"""Smoke test + benchmark: spineax (cuDSS) Cholesky on grounded 2D grid Laplacians.

Run with CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1.
"""

import argparse
import json
import statistics
import time

import cholespy
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spineax import cudss

jax.config.update("jax_enable_x64", True)


# ----------------------------------------------------------------------------- problem
def grid_edges(n: int) -> np.ndarray:
    idx = np.arange(n * n).reshape(n, n)
    h = np.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)
    v = np.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], 1)
    return np.concatenate([h, v], 0)


def grounded_laplacian_structure(n: int, ground: int = 0):
    """CSR pattern of the grounded (full, symmetric) Laplacian and a COO->CSR map.

    Returns offsets, columns, plus (coo_pos, coo_edge, coo_sign) such that
    values = segment_sum(coo_sign * w[coo_edge], coo_pos, nnz).
    """
    N = n * n
    edges = grid_edges(n)
    u, v = edges[:, 0], edges[:, 1]
    e = np.arange(len(edges))
    rows = np.concatenate([u, v, u, v])
    cols = np.concatenate([u, v, v, u])
    eidx = np.concatenate([e, e, e, e])
    sign = np.concatenate([np.ones(2 * len(e)), -np.ones(2 * len(e))])
    keep = (rows != ground) & (cols != ground)
    rows, cols, eidx, sign = rows[keep], cols[keep], eidx[keep], sign[keep]
    # relabel to the grounded system (drop node `ground`)
    rows = rows - (rows > ground)
    cols = cols - (cols > ground)
    M = N - 1
    key = rows.astype(np.int64) * M + cols
    uniq, pos = np.unique(key, return_inverse=True)
    r_u, c_u = uniq // M, uniq % M
    offsets = np.zeros(M + 1, np.int64)
    np.add.at(offsets, r_u + 1, 1)
    offsets = np.cumsum(offsets)
    return dict(
        M=M,
        edges=edges,
        offsets=offsets.astype(np.int32),
        columns=c_u.astype(np.int32),
        rows=r_u.astype(np.int32),
        coo_pos=pos.astype(np.int32),
        coo_edge=eidx.astype(np.int32),
        coo_sign=sign,
        nnz=len(uniq),
    )


def make_values_fn(S):
    edges = jnp.asarray(S["edges"])
    coo_pos = jnp.asarray(S["coo_pos"])
    coo_edge = jnp.asarray(S["coo_edge"])
    coo_sign = jnp.asarray(S["coo_sign"])
    nnz = S["nnz"]

    def values(cond):  # cond: (N,) node conductances
        w = 0.5 * (cond[edges[:, 0]] + cond[edges[:, 1]])
        return jax.ops.segment_sum(coo_sign * w[coo_edge], coo_pos, nnz)

    return values


def lognormal_conductance(n: int, seed: int, decades: float = 4.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n * n)
    z = np.clip(z, -2.0, 2.0) / 2.0  # [-1, 1]
    return 10.0 ** (0.5 * decades * z)  # spans `decades` orders of magnitude


def terminals(n: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(1, n * n), size=k, replace=False) - 1  # grounded idx


# ----------------------------------------------------------------------------- timing
def timeit(fn, reps=3):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), out


def run(n: int, nrhs_list, reps: int = 3, scipy_ref: bool = True):
    res = {"n": n, "N": n * n - 1}
    t0 = time.perf_counter()
    S = grounded_laplacian_structure(n)
    res["host_structure_s"] = time.perf_counter() - t0
    res["nnz"] = S["nnz"]
    values_fn = jax.jit(make_values_fn(S))
    offs = jnp.asarray(S["offsets"])
    cols = jnp.asarray(S["columns"])
    cond0 = jnp.asarray(lognormal_conductance(n, 0))
    vals0 = jax.block_until_ready(values_fn(cond0))

    analyze = jax.jit(
        lambda v: cudss.analyze(v, offs, cols, mtype_id="spd", mview_id="full")
    )
    refac = jax.jit(cudss.refactorize)
    fac = jax.jit(cudss.factorize)
    solve = jax.jit(cudss.solve)

    # analysis (each call mints a fresh registry entry; LRU-bounded)
    jax.block_until_ready(analyze(vals0).id)  # compile
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        tok = analyze(vals0)
        jax.block_until_ready(tok.id)
        ts.append(time.perf_counter() - t)
    res["analysis_s"] = statistics.median(ts)

    # first numeric factorization on the analyzed token
    t = time.perf_counter()
    tok = fac(tok, vals0)
    jax.block_until_ready(tok.id)
    res["factorize_first_incl_compile_s"] = time.perf_counter() - t

    # warm refactorization with NEW values (token threaded through)
    conds = [jnp.asarray(lognormal_conductance(n, s)) for s in range(1, reps + 2)]
    valss = [jax.block_until_ready(values_fn(c)) for c in conds]
    tok = jax.block_until_ready(refac(tok, valss[0]))  # compile
    rb0 = cudss.rebuild_count()
    ts = []
    for v in valss[1:]:
        t = time.perf_counter()
        tok = refac(tok, v)
        jax.block_until_ready(tok.id)
        ts.append(time.perf_counter() - t)
    res["refactorize_s"] = statistics.median(ts)
    res["refactorize_rebuilds"] = cudss.rebuild_count() - rb0
    vals_last = valss[-1]

    # full factorize (no refactorization phase) for comparison
    ts = []
    for _ in range(reps):
        t2 = analyze(vals_last)
        jax.block_until_ready(t2.id)
        t = time.perf_counter()
        t2 = fac(t2, vals_last)
        jax.block_until_ready(t2.id)
        ts.append(time.perf_counter() - t)
    res["factorize_s"] = statistics.median(ts)

    # reference factorization on CPU (CHOLMOD via cholespy)
    data = np.array(vals_last, dtype=np.float64)
    t = time.perf_counter()
    chol = cholespy.CholeskySolverD(
        S["M"],
        S["rows"].copy(),
        S["columns"].copy(),
        data.copy(),
        cholespy.MatrixType.COO,
    )
    res["cholmod_cpu_analyze_factor_s"] = time.perf_counter() - t
    A = sp.csr_matrix((data, S["columns"], S["offsets"]), shape=(S["M"], S["M"]))

    for k in nrhs_list:
        T = terminals(n, k, seed=123)
        B = np.zeros((k, S["M"]))
        B[np.arange(k), T] = 1.0
        Bj = jnp.asarray(B)
        jax.block_until_ready(solve(tok, Bj))  # compile
        res[f"solve_{k}rhs_s"], X = timeit(lambda: solve(tok, Bj), reps)
        X = np.asarray(X)
        # CHOLMOD reference
        Bc = np.ascontiguousarray(B.T)
        Xc = np.empty_like(Bc)
        t = time.perf_counter()
        chol.solve(Bc, Xc)
        res[f"cholmod_cpu_solve_{k}rhs_s"] = time.perf_counter() - t
        Xc = Xc.T
        res[f"relerr_vs_cholmod_{k}rhs"] = float(np.linalg.norm(X - Xc) / np.linalg.norm(Xc))
        res[f"relres_{k}rhs"] = float(np.linalg.norm(A @ X.T - B.T) / np.linalg.norm(B))
        if scipy_ref and k == nrhs_list[0] and S["M"] < 200_000:
            t = time.perf_counter()
            Xs = spla.spsolve(A.tocsc(), B.T)
            res["scipy_spsolve_s"] = time.perf_counter() - t
            Xs = np.asarray(Xs).T
            res[f"relerr_vs_scipy_{k}rhs"] = float(np.linalg.norm(X - Xs) / np.linalg.norm(Xs))
    # condition-ish diagnostic: conductance range
    res["cond_range"] = [float(conds[-1].min()), float(conds[-1].max())]
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000])
    p.add_argument("--nrhs", type=int, nargs="+", default=[20, 100])
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()
    print(jax.devices(), "cuDSS cache capacity", cudss.cache_capacity(), flush=True)
    allres = []
    for n in a.sizes:
        r = run(n, a.nrhs)
        print(json.dumps(r), flush=True)
        allres.append(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(allres, f, indent=1)
