"""cuDSS (spineax) accuracy and timing on the AMG experiment's surfaces.

Closes the gap left by bench_spineax.py, which used an uncorrelated 4-decade
field: here the surfaces, grounding (last vertex) and 20 terminals are exactly
those of amg_exp/common.py, including 6 decades, linear barriers and a 1e-6
conductance floor. Effective resistances are compared pair by pair with CHOLMOD
(cholespy) on the same matrix, using the AMG experiment's error metric.

Run: CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=1 \
     venv12/bin/python contrast_check.py --sizes 100 300 1000
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

from spineax import cudss

from bench_spineax import grounded_laplacian_structure, make_values_fn

jax.config.update("jax_enable_x64", True)

SEED = 0
N_TERMINALS = 20
SURFACES = ["uniform", "grf2", "grf4", "grf6", "barrier", "floor"]


# Surfaces and terminals copied verbatim from amg_exp/common.py, which imports
# JAXScape and cannot be loaded in this environment.
def grf01(n, seed, corr_frac=0.075):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    sigma = corr_frac * n
    filt = np.exp(-2.0 * np.pi**2 * sigma**2 * (kx**2 + ky**2))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    return (field - field.min()) / (field.max() - field.min())


def barrier_surface(n):
    c = np.ones((n, n))
    w = max(1, n // 100)
    r = n // 3
    c[r : r + w, : int(0.8 * n)] = 1e-4
    col = (2 * n) // 3
    c[int(0.2 * n) :, col : col + w] = 1e-4
    for t in np.linspace(0.0, 1.0, 4 * n):
        i = int((0.55 + 0.4 * t) * (n - 1))
        j = int((0.05 + 0.5 * t) * (n - 1))
        c[i, j : j + w] = 1e-4
    return c


def make_surface(name, n):
    if name == "uniform":
        return np.ones((n, n))
    if name.startswith("grf"):
        return 10.0 ** (-float(name[3:]) * (1.0 - grf01(n, SEED)))
    if name == "barrier":
        return barrier_surface(n)
    if name == "floor":
        u = grf01(n, SEED + 1)
        c = 10.0 ** (-3.0 * (1.0 - u))
        c[u < np.quantile(u, 0.25)] = 1e-6
        return c
    raise ValueError(name)


def terminal_nodes(n):
    rng = np.random.default_rng(SEED + 100)
    return np.sort(rng.choice(n * n - 1, N_TERMINALS, replace=False)).astype(np.int32)


def resistances(G):
    d = np.diag(G)
    return d[:, None] + d[None, :] - G - G.T


def rel_error(R, R_ref):
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.max(np.abs(R[mask] - R_ref[mask]) / np.abs(R_ref[mask])))


def main(sizes):
    for n in sizes:
        S = grounded_laplacian_structure(n, ground=n * n - 1)
        M = S["M"]
        values_fn = jax.jit(make_values_fn(S))
        offs, cols = jnp.asarray(S["offsets"]), jnp.asarray(S["columns"])
        T = terminal_nodes(n)
        E = jnp.zeros((T.size, M)).at[jnp.arange(T.size), T].set(1.0)
        analyze = jax.jit(lambda v: cudss.analyze(v, offs, cols, mtype_id="spd", mview_id="full"))
        factorize = jax.jit(cudss.factorize)

        @jax.jit
        def refactor_solve(token, v):
            token = cudss.refactorize(token, v)
            return cudss.solve(token, E), token

        token = None
        for surf in SURFACES:
            c = make_surface(surf, n).ravel()
            v = jax.block_until_ready(values_fn(jnp.asarray(c)))
            if token is None:
                token = factorize(analyze(v), v)
            X, token = refactor_solve(token, v)  # compile on first surface
            jax.block_until_ready(X)
            rb0 = cudss.rebuild_count()
            ts = []
            for _ in range(3):
                t = time.perf_counter()
                X, token = refactor_solve(token, v)
                jax.block_until_ready(X)
                ts.append(time.perf_counter() - t)
            X = np.asarray(X)
            data = np.asarray(v, dtype=np.float64)
            chol = cholespy.CholeskySolverD(
                M, S["rows"].copy(), S["columns"].copy(), data.copy(), cholespy.MatrixType.COO
            )
            B = np.ascontiguousarray(np.asarray(E).T)
            Xc = np.empty_like(B)
            chol.solve(B, Xc)
            Xc = Xc.T
            A = sp.csr_matrix((data, S["columns"], S["offsets"]), shape=(M, M))
            rec = {
                "size": n,
                "surface": surf,
                "log10c": [float(np.log10(c.min())), float(np.log10(c.max()))],
                "refactorize_solve_20rhs_s": statistics.median(ts),
                "rebuilds": cudss.rebuild_count() - rb0,
                "rel_err_vs_cholmod": rel_error(resistances(X[:, T]), resistances(Xc[:, T])),
                "rel_residual": float(
                    np.linalg.norm(A @ X.T - np.asarray(E).T) / np.linalg.norm(np.asarray(E))
                ),
                "finite": bool(np.isfinite(X).all()),
            }
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 300, 1000])
    main(p.parse_args().sizes)
