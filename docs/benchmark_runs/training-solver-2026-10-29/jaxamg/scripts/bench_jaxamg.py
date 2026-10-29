"""Micro-benchmark: JAX-AMG 0.1.4 (AmgX 2.5.0) vs frozen AMJax on a heterogeneous
grounded 2D grid Laplacian (issue vboussange/ilg-toolkit#117).

Run with the env from env.sh (CUDA_VISIBLE_DEVICES=3, JAX_ENABLE_X64=1).
"""

import json
import statistics
import sys
import time

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp

import jaxamg
from jaxamg.preconditioners import _prepare_preconditioner_config
from jaxscape.solvers.amjaxcgsolver import amjax_preconditioner_operator, build_amjax_solver

GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_RHS = 20
RTOL = ATOL = 1e-6  # ilg-toolkit ResistanceSolverConfig defaults
MAX_STEPS = 5000
RESULTS: dict = {"grid": GRID, "n_rhs": N_RHS, "rtol": RTOL, "atol": ATOL}


def log(key, value):
    RESULTS[key] = value
    print(f"{key}: {value}", flush=True)


def timeit(fn, *args, repeats=5):
    jax.block_until_ready(fn(*args))  # compile + warm cache
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


# ---------------------------------------------------------------- problem
def conductance_field(n, seed=0, decades=4.0):
    """Smooth log-normal field spanning `decades` orders of magnitude."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    kk = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    smooth = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.exp(-((kk / 0.03) ** 2))))
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return 10.0 ** (decades * smooth - decades / 2)


def grounded_laplacian(c):
    """Four-neighbour grid Laplacian, mean edge conductance, last vertex grounded."""
    n0, n1 = c.shape
    idx = np.arange(n0 * n1).reshape(n0, n1)
    rows, cols, w = [], [], []
    for a, b, ca, cb in (
        (idx[:, :-1], idx[:, 1:], c[:, :-1], c[:, 1:]),
        (idx[:-1, :], idx[1:, :], c[:-1, :], c[1:, :]),
    ):
        rows.append(a.ravel()); cols.append(b.ravel()); w.append(0.5 * (ca + cb).ravel())
    rows, cols, w = map(np.concatenate, (rows, cols, w))
    N = n0 * n1
    W = sp.coo_matrix((w, (rows, cols)), shape=(N, N)).tocsr()
    W = W + W.T
    L = (sp.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()
    L = L[:-1, :-1].tocsr()
    L.sort_indices()
    return L


def to_bcsr(L):
    return jsp.BCSR(
        (jnp.asarray(L.data), jnp.asarray(L.indices, jnp.int32), jnp.asarray(L.indptr, jnp.int32)),
        shape=L.shape,
    )


c_het = conductance_field(GRID)
log("conductance_log10_range", [float(np.log10(c_het.min())), float(np.log10(c_het.max()))])
L_uni = grounded_laplacian(np.ones((GRID, GRID)))
L_het = grounded_laplacian(c_het)
assert (L_uni.indices == L_het.indices).all() and (L_uni.indptr == L_het.indptr).all()
N = L_het.shape[0]
A_uni, A_het = to_bcsr(L_uni), to_bcsr(L_het)
INDICES, INDPTR = A_het.indices, A_het.indptr
log("n_unknowns", N)
log("nnz", int(L_het.nnz))

rng = np.random.default_rng(1)
sites = rng.choice(N, size=N_RHS, replace=False)
B = jnp.zeros((N_RHS, N)).at[jnp.arange(N_RHS), sites].set(1.0)
b0 = B[0]
structure = jax.ShapeDtypeStruct((N,), jnp.float64)


def bcsr(data):
    return jsp.BCSR((data, INDICES, INDPTR), shape=(N, N))


def operator(data):
    A = bcsr(data)
    return lx.FunctionLinearOperator(
        lambda x: A @ x, structure, tags=(lx.symmetric_tag, lx.positive_semidefinite_tag)
    )


CG = lx.CG(rtol=RTOL, atol=ATOL, max_steps=MAX_STEPS)


def cg_solve(data, b, precond):
    sol = lx.linear_solve(operator(data), b, CG, options={"preconditioner": precond}, throw=False)
    return sol.value, sol.stats["num_steps"], sol.result


def residual(data, x, b):
    return float(jnp.linalg.norm(bcsr(data) @ x - b) / jnp.linalg.norm(b))


# ---------------------------------------------------------------- AMJax
def amjax_precond(A):
    solver = build_amjax_solver(A.to_bcoo(), coarse_solver="pinv")
    return amjax_preconditioner_operator(solver, structure), len(solver.levels)


t0 = time.perf_counter()
P_frozen, levels = amjax_precond(A_uni)
log("amjax_host_setup_uniform_s", time.perf_counter() - t0)
log("amjax_levels", levels)
t0 = time.perf_counter()
P_refreshed, _ = amjax_precond(A_het)
log("amjax_host_setup_heterogeneous_s", time.perf_counter() - t0)

identity = lx.IdentityLinearOperator(structure)
for name, P in (("plain_cg", identity), ("amjax_frozen", P_frozen), ("amjax_refreshed", P_refreshed)):
    f = jax.jit(lambda data, b, P=P: cg_solve(data, b, P))
    x, steps, result = f(A_het.data, b0)
    log(f"{name}_cg_iters_rhs0", int(steps))
    log(f"{name}_converged", bool(result == lx.RESULTS.successful))
    log(f"{name}_relres", residual(A_het.data, x, b0))
    fb = jax.jit(jax.vmap(lambda data, b, P=P: cg_solve(data, b, P)[0], in_axes=(None, 0)))
    log(f"{name}_time_{N_RHS}rhs_s", timeit(fb, A_het.data, B))

apply_frozen = jax.jit(lambda r: P_frozen.mv(r))
log("amjax_vcycle_apply_s", timeit(apply_frozen, b0, repeats=20))

# ---------------------------------------------------------------- JAX-AMG
# Same V-cycle config as the reverted JAXScape JAXAMGCGSolver.
V_CYCLE = {
    "solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS", "interpolator": "D2",
    "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 1.0},
    "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER", "exact_coarse_solve": 1,
}
PRECOND_CFG = _prepare_preconditioner_config(V_CYCLE)


def jaxamg_apply(data, r, reuse_setup):
    return jaxamg.solve(bcsr(data), r, config=PRECOND_CFG, reuse_setup=reuse_setup)[0]


def jaxamg_precond(data, reuse_setup):
    return lx.FunctionLinearOperator(
        lambda r: jaxamg_apply(data, r, reuse_setup),
        structure,
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )


# (cold) first call: cache miss -> AMGX_matrix_upload_all + AMGX_solver_setup
apply_default = jax.jit(lambda data, r: jaxamg_apply(data, r, False))
apply_reuse = jax.jit(lambda data, r: jaxamg_apply(data, r, True))
t0 = time.perf_counter(); jax.block_until_ready(apply_default(A_het.data, b0))
log("jaxamg_first_call_incl_compile_and_setup_s", time.perf_counter() - t0)

# (a) default refresh: every application = replace_coefficients + AMGX_solver_resetup + V-cycle
log("a_jaxamg_apply_default_refresh_s", timeit(apply_default, A_het.data, b0, repeats=20))
# (b) reuse_setup=True: replace_coefficients + V-cycle on the cached hierarchy
jax.block_until_ready(apply_default(A_het.data, b0))  # hierarchy at current matrix
log("b_jaxamg_apply_reuse_setup_s", timeit(apply_reuse, A_het.data, b0, repeats=20))

# Lineax CG + JAX-AMG V-cycle, default refresh (the reverted integration)
f = jax.jit(lambda data, b: cg_solve(data, b, jaxamg_precond(data, False)))
x, steps, result = f(A_het.data, b0)
log("jaxamg_refresh_cg_iters_rhs0", int(steps))
log("jaxamg_refresh_relres", residual(A_het.data, x, b0))
log("jaxamg_refresh_time_1rhs_s", timeit(f, A_het.data, b0, repeats=3))
fb = jax.jit(jax.vmap(lambda data, b: cg_solve(data, b, jaxamg_precond(data, False))[0], in_axes=(None, 0)))
log(f"jaxamg_refresh_time_{N_RHS}rhs_s", timeit(fb, A_het.data, B, repeats=1))

# Lineax CG + JAX-AMG V-cycle, hierarchy set up once at current matrix, then reused.
jax.block_until_ready(apply_default(A_het.data, b0))
f = jax.jit(lambda data, b: cg_solve(data, b, jaxamg_precond(data, True)))
x, steps, result = f(A_het.data, b0)
log("jaxamg_reuse_cg_iters_rhs0", int(steps))
log("jaxamg_reuse_relres", residual(A_het.data, x, b0))
log("jaxamg_reuse_time_1rhs_s", timeit(f, A_het.data, b0, repeats=3))
fb = jax.jit(jax.vmap(lambda data, b: cg_solve(data, b, jaxamg_precond(data, True))[0], in_axes=(None, 0)))
log(f"jaxamg_reuse_time_{N_RHS}rhs_s", timeit(fb, A_het.data, B, repeats=3))

# Frozen-at-uniform AmgX hierarchy (apples-to-apples with AMJax frozen)
jax.block_until_ready(apply_default(A_uni.data, b0))  # hierarchy built on uniform matrix
x, steps, result = f(A_het.data, b0)  # reuse_setup=True keeps the uniform hierarchy
log("jaxamg_frozen_uniform_cg_iters_rhs0", int(steps))
log("jaxamg_frozen_uniform_converged", bool(result == lx.RESULTS.successful))

# (c) AmgX as the full solver: PCG + classical AMG V-cycle, one FFI call per RHS.
PCG_CFG = {
    "solver": "PCG", "preconditioner": V_CYCLE, "convergence": "RELATIVE_INI",
    "tolerance": RTOL, "max_iters": MAX_STEPS, "norm": "L2",
}


def amgx_pcg(data, b, reuse_setup):
    x, info = jaxamg.solve(bcsr(data), b, config=PCG_CFG, reuse_setup=reuse_setup)
    return x, info["iterations"]


pcg_default = jax.jit(lambda data, b: amgx_pcg(data, b, False))
pcg_reuse = jax.jit(lambda data, b: amgx_pcg(data, b, True))
x, iters = pcg_default(A_het.data, b0)  # cache miss for this config -> full setup
log("c_amgx_pcg_iters_rhs0", int(iters))
log("c_amgx_pcg_relres", residual(A_het.data, x, b0))
log("c_amgx_pcg_time_1rhs_default_s", timeit(pcg_default, A_het.data, b0, repeats=10))
log("c_amgx_pcg_time_1rhs_reuse_s", timeit(pcg_reuse, A_het.data, b0, repeats=10))
pb_default = jax.jit(jax.vmap(lambda data, b: amgx_pcg(data, b, False)[0], in_axes=(None, 0)))
pb_reuse = jax.jit(jax.vmap(lambda data, b: amgx_pcg(data, b, True)[0], in_axes=(None, 0)))
log(f"c_amgx_pcg_time_{N_RHS}rhs_default_s", timeit(pb_default, A_het.data, B, repeats=3))
log(f"c_amgx_pcg_time_{N_RHS}rhs_reuse_s", timeit(pb_reuse, A_het.data, B, repeats=3))
X = pb_reuse(A_het.data, B)
log("c_amgx_pcg_max_relres_20rhs", max(residual(A_het.data, X[i], B[i]) for i in range(N_RHS)))

# Reverse-mode through AmgX PCG (jaxamg custom_vjp; adjoint = one more native solve per RHS)
sym = lambda A: jaxamg.with_cache(A, is_symmetric=True)


def loss(data, B):
    xs = jax.vmap(lambda b: jaxamg.solve(sym(bcsr(data)), b, config=PCG_CFG)[0])(B)
    return jnp.sum(xs[jnp.arange(N_RHS), sites])


g = jax.jit(jax.value_and_grad(loss))
log(f"c_amgx_pcg_value_and_grad_{N_RHS}rhs_default_s", timeit(g, A_het.data, B, repeats=3))

with open(f"results_{GRID}.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2)
print(json.dumps(jaxamg.get_solver_cache_info()["single_gpu"]["size"]))
