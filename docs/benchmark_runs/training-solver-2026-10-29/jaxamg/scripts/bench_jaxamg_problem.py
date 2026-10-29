"""Problem construction shared with bench_jaxamg.py (copied verbatim)."""
import sys
import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp
import json, statistics, time
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


