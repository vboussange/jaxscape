"""Shared pieces for the ilg-toolkit#117 AMG preconditioner experiment.

The operator is built exactly as ilg-toolkit does: JAXScape 4-neighbour
GridGraph with mean-of-endpoints edge conductance, graph Laplacian grounded at
the last vertex (``[:-1, :-1]``), float64. Hierarchies go through JAXScape's
``build_amjax_solver`` (device->host CSR, PyAMG, AMJax ``from_pyamg``) so the
host/device costs are the ones a JAXScape-based fix would pay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import cholespy
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pyamg
import scipy.sparse as ssp
from jax.experimental.sparse import BCOO, BCSR

from jaxscape import GridGraph
from jaxscape.solvers import BCOOLinearOperator
from jaxscape.solvers.amjaxcgsolver import (
    _unbatched_bcoo,
    amjax_preconditioner_operator,
    build_amjax_solver,
)
from jaxscape.utils import graph_laplacian, zero_copy_jax_csr_to_scipy_csr

RTOL = 1e-6
ATOL = 1e-6
N_TERMINALS = 20
SEED = 0


# ----------------------------------------------------------------------------
# Surfaces
# ----------------------------------------------------------------------------

def grf01(n: int, seed: int, corr_frac: float = 0.075) -> np.ndarray:
    """Smooth Gaussian random field on an n x n grid, min-max scaled to [0, 1].

    White noise filtered by a Gaussian kernel of std ``corr_frac * n`` pixels
    (periodic FFT convolution), i.e. correlation length ~7.5 % of the side.
    """
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    k = np.fft.fftfreq(n)
    kx, ky = np.meshgrid(k, k, indexing="ij")
    sigma = corr_frac * n
    filt = np.exp(-2.0 * np.pi**2 * sigma**2 * (kx**2 + ky**2))
    field = np.real(np.fft.ifft2(np.fft.fft2(white) * filt))
    field = (field - field.min()) / (field.max() - field.min())
    return field


def barrier_surface(n: int) -> np.ndarray:
    """Background 1.0 with three thin linear barriers of conductance 1e-4.

    A horizontal barrier with a gap, a vertical one with a gap and a diagonal
    one; width ~1 % of the side (>= 1 pixel).
    """
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


def make_surface(name: str, n: int) -> np.ndarray:
    if name == "uniform":
        return np.ones((n, n))
    if name.startswith("grf"):
        span = float(name[3:])
        u = grf01(n, SEED)
        return 10.0 ** (-span * (1.0 - u))
    if name == "barrier":
        return barrier_surface(n)
    if name == "floor":
        # 3-order smooth field, lowest-quartile area clamped to the 1e-6 floor
        # (mimics a softplus + 1e-6 output driven to its floor).
        u = grf01(n, SEED + 1)
        c = 10.0 ** (-3.0 * (1.0 - u))
        c[u < np.quantile(u, 0.25)] = 1e-6
        return c
    raise ValueError(name)


def terminal_nodes(n: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 100)
    # Exclude the grounded last vertex so every RHS is a nonzero unit vector.
    return np.sort(rng.choice(n * n - 1, N_TERMINALS, replace=False)).astype(np.int32)


# ----------------------------------------------------------------------------
# Operator
# ----------------------------------------------------------------------------

def _mean_conductance(left, right):
    return 0.5 * (left + right)


@jax.jit
def reduced_laplacian(c: jax.Array) -> BCOO:
    """Grounded graph Laplacian exactly as JAXScape/ilg-toolkit builds it."""
    graph = GridGraph(c, fun=_mean_conductance)
    return graph_laplacian(graph.get_adjacency_matrix())[:-1, :-1]


def to_scipy_csr(L: BCOO) -> ssp.csr_matrix:
    """Same conversion as jaxscape.build_amjax_solver, then pruned/canonical."""
    m = _unbatched_bcoo(L).sum_duplicates(nse=L.nse)
    A = zero_copy_jax_csr_to_scipy_csr(BCSR.from_bcoo(m))
    A = ssp.csr_matrix(A)
    A.prune()
    A.sum_duplicates()
    return A


@jax.jit
def bcoo_diagonal(L: BCOO) -> jax.Array:
    rows, cols = L.indices[:, 0], L.indices[:, 1]
    vals = jnp.where(rows == cols, L.data, 0.0)
    return jnp.zeros(L.shape[0], L.dtype).at[rows].add(vals, mode="drop")


# ----------------------------------------------------------------------------
# Hierarchies
# ----------------------------------------------------------------------------

@dataclass
class BuiltHierarchy:
    ml: Any                     # amjax.MultilevelSolver (pytree)
    pyamg_h: Any                # pyamg hierarchy (None for derived variants)
    t_total: float              # wall time of the whole rebuild, s
    t_pyamg: float              # host PyAMG setup only, s
    t_other: float              # D2H + AMJax conversion/H2D + pinv, s


def _block(ml):
    jax.block_until_ready(jax.tree_util.tree_leaves(ml))


def build_hierarchy(L: BCOO, pyamg_kwargs: dict) -> BuiltHierarchy:
    """Rebuild through jaxscape.build_amjax_solver, timing the PyAMG part."""
    store: dict[str, Any] = {}

    def timed_sa(A, **kw):
        t = time.perf_counter()
        h = pyamg.smoothed_aggregation_solver(A, **kw)
        store["t"] = time.perf_counter() - t
        store["h"] = h
        return h

    jax.block_until_ready(L.data)
    t0 = time.perf_counter()
    ml = build_amjax_solver(
        L, pyamg_method=timed_sa, pyamg_kwargs=pyamg_kwargs, coarse_solver="pinv"
    )
    _block(ml)
    t_total = time.perf_counter() - t0
    return BuiltHierarchy(ml, store["h"], t_total, store["t"], t_total - store["t"])


def aggops(h) -> list:
    return [lvl.AggOp for lvl in h.levels[:-1]]


def predefined_kwargs(agg: list, smooth: Any = "default", improve: Any = "default"):
    kw: dict[str, Any] = {
        "strength": None,  # aggregates are fixed, skip SoC
        "aggregate": [("predefined", {"AggOp": a}) for a in agg],
        "max_levels": len(agg) + 1,
        "keep": True,
    }
    if smooth != "default":
        kw["smooth"] = smooth
    if improve != "default":
        kw["improve_candidates"] = improve
    return kw


def fine_refreshed(ml_frozen, L: BCOO):
    """Frozen hierarchy with only level-0 A and its Jacobi diagonal refreshed."""
    t0 = time.perf_counter()
    ml = jax.tree_util.tree_map(lambda x: x, ml_frozen)  # shallow pytree copy
    A0 = BCOO.from_scipy_sparse(to_scipy_csr(L)).astype(L.dtype)
    ml.levels[0].A = A0
    ml.levels[0].Dinv = 1.0 / bcoo_diagonal(A0)
    _block(ml)
    return ml, time.perf_counter() - t0


def level_signature(ml) -> list[dict]:
    out = []
    for i, lvl in enumerate(ml.levels):
        d = {"n": int(lvl.A.shape[0]), "A_nse": int(lvl.A.nse)}
        if lvl.P is not None and i < len(ml.levels) - 1:
            d["P_shape"] = list(lvl.P.shape)
            d["P_nse"] = int(lvl.P.nse)
            d["R_nse"] = int(lvl.R.nse)
        out.append(d)
    return out


def leaf_signature(ml):
    leaves, treedef = jax.tree_util.tree_flatten(ml)
    return treedef, [(tuple(x.shape), str(x.dtype)) for x in leaves]


def same_compiled_signature(ml_a, ml_b) -> bool:
    ta, la = leaf_signature(ml_a)
    tb, lb = leaf_signature(ml_b)
    return ta == tb and la == lb


# ----------------------------------------------------------------------------
# Jitted CG solves (vmapped over the 20 one-hot RHS, as JAXScape does)
# ----------------------------------------------------------------------------

TRACE_COUNT: dict[str, int] = {}


def _resistances_from_potentials(V, nodes):
    G = V[nodes]  # (m, m); no terminal is the ground node
    d = jnp.diag(G)
    return d[:, None] + d[None, :] - G - G.T


def make_cg_solve(kind: str, max_steps: int) -> Callable:
    """kind in {'none', 'jacobi', 'amg'}; for 'amg' the AMJax hierarchy is a
    dynamic jit argument so swapping hierarchies with identical leaf
    shapes/treedef reuses the executable."""
    cg = lx.CG(rtol=RTOL, atol=ATOL, max_steps=max_steps, stabilise_every=10)

    def solve(L: BCOO, nodes: jax.Array, ml: Any):
        TRACE_COUNT[kind] = TRACE_COUNT.get(kind, 0) + 1
        n = L.shape[0]
        operator = BCOOLinearOperator(L)
        struct = jax.ShapeDtypeStruct((n,), L.dtype)
        if kind == "none":
            options = {}
        elif kind == "jacobi":
            dinv = 1.0 / bcoo_diagonal(L)
            options = {
                "preconditioner": lx.FunctionLinearOperator(
                    lambda r: dinv * r, struct, tags=lx.positive_semidefinite_tag
                )
            }
        elif kind == "amg":
            options = {"preconditioner": amjax_preconditioner_operator(ml, struct)}
        else:
            raise ValueError(kind)
        basis = jax.nn.one_hot(nodes, n, dtype=L.dtype).T

        def one(b):
            sol = lx.linear_solve(operator, b, solver=cg, options=options, throw=False)
            return sol.value, sol.stats["num_steps"]

        V, steps = jax.vmap(one, in_axes=1, out_axes=(1, 0))(basis)
        return _resistances_from_potentials(V, nodes), steps

    return jax.jit(solve)


def timed_runs(fn, args, repeats: int):
    """First call (compile if needed) + `repeats` warm calls; returns stats."""
    t = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    t_first = time.perf_counter() - t
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t)
    return out, t_first, times


# ----------------------------------------------------------------------------
# Direct reference
# ----------------------------------------------------------------------------

def direct_resistances(L: BCOO, nodes: np.ndarray):
    A = to_scipy_csr(L).tocoo()
    n = A.shape[0]
    t = time.perf_counter()
    solver = cholespy.CholeskySolverD(
        n,
        np.array(A.row, dtype=np.int32, order="C", copy=True),
        np.array(A.col, dtype=np.int32, order="C", copy=True),
        np.array(A.data, dtype=np.float64, order="C", copy=True),
        cholespy.MatrixType.COO,
    )
    rhs = np.zeros((n, nodes.size))
    rhs[nodes, np.arange(nodes.size)] = 1.0
    x = np.empty_like(rhs)
    solver.solve(rhs, x)
    t_direct = time.perf_counter() - t
    G = x[nodes]
    d = np.diag(G)
    return d[:, None] + d[None, :] - G - G.T, t_direct


def rel_error(R, R_ref) -> float:
    R = np.asarray(R)
    mask = ~np.eye(R.shape[0], dtype=bool)
    return float(np.max(np.abs(R[mask] - R_ref[mask]) / np.abs(R_ref[mask])))
