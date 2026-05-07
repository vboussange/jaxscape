import jax
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
import pytest
from equinox import filter_grad, filter_jit
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph
from jaxscape.resistance_distance import (
    p_inv_resistance_distance,
    ResistanceDistance,
)
from jaxscape.solvers import CholmodSolver, PyAMGSolver


# Check availability of optional solvers
try:
    import pyamg

    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False

try:
    import cholespy

    CHOLMOD_AVAILABLE = True
except ImportError:
    CHOLMOD_AVAILABLE = False

# Build list of available solvers
available_solvers = []
if PYAMG_AVAILABLE:
    available_solvers.append(PyAMGSolver())
if CHOLMOD_AVAILABLE:
    available_solvers.append(CholmodSolver())


def build_nx_resistance_distance_matrix(G):
    Rnx_dict = nx.resistance_distance(G, weight="weight", invert_weight=False)
    Rnx = jnp.zeros((G.number_of_nodes(), G.number_of_nodes()))
    node_list = list(G)
    for n, rd in Rnx_dict.items():
        i = node_list.index(n)
        for m, r in rd.items():
            j = node_list.index(m)
            Rnx = Rnx.at[i, j].set(r)
    return Rnx


def test_p_inv_resistance_distance():
    G = nx.grid_2d_graph(2, 3)
    # for u, v in G.edges():
    #     G[u][v]['weight'] = 1

    # simple graph
    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = p_inv_resistance_distance(Ajx)
    Rnx = build_nx_resistance_distance_matrix(G)
    assert jnp.allclose(Rjaxscape, Rnx)

    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.uniform(1, 10)  # Random weight between 1 and 10

    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = p_inv_resistance_distance(Ajx)
    Rnx = build_nx_resistance_distance_matrix(G)
    assert jnp.allclose(Rjaxscape, Rnx)


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_lineax_solver_resistance_distance(solver):
    """
    Tests that the lineax solver implementation of resistance distance
    produces the same result as the pseudo-inverse method.
    """
    key = jr.PRNGKey(42)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.1  # avoid zero permeability
    grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)

    # nodes to nodes
    dist_pinv = ResistanceDistance(solver=None)(grid)
    dist_lineax = ResistanceDistance(solver=solver)(grid)
    assert jnp.allclose(dist_pinv, dist_lineax, rtol=1e-4)


def test_approximate_resistance_distance():
    """
    Tests that the Spielman approximation is jittable and close to the
    pseudo-inverse method.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)

    distance = ResistanceDistance(approximate=True, epsilon=0.05)
    dist_approx = filter_jit(distance)(grid)
    dist_pinv = ResistanceDistance()(grid)

    assert dist_approx.shape == dist_pinv.shape
    assert jnp.allclose(dist_approx, dist_approx.T, atol=1e-5)
    assert jnp.allclose(jnp.diag(dist_approx), 0, atol=1e-5)
    assert jnp.allclose(dist_approx, dist_pinv, atol=8e-2)

    nodes = jnp.array([0, 2])
    sources = jnp.array([0, 1])
    targets = jnp.array([2, 3])
    assert jnp.allclose(
        filter_jit(distance)(grid, nodes=nodes),
        dist_approx[nodes[:, None], nodes[None, :]],
    )
    assert jnp.allclose(
        filter_jit(distance)(grid, sources=sources, targets=targets),
        dist_approx[sources[:, None], targets[None, :]],
    )


def test_approximate_resistance_distance_differentiability():
    """
    Tests that the custom VJP keeps the approximate method compatible with
    jax.grad.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    distance = ResistanceDistance(approximate=True, epsilon=0.1)

    def objective(permeability_raster):
        grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid))

    gradient = filter_jit(filter_grad(objective))(permeability_raster)

    step = 1e-2
    perturbation = jnp.zeros_like(permeability_raster).at[0, 0].set(step)
    finite_difference = (
        objective(permeability_raster + perturbation)
        - objective(permeability_raster - perturbation)
    ) / (2 * step)

    assert isinstance(gradient, jax.Array)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient[0, 0], finite_difference, atol=1e-3)
