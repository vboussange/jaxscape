from importlib.util import find_spec

import jax
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
import pytest
from equinox import filter_grad, filter_jit
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, SpielmanApproximation
from jaxscape.resistance_distance import (
    p_inv_resistance_distance,
    ResistanceDistance,
)
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver, PyAMGSolver


# Check availability of optional solvers
PYAMG_AVAILABLE = find_spec("pyamg") is not None
CHOLMOD_AVAILABLE = find_spec("cholespy") is not None
AMJAX_AVAILABLE = PYAMG_AVAILABLE and find_spec("amjax") is not None

# Build list of available solvers
available_solvers = []
if PYAMG_AVAILABLE:
    available_solvers.append(PyAMGSolver())
if CHOLMOD_AVAILABLE:
    available_solvers.append(CholmodSolver())
if AMJAX_AVAILABLE:
    available_solvers.append(AMJaxCGSolver(rtol=1e-5, atol=1e-5, max_steps=500))

# Spielman projections are randomized; this is the expected absolute error for
# the deterministic seed and epsilon used in the test below.
EXPECTED_APPROXIMATION_ERROR = 8e-2
EXPECTED_DIAGONAL_ERROR = 5e-4
EXPECTED_GRADIENT_ERROR = 5e-3


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
    distance = ResistanceDistance(solver=solver)
    state = distance.init(grid)
    dist_lineax = distance(grid, state=state)
    assert jnp.allclose(dist_pinv, dist_lineax, rtol=1e-4)


def test_approximate_resistance_distance():
    """
    Tests that the Spielman approximation is jittable and close to the
    pseudo-inverse method.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)

    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.05))
    dist_approx = filter_jit(distance)(grid)
    dist_pinv = ResistanceDistance()(grid)

    assert dist_approx.shape == dist_pinv.shape
    assert jnp.allclose(dist_approx, dist_approx.T, atol=1e-5)
    assert jnp.allclose(jnp.diag(dist_approx), 0, atol=EXPECTED_DIAGONAL_ERROR)
    assert jnp.allclose(dist_approx, dist_pinv, atol=EXPECTED_APPROXIMATION_ERROR)

    nodes = jnp.array([0, 2])
    sources = jnp.array([0, 1])
    targets = jnp.array([2, 3])
    assert jnp.allclose(
        filter_jit(distance)(grid, nodes=nodes),
        dist_approx[nodes[:, None], nodes[None, :]],
        atol=1e-3,
    )
    assert jnp.allclose(
        filter_jit(distance)(grid, sources=sources, targets=targets),
        dist_approx[sources[:, None], targets[None, :]],
        atol=1e-3,
    )


def test_approximate_resistance_distance_differentiability():
    """
    Tests that the custom VJP keeps the approximate method compatible with
    jax.grad.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.1))

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
    assert jnp.allclose(
        gradient[0, 0], finite_difference, atol=EXPECTED_GRADIENT_ERROR
    )


def test_approximate_nodes_to_nodes_resistance_distance_differentiability():
    """
    Tests that the selected-node Spielman path remains compatible with jax.grad.
    """
    key = jr.PRNGKey(0)
    permeability_raster = jr.uniform(key, (2, 2)) + 0.5
    nodes = jnp.array([0, 2])
    distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.1))

    def objective(permeability_raster):
        grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)
        return jnp.sum(distance(grid, nodes=nodes))

    gradient = filter_jit(filter_grad(objective))(permeability_raster)

    assert isinstance(gradient, jax.Array)
    assert gradient.shape == permeability_raster.shape
    assert jnp.all(jnp.isfinite(gradient))
