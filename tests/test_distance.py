import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import numpy as np
import pytest
from conftest import DenseCholesky
from equinox import filter_grad, filter_jit
from jaxscape import (
    EuclideanDistance,
    FreeEnergyDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    SpielmanApproximation,
)
from jaxscape.solvers import (
    AbstractReusableStateSolver,
    AMJaxCGSolver,
    CholmodSolver,
    CuDSSSolver,
    PyAMGSolver,
)


DISTANCES = [
    EuclideanDistance(),
    LCPDistance(),
    ResistanceDistance(),
    FreeEnergyDistance(theta=0.1),
]


def test_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (3, 3))  # Start with a uniform permeability

    for distance in DISTANCES:
        distance_jitted = filter_jit(distance)
        grid = GridGraph(permeability_raster, fun=lambda x, y: (x + y) / 2)
        dist = distance_jitted(grid)
        assert dist.shape == (grid.nv, grid.nv)

        dist = distance_jitted(grid, sources=jnp.array([0, 1]), targets=jnp.array([2]))
        assert dist.shape == (2, 1)

        dist = distance_jitted(grid, nodes=jnp.array([0, 1]))
        assert dist.shape == (2, 2)


def test_differentiability():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (2, 2))  # Start with a uniform permeability

    for distance in DISTANCES:

        def objective(permeability_raster):
            grid = GridGraph(permeability_raster, fun=lambda x, y: (x + y) / 2)
            dist = distance(grid)
            return jnp.sum(dist)

        grad_objective = filter_jit(filter_grad(objective))
        dobj = grad_objective(permeability_raster)
        assert isinstance(dobj, jax.Array)


def test_distances_without_a_solver_have_no_state():
    grid = GridGraph(jnp.ones((3, 3)), fun=lambda x, y: (x + y) / 2)
    for distance in DISTANCES:
        assert distance.init(grid) is None
        assert distance.reinit(grid, None) is None


def grid_of(values):
    return GridGraph(values, fun=lambda x, y: (x + y) / 2)


# Large enough for AMJax to build a hierarchy of several levels, and small enough
# for a dense reference.
VALUES = jr.uniform(jr.key(0), (5, 5), minval=0.5, maxval=2.0, dtype=jnp.float64)
NEW_VALUES = VALUES**2 + 0.1
NODES = jnp.array([0, 12, 24, 12])
SOURCES = jnp.array([0, 7])
TARGETS = jnp.array([24, 3, 12])
QUERIES = {
    "all_pairs": lambda distance, grid, state: distance.all_pairs_distance(grid, state),
    "nodes_to_nodes": lambda distance, grid, state: distance.nodes_to_nodes_distance(
        grid, NODES, state
    ),
    "sources_to_targets": lambda distance, grid, state: (
        distance.sources_to_targets_distance(grid, SOURCES, TARGETS, state)
    ),
}
SPIELMAN = SpielmanApproximation(epsilon=0.5, seed=3)
SOLVER_DISTANCES = {
    "exact": lambda solver: ResistanceDistance(solver=solver),
    "spielman": lambda solver: ResistanceDistance(solver=solver, method=SPIELMAN),
    "free-energy": lambda solver: FreeEnergyDistance(0.7, solver=solver),
}


def make_solver(name):
    if name == "cg":
        return lx.CG(rtol=1e-13, atol=1e-13, max_steps=2000)
    if name == "dense-cholesky":
        return DenseCholesky()
    if name == "cholmod":
        pytest.importorskip("cholespy")
        return CholmodSolver()
    pytest.importorskip("pyamg")
    if name == "pyamg":
        return PyAMGSolver(rtol=1e-13, atol=1e-13, maxiter=500)
    pytest.importorskip("amjax")
    return AMJaxCGSolver(rtol=1e-13, atol=1e-13, max_steps=500)


def make_cudss():
    try:
        return CuDSSSolver()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")


def on_gpu(kind, query):
    return pytest.param(kind, "cudss", query, marks=pytest.mark.gpu)


# Each distance with the solvers that reuse a state, and each other solver with
# one distance. Queries take turns: they share their state handling.
LIFECYCLE_CASES = [
    ("exact", "amjax", "all_pairs"),
    ("exact", "dense-cholesky", "sources_to_targets"),
    on_gpu("exact", "nodes_to_nodes"),
    ("spielman", "amjax", "sources_to_targets"),
    ("spielman", "dense-cholesky", "all_pairs"),
    on_gpu("spielman", "sources_to_targets"),
    ("free-energy", "amjax", "nodes_to_nodes"),
    ("free-energy", "dense-cholesky", "nodes_to_nodes"),
    on_gpu("free-energy", "all_pairs"),
    ("exact", "cholmod", "nodes_to_nodes"),
    ("spielman", "pyamg", "all_pairs"),
    ("free-energy", "cg", "sources_to_targets"),
]


def assert_close(actual, expected):
    np.testing.assert_allclose(
        actual, expected, rtol=1e-7, atol=1e-7 * float(np.max(np.abs(expected)))
    )


def squared_distances(values, distance, query, state):
    return jnp.sum(QUERIES[query](distance, grid_of(values), state) ** 2)


@functools.cache
def dense_reference(kind, query, new_values):
    return jax.value_and_grad(squared_distances)(
        NEW_VALUES if new_values else VALUES, SOLVER_DISTANCES[kind](None), query, None
    )


@pytest.mark.parametrize("kind, solver, query", LIFECYCLE_CASES)
def test_state_lifecycle_matches_the_dense_reference(kind, solver, query):
    solver = make_cudss() if solver == "cudss" else make_solver(solver)
    distance = SOLVER_DISTANCES[kind](solver)

    traces = []

    @eqx.filter_jit
    def step(values, state):
        traces.append(1)

        def objective(values, state):
            state = distance.reinit(grid_of(values), state)
            return squared_distances(values, distance, query, state), state

        return eqx.filter_value_and_grad(objective, has_aux=True)(values, state)

    state = distance.init(grid_of(VALUES))
    # The first step solves the graph of the state, and the second a new graph.
    for new_values in (False, True):
        (value, state), gradient = step(NEW_VALUES if new_values else VALUES, state)

        expected_value, expected_gradient = dense_reference(kind, query, new_values)
        assert_close(value, expected_value)
        assert_close(gradient, expected_gradient)
    if isinstance(solver, AbstractReusableStateSolver):
        assert len(traces) == 1


@pytest.mark.parametrize("kind", ["exact", "free-energy"])
def test_amjax_states_serve_any_conductances(kind):
    distance = SOLVER_DISTANCES[kind](make_solver("amjax"))
    state = distance.init(grid_of(VALUES))

    actual = distance(grid_of(NEW_VALUES), nodes=NODES, state=state)

    assert_close(actual, SOLVER_DISTANCES[kind](None)(grid_of(NEW_VALUES), nodes=NODES))
