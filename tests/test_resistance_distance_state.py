"""State-returning resistance-distance queries."""

from importlib.util import find_spec

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jaxscape import ExactResistance, GridGraph, SpielmanApproximation
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver


CHOLMOD_AVAILABLE = find_spec("cholespy") is not None
AMJAX_AVAILABLE = find_spec("pyamg") is not None and find_spec("amjax") is not None

NODES = jnp.array([0, 4, 8, 4], dtype=jnp.int32)
SOURCES = jnp.array([0, 5], dtype=jnp.int32)
TARGETS = jnp.array([8, 3, 5], dtype=jnp.int32)
QUERY_ARGS = {
    "all_pairs": (),
    "nodes_to_nodes": (NODES,),
    "sources_to_targets": (SOURCES, TARGETS),
}
METHODS = {
    "exact": ExactResistance(),
    "spielman": SpielmanApproximation(epsilon=0.5, seed=3),
}


def grid_graph(permeability):
    return GridGraph(permeability, fun=lambda x, y: (x + y) / 2)


def permeability(scale=1.0):
    return scale * jnp.linspace(0.5, 2.0, 9, dtype=jnp.float64).reshape(3, 3)


def query(distance, name, graph, state):
    method = getattr(distance, f"{name}_distance")
    return method(graph, *QUERY_ARGS[name], state)


def query_and_state(distance, name, graph, state):
    method = getattr(distance, f"{name}_distance_and_state")
    return method(graph, *QUERY_ARGS[name], state)


def _no_state(distance, graph):
    return None


def _full_state(distance, graph):
    return distance.init(graph)


def _preconditioner_state(distance, graph):
    return distance.init_preconditioner(graph)


SUPPLIED_STATE_CASES = [
    pytest.param(None, _no_state, id="no-solver"),
    pytest.param(lx.CG(rtol=1e-10, atol=1e-10), _full_state, id="cg"),
    pytest.param(
        CholmodSolver() if CHOLMOD_AVAILABLE else None,
        _full_state,
        id="cholmod",
        marks=pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="needs cholespy"),
    ),
    pytest.param(
        AMJaxCGSolver(rtol=1e-10, atol=1e-10, max_steps=100)
        if AMJAX_AVAILABLE
        else None,
        _full_state,
        id="amjax-full",
        marks=pytest.mark.skipif(not AMJAX_AVAILABLE, reason="needs amjax"),
    ),
    pytest.param(
        AMJaxCGSolver(rtol=1e-10, atol=1e-10, max_steps=100)
        if AMJAX_AVAILABLE
        else None,
        _preconditioner_state,
        id="amjax-preconditioner",
        marks=pytest.mark.skipif(not AMJAX_AVAILABLE, reason="needs amjax"),
    ),
]


@pytest.mark.parametrize("method", METHODS.values(), ids=METHODS.keys())
@pytest.mark.parametrize("name", QUERY_ARGS)
@pytest.mark.parametrize("solver, make_state", SUPPLIED_STATE_CASES)
def test_state_returning_query_matches_query_and_returns_supplied_state(
    solver, make_state, name, method
):
    distance = ResistanceDistance(solver=solver, method=method)
    graph = grid_graph(permeability())
    state = make_state(distance, graph)

    distances, returned_state = query_and_state(distance, name, graph, state)

    np.testing.assert_array_equal(distances, query(distance, name, graph, state))
    assert eqx.tree_equal(returned_state, state)


INITIALIZING_CASES = [
    pytest.param(None, id="no-solver"),
    pytest.param(lx.CG(rtol=1e-10, atol=1e-10), id="cg"),
    pytest.param(
        CholmodSolver() if CHOLMOD_AVAILABLE else None,
        id="cholmod",
        marks=pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="needs cholespy"),
    ),
]


@pytest.mark.parametrize("method", METHODS.values(), ids=METHODS.keys())
@pytest.mark.parametrize("name", QUERY_ARGS)
@pytest.mark.parametrize("solver", INITIALIZING_CASES)
def test_state_returning_query_without_state_returns_initialized_state(
    solver, name, method
):
    distance = ResistanceDistance(solver=solver, method=method)
    graph = grid_graph(permeability())

    distances, returned_state = query_and_state(distance, name, graph, None)

    np.testing.assert_array_equal(distances, query(distance, name, graph, None))
    # Queries are compiled, so they initialize the state from traced values.
    assert eqx.tree_equal(returned_state, eqx.filter_jit(distance.init)(graph))
