"""State-returning resistance-distance queries.

All queries share one code path, so each solver case runs one query and
method, and the cases of each test cycle through every query and method.
"""

import itertools
from importlib.util import find_spec
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax import Array
from jaxscape import ExactResistance, GridGraph, SpielmanApproximation
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver
from jaxscape.solvers.operator import AbstractRefreshingSolver


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


def paired_with_queries(cases):
    """Pair each case with a query name and method, cycling through all six."""
    pairs = itertools.cycle(itertools.product(QUERY_ARGS, METHODS))
    return [
        pytest.param(
            *case.values,
            name,
            METHODS[method],
            id=f"{case.id}-{name}-{method}",
            marks=case.marks,
        )
        for case, (name, method) in zip(cases, pairs)
    ]


def grid_graph(permeability):
    return GridGraph(permeability, fun=lambda x, y: (x + y) / 2)


def permeability(power=1.0):
    return jnp.linspace(0.5, 2.0, 9, dtype=jnp.float64).reshape(3, 3) ** power


def query(distance, name, graph, state):
    method = getattr(distance, f"{name}_distance")
    return method(graph, *QUERY_ARGS[name], state)


def query_and_state(distance, name, graph, state):
    method = getattr(distance, f"{name}_distance_and_state")
    return method(graph, *QUERY_ARGS[name], state)


@final
class _CountingState(eqx.Module):
    matrix: Array
    refreshes: Array


@final
class _RefreshCountingSolver(AbstractRefreshingSolver):
    """Dense solver whose refresh step builds a new state for each operator."""

    def init(self, operator, options):
        del options
        return _CountingState(operator.as_matrix().todense(), jnp.array(0))

    def refresh_state(self, operator, options, state):
        del options
        return _CountingState(operator.as_matrix().todense(), state.refreshes + 1)

    def compute(self, state, vector, options):
        del options
        return jnp.linalg.solve(state.matrix, vector), lx.RESULTS.successful, {}

    def transpose(self, state, options):
        return _CountingState(state.matrix.T, state.refreshes), options

    def conj(self, state, options):
        return state, options

    def assume_full_rank(self):
        return True


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


@pytest.mark.parametrize(
    "solver, make_state, name, method", paired_with_queries(SUPPLIED_STATE_CASES)
)
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
    pytest.param(_RefreshCountingSolver(), id="refreshing"),
]


@pytest.mark.parametrize(
    "solver, name, method", paired_with_queries(INITIALIZING_CASES)
)
def test_state_returning_query_without_state_starts_from_init(solver, name, method):
    distance = ResistanceDistance(solver=solver, method=method)
    graph = grid_graph(permeability())

    distances, returned_state = query_and_state(distance, name, graph, None)

    # As `distance(graph)` does, the query starts from `distance.init(graph)`.
    expected, expected_state = query_and_state(
        distance, name, graph, distance.init(graph)
    )
    np.testing.assert_array_equal(distances, expected)
    np.testing.assert_array_equal(distances, query(distance, name, graph, None))
    # Array for array: a host factorization is a new object at each `init`.
    assert eqx.tree_equal(array_leaves(returned_state), array_leaves(expected_state))


@pytest.mark.parametrize(
    "name, method",
    [
        pytest.param("all_pairs", METHODS["exact"], id="all_pairs-exact"),
        pytest.param(
            "sources_to_targets", METHODS["spielman"], id="sources_to_targets-spielman"
        ),
    ],
)
def test_state_returning_query_returns_refreshed_state(name, method):
    distance = ResistanceDistance(solver=_RefreshCountingSolver(), method=method)
    reference = ResistanceDistance(method=method)
    state = distance.init(grid_graph(permeability()))

    for step, power in enumerate([2.0, -1.0], start=1):
        graph = grid_graph(permeability(power))
        distances, state = query_and_state(distance, name, graph, state)

        # Solves use the state refreshed for the current graph, not the
        # supplied one, and the query hands that refreshed state back.
        assert state.refreshes == step
        np.testing.assert_allclose(
            distances, query(reference, name, graph, None), rtol=1e-10, atol=1e-12
        )


def array_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]


def _sum_inexact_leaves(tree):
    leaves = jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
    return sum((jnp.sum(leaf) for leaf in leaves), jnp.zeros(()))


# Supplied states that come back unchanged are covered by the default solver
# and the AMJax preconditioner, whose solves use a completed state. The other
# cases return a state the call builds from the differentiated graph.
DIFFERENTIATED_CASES = [
    SUPPLIED_STATE_CASES[0],
    SUPPLIED_STATE_CASES[-1],
    pytest.param(lx.CG(rtol=1e-10, atol=1e-10), _no_state, id="cg-initialized"),
    pytest.param(
        CholmodSolver() if CHOLMOD_AVAILABLE else None,
        _no_state,
        id="cholmod-initialized",
        marks=pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="needs cholespy"),
    ),
    pytest.param(_RefreshCountingSolver(), _full_state, id="refreshing"),
    pytest.param(_RefreshCountingSolver(), _no_state, id="refreshing-initialized"),
]


@pytest.mark.parametrize(
    "solver, make_state, name, method", paired_with_queries(DIFFERENTIATED_CASES)
)
def test_state_returning_query_gradients_ignore_returned_state(
    solver, make_state, name, method
):
    distance = ResistanceDistance(solver=solver, method=method)
    state = make_state(distance, grid_graph(permeability()))

    def loss(values, state):
        return jnp.sum(query(distance, name, grid_graph(values), state) ** 2)

    def loss_with_state(values, state):
        distances, state = query_and_state(distance, name, grid_graph(values), state)
        # Any gradient through the returned state would change the total.
        return jnp.sum(distances**2) + _sum_inexact_leaves(state), state

    values = permeability(2.0)
    gradient, returned_state = eqx.filter_jit(
        eqx.filter_grad(loss_with_state, has_aux=True)
    )(values, state)
    expected_gradient = eqx.filter_jit(eqx.filter_grad(loss))(values, state)

    np.testing.assert_array_equal(gradient, expected_gradient)
    _, expected_state = eqx.filter_jit(query_and_state)(
        distance, name, grid_graph(values), state
    )
    assert eqx.tree_equal(returned_state, expected_state)
