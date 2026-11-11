"""`CholmodSolver` reuses its factorization while the matrix values stay the same."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, ResistanceDistance, SpielmanApproximation
from jaxscape.solvers import BCOOLinearOperator, CholmodSolver, cholmodsolver


pytest.importorskip("cholespy")


VALUES = jnp.linspace(0.5, 2.0, 25, dtype=jnp.float64).reshape(5, 5)
NODES = jnp.array([0, 12, 24])


def grid_of(values):
    return GridGraph(values, fun=lambda x, y: (x + y) / 2)


@pytest.fixture
def factorizations(monkeypatch):
    """Count the native factorizations, which run on host callback threads."""
    calls = []
    factorize = cholmodsolver._factorize_host

    def counted(*args):
        calls.append(1)
        return factorize(*args)

    monkeypatch.setattr(cholmodsolver, "_factorize_host", counted)
    return calls


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_factorizations_follow_the_matrix_values(factorizations, compiled):
    distance = ResistanceDistance(solver=CholmodSolver())

    def query(values, state):
        return distance(grid_of(values), nodes=NODES, state=state)

    if compiled:
        query = jax.jit(query)
    state = distance.init(grid_of(VALUES))
    assert len(factorizations) == 1

    # Queries on the graph of the state share its factorization.
    query(VALUES, state).block_until_ready()
    query(VALUES, state).block_until_ready()
    assert len(factorizations) == 1

    # New values factorize once, and the state keeps that factorization.
    state = distance.reinit(grid_of(2 * VALUES), state)
    actual = query(2 * VALUES, state)
    query(2 * VALUES, state).block_until_ready()
    assert len(factorizations) == 2
    expected = ResistanceDistance(solver=None)(grid_of(VALUES), nodes=NODES) / 2
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_states_of_one_pattern_used_alternately_give_their_own_exact_results():
    distance = ResistanceDistance(solver=CholmodSolver())
    reference = ResistanceDistance(solver=None)
    values = {"first": VALUES, "second": 2 * VALUES[::-1]}
    states = {name: distance.init(grid_of(v)) for name, v in values.items()}

    for name in ["first", "second", "first", "second"]:
        actual = distance(grid_of(values[name]), nodes=NODES, state=states[name])

        expected = reference(grid_of(values[name]), nodes=NODES)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_a_used_up_state_stays_exact():
    distance = ResistanceDistance(solver=CholmodSolver())
    used_up = distance.init(grid_of(VALUES))
    other = 2 * VALUES[::-1]
    reinitialized = distance.reinit(grid_of(other), used_up)

    # The query with `reinitialized` uses up `used_up`.
    for values, state in [(other, reinitialized), (VALUES, used_up)]:
        actual = distance(grid_of(values), nodes=NODES, state=state)

        expected = ResistanceDistance(solver=None)(grid_of(values), nodes=NODES)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_adjoint_solves_reuse_the_forward_factorization(factorizations):
    method = SpielmanApproximation(epsilon=0.5, seed=3)
    distance = ResistanceDistance(solver=CholmodSolver(), method=method)
    state = distance.init(grid_of(VALUES))

    jax.grad(lambda v: jnp.sum(distance(grid_of(v), nodes=NODES, state=state)))(VALUES)

    assert len(factorizations) == 1


def test_a_deferred_factorization_runs_with_the_first_solve(factorizations):
    distance = ResistanceDistance(solver=CholmodSolver(factorize_in_init=False))
    state = distance.init(grid_of(VALUES))
    assert len(factorizations) == 0

    distance(grid_of(VALUES), nodes=NODES, state=state).block_until_ready()

    assert len(factorizations) == 1


def test_a_compiled_call_without_a_state_factorizes_each_time(factorizations):
    """Published benchmarks time this path, so a cache must not outlive its call."""
    distance = ResistanceDistance(solver=CholmodSolver())
    compiled = jax.jit(lambda values: distance(grid_of(values), nodes=NODES))

    compiled(VALUES).block_until_ready()
    compiled(VALUES).block_until_ready()

    assert len(factorizations) == 2


def test_solves_in_the_precision_of_the_right_hand_side():
    matrix = BCOO.fromdense(jnp.array([[2.0, -1.0], [-1.0, 2.0]], dtype=jnp.float64))
    solver = CholmodSolver()
    state = solver.init(BCOOLinearOperator(matrix), {})

    for dtype in (jnp.float64, jnp.float32):
        solution, _, _ = solver.compute(state, jnp.ones(2, dtype), {})

        np.testing.assert_allclose(solution, [1.0, 1.0], rtol=1e-6)
