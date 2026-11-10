"""`CholmodSolver` reuses its factorization while the matrix values stay the same."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import (
    ConnectivityAnalysis,
    GridGraph,
    ResistanceDistance,
    SensitivityAnalysis,
    SpielmanApproximation,
)
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


@pytest.fixture
def compilations():
    """Name the programs that JAX traces or compiles while the test runs."""
    events = {
        "/jax/core/compile/jaxpr_trace_duration",
        "/jax/core/compile/backend_compile_duration",
    }
    names = []

    def listen(event, duration_secs, **kwargs):
        del duration_secs
        if event in events:
            names.append(kwargs.get("fun_name"))

    jax.monitoring.register_event_duration_secs_listener(listen)
    yield names
    jax.monitoring.unregister_event_duration_listener(listen)


def windowed(kind, solver, seed):
    """A windowed analysis of random rasters of one shape, with the default
    proximity, whose profile a window query computes."""
    rng = np.random.default_rng(seed)
    return kind(
        quality_raster=jnp.asarray(rng.uniform(0, 1, (10, 11))),
        permeability_raster=jnp.asarray(10 ** rng.uniform(-1, 0, (10, 11))),
        distance=ResistanceDistance(solver=solver),
        dependency_range=3,
        batch_size=8,
    )


RUNS = {
    "connectivity": (ConnectivityAnalysis, lambda analysis: analysis.run()),
    "permeability-sensitivity": (
        SensitivityAnalysis,
        lambda analysis: analysis.run("permeability"),
    ),
}


@pytest.mark.parametrize("run", RUNS)
def test_a_new_analysis_reuses_the_compiled_programs(compilations, run):
    kind, run = RUNS[run]
    run(windowed(kind, CholmodSolver(), seed=0)).block_until_ready()
    expected = run(windowed(kind, None, seed=1))

    compilations.clear()
    actual = run(windowed(kind, CholmodSolver(), seed=1))

    assert compilations == []
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


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


def test_states_of_one_pattern_keep_their_own_factorizations(factorizations):
    distance = ResistanceDistance(solver=CholmodSolver())
    first = distance.init(grid_of(VALUES))
    distance(grid_of(VALUES), nodes=NODES, state=first).block_until_ready()

    # The query compiled for `first` solves with the factorization of `second`.
    second = distance.init(grid_of(2 * VALUES))
    actual = distance(grid_of(2 * VALUES), nodes=NODES, state=second)
    distance(grid_of(VALUES), nodes=NODES, state=first).block_until_ready()

    assert len(factorizations) == 2
    expected = ResistanceDistance(solver=None)(grid_of(VALUES), nodes=NODES) / 2
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_a_used_up_state_factorizes_again_and_stays_exact(factorizations):
    distance = ResistanceDistance(solver=CholmodSolver())
    used_up = distance.init(grid_of(VALUES))
    state = distance.reinit(grid_of(2 * VALUES), used_up)
    distance(grid_of(2 * VALUES), nodes=NODES, state=state).block_until_ready()

    actual = distance(grid_of(VALUES), nodes=NODES, state=used_up)

    assert len(factorizations) == 3
    expected = ResistanceDistance(solver=None)(grid_of(VALUES), nodes=NODES)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_a_state_evicted_by_newer_ones_factorizes_again_and_stays_exact(
    factorizations,
):
    distance = ResistanceDistance(solver=CholmodSolver())
    evicted = distance.init(grid_of(VALUES))
    # Documented: the process keeps the factorizations of the 8 most recently
    # used states.
    for scale in range(2, 10):
        distance.init(grid_of(scale * VALUES))

    actual = distance(grid_of(VALUES), nodes=NODES, state=evicted)

    assert len(factorizations) == 10
    expected = ResistanceDistance(solver=None)(grid_of(VALUES), nodes=NODES)
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
