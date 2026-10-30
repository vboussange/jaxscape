"""`AbstractSparseDirectSolver` at its public boundaries.

Each test runs against `DenseCholesky`, on the CPU, and against `CuDSSSolver`, on
an NVIDIA GPU with the `gpu` marker. The second skips where the GPU is missing.
"""

import itertools

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from conftest import dense_cholesky, DenseCholesky
from jax.experimental.sparse import BCOO
from jaxscape import Graph, GridGraph, ResistanceDistance
from jaxscape.solvers import (
    AbstractSparseDirectSolver,
    BCOOLinearOperator,
    CuDSSSolver,
)
from jaxscape.solvers.operator import _batched_linear_solve, _linear_solve
from jaxscape.utils import graph_laplacian


CUDSS = pytest.param("cudss", marks=pytest.mark.gpu)


def make_solver(kind):
    if kind == "dense":
        return DenseCholesky()
    if kind == "dense-nan":
        return DenseCholesky(nan_pivots=True)
    try:
        return CuDSSSolver()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")


@pytest.fixture(params=["dense", CUDSS])
def solver(request):
    return make_solver(request.param)


def with_failing_solvers(*cases):
    """Pair each case with cuDSS on a GPU, and in turn with a CPU solver whose
    factorizations leave finite garbage or NaN, and poison their handle."""
    modes = itertools.cycle(["dense", "dense-nan"])
    params = []
    for case in cases:
        params.append(pytest.param(next(modes), *case))
        params.append(pytest.param("cudss", *case, marks=pytest.mark.gpu))
    return params


def grid_of(conductance):
    return GridGraph(conductance, fun=lambda x, y: (x + y) / 2)


def surface(name, shape=(6, 7)):
    """Small conductance rasters, from smooth to six orders of contrast."""
    rng = np.random.default_rng(0)
    if name == "smooth":
        values = np.linspace(0.5, 2.0, shape[0] * shape[1]).reshape(shape)
    elif name == "six_decades":
        values = 10.0 ** rng.uniform(-6.0, 0.0, shape)
    elif name == "barrier_with_floor":
        # A two-cell-wide barrier of zero conductance, raised to a 1e-6 floor.
        values = 10.0 ** rng.uniform(-2.0, 0.0, shape)
        values[:, 3:5] = 0.0
        values = np.maximum(values, 1e-6)
    elif name == "disconnected":
        # Without the floor, the vertices left of the barrier float, and the
        # grounded Laplacian is singular.
        values = np.array(surface("barrier_with_floor"))
        values[:, 3:5] = 0.0
    elif name == "negative":
        # The edges around this cell have negative conductances.
        values = np.array(surface("barrier_with_floor"))
        values[2, 2] = -10.0
    else:
        raise ValueError(name)
    return jnp.asarray(values, dtype=jnp.float64)


def distance_rtol(dtype, name="smooth"):
    """Relative tolerance of distances against the float64 dense reference."""
    if dtype == jnp.float64:
        return 1e-8
    # A 1e-6 floor across a two-cell barrier costs float32 most of its seven
    # digits: distances across it were 8.5e-2 off with cuDSS on an RTX 3090.
    return 2e-1 if name == "barrier_with_floor" else 1e-4


def grounded_laplacian(conductance):
    return graph_laplacian(grid_of(conductance).get_adjacency_matrix())[:-1, :-1]


@pytest.mark.parametrize(
    "compiled, dtype",
    [(False, jnp.float64), (True, jnp.float32)],
    ids=["eager-f64", "jit-f32"],
)
def test_distances_match_dense_reference_across_contrast(solver, compiled, dtype):
    distance = ResistanceDistance(solver=solver)
    nodes = jnp.array([0, 9, 41, 20, 9])

    def distances(grid, state):
        return distance(grid, nodes=nodes, state=state)

    if compiled:
        distances = jax.jit(distances)
    for name in ("smooth", "six_decades", "barrier_with_floor"):
        grid = grid_of(surface(name).astype(dtype))
        actual = distances(grid, distance.init(grid))

        expected = ResistanceDistance()(grid_of(surface(name)), nodes=nodes)
        assert actual.dtype == dtype
        rtol = distance_rtol(dtype, name)
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=rtol * float(np.max(expected))
        )


def test_a_single_edge_gives_the_exact_distance(solver):
    # The grounded Laplacian is 1 x 1.
    conductance = jnp.array([[0.0, 4.0], [4.0, 0.0]], dtype=jnp.float32)
    graph = Graph(BCOO.fromdense(conductance))
    distance = ResistanceDistance(solver=solver)

    actual = distance(graph, state=distance.init(graph))

    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, [[0.0, 0.25], [0.25, 0.0]], rtol=1e-6)


def ring_graph(chords):
    """A ring of 10 vertices with `chords`, of distinct conductances."""
    n = 10
    adjacency = np.zeros((n, n))
    edges = [(i, (i + 1) % n) for i in range(n)] + chords
    for k, (u, v) in enumerate(edges):
        adjacency[u, v] = adjacency[v, u] = 1.0 + k / 4
    return Graph(BCOO.fromdense(jnp.asarray(adjacency)))


def test_a_state_rejects_graphs_of_another_sparsity_pattern(solver):
    distance = ResistanceDistance(solver=solver)
    state = distance.init(ring_graph([(0, 5), (2, 7)]))
    # As many stored entries as the pattern of the state, at other positions.
    moved_chord = ring_graph([(0, 4), (2, 7)])
    # More stored entries.
    extra_chord = ring_graph([(0, 5), (2, 7), (4, 9)])

    for graph in (moved_chord, extra_chord):
        with pytest.raises(Exception, match="sparse indices differ"):
            jax.block_until_ready(distance(graph, state=state))


def sparse_layout(dense):
    """Store a symmetric matrix as JAXScape's sparse code paths do: unsorted,
    with duplicate entries splitting values, and padding entries interspersed.
    """
    n = dense.shape[0]
    rows, cols = np.nonzero(np.asarray(dense))
    values = np.asarray(dense)[rows, cols]
    split = np.arange(rows.size) % 3 == 0
    rows = np.concatenate([rows, rows[split], [n, n, n]])
    cols = np.concatenate([cols, cols[split], [n, n, n]])
    values = np.concatenate([values, values[split], [0.0, 0.0, 0.0]])
    values[: split.size][split] /= 2
    values[split.size : split.size + split.sum()] /= 2
    order = np.random.default_rng(1).permutation(rows.size)
    indices = np.stack([rows[order], cols[order]], axis=1)
    return BCOO(
        (jnp.asarray(values[order]), jnp.asarray(indices, dtype=jnp.int32)),
        shape=(n, n),
    )


@pytest.mark.parametrize("n_rhs", [None, 4])
def test_sparse_solves_and_gradients_match_dense_reference(solver, n_rhs):
    """Values and gradients with respect to the stored matrix values and to the
    right-hand sides, for one right-hand side and for vmapped batches."""
    n = 9
    matrix = (
        np.diag(np.linspace(2.5, 4.0, n))
        - np.diag(np.full(n - 1, 1.0), 1)
        - np.diag(np.full(n - 1, 1.0), -1)
    )
    A = sparse_layout(matrix)
    state = solver.init(BCOOLinearOperator(A), {})
    rng = np.random.default_rng(3)
    rhs_shape = (n,) if n_rhs is None else (n, n_rhs)
    b = jnp.asarray(rng.standard_normal(rhs_shape))
    weights = jnp.asarray(rng.standard_normal(rhs_shape))

    def sparse_solve(A, b):
        solve = _linear_solve if n_rhs is None else _batched_linear_solve
        return solve(A, b, solver, state=state)

    def gradients(solve):
        def objective(data, b):
            return jnp.sum(weights * solve(BCOO((data, A.indices), shape=A.shape), b))

        return jax.grad(objective, argnums=(0, 1))(A.data, b)

    expected_solution = np.linalg.solve(matrix, b)
    np.testing.assert_allclose(sparse_solve(A, b), expected_solution, rtol=1e-12)
    actual = gradients(sparse_solve)
    expected = gradients(lambda A, b: jnp.linalg.solve(A.todense(), b))
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-10, atol=1e-12)


def test_reinit_reuses_the_analysis_and_keeps_unchanged_values():
    solver = DenseCholesky()
    distance = ResistanceDistance(solver=solver)
    smooth = surface("smooth")
    state = distance.init(grid_of(smooth))
    assert state.factorization.refactorizations == 0

    # Unchanged values keep the factorization.
    state = distance.reinit(grid_of(smooth), state)
    assert state.factorization.refactorizations == 0
    # So do differences within the rounding of assembling the matrix.
    state = distance.reinit(grid_of(smooth * (1 + 2e-16)), state)
    assert state.factorization.refactorizations == 0

    # Each change of values refactorizes on the same analysis.
    for scale, count in ((2.0, 1), (3.0, 2)):
        state = distance.reinit(grid_of(scale * smooth), state)
        assert state.factorization.refactorizations == count


def test_reinit_compares_each_value_with_the_factorized_one():
    distance = ResistanceDistance(solver=DenseCholesky())
    smooth = surface("smooth")
    eps = jnp.finfo(smooth.dtype).eps

    # Each of these steps stays within rounding of the last, and together they
    # leave the values that were factorized.
    state = distance.init(grid_of(smooth))
    state = distance.reinit(grid_of(smooth * (1 + 12 * eps)), state)
    assert state.factorization.refactorizations == 0
    state = distance.reinit(grid_of(smooth * (1 + 24 * eps)), state)
    assert state.factorization.refactorizations == 1

    # A change far below the largest value is a change of its own value.
    tiny = smooth.at[0, :2].set(1e-16)
    state = distance.init(grid_of(tiny))
    state = distance.reinit(grid_of(tiny.at[0, :2].set(3e-16)), state)
    assert state.factorization.refactorizations == 1


class MinimalCholesky(AbstractSparseDirectSolver):
    """The least a backend implements: it has no `refactorize`."""

    def factorize(self, pattern, values):
        return dense_cholesky(pattern.n, pattern.rows, pattern.columns, values)

    def solve(self, factorization, vector):
        return jax.scipy.linalg.cho_solve((factorization, True), vector)

    def residual_tolerance(self, dtype):
        return 1e-4


def test_a_backend_needs_only_factorize_solve_and_a_tolerance():
    distance = ResistanceDistance(solver=MinimalCholesky())
    smooth = surface("smooth")
    nodes = jnp.array([0, 9, 41])

    state = distance.reinit(grid_of(2 * smooth), distance.init(grid_of(smooth)))
    actual = distance(grid_of(2 * smooth), nodes=nodes, state=state)

    expected = ResistanceDistance()(grid_of(2 * smooth), nodes=nodes)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_operators_without_a_bcoo_matrix_are_rejected():
    operator = lx.MatrixLinearOperator(jnp.eye(2), lx.positive_semidefinite_tag)
    with pytest.raises(TypeError, match="DenseCholesky requires a BCOO"):
        DenseCholesky().init(operator, {})


def test_a_query_on_a_new_graph_solves_it_with_an_outdated_state(solver):
    distance = ResistanceDistance(solver=solver)
    nodes = jnp.array([0, 9, 41])
    outdated = distance.init(grid_of(surface("smooth")))
    query = jax.jit(lambda grid, state: distance(grid, nodes=nodes, state=state))

    # The first query uses up the factorization of `outdated`, on cuDSS.
    for name in ("six_decades", "barrier_with_floor", "six_decades"):
        actual = query(grid_of(surface(name)), outdated)

        expected = ResistanceDistance()(grid_of(surface(name)), nodes=nodes)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-12)


def test_queries_without_a_state_initialize_one_outside_jit(solver):
    distance = ResistanceDistance(solver=solver)
    grid = grid_of(surface("smooth"))
    nodes = jnp.array([0, 9, 41])

    np.testing.assert_allclose(
        distance(grid, nodes=nodes),
        ResistanceDistance()(grid, nodes=nodes),
        rtol=1e-8,
    )
    with pytest.raises(ValueError, match=r"outside `jax\.jit`.*distance\.init"):
        jax.jit(lambda grid: distance(grid, nodes=nodes))(grid)


def raises_failed_solve():
    """Expect Lineax's error for a failed solve.

    Its exception type depends on the compilation wrapper and the JAX version.
    """
    return pytest.raises(Exception, match="linear solver returned non-finite")


@pytest.mark.parametrize(
    "kind, name, compiled, dtype",
    with_failing_solvers(
        ("disconnected", False, jnp.float64),
        ("negative", True, jnp.float32),
    ),
)
def test_a_failed_factorization_raises(kind, name, compiled, dtype):
    grid = grid_of(surface(name).astype(dtype))
    distance = ResistanceDistance(solver=make_solver(kind))
    # Nothing reports a failure during initialization.
    state = distance.init(grid)

    def distances(grid, state):
        return distance(grid, state=state)

    if compiled:
        distances = jax.jit(distances)
    with raises_failed_solve():
        jax.block_until_ready(distances(grid, state))


@pytest.mark.parametrize(
    "kind, name, dtype",
    with_failing_solvers(
        ("disconnected", jnp.float32),
        ("negative", jnp.float64),
    ),
)
def test_a_failed_refactorization_raises_mid_optimization(kind, name, dtype):
    # Pivots of this surface stay above cuDSS's float32 pivot epsilon.
    valid = surface("smooth").astype(dtype)
    distance = ResistanceDistance(solver=make_solver(kind))
    state = distance.init(grid_of(valid))

    @jax.jit
    def step(conductance, state):
        grid = grid_of(conductance)
        state = distance.reinit(grid, state)
        return distance(grid, state=state), state

    _, state = step(2 * valid, state)
    with raises_failed_solve():
        jax.block_until_ready(step(surface(name).astype(dtype), state))


def test_a_failed_refactorization_is_retried_on_a_fresh_analysis():
    """A factorization can pass its check and still leave its handle unusable. The
    next refactorization then fails its check, and is retried on a new analysis."""
    smooth = surface("smooth")
    solver = DenseCholesky(poison_at=1)
    operators = [BCOOLinearOperator(grounded_laplacian(k * smooth)) for k in (1, 2, 3)]
    b = jnp.ones(operators[0].in_size())

    state = solver.init(operators[0], {})
    state = solver.reinit(operators[1], {}, state)
    assert not state.failed and state.factorization.poisoned
    state = solver.reinit(operators[2], {}, state)

    assert not state.failed and state.factorization.refactorizations == 0
    solution = lx.linear_solve(operators[2], b, solver, state=state)
    expected = np.linalg.solve(np.asarray(operators[2].matrix.todense()), b)
    np.testing.assert_allclose(solution.value, expected, rtol=1e-8)


@pytest.mark.parametrize(
    "kind, name, reinitialized, dtype",
    with_failing_solvers(
        ("disconnected", False, jnp.float64),
        ("negative", True, jnp.float64),
        ("negative", False, jnp.float32),
    ),
)
def test_without_throw_a_failure_returns_a_failed_result_and_recovers(
    kind, name, reinitialized, dtype
):
    solver = make_solver(kind)
    invalid = BCOOLinearOperator(grounded_laplacian(surface(name).astype(dtype)))
    valid_matrix = grounded_laplacian(surface("smooth").astype(dtype))
    valid = BCOOLinearOperator(valid_matrix)
    # Current enters at every vertex, the floating ones included.
    b = jnp.ones(invalid.in_size(), dtype)

    @jax.jit
    def solve(operator, state):
        solution = lx.linear_solve(operator, b, solver, state=state, throw=False)
        return solution.value, solution.result

    reinit = jax.jit(lambda operator, state: solver.reinit(operator, {}, state))
    if reinitialized:
        state = reinit(invalid, solver.init(valid, {}))
    else:
        state = solver.init(invalid, {})

    value, result = solve(invalid, state)

    assert result != lx.RESULTS.successful
    assert np.all(np.isnan(value))
    # The next factorization, of valid values, is exact.
    value, result = solve(valid, reinit(valid, state))
    assert result == lx.RESULTS.successful
    expected = np.linalg.solve(np.asarray(valid_matrix.todense(), np.float64), b)
    np.testing.assert_allclose(value, expected, rtol=distance_rtol(dtype))
