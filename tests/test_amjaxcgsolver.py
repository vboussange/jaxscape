import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator, SparseIterativeState
from jaxscape.solvers.operator import _batched_linear_solve


def avg(x, y):
    return (x + y) / 2


@pytest.fixture
def amjax_dependencies():
    pytest.importorskip("amjax")
    return pytest.importorskip("pyamg")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hierarchy_preserves_matrix_dtype(amjax_dependencies, dtype):
    matrix = BCOO.from_scipy_sparse(
        amjax_dependencies.gallery.poisson((6, 6), format="csr", dtype=dtype)
    )
    solver = AMJaxCGSolver(coarse_solver="pinv")
    state = solver.init(BCOOLinearOperator(matrix), {})

    assert isinstance(state, SparseIterativeState)
    assert len(state.preconditioner.levels) > 1
    for leaf in jax.tree.leaves(state.preconditioner):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            assert leaf.dtype == dtype


def test_reinitialized_state_jitted_values_and_gradients(amjax_dependencies):
    """Solves and resistance gradients with the state of other values match the
    exact ones."""
    scipy_matrix = amjax_dependencies.gallery.poisson(
        (6, 6), format="csr", dtype=np.float64
    )
    reference = BCOO.from_scipy_sparse(scipy_matrix)
    scipy_matrix.setdiag(scipy_matrix.diagonal() + np.linspace(0.1, 0.5, 36))
    current = BCOO.from_scipy_sparse(scipy_matrix)
    rhs = jnp.stack([jnp.ones(36), jnp.linspace(-1.0, 2.0, 36)], axis=1)
    solver = AMJaxCGSolver(rtol=1e-10, max_steps=500, coarse_solver="pinv")
    state = solver.init(BCOOLinearOperator(reference), {})

    def solve(data, rhs):
        matrix = BCOO((data, current.indices), shape=current.shape)
        return _batched_linear_solve(matrix, rhs, solver, state=state)

    actual = jax.jit(solve)(current.data, rhs)
    expected = jnp.linalg.solve(current.todense(), rhs)
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)

    permeability = jnp.ones((6, 6), dtype=jnp.float64)
    distance = ResistanceDistance(solver=solver)
    resistance_state = distance.init(GridGraph(permeability, fun=avg))
    nodes = jnp.array([0, 12, 35, 12], dtype=jnp.int32)

    def objective(values, distance, state=None):
        graph = GridGraph(values, fun=avg)
        return jnp.sum(distance(graph, nodes=nodes, state=state) ** 2)

    current_permeability = permeability + jnp.linspace(0.1, 0.5, 36).reshape(6, 6)
    value, gradient = jax.jit(
        jax.value_and_grad(lambda p: objective(p, distance, resistance_state))
    )(current_permeability)
    expected_value, expected_gradient = jax.value_and_grad(objective)(
        current_permeability, ResistanceDistance(solver=None)
    )
    np.testing.assert_allclose(value, expected_value, rtol=1e-8)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("size", [1, 4, 9])
def test_single_level_hierarchy_runs_without_a_preconditioner(amjax_dependencies, size):
    matrix = BCOO.from_scipy_sparse(
        amjax_dependencies.gallery.poisson((size,), format="csr", dtype=np.float64)
    )
    rhs = jnp.stack([jnp.ones(size), jnp.arange(size, dtype=jnp.float64)], axis=1)
    solver = AMJaxCGSolver(rtol=1e-10, max_steps=50)
    state = solver.init(BCOOLinearOperator(matrix), {})
    assert len(state.preconditioner.levels) == 1

    @jax.jit
    def solve(rhs):
        return _batched_linear_solve(matrix, rhs, solver, state=state)

    expected = jnp.linalg.solve(matrix.todense(), rhs)
    np.testing.assert_allclose(solve(rhs), expected, rtol=1e-8, atol=1e-9)

    permeability = jnp.linspace(0.8, 1.4, size + 1).reshape(1, -1)
    distance = ResistanceDistance(solver=solver)
    resistance_state = distance.init(GridGraph(permeability, fun=avg))

    def objective(values, distance, state=None):
        return jnp.sum(distance(GridGraph(values, fun=avg), state=state))

    gradient = jax.jit(jax.grad(lambda p: objective(p, distance, resistance_state)))(
        permeability
    )
    expected = jax.grad(objective)(permeability, ResistanceDistance(solver=None))
    np.testing.assert_allclose(gradient, expected, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize("dtype, rtol", [(jnp.float32, 1e-4), (jnp.float64, 1e-7)])
def test_reinit_solves_high_contrast_values_in_few_steps(
    amjax_dependencies, dtype, rtol
):
    """The state of a graph of uniform conductance, as windowed analyses build
    it, solves a graph of conductance contrast 1e4 in few iterations."""
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-4, 0, (24, 24)))
    nodes = jnp.array([0, 100, 300, 575])
    distance = ResistanceDistance(solver=AMJaxCGSolver(max_steps=60))
    state = distance.init(GridGraph(jnp.ones((24, 24), dtype), fun=avg))
    graph = GridGraph(permeability.astype(dtype), fun=avg)

    actual = distance(graph, nodes=nodes, state=distance.reinit(graph, state))
    expected = ResistanceDistance(solver=None)(
        GridGraph(permeability, fun=avg), nodes=nodes
    )
    np.testing.assert_allclose(actual, expected, rtol=rtol)


def test_reinitialized_levels_at_high_contrast(amjax_dependencies):
    """With the state of a graph of uniform conductance and the values of a
    graph of conductance contrast 1e8, the matrix of each level is the Galerkin
    product `R A P` of the level above, and the V-cycle is symmetric positive
    definite."""
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-8, 0, (24, 24)))
    distance = ResistanceDistance()
    uniform = distance.system_matrix(GridGraph(jnp.ones_like(permeability), fun=avg))
    matrix = distance.system_matrix(GridGraph(permeability, fun=avg))
    solver = AMJaxCGSolver()
    state = solver.init(BCOOLinearOperator(uniform), {})
    state = solver.reinit(BCOOLinearOperator(matrix), {}, state)

    expected = np.asarray(matrix.todense())
    for level in state.preconditioner.multilevel_solver(state.values).levels:
        scale = np.abs(expected).max()
        np.testing.assert_allclose(level.A.todense(), expected, atol=1e-14 * scale)
        np.testing.assert_allclose(level.Dinv, 1 / np.diag(expected), rtol=1e-12)
        if level.P is not None:
            expected = level.R.todense() @ expected @ level.P.todense()

    v_cycle = solver.precondition(state.preconditioner, state.pattern, state.values)
    approximate_inverse = np.asarray(v_cycle(jnp.eye(state.pattern.n)))
    scale = np.abs(approximate_inverse).max()
    np.testing.assert_allclose(
        approximate_inverse, approximate_inverse.T, atol=1e-12 * scale
    )
    assert np.linalg.eigvalsh(approximate_inverse).min() > 0


def test_state_built_inside_jit_solves_the_graph(amjax_dependencies):
    """Inside `jax.jit`, the aggregation of the hierarchy is that of a graph of
    permeability 1."""
    rng = np.random.default_rng(0)
    permeability = jnp.asarray(10 ** rng.uniform(-1, 0, (8, 9)))
    nodes = jnp.array([0, 30, 71])
    solver = AMJaxCGSolver(rtol=1e-10, coarse_solver="pinv")

    def query(permeability, distance):
        return distance(GridGraph(permeability, fun=avg), nodes=nodes)

    actual = jax.jit(lambda p: query(p, ResistanceDistance(solver=solver)))(
        permeability
    )
    expected = query(permeability, ResistanceDistance(solver=None))
    np.testing.assert_allclose(actual, expected, rtol=1e-7)


def test_missing_amjax_dependency_reports_extra(monkeypatch):
    from jaxscape.solvers import amjaxcgsolver

    monkeypatch.setattr(amjaxcgsolver, "AMJAX_AVAILABLE", False)
    with pytest.raises(ImportError, match="--extra amjax"):
        AMJaxCGSolver()
