import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import (
    amjax_preconditioner_operator,
    AMJaxCGSolver,
    AMJaxCGSolverState,
    BCOOLinearOperator,
    build_amjax_solver,
)
from jaxscape.solvers.operator import _batched_linear_solve


@pytest.fixture
def amjax_dependencies():
    pytest.importorskip("amjax")
    return pytest.importorskip("pyamg")


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_hierarchy_preserves_matrix_dtype(amjax_dependencies, dtype):
    with jax.enable_x64():
        matrix = BCOO.from_scipy_sparse(
            amjax_dependencies.gallery.poisson((6, 6), format="csr", dtype=dtype)
        )
        hierarchy = build_amjax_solver(matrix, coarse_solver="pinv")

        assert len(hierarchy.levels) > 1
        for leaf in jax.tree.leaves(hierarchy):
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.inexact):
                assert leaf.dtype == dtype


def test_frozen_preconditioner_jitted_values_and_gradients(amjax_dependencies):
    """Current matrix solves and resistance gradients reuse a frozen hierarchy."""
    with jax.enable_x64():
        scipy_matrix = amjax_dependencies.gallery.poisson(
            (6, 6), format="csr", dtype=np.float64
        )
        reference = BCOO.from_scipy_sparse(scipy_matrix)
        scipy_matrix.setdiag(scipy_matrix.diagonal() + np.linspace(0.1, 0.5, 36))
        current = BCOO.from_scipy_sparse(scipy_matrix)
        rhs = jnp.stack([jnp.ones(36), jnp.linspace(-1.0, 2.0, 36)], axis=1)
        solver = AMJaxCGSolver(
            rtol=1e-10, atol=1e-10, max_steps=500, coarse_solver="pinv"
        )
        state = solver.init(BCOOLinearOperator(reference), {})

        def solve(data, rhs):
            matrix = BCOO((data, current.indices), shape=current.shape)
            return _batched_linear_solve(matrix, rhs, solver, state=state)

        def dense_solve(data, rhs):
            matrix = BCOO((data, current.indices), shape=current.shape)
            return jnp.linalg.solve(matrix.todense(), rhs)

        actual = jax.jit(solve)(current.data, rhs)
        expected = dense_solve(current.data, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)

        permeability = jnp.ones((6, 6), dtype=jnp.float64)
        distance = ResistanceDistance(solver=solver)
        graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        resistance_state = distance.init(graph)
        nodes = jnp.array([0, 12, 35, 12], dtype=jnp.int32)

        def objective(values):
            graph = GridGraph(values, fun=lambda x, y: (x + y) / 2)
            return jnp.sum(distance(graph, nodes=nodes, state=resistance_state) ** 2)

        def dense_objective(values):
            graph = GridGraph(values, fun=lambda x, y: (x + y) / 2)
            return jnp.sum(ResistanceDistance()(graph, nodes=nodes) ** 2)

        current_permeability = permeability + jnp.linspace(0.1, 0.5, 36).reshape(6, 6)
        value, gradient = jax.jit(jax.value_and_grad(objective))(current_permeability)
        expected_value, expected_gradient = jax.value_and_grad(dense_objective)(
            current_permeability
        )
        np.testing.assert_allclose(value, expected_value, rtol=1e-8)
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=1e-8, atol=1e-9
        )


@pytest.mark.parametrize("size", [1, 4, 9])
def test_single_level_hierarchy_uses_cg(amjax_dependencies, size):
    with jax.enable_x64():
        matrix = BCOO.from_scipy_sparse(
            amjax_dependencies.gallery.poisson((size,), format="csr", dtype=np.float64)
        )
        hierarchy = build_amjax_solver(matrix)
        assert len(hierarchy.levels) == 1
        rhs = jnp.stack([jnp.ones(size), jnp.arange(size, dtype=jnp.float64)], axis=1)
        operator = BCOOLinearOperator(matrix)
        preconditioner = amjax_preconditioner_operator(
            hierarchy, operator.in_structure()
        )
        assert isinstance(preconditioner, lx.IdentityLinearOperator)
        solver = AMJaxCGSolver(rtol=1e-10, atol=1e-10, max_steps=50)
        state = solver.init(operator, {})

        @jax.jit
        def solve(rhs):
            return _batched_linear_solve(matrix, rhs, solver, state=state)

        np.testing.assert_allclose(
            solve(rhs), jnp.linalg.solve(matrix.todense(), rhs), rtol=1e-8, atol=1e-9
        )
        permeability = jnp.linspace(0.8, 1.4, size + 1).reshape(1, -1)
        distance = ResistanceDistance(solver=solver)
        graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
        resistance_state = distance.init(graph)

        def objective(values, distance, state=None):
            graph = GridGraph(values, fun=lambda x, y: (x + y) / 2)
            return jnp.sum(distance(graph, state=state))

        gradient = jax.jit(jax.grad(
            lambda values: objective(values, distance, resistance_state)
        ))(permeability)
        expected = jax.grad(
            lambda values: objective(values, ResistanceDistance())
        )(permeability)
        np.testing.assert_allclose(gradient, expected, rtol=1e-8, atol=1e-9)


def test_missing_amjax_dependency_reports_extra(monkeypatch):
    from jaxscape.solvers import amjaxcgsolver

    monkeypatch.setattr(amjaxcgsolver, "AMJAX_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"jaxscape\[amjax\]"):
        AMJaxCGSolver()
    with pytest.raises(ImportError, match=r"jaxscape\[amjax\]"):
        build_amjax_solver(None)


def test_deprecated_state_methods_are_aliases(amjax_dependencies):
    matrix = BCOO.from_scipy_sparse(
        amjax_dependencies.gallery.poisson((6, 6), format="csr", dtype=np.float64)
    )
    operator = BCOOLinearOperator(matrix)
    solver = AMJaxCGSolver(rtol=1e-8, atol=1e-8, max_steps=100)
    distance = ResistanceDistance(solver=solver)
    grid = GridGraph(jnp.ones((3, 3)), fun=lambda x, y: (x + y) / 2)

    with pytest.warns(DeprecationWarning, match="use `init`"):
        state = solver.init_preconditioner(operator)
    with pytest.warns(DeprecationWarning, match="use `reinit`"):
        state = solver.materialize_state(operator, {}, state)
    with pytest.warns(DeprecationWarning, match="use `init`"):
        distance_state = distance.init_preconditioner(grid)

    assert isinstance(state, AMJaxCGSolverState)
    assert isinstance(distance_state, AMJaxCGSolverState)
