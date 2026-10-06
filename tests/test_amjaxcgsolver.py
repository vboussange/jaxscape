import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape.solvers import (
    amjax_preconditioner_operator,
    AMJaxCGSolver,
    batched_linear_solve,
    BCOOLinearOperator,
    build_amjax_solver,
)


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
    """Current matrix and RHS derivatives must not use the reference operator."""
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
        state = solver.init_preconditioner(BCOOLinearOperator(reference))
        assert not state.has_cg_state

        def solve(data, rhs):
            matrix = BCOO((data, current.indices), shape=current.shape)
            return batched_linear_solve(matrix, rhs, solver, state=state)

        def dense_solve(data, rhs):
            matrix = BCOO((data, current.indices), shape=current.shape)
            return jnp.linalg.solve(matrix.todense(), rhs)

        actual = jax.jit(solve)(current.data, rhs)
        expected = dense_solve(current.data, rhs)
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)

        def objective(data, rhs):
            return jnp.sum(solve(data, rhs) ** 2)

        def dense_objective(data, rhs):
            return jnp.sum(dense_solve(data, rhs) ** 2)

        value, grads = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
            current.data, rhs
        )
        expected_value, expected_grads = jax.value_and_grad(
            dense_objective, argnums=(0, 1)
        )(current.data, rhs)
        np.testing.assert_allclose(value, expected_value, rtol=1e-8)
        for actual_grad, expected_grad in zip(grads, expected_grads):
            np.testing.assert_allclose(
                actual_grad, expected_grad, rtol=1e-8, atol=1e-9
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
        state = solver.init_preconditioner(operator)

        @jax.jit
        def solve(rhs):
            return batched_linear_solve(matrix, rhs, solver, state=state)

        np.testing.assert_allclose(
            solve(rhs), jnp.linalg.solve(matrix.todense(), rhs), rtol=1e-8, atol=1e-9
        )
        gradient = jax.jit(jax.grad(lambda b: jnp.sum(solve(b))))(rhs)
        expected = jnp.linalg.solve(matrix.todense().T, jnp.ones_like(rhs))
        np.testing.assert_allclose(gradient, expected, rtol=1e-8, atol=1e-9)


def test_missing_amjax_dependency_reports_extra(monkeypatch):
    from jaxscape.solvers import amjaxcgsolver

    monkeypatch.setattr(amjaxcgsolver, "AMJAX_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"jaxscape\[amjax\]"):
        AMJaxCGSolver()
    with pytest.raises(ImportError, match=r"jaxscape\[amjax\]"):
        build_amjax_solver(None)
