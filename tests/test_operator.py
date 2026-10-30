from importlib.util import find_spec

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jaxscape.solvers import (
    amjax_preconditioner_operator,
    AMJaxCGSolver,
    BCOOLinearOperator,
    build_amjax_solver,
    CholmodSolver,
    PyAMGSolver,
)
from jaxscape.solvers.operator import (
    _batched_linear_solve,
    _linear_solve,
    _solver_state,
)


# Check availability of optional solvers
PYAMG_AVAILABLE = find_spec("pyamg") is not None
CHOLMOD_AVAILABLE = find_spec("cholespy") is not None
AMJAX_AVAILABLE = PYAMG_AVAILABLE and find_spec("amjax") is not None

if PYAMG_AVAILABLE:
    from pyamg.gallery import poisson
else:
    poisson = None

# Build list of available solvers
available_solvers = []
if PYAMG_AVAILABLE:
    solver = PyAMGSolver()
    available_solvers.append(solver)
if CHOLMOD_AVAILABLE:
    solver = CholmodSolver()
    available_solvers.append(solver)
if AMJAX_AVAILABLE:
    solver = AMJaxCGSolver(rtol=1e-5, atol=1e-5, max_steps=500)
    available_solvers.append(solver)


def test_bcoo_linear_operator_mv():
    """Test BCOOLinearOperator.mv method by comparing to dense matrix multiplication."""
    key = jax.random.PRNGKey(42)
    dense_matrix = jax.random.normal(key, (5, 5))
    bcoo_matrix = BCOO.fromdense(dense_matrix)
    operator = BCOOLinearOperator(bcoo_matrix)
    vector = jax.random.normal(jax.random.split(key)[0], (5,))
    result_sparse = operator.mv(vector)
    result_dense = dense_matrix @ vector
    assert jnp.allclose(result_sparse, result_dense, rtol=1e-6)


gradient_solvers: dict[str, lx.AbstractLinearSolver] = {
    "cg": lx.CG(rtol=1e-13, atol=1e-13, max_steps=1000)
}
if CHOLMOD_AVAILABLE:
    gradient_solvers["cholmod"] = CholmodSolver()


@pytest.mark.parametrize("batched", [False, True], ids=["single", "batched"])
@pytest.mark.parametrize("solver", gradient_solvers.values(), ids=gradient_solvers)
def test_sparse_solve_gradients_match_dense_reference(solver, batched):
    """Gradients with respect to the stored matrix values, padding included,
    and to the right-hand sides, through single and vmapped solves."""
    n = 8
    rng = np.random.default_rng(0)
    factor = rng.standard_normal((n, n))
    dense = factor @ factor.T + n * np.eye(n)
    A = BCOO.fromdense(jnp.asarray(dense), nse=n * n + 3)
    rhs_shape = (n, 3) if batched else (n,)
    b = jnp.asarray(rng.standard_normal(rhs_shape))
    weights = jnp.asarray(rng.standard_normal(rhs_shape))
    sparse_solve = _batched_linear_solve if batched else _linear_solve

    def gradients(solve):
        def objective(data, b):
            x = solve(BCOO((data, A.indices), shape=A.shape), b)
            return jnp.sum(weights * x)

        return jax.grad(objective, argnums=(0, 1))(A.data, b)

    data_gradient, rhs_gradient = gradients(lambda A, b: sparse_solve(A, b, solver))

    expected = gradients(lambda A, b: jnp.linalg.solve(A.todense(), b))
    np.testing.assert_allclose(data_gradient, expected[0], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(rhs_gradient, expected[1], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(data_gradient[n * n :], 0.0)


def test_solver_state_carries_no_gradient():
    A = BCOO.fromdense(jnp.array([[2.0, -1.0], [-1.0, 2.0]]))

    def sum_of_state(data):
        matrix = BCOO((data, A.indices), shape=A.shape)
        state = _solver_state(matrix, lx.CG(rtol=1e-6, atol=1e-6), None, None)
        leaves = jax.tree.leaves(eqx.filter(state, eqx.is_inexact_array))
        return sum(jnp.sum(leaf) for leaf in leaves)

    assert sum_of_state(A.data) != 0
    np.testing.assert_array_equal(jax.grad(sum_of_state)(A.data), 0.0)


def solver_state(solver, A):
    if isinstance(solver, AMJaxCGSolver):
        return solver.init(BCOOLinearOperator(A), {})
    return None


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_solver(solver):
    A_scipy = poisson((10, 10), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)
    b = jnp.ones(A_jax.shape[0], dtype=A_jax.dtype)
    state = solver_state(solver, A_jax)
    x = _linear_solve(A_jax, b, solver, state=state)
    residual = A_jax @ x - b
    assert (
        jnp.linalg.norm(residual) < 1e-4
    ), f"Residual too large: {jnp.linalg.norm(residual)}"

    B = jnp.stack([b, 2 * b, 3 * b], axis=-1)
    X = _batched_linear_solve(A_jax, B, solver, state=state)
    X_reference = jnp.linalg.solve(A_jax.todense(), B)
    rtol, atol = (1e-5, 1e-6) if isinstance(solver, CholmodSolver) else (2e-3, 1e-4)
    assert jnp.allclose(X, X_reference, rtol=rtol, atol=atol)
    residuals = A_jax @ X - B
    assert (
        jnp.linalg.norm(residuals) < 3 * 1e-4
    ), f"Residual too large: {jnp.linalg.norm(residuals)}"


@pytest.mark.skipif(not AMJAX_AVAILABLE, reason="AMJax not available")
def test_amjax_state_uses_current_operator():
    A_scipy = poisson((5, 5), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)
    A_scaled = BCOO(
        (1.2 * A_jax.data, A_jax.indices),
        shape=A_jax.shape,
        indices_sorted=A_jax.indices_sorted,
        unique_indices=A_jax.unique_indices,
    )
    b = jnp.ones(A_jax.shape[0], dtype=A_jax.data.dtype)
    solver = AMJaxCGSolver(rtol=1e-5, atol=1e-5, max_steps=500)

    state = solver.init(BCOOLinearOperator(A_jax), {})

    x = _linear_solve(A_scaled, b, solver, state=state)
    residual = A_scaled @ x - b
    assert jnp.linalg.norm(residual) < 1e-4


@pytest.mark.skipif(not AMJAX_AVAILABLE, reason="AMJax not available")
def test_amjax_preconditioner_operator():
    A_scipy = poisson((5, 5), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)
    b = jnp.ones(A_jax.shape[0], dtype=A_jax.data.dtype)

    amjax_solver = build_amjax_solver(A_jax)
    preconditioner = amjax_preconditioner_operator(
        amjax_solver,
        jax.eval_shape(lambda: b),
    )

    assert preconditioner.in_structure().shape == b.shape
    assert preconditioner.out_structure().shape == b.shape
    assert jnp.all(jnp.isfinite(preconditioner.mv(b)))
