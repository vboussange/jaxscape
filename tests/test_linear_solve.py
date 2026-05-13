from importlib.util import find_spec

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO
from jaxscape.solvers import (
    amjax_preconditioner_operator,
    AMJaxCGSolver,
    batched_linear_solve,
    BCOOLinearOperator,
    build_amjax_solver,
    CholmodSolver,
    linear_solve,
    PyAMGSolver,
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


def solver_state(solver, A):
    if isinstance(solver, AMJaxCGSolver):
        return solver.init(BCOOLinearOperator(A), {})
    return None


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_solver(solver):
    A_scipy = poisson((10, 10), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)
    b = jnp.ones(A_jax.shape[0])
    state = solver_state(solver, A_jax)
    x = linear_solve(A_jax, b, solver, state=state)
    residual = A_jax @ x - b
    assert (
        jnp.linalg.norm(residual) < 1e-4
    ), f"Residual too large: {jnp.linalg.norm(residual)}"

    B = jnp.stack([b, 2 * b, 3 * b], axis=-1)
    X = batched_linear_solve(A_jax, B, solver, state=state)
    residuals = A_jax @ X - B
    assert (
        jnp.linalg.norm(residuals) < 3 * 1e-4
    ), f"Residual too large: {jnp.linalg.norm(residuals)}"


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_solver_differentiability(solver):
    """Test that the solver is differentiable."""
    A_scipy = poisson((5, 5), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)
    b = jnp.ones(A_jax.shape[0], dtype=A_jax.data.dtype)
    state = solver_state(solver, A_jax)

    def objective(A_data):
        # Modify the matrix data slightly
        A_modified = BCOO((A_data, A_jax.indices), shape=A_jax.shape)
        x = linear_solve(A_modified, b, solver, state=state)
        return jnp.sum(x**2)

    grad_objective = jax.jit(jax.grad(objective))
    grad_result = grad_objective(A_jax.data)
    assert isinstance(grad_result, jax.Array)
    assert grad_result.shape == A_jax.data.shape
    assert jnp.all(jnp.isfinite(grad_result))


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
