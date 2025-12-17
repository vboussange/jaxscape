import jax
import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO
from jaxscape.solvers import (
    batched_linear_solve,
    BCOOLinearOperator,
    CholmodSolver,
    linear_solve,
    PyAMGSolver,
)


# Check availability of optional solvers
try:
    import pyamg
    from pyamg.gallery import poisson

    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False

try:
    import cholespy

    CHOLMOD_AVAILABLE = True
except ImportError:
    CHOLMOD_AVAILABLE = False

# Build list of available solvers
available_solvers = []
if PYAMG_AVAILABLE:
    available_solvers.append(PyAMGSolver())
if CHOLMOD_AVAILABLE:
    available_solvers.append(CholmodSolver())


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


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_solver(solver):
    if not PYAMG_AVAILABLE:
        pytest.skip("pyamg not available for generating test matrix")
    A_scipy = poisson((10, 10), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)
    b = jnp.ones(A_jax.shape[0])
    x = linear_solve(A_jax, b, solver)
    residual = A_jax @ x - b
    assert (
        jnp.linalg.norm(residual) < 1e-4
    ), f"Residual too large: {jnp.linalg.norm(residual)}"

    B = jnp.stack([b, 2 * b, 3 * b], axis=-1)
    X = batched_linear_solve(A_jax, B, solver)
    residuals = A_jax @ X - B
    assert (
        jnp.linalg.norm(residuals) < 3 * 1e-4
    ), f"Residual too large: {jnp.linalg.norm(residuals)}"


@pytest.mark.skipif(len(available_solvers) == 0, reason="No solvers available")
@pytest.mark.parametrize("solver", available_solvers)
def test_solver_differentiability(solver):
    """Test that the solver is differentiable."""
    if not PYAMG_AVAILABLE:
        pytest.skip("pyamg not available for generating test matrix")
    A_scipy = poisson((5, 5), format="coo", dtype="float32")
    A_jax = BCOO.from_scipy_sparse(A_scipy)

    def objective(A_data):
        # Modify the matrix data slightly
        A_modified = BCOO((A_data, A_jax.indices), shape=A_jax.shape)
        b = jnp.ones(A_modified.shape[0])
        x = linear_solve(A_modified, b, solver)
        return jnp.sum(x**2)

    grad_objective = jax.jit(jax.grad(objective))
    grad_result = grad_objective(A_jax.data)
    assert isinstance(grad_result, jax.Array)
    assert grad_result.shape == A_jax.data.shape
