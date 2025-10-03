import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from pyamg.gallery import poisson
from jax.experimental.sparse import BCOO
from jaxscape.linear_solve import BCOOLinearOperator, PyAMGSolver, linear_solve

def test_bcoo_linear_operator_mv():
    """Test BCOOLinearOperator.mv method by comparing to dense matrix multiplication."""
    # Create a small dense matrix
    key = jax.random.PRNGKey(42)
    dense_matrix = jax.random.normal(key, (5, 5))

    # Convert to BCOO sparse format
    bcoo_matrix = BCOO.fromdense(dense_matrix)

    # Create the linear operator
    operator = BCOOLinearOperator(bcoo_matrix)

    # Create a test vector
    vector = jax.random.normal(jax.random.split(key)[0], (5,))

    # Test mv method
    result_sparse = operator.mv(vector)

    # Compare to dense multiplication
    result_dense = dense_matrix @ vector

    # They should be approximately equal
    assert jnp.allclose(result_sparse, result_dense, rtol=1e-6)


@pytest.mark.parametrize(
    "solver",
    (
        PyAMGSolver(tol=1e-9),
        # TODO: add PyAMGXSolver, do later
    ),
)

solver = PyAMGSolver(tol=1e-10, accel=None) # For quick testing without pytest
def test_custom_solver(solver):
    """Test custom solvers using pyamg.gallery.poisson example."""
    # Create a 2D Poisson problem as suggested in the TODO
    A_scipy = poisson((10, 10), format="csr", dtype="float32")

    # Convert to JAX BCOO format
    A_jax = BCOO.from_scipy_sparse(A_scipy)

    # Create a right-hand side vector (e.g., all ones)
    b = jnp.ones(A_jax.shape[0])

    x = linear_solve(A_jax, b, solver)

    # Verify the solution by checking Ax ≈ b
    residual = A_jax @ x - b
    assert jnp.linalg.norm(residual) < 1e-4, f"Residual too large: {jnp.linalg.norm(residual)}"