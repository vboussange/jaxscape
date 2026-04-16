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


def _build_spd_system(size: int) -> BCOO:
    diagonal = 4.0 * jnp.ones(size, dtype=jnp.float32)
    off_diagonal = -1.0 * jnp.ones(size - 1, dtype=jnp.float32)
    dense = jnp.diag(diagonal)
    dense = dense + jnp.diag(off_diagonal, k=1)
    dense = dense + jnp.diag(off_diagonal, k=-1)
    return BCOO.fromdense(dense)


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
    A_jax = _build_spd_system(10)
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
    A_jax = _build_spd_system(5)

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


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="CholmodSolver unavailable")
def test_cholmod_batched_solver_matches_columnwise_solves():
    A_jax = _build_spd_system(5)
    b = jnp.ones(A_jax.shape[0], dtype=jnp.float32)
    B = jnp.stack([b, 2 * b, 3 * b], axis=-1)
    solver = CholmodSolver()

    X_batched = batched_linear_solve(A_jax, B, solver)
    X_columnwise = jnp.stack(
        [linear_solve(A_jax, B[:, i], solver) for i in range(B.shape[1])],
        axis=-1,
    )

    assert X_batched.shape == B.shape
    assert jnp.allclose(X_batched, X_columnwise, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="CholmodSolver unavailable")
def test_cholmod_batched_solver_is_differentiable():
    A_jax = _build_spd_system(4)
    B = jnp.stack(
        [
            jnp.ones(A_jax.shape[0], dtype=jnp.float32),
            2 * jnp.ones(A_jax.shape[0], dtype=jnp.float32),
        ],
        axis=-1,
    )
    solver = CholmodSolver()

    def objective(A_data):
        A_modified = BCOO((A_data, A_jax.indices), shape=A_jax.shape)
        X = batched_linear_solve(A_modified, B, solver)
        return jnp.sum(X**2)

    grad_result = jax.grad(objective)(A_jax.data)

    assert isinstance(grad_result, jax.Array)
    assert grad_result.shape == A_jax.data.shape
    assert jnp.all(jnp.isfinite(grad_result))


@pytest.mark.skipif(not CHOLMOD_AVAILABLE, reason="CholmodSolver unavailable")
def test_cholmod_batched_solver_matches_dense_solve_and_gradients():
    A_jax = _build_spd_system(4)
    A_dense = A_jax.todense()
    B = jnp.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    solver = CholmodSolver()

    X_batched = batched_linear_solve(A_jax, B, solver)
    X_dense = jnp.linalg.solve(A_dense, B)

    assert jnp.allclose(X_batched, X_dense, rtol=1e-5, atol=1e-6)

    def objective_sparse(A_data, rhs):
        A_sparse = BCOO((A_data, A_jax.indices), shape=A_jax.shape)
        return jnp.sum(batched_linear_solve(A_sparse, rhs, solver) ** 2)

    def objective_dense(matrix, rhs):
        return jnp.sum(jnp.linalg.solve(matrix, rhs) ** 2)

    sparse_grad_A, sparse_grad_B = jax.grad(objective_sparse, argnums=(0, 1))(
        A_jax.data,
        B,
    )
    dense_grad_A, dense_grad_B = jax.grad(objective_dense, argnums=(0, 1))(
        A_dense,
        B,
    )
    dense_grad_entries = dense_grad_A[A_jax.indices[:, 0], A_jax.indices[:, 1]]

    assert jnp.allclose(sparse_grad_A, dense_grad_entries, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(sparse_grad_B, dense_grad_B, rtol=1e-5, atol=1e-6)
