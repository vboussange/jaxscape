from importlib.util import find_spec

import jax
import jax.numpy as jnp
import jaxscape.solvers.cudsssolver as cudsssolver_module
import pytest
from jax.experimental.sparse import BCOO, BCSR
from jaxscape.solvers import (
    batched_linear_solve,
    BCOOLinearOperator,
    CholmodSolver,
    CuDSSSolver,
    linear_solve,
    PyAMGSolver,
)


# Check availability of optional solvers
PYAMG_AVAILABLE = find_spec("pyamg") is not None
if PYAMG_AVAILABLE:
    from pyamg.gallery import poisson
else:
    poisson = None

CHOLMOD_AVAILABLE = find_spec("cholespy") is not None

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


def test_cudss_solver_requires_spineax(monkeypatch):
    monkeypatch.setattr(cudsssolver_module, "CUDSS_AVAILABLE", False)

    with pytest.raises(ImportError, match="spineax with cuDSS support"):
        CuDSSSolver()


def test_cudss_solver_uses_lineax_custom_vjp(monkeypatch):
    class FakeCuDSSBackend:
        def __call__(
            self,
            rhs,
            csr_values,
            csr_offsets,
            csr_columns,
            *,
            device_id,
            mtype_id,
            mview_id,
        ):
            del device_id, mtype_id, mview_id

            n = csr_offsets.shape[0] - 1
            dense = BCSR(
                (csr_values, csr_columns, csr_offsets), shape=(n, n)
            ).todense()
            solution = jnp.linalg.solve(dense, rhs)
            inertia = jnp.array([n, 0], dtype=jnp.int32)
            return solution, inertia

    monkeypatch.setattr(cudsssolver_module, "CUDSS_AVAILABLE", True)
    monkeypatch.setattr(cudsssolver_module, "cudss_solve", FakeCuDSSBackend())

    A = jnp.array(
        [
            [4.0, 1.0, 0.0],
            [1.0, 3.0, 1.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=jnp.float32,
    )
    A_bcoo = BCOO.fromdense(A)
    solver = CuDSSSolver()
    b = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    x = linear_solve(A_bcoo, b, solver)
    assert jnp.allclose(A @ x, b, rtol=1e-5)

    B = jnp.stack([b, 2 * b], axis=1)
    X = batched_linear_solve(A_bcoo, B, solver)
    assert jnp.allclose(A @ X, B, rtol=1e-5)

    def objective(data):
        A_dynamic = BCOO((data, A_bcoo.indices), shape=A_bcoo.shape)
        return jnp.sum(linear_solve(A_dynamic, b, solver) ** 2)

    grad = jax.grad(objective)(A_bcoo.data)
    assert grad.shape == A_bcoo.data.shape
