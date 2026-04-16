import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO

from .cholmodsolver import CholmodSolver


class BCOOLinearOperator(lx.MatrixLinearOperator):
    """`lineax.MatrixLinearOperator` wrapper for `jax.experimental.sparse.BCOO`
    matrices."""

    def __check_init__(self) -> None:
        if not isinstance(self.matrix, BCOO):
            raise ValueError("The operator must be a BCOO matrix.")

    def mv(self, vector: Array) -> Array:
        return self.matrix @ vector


@lx.is_positive_semidefinite.register(BCOOLinearOperator)
def _(op: BCOOLinearOperator) -> bool:
    return True


def linear_solve(A: BCOO, b: Array, solver: lx.AbstractLinearSolver) -> Array:
    operator = BCOOLinearOperator(A)
    return lx.linear_solve(operator, b, solver=solver).value  # works


def _cholmod_matrix_rhs_solve_impl(A: BCOO, B: Array, solver: CholmodSolver) -> Array:
    result_shape = jax.ShapeDtypeStruct(B.shape, B.dtype)
    return eqx.filter_pure_callback(
        solver._compute_host,
        A,
        B,
        result_shape_dtypes=result_shape,
        vmap_method="expand_dims",
    )


def _cholmod_matrix_rhs_solve(
    A: BCOO,
    B: Array,
    solver: CholmodSolver,
) -> Array:
    @jax.custom_vjp
    def solve(current_A: BCOO, current_B: Array) -> Array:
        return _cholmod_matrix_rhs_solve_impl(current_A, current_B, solver)

    def solve_fwd(current_A: BCOO, current_B: Array):
        X = _cholmod_matrix_rhs_solve_impl(current_A, current_B, solver)
        return X, (current_A, current_B, X)

    def solve_bwd(residuals, G: Array):
        current_A, current_B, X = residuals
        adjoint = _cholmod_matrix_rhs_solve_impl(current_A, G, solver)

        dense_grad = -(adjoint @ X.T)
        row_indices = current_A.indices[:, 0]
        col_indices = current_A.indices[:, 1]
        data_grad = dense_grad[row_indices, col_indices].astype(current_A.data.dtype)
        a_grad = BCOO((data_grad, current_A.indices), shape=current_A.shape)
        b_grad = adjoint.astype(current_B.dtype)
        return a_grad, b_grad

    solve.defvjp(solve_fwd, solve_bwd)
    return solve(A, B)


def batched_linear_solve(A: BCOO, B: Array, solver: lx.AbstractLinearSolver) -> Array:
    if B.ndim == 1:
        return linear_solve(A, B, solver)

    if isinstance(solver, CholmodSolver):
        return _cholmod_matrix_rhs_solve(A, B, solver)

    return jax.lax.map(lambda b: linear_solve(A, b, solver), B.T).T
