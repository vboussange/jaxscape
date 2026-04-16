import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from typing import cast
from jax import Array
from jax.experimental import sparse as jsparse
from jax.experimental.sparse import BCOO


class BCOOLinearOperator(lx.MatrixLinearOperator):
    """`lineax.MatrixLinearOperator` wrapper for `jax.experimental.sparse.BCOO`
    matrices."""

    def __init__(self, matrix: BCOO):
        super().__init__(cast(Array, matrix))

    def __check_init__(self) -> None:
        if not isinstance(self.matrix, BCOO):
            raise ValueError("The operator must be a BCOO matrix.")

    def mv(self, vector: Array) -> Array:
        matrix = cast(BCOO, self.matrix)
        return matrix @ vector


_batched_bcoo_matvec = jsparse.sparsify(
    jax.vmap(lambda matrix, column: matrix @ column, in_axes=(None, 1), out_axes=1)
)


def _apply_batched_bcoo_matvec(matrix: BCOO, vector: Array) -> Array:
    if vector.ndim == 1:
        return matrix @ vector

    flat_vector = vector.reshape((vector.shape[0], -1))
    flat_result = _batched_bcoo_matvec(matrix, flat_vector)
    return flat_result.reshape((matrix.shape[0],) + vector.shape[1:])


class _BatchedBCOOLinearOperator(BCOOLinearOperator):
    rhs_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, matrix: BCOO, rhs_shape: tuple[int, ...]):
        super().__init__(matrix)
        object.__setattr__(self, "rhs_shape", rhs_shape)

    def __check_init__(self) -> None:
        super().__check_init__()
        matrix = cast(BCOO, self.matrix)
        if len(self.rhs_shape) < 2:
            raise ValueError("rhs_shape must describe a batched right-hand side.")
        if self.rhs_shape[0] != matrix.shape[1]:
            raise ValueError("rhs_shape[0] must match the operator input dimension.")

    def mv(self, vector: Array) -> Array:
        matrix = cast(BCOO, self.matrix)
        return _apply_batched_bcoo_matvec(matrix, vector)

    def transpose(self) -> "_BatchedBCOOLinearOperator":
        matrix = cast(BCOO, self.matrix)
        return _BatchedBCOOLinearOperator(
            cast(BCOO, matrix.T),
            (matrix.shape[0],) + self.rhs_shape[1:],
        )

    def in_structure(self):
        matrix = cast(BCOO, self.matrix)
        return jax.ShapeDtypeStruct(shape=self.rhs_shape, dtype=matrix.dtype)

    def out_structure(self):
        matrix = cast(BCOO, self.matrix)
        return jax.ShapeDtypeStruct(
            shape=(matrix.shape[0],) + self.rhs_shape[1:],
            dtype=matrix.dtype,
        )


@lx.is_positive_semidefinite.register(BCOOLinearOperator)
def _(op: BCOOLinearOperator) -> bool:
    return True


@lx.conj.register(_BatchedBCOOLinearOperator)
def _(op: _BatchedBCOOLinearOperator) -> _BatchedBCOOLinearOperator:
    current_matrix = cast(BCOO, op.matrix)
    matrix = BCOO(
        (jnp.conj(current_matrix.data), current_matrix.indices),
        shape=current_matrix.shape,
        indices_sorted=current_matrix.indices_sorted,
        unique_indices=current_matrix.unique_indices,
    )
    return _BatchedBCOOLinearOperator(matrix, op.rhs_shape)


def _make_linear_operator(A: BCOO, rhs_shape: tuple[int, ...]) -> BCOOLinearOperator:
    if len(rhs_shape) == 1:
        return BCOOLinearOperator(A)
    return _BatchedBCOOLinearOperator(A, rhs_shape)


def linear_solve(A: BCOO, b: Array, solver: lx.AbstractLinearSolver) -> Array:
    operator = _make_linear_operator(A, b.shape)
    return lx.linear_solve(operator, b, solver=solver).value  # works


def batched_linear_solve(A: BCOO, B: Array, solver: lx.AbstractLinearSolver) -> Array:
    return linear_solve(A, B, solver)
