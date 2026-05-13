from typing import Any

import jax
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO


class BCOOLinearOperator(lx.MatrixLinearOperator):
    """`lineax.MatrixLinearOperator` wrapper for `jax.experimental.sparse.BCOO`
    matrices."""

    def __check_init__(self) -> None:
        if not isinstance(self.matrix, BCOO):
            raise ValueError("The operator must be a BCOO matrix.")

    def mv(self, vector: Array) -> Array:
        return self.matrix @ vector

    def transpose(self) -> "BCOOLinearOperator":
        return BCOOLinearOperator(self.matrix.T, self.tags)


@lx.is_positive_semidefinite.register(BCOOLinearOperator)
def _(op: BCOOLinearOperator) -> bool:
    return True


def linear_solve(
    A: BCOO,
    b: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
) -> Array:
    operator = BCOOLinearOperator(A)
    if state is None:
        return lx.linear_solve(operator, b, solver=solver, options=options).value
    return lx.linear_solve(
        operator,
        b,
        solver=solver,
        options=options,
        state=state,
    ).value


def batched_linear_solve(
    A: BCOO,
    B: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
) -> Array:
    return jax.vmap(
        lambda A, b: linear_solve(A, b, solver, options=options, state=state),
        in_axes=(None, 1),
        out_axes=1,
    )(A, B)
