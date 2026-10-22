from typing import Any

import equinox as eqx
import jax
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO


class BCOOLinearOperator(lx.MatrixLinearOperator):
    """`lineax.MatrixLinearOperator` wrapper for `jax.experimental.sparse.BCOO`
    matrices."""

    def __init__(self, matrix: BCOO, tags: object | frozenset[object] = ()):
        super().__init__(matrix, tags)  # type: ignore[arg-type]

    def __check_init__(self) -> None:
        if not isinstance(self.matrix, BCOO):
            raise ValueError("The operator must be a BCOO matrix.")

    def mv(self, vector: Array) -> Array:
        return self.matrix @ vector

    def transpose(self) -> "BCOOLinearOperator":
        matrix_transpose = self.matrix.T
        assert isinstance(matrix_transpose, BCOO)
        return BCOOLinearOperator(matrix_transpose, self.tags)


@lx.is_positive_semidefinite.register(BCOOLinearOperator)
def _(op: BCOOLinearOperator) -> bool:
    return True


def _linear_solve(
    A: BCOO,
    b: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
) -> Array:
    operator = BCOOLinearOperator(A)
    options = _solver_options(options)
    solve_state = _solver_state_from_operator(operator, solver, options, state)
    if solve_state is None:
        return lx.linear_solve(operator, b, solver=solver, options=options).value
    return lx.linear_solve(
        operator,
        b,
        solver=solver,
        options=options,
        state=solve_state,
    ).value


def _batched_linear_solve(
    A: BCOO,
    B: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
) -> Array:
    """Prepare the current operator once, then solve independent RHS columns."""
    solve_state = _solver_state(A, solver, options, state)
    return _batched_linear_solve_impl(A, B, solver, options=options, state=solve_state)


def _batched_linear_solve_impl(
    A: BCOO,
    B: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any,
) -> Array:
    operator = BCOOLinearOperator(A)
    # Keep this as a vmapped Lineax solve, even for callback-backed solvers.
    # Their `filter_pure_callback(..., vmap_method="expand_dims")` compute paths
    # already coalesce the RHS batch into one host callback; direct callback
    # fast paths measured slower because they bypass Lineax's primitive batching.
    return jax.vmap(
        lambda b: lx.linear_solve(
            operator,
            b,
            solver=solver,
            options=options,
            state=state,
        ).value,
        in_axes=1,
        out_axes=1,
    )(B)


def _solver_state(
    A: BCOO,
    solver: lx.AbstractLinearSolver,
    options: dict[str, Any] | None,
    state: Any,
) -> Any:
    options = _solver_options(options)
    operator = BCOOLinearOperator(A)
    solve_state = _solver_state_from_operator(operator, solver, options, state)
    if solve_state is not None:
        return solve_state
    return solver.init(_stop_gradient_operator(operator), options)


def _solver_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    return options


def _solver_state_from_operator(
    operator: lx.AbstractLinearOperator,
    solver: lx.AbstractLinearSolver,
    options: dict[str, Any],
    state: Any,
) -> Any:
    if state is None:
        return None
    materialize_state = getattr(solver, "materialize_state", None)
    if materialize_state is None:
        return state
    return materialize_state(_stop_gradient_operator(operator), options, state)


def _stop_gradient_operator(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    dynamic_operator, static_operator = eqx.partition(operator, eqx.is_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic_operator), static_operator)
