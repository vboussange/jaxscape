import abc
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


class AbstractReusableStateSolver(lx.AbstractLinearSolver):
    """Lineax solver whose state can serve another operator of the same sparsity.

    Subclass it when part of a state outlives the matrix values, such as a sparse
    analysis or a multigrid hierarchy, and implement `reinit`. Distance queries
    call `reinit` on the state they receive before solving, so they solve the
    graph they are given with any state of the same sparsity pattern.
    """

    @abc.abstractmethod
    def reinit(
        self,
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
        state: Any,
    ) -> Any:
        """Return the state that solves `operator`, reusing what it can from `state`.

        It runs inside `jax.jit`. A query calls it on every state it receives, so
        it should do no expensive work when `state` already solves `operator`.

        **Arguments:**

        - `operator`: The operator to solve. It has the sparsity pattern of the
          operator `state` was built for, possibly different values, and no
          gradient.
        - `options`: Solver options, as passed to `init`.
        - `state`: A state returned by `init` or `reinit`.

        **Returns:**

        The state to solve `operator` with, with the structure of a state from
        `init`, so that a compiled loop can carry it.
        """


def _linear_solve(
    A: BCOO,
    b: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
) -> Array:
    options = _solver_options(options)
    solve_state = _solver_state(A, solver, options, state)
    return lx.linear_solve(
        BCOOLinearOperator(A), b, solver=solver, options=options, state=solve_state
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
    """Return the state that solves `A`, built from its values without gradient.

    Without a supplied `state`, it is the state `solver.init` builds. An
    `AbstractReusableStateSolver` reinitializes the supplied state for `A`. Any
    other solver solves with the supplied state, which must belong to `A`.
    """
    options = _solver_options(options)
    operator = _stop_gradient(BCOOLinearOperator(A))
    if state is None:
        return solver.init(operator, options)
    if isinstance(solver, AbstractReusableStateSolver):
        return solver.reinit(operator, options, state)
    return state


def _solver_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    return options


def _stop_gradient(tree: Any) -> Any:
    """Stop gradients through the arrays of `tree`, which may hold other leaves."""
    dynamic, static = eqx.partition(tree, eqx.is_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)
