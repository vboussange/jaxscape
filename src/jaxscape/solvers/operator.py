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


class AbstractRefreshingSolver(lx.AbstractLinearSolver):
    """Lineax solver whose state is refreshed for each operator it solves.

    Subclass it for a solver whose state depends on the matrix values, such
    as a sparse factorization, and implement `refresh_state`. Before solving,
    every resistance-distance query refreshes the state it receives for the
    current grounded Laplacian, solves with the refreshed state, and returns
    that state from its `*_distance_and_state` form.
    """

    @abc.abstractmethod
    def refresh_state(
        self,
        operator: lx.AbstractLinearOperator,
        options: dict[str, Any],
        state: Any,
    ) -> Any:
        """Refresh `state` for the values of `operator`.

        **Arguments:**

        - `operator`: The operator to solve. It has the sparsity pattern of the
          operator `state` was built for, possibly different values, and no
          gradient.
        - `options`: Solver options, as passed to `init`.
        - `state`: A state returned by `init` or `refresh_state`.

        **Returns:**

        The state to solve `operator` with, and to pass to the next call.
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
    solve_state, _ = _solver_states(A, solver, options, state)
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
    solve_state, _ = _solver_states(A, solver, options, state)
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


def _solver_states(
    A: BCOO,
    solver: lx.AbstractLinearSolver,
    options: dict[str, Any] | None,
    state: Any,
) -> tuple[Any, Any]:
    """Return the state that solves `A` and the state to carry to the next call.

    Both are built from the values of `A` without gradient. Without a supplied
    `state`, both are the state `solver.init` builds for `A`. An
    `AbstractRefreshingSolver` refreshes the supplied state, and both are the
    refreshed state. A solver with `materialize_state`, such as
    `AMJaxCGSolver`, completes the supplied state for `A` to solve with, and
    carries the supplied state forward. Any other solver solves with the
    supplied state and carries it forward.
    """
    options = _solver_options(options)
    operator = _stop_gradient_operator(BCOOLinearOperator(A))
    if state is None:
        state = solver.init(operator, options)
        return state, state
    if isinstance(solver, AbstractRefreshingSolver):
        state = solver.refresh_state(operator, options, state)
        return state, state
    materialize_state = getattr(solver, "materialize_state", None)
    if materialize_state is None:
        return state, state
    return materialize_state(operator, options, state), state


def _solver_options(options: dict[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    return options


def _stop_gradient_operator(
    operator: lx.AbstractLinearOperator,
) -> lx.AbstractLinearOperator:
    dynamic_operator, static_operator = eqx.partition(operator, eqx.is_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic_operator), static_operator)
