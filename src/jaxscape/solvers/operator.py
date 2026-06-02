from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
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


def linear_solve(
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


def batched_linear_solve(
    A: BCOO,
    B: Array,
    solver: lx.AbstractLinearSolver,
    *,
    options: dict[str, Any] | None = None,
    state: Any = None,
) -> Array:
    return _batched_linear_solve(
        (A.data, B),
        A.indices,
        A.shape,
        A.indices_sorted,
        A.unique_indices,
        solver,
        options,
        state,
    )


# This custom VJP is deliberately scoped to the batched helper. The tempting
# alternative is to expose `B` as one matrix-shaped Lineax RHS and rely on
# Lineax's own differentiation rule, but iterative solvers such as CG then use
# one global set of scalar reductions across all RHS columns. That is a block
# solve with shared step sizes, not the independent column solves represented by
# `vmap(linear_solve)`. Keeping the forward pass vmapped preserves solver
# semantics, while the backward rule below applies the standard full-rank linear
# solve adjoint without passing through JAX's broken transpose rule for vmapped
# `BCOO @ vector` products.
# TODO: this could be not needed anymore thanks to fix https://github.com/jax-ml/jax/issues/37647#event-25863292581
@eqx.filter_custom_vjp
def _batched_linear_solve(
    solve_args: tuple[Array, Array],
    indices: Array,
    shape: tuple[int, int],
    indices_sorted: bool,
    unique_indices: bool,
    solver: lx.AbstractLinearSolver,
    options: dict[str, Any] | None,
    state: Any,
) -> Array:
    A_data, B = solve_args
    A = BCOO(
        (A_data, indices),
        shape=shape,
        indices_sorted=indices_sorted,
        unique_indices=unique_indices,
    )
    solve_state = _solver_state(A, solver, options, state)
    return _batched_linear_solve_impl(A, B, solver, options=options, state=solve_state)


@_batched_linear_solve.def_fwd
def _batched_linear_solve_fwd(
    perturbed,
    solve_args: tuple[Array, Array],
    indices: Array,
    shape: tuple[int, int],
    indices_sorted: bool,
    unique_indices: bool,
    solver: lx.AbstractLinearSolver,
    options: dict[str, Any] | None,
    state: Any,
) -> tuple[Array, tuple[BCOO, Array, Any]]:
    del perturbed
    A_data, B = solve_args
    A = BCOO(
        (A_data, indices),
        shape=shape,
        indices_sorted=indices_sorted,
        unique_indices=unique_indices,
    )
    solve_state = _solver_state(A, solver, options, state)
    X = _batched_linear_solve_impl(A, B, solver, options=options, state=solve_state)
    return X, (A, X, solve_state)


@_batched_linear_solve.def_bwd
def _batched_linear_solve_bwd(
    residuals: tuple[BCOO, Array, Any],
    cotangent: Array | None,
    perturbed,
    solve_args: tuple[Array, Array],
    indices: Array,
    shape: tuple[int, int],
    indices_sorted: bool,
    unique_indices: bool,
    solver: lx.AbstractLinearSolver,
    options: dict[str, Any] | None,
    state: Any,
) -> tuple[Array | None, Array | None]:
    del solve_args, indices, shape, indices_sorted, unique_indices, state
    A, X, solve_state = residuals
    A_data_perturbed, B_perturbed = perturbed

    if cotangent is None or not (A_data_perturbed or B_perturbed):
        return None, None

    options = _solver_options(options)
    transpose_state, transpose_options = solver.transpose(solve_state, options)
    A_transpose = A.T
    assert isinstance(A_transpose, BCOO)
    adjoint = _batched_linear_solve_impl(
        A_transpose,
        cotangent,
        solver,
        options=transpose_options,
        state=transpose_state,
    )

    grad_A_data = None
    if A_data_perturbed:
        # For A X = B and loss cotangent G, solve A.T Lambda = G. Then
        # dL/dA = -Lambda X.T. Gather only the stored sparse entries so the
        # cotangent has exactly the primal BCOO data shape; this avoids the JAX
        # sparse transpose bug that returns a broadcasted vmapped cotangent.
        rows = A.indices[:, 0]
        cols = A.indices[:, 1]
        grad_A_data = -jnp.sum(adjoint[rows] * X[cols], axis=1)

    grad_B = adjoint if B_perturbed else None
    return grad_A_data, grad_B


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
