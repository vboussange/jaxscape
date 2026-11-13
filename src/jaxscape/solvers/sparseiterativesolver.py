"""Sparse iterative solvers: conjugate gradients with a preconditioner."""

import abc
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array
from lineax import RESULTS

from .sparsesolver import (
    _product,
    AbstractSparseSolver,
    AbstractSparseState,
    CSRPattern,
)


@final
class SparseIterativeState[Preconditioner](AbstractSparseState):
    """State of an `AbstractSparseIterativeSolver` for one sparsity pattern.

    **Attributes:**

    - `preconditioner`: What the solver keeps across values, from which each
      solve applies an approximate inverse of the matrix.
    - `pattern`: The CSR pattern of the matrix.
    - `values`: The CSR values of the matrix.
    """

    preconditioner: Preconditioner
    pattern: CSRPattern
    values: Array


class AbstractSparseIterativeSolver[Preconditioner](
    AbstractSparseSolver[SparseIterativeState[Preconditioner]]
):
    """Conjugate gradients on sparse symmetric positive definite matrices, with a
    preconditioner.

    Subclass it with `preconditioner`, which `init` builds once, `precondition`,
    which applies it to the current values at each solve, `matvec`, the product
    of the current matrix with vectors, and `vector_shape`, the shape of those
    vectors. `csr_matvec` gives the product of the CSR values, with vectors of
    shape `(n,)`. The base class provides the Lineax solver, as
    `AbstractSparseSolver` describes: `reinit` keeps the preconditioner and
    replaces the values. Values other than those the preconditioner was built
    from change the number of iterations, and a solve that then exceeds
    `max_steps` fails.

    A batch of right-hand sides, as `jax.vmap` makes, runs as one solve with the
    right-hand sides contiguous in memory, until all have converged. A solve
    that has not converged after `max_steps` iterations fails with
    `lineax.RESULTS.max_steps_reached`, and one whose solution is not finite,
    as those of singular matrices can be, with `lineax.RESULTS.singular`.

    **Attributes:**

    - `rtol`: Largest Euclidean norm, relative to that of the right-hand side,
      of the residual that conjugate gradients update at each iteration. In
      exact arithmetic it equals the residual `b - A x` of the solution `x`.
      In float32, rounding stops the decrease of `b - A x`, so that solves
      can converge with a residual `b - A x` above `rtol`.
    - `max_steps`: Largest number of iterations.
    """

    rtol: eqx.AbstractVar[float]
    max_steps: eqx.AbstractVar[int]

    def __check_init__(self):
        if self.rtol < 0:
            raise ValueError("`rtol` must be non-negative.")
        if self.max_steps < 1:
            raise ValueError("`max_steps` must be positive.")

    @abc.abstractmethod
    def preconditioner(self, pattern: CSRPattern, values: Array) -> Preconditioner:
        """Build what the state keeps across values. `init` calls it.

        It may read concrete arrays or run host code, as `init` runs outside
        `jax.jit`, apart from the states that `distance.init(graph)` builds from
        the shape and contiguity of a `GridGraph`, with concrete values. Those
        build eagerly, under `jax.ensure_compile_time_eval`, where
        `eqx.error_if` raises whatever its condition: it must not call
        `eqx.error_if`.

        **Arguments:**

        - `pattern`: The CSR pattern of the matrix.
        - `values`: The CSR values of the matrix.

        **Returns:**

        A pytree, which states carry through `jax.jit`.
        """

    @abc.abstractmethod
    def precondition(
        self, preconditioner: Preconditioner, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        """Return the preconditioner of the current values. Each solve calls it
        once, inside `jax.jit`.

        **Arguments:**

        - `preconditioner`: What `preconditioner` built.
        - `pattern`: The CSR pattern of the matrix.
        - `values`: The current CSR values of the matrix.

        **Returns:**

        A function that applies a symmetric positive definite approximation of
        the inverse of the matrix to residuals of `m` right-hand sides, with
        shape `vector_shape(...) + (m,)`.
        """

    @abc.abstractmethod
    def matvec(
        self, preconditioner: Preconditioner, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        """Return the product of the matrix with vectors. Each solve calls it
        once, inside `jax.jit`. `csr_matvec(pattern, values)` gives the product
        of the CSR values.

        **Arguments:**

        - `preconditioner`: What `preconditioner` built.
        - `pattern`: The CSR pattern of the matrix.
        - `values`: The current CSR values of the matrix.

        **Returns:**

        A function that multiplies the matrix with `m` vectors, with shape
        `vector_shape(...) + (m,)`.
        """

    @abc.abstractmethod
    def vector_shape(
        self, preconditioner: Preconditioner, pattern: CSRPattern
    ) -> tuple[int, ...]:
        """Return the shape in which `matvec` and `precondition` take vectors:
        `(pattern.n,)`, or another shape of `n` entries.

        Conjugate gradients carry their vectors in this shape, with a last axis
        of right-hand sides: a solver of rasters takes them as rasters, which
        XLA would copy at each iteration if they were reshaped there.

        **Arguments:**

        - `preconditioner`: What `preconditioner` built.
        - `pattern`: The CSR pattern of the matrix.

        **Returns:**

        A shape of `n` entries.
        """

    def _init(
        self, pattern: CSRPattern, values: Array
    ) -> SparseIterativeState[Preconditioner]:
        preconditioner = self.preconditioner(pattern, values)
        return SparseIterativeState(
            preconditioner=preconditioner, pattern=pattern, values=values
        )

    def _reinit(
        self, state: SparseIterativeState[Preconditioner], values: Array
    ) -> SparseIterativeState[Preconditioner]:
        return SparseIterativeState(
            preconditioner=state.preconditioner, pattern=state.pattern, values=values
        )

    def _solve(
        self, state: SparseIterativeState[Preconditioner], vector: Array
    ) -> tuple[Array, RESULTS]:
        solution, solved = _solve(self, state, vector)
        # Singular matrices, such as the grounded Laplacians of disconnected
        # graphs, give non-finite solutions, or solves that do not converge.
        failed = RESULTS.where(
            jnp.all(jnp.isfinite(solution)),
            RESULTS.max_steps_reached,
            RESULTS.singular,
        )
        return solution, RESULTS.where(solved, RESULTS.successful, failed)


def csr_matvec(pattern: CSRPattern, values: Array) -> Callable[[Array], Array]:
    """Return the product of the matrix of CSR `pattern` and `values` with
    vectors, as `AbstractSparseIterativeSolver.matvec` returns it for a
    `vector_shape` of `(pattern.n,)`.

    **Arguments:**

    - `pattern`: The CSR pattern of the matrix.
    - `values`: The CSR values of the matrix.

    **Returns:**

    A function that multiplies the matrix with `m` vectors, with shape `(n, m)`.
    """
    return jax.vmap(
        lambda vector: _product(pattern, values, vector), in_axes=1, out_axes=1
    )


def _ratio(numerator: Array, denominator: Array) -> Array:
    nonzero = denominator != 0
    return jnp.where(nonzero, numerator / jnp.where(nonzero, denominator, 1), 0)


def _dot(x: Array, y: Array) -> Array:
    return jnp.sum(x * y, axis=tuple(range(x.ndim - 1)))


# Each solve traces it, and Lineax traces a solve up to three times: a jitted
# function is traced once for its shapes.
@eqx.filter_jit
def _cg(
    solver: AbstractSparseIterativeSolver, state: SparseIterativeState, b: Array
) -> tuple[Array, Array]:
    """Solve for the columns of `b`, of shape `(n, m)`, in the vector shape of
    `solver`. All columns iterate until the last one converges, which costs no
    time.

    **Returns:**

    The solutions, and whether each column converged.
    """
    args = (state.preconditioner, state.pattern, state.values)
    matvec, precondition = solver.matvec(*args), solver.precondition(*args)
    shape = b.shape
    b = b.reshape(solver.vector_shape(state.preconditioner, state.pattern) + (-1,))
    tolerance = (solver.rtol**2) * _dot(b, b)

    def step(carry):
        x, r, p, rz, squared_residual, steps = carry
        z = precondition(r)
        rz_new = _dot(r, z)
        # The first step starts from p = 0.
        p = z + _ratio(rz_new, rz) * p
        q = matvec(p)
        alpha = _ratio(rz_new, _dot(p, q))
        r = r - alpha * q
        return x + alpha * p, r, p, rz_new, _dot(r, r), steps + 1

    def unconverged(carry):
        return jnp.any(carry[4] > tolerance) & (carry[5] < solver.max_steps)

    # One preconditioner application in the program, inside the loop, halves
    # the compilation of a multigrid cycle.
    zeros = jnp.zeros_like(b)
    carry = (zeros, b, zeros, jnp.ones_like(tolerance), _dot(b, b), 0)
    x, _, _, _, squared_residual, _ = lax.while_loop(unconverged, step, carry)
    # A NaN residual compares false.
    return x.reshape(shape), squared_residual <= tolerance


def _solve(
    solver: AbstractSparseIterativeSolver, state: SparseIterativeState, vector: Array
) -> tuple[Array, Array]:
    """The solution of one right-hand side, and whether it converged. Under
    `jax.vmap`, the right-hand sides of a batch solve together."""

    @jax.custom_batching.custom_vmap
    def solve_one(state, vector):
        x, solved = _cg(solver, state, vector[:, None])
        return x[:, 0], solved[0]

    @solve_one.def_vmap
    def solve_batch(axis_size: int, in_batched: Any, state, vectors):
        state_batched, vectors_batched = in_batched
        if any(jax.tree.leaves(state_batched)):
            # Other matrices for the right-hand sides: one solve each.
            axes = jax.tree.map(lambda batched: 0 if batched else None, state_batched)
            one = jax.vmap(
                lambda state, vector: solve_one.fun(state, vector),
                in_axes=(axes, 0 if vectors_batched else None),
            )
            return one(state, vectors), (True, True)
        if not vectors_batched:
            vectors = jnp.broadcast_to(vectors, (axis_size,) + vectors.shape)
        x, solved = _cg(solver, state, vectors.T)
        return (x.T, solved), (True, True)

    return solve_one(state, vector)
