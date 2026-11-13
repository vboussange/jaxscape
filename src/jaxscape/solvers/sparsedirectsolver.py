"""Sparse direct solvers assembled from factorization and solve phases."""

import abc
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
from lineax import RESULTS

from .sparsesolver import (
    _product,
    AbstractSparseSolver,
    AbstractSparseState,
    CSRPattern,
)


@final
class SparseDirectState[Factorization](AbstractSparseState):
    """State of an `AbstractSparseDirectSolver` for one sparsity pattern.

    **Attributes:**

    - `factorization`: The factorization the solver's phases return.
    - `pattern`: The CSR pattern of the matrix.
    - `values`: The CSR values that `factorization` factorizes.
    - `failed`: Whether `factorization` failed its check. Solves with a failed
      state fail, and the next `reinit` analyses its values afresh.
    """

    factorization: Factorization
    pattern: CSRPattern
    values: Array
    failed: Array


class AbstractSparseDirectSolver[Factorization](
    AbstractSparseSolver[SparseDirectState[Factorization]]
):
    """Direct solver for sparse symmetric positive definite matrices.

    Subclass it with the phases of a sparse factorization library: `factorize`,
    `solve`, `residual_tolerance` and, when the library can reuse an analysis,
    `refactorize`. The base class provides the Lineax solver, as
    `AbstractSparseSolver` describes: it factorizes the matrix again when its
    values change, and checks the results.

    `reinit` returns its state itself when the state passed its check and the
    values are those it factorizes, up to the rounding of assembling the
    matrix. Otherwise it refactorizes the state, whose factorization may no
    longer be valid afterwards, and analyses the values afresh when the state
    failed its check or when the refactorization fails its own.

    The checks assume nothing from the library. Each factorization solves
    `A x = A 1`, and fails when the backward error
    `max|A 1 - A x| / (||A||_inf max|x|)` is not finite or exceeds the square root
    of the machine epsilon. Each solve fails when its solution `x` of the scaled
    right-hand side `b` is not finite or `max|b - A x|` exceeds
    `residual_tolerance` times `max|b|`, and reports `lineax.RESULTS.singular`.

    A refactorization that fails its check is retried on a fresh analysis, and the
    next `reinit` of a failed state analyses its values afresh, so a failure leaves
    later values exact. The checks cost one solve of one right-hand side and two
    sparse matrix-vector products per factorization, and one sparse matrix-vector
    product per right-hand side.
    """

    @abc.abstractmethod
    def factorize(self, pattern: CSRPattern, values: Array) -> Factorization:
        """Analyse `pattern` and factorize `values` on a new handle.

        **Arguments:**

        - `pattern`: The CSR pattern of the matrix.
        - `values`: The CSR values of the matrix.

        **Returns:**

        A pytree to pass to `solve` and `refactorize`, with the same structure for
        every `pattern` and `values`, so that it can pass through `jax.lax.cond`.
        """

    def refactorize(
        self, factorization: Factorization, pattern: CSRPattern, values: Array
    ) -> Factorization:
        """Factorize new `values`. Defaults to a new analysis with `factorize`.

        **Arguments:**

        - `factorization`: A factorization of the same pattern. It need not stay
          valid after this call.
        - `pattern`: The CSR pattern of the matrix.
        - `values`: The new CSR values.

        **Returns:**

        The factorization of `values`.
        """
        del factorization
        return self.factorize(pattern, values)

    @abc.abstractmethod
    def solve(self, factorization: Factorization, vector: Array) -> Array:
        """Solve for one right-hand side, also under `jax.vmap` over `vector`.

        **Arguments:**

        - `factorization`: The factorization of the matrix.
        - `vector`: The right-hand side, with shape `(n,)`.

        **Returns:**

        The solution, with shape `(n,)`.
        """

    @abc.abstractmethod
    def residual_tolerance(self, dtype: Any) -> float:
        """Largest residual of a solution, relative to its right-hand side.

        **Arguments:**

        - `dtype`: The dtype of the solution.

        **Returns:**

        The tolerance of the infinity norms of the residual and the right-hand
        side, above the residuals this library leaves on valid matrices.
        """

    def _init(
        self, pattern: CSRPattern, values: Array
    ) -> SparseDirectState[Factorization]:
        return _analysed_state(self, pattern, values)

    def _reinit(
        self, state: SparseDirectState[Factorization], values: Array
    ) -> SparseDirectState[Factorization]:
        pattern = state.pattern

        def analysed(factorization: Any, values: Array) -> tuple[Any, Array]:
            del factorization
            return self._analysed(pattern, values)

        def refactorized(factorization: Any, values: Array) -> tuple[Any, Array]:
            factorization = self.refactorize(factorization, pattern, values)
            failed = self._check_fails(factorization, pattern, values)
            # A factorization can pass its check and still leave its handle
            # unusable, as one with vertices cut off from the ground vertex did
            # with cuDSS, so a failed refactorization is retried.
            return jax.lax.cond(
                failed,
                analysed,
                lambda factorization, _: (factorization, failed),
                factorization,
                values,
            )

        def updated(state: Any, values: Array) -> Any:
            factorization, failed = jax.lax.cond(
                state.failed, analysed, refactorized, state.factorization, values
            )
            return SparseDirectState(
                factorization=factorization,
                pattern=pattern,
                values=values,
                failed=failed,
            )

        tolerance = 16 * jnp.finfo(values.dtype).eps * jnp.abs(state.values)
        unchanged = ~state.failed & jnp.all(jnp.abs(values - state.values) <= tolerance)
        return jax.lax.cond(unchanged, lambda state, _: state, updated, state, values)

    def _solve(
        self, state: SparseDirectState[Factorization], vector: Array
    ) -> tuple[Array, RESULTS]:
        solution = self.solve(state.factorization, vector)
        product = _product(state.pattern, state.values, solution)
        tolerance = self.residual_tolerance(solution.dtype)
        # A NaN residual compares false. The scaling of `vector` keeps float32
        # solutions of tiny right-hand sides normal, so that they pass.
        solved = ~state.failed & (
            jnp.max(jnp.abs(vector - product)) <= tolerance * jnp.max(jnp.abs(vector))
        )
        return solution, RESULTS.where(solved, RESULTS.successful, RESULTS.singular)

    def _analysed(self, pattern: CSRPattern, values: Array) -> tuple[Any, Array]:
        factorization = self.factorize(pattern, values)
        return factorization, self._check_fails(factorization, pattern, values)

    def _check_fails(
        self, factorization: Factorization, pattern: CSRPattern, values: Array
    ) -> Array:
        """Whether `factorization` of `values` fails its check, a solve of `A x = A 1`.

        On grounded grid Laplacians of up to 1000 x 1000 cells, with up to six
        orders of magnitude of conductance contrast and a 1e-6 conductance floor,
        cuDSS factorizations had backward errors of at most 1.1e-6 in float32 and
        1.0e-15 in float64; those that met a non-positive pivot had at least
        3.9e-2, or were not finite.
        """
        b = _product(pattern, values, jnp.ones(pattern.n, values.dtype))
        x = self.solve(factorization, b)
        residual = jnp.max(jnp.abs(b - _product(pattern, values, x)))
        norm = jnp.max(
            jax.ops.segment_sum(
                jnp.abs(values),
                pattern.rows,
                num_segments=pattern.n,
                indices_are_sorted=True,
            )
        )
        tolerance = jnp.sqrt(jnp.finfo(values.dtype).eps)
        # A NaN residual compares false.
        return ~(residual <= tolerance * norm * jnp.max(jnp.abs(x)))


@eqx.filter_jit
def _analysed_state(
    solver: AbstractSparseDirectSolver, pattern: CSRPattern, values: Array
) -> SparseDirectState:
    """The state of `solver` for the CSR `values` of a matrix of `pattern`,
    analysed afresh, in one compiled program."""
    factorization, failed = solver._analysed(pattern, values)
    return SparseDirectState(
        factorization=factorization, pattern=pattern, values=values, failed=failed
    )
