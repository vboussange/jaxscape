"""Sparse direct solvers assembled from factorization and solve phases."""

import abc
from typing import Any, final, Generic, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from lineax import AbstractLinearOperator, RESULTS

from .operator import AbstractReusableStateSolver


_Factorization = TypeVar("_Factorization")


@final
class CSRPattern(eqx.Module):
    """Compressed sparse row pattern of a BCOO matrix.

    Duplicate entries of the BCOO matrix share one CSR value, and its padding
    entries, whose indices are out of bounds, are left out.

    **Attributes:**

    - `offsets`: Start of each row in `columns`, with shape `(n + 1,)`.
    - `columns`: Column of each CSR value, sorted within each row.
    - `rows`: Row of each CSR value.
    - `value_map`: For each stored entry of the BCOO matrix, the index of its CSR
      value, or the number of CSR values for padding entries.
    """

    offsets: Array
    columns: Array
    rows: Array
    value_map: Array

    @property
    def n(self) -> int:
        return self.offsets.shape[0] - 1

    @property
    def nnz(self) -> int:
        return self.columns.shape[0]


@final
class SparseDirectState(eqx.Module, Generic[_Factorization]):
    """State of an `AbstractSparseDirectSolver` for one sparsity pattern.

    **Attributes:**

    - `factorization`: The factorization the solver's phases return.
    - `pattern`: The CSR pattern of the matrix.
    - `values`: The CSR values that `factorization` factorizes.
    - `failed`: Whether `factorization` failed its check. Solves with a failed
      state fail, and the next `reinit` analyses its values afresh.
    """

    factorization: _Factorization
    pattern: CSRPattern
    values: Array
    failed: Array


class AbstractSparseDirectSolver(AbstractReusableStateSolver, Generic[_Factorization]):
    """Direct solver for sparse symmetric positive definite matrices.

    Subclass it with the phases of a sparse factorization library: `factorize`,
    `solve`, `residual_tolerance` and, when the library can reuse an analysis,
    `refactorize`. The base class provides the Lineax solver: it derives the CSR
    pattern once, refactorizes when the values change, and checks results.

    The checks assume nothing from the library. Each factorization solves
    `A x = A 1`, and fails when the backward error
    `max|A 1 - A x| / (||A||_inf max|x|)` is not finite or exceeds the square root
    of the machine epsilon. Each solve fails when its solution `x` of `b` is not
    finite or `max|b - A x|` exceeds `residual_tolerance` times `max|b|`. A failed
    solve returns NaN and reports `lineax.RESULTS.singular`, so it raises with
    `throw=True` and returns a failed result with `throw=False`. On GPUs, Lineax may
    only print the error, and the failed distances are then NaN.

    A refactorization that fails its check is retried on a fresh analysis, and the
    next `reinit` of a failed state analyses its values afresh, so a failure leaves
    later values exact. The checks cost one solve of one right-hand side and two
    sparse matrix-vector products per factorization, and one sparse matrix-vector
    product per right-hand side.

    !!! warning

        `init` reads concrete sparse indices, so create the state outside
        `jax.jit`. A state belongs to one sparsity pattern: for a `GridGraph`, the
        raster shape and contiguity, with any conductances. `reinit` raises an
        error for an operator with other sparse indices.
    """

    @abc.abstractmethod
    def factorize(self, pattern: CSRPattern, values: Array) -> _Factorization:
        """Analyse `pattern` and factorize `values` on a new handle.

        **Arguments:**

        - `pattern`: The CSR pattern of the matrix.
        - `values`: The CSR values of the matrix.

        **Returns:**

        A pytree to pass to `solve` and `refactorize`, with the same structure for
        every `pattern` and `values`, so that it can pass through `jax.lax.cond`.
        """

    def refactorize(
        self, factorization: _Factorization, pattern: CSRPattern, values: Array
    ) -> _Factorization:
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
    def solve(self, factorization: _Factorization, vector: Array) -> Array:
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

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> SparseDirectState[_Factorization]:
        del options
        matrix = self._matrix(operator)
        pattern = self._pattern(matrix)
        values = _csr_values(matrix.data, pattern)
        factorization, failed = self._analysed(pattern, values)
        return SparseDirectState(factorization, pattern, values, failed)

    def reinit(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        state: SparseDirectState[_Factorization],
    ) -> SparseDirectState[_Factorization]:
        """Return the state of the values of `operator`, keeping the analysis.

        **Arguments:**

        - `operator`: BCOO-backed operator with the sparse indices `state` was
          initialized for.
        - `options`: Unused.
        - `state`: State from `init` or an earlier call. Its factorization may no
          longer be valid afterwards.

        **Returns:**

        `state` itself when it passed its check and the values are those it
        factorizes, up to the rounding of assembling the matrix. Otherwise a state
        that refactorizes `state`, analysing the values afresh when `state` failed
        its check or when the refactorization fails its own.
        """
        del options
        pattern = state.pattern
        data = self._checked_data(operator, pattern)
        if data.dtype != state.values.dtype:
            raise ValueError(
                f"{type(self).__name__}: the operator has dtype {data.dtype} and the "
                f"state {state.values.dtype}. A state keeps the dtype it was "
                "initialized with; initialize a new one for this precision."
            )
        values = _csr_values(data, pattern)

        def analysed(factorization: Any, values: Array) -> tuple[Any, Array]:
            del factorization
            return self._analysed(pattern, values)

        def refactorized(factorization: Any, values: Array) -> tuple[Any, Array]:
            factorization = self.refactorize(factorization, pattern, values)
            failed = self._check_fails(factorization, pattern, values)
            # A factorization can pass its check and still leave its handle
            # unusable, as one with vertices cut off from the ground vertex did
            # with cuDSS on an RTX 3090, so a failed refactorization is retried.
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
            return SparseDirectState(factorization, pattern, values, failed)

        tolerance = 16 * jnp.finfo(values.dtype).eps * jnp.abs(state.values)
        unchanged = ~state.failed & jnp.all(jnp.abs(values - state.values) <= tolerance)
        return jax.lax.cond(unchanged, lambda state, _: state, updated, state, values)

    def compute(
        self,
        state: SparseDirectState[_Factorization],
        vector: Array,
        options: dict[str, Any],
    ) -> tuple[Array, RESULTS, dict[str, Any]]:
        del options
        solution = self.solve(state.factorization, vector)
        product = _product(state.pattern, state.values, solution)
        tolerance = self.residual_tolerance(solution.dtype)
        # A NaN residual compares false.
        solved = ~state.failed & (
            jnp.max(jnp.abs(vector - product)) <= tolerance * jnp.max(jnp.abs(vector))
        )
        # A failed solve can leave finite solutions. They become NaN, so that
        # their values never look valid.
        solution = jnp.where(solved, solution, jnp.nan)
        return solution, RESULTS.where(solved, RESULTS.successful, RESULTS.singular), {}

    def transpose(self, state: SparseDirectState[_Factorization], options: dict):
        return state, options

    def conj(self, state: SparseDirectState[_Factorization], options: dict):
        return state, options

    def assume_full_rank(self):
        return True

    def _analysed(self, pattern: CSRPattern, values: Array) -> tuple[Any, Array]:
        factorization = self.factorize(pattern, values)
        return factorization, self._check_fails(factorization, pattern, values)

    def _check_fails(
        self, factorization: _Factorization, pattern: CSRPattern, values: Array
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

    def _matrix(self, operator: AbstractLinearOperator) -> BCOO:
        matrix = operator.as_matrix()
        if not isinstance(matrix, BCOO):
            raise TypeError(f"{type(self).__name__} requires a BCOO-backed operator.")
        return jax.lax.stop_gradient(matrix)

    def _pattern(self, matrix: BCOO) -> CSRPattern:
        try:
            indices = np.asarray(matrix.indices, dtype=np.int64)
        except jax_errors.TracerArrayConversionError:
            raise ValueError(
                f"{type(self).__name__} derives its sparsity pattern from concrete "
                "sparse indices, so it initializes outside `jax.jit`. Create the "
                "state with `distance.init(graph)` (or `solver.init(operator, {})`) "
                "before compiling, and pass it as `state`."
            ) from None
        n = matrix.shape[0]
        valid = np.all((indices >= 0) & (indices < np.asarray(matrix.shape)), axis=1)
        keys = indices[valid, 0] * n + indices[valid, 1]
        unique_keys, slots = np.unique(keys, return_inverse=True)
        row_counts = np.bincount(unique_keys // n, minlength=n)
        value_map = np.full(indices.shape[0], unique_keys.shape[0])
        value_map[valid] = slots
        return CSRPattern(
            offsets=jnp.asarray(
                np.concatenate([[0], np.cumsum(row_counts)]), dtype=jnp.int32
            ),
            columns=jnp.asarray(unique_keys % n, dtype=jnp.int32),
            rows=jnp.asarray(unique_keys // n, dtype=jnp.int32),
            value_map=jnp.asarray(value_map, dtype=jnp.int32),
        )

    def _checked_data(
        self, operator: AbstractLinearOperator, pattern: CSRPattern
    ) -> Array:
        """Return the stored values of `operator` after checking its sparse indices.

        Each stored entry must sit where `pattern` puts it: at the row and column
        of its CSR value, or out of bounds for padding entries. Otherwise it
        raises, at trace time for another number of stored entries and at run time
        for other indices.
        """
        matrix = self._matrix(operator)
        mismatch = (
            f"{type(self).__name__}: the operator's sparse indices differ from those "
            "the state was initialized for. A state belongs to one sparsity pattern; "
            "initialize a new one, e.g. with `distance.init(graph)`, for this graph "
            "geometry."
        )
        if matrix.indices.shape != (pattern.value_map.shape[0], 2):
            raise ValueError(mismatch)
        rows, columns = matrix.indices[:, 0], matrix.indices[:, 1]
        in_bounds = (
            (rows >= 0) & (rows < pattern.n) & (columns >= 0) & (columns < pattern.n)
        )
        stored = pattern.value_map < pattern.nnz
        slot = jnp.where(stored, pattern.value_map, 0)
        same_entry = (pattern.rows[slot] == rows) & (pattern.columns[slot] == columns)
        matches = jnp.all(jnp.where(stored, in_bounds & same_entry, ~in_bounds))
        return eqx.error_if(matrix.data, ~matches, mismatch)


def _csr_values(data: Array, pattern: CSRPattern) -> Array:
    values = jnp.zeros(pattern.nnz, dtype=data.dtype)
    return values.at[pattern.value_map].add(data, mode="drop")


def _product(pattern: CSRPattern, values: Array, vector: Array) -> Array:
    """The product of the matrix with CSR `values` and `vector`."""
    return jax.ops.segment_sum(
        values * vector[pattern.columns],
        pattern.rows,
        num_segments=pattern.n,
        indices_are_sorted=True,
    )
