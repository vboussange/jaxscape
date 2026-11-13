"""Sparse solvers of symmetric positive definite matrices, with a state for one
sparsity pattern."""

import abc
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from lineax import AbstractLinearOperator, RESULTS

from .operator import AbstractReusableStateSolver


@final
class CSRPattern(eqx.Module):
    """Compressed sparse row pattern of a BCOO matrix.

    Duplicate entries of the BCOO matrix share one CSR value, and its padding
    entries, whose indices are out of bounds, are left out.

    **Attributes:**

    - `offsets`: Start of each row in `columns`, with shape `(n + 1,)`.
    - `columns`: Column of each CSR value, sorted within each row, with shape
      `(nnz,)`.
    - `value_map`: For each stored entry of the BCOO matrix, the index of its CSR
      value, or `nnz` for padding entries.
    - `max_row_length`: Number of CSR values of the longest row.
    - `n`: Number of rows, and of columns, of the matrix.
    - `nnz`: Number of CSR values.
    - `rows`: Row of each CSR value, with shape `(nnz,)`, computed from
      `offsets` at each access.
    """

    offsets: Array
    columns: Array
    value_map: Array
    max_row_length: int = eqx.field(static=True)

    @property
    def n(self) -> int:
        return self.offsets.shape[0] - 1

    @property
    def nnz(self) -> int:
        return self.columns.shape[0]

    @property
    def rows(self) -> Array:
        vertices = jnp.arange(self.n, dtype=self.offsets.dtype)
        counts = jnp.diff(self.offsets)
        return jnp.repeat(vertices, counts, total_repeat_length=self.nnz)


class AbstractSparseState(eqx.Module):
    """State of an `AbstractSparseSolver` for one sparsity pattern.

    **Attributes:**

    - `pattern`: The CSR pattern of the matrix, which `init` reads from concrete
      sparse indices.
    - `values`: The CSR values of the matrix, which `reinit` replaces.
    """

    pattern: eqx.AbstractVar[CSRPattern]
    values: eqx.AbstractVar[Array]


class AbstractSparseSolver[State: AbstractSparseState](AbstractReusableStateSolver):
    """Lineax solver of sparse symmetric positive definite matrices, with a state
    for one sparsity pattern.

    Subclass `AbstractSparseDirectSolver` with the phases of a sparse
    factorization library, or `AbstractSparseIterativeSolver` with a
    preconditioner. This class provides what they share. `init` reads the CSR
    pattern of the matrix from concrete sparse indices, and `reinit` replaces
    its values inside `jax.jit`, after checking that the sparse indices are
    those of the pattern. The state holds the pattern, the values, and the
    factorization or preconditioner of the solver.

    Each solve scales the right-hand side exactly, by a power of two, to a
    largest magnitude in [0.5, 1), so that its solution keeps the precision of
    the dtype: float32 right-hand sides of about 1e-37 had subnormal solutions.
    Right-hand sides below the smallest normal number solve to 0. A failed solve
    returns NaN and reports its failure through `lineax.RESULTS`, so it raises
    with `throw=True` and returns a failed result with `throw=False`. On GPUs,
    Lineax may only print the error, and the failed distances are then NaN.

    The matrix is symmetric: the state of a solver is also that of the
    transposed matrix.

    !!! warning

        `init` reads concrete sparse indices: `distance.init(graph)` builds the
        state of a `GridGraph` from its shape and contiguity, also inside
        `jax.jit`, and that of another graph outside `jax.jit`. A state belongs
        to one sparsity pattern: for a `GridGraph`, the raster shape and
        contiguity, with any conductances. `reinit` raises an error for an
        operator with other sparse indices.
    """

    @abc.abstractmethod
    def _init(self, pattern: CSRPattern, values: Array) -> State:
        """The state of the matrix of CSR `pattern` and `values`."""

    @abc.abstractmethod
    def _reinit(self, state: State, values: Array) -> State:
        """The state of the matrix of the pattern of `state` and `values`. It
        runs inside `jax.jit`."""

    @abc.abstractmethod
    def _solve(self, state: State, vector: Array) -> tuple[Array, RESULTS]:
        """The solution of `vector`, scaled to a largest magnitude in [0.5, 1),
        and whether the solve succeeded. It runs under `jax.vmap` over `vector`
        too."""

    def init(self, operator: AbstractLinearOperator, options: dict[str, Any]) -> State:
        """Return the state of `operator`, for its sparsity pattern and values.

        It reads the sparsity pattern from concrete sparse indices: it raises
        `ValueError` for traced sparse indices, as inside `jax.jit`, and
        `TypeError` for an operator that is not BCOO-backed. A solve uses the
        matrix that its state holds, whatever operator `lineax.linear_solve`
        receives: before solving another matrix of the same sparsity pattern,
        call `reinit` with its operator.

        **Arguments:**

        - `operator`: BCOO-backed operator of a symmetric positive definite
          matrix.
        - `options`: Unused.

        **Returns:**

        The state of `operator`, which keeps its dtype.
        """
        del options
        matrix = _matrix(self, operator)
        pattern = _pattern(self, matrix)
        return self._init(pattern, _csr_values(matrix.data, pattern))

    def reinit(
        self, operator: AbstractLinearOperator, options: dict[str, Any], state: State
    ) -> State:
        """Return the state of the values of `operator`, keeping its pattern.

        **Arguments:**

        - `operator`: BCOO-backed operator with the sparse indices `state` was
          initialized for.
        - `options`: Unused.
        - `state`: State from `init` or an earlier call.

        **Returns:**

        The state of the values of `operator`.
        """
        del options
        data = _checked_data(self, operator, state.pattern)
        if data.dtype != state.values.dtype:
            raise ValueError(
                f"{type(self).__name__}: the operator has dtype {data.dtype} and the "
                f"state {state.values.dtype}. A state keeps the dtype it was "
                "initialized with; initialize a new one for this precision."
            )
        return self._reinit(state, _csr_values(data, state.pattern))

    def compute(
        self, state: State, vector: Array, options: dict[str, Any]
    ) -> tuple[Array, RESULTS, dict[str, Any]]:
        del options
        norm = jnp.max(jnp.abs(vector))
        exponent = jnp.frexp(norm)[1]
        normal = norm >= jnp.finfo(vector.dtype).tiny
        scaled = jnp.where(normal, jnp.ldexp(vector, -exponent), 0)
        solution, result = self._solve(state, scaled)
        # A failed solve can leave finite solutions. They become NaN, so that
        # their values never look valid.
        solved = result == RESULTS.successful
        solution = jnp.where(solved, jnp.ldexp(solution, exponent), jnp.nan)
        return solution, result, {}

    def transpose(self, state: State, options: dict[str, Any]):
        return state, options

    def conj(self, state: State, options: dict[str, Any]):
        return state, options

    def assume_full_rank(self):
        return True


def _matrix(solver: AbstractSparseSolver, operator: AbstractLinearOperator) -> BCOO:
    matrix = operator.as_matrix()
    if not isinstance(matrix, BCOO):
        raise TypeError(f"{type(solver).__name__} requires a BCOO-backed operator.")
    return jax.lax.stop_gradient(matrix)


def _pattern(solver: AbstractSparseSolver, matrix: BCOO) -> CSRPattern:
    try:
        indices = np.asarray(matrix.indices, dtype=np.int64)
    except jax_errors.TracerArrayConversionError:
        raise ValueError(
            f"{type(solver).__name__} reads its sparsity pattern from concrete "
            "sparse indices, so it initializes outside `jax.jit`, except for the "
            "state of a `GridGraph`, which `distance.init(graph)` builds from its "
            "shape and contiguity. Create the state with `distance.init(graph)` "
            "(or `solver.init(operator, {})`) before compiling, and pass it as "
            "`state`."
        ) from None
    n = matrix.shape[0]
    valid = np.all((indices >= 0) & (indices < np.asarray(matrix.shape)), axis=1)
    keys = indices[valid, 0] * n + indices[valid, 1]
    unique_keys, slots = np.unique(keys, return_inverse=True)
    nnz = unique_keys.shape[0]
    row_counts = np.bincount(unique_keys // n, minlength=n)
    value_map = np.full(indices.shape[0], nnz)
    value_map[valid] = slots
    return CSRPattern(
        offsets=jnp.asarray(np.concatenate([[0], np.cumsum(row_counts)]), jnp.int32),
        columns=jnp.asarray(unique_keys % n, jnp.int32),
        value_map=jnp.asarray(value_map, jnp.int32),
        max_row_length=int(row_counts.max(initial=0)),
    )


def _checked_data(
    solver: AbstractSparseSolver,
    operator: AbstractLinearOperator,
    pattern: CSRPattern,
) -> Array:
    """Return the stored values of `operator` after checking its sparse indices.

    Each stored entry must sit where `pattern` puts it: at the row and column of
    its CSR value, or out of bounds for padding entries. Otherwise it raises, at
    trace time for another number of stored entries and at run time for other
    indices.
    """
    matrix = _matrix(solver, operator)
    mismatch = (
        f"{type(solver).__name__}: the operator's sparse indices differ from those "
        "the state was initialized for. A state belongs to one sparsity pattern; "
        "initialize a new one for this graph, e.g. with `distance.init(graph)`."
    )
    if matrix.indices.shape != (pattern.value_map.shape[0], 2):
        raise ValueError(mismatch)
    rows, columns = matrix.indices[:, 0], matrix.indices[:, 1]
    in_bounds = (
        (rows >= 0) & (rows < pattern.n) & (columns >= 0) & (columns < pattern.n)
    )
    stored = pattern.value_map < pattern.nnz
    slot = jnp.where(stored, pattern.value_map, 0)
    row = jnp.where(in_bounds, rows, 0)
    same_entry = (
        (pattern.offsets[row] <= slot)
        & (slot < pattern.offsets[row + 1])
        & (pattern.columns[slot] == columns)
    )
    matches = jnp.all(jnp.where(stored, in_bounds & same_entry, ~in_bounds))
    return eqx.error_if(matrix.data, ~matches, mismatch)


def _csr_values(data: Array, pattern: CSRPattern) -> Array:
    values = jnp.zeros(pattern.nnz, dtype=data.dtype)
    return values.at[pattern.value_map].add(data, mode="drop")


def _padded_rows(pattern: CSRPattern, values: Array) -> tuple[Array, Array]:
    """The CSR values and columns of each row, padded to `max_row_length`
    entries with entries of value 0 on the diagonal.

    **Returns:**

    The values and the columns, with shape `(n, max_row_length)`.
    """
    entry = jnp.arange(pattern.max_row_length)
    stored = entry < jnp.diff(pattern.offsets)[:, None]
    # Padding entries read at position `nnz`, past the CSR values, which gives
    # the fill value. Masking the positions, rather than the values read, also
    # keeps XLA:CPU in jaxlib 0.12 from miscompiling the reads: it returned
    # wrong values for some shapes.
    position = jnp.where(stored, pattern.offsets[:-1, None] + entry, pattern.nnz)
    padded = values.at[position].get(mode="fill", fill_value=0)
    columns = pattern.columns.at[position].get(mode="fill", fill_value=0)
    diagonal = jnp.arange(pattern.n, dtype=columns.dtype)[:, None]
    return padded, jnp.where(stored, columns, diagonal)


def _product(pattern: CSRPattern, values: Array) -> Callable[[Array], Array]:
    """The product of the matrix of CSR `pattern` and `values` with vectors of
    shape `(n,)`.

    When the padded rows (see `_padded_rows`) hold at most twice as many
    entries as the CSR values, and products unroll them, the product adds up
    one term per column of the padded rows, and `jax.vmap` gathers each term
    from the whole batch of vectors. The residual checks of a batch of
    solutions then cost little beside their solve, and less than one sum over
    the padded rows. Otherwise it sums over the CSR values, and under
    `jax.vmap` over vectors alone, it computes the products one vector after
    another: XLA gathers the entries of a batch of vectors as a slice per
    index, which reads the batch in strides of the vector size and runs slower.

    It prepares the padded rows, or the rows of the CSR values, once, outside
    the loops that apply the product.
    """
    length = pattern.max_row_length
    if pattern.n * length > 2 * pattern.nnz or length > _PADDED_ROW_LENGTH:
        rows = pattern.rows
        return lambda vector: _summed_product(rows, pattern.columns, values, vector)
    padded, columns = _padded_rows(pattern, values)

    def product(vector: Array) -> Array:
        result = jnp.zeros(pattern.n, jnp.result_type(values, vector))
        for k in range(length):
            result = result + padded[:, k] * vector[columns[:, k]]
        return result

    return product


def _sum(rows: Array, columns: Array, values: Array, vector: Array) -> Array:
    return jax.ops.segment_sum(
        values * vector[columns],
        rows,
        num_segments=vector.shape[0],
        indices_are_sorted=True,
    )


@jax.custom_batching.custom_vmap
def _summed_product(rows: Array, columns: Array, values: Array, vector: Array):
    return _sum(rows, columns, values, vector)


@_summed_product.def_vmap
def _(
    axis_size: int,
    in_batched: list[bool],
    rows: Array,
    columns: Array,
    values: Array,
    vector: Array,
):
    del axis_size
    if in_batched[3] and not any(in_batched[:3]):
        return jax.lax.map(
            lambda vector: _sum(rows, columns, values, vector), vector
        ), True
    axes = [0 if batched else None for batched in in_batched]
    return jax.vmap(_sum, in_axes=axes)(rows, columns, values, vector), True


# Longest rows that products pad (see `_product`). They add up one term per
# entry of a row, which XLA compiles separately, so their compile time grows
# with the length of the rows; products of longer rows sum over the CSR values.
_PADDED_ROW_LENGTH = 32
