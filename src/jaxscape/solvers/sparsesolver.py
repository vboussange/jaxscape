"""Sparse solvers of symmetric positive definite matrices, with a state for one
sparsity pattern."""

import abc
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
    - `columns`: Column of each CSR value, sorted within each row.
    - `rows`: Row of each CSR value.
    - `value_map`: For each stored entry of the BCOO matrix, the index of its CSR
      value, or the number of CSR values for padding entries.
    - `ell_slots`, `ell_columns`: The CSR values and columns of each row, padded
      to the longest row with the number of CSR values and column 0, with shape
      `(n, width)`, or `None` when they would hold more than twice as many
      entries as the CSR values, or rows longer than products unroll. Products
      of the matrix with vectors add up one term per column of them.
    """

    offsets: Array
    columns: Array
    rows: Array
    value_map: Array
    ell_slots: Array | None = None
    ell_columns: Array | None = None

    @property
    def n(self) -> int:
        return self.offsets.shape[0] - 1

    @property
    def nnz(self) -> int:
        return self.columns.shape[0]


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
    offsets = np.concatenate([[0], np.cumsum(row_counts)])
    columns = unique_keys % n
    value_map = np.full(indices.shape[0], nnz)
    value_map[valid] = slots
    ell_slots = ell_columns = None
    width = int(row_counts.max(initial=0))
    if n * width <= 2 * nnz and width <= _ELL_WIDTH:
        position = np.arange(width)
        stored = position[None, :] < row_counts[:, None]
        ell = np.where(stored, offsets[:-1, None] + position[None, :], nnz)
        ell_slots = jnp.asarray(ell, dtype=jnp.int32)
        ell_columns = jnp.asarray(np.append(columns, 0)[ell], dtype=jnp.int32)
    return CSRPattern(
        offsets=jnp.asarray(offsets, dtype=jnp.int32),
        columns=jnp.asarray(columns, dtype=jnp.int32),
        rows=jnp.asarray(unique_keys // n, dtype=jnp.int32),
        value_map=jnp.asarray(value_map, dtype=jnp.int32),
        ell_slots=ell_slots,
        ell_columns=ell_columns,
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
    same_entry = (pattern.rows[slot] == rows) & (pattern.columns[slot] == columns)
    matches = jnp.all(jnp.where(stored, in_bounds & same_entry, ~in_bounds))
    return eqx.error_if(matrix.data, ~matches, mismatch)


def _csr_values(data: Array, pattern: CSRPattern) -> Array:
    values = jnp.zeros(pattern.nnz, dtype=data.dtype)
    return values.at[pattern.value_map].add(data, mode="drop")


@jax.custom_batching.custom_vmap
def _product(pattern: CSRPattern, values: Array, vector: Array) -> Array:
    """The product of the matrix with CSR `values` and `vector`.

    With rows padded to the longest (see `CSRPattern`), it adds up one term per
    column of the padded rows, and `jax.vmap` gathers each term from the whole
    batch of vectors. The residual checks of a batch of solutions then cost
    little beside their solve, and less than one sum over the padded rows.
    Without padded rows, under `jax.vmap` over vectors alone, it computes the
    products one vector after another. XLA gathers the entries of a batch of
    vectors as a slice per index, which reads the batch in strides of the
    vector size and runs slower.
    """
    return _product_of(pattern, values, vector)


@_product.def_vmap
def _(
    axis_size: int, in_batched: Any, pattern: CSRPattern, values: Array, vector: Array
):
    del axis_size
    vectors_alone = in_batched[2] and not any(jax.tree.leaves(in_batched[:2]))
    if pattern.ell_slots is None and vectors_alone:
        return jax.lax.map(
            lambda vector: _product_of(pattern, values, vector), vector
        ), True
    axes = jax.tree.map(lambda batched: 0 if batched else None, in_batched)
    return jax.vmap(_product_of, in_axes=axes)(pattern, values, vector), True


def _product_of(pattern: CSRPattern, values: Array, vector: Array) -> Array:
    if pattern.ell_slots is None or pattern.ell_columns is None:
        return jax.ops.segment_sum(
            values * vector[pattern.columns],
            pattern.rows,
            num_segments=pattern.n,
            indices_are_sorted=True,
        )
    padded = jnp.append(values, jnp.zeros((1,), values.dtype))[pattern.ell_slots]
    product = jnp.zeros(pattern.n, jnp.result_type(values, vector))
    for k in range(padded.shape[1]):
        product = product + padded[:, k] * vector[pattern.ell_columns[:, k]]
    return product


# Longest rows that products pad (see `CSRPattern`). They add up one term per
# entry of a row, which XLA compiles separately, so their compile time grows
# with the length of the rows; products of longer rows sum over the CSR values.
_ELL_WIDTH = 32
