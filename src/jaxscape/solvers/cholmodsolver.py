from typing import Any, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array as JaxArray, errors as jax_errors
from jax.experimental.sparse import BCOO
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, RESULTS
from lineax._solver.misc import (
    pack_structures,
    PackedStructures,
    transpose_packed_structures,
    unravel_solution,
)

from .operator import AbstractReusableStateSolver


try:
    import cholespy as _cholespy

    CHOLESPY_AVAILABLE = True
except ImportError:
    _cholespy = None
    CHOLESPY_AVAILABLE = False


class _CholespyFactor(eqx.Module):
    # Native cholespy factors are host objects, so keep them static leaves rather
    # than asking JAX to trace or transfer them.
    solver: Any = eqx.field(static=True)
    solve_dtype: np.dtype = eqx.field(static=True)
    size: int = eqx.field(static=True)


@jax.tree_util.register_static
class _FactorCache:
    """The last factorization of a solver state, kept on the host.

    A state keeps one cache for its whole life, so its structure stays the same
    under `jax.jit` whatever the matrix values. A factorization is a function of
    the matrix, so caching it leaves host callbacks pure.
    """

    def __init__(self) -> None:
        self._entry: tuple[np.ndarray, _CholespyFactor] | None = None

    def factor_of(self, A_bcoo: BCOO, solve_dtype: np.dtype) -> _CholespyFactor:
        # The first slice of any callback batch, in NumPy: this runs on every solve.
        data = np.asarray(A_bcoo.data)
        data = data.reshape(-1, data.shape[-1])[0]
        entry = self._entry
        if (
            entry is not None
            and entry[1].solve_dtype == solve_dtype
            and np.array_equal(entry[0], data)
        ):
            return entry[1]
        factor = _factorize_host(_unbatched_bcoo(A_bcoo), solve_dtype)
        self._entry = (np.array(data), factor)
        return factor


_CholmodSolverState: TypeAlias = tuple[BCOO, PackedStructures, _FactorCache | None]


class CholmodSolver(AbstractReusableStateSolver):
    """
    A linear solver that uses CHOLMOD (via cholespy) to solve a sparse linear system.
    Uses direct Cholesky factorization for symmetric positive definite matrices.
    A state keeps its last factorization, which solves reuse while the matrix values
    stay the same.


    !!! example

        ```python
        from jaxscape.solvers import CholmodSolver

        solver = CholmodSolver()
        distance = ResistanceDistance(solver=solver)
        state = distance.init(grid)  # pre-factorizes the grounded Laplacian
        dist = distance(grid, state=state)
        ```

    !!! warning
        `cholespy` must be installed to use this solver.

    !!! warning
        Float64 callback-backed solves require process-wide x64 support to be
        enabled before JAX work starts, e.g. via `JAX_ENABLE_X64=1` or
        `jax.config.update("jax_enable_x64", True)`. A thread-local
        `with jax.enable_x64()` block is not sufficient for host callback
        threads, and can surface as a callback dtype mismatch such as
        `Expected: float64, Actual: float32`.
    """

    factorize_in_init: bool = True

    def __check_init__(self):
        if not CHOLESPY_AVAILABLE:
            raise ImportError(
                "cholespy is required for CholmodSolver. "
                "Install it with: pip install cholespy"
            )

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _CholmodSolverState:
        del options

        A_bcoo = _bcoo_matrix(operator)
        packed_structures = pack_structures(operator)
        # `init` may be called inside a traced Lineax solve. In that case the
        # sparse buffers are abstract values, so defer native factorization to
        # the host callback where concrete arrays are available. A compiled
        # function would keep a cache across its calls, so each solve factorizes.
        if not _can_materialize_bcoo(A_bcoo):
            return A_bcoo, packed_structures, None
        cache = _FactorCache()
        solve_dtype = _factor_dtype_from_matrix(A_bcoo)
        if self.factorize_in_init and solve_dtype is not None:
            cache.factor_of(A_bcoo, solve_dtype)
        return A_bcoo, packed_structures, cache

    def reinit(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        state: _CholmodSolverState,
    ) -> _CholmodSolverState:
        """Point `state` at `operator`. Solves factorize new values when they run."""
        del options
        return _bcoo_matrix(operator), pack_structures(operator), state[2]

    def compute(
        self,
        state: _CholmodSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """
        Compute the solution to the linear system.
        """
        del options
        b_jax = vector
        A_bcoo, packed_structures, cache = state
        result_shape = jax.ShapeDtypeStruct(b_jax.shape, b_jax.dtype)

        def solve_host(A_bcoo: BCOO, b_jax: JaxArray) -> np.ndarray:
            solve_dtype = _solve_dtype_from_rhs(b_jax)
            if cache is None:
                factor = _factorize_host(_unbatched_bcoo(A_bcoo), solve_dtype)
            else:
                factor = cache.factor_of(A_bcoo, solve_dtype)
            return _solve_with_factor_host(factor, b_jax)

        # `expand_dims` coalesces vmapped RHS solves into one batched host callback.
        solution = eqx.filter_pure_callback(
            solve_host,
            A_bcoo,
            b_jax,
            result_shape_dtypes=result_shape,
            vmap_method="expand_dims",
        )

        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CholmodSolverState, options: dict[str, Any]):
        del options
        A_bcoo, packed_structures, cache = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        A_bcoo_T = A_bcoo.T
        transpose_state = (A_bcoo_T, transposed_packed_structures, cache)
        return transpose_state, {}

    def conj(self, state: _CholmodSolverState, options: dict[str, Any]):
        del options
        A_bcoo, packed_structures, cache = state
        A_conj = BCOO(
            (jnp.conj(A_bcoo.data), A_bcoo.indices),
            shape=A_bcoo.shape,
            indices_sorted=A_bcoo.indices_sorted,
            unique_indices=A_bcoo.unique_indices,
        )
        conj_state = (A_conj, packed_structures, cache)
        return conj_state, {}

    def assume_full_rank(self):
        return True


def _bcoo_matrix(operator: AbstractLinearOperator) -> BCOO:
    A_bcoo = operator.as_matrix()
    if not isinstance(A_bcoo, BCOO):
        raise ValueError("CholmodSolver requires a BCOO-backed operator.")
    return A_bcoo


def _can_materialize_bcoo(A_bcoo: BCOO) -> bool:
    leaves = jax.tree.leaves((A_bcoo.data, A_bcoo.indices))
    try:
        for leaf in leaves:
            # Converting to NumPy is the relevant test: concrete host/device
            # arrays can be materialized for cholespy, tracers cannot.
            np.asarray(leaf)
    except (jax_errors.TracerArrayConversionError, TypeError):
        return False
    return True


def _factor_dtype_from_matrix(A_bcoo: BCOO) -> np.dtype | None:
    dtype = np.dtype(A_bcoo.data.dtype)
    if dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        return dtype
    return None


def _solve_dtype_from_rhs(rhs: JaxArray) -> np.dtype:
    dtype = np.dtype(rhs.dtype)
    if dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        return dtype
    raise TypeError(
        "CholmodSolver supports only float32 and float64 right-hand sides; "
        f"got {dtype}."
    )


def _factorize_host(A_bcoo: BCOO, solve_dtype: np.dtype) -> _CholespyFactor:
    # CHOLMOD expects canonical sparse input. Duplicate BCOO entries can arise
    # from sparse arithmetic and may trigger native cholespy aborts.
    A_bcoo = _unbatched_bcoo(A_bcoo).sum_duplicates()
    if A_bcoo.shape[0] != A_bcoo.shape[1]:
        raise ValueError("CholmodSolver requires a square operator.")

    # cholespy rejects read-only JAX -> NumPy views; make writable C-order host
    # buffers while preserving the matrix value dtype.
    indices = _writable_c_array(A_bcoo.indices, dtype=np.int32)
    data = _writable_c_array(A_bcoo.data)
    if np.iscomplexobj(data) or not np.issubdtype(data.dtype, np.floating):
        raise TypeError(
            "CholmodSolver supports only real floating-point sparse matrices."
        )

    solver_cls = _cholespy_solver_cls(solve_dtype)
    assert _cholespy is not None
    matrix_type = getattr(_cholespy, "MatrixType")
    solver = solver_cls(
        A_bcoo.shape[0],
        _writable_c_array(indices[:, 0], dtype=np.int32),
        _writable_c_array(indices[:, 1], dtype=np.int32),
        data,
        matrix_type.COO,
    )
    return _CholespyFactor(solver, solve_dtype, A_bcoo.shape[0])


def _solve_with_factor_host(factor: _CholespyFactor, b_jax: JaxArray) -> np.ndarray:
    if b_jax.ndim == 0:
        raise ValueError("Right-hand side must have at least one dimension.")

    # The cholespy solve API is in-place and has no implicit casting: RHS and
    # output buffers must exactly match CholeskySolverF/D precision.
    rhs = _writable_c_array(b_jax, dtype=factor.solve_dtype)
    if rhs.shape[-1] != factor.size:
        raise ValueError(
            "The last dimension of b must match the operator dimension."
        )

    rhs_shape = rhs.shape
    # Lineax gives vector-shaped RHS values; cholespy's fast path expects
    # `(n_rows, n_rhs)`, so flatten any leading callback/vmap dimensions.
    rhs_matrix = _writable_c_array(rhs.reshape((-1, factor.size)).T)
    solution_matrix = np.empty_like(rhs_matrix)
    factor.solver.solve(rhs_matrix, solution_matrix)
    return solution_matrix.T.reshape(rhs_shape)


def _cholespy_solver_cls(solve_dtype: np.dtype) -> Any:
    assert _cholespy is not None
    if solve_dtype == np.dtype(np.float32):
        return getattr(_cholespy, "CholeskySolverF")
    if solve_dtype == np.dtype(np.float64):
        return getattr(_cholespy, "CholeskySolverD")
    raise TypeError(
        "CholmodSolver supports only float32 and float64 right-hand sides; "
        f"got {solve_dtype}."
    )


def _writable_c_array(array: Any, dtype: Any | None = None) -> np.ndarray:
    array_np = np.asarray(array)
    return np.require(array_np, dtype=dtype, requirements=["C", "W"])


def _unbatched_bcoo(matrix: BCOO) -> BCOO:
    if matrix.n_batch == 0:
        return matrix

    # `pure_callback(..., vmap_method="expand_dims")` can wrap the shared matrix
    # in a batch dimension while batching only RHS values. The first slice is the
    # matrix to factor; RHS batching is handled separately in `_solve_with_factor_host`.
    batch_index = (0,) * matrix.n_batch
    return BCOO(
        (matrix.data[batch_index], matrix.indices[batch_index]),
        shape=matrix.shape[-2:],
        indices_sorted=matrix.indices_sorted,
        unique_indices=matrix.unique_indices,
    )
