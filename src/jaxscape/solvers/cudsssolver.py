"""Sparse Cholesky on NVIDIA GPUs with cuDSS, through spineax's phase API."""

import importlib
import importlib.util
from types import ModuleType
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from lineax import AbstractLinearOperator, RESULTS

from .operator import AbstractRefreshingSolver


@final
class CuDSSSolverState(eqx.Module):
    """State of `CuDSSSolver` for one sparsity pattern.

    **Attributes:**

    - `token`: spineax's factorization token. It holds the CSR sparsity pattern
      and the values of the last factorization.
    - `value_map`: For each stored entry of the BCOO matrix, the index of its
      CSR value, or the number of CSR values for padding entries.
    - `rows`: The row of each CSR value.
    """

    token: Any
    value_map: Array
    rows: Array


@final
class CuDSSSolver(AbstractRefreshingSolver):
    """Sparse Cholesky solver running NVIDIA cuDSS on the GPU through spineax.

    It solves symmetric positive definite systems, such as grounded graph
    Laplacians, with a direct factorization whose cost and accuracy stay the
    same however heterogeneous the conductances are.

    `init` derives the CSR sparsity pattern of the BCOO matrix once: duplicate
    entries are summed and padding entries are left out. It then runs cuDSS
    analysis and factorization. Each resistance-distance query refactorizes
    the state for the current matrix values into a new state, and all
    right-hand sides of the query share that factorization in one multi-RHS
    cuDSS solve. The `*_distance_and_state` queries return the new state, so
    an optimization loop passes it from one step to the next and only pays
    for refactorization. Gradients of exact resistance distances reuse the
    forward solutions. Gradients of Spielman approximate distances and of
    sparse linear solves solve the adjoint system with the forward
    factorization, because the matrix is symmetric and the solver is its own
    transpose. No gradient is taken through analysis or factorization.

    It factorizes and solves in the dtype of the matrix, float32 or float64.
    spineax reports no factorization status, so each solve checks its solution
    `x` against the factorized matrix `A`: a solution that is not finite, or
    whose residual `max|b - A x|` exceeds `max|b|` times 1e-4 in float64 or
    1e-1 in float32, becomes NaN and reports `lineax.RESULTS.singular`. A
    matrix that is not positive definite, or is numerically singular,
    therefore makes the solve raise with `throw=True` and return a failed
    result with `throw=False`. The check costs one sparse matrix-vector
    product per right-hand side.

    !!! example

        ```python
        # Runs on an NVIDIA GPU with the `cudss` extra installed.
        import jax

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp
        from jaxscape import GridGraph, ResistanceDistance
        from jaxscape.solvers import CuDSSSolver


        def landscape(permeability):
            return GridGraph(permeability, fun=lambda x, y: (x + y) / 2)


        distance = ResistanceDistance(solver=CuDSSSolver())
        nodes = jnp.array([0, 45, 99])
        # Analysis and factorization, outside `jax.jit`.
        state = distance.init(landscape(jnp.ones((10, 10))))


        @jax.jit
        def step(permeability, state):
            # Refactorizes `state` and returns the refactorized state.
            return distance.nodes_to_nodes_distance_and_state(
                landscape(permeability), nodes, state
            )


        for seed in range(3):
            key = jax.random.key(seed)
            permeability = jax.random.uniform(key, (10, 10), minval=1e-3)
            distances, state = step(permeability, state)
        ```

    !!! warning

        `CuDSSSolver` requires an NVIDIA GPU, CUDA-enabled JAX and spineax with
        cuDSS below 1.0. The `cudss` extra, `pip install 'jaxscape[cudss]'`,
        installs spineax 0.0.5, its CUDA 13 build, and cuDSS below 1.0 on Linux
        x86-64 with Python 3.12 or newer. The
        [linear solvers guide](https://vboussange.github.io/jaxscape/api/linear_solvers/)
        describes a spineax build for CUDA 12. Constructing `CuDSSSolver`
        elsewhere raises an error naming what is missing; it has no CPU
        fallback.

    !!! warning

        `init` reads concrete sparse indices, so create the state outside
        `jax.jit`. A state belongs to one sparsity pattern, i.e. one graph
        geometry with any conductances; `refresh_state` maps values onto that
        pattern by position and does not compare sparse indices. A query uses
        up the factorization of the state it is given, because cuDSS handles
        cannot be copied (spineax issue #27). Pass the state that a
        `*_distance_and_state` query returns to the next call. Passing a
        used-up state again gives correct results and repeats the analysis.
        Query forms that return distances only, such as
        `distance(graph, state=state)`, discard the refactorized state, so
        each further call with the same `state` repeats the analysis.
    """

    def __check_init__(self):
        _load_cudss()

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> CuDSSSolverState:
        del options
        cudss = _load_cudss()
        matrix = jax.lax.stop_gradient(_bcoo_matrix(operator))
        offsets, columns, rows, value_map = _csr_pattern(matrix)
        values = _csr_values(matrix.data, value_map, columns.shape[0])
        token = cudss.analyze(values, offsets, columns, mtype_id="spd", mview_id="full")
        token = cudss.factorize(token, values)
        return CuDSSSolverState(token=token, value_map=value_map, rows=rows)

    def refresh_state(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        state: CuDSSSolverState,
    ) -> CuDSSSolverState:
        """Refactorize `state` for the values of `operator`.

        **Arguments:**

        - `operator`: BCOO-backed operator with the sparsity pattern `state`
          was initialized for, and possibly different values.
        - `options`: Unused.
        - `state`: State from `init` or an earlier call.

        **Returns:**

        A new state factorized for `operator`, to solve with and to pass to
        the next call. The factorization of `state` is used up.
        """
        del options
        matrix = _bcoo_matrix(operator)
        values = _csr_values(matrix.data, state.value_map, state.token.nnz)
        token = _load_cudss().refactorize(state.token, values)
        return CuDSSSolverState(token=token, value_map=state.value_map, rows=state.rows)

    def compute(
        self,
        state: CuDSSSolverState,
        vector: Array,
        options: dict[str, Any],
    ) -> tuple[Array, RESULTS, dict[str, Any]]:
        del options
        solution = _load_cudss().solve(state.token, vector)
        solved = _solves(state, solution, vector)
        # spineax reports no factorization status, and a failed factorization
        # can leave finite solutions. Failed solutions become NaN, so that
        # their values never look valid.
        solution = jnp.where(solved, solution, jnp.nan)
        return solution, RESULTS.where(solved, RESULTS.successful, RESULTS.singular), {}

    def transpose(self, state: CuDSSSolverState, options: dict[str, Any]):
        return state, options

    def conj(self, state: CuDSSSolverState, options: dict[str, Any]):
        return state, options

    def assume_full_rank(self):
        return True


def _load_cudss() -> ModuleType:
    """Return spineax's cuDSS module, or raise an error naming what is missing.

    Importing `spineax.cudss` initializes JAX's backends and loads the CUDA
    libraries, so it happens only here, when a `CuDSSSolver` is used.
    """
    if importlib.util.find_spec("spineax") is None:
        raise ImportError(
            "CuDSSSolver requires spineax, which is not installed. Install it "
            "with `pip install 'jaxscape[cudss]'` on Linux x86-64 with Python "
            "3.12 or newer."
        )
    backend = jax.default_backend()
    if backend != "gpu" or "cuda" not in _platform_version().lower():
        raise RuntimeError(
            "CuDSSSolver requires an NVIDIA GPU and CUDA-enabled JAX; JAX's "
            f"default backend is {backend!r} ({_platform_version()!r}). Install "
            "a JAX CUDA plugin, such as `pip install 'jax[cuda13]'`, and run on "
            "an NVIDIA GPU. CuDSSSolver has no CPU fallback."
        )
    try:
        return importlib.import_module("spineax.cudss")
    except Exception as error:
        cause = error.__cause__ or error
        raise ImportError(
            "CuDSSSolver could not load the cuDSS runtime through spineax: "
            f"{cause}. spineax 0.0.5 needs cuDSS below 1.0 "
            "(`pip install 'nvidia-cudss-cu13>=0.8,<1'`) and the CUDA 13 "
            "runtime, or a spineax build against your CUDA version."
        ) from error


def _platform_version() -> str:
    return str(jax.devices()[0].client.platform_version)


def _bcoo_matrix(operator: AbstractLinearOperator) -> BCOO:
    matrix = operator.as_matrix()
    if not isinstance(matrix, BCOO):
        raise TypeError("CuDSSSolver requires a BCOO-backed operator.")
    return matrix


def _csr_pattern(matrix: BCOO) -> tuple[Array, Array, Array, Array]:
    """CSR pattern of `matrix`, the row of each CSR value, and the map from its
    stored entries to CSR values.

    The pattern is sorted and unique: duplicate entries share one CSR value and
    padding entries, whose indices are out of bounds, are left out.
    """
    try:
        indices = np.asarray(matrix.indices, dtype=np.int64)
    except jax_errors.TracerArrayConversionError:
        raise ValueError(
            "CuDSSSolver derives its sparsity pattern from concrete sparse "
            "indices, so it initializes outside `jax.jit`. Create the state "
            "with `distance.init(graph)` (or `solver.init(operator, {})`) "
            "before compiling, and pass it as `state`."
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
    return (
        jnp.asarray(offsets, dtype=jnp.int32),
        jnp.asarray(columns, dtype=jnp.int32),
        jnp.asarray(unique_keys // n, dtype=jnp.int32),
        jnp.asarray(value_map, dtype=jnp.int32),
    )


def _csr_values(data: Array, value_map: Array, nnz: int) -> Array:
    return jnp.zeros(nnz, dtype=data.dtype).at[value_map].add(data, mode="drop")


# Largest residual of a solution, relative to its right-hand side, in the
# infinity norm. A failed Cholesky factorization leaves solutions with NaN or
# with residuals of the order of the right-hand side. A successful one of
# grounded Laplacians with six orders of conductance contrast and a 1e-6 floor
# keeps residuals below 1e-7 in float64, up to 1000 x 1000 rasters (#40), and
# mostly below 1e-1 in float32.
_RESIDUAL_TOLERANCE = {jnp.dtype(jnp.float32): 1e-1, jnp.dtype(jnp.float64): 1e-4}


def _solves(state: CuDSSSolverState, solution: Array, vector: Array) -> Array:
    """Whether `solution` solves the factorized matrix for `vector`.

    It is false for non-finite solutions and for residuals above the tolerance
    of their precision. It costs one sparse matrix-vector product with the
    factorized matrix and two maximum reductions.
    """
    token = state.token
    product = jax.ops.segment_sum(
        token.values * solution[token.columns],
        state.rows,
        num_segments=token.n,
        indices_are_sorted=True,
    )
    residual = jnp.max(jnp.abs(vector - product))
    tolerance = _RESIDUAL_TOLERANCE[jnp.dtype(solution.dtype)]
    # A NaN residual compares false.
    return residual <= tolerance * jnp.max(jnp.abs(vector))
