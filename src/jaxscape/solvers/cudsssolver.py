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
    """

    token: Any
    value_map: Array


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
    forward solutions.

    !!! example

        ```python
        import jax
        from jaxscape import GridGraph, ResistanceDistance
        from jaxscape.solvers import CuDSSSolver

        distance = ResistanceDistance(solver=CuDSSSolver())
        state = distance.init(grid)  # analysis and factorization

        @jax.jit
        def step(permeability, state):
            grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
            # Refactorizes `state` and returns the refactorized state.
            return distance.nodes_to_nodes_distance_and_state(grid, nodes, state)

        for permeability in permeabilities:
            distances, state = step(permeability, state)
        ```

    !!! warning

        `CuDSSSolver` requires an NVIDIA GPU, CUDA-enabled JAX and the `cudss`
        extra, `pip install 'jaxscape[cudss]'`, which installs spineax 0.0.5
        and cuDSS below 1.0 on Linux x86-64 with Python 3.12 or newer.
        Constructing it elsewhere raises an error naming what is missing; it
        has no CPU fallback.

    !!! warning

        `init` reads concrete sparse indices, so create the state outside
        `jax.jit`. A state belongs to one sparsity pattern, i.e. one graph
        geometry with any conductances. A query uses up the factorization of
        the state it is given, because cuDSS handles cannot be copied (spineax
        issue #27). Pass the state that a `*_distance_and_state` query returns
        to the next call. Passing a used-up state again gives correct results
        and repeats the analysis.
    """

    def __check_init__(self):
        _load_cudss()

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> CuDSSSolverState:
        del options
        cudss = _load_cudss()
        matrix = jax.lax.stop_gradient(_bcoo_matrix(operator))
        offsets, columns, value_map = _csr_pattern(matrix)
        values = _csr_values(matrix.data, value_map, columns.shape[0])
        token = cudss.analyze(values, offsets, columns, mtype_id="spd", mview_id="full")
        token = cudss.factorize(token, values)
        return CuDSSSolverState(token=token, value_map=value_map)

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
        return CuDSSSolverState(token=token, value_map=state.value_map)

    def compute(
        self,
        state: CuDSSSolverState,
        vector: Array,
        options: dict[str, Any],
    ) -> tuple[Array, RESULTS, dict[str, Any]]:
        del options
        return _load_cudss().solve(state.token, vector), RESULTS.successful, {}

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


def _csr_pattern(matrix: BCOO) -> tuple[Array, Array, Array]:
    """CSR pattern of `matrix` and the map from its stored entries to CSR values.

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
        jnp.asarray(value_map, dtype=jnp.int32),
    )


def _csr_values(data: Array, value_map: Array, nnz: int) -> Array:
    return jnp.zeros(nnz, dtype=data.dtype).at[value_map].add(data, mode="drop")
