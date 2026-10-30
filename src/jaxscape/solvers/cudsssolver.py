"""Sparse Cholesky on NVIDIA GPUs with cuDSS, through spineax's phase API."""

import importlib
import importlib.util
from types import ModuleType
from typing import Any, final

import jax
import jax.numpy as jnp
from jaxtyping import Array

from .sparsedirectsolver import AbstractSparseDirectSolver, CSRPattern


# Finding the spec imports nothing: spineax and the CUDA libraries load only
# when a `CuDSSSolver` is constructed.
CUDSS_AVAILABLE = importlib.util.find_spec("spineax") is not None


@final
class CuDSSSolver(AbstractSparseDirectSolver[Any]):
    """Sparse Cholesky solver running NVIDIA cuDSS on the GPU through spineax.

    Solves float32 and float64 systems. One query shares a factorization across
    its right-hand sides in one multi-RHS cuDSS solve, and it is not
    differentiated: gradients reuse the forward solutions or factorization. See
    `AbstractSparseDirectSolver` for the checks of each factorization and solve,
    which spineax does not report itself.

    !!! warning

        It needs an NVIDIA GPU of compute capability 7.5 or newer and CUDA-enabled
        JAX, and raises an error naming what is missing elsewhere. The `cudss`
        extra, `pip install 'jaxscape[cudss]'`, installs spineax 0.0.5 and cuDSS
        below 1.0 for CUDA 13, on Linux x86-64 with Python 3.12 or newer. It brings
        JAX's CUDA 13 plugin, so leave out the `cuda` extra. For CUDA 12, build
        spineax from source as in
        [these notes](https://github.com/vboussange/jaxscape/issues/49#issuecomment-6501729059).

    !!! warning

        cuDSS allocates factorizations outside JAX's memory pool, so leave room
        with `XLA_PYTHON_CLIENT_PREALLOCATE=false`. spineax keeps the 8 most
        recently used factorizations in device memory; `SPINEAX_FACTOR_CACHE` sets
        that number.

    !!! note

        `reinit` uses up the factorization of the state it refactorizes, because
        cuDSS handles cannot be copied (spineax issue #27). A used-up or evicted
        state stays correct, and spineax repeats its costly analysis, so pass each
        new state to the next `reinit`. In float32, vertices cut off from the ground
        vertex can pass the solve check on grids of 300 x 300 cells or more; use
        float64 when conductances can be zero.
    """

    def __check_init__(self):
        _load_cudss()

    def factorize(self, pattern: CSRPattern, values: Array) -> Any:
        cudss = _load_cudss()
        token = cudss.analyze(
            values, pattern.offsets, pattern.columns, mtype_id="spd", mview_id="full"
        )
        return cudss.factorize(token, values)

    def refactorize(
        self, factorization: Any, pattern: CSRPattern, values: Array
    ) -> Any:
        del pattern
        return _load_cudss().refactorize(factorization, values)

    def solve(self, factorization: Any, vector: Array) -> Array:
        return _load_cudss().solve(factorization, vector)

    def residual_tolerance(self, dtype: Any) -> float:
        return _RESIDUAL_TOLERANCE[jnp.dtype(dtype)]


def _load_cudss() -> ModuleType:
    """Return spineax's cuDSS module, or raise an error naming what is missing.

    Importing `spineax.cudss` initializes JAX's backends and loads the CUDA
    libraries, so it happens only here, when a `CuDSSSolver` is used.
    """
    if not CUDSS_AVAILABLE:
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


# Largest residual of a solution, relative to its right-hand side, in the
# infinity norm. A successful factorization of grounded Laplacians with six
# orders of conductance contrast and a 1e-6 floor keeps residuals of basis
# right-hand sides below 1.2e-11 in float64 and 6.6e-3 in float32, up to
# 1000 x 1000 grids. A grounded Laplacian with vertices disconnected from the
# ground vertex leaves residuals of 3e-2 to 0.16 in float64; in float32 they
# ranged from 1.5e-3 on 1000 x 1000 grids to 0.5 on 100 x 100 grids.
_RESIDUAL_TOLERANCE = {jnp.dtype(jnp.float32): 1e-1, jnp.dtype(jnp.float64): 1e-4}
