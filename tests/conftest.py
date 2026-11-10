"""Core pinning of pytest-xdist workers, precision configuration and a reference
sparse direct solver for the test suite."""

import os


# A JAX process sizes its CPU thread pools to the cores it may run on: on
# Linux, each pytest-xdist worker runs on one core, set before JAX starts, so
# that `pytest -n auto` does not oversubscribe the machine. The threads that
# NumPy's BLAS started on import, before this file, take that core too.
_worker = os.environ.get("PYTEST_XDIST_WORKER")
if _worker and hasattr(os, "sched_setaffinity"):
    _cores = sorted(os.sched_getaffinity(0))
    _core = {_cores[int(_worker.removeprefix("gw")) % len(_cores)]}
    for _thread in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(_thread), _core)

from types import SimpleNamespace
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jax import Array
from jaxscape.solvers import AbstractSparseDirectSolver, CSRPattern, cudsssolver


# Native callbacks execute on worker threads, where a thread-local enable_x64
# context is insufficient. Float32 coverage creates explicit float32 inputs.
jax.config.update("jax_enable_x64", True)


def _memory_mappings() -> int:
    """The number of memory mappings of this process, or 0 where Linux's
    `/proc` file system is missing."""
    try:
        with open("/proc/self/maps") as maps:
            return sum(1 for _ in maps)
    except OSError:
        return 0


@pytest.fixture(autouse=True)
def _release_compiled_programs():
    """Release the compiled programs after a test once the process holds many.

    Every XLA:CPU executable keeps memory mappings for its code. A process that
    held the programs of the whole suite reached Linux's default limit of 65530
    mappings (`vm.max_map_count`), and XLA then crashed while compiling.
    Releasing them after every test slows the suite down: later tests
    recompile the programs they share.
    """
    yield
    if _memory_mappings() > 30_000:
        jax.clear_caches()


class DenseFactorization(eqx.Module):
    factor: Array
    refactorizations: Array
    poisoned: Array


def dense_cholesky(n: int, rows: Array, columns: Array, values: Array) -> Array:
    dense = jnp.zeros((n, n), values.dtype).at[rows, columns].set(values)
    return jnp.linalg.cholesky(dense)


@final
class DenseCholesky(AbstractSparseDirectSolver[DenseFactorization]):
    """Sparse direct solver in pure JAX, which fails as cuDSS does on a GPU.

    A factorization that meets a non-positive pivot reports nothing, and leaves
    solutions that do not solve the matrix: finite garbage, or NaN with
    `nan_pivots`. It also poisons its handle: every later refactorization gives the
    same garbage, until a fresh analysis. `refactorizations` counts refactorizations
    since the last analysis. With `poison_at`, that refactorization is valid and
    poisons its handle.
    """

    nan_pivots: bool = False
    poison_at: int | None = None

    def factorize(self, pattern: CSRPattern, values: Array) -> DenseFactorization:
        return self._factorized(pattern, values, jnp.array(0), jnp.array(False))

    def refactorize(
        self, factorization: DenseFactorization, pattern: CSRPattern, values: Array
    ) -> DenseFactorization:
        count = factorization.refactorizations + 1
        return self._factorized(pattern, values, count, factorization.poisoned)

    def _factorized(
        self, pattern: CSRPattern, values: Array, count: Array, poisoned: Array
    ) -> DenseFactorization:
        factor = dense_cholesky(pattern.n, pattern.rows, pattern.columns, values)
        poisoned = poisoned | ~jnp.all(jnp.isfinite(factor))
        fill = jnp.nan if self.nan_pivots else 0.0
        garbage = jnp.where(jnp.eye(pattern.n, dtype=bool), 1.0, fill)
        factor = jnp.where(poisoned, garbage.astype(values.dtype), factor)
        if self.poison_at is not None:
            poisoned = poisoned | (count == self.poison_at)
        return DenseFactorization(factor, count, poisoned)

    def solve(self, factorization: DenseFactorization, vector: Array) -> Array:
        return jax.scipy.linalg.cho_solve((factorization.factor, True), vector)

    def residual_tolerance(self, dtype: Any) -> float:
        return 1e-4 if dtype == jnp.float64 else 1e-1


class _StubToken(eqx.Module):
    factor: Array
    offsets: Array
    columns: Array


@pytest.fixture
def stub_spineax(monkeypatch):
    """Run `CuDSSSolver` on dense Cholesky, for what does not need a GPU."""

    def factorize(token, values):
        n = token.offsets.shape[0] - 1
        rows = jnp.repeat(
            jnp.arange(n), jnp.diff(token.offsets), total_repeat_length=values.shape[0]
        )
        factor = dense_cholesky(n, rows, token.columns, values)
        return _StubToken(factor, token.offsets, token.columns)

    def analyze(values, offsets, columns, *, mtype_id, mview_id):
        assert (mtype_id, mview_id) == ("spd", "full")
        n = offsets.shape[0] - 1
        return _StubToken(jnp.zeros((n, n), values.dtype), offsets, columns)

    stub = SimpleNamespace(
        analyze=analyze,
        factorize=factorize,
        refactorize=factorize,
        solve=lambda token, b: jax.scipy.linalg.cho_solve((token.factor, True), b),
        rebuild_count=lambda: 0,
    )
    monkeypatch.setattr(cudsssolver, "_load_cudss", lambda: stub)
    return stub
