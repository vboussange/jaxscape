"""Precision configuration and a reference sparse direct solver for the test suite."""

from types import SimpleNamespace
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.experimental import io_callback
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
    Releasing them after every test took 22 minutes instead of 13.
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
    id: Array


class _StubFactorizations:
    """The factorizations of `stub_spineax`, which it keeps as spineax does.

    An analysis mints an id. A factorization or refactorization uses up the id
    of its input and mints a new one. A phase on an id that is no longer kept
    rebuilds its factorization, which `rebuilds` counts, and keeps the id again.
    Compiled programs call it from every test, so tests compare counts.
    """

    def __init__(self) -> None:
        self.kept: set[int] = set()
        self.last = 0
        self.analyses = 0
        self.rebuilds = 0

    def analyse(self) -> np.ndarray:
        self.analyses += 1
        return self._mint()

    def use_up(self, id: np.ndarray) -> np.ndarray:
        self.use(id)
        self.kept.remove(int(id))
        return self._mint()

    def use(self, id: np.ndarray) -> None:
        if int(id) not in self.kept:
            self.rebuilds += 1
            self.kept.add(int(id))

    def _mint(self) -> np.ndarray:
        self.last += 1
        self.kept.add(self.last)
        return np.array(self.last, np.int32)


STUB_FACTORIZATIONS = _StubFactorizations()
_ID = jax.ShapeDtypeStruct((), jnp.int32)


@pytest.fixture
def stub_spineax(monkeypatch):
    """Run `CuDSSSolver` on dense Cholesky, for what does not need a GPU.

    Its factorizations are kept and used up as spineax keeps them, and its
    `analyses` and `rebuild_count` count the analyses and rebuilds that ran.
    """
    kept = STUB_FACTORIZATIONS

    def factorize(token, values):
        n = token.offsets.shape[0] - 1
        rows = jnp.repeat(
            jnp.arange(n), jnp.diff(token.offsets), total_repeat_length=values.shape[0]
        )
        factor = dense_cholesky(n, rows, token.columns, values)
        id = io_callback(kept.use_up, _ID, token.id)
        return _StubToken(factor, token.offsets, token.columns, id)

    def analyze(values, offsets, columns, *, mtype_id, mview_id):
        assert (mtype_id, mview_id) == ("spd", "full")
        n = offsets.shape[0] - 1
        id = io_callback(kept.analyse, _ID)
        return _StubToken(jnp.zeros((n, n), values.dtype), offsets, columns, id)

    def solve(token, b):
        jax.debug.callback(kept.use, token.id)
        return jax.scipy.linalg.cho_solve((token.factor, True), b)

    stub = SimpleNamespace(
        analyze=analyze,
        factorize=factorize,
        refactorize=factorize,
        solve=solve,
        analyses=lambda: kept.analyses,
        rebuild_count=lambda: kept.rebuilds,
    )
    monkeypatch.setattr(cudsssolver, "_load_cudss", lambda: stub)
    return stub
