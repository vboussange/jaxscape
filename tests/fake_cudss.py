"""CPU test double of spineax 0.0.5's cuDSS phase API (`spineax.cudss`).

`CuDSSSolver` needs an NVIDIA GPU, spineax and cuDSS. This double reproduces,
on the CPU, the parts of spineax's public phase API that JAXScape relies on, so
that the solver's plumbing runs in CI:

- `analyze`, `factorize`, `refactorize` and `solve` with spineax's signatures,
  trace-time checks and error messages, and a token pytree with spineax's
  leaves and static fields;
- a process-global host registry of factorizations with LRU eviction
  (`SPINEAX_FACTOR_CACHE`, default 8), where each numeric phase uses up the
  token it is given and re-registers its entry under a fresh id;
- a stale, evicted or released token that still gives correct results by
  rebuilding the factorization, counted by `rebuild_count()`;
- the CSR structure fingerprint, checked against the analysed pattern;
- spineax's batching rules: a vmapped solve with an unbatched token is one
  multi-RHS solve, and a vmapped numeric phase with an unbatched token raises.

Factorization uses dense Cholesky on the host and is meant for small systems.
Like spineax, the double reports no factorization status. What a failed
factorization leaves in the factor is set by `failed_pivots`, because cuDSS's
behaviour is not documented for it:

- `"nan"` (the default): a matrix that is not numerically positive definite
  gives a NaN factor, so its solutions are NaN. A singular matrix whose
  pivots round to small positive values still factorizes, into finite
  solutions that do not solve it.
- `"perturbed"`: every pivot below cuDSS's default pivot epsilon (1e-13 in
  float64, 1e-5 in float32) is replaced by that epsilon, which leaves finite
  solutions that do not solve the matrix wherever the factorization stays
  finite.

Besides the spineax API, the double records what ran, for tests: `analyses`,
`factorizations` (numeric factorizations, rebuilds included) and
`solve_calls` (the number of right-hand sides of each solve call).
"""

import collections
import dataclasses
import itertools
import os
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg


_MTYPE_IDS = {
    "general": 0,
    "symmetric": 1,
    "hermitian": 2,
    "spd": 3,
    "symmetric_positive_definite": 3,
    "hpd": 4,
    "hermitian_positive_definite": 4,
}
_MVIEW_IDS = {"full": 0, "upper": 1, "lower": 2}
_REORDERING_IDS = {
    "default": 0,
    "btf_colamd": 1,
    "colamd": 2,
    "amd": 3,
    "nested_dissection": 4,
    "none": 5,
}
_MEMORY_IDS = {"device": 0, "hybrid": 1}
_DTYPES = tuple(
    jnp.dtype(dtype)
    for dtype in (jnp.float32, jnp.float64, jnp.complex64, jnp.complex128)
)


class FactorToken(eqx.Module):
    """Same leaves and static fields as `spineax.cudss.FactorToken`."""

    id: jax.Array
    values: jax.Array
    offsets: jax.Array
    columns: jax.Array
    phase: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    dtype: Any = eqx.field(static=True)
    n: int = eqx.field(static=True)
    nnz: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    mtype_id: int = eqx.field(static=True)
    mview_id: int = eqx.field(static=True)
    device_id: int = eqx.field(static=True)
    reordering_id: int = eqx.field(static=True)
    memory_id: int = eqx.field(static=True)


def _check_dtype(dtype: Any) -> None:
    if jnp.dtype(dtype) not in _DTYPES:
        raise ValueError(
            f"spineax tokens: unsupported dtype {dtype} "
            "(supported: f32, f64, c64, c128)"
        )


def _knob_id(table: dict[str, int], value: str | int, what: str) -> int:
    if not isinstance(value, str):
        return int(value)
    try:
        return table[value]
    except KeyError:
        raise ValueError(
            f"spineax tokens: unknown {what} {value!r} (one of: {', '.join(table)})"
        ) from None


def _structure_fingerprint(offsets: jax.Array, columns: jax.Array) -> jax.Array:
    # spineax's position-weighted uint32 checksum, computed outside the host
    # call as spineax computes it on the device.
    w_off = jnp.arange(offsets.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2654435761
    ) + jnp.uint32(0x9E3779B9)
    w_col = jnp.arange(columns.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2246822519
    ) + jnp.uint32(0x85EBCA6B)
    h_off = jnp.sum(
        (offsets.astype(jnp.uint32) + jnp.uint32(1)) * w_off, dtype=jnp.uint32
    )
    h_col = jnp.sum(
        (columns.astype(jnp.uint32) + jnp.uint32(1)) * w_col, dtype=jnp.uint32
    )
    return jnp.stack([h_off, h_col])


def _require_factorized(token: FactorToken, op: str) -> None:
    if token.phase != "factorized":
        raise ValueError(
            f"spineax tokens: {op} requires a factorized token (call factorize first)"
        )


def _id_struct() -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct((1,), jnp.int32)


class FakeCuDSS:
    """Drop-in replacement for the `spineax.cudss` module on the CPU."""

    FactorToken = FactorToken

    def __init__(self) -> None:
        self.reset()

    # ---------------------------------------------------------------- spies
    def reset(self) -> None:
        """Empty the registry, zero every counter and restore NaN factors for
        failed factorizations."""
        self._entries: collections.OrderedDict[
            int, dict[str, Any]
        ] = collections.OrderedDict()
        self._ids = itertools.count(1)
        self._rebuilds = 0
        self.analyses = 0
        self.factorizations = 0
        self.solve_calls: list[int] = []
        self.failed_pivots = "nan"

    # --------------------------------------------------------- public API
    def analyze(
        self,
        csr_values: jax.Array,
        csr_offsets: jax.Array,
        csr_columns: jax.Array,
        *,
        mtype_id: str | int = "symmetric",
        mview_id: str | int = "upper",
        device_id: int = 0,
        reordering: str | int = "default",
        memory: str | int = "device",
    ) -> FactorToken:
        _check_dtype(csr_values.dtype)
        if csr_values.ndim != 1:
            raise NotImplementedError(
                "The CPU double models single systems only (1-D values)."
            )
        mtype = _knob_id(_MTYPE_IDS, mtype_id, "mtype")
        mview = _knob_id(_MVIEW_IDS, mview_id, "mview")
        offsets = csr_offsets.astype(jnp.int32)
        columns = csr_columns.astype(jnp.int32)
        token_id = jax.pure_callback(
            self._host_analyze,
            _id_struct(),
            _structure_fingerprint(offsets, columns),
        )
        return FactorToken(
            id=token_id,
            values=csr_values,
            offsets=offsets,
            columns=columns,
            phase="analyzed",
            kind="single",
            dtype=jnp.dtype(csr_values.dtype),
            n=int(offsets.shape[-1] - 1),
            nnz=int(columns.shape[-1]),
            batch_size=1,
            mtype_id=mtype,
            mview_id=mview,
            device_id=int(device_id),
            reordering_id=_knob_id(_REORDERING_IDS, reordering, "reordering"),
            memory_id=_knob_id(_MEMORY_IDS, memory, "memory"),
        )

    def factorize(self, token: FactorToken, csr_values: jax.Array) -> FactorToken:
        return self._numeric(token, csr_values, refactor=False)

    def refactorize(self, token: FactorToken, csr_values: jax.Array) -> FactorToken:
        return self._numeric(token, csr_values, refactor=True)

    def solve(
        self, token: FactorToken, b: jax.Array, ir_nsteps: int | None = None
    ) -> jax.Array:
        _require_factorized(token, "solve")
        if jnp.dtype(b.dtype) != token.dtype:
            raise ValueError(
                f"spineax tokens: rhs dtype {b.dtype} != token dtype {token.dtype}"
            )
        if b.shape[-1] != token.n:
            raise ValueError(
                f"spineax tokens: rhs trailing dim {b.shape[-1]} != token n {token.n}"
            )
        if ir_nsteps:
            raise NotImplementedError("The CPU double has no iterative refinement.")
        token_ng = jax.lax.stop_gradient(token)

        def matvec(x):
            rows = jnp.repeat(
                jnp.arange(token.n, dtype=jnp.int32),
                jnp.diff(token.offsets),
                total_repeat_length=token.nnz,
            )
            return (
                jnp.zeros_like(x)
                .at[..., rows]
                .add(token.values * jnp.take(x, token.columns, axis=-1))
            )

        def solve_fn(_matvec, rhs):
            return self._solve_id(
                token_ng.id,
                token_ng.offsets,
                token_ng.columns,
                token_ng.values,
                rhs,
            )

        # spineax wraps symmetric and SPD solves the same way.
        return jax.lax.custom_linear_solve(
            matvec, b, solve_fn, solve_fn, symmetric=True
        )

    def release(self, token: FactorToken) -> bool:
        token_id = int(np.asarray(jax.device_get(token.id)).ravel()[0])
        return self._entries.pop(token_id, None) is not None

    def registry_size(self) -> int:
        return len(self._entries)

    def cache_capacity(self) -> int:
        return int(os.environ.get("SPINEAX_FACTOR_CACHE", 8))

    def rebuild_count(self) -> int:
        return self._rebuilds

    # ------------------------------------------------------- traced phases
    def _numeric(
        self, token: FactorToken, csr_values: jax.Array, *, refactor: bool
    ) -> FactorToken:
        op = "refactorize" if refactor else "factorize"
        if refactor:
            _require_factorized(token, op)
        if jnp.dtype(csr_values.dtype) != token.dtype:
            raise ValueError(
                f"spineax tokens: {op} values dtype {csr_values.dtype} != "
                f"token dtype {token.dtype}"
            )
        if csr_values.shape[-1] != token.nnz:
            raise ValueError(
                f"spineax tokens: {op} values size {csr_values.shape[-1]} != "
                f"token nnz {token.nnz}"
            )
        token_id = self._numeric_id(
            token.id, token.offsets, token.columns, csr_values, op
        )
        return dataclasses.replace(
            token, id=token_id, values=csr_values, phase="factorized"
        )

    def _numeric_id(self, token_id, offsets, columns, values, op):
        @jax.custom_batching.custom_vmap
        def numeric_id(token_id, offsets, columns, values):
            return jax.pure_callback(
                self._host_numeric,
                _id_struct(),
                token_id,
                offsets,
                columns,
                _structure_fingerprint(offsets, columns),
                values,
            )

        @numeric_id.def_vmap
        def _(axis_size, in_batched, token_id, offsets, columns, values):
            if not in_batched[0]:
                raise ValueError(
                    f"spineax tokens: vmap({op}) with an unbatched token and "
                    "batched values — one entry cannot hold a batch of "
                    "factorizations. vmap(analyze) over the batch first."
                )
            raise NotImplementedError("The CPU double models single systems only.")

        return numeric_id(token_id, offsets, columns, values)

    def _solve_id(self, token_id, offsets, columns, values, b):
        @jax.custom_batching.custom_vmap
        def solve_id(token_id, offsets, columns, values, b):
            # Leading axes of `b` are right-hand sides of one solve call.
            return jax.pure_callback(
                self._host_solve,
                jax.ShapeDtypeStruct(b.shape, b.dtype),
                token_id,
                offsets,
                columns,
                _structure_fingerprint(offsets, columns),
                values,
                b,
            )

        @solve_id.def_vmap
        def _(axis_size, in_batched, token_id, offsets, columns, values, b):
            token_batched = any(in_batched[:4])
            if token_batched:
                raise NotImplementedError("The CPU double models single systems only.")
            if not in_batched[4]:
                b = jnp.broadcast_to(b, (axis_size,) + b.shape)
            # An unbatched token: the batch axis is more right-hand sides.
            return solve_id(token_id, offsets, columns, values, b), True

        return solve_id(token_id, offsets, columns, values, b)

    # ------------------------------------------------------- host registry
    def _insert(self, entry: dict[str, Any], token_id: int | None = None) -> int:
        while len(self._entries) >= self.cache_capacity():
            self._entries.popitem(last=False)
        if token_id is None:
            token_id = next(self._ids)
        self._entries[token_id] = entry
        return token_id

    def _resident(self, token_id: int, fingerprint: np.ndarray) -> dict | None:
        entry = self._entries.get(token_id)
        if entry is None:
            return None
        self._entries.move_to_end(token_id)
        if not np.array_equal(entry["fingerprint"], fingerprint):
            raise RuntimeError(
                "spineax token: CSR structure fingerprint mismatch — the "
                "offsets/columns behind this token differ from the pattern "
                "that was analyzed"
            )
        return entry

    def _factor(self, values, offsets, columns, fingerprint) -> dict[str, Any]:
        self.factorizations += 1
        n = offsets.shape[0] - 1
        matrix = np.zeros((n, n), dtype=values.dtype)
        rows = np.repeat(np.arange(n), np.diff(offsets))
        matrix[rows, columns] = values
        if self.failed_pivots == "perturbed":
            factor = _cholesky_perturbing_pivots(matrix)
        else:
            try:
                factor = np.linalg.cholesky(matrix)
            except np.linalg.LinAlgError:
                factor = np.full((n, n), np.nan, dtype=values.dtype)
        return {"factor": factor, "fingerprint": np.array(fingerprint)}

    def _host_analyze(self, fingerprint):
        self.analyses += 1
        entry = {"factor": None, "fingerprint": np.array(fingerprint)}
        return np.asarray([self._insert(entry)], dtype=np.int32)

    def _host_numeric(self, token_id, offsets, columns, fingerprint, values):
        token_id = int(token_id[0])
        if self._resident(token_id, fingerprint) is None:
            # Self-heal: analysis and factorization of the call's own values.
            self._rebuilds += 1
            self.analyses += 1
            entry = self._factor(values, offsets, columns, fingerprint)
            return np.asarray([self._insert(entry)], dtype=np.int32)
        entry = self._factor(values, offsets, columns, fingerprint)
        # The numeric phase uses up the input id: re-register under a new one.
        del self._entries[token_id]
        return np.asarray([self._insert(entry)], dtype=np.int32)

    def _host_solve(self, token_id, offsets, columns, fingerprint, values, b):
        token_id = int(token_id[0])
        entry = self._resident(token_id, fingerprint)
        if entry is None:
            # Self-heal from the token's own values, under the caller's id.
            self._rebuilds += 1
            self.analyses += 1
            entry = self._factor(values, offsets, columns, fingerprint)
            self._insert(entry, token_id)
        n = offsets.shape[0] - 1
        rhs = np.asarray(b).reshape(-1, n)
        self.solve_calls.append(rhs.shape[0])
        factor = entry["factor"]
        if factor is None:
            raise RuntimeError("spineax token: solve on an analysed-only entry")
        solution = scipy.linalg.cho_solve((factor, True), rhs.T, check_finite=False)
        return solution.T.reshape(b.shape).astype(b.dtype)


_PIVOT_EPSILON = {np.dtype(np.float32): 1e-5, np.dtype(np.float64): 1e-13}


def _cholesky_perturbing_pivots(matrix: np.ndarray) -> np.ndarray:
    """Lower Cholesky factor, with each pivot below cuDSS's default pivot
    epsilon replaced by the epsilon."""
    epsilon = _PIVOT_EPSILON[matrix.dtype]
    factor = np.zeros_like(matrix)
    with np.errstate(all="ignore"):
        for k in range(matrix.shape[0]):
            pivot = matrix[k, k] - factor[k, :k] @ factor[k, :k]
            factor[k, k] = np.sqrt(pivot if pivot > epsilon else epsilon)
            factor[k + 1 :, k] = (
                matrix[k + 1 :, k] - factor[k + 1 :, :k] @ factor[k, :k]
            ) / factor[k, k]
    return factor


FAKE_CUDSS = FakeCuDSS()
"""The process-global double, like spineax's process-global registry.

Compiled functions keep references to the host calls of the double that traced
them, so tests share this instance and call `reset()` between tests.
"""
