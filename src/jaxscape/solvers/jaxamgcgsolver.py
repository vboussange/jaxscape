"""Optional native JAX-AMG preconditioning for Lineax CG."""

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree


@final
class _JAXAMGCGSolverState(eqx.Module):
    cg_state: PyTree[Any]
    preconditioner: lx.AbstractLinearOperator


@final
class JAXAMGCGSolver(lx.AbstractLinearSolver):
    """Lineax CG preconditioned by one native JAX-AMG V-cycle on NVIDIA GPUs.

    Requires ``jaxscape[jaxamg]``, CUDA-enabled JAX, and native CUDA/AmgX
    libraries. Only real float32/float64 sparse SPD systems are supported.
    Setup can execute inside JIT; native refresh and reuse follow JAX-AMG's
    defaults and may occur on every preconditioner application.

    For changing matrices, omit ``state`` so CG and the preconditioner bind to
    the current values on each call. A state returned by ``init`` belongs to
    that fixed matrix, unlike AMJax's reusable preconditioner-only state.
    Implicit derivatives are handled by Lineax/JAXScape, not AMG construction.

    The fixed symmetric V-cycle uses matching L1-Jacobi pre/post smoothing
    and an exact coarse solve. Custom AMG configurations and replacement
    ``options['preconditioner']`` are deliberately unsupported: arbitrary
    preconditioners need not satisfy CG's symmetry/definiteness requirements.
    Lineax CG options such as ``y0`` remain available.
    """

    rtol: float = 1e-3
    atol: float = 1e-3
    max_steps: int | None = None
    stabilise_every: int | None = 10

    def __check_init__(self):
        _load_jaxamg()
        _check_backend()
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if isinstance(self.atol, (int, float)) and self.atol < 0:
            raise ValueError("Tolerances must be non-negative.")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive or None.")

    def _cg_solver(self) -> lx.CG:
        return lx.CG(
            rtol=self.rtol,
            atol=self.atol,
            max_steps=self.max_steps,
            stabilise_every=self.stabilise_every,
        )

    def init(
        self, operator: lx.AbstractLinearOperator, options: dict[str, Any]
    ) -> _JAXAMGCGSolverState:
        _check_options(options)
        _check_backend()
        # Do not materialize an arbitrary operator: as_matrix() may allocate
        # dense storage. JAXScape's BCOOLinearOperator is matrix-backed.
        if not isinstance(operator, lx.MatrixLinearOperator) or not isinstance(
            operator.matrix, BCOO
        ):
            raise TypeError("JAXAMGCGSolver requires a BCOO-backed sparse operator.")
        matrix = operator.matrix
        if matrix.dtype not in (jnp.dtype("float32"), jnp.dtype("float64")):
            raise TypeError(
                "JAXAMGCGSolver supports only float32 and float64 matrices."
            )
        matrix = _native_csr(matrix)
        # L1-Jacobi scales by absolute row sums, keeping relaxation stable for
        # SPD matrices. Classical AMG uses R=P.T and Galerkin coarse operators;
        # paired symmetric smooths plus an exact coarse solve give an SPD cycle.
        apply = _load_jaxamg().make_preconditioner(
            matrix,
            config={
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "selector": "PMIS",
                "interpolator": "D2",
                "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 1.0},
                "presweeps": 1,
                "postsweeps": 1,
                "max_iters": 1,
                "cycle": "V",
                "coarse_solver": "DENSE_LU_SOLVER",
                "exact_coarse_solve": 1,
            },
        )

        def precondition(rhs):
            try:
                return apply(rhs)
            except (ImportError, OSError) as error:
                raise ImportError(
                    "JAXAMGCGSolver could not load the native CUDA/AmgX backend. "
                    "Install `jaxscape[jaxamg]` against your CUDA/AmgX installation "
                    "and make libamgxsh.so and CUDA libraries discoverable; see "
                    "https://jx-wang-s-group.github.io/JAX-AMG/install/."
                ) from error

        preconditioner = lx.FunctionLinearOperator(
            precondition,
            operator.in_structure(),
            tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
        )
        return _JAXAMGCGSolverState(
            self._cg_solver().init(operator, options), preconditioner
        )

    def compute(
        self,
        state: _JAXAMGCGSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], lx.RESULTS, dict[str, Any]]:
        _check_options(options)
        return self._cg_solver().compute(
            state.cg_state, vector, {**options, "preconditioner": state.preconditioner}
        )

    def transpose(self, state: _JAXAMGCGSolverState, options: dict[str, Any]):
        _check_options(options)
        cg_state, cg_options = self._cg_solver().transpose(
            state.cg_state, {"preconditioner": state.preconditioner}
        )
        return _JAXAMGCGSolverState(cg_state, cg_options["preconditioner"]), {}

    def conj(self, state: _JAXAMGCGSolverState, options: dict[str, Any]):
        # Only real matrices and their real preconditioners are supported.
        _check_options(options)
        return state, options

    def assume_full_rank(self):
        return True


def _load_jaxamg():
    try:
        import jaxamg
    except (ImportError, OSError) as error:
        raise ImportError(
            "JAXAMGCGSolver requires JAX-AMG. Install `jaxscape[jaxamg]` "
            "and its native CUDA/AmgX prerequisites; see "
            "https://jx-wang-s-group.github.io/JAX-AMG/install/."
        ) from error
    return jaxamg


def _check_backend():
    if jax.default_backend() != "gpu" or "cuda" not in (
        jax.devices()[0].client.platform_version.lower()
    ):
        raise RuntimeError(
            "JAXAMGCGSolver requires an NVIDIA GPU and CUDA-enabled JAX. "
            "Select a CUDA GPU backend; CPU fallback is not supported."
        )


def _check_options(options):
    if "preconditioner" in options:
        raise ValueError(
            "JAXAMGCGSolver supplies its own CG-compatible preconditioner."
        )


def _native_csr(matrix: BCOO) -> BCSR:
    """Canonical CSR with static capacity and no out-of-bounds padding.

    AmgX treats the data buffer length as nnz, so JAX's sentinel padding cannot
    be passed through. Fill unused capacity with distinct explicit zero entries
    in the same matrix. Among the first `capacity` possible coordinates, at
    least `capacity - valid_count` are unused. All arrays remain O(capacity),
    including under JIT; neither host sparsity discovery nor densification is
    needed. Numerical zeros are retained so changing values keep their pattern.
    """
    if matrix.ndim != 2 or matrix.n_batch or matrix.n_dense:
        raise ValueError("JAXAMGCGSolver requires an unbatched 2D sparse matrix.")
    n, m = matrix.shape
    if n != m or n == 0 or matrix.nse == 0:
        raise ValueError("JAXAMGCGSolver requires a nonempty square sparse matrix.")
    capacity = min(matrix.nse, n * n)
    matrix = matrix.sum_duplicates(nse=capacity, remove_zeros=False)
    valid = jnp.all(matrix.indices < n, axis=1)
    count = jnp.sum(valid, dtype=matrix.indices.dtype)
    candidates = jnp.arange(capacity, dtype=matrix.indices.dtype)
    # Only keys inside the candidate prefix matter. Cap rows first to avoid
    # computing n*n (which may overflow int32 on large sparse grids).
    rows = jnp.minimum(matrix.indices[:, 0], capacity // n)
    keys = jnp.where(
        matrix.indices[:, 0] > capacity // n,
        capacity,
        jnp.minimum(rows * n + matrix.indices[:, 1], capacity),
    )
    positions = jnp.searchsorted(keys, candidates)
    occupied = keys[jnp.minimum(positions, capacity - 1)] == candidates
    unused = jnp.nonzero(~occupied, size=capacity, fill_value=0)[0]
    replacement = unused[jnp.maximum(candidates - count, 0)]
    fill_indices = jnp.stack((replacement // n, replacement % n), axis=1)
    indices = jnp.where(valid[:, None], matrix.indices, fill_indices)
    data = jnp.where(valid, matrix.data, 0)
    canonical = BCOO((data, indices), shape=matrix.shape, unique_indices=True)
    return BCSR.from_bcoo(canonical.sort_indices())
