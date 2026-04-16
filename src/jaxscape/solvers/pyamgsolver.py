from collections.abc import Callable
from typing import Any, Optional, TypeAlias, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, PyTree
from lineax import AbstractLinearOperator, AbstractLinearSolver, RESULTS
from lineax._solver.misc import (
    pack_structures,
    PackedStructures,
    transpose_packed_structures,
    unravel_solution,
)
from scipy.sparse.linalg import cg as scipy_cg

from jaxscape.utils import zero_copy_jax_csr_to_scipy_csr

from ._callback import (
    collapse_broadcasted_bcoo,
    flatten_callback_rhs,
    restore_callback_rhs,
)


try:
    import pyamg  # pyright: ignore[reportMissingImports]

    PYAMG_AVAILABLE = True
except ImportError:
    pyamg = None
    PYAMG_AVAILABLE = False

_PyAMGSolverState: TypeAlias = tuple[BCOO, PackedStructures]


class PyAMGSolver(AbstractLinearSolver):
    """
    A linear solver that uses PyAMG to solve a sparse linear system.

    !!! example
        ```python
        from jaxscape.solvers import PyAMGSolver

        solver = PyAMGSolver(rtol=1e-6, maxiter=100_000)
        distance = ResistanceDistance(solver=solver)
        dist = distance(grid)

        # Custom AMG method
        import pyamg
        solver = PyAMGSolver(pyamg_method=pyamg.ruge_stuben_solver)
        ```

    !!! warning
        PyAMG must be installed to use this solver.
    """

    rtol: float = 1e-6
    maxiter: int = 100_000
    pyamg_method: Optional[Callable] = None

    def __check_init__(self):
        if not PYAMG_AVAILABLE:
            raise ImportError(
                "pyamg is required for PyAMGSolver. "
                "Install it with: pip install pyamg"
            )

        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")

        if self.pyamg_method is None:
            assert pyamg is not None
            object.__setattr__(self, "pyamg_method", pyamg.smoothed_aggregation_solver)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _PyAMGSolverState:
        del options

        A_bcoo = cast(BCOO, operator.as_matrix())
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _compute_host(
        self, A_bcoo: BCOO, b_jax: Array, rtol: float, maxiter: int
    ) -> Array:
        callback_batched = A_bcoo.n_batch > 0
        A_bcoo = collapse_broadcasted_bcoo(A_bcoo)

        A_bcsr = BCSR.from_bcoo(A_bcoo)
        A_scipy = cast(Any, zero_copy_jax_csr_to_scipy_csr(A_bcsr))
        assert self.pyamg_method is not None
        ml = self.pyamg_method(A_scipy)
        M = ml.aspreconditioner()

        rhs_size = A_scipy.shape[0]
        b_flat_batch, rhs_shape = flatten_callback_rhs(
            b_jax,
            rhs_size,
            callback_batched,
        )
        b_host = np.asarray(b_flat_batch)

        x_host = np.zeros_like(b_host)
        for i in range(b_host.shape[1]):
            column = b_host[:, i]
            x_host[:, i], info = scipy_cg(
                A_scipy,
                column,
                rtol=rtol,
                maxiter=int(maxiter),
                M=M,
            )
            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")

        x = jnp.asarray(x_host, dtype=b_jax.dtype)
        return restore_callback_rhs(x, rhs_shape, callback_batched)

    def compute(
        self,
        state: _PyAMGSolverState,
        vector: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        del options

        A_bcoo, packed_structures = state
        result_shape = jax.ShapeDtypeStruct(vector.shape, vector.dtype)

        def solve_host(current_A: BCOO, current_b: Array) -> Array:
            return self._compute_host(current_A, current_b, self.rtol, self.maxiter)

        solution = eqx.filter_pure_callback(
            solve_host,
            A_bcoo,
            vector,
            result_shape_dtypes=result_shape,
            vmap_method="broadcast_all",
        )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _PyAMGSolverState, options: dict[str, Any]):
        del options

        A_bcoo, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        A_bcoo_T = A_bcoo.T
        transpose_state = (A_bcoo_T, transposed_packed_structures)
        return transpose_state, {}

    def conj(self, state: _PyAMGSolverState, options: dict[str, Any]):
        del options

        A_bcoo, packed_structures = state
        A_conj = BCOO(
            (jnp.conj(A_bcoo.data), A_bcoo.indices),
            shape=A_bcoo.shape,
            indices_sorted=A_bcoo.indices_sorted,
            unique_indices=A_bcoo.unique_indices,
        )
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def assume_full_rank(self):
        return True
