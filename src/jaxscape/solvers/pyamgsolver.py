from collections.abc import Callable
from typing import Any, Optional, TypeAlias

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


try:
    import pyamg

    PYAMG_AVAILABLE = True
except ImportError:
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
    maxiter: float = 100_000
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
            object.__setattr__(self, "pyamg_method", pyamg.smoothed_aggregation_solver)

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _PyAMGSolverState:
        del options

        A_bcoo = operator.as_matrix()
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _compute_host(
        self, A_bcoo: BCOO, b_jax: Array, rtol: float, maxiter: int
    ) -> Array:
        if A_bcoo.n_batch > 0:
            # ugly trick to remove batch dimension
            # specific to BCOO behavior
            A_bcoo = BCOO(
                (A_bcoo.data.squeeze(), A_bcoo.indices.squeeze()), shape=A_bcoo.shape
            )

        A_bcsr = BCSR.from_bcoo(A_bcoo)
        A_scipy = zero_copy_jax_csr_to_scipy_csr(A_bcsr)
        ml = self.pyamg_method(A_scipy)
        M = ml.aspreconditioner()

        if b_jax.ndim == 0:
            raise ValueError("Right-hand side must have at least one dimension.")
        b_scipy = np.from_dlpack(b_jax.T)

        rhs_size = b_scipy.shape[0]
        if rhs_size != A_scipy.shape[0]:
            raise ValueError(
                "The first dimension of b must match the operator dimension."
            )

        flat_batch = b_scipy.reshape((rhs_size, -1))
        x_np = np.zeros_like(flat_batch)
        for i in range(flat_batch.shape[1]):
            column = flat_batch[:, i]
            x_np[:, i], info = scipy_cg(
                A_scipy, column, rtol=rtol, maxiter=maxiter, M=M
            )
            if info != 0:
                raise RuntimeError(f"CG did not converge, info={info}")

        x_np = x_np.reshape(b_scipy.shape)
        return jnp.asarray(x_np, copy=False).T

    def compute(
        self,
        state: _PyAMGSolverState,
        b_jax: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        A_bcoo, packed_structures = state
        result_shape = jax.ShapeDtypeStruct(b_jax.shape, b_jax.dtype)
        solution = eqx.filter_pure_callback(
            self._compute_host,
            A_bcoo,
            b_jax,
            self.rtol,
            self.maxiter,
            result_shape_dtypes=result_shape,
            vmap_method="expand_dims",
        )
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _PyAMGSolverState, options: dict[str, Any]):
        A_bcoo, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        A_bcoo_T = A_bcoo.T
        transpose_state = (A_bcoo_T, transposed_packed_structures)
        return transpose_state, {}

    def conj(self, state: _PyAMGSolverState, options: dict[str, Any]):
        A_bcoo, _, packed_structures = state
        A_conj = BCOO(
            (jnp.conj(A_bcoo.data), A_bcoo.indices, A_bcoo.indptr), shape=A_bcoo.shape
        )
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False
