import jax
import equinox as eqx
import lineax as lx
from lineax import AbstractLinearSolver, AbstractLinearOperator, RESULTS
import numpy as np
import jax.numpy as jnp
from typing import Any
from typing_extensions import TypeAlias
from jaxtyping import Array, PyTree
from jax.experimental.sparse import BCSR, BCOO
from lineax._solver.misc import (
    pack_structures,
    PackedStructures,
    transpose_packed_structures,
    unravel_solution,
)
from jaxscape.utils import zero_copy_jax_csr_to_scipy_csr

try:
    from cholespy import CholeskySolverD, CholeskySolverF, MatrixType
    CHOLESPY_AVAILABLE = True
except ImportError:
    CHOLESPY_AVAILABLE = False

_CholmodSolverState: TypeAlias = tuple[BCOO, PackedStructures]


class CholmodSolver(AbstractLinearSolver):
    """
    A linear solver that uses CHOLMOD (via cholespy) to solve a sparse linear system.
    Uses direct Cholesky factorization for symmetric positive definite matrices.
    """
    use_float32: bool = False

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
        
        A_bcoo = operator.as_matrix()
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _compute_host(self, A_bcoo, b_jax):
        """
        Solve the linear system using CHOLMOD via cholespy.
        
        Args:
            A_bcoo: Sparse matrix in BCOO format
            b_jax: Right-hand side vector(s)
            
        Returns:
            Solution vector(s)
        """
        # Convert BCOO to CSR format for cholespy
        if b_jax.ndim == 1:
            A_bcsr = BCSR.from_bcoo(A_bcoo)
        elif b_jax.ndim == 2:
            # Unbatch the BCOO matrix for 2D case
            A_bcsr = BCSR.from_bcoo(
                BCOO((A_bcoo.data[0, :], A_bcoo.indices[0, :, :]), shape=A_bcoo.shape)
            )
        
        # Get CSR arrays
        data_np = np.from_dlpack(A_bcsr.data)
        indices_np = np.from_dlpack(A_bcsr.indices)
        indptr_np = np.from_dlpack(A_bcsr.indptr)
        
        # Convert to appropriate dtype
        dtype = np.float32 if self.use_float32 else np.float64
        CholeskySolver = CholeskySolverF if self.use_float32 else CholeskySolverD
        
        data_np = data_np.astype(dtype)
        b_np = np.from_dlpack(b_jax).astype(dtype)
        
        n = A_bcoo.shape[0]
        
        # Initialize the Cholesky solver with CSR format
        solver = CholeskySolver(
            n,
            indptr_np,
            indices_np,
            data_np,
            MatrixType.CSR
        )
        
        if b_jax.ndim == 1:
            # Single RHS vector
            b_np = b_np.reshape(-1, 1)
            x_np = np.zeros_like(b_np)
            solver.solve(b_np, x_np)
            x_np = x_np.reshape(-1)
        elif b_jax.ndim == 2:
            # Multiple RHS vectors
            b_np = b_np.T  # cholespy expects (n, n_rhs)
            x_np = np.zeros_like(b_np)
            solver.solve(b_np, x_np)
            x_np = x_np.T
        
        x_jax = jnp.asarray(x_np, copy=False)
        return x_jax

    def compute(
        self,
        state: _CholmodSolverState,
        b_jax: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        """
        Compute the solution to the linear system.
        """
        A_bcoo, packed_structures = state
        result_shape = jax.ShapeDtypeStruct(b_jax.shape, b_jax.dtype)
        
        solution = eqx.filter_pure_callback(
            self._compute_host,
            A_bcoo,
            b_jax,
            result_shape_dtypes=result_shape,
            vmap_method="broadcast_all",
        )
        
        solution = unravel_solution(solution, packed_structures)
        return solution, RESULTS.successful, {}

    def transpose(self, state: _CholmodSolverState, options: dict[str, Any]):
        """
        Handle transpose of the operator.
        For symmetric matrices (like those from Cholesky), transpose is the same.
        """
        A_bcoo, packed_structures = state
        transposed_packed_structures = transpose_packed_structures(packed_structures)
        A_bcoo_T = A_bcoo.T
        transpose_state = (A_bcoo_T, transposed_packed_structures)
        return transpose_state, {}

    def conj(self, state: _CholmodSolverState, options: dict[str, Any]):
        """Handle conjugate of the operator."""
        A_bcoo, packed_structures = state
        A_conj = BCOO(
            (jnp.conj(A_bcoo.data), A_bcoo.indices), 
            shape=A_bcoo.shape
        )
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def allow_dependent_columns(self, operator):
        """Cholesky requires independent columns."""
        return False

    def allow_dependent_rows(self, operator):
        """Cholesky requires independent rows."""
        return False
