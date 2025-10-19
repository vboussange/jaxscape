"""Utilities for linear solves."""

import jax
import equinox as eqx
import lineax as lx
from lineax import AbstractLinearSolver, AbstractLinearOperator, RESULTS
import numpy as np
from scipy.sparse.linalg import cg as scipy_cg
import jax.numpy as jnp
from typing import Any, Callable, Optional
from typing_extensions import TypeAlias
import pyamg
from jaxtyping import Array, PyTree
from jax.experimental.sparse import BCSR, BCOO
from lineax._solver.misc import (
    pack_structures,
    PackedStructures,
    transpose_packed_structures,
    unravel_solution,
)
from jaxscape.utils import zero_copy_jax_csr_to_scipy_csr

_PyAMGSolverState: TypeAlias = tuple[BCOO, PackedStructures]

class PyAMGSolver(AbstractLinearSolver):
    """
    A linear solver that uses PyAMG to solve a sparse linear system.
    """
    rtol: float = 1e-6
    maxiter : float = 100_000
    pyamg_method: Callable = pyamg.smoothed_aggregation_solver

    def __check_init__(self):
        if isinstance(self.rtol, (int, float)) and self.rtol < 0:
            raise ValueError("Tolerances must be non-negative.")

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _PyAMGSolverState:
        del options
        
        A_bcoo = operator.as_matrix()
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _compute_host(self, A_bcoo, b_jax, rtol, maxiter):
        if b_jax.ndim == 1:
            A_bcsr = BCSR.from_bcoo(A_bcoo)
            A_scipy = zero_copy_jax_csr_to_scipy_csr(A_bcsr)
            ml = self.pyamg_method(A_scipy)  # construct the multigrid hierarchy
            M = ml.aspreconditioner()  # create preconditioner
            b_scipy = np.from_dlpack(b_jax)
            x_np, info = scipy_cg(A_scipy, b_scipy, rtol=rtol, maxiter=maxiter, M=M)  # solve Ax=b
            x_jax = jnp.asarray(x_np, copy=False)
        elif b_jax.ndim == 2:
            A_bcsr = BCSR.from_bcoo(BCOO((A_bcoo.data[0, :], A_bcoo.indices[0, :, :]), shape=A_bcoo.shape)) # need to unbatch the BCOO matrix
            A_scipy = zero_copy_jax_csr_to_scipy_csr(A_bcsr)
            ml = self.pyamg_method(A_scipy)  # construct the multigrid hierarchy
            M = ml.aspreconditioner()  # create preconditioner
            b_scipy = np.from_dlpack(b_jax).transpose()
            x_np = np.zeros(b_scipy.shape, dtype=b_scipy.dtype)
            for i in range(b_scipy.shape[1]):
                _b = b_scipy[:, i]                                       # extract the i-th right-hand side
                x_np[:, i], info = scipy_cg(A_scipy, _b, rtol=rtol, maxiter=maxiter, M=M)  # solve Ax=b
            x_jax = jnp.asarray(x_np, copy=False)
        return x_jax.transpose()

    def compute(
        self,
        state: _PyAMGSolverState,
        b_jax: PyTree[Array],
        options: dict[str, Any],
    ) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:

        A_bcoo, packed_structures = state # TODO: change
        result_shape = jax.ShapeDtypeStruct(b_jax.shape, b_jax.dtype)
        solution = eqx.filter_pure_callback(
            self._compute_host,
            A_bcoo,
            b_jax,
            self.rtol,
            self.maxiter,
            result_shape_dtypes=result_shape,
            vmap_method="broadcast_all",
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
        A_conj = BCOO((jnp.conj(A_bcoo.data), A_bcoo.indices, A_bcoo.indptr), shape=A_bcoo.shape)
        conj_state = (A_conj, packed_structures)
        return conj_state, {}

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False