"""Utilities for linear solves."""

import jax
import equinox as eqx
import lineax as lx
from lineax import AbstractLinearSolver, AbstractLinearOperator, RESULTS
import numpy as np
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

class BCOOLinearOperator(lx.MatrixLinearOperator):
    def __check_init__(self):
        if not isinstance(self.matrix, BCOO):
            raise ValueError("The operator must be a BCOO matrix.")

    def mv(self, vector):
        return self.matrix @ vector

@lx.is_positive_semidefinite.register(BCOOLinearOperator)
def _(op):
    return True

_PyAMGSolverState: TypeAlias = tuple[BCOO, PackedStructures]


class PyAMGSolver(AbstractLinearSolver):
    """
    A linear solver that uses PyAMG to solve a sparse linear system.
    """
    tol: float
    method: Callable = pyamg.ruge_stuben_solver
    accel: str = "gmres" #TODO: use

    def __check_init__(self):
        if isinstance(self.tol, (int, float)) and self.tol < 0:
            raise ValueError("Tolerances must be non-negative.")

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> _PyAMGSolverState:
        del options
        
        A_bcoo = operator.as_matrix()
        packed_structures = pack_structures(operator)
        return A_bcoo, packed_structures

    def _compute_host(self, A_bcoo, b_jax, tol):
        if b_jax.ndim == 1:
            A_bcsr = BCSR.from_bcoo(A_bcoo)
            A_scipy = zero_copy_jax_csr_to_scipy_csr(A_bcsr)
            ml = self.method(A_scipy)  # construct the multigrid hierarchy
            b_scipy = np.from_dlpack(b_jax)
            x_scipy = ml.solve(b_scipy, tol=tol, accel=self.accel)  # solve Ax=b
            x_jax = jnp.asarray(x_scipy, copy=False)
        elif b_jax.ndim == 2:
            A_bcsr = BCSR.from_bcoo(BCOO((A_bcoo.data[0, :], A_bcoo.indices[0, :, :]), shape=A_bcoo.shape)) # need to unbatch the BCOO matrix
            A_scipy = zero_copy_jax_csr_to_scipy_csr(A_bcsr)
            ml = pyamg.ruge_stuben_solver(A_scipy) 
            b_scipy = np.from_dlpack(b_jax).transpose()
            x_np = np.zeros(b_scipy.shape, dtype=b_scipy.dtype)
            for i in range(b_scipy.shape[1]):
                _b = b_scipy[:, i]                                       # extract the i-th right-hand side
                _x = ml.solve(_b, tol=tol, accel=self.accel)
                x_np[:, i] = _x
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
            self.tol,
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
    
def linear_solve(A, b, solver):
    operator = BCOOLinearOperator(A)
    return lx.linear_solve(operator, b, solver=solver).value # works

def batched_linear_solve(A, B, solver):
    return jax.vmap(lambda A, b: linear_solve(A, b, solver), in_axes=(None, 1), out_axes=1)(A, B)
