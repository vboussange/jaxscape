"""Utilities for linear solves."""

import jax
import lineax as lx
import jax.numpy as jnp
from typing_extensions import TypeAlias
from lineax import AbstractLinearSolver, AbstractLinearOperator, conj, RESULTS
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.gallery import poisson
from jax import ShapeDtypeStruct
import pyamg
from typing import Any, Optional
from jaxtyping import Array, PyTree
from scipy.sparse import csr_matrix
from jax.experimental.sparse import BCSR, BCOO

class SparseMatrixLinearOperator(lx.MatrixLinearOperator):
    # def __init__(self, matrix):
    #     # TODO: we should check that the matrix is indeed positive_semidefinite
    #     # TODO: we should check that matrix is a bcoo matrix
    #     self.matrix = matrix
    def mv(self, vector):
        return self.matrix @ vector
    
@lx.is_positive_semidefinite.register(SparseMatrixLinearOperator)
def _(op):
    return True

_AMGState: TypeAlias = tuple[BCSR, ShapeDtypeStruct]

class AMGSolver(lx.AbstractLinearSolver):
    def init(self, operator: SparseMatrixLinearOperator, options: dict[str, Any]):
        # TODO: options should served to store args to ass to amg solve
        A_bcsr = BCSR.from_bcoo(operator.matrix)
        out_shape = (A_bcsr.shape[0], 1)
        out_dtype = A_bcsr.dtype
        out_spec = ShapeDtypeStruct(out_shape, out_dtype)
        return A_bcsr, out_spec
        
    def pyamg_solve(data, indices, indptr, b):
        op = csr_matrix((data, indices, indptr))
        ml = pyamg.ruge_stuben_solver(op) # construct the multigrid hierarchy
        b = ml.solve(b, tol=1e-10) # solve Ax=b to a tolerance of 1e-10 #TODO: to fix
        return b

    def compute(self, state: _AMGState, vector: PyTree[Array], options: dict[str, Any]) -> tuple[PyTree[Array], RESULTS, dict[str, Any]]:
        A_bcsr, out_spec = state
        x = jax.pure_callback(self.pyamg_solve, out_spec, (A_bcsr.data, A_bcsr.indices, A_bcsr.indptr, vector))
        # result = RESULTS.where(breakdown, RESULTS.breakdown, result)
        return x, None, None
    
    def transpose(self, state: _AMGState, options: dict[str, Any]):
        del options
        operator = state
        transpose_options = {}
        return operator.transpose(), transpose_options

    def conj(self, state: _AMGState, options: dict[str, Any]):
        del options
        operator = state
        conj_options = {}
        return conj(operator), conj_options

    def allow_dependent_columns(self, operator):
        return False

    def allow_dependent_rows(self, operator):
        return False

# Define operator and solve with GMRES
# TODO: this cannot work because we can't jit compile a function that relies on a non-jittable object
# which in this case would be `ml`
# class AMGPreconditioner(lx.FunctionLinearOperator):
#     def __init__(self, A):
#         # A = A.to_scipy_sparse()
#         ml = smoothed_aggregation_solver(A)
#         out_shape = A.shape[0]
#         out_dtype = A.dtype
#         out_spec = ShapeDtypeStruct(out_shape, out_dtype) # TODO: this fails
#         # 
#         M = ml.aspreconditioner(cycle='V')
#         super().__init__(lambda x: jax.pure_callback(M, out_spec, x), 
#                          out_spec, 
#                          tags=[lx.positive_semidefinite_tag])

# _AMGState: TypeAlias = tuple[AbstractLinearOperator, bool]

# class AMGSolver(lx.AbstractLinearSolve[_AMGState], strict=True):
#     """Abstract base class for all linear solvers."""

#     @abc.abstractmethod
#     def init(
#         self, operator: AbstractLinearOperator, options: dict[str, Any]
#     ) -> _SolverState:
#         """Do any initial computation on just the `operator`.

#         For example, an LU solver would compute the LU decomposition of the operator
#         (and this does not require knowing the vector yet).

#         It is common to need to solve the linear system `Ax=b` multiple times in
#         succession, with the same operator `A` and multiple vectors `b`. This method
#         improves efficiency by making it possible to re-use the computation performed
#         on just the operator.