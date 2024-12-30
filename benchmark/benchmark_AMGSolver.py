"""
Benchmarking the AMG solver against the other lx solvers.
TODO: work in progress
"""

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.scipy.sparse.linalg import cg as jcg
import numpy as np
import pyamg
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.gallery import poisson
import lineax as lx
# Enforce CPU usage
jax.config.update("jax_platform_name", "cpu")
from jax.experimental.sparse import BCOO, BCSR, sparsify
import timeit
import equinox as eqx
from jaxscape.utils import BCOO_to_csr

@jax.jit
def get_diagonal(matrix):
    is_diag = matrix.indices[:, 0] == matrix.indices[:, 1]
    diag_values = jnp.where(is_diag, matrix.data, 0)
    diag = jnp.zeros(matrix.shape[0], dtype=matrix.data.dtype)
    diag = diag.at[matrix.indices[:, 0]].add(diag_values)
    return diag

def benchmark(method, func):
    times = timeit.repeat(func, number=1, repeat=10)
    _, error = func()
    print(f"{method} solve error: {error:2e}")
    print(f"{method} min time: {min(times)}\n")
    
def pyamg_solve(A, b):
    ml = pyamg.ruge_stuben_solver(A)           # construct the multigrid hierarchy
    x = ml.solve(b, tol=1e-10)
    return x

def _pyamg_solve(A, b)

class SparseMatrixLinearOperator(lx.MatrixLinearOperator):
    matrix: jax.Array
    def __init__(self, matrix):
        # TODO: we should check that the matrix is indeed positive_semidefinite
        self.matrix = matrix
    def mv(self, vector):
        return self.matrix @ vector
    

class AMGPreconditioner(lx.MatrixLinearOperator):
    def __init__(self, A):
        A_bcsr = BCSR.from_bcoo(A)
        M = jax.pure_callback(preconditionero)
        super().__init__(lambda x: jax.pure_callback(M, out_spec, x), 
                         out_spec, 
                         tags=[lx.positive_semidefinite_tag])

@jax.jit
def pyamg_cg_solve(A, B):
    preconditioner = AMGPreconditioner(A)
    operator = SparseMatrixLinearOperator(A)
    
    solver = lx.CG(atol=1e-5, 
                      rtol=1e-5,
                      max_steps=30)
    
    X = eqx.filter_vmap(lx.linear_solve, 
                        in_axes = (None, 0, None, None, None))(operator, 
                        B, 
                        solver=solver, 
                        throw=False, 
                        options={"preconditioner":preconditioner}).value
    error = jnp.linalg.norm(B - (A @ X))
    return X, error

if __name__ == "__main__":
    
    # Original setup
    A = poisson((20, 20), format='csr')  # matrix
    L = 10 # landmarks
    B = np.random.rand(A.shape[0], L)                   # pick a random right hand side

    A_jax = BCOO.from_scipy_sparse(A)
    B_jax = jnp.array(B)

    benchmark("pyamg_solve", lambda: jax.block_until_ready(pyamg_solve(A, B)))
    benchmark("cg_solve", lambda: jax.block_until_ready(pyamg_cg_solve(A_jax, B_jax)))
