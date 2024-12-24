"""
This file benchmarks the performance of the lineax GMRES solver against the pyamg solver and the jax GMRES solver for batch mode.
"""

import pyamg
import numpy as np

from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import jax
import lineax as lx

import timeit

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
    
def pyamg_solve(A, B):
    ml = pyamg.ruge_stuben_solver(A)           # construct the multigrid hierarchy
    
    X = np.empty_like(B)
    for j in range(X.shape[1]):
        X[:, j] = ml.solve(B[:, j], tol=1e-10)

    error = np.linalg.norm(B - A @ X)          # compute norm of residual vector
    return X, error

@jax.jit
def cg_solve(A, B):    
    # def single_solve(b):
    X, _ = jax.scipy.sparse.linalg.cg(A,
                                    B,
                                    tol=1e-10,
                                    atol=1e-10,
                                    maxiter=10000)
    #     return x
    # X = jax.vmap(single_solve, in_axes=1, out_axes=1)(B)
    error = jnp.linalg.norm(B - (A @ X))

    return X, error

@jax.jit
def lineax_gmres_solve(A, B):
    # TODO: this fails
    in_structure = jax.eval_shape(lambda: B[:, 0])

    jacobi = get_diagonal(A)
    pc = lambda x: x / jacobi
    preconditioner = lx.FunctionLinearOperator(pc, in_structure, tags=[lx.positive_semidefinite_tag])
    
    op = lambda x: A @ x
    operator = lx.FunctionLinearOperator(op, in_structure)
    
    solver = lx.GMRES(atol=1e-5, 
                      rtol=1e-5,
                      max_steps=maxiter,
                      restart=50)
    
    def solve_single(b):
        x = lx.linear_solve(operator, 
                            b, 
                            solver=solver, 
                            throw=False, 
                            options={"preconditioner":preconditioner}).value
        return x
    
    X = jax.vmap(solve_single, in_axes=1, out_axes=1)(B)
    error = jnp.linalg.norm(B - (A @ X))

    return X, error

@jax.jit
def lineax_cg_solve(A, B):
    in_structure = jax.eval_shape(lambda: B[:, 0])

    jacobi = get_diagonal(A)
    pc = lambda x: x / jacobi
    preconditioner = lx.FunctionLinearOperator(pc, in_structure, tags=[lx.positive_semidefinite_tag])
    
    op = lambda x: A @ x
    operator = lx.FunctionLinearOperator(op, in_structure)
    
    solver = lx.CG(atol=1e-5, 
                      rtol=1e-5,
                      max_steps=maxiter)
    X = lx.linear_solve(operator, 
                        B, 
                        solver=solver, 
                        throw=False, 
                        options={"preconditioner":preconditioner}).value
    
    error = jnp.linalg.norm(B - (A @ X))

    return X, error

@jax.jit
def lineax_auto_solve(A, B):    
    operator = lx.MatrixLinearOperator(A.todense())
    
    def solve_single(b):
        x = lx.linear_solve(operator, 
                            b, 
                            throw=False).value
        return x
    
    X = jax.vmap(solve_single, in_axes=1, out_axes=1)(B)
    error = jnp.linalg.norm(B - (A @ X))

    return X, error

if __name__ == "__main__":

    A = pyamg.gallery.poisson((100, 100), format='csr')  # 2D Poisson problem on 500x500 grid
    L = 10 # landmarks
    B = np.random.rand(A.shape[0], L)                   # pick a random right hand side

    A_jax = BCOO.from_scipy_sparse(A)
    B_jax = jnp.array(B)
    assert jnp.allclose(A.diagonal(), get_diagonal(A_jax))

    maxiter = 100

    benchmark("pyamg_solve", lambda: jax.block_until_ready(pyamg_solve(A, B)))
    benchmark("cg_solve", lambda: jax.block_until_ready(cg_solve(A_jax, B_jax)))
    benchmark("lineax_gmres_solve", lambda: jax.block_until_ready(lineax_gmres_solve(A_jax, B_jax))) #TODO: to fix, fails
    benchmark("lineax_cg_solve", lambda: jax.block_until_ready(lineax_cg_solve(A_jax, B_jax))) 
    benchmark("lineax_auto_solve", lambda: jax.block_until_ready(lineax_auto_solve(A_jax, B_jax)))