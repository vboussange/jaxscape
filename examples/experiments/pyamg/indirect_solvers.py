"""Compares performance of indirect solvers, see also batch_mode"""

import pyamg
import numpy as np
import jax.experimental.sparse as js
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
    
def pyamg_solve(A, b):
    ml = pyamg.ruge_stuben_solver(A)           # construct the multigrid hierarchy
    x = ml.solve(b, tol=1e-10)                 # solve Ax=b to a tolerance of 1e-10
    error = np.linalg.norm(b - A @ x)          # compute norm of residual vector
    return x, error

@jax.jit
def bigcstab_solve(A, b):
    jacobi = get_diagonal(A)
    pc = lambda x: x * (1. / jacobi)
    x, info = jax.scipy.sparse.linalg.bicgstab(A,
                                                b,
                                                M=pc,
                                                tol=1e-10,
                                                atol=1e-10)

    error = jnp.linalg.norm(b - (A @ x))
    return x, error

@jax.jit
def bigcstab_no_precond_solve(A, b):
    x, info = jax.scipy.sparse.linalg.bicgstab(A,
                                                b,
                                                tol=1e-10,
                                                atol=1e-10)

    error = jnp.linalg.norm(b - (A @ x))
    return x, error

@jax.jit
def gmres_solve_incremental(A, b):
    jacobi = get_diagonal(A)
    pc = lambda x: x * (1. / jacobi)
    x, info = jax.scipy.sparse.linalg.gmres(A,
                                            b,
                                            M=pc,
                                            tol=1e-10,
                                            atol=1e-10,
                                            maxiter=maxiter,
                                            restart=40,
                                            solve_method="incremental")
    # Note: incremental is very slow, but permits better convergence
    error = jnp.linalg.norm(b - (A @ x))
    return x, error

@jax.jit
def gmres_solve(A, b):
    jacobi = get_diagonal(A)
    pc = lambda x: x * (1. / jacobi)
    x, info = jax.scipy.sparse.linalg.gmres(A,
                                            b,
                                            M=pc,
                                            tol=1e-10,
                                            atol=1e-10,
                                            maxiter=maxiter,
                                            restart=40)
    error = jnp.linalg.norm(b - (A @ x))
    return x, error

@jax.jit
def cg_solve(A, b):
    jacobi = get_diagonal(A)
    pc = lambda x: x * (1. / jacobi)
    x, info = jax.scipy.sparse.linalg.cg(A,
                                        b,
                                        M=pc,
                                        tol=1e-10,
                                        atol=1e-10,
                                        maxiter=maxiter, 
                                        )
    error = jnp.linalg.norm(b - (A @ x))
    return x, error

@jax.jit
def spsolve_solve(A, b):
    x = js.linalg.spsolve(A.data,
                            A.indices,
                            A.indptr,
                            b,
                            tol=1e-10)
    error = jnp.linalg.norm(b - (A @ x))
    return x, error

@jax.jit
def lineax_gmres_solve(A, b):
    in_structure = jax.eval_shape(lambda: b)

    jacobi = get_diagonal(A)
    pc = lambda x: x * (1. / jacobi)
    preconditioner = lx.FunctionLinearOperator(pc, in_structure, tags=[lx.positive_semidefinite_tag])
    
    op = lambda x: A @ x
    operator = lx.FunctionLinearOperator(op, in_structure)
    
    solver = lx.GMRES(atol=1e-5, 
                      rtol=1e-5,
                      max_steps=maxiter,
                      restart=50)
    x = lx.linear_solve(operator, b, solver=solver, throw=False, options={"preconditioner":preconditioner}).value
    error = jnp.linalg.norm(b - (A @ x))

    return x, error

@jax.jit
def lineax_bigcstab_solve(A, b):
    in_structure = jax.eval_shape(lambda: b)

    jacobi = get_diagonal(A)
    pc = lambda x: x * (1. / jacobi)
    preconditioner = lx.FunctionLinearOperator(pc, in_structure, tags=[lx.positive_semidefinite_tag])
    
    op = lambda x: A @ x
    operator = lx.FunctionLinearOperator(op, in_structure)
    
    solver = lx.GMRES(atol=1e-5, 
                      rtol=1e-5,
                      max_steps=maxiter,
                      restart=50)
    x = lx.linear_solve(operator, b, solver=solver, throw=False, options={"preconditioner":preconditioner}).value
    error = jnp.linalg.norm(b - (A @ x))

    return x, error

A = pyamg.gallery.poisson((600, 600), format='csr')  # 2D Poisson problem on 500x500 grid
b = np.random.rand(A.shape[0])                   # pick a random right hand side

A_jax = js.BCOO.from_scipy_sparse(A)
b_jax = jnp.array(b)
Abcr = js.BCSR.from_bcoo(A_jax)

assert jnp.allclose(A.diagonal(), get_diagonal(A_jax))

maxiter = 100

benchmark("pyamg_solve", lambda: jax.block_until_ready(pyamg_solve(A, b)))
benchmark("bigcstab_solve", lambda: jax.block_until_ready(bigcstab_solve(A_jax, b_jax)))
benchmark("bigcstab_no_precond_solve", lambda: jax.block_until_ready(bigcstab_no_precond_solve(A_jax, b_jax)))
benchmark("gmres_solve", lambda: jax.block_until_ready(gmres_solve(A_jax, b_jax)))
benchmark("cg_solve", lambda: jax.block_until_ready(cg_solve(A_jax, b_jax)))
benchmark("spsolve_solve", lambda: jax.block_until_ready(spsolve_solve(Abcr, b_jax)))

benchmark("lineax_gmres_solve", lambda: jax.block_until_ready(lineax_gmres_solve(A_jax, b_jax)))
benchmark("lineax_bigcstab_solve", lambda: jax.block_until_ready(lineax_bigcstab_solve(A_jax, b_jax)))

"""
Seems like lineax.gmres achieves good performance up until ...

"""