"""Test LSMR solver"""

import jax
import timeit
import lineax as lx
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
import pyamg
import numpy as np
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.gallery import poisson
from scipy.sparse.linalg import cg

def benchmark(method, func):
    times = timeit.repeat(func, number=1, repeat=10)
    _, error = func()
    print(f"{method} solve error: {error:2e}")
    print(f"{method} min time: {min(times)}\n")
    
@jax.jit
def linalg_solve(A, B):
    x = jnp.linalg.solve(A, B)
    error = jnp.linalg.norm(B - (A @ x))
    return x, error

@jax.jit
def lsmr_solve(A, b):
    def f(x):
        return A @ x
    in_structure = jax.eval_shape(lambda: b)
    operator = lx.FunctionLinearOperator(f, in_structure)
    solver = lx.LSMR(atol=1e-5, rtol=1e-5)
    x = lx.linear_solve(operator, b, solver=solver).value
    error = jnp.linalg.norm(b - (A @ x))
    return x, error

def amg_solve(A, B):
    ml = pyamg.ruge_stuben_solver(A)
    x = ml.solve(B, tol=1e-10)
    error = jnp.linalg.norm(B-A @ x)
    return x, error

def amg_cg_solve(A, B):
    ml = smoothed_aggregation_solver(A)
    M = ml.aspreconditioner(cycle='V')
    x, _ = cg(A, B, rtol = 1e-5, maxiter=30, M=M) 
    error = jnp.linalg.norm(B - (A @ x))
    return x, error

# def lineax_solve():
#     operator = lx.MatrixLinearOperator(A)
#     solver = lx.LU()
#     state = solver.init(operator, options={})
#     def solve_single(b):
#         x = lx.linear_solve(operator, b, solver=solver, state=state).value
#         return x
#     x = vmap(solve_single, in_axes=1, out_axes=1)(B)
#     error = jnp.linalg.norm(B - (A @ x))
#     return x, error



A = pyamg.gallery.poisson((50,50), format='csr')  # 2D Poisson problem on 500x500 grid
b = np.random.rand(A.shape[0])                      # pick a random right hand side

A_jax = BCOO.from_scipy_sparse(A)
b_jax = jnp.array(b)

# linalg_solve(A_jax, b_jax)
lsmr_solve(A_jax, b_jax)

benchmark("amg_solve", lambda: jax.block_until_ready(amg_solve(A, b)))
benchmark("amg_cg_solve", lambda: jax.block_until_ready(amg_cg_solve(A, b)))
benchmark("lsmr_solve", lambda: jax.block_until_ready(lsmr_solve(A_jax, b_jax)))