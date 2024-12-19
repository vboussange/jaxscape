"""
Wrapping pyamg aspreconditioner with pure_callback and calling jax cg

TODO: current implementation of jax.cg does not look good - instead you would better have that of lineax
"""


import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.scipy.sparse.linalg import cg as jcg
import numpy as np
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.gallery import poisson
import lineax as lx
from jax.experimental.sparse import BCOO
import timeit
from scipy.sparse.linalg import cg

def benchmark(method, func):
    times = timeit.repeat(func, number=1, repeat=10)
    _, error = func()
    print(f"{method} solve error: {error:2e}")
    print(f"{method} min time: {min(times)}\n")

def M_jax(x):
    # x is a JAX array.
    # We must provide shape and dtype info to pure_callback.
    out_shape = x.shape
    out_dtype = x.dtype
    out_spec = ShapeDtypeStruct(out_shape, out_dtype)
    return jax.pure_callback(M, out_spec, x)

@jax.jit
def jcg_solve(A_jax, B_jax):
    X, _ = jcg(A_jax, 
               B_jax, 
               M=M_jax, atol=1e-10, tol=1e-10)
    error = jnp.linalg.norm(B_jax - (A_jax @ X))
    return X, error

def pyamg_solve(A, B):    
    X = np.empty_like(B)
    for j in range(X.shape[1]):
        X[:, j] = ml.solve(B[:, j], tol=1e-10)

    error = np.linalg.norm(B - A @ X)          # compute norm of residual vector
    return X, error


def pyamg_cg_solve(A, B): 
    X = np.empty_like(B)
    for j in range(X.shape[1]):
        X[:, j], _ = cg(A, B[:, j], rtol=1e-8, maxiter=30, M=M)

    error = np.linalg.norm(B - A @ X) # compute norm of residual vector
    return X, error


# Original setup
A = poisson((500, 500), format='csr')  # matrix
b = np.random.rand(A.shape[0])         # random RHS
ml = smoothed_aggregation_solver(A)    # AMG solver
M = ml.aspreconditioner(cycle='V')     # preconditioner (M is a linear operator)

# Convert A and b to JAX arrays
A_jax = BCOO.from_scipy_sparse(A)
b_jax = jnp.array(b)

num_rhs = 10
B = np.random.rand(A.shape[0], num_rhs)
B_jax = jnp.array(B)

device = jax.devices("cpu")[0]
benchmark("jcg_solve, cpu", lambda: jax.block_until_ready(jcg_solve(jax.device_put(A_jax, device), 
                                                                    jax.device_put(B_jax, device))))
device = jax.devices("gpu")[0]
benchmark("jcg_solve, gpu", lambda: jax.block_until_ready(jcg_solve(jax.device_put(A_jax, device), 
                                                                    jax.device_put(B_jax, device))))

benchmark("pyamg_solve", lambda: jax.block_until_ready(pyamg_solve(A, B)))
benchmark("pyamg_cg_solve", lambda: jax.block_until_ready(pyamg_cg_solve(A, B)))
"""
jcg_solve, cpu solve error: 2.275686e+00
jcg_solve, cpu min time: 7.084284259006381


jcg_solve, gpu solve error: 2.397763e+00
jcg_solve, gpu min time: 4.26438863799558

pyamg_solve solve error: 2.956911e-08
pyamg_solve min time: 6.149242135987151

pyamg_cg_solve solve error: 2.738978e-06
pyamg_cg_solve min time: 3.321608076017583
"""