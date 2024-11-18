import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres
from jax.experimental import sparse
from jax.experimental.sparse import random_bcoo
import jax.random as jr
import timeit
import lineax as lx

# Define matrix size and seed
N = 10
key = jr.PRNGKey(0)

# Generate random sparse matrix and construct L matrix
W = random_bcoo(key, (N, N), nse=int(0.0001))
L = sparse.eye(W.shape[0], dtype=W.dtype, index_dtype=W.indices.dtype) - W
b = jnp.identity(W.shape[0], dtype=W.dtype)

# Function to benchmark GMRES solve
def gmres_solve():
    x, _ = gmres(L, b, 
                atol=1e-6,
                solve_method="incremental", 
                maxiter=1000, 
                restart=50
                 )
    error = jnp.linalg.norm(b - (L @ x))
    return x, error

# Function to benchmark matrix inverse solve
def inverse_solve():
    x = jnp.linalg.inv(L.todense()) @ b
    error = jnp.linalg.norm(b - (L @ x))
    return x, error

# Function to benchmark lineax solve
def lineax_solve():
    def f(x):
        return L @ x
    in_structure = jax.eval_shape(lambda: b)
    operator = lx.FunctionLinearOperator(f, in_structure)
    solver = lx.GMRES(atol=1e-6, rtol=1e-3)
    x = lx.linear_solve(operator, b, solver=solver).value
    error = jnp.linalg.norm(b - (L @ x))
    return x, error

# Benchmarking each method
def benchmark(method, func):
    print(f"Benchmarking {method}...")
    time_taken = timeit.timeit(func, number=10) / 10
    _, error = func()
    print(f"{method} solve error: {error:2e}")
    print(f"{method} average time: {time_taken * 1e3:.2f} ms\n")

# Running benchmarks
benchmark("GMRES", gmres_solve)
benchmark("Inverse", inverse_solve)
benchmark("Lineax", lineax_solve)


# TODO: it is unclear whether the lineax batch solve mode using
# `FunctionLinearOperator` and `eval_shape` is as efficient as the `inv` function.
# You should also test these in a realistic setting in the `_test_lineax.py` file.