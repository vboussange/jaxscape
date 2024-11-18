import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import jax.random as jr
import jax
from jaxscape.gridgraph import GridGraph
from jaxscape.utils import bcoo_diag
from jax.experimental import sparse
from jax.scipy.sparse.linalg import gmres, cg
import timeit
from jax import jit
import equinox as eqx

N = 10
key = jr.PRNGKey(0)  # Random seed is explicit in JAX
permeability_raster = jr.uniform(key, (N, N))  # Start with a uniform permeability
activities = jnp.ones(permeability_raster.shape, dtype=bool)
grid = GridGraph(activities, permeability_raster)
A = grid.get_adjacency_matrix()
D = bcoo_diag(A.sum(axis=1).todense())  # Degree matrix
L = D - A  # Laplacian matrix
I = bcoo_diag(jnp.ones(L.shape[0]))
Idense = I.todense()
eps = 5e-7
Lreg = L + eps * I

# Function to benchmark GMRES solve
@jit
def gmres_solve():
    x, _ = gmres(Lreg, Idense, 
                atol=1e-6,
                solve_method="incremental", 
                maxiter=1000, 
                restart=50
                 )
    error = jnp.linalg.norm(Idense - (Lreg @ x))
    return x, error

# Function to benchmark matrix inverse solve
@jit
def inverse_solve():
    x = jnp.linalg.inv(Lreg.todense())
    error = jnp.linalg.norm(Idense - (Lreg @ x))
    return x, error

# Function to benchmark lineax solve
def lineax_solve(solver):
    def f(x):
        return Lreg @ x
    in_structure = jax.eval_shape(lambda: Idense)
    operator = lx.FunctionLinearOperator(f, in_structure)
    @eqx.filter_jit
    def solve():
        x = lx.linear_solve(operator, Idense, solver=solver).value
        error = jnp.linalg.norm(Idense - (Lreg @ x))
        return x, error

    return solve()

# def lineax_solve(solver):
#     def f(x):
#         return Lreg @ x
#     in_structure = jax.eval_shape(lambda: Idense)
#     operator = lx.FunctionLinearOperator(f, in_structure)
#     x = lx.linear_solve(operator, Idense, solver=solver).value
#     error = jnp.linalg.norm(Idense - (Lreg @ x))
#     return x, error

# Benchmarking each method
def benchmark(method, func):
    print(f"Benchmarking {method}...")
    time_taken = timeit.timeit(func, number=10) / 10
    _, error = func()
    print(f"{method} solve error: {error:2e}")
    print(f"{method} average time: {time_taken * 1e3:.2f} ms\n")

# Running benchmarks
# benchmark("GMRES", gmres_solve) # does not converge...
benchmark("Inverse", inverse_solve)
# Inverse solve error: 4.254667e-01
# Inverse average time: 10.93 ms

# solver = lx.GMRES(atol=1e-3, rtol=1e-3) # similarly, does not converge
# benchmark("Lineax",lambda: lineax_solve(solver))

solver = lx.LU()
benchmark("Lineax",lambda: lineax_solve(solver))

# does not seem to work
solver = lx.SVD()
benchmark("Lineax",lambda: lineax_solve(solver))
# Lineax solve error: 3.837553e-01
# Lineax average time: 2714.45 ms