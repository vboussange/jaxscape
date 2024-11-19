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

N = 20
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

@jit
def linalg_solve():
    x = jnp.linalg.solve(Lreg.todense(), Idense)
    error = jnp.linalg.norm(Idense - (Lreg @ x))
    return x, error

# Function to benchmark lineax solve
def lineax_sparse_solve(solver):
    def f(x):
        return Lreg @ x
    in_structure = jax.ShapeDtypeStruct((Lreg.shape[0],), Lreg.dtype)
    operator = lx.FunctionLinearOperator(f, in_structure)
    def solve_single(b):
        x = lx.linear_solve(operator, b, solver=solver).value
        return x
    X = jax.vmap(solve_single, in_axes=1, out_axes=1)(I)
    error = jnp.linalg.norm(Idense - (Lreg @ X))
    return X, error

# Function to benchmark lineax solve
def lineax_solve(solver):
    operator = lx.MatrixLinearOperator(Lreg.todense())
    state = solver.init(operator, options={})
    def solve_single(b):
        x = lx.linear_solve(operator, b, solver=solver, state=state).value
        return x
    X = jax.vmap(solve_single, in_axes=1, out_axes=1)(Idense)
    error = jnp.linalg.norm(Idense - (Lreg @ X))
    return X, error

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

benchmark("linalg.solve", linalg_solve)
# solver = lx.GMRES(atol=1e-3, rtol=1e-3) # similarly, does not converge
# benchmark("Lineax",lambda: lineax_solve(solver))

myfun = jit(lambda: lineax_solve(lx.LU()))
benchmark("Lineax", myfun)

myfun = jit(lambda: lineax_solve(lx.SVD()))
benchmark("Lineax", myfun)
# Lineax solve error: 3.837553e-01
# Lineax average time: 2714.45 ms

myfun = jit(lambda: lineax_sparse_solve(lx.LU()))
benchmark("Lineax", myfun)
# does not work, needs some nbatch

