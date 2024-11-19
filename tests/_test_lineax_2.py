import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
import lineax as lx
import timeit


N = 20
key = jr.PRNGKey(0)
A = jr.uniform(key, (N, N))
B = jnp.eye(N, N)

# Function to benchmark matrix inverse solve
@jit
def inverse_solve():
    x = jnp.linalg.inv(A)
    error = jnp.linalg.norm(B - (A @ x))
    return x, error

@jit
def linalg_solve():
    x = jnp.linalg.solve(A, B)
    error = jnp.linalg.norm(B - (A @ x))
    return x, error

@jit
def lineax_solve():
    operator = lx.MatrixLinearOperator(A)
    solver = lx.LU()
    state = solver.init(operator, options={})
    def solve_single(b):
        x = lx.linear_solve(operator, b, solver=solver, state=state).value
        return x
    x = vmap(solve_single, in_axes=1, out_axes=1)(B)
    error = jnp.linalg.norm(B - (A @ x))
    return x, error

# Benchmarking each method
def benchmark(method, func):
    time_taken = timeit.timeit(func, number=10) / 10
    _, error = func()
    print(f"{method} solve error: {error:2e}")
    print(f"{method} average time: {time_taken * 1e3:.2f} ms\n")

# Running benchmarks
benchmark("Inverse", inverse_solve)
# Inverse solve error: 4.254667e-01
# Inverse average time: 10.93 ms

benchmark("linalg.solve", linalg_solve)
# linalg.solve solve error: 6.581411e-06
# linalg.solve average time: 0.03 ms

benchmark("Lineax", lineax_solve)
# Lineax solve error: 6.581411e-06
# Lineax average time: 0.13 ms