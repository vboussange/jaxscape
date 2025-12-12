from jaxscape import GridGraph
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import time


N = 1000
key = jr.PRNGKey(0)
permeability = jr.normal(key, (N, N))

def objective(permeability):
    activities = jnp.ones_like(permeability, dtype="bool")
    grid = GridGraph(activities=activities, 
                     grid=permeability, 
                     nb_active=activities.size)
    A = grid.get_adjacency_matrix()
    return A.sum()

grad_objective = eqx.filter_jit(eqx.filter_grad(objective))

# compiling
grad_objective(permeability)

# Benchmark
start_time = time.time()
for _ in range(10):
    grad_objective(permeability).block_until_ready()
end_time = time.time()

print(f"Average execution time: {(end_time - start_time) / 100:.6f} seconds")