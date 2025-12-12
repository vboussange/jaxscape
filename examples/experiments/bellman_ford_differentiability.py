"""
This script is a minimal working example to demonstrate that the Bellman Ford algorithm is not differentiable.
"""


import jax
import jax.numpy as jnp
from jaxscape.distance import AbstractDistance
from jax.experimental.sparse import BCOO
from jax import lax, ops
import equinox
import jax.random as jr
from jax import jit, grad
from jaxscape import GridGraph

def bellman_ford(W_data, W_indices, N, source: int):
    D = jnp.full(N, jnp.inf)
    D = D.at[source].set(0.0)
    
    @equinox.filter_checkpoint
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=N)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=N - 1)
    return D

    
def test_bellman_ford_differentiability():
    permeability_raster = jnp.ones((2, 2))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities,
                    grid = permeability_raster,
                    nb_active = permeability_raster.size)
    A = grid.get_adjacency_matrix()
    W_indices, W_data = A.indices, A.data
    def sum_bellman_ford(W_data):
        distances_jax = bellman_ford(W_data, W_indices, A.shape[0], 0)
        return jnp.sum(distances_jax)
    
    grad_sum_bellman_ford = jit(grad(sum_bellman_ford))
    sensitivity_bellman_ford = grad_sum_bellman_ford(W_data)
    print(sensitivity_bellman_ford)
