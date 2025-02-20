import pytest
import jax
import jax.numpy as jnp
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit
import jax.random as jr

def test_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))

    grid = GridGraph(vertex_weights = permeability_raster)
    distance = EuclideanDistance()
    dist = distance(grid)
    assert dist[0,0] == 0
    source_idx = 0
    target_idx = 4
    source_xy_coord = grid.index_to_coord(jnp.array([source_idx, target_idx]))
    assert dist[source_idx, target_idx] == jnp.sqrt(jnp.sum((source_xy_coord[0,:] - source_xy_coord[1,:])**2))
    
def test_differentiability_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    distance = EuclideanDistance()

    def objective(permeability_raster):
        grid = GridGraph(vertex_weights = permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)
        
    grad_objective = grad(objective)
    dobj = grad_objective(permeability_raster) # 1.26 ms ± 13.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    assert isinstance(dobj, jax.Array)
    

def test_jit_differentiability_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    distance = EuclideanDistance()

    def objective(permeability_raster):
        grid = GridGraph(vertex_weights = permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)
        
    grad_objective = jit(grad(objective))
    dobj = grad_objective(permeability_raster) # 70.6 μs ± 4.6 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    assert isinstance(dobj, jax.Array)

if __name__ == "__main__":
    pytest.main()
