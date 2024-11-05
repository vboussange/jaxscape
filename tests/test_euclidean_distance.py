import pytest
import jax
import jax.numpy as jnp
from connectax.gridgraph import GridGraph
from connectax.euclidean_distance import euclidean_distance_matrix
from connectax.connectivity import functional_habitat

import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit

import jax.random as jr
def test_euclidean_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities, permeability_raster)
    dist = euclidean_distance_matrix(grid, 1.)
    assert dist[0,0] == 0
    source_idx = 0
    target_idx = 4
    source_xy_coord = grid.active_vertex_index_to_coord(jnp.array([source_idx, target_idx]))
    assert dist[source_idx, target_idx] == jnp.sqrt(jnp.sum((source_xy_coord[0,:] - source_xy_coord[1,:])**2))
    
import jax.random as jr
def test_differentiability_euclidean_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    def objective(permeability_raster):
        grid = GridGraph(activities, permeability_raster)
        dist = euclidean_distance_matrix(grid, 1.)
        active_ij = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active()))
        q = permeability_raster[active_ij[:,0], active_ij[:,1]]
        func = functional_habitat(q, dist)
        return func
        
    grad_objective = grad(objective)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)

if __name__ == "__main__":
    pytest.main()
