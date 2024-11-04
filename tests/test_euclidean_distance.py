import pytest
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from connectax.gridgraph import GridGraph  # Replace 'your_module' with the actual module name
from connectax.euclidean_distance import euclidean_distance_matrix  # Replace 'your_module' with the actual module name

import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit

import jax.random as jr
def test_euclidean_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities, permeability_raster)
    mat = euclidean_distance_matrix(grid, 1.)
    assert mat[0,0] == 0
    source_idx = 0
    target_idx = 4
    source_xy_coord = grid.active_vertex_coordinate(jnp.array([source_idx, target_idx]))
    assert mat[source_idx, target_idx] == jnp.sqrt(jnp.sum((source_xy_coord[0,:] - source_xy_coord[1,:])**2))

if __name__ == "__main__":
    pytest.main()
