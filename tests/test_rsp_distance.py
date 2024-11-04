import pytest
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from connectax.gridgraph import GridGraph  # Replace 'your_module' with the actual module name
from connectax.rsp_distance import rsp_distance  # Replace 'your_module' with the actual module name

import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit

import jax.random as jr
def test_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities, permeability_raster)
    A = grid.adjacency_matrix()
    theta = 0.01
    mat = rsp_distance(A, theta)
    # TODO: implement proper test
    
    
# TODO: implement differentiability
    
if __name__ == "__main__":
    pytest.main()
