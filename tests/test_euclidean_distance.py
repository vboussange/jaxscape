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
    