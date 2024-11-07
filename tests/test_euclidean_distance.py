import pytest
import jax
import jax.numpy as jnp
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.landscape import Landscape
from jaxscape.gridgraph import GridGraph
import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit

import jax.random as jr
def test_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities,
                              vertex_weights = permeability_raster)
    distance = EuclideanDistance(res=jnp.array(1.))
    dist = distance(grid)
    assert dist[0,0] == 0
    source_idx = 0
    target_idx = 4
    source_xy_coord = grid.active_vertex_index_to_coord(jnp.array([source_idx, target_idx]))
    assert dist[source_idx, target_idx] == jnp.sqrt(jnp.sum((source_xy_coord[0,:] - source_xy_coord[1,:])**2))
    
import jax.random as jr
def test_differentiability_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    D = 1.
    distance = EuclideanDistance(res=1.)

    def objective(permeability_raster):
        grid = GridGraph(activities=activities,
                                    vertex_weights = permeability_raster)
        dist = distance(grid)
        proximity = jnp.exp(-dist / D)
        landscape = Landscape(permeability_raster, proximity)
        func = landscape.equivalent_connected_habitat()
        return func
        
    grad_objective = grad(objective)
    dobj = grad_objective(permeability_raster) # 1.26 ms ± 13.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    assert isinstance(dobj, jax.Array)
    

def test_jit_differentiability_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    nb_active = int(activities.sum())
    D = 1.
    distance = EuclideanDistance(res=1.)

    def objective(permeability_raster):
        grid = GridGraph(activities=activities,
                        vertex_weights = permeability_raster,
                        nb_active=nb_active)
        dist = distance(grid)
        proximity = jnp.exp(-dist / D)
        landscape = Landscape(permeability_raster, proximity, nb_active=nb_active)
        func = landscape.equivalent_connected_habitat()
        return func
        
    grad_objective = jit(grad(objective))
    dobj = grad_objective(permeability_raster) # 70.6 μs ± 4.6 μs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
    assert isinstance(dobj, jax.Array)

if __name__ == "__main__":
    pytest.main()
