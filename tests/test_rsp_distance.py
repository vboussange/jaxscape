import pytest
import jax
import jax.numpy as jnp
from connectax.gridgraph import GridGraph
from connectax.rsp_distance import rsp_distance
from connectax.connectivity import functional_habitat


import jax.random as jr
def test_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities, permeability_raster)
    A = grid.adjacency_matrix()
    theta = 0.01
    mat = rsp_distance(A, theta)
    assert isinstance(mat, jax.Array)
    # TODO: implement proper test
    
    
import jax.random as jr
def test_differentiability_euclidean_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    def objective(permeability_raster):
        grid = GridGraph(activities, permeability_raster)
        theta = jnp.array(0.01)
        affinity = grid.adjacency_matrix()
        dist = rsp_distance(affinity, theta)
        active_ij = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active()))
        q = permeability_raster[active_ij[:,0], active_ij[:,1]]
        func = functional_habitat(q, dist)
        return func
        
    grad_objective = grad(objective)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)
    # TODO: implement proper test
    
if __name__ == "__main__":
    pytest.main()
