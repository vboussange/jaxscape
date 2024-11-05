import pytest
import jax
import jax.numpy as jnp
from connectax.gridgraph import GridGraph
from connectax.rsp_distance import rsp_distance
from connectax.connectivity import functional_habitat, BCOO_to_sparse, get_largest_component
import xarray as xr
from pathlib import Path
from scipy.sparse.csgraph import connected_components
import numpy as np
import jax.random as jr
from jax.experimental.sparse import BCOO

jax.config.update("jax_enable_x64", True)


# conscape test, does not work because cost matrix 
# in ConScape test is not -log(x). 
# TODO: to be implemented
# expected_cost_conscape = jnp.array([
#                                     [0.0,      1.01848, 1.01848, 2.01848],
#                                     [1.01848,  0.0,     2.01848, 1.01848],
#                                     [1.01848,  2.01848, 0.0,     1.01848],
#                                     [2.01848,  1.01848, 1.01848, 0.0]
#                                 ])
# def test_rsp_distance_matrix():
#     permeability_raster = jnp.ones((2, 2))
#     activities = jnp.ones(permeability_raster.shape, dtype=bool)
#     grid = GridGraph(activities, permeability_raster)
#     A = grid.adjacency_matrix()
#     theta = jnp.array(2.)
#     mat = rsp_distance(A, theta)
#     assert jnp.allclose(mat, expected_cost_conscape, atol = 1e-4)

# test with more complex setting with not activation of a node
def test_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    activities = activities.at[0, 0].set(False)
    grid = GridGraph(activities, permeability_raster)
    A = grid.adjacency_matrix()
    theta = 0.01
    mat = rsp_distance(A, theta)
    assert isinstance(mat, jax.Array)
    assert A.shape[0] == 99
    
# test with true raster
# TOFIX: this is not working
def test_rsp_distance_matrix():
    sp_name = "Salmo trutta"
    conscape_dist_path = f"data/{sp_name}_conscape_rsp_distance.csv"
    expected_cost_conscape = jnp.array(np.loadtxt(conscape_dist_path, delimiter=","))
    
    
    # Ideally, we construct A based on the raster, but it seems that there is a
    # problem of shifted vertices when doing so 
    # TODO: this is a quick fix that
    # should be worked out
    conscape_A_path = "data/Salmo trutta_conscape_affinity_matrix.csv"
    A_pruned = BCOO.fromdense(jnp.array(np.loadtxt(conscape_A_path, delimiter=",")))

    # path = Path("data/habitat_suitability.nc")
    # with xr.open_dataset(path, engine="netcdf4", decode_coords="all") as da: 
    #     habitat_suitability = da[sp_name] / 100
    #     da.close()
    # permeability_raster = jnp.array(habitat_suitability.data[0,:,:], dtype="float64")
    # permeability_raster = jnp.nan_to_num(permeability_raster, nan=0.).T
    # # activities = jnp.ones(permeability_raster.shape, dtype=bool)
    # activities = ~jnp.isnan(permeability_raster)
    # grid = GridGraph(activities, permeability_raster)
    # A = grid.adjacency_matrix()
    
    # Anp = BCOO_to_sparse(A)
    
    # _, labels = connected_components(Anp, directed=True, connection="strong")
    # connected_vertices = get_largest_component(labels)
    # A_pruned = Anp.tocsr()[connected_vertices, :][:, connected_vertices]
    # A_pruned = BCOO.from_scipy_sparse(A_pruned)


    theta = jnp.array(0.01)
    mat = rsp_distance(A_pruned, theta)
    assert jnp.allclose(mat, expected_cost_conscape, atol = 1e-4)    
    
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
