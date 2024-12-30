import pytest
import jax
import jax.numpy as jnp
from jax import grad, jit
from jaxscape.rsp_distance import RSPDistance, rsp_distance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
from jaxscape.utils import BCOO_to_coo, get_largest_component_label
from pathlib import Path
from scipy.sparse.csgraph import connected_components
import numpy as np
import jax.random as jr
from jax.experimental.sparse import BCOO
from jaxscape.utils import mapnz

jax.config.update("jax_enable_x64", True)

def test_rsp_distance_matrix():
    # This is the base example taken from ConScape
    expected_cost_conscape = jnp.array([
                                        [0.0,      1.01848, 1.01848, 2.01848],
                                        [1.01848,  0.0,     2.01848, 1.01848],
                                        [1.01848,  2.01848, 0.0,     1.01848],
                                        [2.01848,  1.01848, 1.01848, 0.0]
                                    ])
    permeability_raster = jnp.ones((2, 2))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability_raster)
    theta = jnp.array(2.)
    distance = RSPDistance(theta, cost= lambda x: x)
    mat = distance(grid)
    assert jnp.allclose(mat, expected_cost_conscape, atol = 1e-4)

# test with more complex setting with not activation of a node
def test_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    activities = activities.at[0, 0].set(False)
    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability_raster)
    theta = jnp.array(0.01)
    distance = RSPDistance(theta)
    cost = distance.cost_matrix(grid)
    assert cost.sum() > 0

    mat = distance(grid)
    assert jnp.any(~jnp.isnan(mat))
    assert isinstance(mat, jax.Array)
    assert grid.nb_active == 99
    
def test_rsp_distance():
    A = BCOO.fromdense(jnp.array(
    [
        [0, 1.2, 1.2, 0, 0, 0],
        [1.2, 0, 0, 1.2, 0, 0],
        [1.2, 0, 0, 0, 1.2, 0],
        [0, 1.5, 0, 0, 0, 1.5],
        [0, 0, 1.5, 0, 0, 1.5],
        [0, 0, 0, 1.5, 1.5, 0]
    ],
    dtype="float32",
    ))
    theta = jnp.array(1.)
    C = mapnz(A, lambda x: -jnp.log(x))
    dist = rsp_distance(theta, A, C)
    assert isinstance(dist, jax.Array)
    
# test with true raster
def test_rsp_distance_matrix():
    
    raster_path = Path(__file__).parent / "data/habitat_suitability.csv"
    habitat_suitability = jnp.array(np.loadtxt(raster_path, delimiter=","))

    conscape_dist_path = Path(__file__).parent /  "data/conscape_rsp_distance_to_i=19_j=6.csv"
    expected_cost_conscape = jnp.array(np.loadtxt(conscape_dist_path, delimiter=","))
    activities = habitat_suitability > 0
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_suitability)
    
    # pruning grid graph to have only connected vertices active
    A = grid.get_adjacency_matrix()
    Anp = BCOO_to_coo(A)
    _, labels = connected_components(Anp, directed=True, connection="strong")
    label = get_largest_component_label(labels)
    vertex_belongs_to_largest_component_node = labels == label
    activities_pruned = grid.node_values_to_array(vertex_belongs_to_largest_component_node)
    activities_pruned = activities_pruned == True
    graph_pruned = GridGraph(activities=activities_pruned, 
                             vertex_weights=habitat_suitability)
    
    
    # calculating distance to vertex 19, 6 in julia coordinates (corresponding to vertex 18, 5 in python coordinate)
    theta = jnp.array(0.01)
    distance = RSPDistance(theta)
    mat = distance(graph_pruned)
    vertex_index = graph_pruned.coord_to_active_vertex_index(18, 5)
    expected_cost = graph_pruned.node_values_to_array(mat[:, vertex_index])

    assert jnp.allclose(expected_cost[~jnp.isnan(expected_cost)], expected_cost_conscape[~jnp.isnan(expected_cost_conscape)], rtol = 1e-6)
    assert jnp.allclose(jnp.isnan(expected_cost), jnp.isnan(expected_cost_conscape))    

def test_differentiability_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    D = 1.
    theta = jnp.array(0.01)
    distance = RSPDistance(theta)

    def objective(permeability_raster):
        grid = GridGraph(activities=activities, vertex_weights=permeability_raster)
        dist = distance(grid)
        proximity = jnp.exp(-dist / D)
        landscape = ExplicitGridGraph(activities=activities, 
                                  vertex_weights=permeability_raster, 
                                  adjacency_matrix=proximity)
        func = landscape.equivalent_connected_habitat()
        return func
        
    grad_objective = grad(objective)
    # %timeit grad_objective(permeability_raster) # 71.2 ms ± 16.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)
    # TODO: implement proper test
    

def test_jit_differentiability_rsp_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    nb_active = int(activities.sum())
    D = 1.
    theta = jnp.array(0.01)
    distance = RSPDistance(theta=theta)

    def objective(permeability_raster):
        grid = GridGraph(activities=activities,
                        vertex_weights = permeability_raster,
                        nb_active=nb_active)
        dist = distance(grid)
        proximity = jnp.exp(-dist / D)
        landscape = ExplicitGridGraph(activities=activities, 
                            vertex_weights=permeability_raster, 
                            adjacency_matrix=proximity,
                            nb_active=nb_active)
        func = landscape.equivalent_connected_habitat()
        return func
        
    grad_objective = jit(grad(objective))
    # %timeit grad_objective(permeability_raster) # 13 μs ± 4.18 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)
    
if __name__ == "__main__":
    pytest.main()
