import pytest
import jax
import jax.numpy as jnp
from jax import grad, jit
from jaxscape.resistance_distance import ResistanceDistance, resistance_distance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
from jaxscape.utils import BCOO_to_sparse, get_largest_component_label
from jaxscape.utils import well_adapted_movement
from pathlib import Path
from scipy.sparse.csgraph import connected_components
import numpy as np
import jax.random as jr
from jax.experimental.sparse import BCOO
import networkx as nx
import jax.random as jr

# jax.config.update("jax_enable_x64", True)

def test_resistance_distance():
    permeability_raster = jnp.ones((2, 2))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability_raster)

    distance = ResistanceDistance()
    mat = distance(grid)
    assert jnp.allclose(mat)
    
def test_resistance_distance_matrix():
    G = nx.grid_2d_graph(2, 3)
    
    # simple graph
    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = resistance_distance(Ajx)
    Rnx_dict = nx.resistance_distance(G)
    Rnx = jnp.zeros(Rjaxscape.shape)
    node_list = list(G)
    for n, rd in Rnx_dict.items():
        i = node_list.index(n)
        for m, r in rd.items():
            j = node_list.index(m)
            Rnx = Rnx.at[i, j].set(r)
    assert jnp.allclose(Rjaxscape, Rnx)
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(1, 10)  # Random weight between 1 and 10

    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = resistance_distance(Ajx)
    Rnx_dict = nx.resistance_distance(G, weight="weight", invert_weight=False)
    Rnx = jnp.zeros(Rjaxscape.shape)
    node_list = list(G)
    for n, rd in Rnx_dict.items():
        i = node_list.index(n)
        for m, r in rd.items():
            j = node_list.index(m)
            Rnx = Rnx.at[i, j].set(r)
    assert jnp.allclose(Rjaxscape, Rnx)
    
def test_differentiability_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    D = 1.
    distance = ResistanceDistance()

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
    
def test_jit_differentiability_rsp_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    nb_active = int(activities.sum())
    D = 1.
    distance = ResistanceDistance()

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
    