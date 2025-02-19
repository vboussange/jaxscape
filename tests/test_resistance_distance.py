import jax
import jax.numpy as jnp
from equinox import filter_jit, filter_grad
from jaxscape.resistance_distance import ResistanceDistance, p_inv_resistance_distance, lineax_solver_resistance_distance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph

import numpy as np
import jax.random as jr
from jax.experimental.sparse import BCOO
import networkx as nx
import jax.random as jr
import lineax as lx

def build_nx_resistance_distance_matrix(G):
    Rnx_dict = nx.resistance_distance(G, weight="weight", invert_weight=False)
    Rnx = jnp.zeros((G.number_of_nodes(), G.number_of_nodes()))
    node_list = list(G)
    for n, rd in Rnx_dict.items():
        i = node_list.index(n)
        for m, r in rd.items():
            j = node_list.index(m)
            Rnx = Rnx.at[i, j].set(r)
    return Rnx

def test_resistance_distance():
    permeability_raster = jnp.ones((2, 2))
    grid = GridGraph(vertex_weights=permeability_raster)

    distance = ResistanceDistance()
    mat = filter_jit(distance)(grid)
    assert isinstance(mat, jax.Array)
    
def test_p_inv_resistance_distance():
    G = nx.grid_2d_graph(2, 3)
    # for u, v in G.edges():
    #     G[u][v]['weight'] = 1

    # simple graph
    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = p_inv_resistance_distance(Ajx)
    Rnx = build_nx_resistance_distance_matrix(G)
    assert jnp.allclose(Rjaxscape, Rnx)
    
    # Add random weights to edges
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(1, 10)  # Random weight between 1 and 10

    A = nx.adjacency_matrix(G)
    Ajx = BCOO.from_scipy_sparse(A)
    Rjaxscape = p_inv_resistance_distance(Ajx)
    Rnx = build_nx_resistance_distance_matrix(G)
    assert jnp.allclose(Rjaxscape, Rnx)
    
# def test__landmark_resistance_distance():

#     key = jr.PRNGKey(0)  # Random seed is explicit in JAX
#     permeability_raster = jr.uniform(key, (11, 11))  # Start with a uniform permeability
#     activities = jnp.ones(permeability_raster.shape, dtype=bool)
#     grid = GridGraph(activities=activities, 
#                      vertex_weights=permeability_raster)
#     coarse_matrix = coarse_graining(grid, 2) 
#     landmarks = coarse_matrix.indices
#     Rjaxscape = _landmark_resistance_distance(Ajx, landmarks)
#     Rnx_dict = nx.resistance_distance(G)
#     Rnx = jnp.zeros(Rjaxscape.shape)
#     node_list = list(G)
#     for n, rd in Rnx_dict.items():
#         i = node_list.index(n)
#         for m, r in rd.items():
#             j = node_list.index(m)
#             Rnx = Rnx.at[i, j].set(r)
#     assert jnp.allclose(Rjaxscape, Rnx)
    
#     # Add random weights to edges
#     for u, v in G.edges():
#         G[u][v]['weight'] = np.random.uniform(1, 10)  # Random weight between 1 and 10

#     A = nx.adjacency_matrix(G)
#     Ajx = BCOO.from_scipy_sparse(A)
#     Rjaxscape = resistance_distance(Ajx)
#     Rnx_dict = nx.resistance_distance(G, weight="weight", invert_weight=False)
#     Rnx = jnp.zeros(Rjaxscape.shape)
#     node_list = list(G)
#     for n, rd in Rnx_dict.items():
#         i = node_list.index(n)
#         for m, r in rd.items():
#             j = node_list.index(m)
#             Rnx = Rnx.at[i, j].set(r)
#     assert jnp.allclose(Rjaxscape, Rnx)
    
def test_differentiability_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    distance = ResistanceDistance()

    def objective(permeability_raster):
        grid = GridGraph(permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)
    
    grad_objective = filter_grad(objective)
    # %timeit grad_objective(permeability_raster) # 71.2 ms ± 16.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)
    
def test_jit_differentiability_rsp_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    distance = ResistanceDistance()

    def objective(permeability_raster):
        grid = GridGraph(vertex_weights = permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)
        
    grad_objective = filter_jit(filter_grad(objective))
    # %timeit grad_objective(permeability_raster) # 13 μs ± 4.18 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)
    
def test_lineax_solver_resistance_distance():
    G = nx.grid_2d_graph(2, 3)
    
    # simple graph
    A = nx.adjacency_matrix(G)
    Ajax = BCOO.from_scipy_sparse(A)
    target = jnp.array([0])
    solver = lx.GMRES(atol=1e-5, 
                      rtol=1e-5,
                      max_steps=100,
                      restart=50)
    
    Rnx = build_nx_resistance_distance_matrix(G)
    Rjaxscape = lineax_solver_resistance_distance(Ajax, target, solver)
    
    assert jnp.allclose(Rjaxscape, Rnx[:, target].flatten())
    