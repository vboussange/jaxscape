import jax
import jax.numpy as jnp
from jaxscape.resistance_distance import (
    ResistanceDistance,
    p_inv_resistance_distance,
)
from jaxscape.gridgraph import GridGraph
from jaxscape.solvers import PyAMGSolver, CholmodSolver

import numpy as np
import jax.random as jr
from jax.experimental.sparse import BCOO
import networkx as nx

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

def test_lineax_solver_resistance_distance():
    """
    Tests that the lineax solver implementation of resistance distance
    produces the same result as the pseudo-inverse method.
    """
    key = jr.PRNGKey(42)
    permeability_raster = jr.uniform(key, (2, 2))
    grid = GridGraph(vertex_weights=permeability_raster, fun= lambda x, y: (x+y)/2)

    # nodes to nodes
    dist_pinv = ResistanceDistance(solver=None)(grid)
    
    for solver in [PyAMGSolver(), CholmodSolver()]:
        dist_lineax = ResistanceDistance(solver=solver)(grid)
        assert jnp.allclose(dist_pinv, dist_lineax, rtol=1e-4)