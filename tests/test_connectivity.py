# from connectax.connectivity import strongly_connected_components
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax import random
import jax.random as jr
from connectax.connectivity import Landscape
from connectax.rsp_distance import well_adapted_movement

def test_Landscape():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    grid = Landscape(activities=activities, 
                    vertex_weights=permeability_raster,
                    cost=well_adapted_movement)
    adjacency_matrix = grid.adjacency_matrix()
    cost_matrix = grid.cost_matrix()
    assert isinstance(adjacency_matrix, BCOO)
    assert isinstance(cost_matrix, BCOO)

# def test_strongly_connected_components():
#     # Helper function to build a sparse BCOO graph
#     def build_bcoo_graph(n, edges):
#         # Create a dense adjacency matrix
#         adj_matrix = jnp.zeros((n, n), dtype=jnp.int32)
#         for src, dst in edges:
#             adj_matrix = adj_matrix.at[src, dst].set(1)
#         return BCOO.fromdense(adj_matrix)

#     # Test Case 1: Simple graph with two SCCs {0, 1, 2} and {3, 4}
#     n1 = 5
#     edges1 = [(0, 1), (1, 2), (2, 0), (3, 4)]
#     graph1 = build_bcoo_graph(n1, edges1)
#     expected_sccs1 = [[0, 1, 2], [3], [4]]

#     # Test Case 2: Single node with self-loop
#     n2 = 1
#     edges2 = [(0, 0)]
#     graph2 = build_bcoo_graph(n2, edges2)
#     expected_sccs2 = [[0]]

#     # Test Case 3: Disconnected nodes
#     n3 = 3
#     edges3 = []
#     graph3 = build_bcoo_graph(n3, edges3)
#     expected_sccs3 = [[0], [1], [2]]

#     # Test Case 4: Complex graph with nested SCCs {0, 1, 2} and {3, 4, 5, 6}
#     n4 = 7
#     edges4 = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 6), (6, 3)]
#     graph4 = build_bcoo_graph(n4, edges4)
#     expected_sccs4 = [[0, 1, 2], [3, 4, 5, 6]]

#     # Test case dictionary
#     test_cases = [
#         (graph1, expected_sccs1),
#         (graph2, expected_sccs2),
#         (graph3, expected_sccs3),
#         (graph4, expected_sccs4)
#     ]

#     # Run each test case
#     for i, (graph, expected_sccs) in enumerate(test_cases, 1):
#         computed_sccs = strongly_connected_components(graph)
        
#         # Sort and compare the computed and expected SCCs
#         computed_sccs_sorted = [sorted(scc.tolist()) for scc in computed_sccs]
#         expected_sccs_sorted = [sorted(scc) for scc in expected_sccs]

#         assert sorted(computed_sccs_sorted) == sorted(expected_sccs_sorted), \
#             f"Test case {i} failed. Expected {expected_sccs_sorted}, got {computed_sccs_sorted}"
#         print(f"Test case {i} passed.")

# # Run the test function
# test_strongly_connected_components()
