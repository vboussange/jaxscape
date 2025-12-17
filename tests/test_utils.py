# from jaxscape.connectivity import strongly_connected_components
import jax.numpy as jnp
import jax.random as jr
from jax.experimental.sparse import BCOO, random_bcoo
from jaxscape.utils import (
    bcoo_at_set,
    bcoo_diag,
    bcoo_triu,
    connected_component_labels,
    padding,
)


def test_bcoo_diag():
    diagonal = jnp.array([1, 2, 3])
    sparse_matrix = bcoo_diag(diagonal)

    assert isinstance(sparse_matrix, BCOO)
    assert sparse_matrix.shape == (3, 3)
    assert sparse_matrix.nse == 3
    assert jnp.array_equal(sparse_matrix.todense(), jnp.diag(diagonal))

    diagonal = jnp.array([4, 5])
    sparse_matrix = bcoo_diag(diagonal)

    assert isinstance(sparse_matrix, BCOO)
    assert sparse_matrix.shape == (2, 2)
    assert sparse_matrix.nse == 2
    assert jnp.array_equal(sparse_matrix.todense(), jnp.diag(diagonal))

    diagonal = jnp.array([])
    sparse_matrix = bcoo_diag(diagonal)

    assert isinstance(sparse_matrix, BCOO)
    assert sparse_matrix.shape == (0, 0)
    assert sparse_matrix.nse == 0
    assert jnp.array_equal(sparse_matrix.todense(), jnp.diag(diagonal))


def test_bcoo_tril():
    # test tril, triu
    M = jnp.ones((10, 10))
    Mbcoo = BCOO.fromdense(M)

    Mbcoo_triu = bcoo_triu(Mbcoo, 1).todense()
    Mtriu = jnp.triu(M, 1)
    assert jnp.allclose(Mbcoo_triu, Mtriu)


def test_bcoo_triu():
    # test tril, triu
    M = jnp.ones((10, 10))
    Mbcoo = BCOO.fromdense(M)

    Mbcoo_triu = bcoo_triu(Mbcoo, 1).todense()
    Mtriu = jnp.triu(M, 1)
    assert jnp.allclose(Mbcoo_triu, Mtriu)


def test_bcoo_at_set():
    # testing one index
    orig_mat_sparse = random_bcoo(jr.PRNGKey(0), (5, 5), nse=0.1)
    mat_sparse = bcoo_at_set(
        orig_mat_sparse, jnp.array([1]), jnp.array([1]), jnp.array([1.0])
    )
    mat_dense = orig_mat_sparse.todense().at[1, 1].set(1.0)
    assert jnp.allclose(mat_sparse.todense(), mat_dense)

    # testing multiple indices
    mat_sparse = bcoo_at_set(
        orig_mat_sparse, jnp.array([1, 2]), jnp.array([1, 1]), jnp.array([1.0, 42])
    )
    mat_dense = orig_mat_sparse.todense().at[[1, 2], [1, 1]].set([1.0, 42])
    assert jnp.allclose(mat_sparse.todense(), mat_dense)


def test_padding():
    raster = jnp.ones((2300, 3600))
    buffer_size = 50
    window_size = 3
    padded_raster = padding(raster, buffer_size, window_size)

    for i in range(2):
        assert (padded_raster.shape[i] - 2 * buffer_size) % window_size == 0

    # other test
    raster = jnp.ones((230, 360))
    buffer_size = 3
    window_size = 27
    padded_raster = padding(raster, buffer_size, window_size)

    for i in range(2):
        assert (padded_raster.shape[i] - 2 * buffer_size) % window_size == 0


def test_connected_component_labels():
    # Graph with two components: {0,1} and {2,3}
    adjacency = jnp.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    A = BCOO.fromdense(adjacency)

    labels = connected_component_labels(A)

    assert bool(labels[0] == labels[1])
    assert bool(labels[2] == labels[3])
    assert bool(labels[0] != labels[2])


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
