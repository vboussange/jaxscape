import jax
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
import pytest
from jax import grad
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph, GridTopology, QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from networkx import grid_2d_graph


def _harmonic(x, y):
    return 2 * x * y / (x + y)


# TODO: implement test ROOK_CONTIGUITY vs QUEEN_CONTIGUITY


@pytest.fixture
def sample_gridgraph():
    grid = jnp.ones((2, 3))  # Uniform weights for simplicity
    grid = GridGraph(grid)
    return grid


def test_gridgraph_properties(sample_gridgraph):
    assert sample_gridgraph.height == 2
    assert sample_gridgraph.width == 3


def test_coord_to_index(sample_gridgraph):
    assert sample_gridgraph.coord_to_index(0, 0) == 0
    assert sample_gridgraph.coord_to_index(1, 2) == 5
    # assert sample_gridgraph.coord_to_index(2, 2) == -1  # Out of bounds


def test_index_to_coord(sample_gridgraph):
    assert jnp.array_equal(sample_gridgraph.index_to_coord(0), jnp.array([[0, 0]]))
    assert jnp.array_equal(sample_gridgraph.index_to_coord(5), jnp.array([[1, 2]]))
    # assert sample_gridgraph.index_to_coord(6) == (-1, -1)  # Out of bounds


def test_active_vertices_coordinates(sample_gridgraph):
    active_coords = sample_gridgraph.index_to_coord(jnp.arange(4))
    expected_coords = jnp.array([[0, 0], [0, 1], [0, 2], [1, 0]])
    assert jnp.array_equal(active_coords, expected_coords)


def test_node_values_to_raster():
    grid = jnp.ones((2, 3))
    grid = GridGraph(grid)

    values = jnp.array([10, 20, 30, 40, 50, 60])

    # Expected raster output based on the grid configuration and active values
    expected_raster = jnp.array(
        [
            [10, 20, 30],
            [40, 50, 60],
        ]
    )

    output_raster = grid.node_values_to_array(values)

    assert jnp.allclose(output_raster, expected_raster)


def test_array_to_node_values():
    grid = jnp.ones((2, 3))
    grid = GridGraph(grid)

    # Define values for the active nodes
    expected_values = jnp.array([10, 20, 30, 40, 50, 60])

    # Expected raster output based on the grid configuration and active values
    raster = jnp.array(
        [
            [10, 20, 30],
            [40, 50, 60],
        ]
    )

    values = grid.array_to_node_values(raster)

    assert jnp.allclose(expected_values, values)


# test against networkx
def test_adjacency_matrix():
    permeability_raster = jnp.ones((2, 3))
    grid = GridGraph(permeability_raster)
    edge_weights = grid.get_adjacency_matrix()

    assert isinstance(edge_weights, BCOO)  # Ensure that the output is a sparse matrix
    assert edge_weights.shape == (6, 6)  # 2x3 grid flattened to 6 nodes

    # trivial connectivity
    G = grid_2d_graph(2, 3, create_using=nx.DiGraph)
    assert jnp.array_equal(nx.adjacency_matrix(G).toarray(), edge_weights.todense())

    # not all nodes active, not trivial connectivity
    for edge in [((0, 1), (0, 2)), ((1, 2), (0, 2))]:
        G[edge[0]][edge[1]]["weight"] = 5
    permeability_raster = permeability_raster.at[0, 2].set(5)
    grid = GridGraph(permeability_raster)
    adj_matrix = grid.get_adjacency_matrix().todense()
    adj_matrix_nx = nx.adjacency_matrix(G, weight="weight").todense()
    assert jnp.array_equal(adj_matrix, adj_matrix_nx)


# test of custom fun
def test_adjacency_matrix_custom_fun():
    permeability_raster = jnp.ones((2, 3))
    grid = GridGraph(permeability_raster, fun=lambda x, y: 4 * (x + y))
    edge_weights = grid.get_adjacency_matrix()
    assert jnp.all(edge_weights.data[edge_weights.data > 0] == 8)


def test_differentiability_adjacency_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability

    def objective(permeability_raster):
        grid = GridGraph(permeability_raster)
        edge_weights = grid.get_adjacency_matrix()
        return edge_weights.sum()

    grad_objective = grad(objective)
    permeability_gradient = grad_objective(permeability_raster)
    assert permeability_gradient[0, 0] == 2
    assert permeability_gradient[0, 1] == 3
    assert permeability_gradient[1, 1] == 4


@pytest.mark.parametrize("neighbors", [ROOK_CONTIGUITY, QUEEN_CONTIGUITY])
def test_a_topology_gives_the_adjacency_matrix_of_each_surface(neighbors):
    topology = GridTopology((4, 7), neighbors)
    for seed in range(3):
        surface = jr.uniform(jr.key(seed), (4, 7), minval=0.1, maxval=2.0)

        actual = GridGraph(surface, fun=_harmonic, topology=topology)
        expected = GridGraph(surface, fun=_harmonic, neighbors=neighbors)

        actual, expected = (
            actual.get_adjacency_matrix(),
            expected.get_adjacency_matrix(),
        )
        assert actual.shape == expected.shape
        assert actual.indices.dtype == expected.indices.dtype
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(actual.data, expected.data)


@pytest.mark.parametrize("neighbors", [ROOK_CONTIGUITY, QUEEN_CONTIGUITY])
def test_a_topology_keeps_float32_gradients_accurate(neighbors):
    # Each pixel's gradient, the corner one included, keeps float32 accuracy
    # on a large raster, against float64 gradients without topology.
    shape = (512, 512)
    topology = GridTopology(shape, neighbors)
    surface = jr.uniform(jr.key(0), shape, jnp.float64, minval=0.1, maxval=2.0)
    weights = jr.uniform(jr.key(1), (surface.size * neighbors.shape[0],))

    def objective(surface, topology):
        if topology is None:
            graph = GridGraph(surface, fun=_harmonic, neighbors=neighbors)
        else:
            graph = GridGraph(surface, fun=_harmonic, topology=topology)
        data = graph.get_adjacency_matrix().data
        return jnp.sum(weights.astype(surface.dtype) * data)

    actual = grad(objective)(surface.astype(jnp.float32), topology)
    expected = grad(objective)(surface, None)

    np.testing.assert_allclose(actual, expected, rtol=1e-5)


def test_a_graph_takes_its_contiguity_from_its_topology():
    topology = GridTopology((4, 7), QUEEN_CONTIGUITY)

    graph = GridGraph(jnp.ones((4, 7)), topology=topology)

    np.testing.assert_array_equal(graph.neighbors, QUEEN_CONTIGUITY)
    with pytest.raises(ValueError, match="contiguity"):
        GridGraph(jnp.ones((4, 7)), neighbors=QUEEN_CONTIGUITY, topology=topology)


def test_a_topology_of_another_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        GridGraph(jnp.ones((7, 4)), topology=GridTopology((4, 7)))


def test_a_topology_needs_concrete_neighbors():
    def build(neighbors):
        return GridTopology((4, 7), neighbors).indices

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(build)(ROOK_CONTIGUITY)
