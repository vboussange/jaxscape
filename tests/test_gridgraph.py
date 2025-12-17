import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import pytest
from jax import grad
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph
from networkx import grid_2d_graph


# TODO: implement test ROOK_CONTIGUITY vs QUEEN_CONTIGUITY


@pytest.fixture
def sample_gridgraph():
    grid = jnp.ones((2, 3))  # Uniform weights for simplicity
    grid = GridGraph(grid)
    sample_gridgraph = grid  # to comment
    return grid


def test_gridgraph_properties(sample_gridgraph):
    assert sample_gridgraph.height == 2
    assert sample_gridgraph.width == 3


def test_coord_to_index(sample_gridgraph):
    assert sample_gridgraph.coord_to_index(0, 0) == 0
    assert sample_gridgraph.coord_to_index(1, 2) == 5
    # assert sample_gridgraph.coord_to_index(2, 2) == -1  # Out of bounds


def test_index_to_coord(sample_gridgraph):
    assert jnp.array_equal(sample_gridgraph.index_to_coord(0), [[0, 0]])
    assert jnp.array_equal(sample_gridgraph.index_to_coord(5), [[1, 2]])
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
