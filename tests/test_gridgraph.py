import pytest
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from connectax.gridgraph import GridGraph, ROOK_CONTIGUITY  # Replace 'your_module' with the actual module name
import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit


@pytest.fixture
def sample_gridgraph():
    activity = jnp.array([[1, 0, 1],
                          [1, 1, 0]])  # 2x3 grid with some inactive cells
    vertex_weights = jnp.ones((2, 3))  # Uniform weights for simplicity
    grid = GridGraph(activity, vertex_weights)
    return grid


def test_gridgraph_properties(sample_gridgraph):
    assert sample_gridgraph.height == 2
    assert sample_gridgraph.width == 3


def test_coord_to_index(sample_gridgraph):
    assert sample_gridgraph.coord_to_index(0, 0) == 0
    assert sample_gridgraph.coord_to_index(1, 2) == 5
    # assert sample_gridgraph.coord_to_index(2, 2) == -1  # Out of bounds


def test_index_to_coord(sample_gridgraph):
    assert sample_gridgraph.index_to_coord(0) == (0, 0)
    assert sample_gridgraph.index_to_coord(5) == (1, 2)
    # assert sample_gridgraph.index_to_coord(6) == (-1, -1)  # Out of bounds


def test_vertex_active(sample_gridgraph):
    assert sample_gridgraph.vertex_active(0) == 1
    assert sample_gridgraph.vertex_active(1) == 0
    assert sample_gridgraph.vertex_active(5) == 0


def test_vertex_active_coord(sample_gridgraph):
    assert sample_gridgraph.vertex_active_coord(0, 0) == 1
    assert sample_gridgraph.vertex_active_coord(0, 1) == 0
    assert sample_gridgraph.vertex_active_coord(1, 2) == 0


def test_nb_active(sample_gridgraph):
    assert sample_gridgraph.nb_active() == 4  # There are four active vertices


def test_all_active(sample_gridgraph):
    assert not sample_gridgraph.all_active()  # Not all vertices are active


def test_list_active_vertices(sample_gridgraph):
    active_vertices = sample_gridgraph.list_active_vertices()
    expected_active_vertices = jnp.array([0, 2, 3, 4])
    assert jnp.array_equal(active_vertices, expected_active_vertices)


def test_active_vertices_coordinates(sample_gridgraph):
    active_coords = sample_gridgraph.active_vertex_index_to_coord(jnp.arange(sample_gridgraph.nb_active()))
    expected_coords = jnp.array([[0, 0], [0, 2], [1, 0], [1, 1]])
    assert jnp.array_equal(active_coords, expected_coords)

def test_adjacency_matrix():
    permeability_raster = jnp.ones((2,3))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities, permeability_raster)
    edge_weights = grid.adjacency_matrix()
    
    assert isinstance(edge_weights, BCOO)  # Ensure that the output is a sparse matrix
    assert edge_weights.shape == (6, 6)  # 2x3 grid flattened to 6 nodes
    
    G = grid_2d_graph(2, 3, create_using=nx.DiGraph)
    assert jnp.array_equal(nx.adjacency_matrix(G).toarray(), edge_weights.todense())
    

    G.remove_node((0,0))
    activities2 = activities.at[0,0].set(False)
    grid2 = GridGraph(activities2, permeability_raster)
    edge_weights2 = grid2.adjacency_matrix()
    assert jnp.array_equal(nx.adjacency_matrix(G).toarray(), edge_weights2.todense())

import jax.random as jr
def test_differentiability_adjacency_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    def objective(permeability_raster):
        grid = GridGraph(activities, permeability_raster)
        edge_weights = grid.adjacency_matrix()
        return edge_weights.sum()
        
    grad_objective = grad(objective)
    permeability_gradient = grad_objective(permeability_raster)
    assert permeability_gradient[0, 0] == 2
    assert permeability_gradient[0, 1] == 3
    assert permeability_gradient[1, 1] == 4

if __name__ == "__main__":
    pytest.main()
