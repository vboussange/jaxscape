import pytest
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph, ROOK_CONTIGUITY  # Replace 'your_module' with the actual module name
import networkx as nx
from networkx import grid_2d_graph
from jax import grad, jit
import jax.random as jr
import numpy as np

# TODO: implement test ROOK_CONTIGUITY vs QUEEN_CONTIGUITY

@pytest.fixture
def sample_gridgraph():
    activity = jnp.array([[True, False, True],
                          [True, True, False]])  # 2x3 grid with some inactive cells
    vertex_weights = jnp.ones((2, 3))  # Uniform weights for simplicity
    grid = GridGraph(activity, vertex_weights)
    sample_gridgraph = grid # to comment
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
    assert sample_gridgraph.nb_active == 4  # There are four active vertices


def test_all_active(sample_gridgraph):
    assert not sample_gridgraph.all_active()  # Not all vertices are active


def test_list_active_vertices(sample_gridgraph):
    active_vertices = sample_gridgraph.list_active_vertices()
    expected_active_vertices = jnp.array([0, 2, 3, 4])
    assert jnp.array_equal(active_vertices, expected_active_vertices)


def test_active_vertices_coordinates(sample_gridgraph):
    active_coords = sample_gridgraph.active_vertex_index_to_coord(jnp.arange(sample_gridgraph.nb_active))
    expected_coords = jnp.array([[0, 0], [0, 2], [1, 0], [1, 1]])
    assert jnp.array_equal(active_coords, expected_coords)
    
def test_node_values_to_raster():
    activities = jnp.array([
        [True, False, True],
        [False, True, False],
        [True, False, True]
    ])
    vertex_weights = jnp.ones_like(activities)
    grid = GridGraph(activities, vertex_weights)


    # Define values for the active nodes
    active_values = jnp.array([10, 20, 30, 40, 50])  # Should match the number of active nodes

    # Expected raster output based on the grid configuration and active values
    expected_raster = jnp.array([
        [10, np.nan, 20],
        [np.nan, 30, np.nan],
        [40, np.nan, 50]
    ])
    
    # # Generate expected raster dynamically
    # expected_raster = jnp.full((grid.height, grid.width), jnp.nan)
    # active_coords = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active()))
    # for coord, value in zip(active_coords, active_values):
    #     expected_raster = expected_raster.at[tuple(coord)].set(value)

    output_raster = grid.node_values_to_array(active_values)

    assert jnp.array_equal(jnp.isnan(output_raster), jnp.isnan(expected_raster)), "NaN positions mismatch"
    assert jnp.allclose(output_raster[~jnp.isnan(output_raster)], expected_raster[~jnp.isnan(expected_raster)]), "Values mismatch"


# test against networkx
def test_adjacency_matrix():
    permeability_raster = jnp.ones((2,3))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities, permeability_raster)
    edge_weights = grid.get_adjacency_matrix()
    
    assert isinstance(edge_weights, BCOO)  # Ensure that the output is a sparse matrix
    assert edge_weights.shape == (6, 6)  # 2x3 grid flattened to 6 nodes
    
    # trivial connectivity
    G = grid_2d_graph(2, 3, create_using=nx.DiGraph)
    assert jnp.array_equal(nx.adjacency_matrix(G).toarray(), edge_weights.todense())

    # not all nodes active
    G.remove_node((0,0))
    activities = activities.at[0,0].set(False)
    grid = GridGraph(activities, permeability_raster)
    adj_matrix = grid.get_adjacency_matrix().todense()
    adj_matrix_nx = nx.adjacency_matrix(G).todense()
    assert jnp.array_equal(adj_matrix, adj_matrix_nx)
    
    
    # not all nodes active, not trivial connectivity
    for edge in [((0, 1), (0, 2)), ((1,2), (0, 2))]:
        G[edge[0]][edge[1]]['weight'] = 5
    permeability_raster = permeability_raster.at[0,2].set(5)
    grid = GridGraph(activities, permeability_raster)
    adj_matrix = grid.get_adjacency_matrix().todense()
    adj_matrix_nx = nx.adjacency_matrix(G, weight='weight').todense()
    assert jnp.array_equal(adj_matrix, adj_matrix_nx)

def test_differentiability_adjacency_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    def objective(permeability_raster):
        grid = GridGraph(activities, permeability_raster)
        edge_weights = grid.get_adjacency_matrix()
        return edge_weights.sum()
        
    grad_objective = grad(objective)
    permeability_gradient = grad_objective(permeability_raster)
    assert permeability_gradient[0, 0] == 2
    assert permeability_gradient[0, 1] == 3
    assert permeability_gradient[1, 1] == 4
    
def test_ExplicitGridGraph():
    permeability_raster = jnp.ones((10, 10)) 
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    grid = GridGraph(activities=activities, 
                    vertex_weights=permeability_raster)
    A = grid.get_adjacency_matrix()
    permeability_raster = permeability_raster.at[1,1].set(0.)
    landscape = ExplicitGridGraph(activities=activities, 
                                vertex_weights = permeability_raster,
                                adjacency_matrix = A)
    func = landscape.equivalent_connected_habitat()
    assert func == jnp.sqrt((A.sum() - 8.)) # we lose 1 vertex connected to 4 neighbors, as its quality is set 0.