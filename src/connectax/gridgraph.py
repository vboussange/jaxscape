from typing import Callable, Tuple
import jax.numpy as jnp
from jax import jit
import numpy as np
import networkx as nx
from jax.experimental.sparse import BCOO

# Neighboring indices in the grid
ROOK_CONTIGUITY = [
                jnp.array((1, 0)),  # down
                jnp.array((-1, 0)),  # up
                jnp.array((0, 1)),  # right
                jnp.array((0, -1)),  # left
                ]

class GridGraph():
    def __init__(self,
                 activities,
                 vertex_weights,
                 adjacency_matrix = None,
                 contiguity = ROOK_CONTIGUITY):
        """
        Initializes the GridGraph object.
        """
        assert activities.shape == vertex_weights.shape
        assert activities.dtype == "bool"
        self.activities = activities
        self.vertex_weights = vertex_weights
        self.neighbors = contiguity
        self._adjacency_matrix = adjacency_matrix

    @property
    def height(self):
        """Get the height of the grid (number of rows)."""
        return self.activities.shape[0]

    @property
    def width(self):
        """Get the width of the grid (number of columns)."""
        return self.activities.shape[1]
        
    def adjacency_matrix(self):
        if self._adjacency_matrix is None:
            self._adjacency_matrix = build_adjacency_matrix(self)
        return self._adjacency_matrix

    # TODO: index do not correspond to the entries in the adjacency matrix
    # the adjacency matrix considers only active nodes
    # this is misleading - we should consider an alternative
    def coord_to_index(self, i, j):
        """Convert (i, j) grid coordinates to the associated passive vertex index."""
        return jnp.ravel_multi_index((i,j), self.activities.shape)

    def index_to_coord(self, v):
        """Convert passive vertex index `v` to (i, j) grid coordinates."""
        return jnp.unravel_index(v, self.activities.shape)

    def vertex_active(self, v):
        """Check if passive vertex index v is active."""
        return self.activities.ravel()[v]

    def vertex_active_coord(self, i, j):
        """Check if vertex at (i, j) is active."""
        return self.activities[i, j]

    def nb_active(self):
        """Count the number of active vertices."""
        return self.activities.sum()

    def all_active(self):
        """Check if all vertices are active."""
        return self.nb_active() == self.activities.size

    def list_active_vertices(self):
        """Return a list of active vertices in passive vertex index."""
        return jnp.nonzero(self.activities.ravel())[0]

    def active_vertex_index_to_coord(self, v):
        """Get (i,j) coordinates of active vertex index `v`."""
        passive_index = self.list_active_vertices()[v]
        return jnp.column_stack(self.index_to_coord(passive_index))
    
    def coord_to_active_vertex_index(self, i, j):
        """Get (i,j) coordinates of active vertex index `v`."""
        num_nodes = self.nb_active()
        source_xy_coord = self.active_vertex_index_to_coord(jnp.arange(num_nodes))
        active_map = jnp.zeros_like(self.activities, dtype=int) - 1
        active_map = active_map.at[source_xy_coord[:,0], source_xy_coord[:,1]].set(jnp.arange(num_nodes))  # -1 if not an active vertex
        return active_map[i, j]
    
    def node_values_to_raster(self, values):
        canvas = jnp.full((self.height, self.width), jnp.nan)
        vertices_coord = self.active_vertex_index_to_coord(jnp.arange(self.nb_active()))
        canvas = canvas.at[vertices_coord[:,0], vertices_coord[:,1]].set(values)
        return canvas

    def __repr__(self):
        return f"GridGraph of size {self.height}x{self.width}"
    
    
def build_adjacency_matrix(gridgraph):
    """
    Create a differentiable adjacency matrix based on the vertices and
    contiguity pattern of `gridgraph`.
    """
    assert isinstance(gridgraph, GridGraph)
    # Get shape of raster
    activities = gridgraph.activities
    nrows, ncols = activities.shape
    num_nodes = gridgraph.nb_active()
    neighbors = gridgraph.neighbors
    permeability_raster = gridgraph.vertex_weights
    
    source_xy_coord = gridgraph.active_vertex_index_to_coord(jnp.arange(num_nodes))
    active_map = jnp.zeros_like(activities, dtype=int) - 1
    active_map = active_map.at[source_xy_coord[:,0], source_xy_coord[:,1]].set(jnp.arange(num_nodes))  # -1 if not an active vertex
    
    rows, cols, values = [], [], []
    
    for neighb in neighbors:
        candidate_target_xy_coord = source_xy_coord + neighb
        
        # filter edges with out out-of-bound indices
        edge_validity = (candidate_target_xy_coord[:, 0] >= 0) & (candidate_target_xy_coord[:, 0] < nrows) & (candidate_target_xy_coord[:, 1] >= 0) & (candidate_target_xy_coord[:, 1] < ncols)

        # filter out edges with inactive target pixels
        edge_validity = edge_validity * activities[candidate_target_xy_coord[:,0], candidate_target_xy_coord[:,1]]
        
        source_node_coord = active_map[source_xy_coord[:, 0][edge_validity], source_xy_coord[:, 1][edge_validity]]
        target_node_coord = active_map[candidate_target_xy_coord[:, 0][edge_validity], candidate_target_xy_coord[:, 1][edge_validity]]
        
        # Connectivity values based on permeability
        value = permeability_raster[candidate_target_xy_coord[:, 0][edge_validity], candidate_target_xy_coord[:, 1][edge_validity]]

        values.extend(value)
        rows.extend(source_node_coord)
        cols.extend(target_node_coord)

    # Stack results into arrays for COO format
    data = jnp.array(values)
    row_col_indices = jnp.vstack([jnp.array(rows), jnp.array(cols)]).T

    A = BCOO((data, row_col_indices), shape=(num_nodes, num_nodes))

    return A