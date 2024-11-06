from typing import Callable, Tuple
import jax.numpy as jnp
from jax import jit
import numpy as np
import networkx as nx
from jax.experimental.sparse import BCOO
from functools import partial
import equinox as eqx
from jax import lax

# Neighboring indices in the grid
ROOK_CONTIGUITY = jnp.array([
                (1, 0),  # down
                (-1, 0),  # up
                (0, 1),  # right
                (0, -1),  # left
                ])

class GridGraph(eqx.Module):
    activities: jnp.ndarray
    vertex_weights: jnp.ndarray
    nb_active: int = eqx.field(static=True)
    
    def __init__(self,
                 activities,
                 vertex_weights):
        """
        Initializes the GridGraph object.
        """
        assert activities.shape == vertex_weights.shape
        assert activities.dtype == "bool"
        self.activities = activities
        self.vertex_weights = vertex_weights
        self.nb_active = int(activities.sum())

    @property
    def height(self):
        """Get the height of the grid (number of rows)."""
        return self.activities.shape[0]

    @property
    def width(self):
        """Get the width of the grid (number of columns)."""
        return self.activities.shape[1]

    @jit
    def coord_to_index(self, i, j):
        """Convert (i, j) grid coordinates to the associated passive vertex index."""
        num_columns = self.activities.shape[1]  # Get the number of columns in the grid
        return i * num_columns + j
    @jit
    def index_to_coord(self, v):
        """Convert passive vertex index `v` to (i, j) grid coordinates."""
        num_columns = self.activities.shape[1]  # Get the number of columns in the grid
        i = v // num_columns  # Row index
        j = v % num_columns   # Column index
        return (i, j)

    @jit
    def vertex_active(self, v):
        """Check if passive vertex index v is active."""
        return self.activities.ravel()[v]

    @jit
    def vertex_active_coord(self, i, j):
        """Check if vertex at (i, j) is active."""
        return self.activities[i, j]

    @jit
    def all_active(self):
        """Check if all vertices are active."""
        return self.nb_active == self.activities.size

    @jit
    def list_active_vertices(self):
        """Return a list of active vertices in passive vertex index."""
        rows, cols = jnp.nonzero(self.activities, size=self.nb_active)  # No `size` argument used here
        return self.coord_to_index(rows, cols)

    
    @jit
    def active_vertex_index_to_coord(self, v):
        """Get (i,j) coordinates of active vertex index `v`."""
        passive_index = self.list_active_vertices()[v]
        return jnp.column_stack(self.index_to_coord(passive_index))
    
    @jit
    def coord_to_active_vertex_index(self, i, j):
        """Get (i,j) coordinates of active vertex index `v`."""
        num_nodes = self.nb_active
        source_xy_coord = self.active_vertex_index_to_coord(jnp.arange(num_nodes))
        active_map = jnp.zeros_like(self.activities, dtype=int) - 1
        active_map = active_map.at[source_xy_coord[:,0], source_xy_coord[:,1]].set(jnp.arange(num_nodes))  # -1 if not an active vertex
        return active_map[i, j]
    
    @jit
    def node_values_to_raster(self, values):
        canvas = jnp.full((self.height, self.width), jnp.nan)
        vertices_coord = self.active_vertex_index_to_coord(jnp.arange(self.nb_active))
        canvas = canvas.at[vertices_coord[:,0], vertices_coord[:,1]].set(values)
        return canvas

    def __repr__(self):
        return f"GridGraph of size {self.height}x{self.width}"
    
    @jit
    def get_adjacency_matrix(self, neighbors=ROOK_CONTIGUITY):
        """
        Create a differentiable adjacency matrix based on the vertices and
        contiguity pattern of gridgraph.
        """
        # Get shape of raster
        activities = self.activities
        nrows, ncols = activities.shape
        num_nodes = self.nb_active
        permeability_raster = self.vertex_weights
        # Get coordinates of active nodes
        source_xy_coord = self.active_vertex_index_to_coord(jnp.arange(num_nodes))
        # Build a map from (i,j) to active node indices
        active_map = jnp.full_like(activities, fill_value=-1, dtype=int)
        active_map = active_map.at[source_xy_coord[:,0], source_xy_coord[:,1]].set(jnp.arange(num_nodes))

        num_neighbors = neighbors.shape[0]
        # Compute candidate target coordinates
        candidate_target_xy_coord = source_xy_coord[:, None, :] + neighbors[None, :, :]  # Shape (num_nodes, num_neighbors, 2)

        # Compute edge validity
        in_bounds = (
            (candidate_target_xy_coord[..., 0] >= 0) & (candidate_target_xy_coord[..., 0] < nrows) &
            (candidate_target_xy_coord[..., 1] >= 0) & (candidate_target_xy_coord[..., 1] < ncols)
        )

        # Check if target nodes are active
        candidate_i = candidate_target_xy_coord[..., 0].clip(0, nrows - 1)
        candidate_j = candidate_target_xy_coord[..., 1].clip(0, ncols - 1)
        target_active = activities[candidate_i, candidate_j]
        edge_validity = in_bounds & target_active

        # For invalid edges, set indices to 0
        candidate_target_xy_coord_valid = jnp.where(edge_validity[..., None], candidate_target_xy_coord, 0)

        # Get target node indices
        target_node_indices = active_map[candidate_target_xy_coord_valid[..., 0], candidate_target_xy_coord_valid[..., 1]]
        target_node_indices = jnp.where(edge_validity, target_node_indices, 0)

        # Source node indices
        source_node_indices = jnp.broadcast_to(jnp.arange(num_nodes)[:, None], (num_nodes, num_neighbors))
        source_node_indices = jnp.where(edge_validity, source_node_indices, 0)

        # Get values (edge weights)
        values = permeability_raster[candidate_target_xy_coord_valid[..., 0], candidate_target_xy_coord_valid[..., 1]]
        values = jnp.where(edge_validity, values, 0.0)

        # Flatten arrays
        source_node_indices = source_node_indices.ravel()
        target_node_indices = target_node_indices.ravel()
        values = values.ravel()

        # Construct indices array
        row_col_indices = jnp.stack([source_node_indices, target_node_indices], axis=1)

        # Create BCOO matrix
        A = BCOO((values, row_col_indices), shape=(num_nodes, num_nodes))

        return A