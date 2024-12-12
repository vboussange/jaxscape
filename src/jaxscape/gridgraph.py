import jax.numpy as jnp
from jax import jit
from jax.experimental.sparse import BCOO
import equinox as eqx
import abc

# Neighboring indices in the grid
ROOK_CONTIGUITY = jnp.array([
                (1, 0),  # down
                (-1, 0),  # up
                (0, 1),  # right
                (0, -1),  # left
                ])

# Neighboring indices in the grid
QUEEN_CONTIGUITY = jnp.array([
                (1, 0),    # down
                (-1, 0),   # up
                (0, 1),    # right
                (0, -1),   # left
                (1, 1),    # down-right
                (1, -1),   # down-left
                (-1, 1),   # up-right
                (-1, -1)   # up-left
                ])
    

class GridGraph(eqx.Module):
    activities: jnp.ndarray
    vertex_weights: jnp.ndarray
    nb_active: int = eqx.field(static=True)
    
    def __init__(self,
                 activities,
                 vertex_weights,
                 nb_active=None):
        """
        Initializes a GridGraph object.
        `activities` is a boolean array of shape (height, width) indicating which vertices should be included in the graph.
        `vertex_weights` is a 2D array of shape (height, width) containing the weights of each vertex.
        `nb_active` is the number of active vertices in the graph, and should be defined in a jit context.
        """
        assert activities.shape == vertex_weights.shape
        assert activities.dtype == "bool"
        # assert activities.sum() == nb_active
        self.activities = activities
        self.vertex_weights = vertex_weights
        if nb_active is None:
            self.nb_active = int(activities.sum())
        else:
            self.nb_active = nb_active
            
    def __repr__(self):
        return f"GridGraph of size {self.height}x{self.width}"
    
    @property
    def height(self):
        """Get the height of the grid (number of rows)."""
        return self.activities.shape[0]

    @property
    def width(self):
        """Get the width of the grid (number of columns)."""
        return self.activities.shape[1]

    # TODO: `coord_to_index` and `index_to_coord` are not used and could be removed
    # @jit
    def coord_to_index(self, i, j):
        """Convert (i, j) grid coordinates to the associated passive vertex index."""
        num_columns = self.activities.shape[1]  # Get the number of columns in the grid
        return i * num_columns + j
    # @jit
    def index_to_coord(self, v):
        """Convert passive vertex index `v` to (i, j) grid coordinates."""
        num_columns = self.activities.shape[1]  # Get the number of columns in the grid
        i = v // num_columns  # Row index
        j = v % num_columns   # Column index
        return (i, j)

    # @jit
    def vertex_active(self, v):
        """Check if passive vertex index v is active."""
        return self.activities.ravel()[v]

    # @jit
    def vertex_active_coord(self, i, j):
        """Check if vertex at (i, j) is active."""
        return self.activities[i, j]

    # @jit
    def all_active(self):
        """Check if all vertices are active."""
        return self.nb_active == self.activities.size

    # @jit
    def list_active_vertices(self):
        """Return a list of active vertices in passive vertex index."""
        rows, cols = jnp.nonzero(self.activities, size=self.nb_active)  # No `size` argument used here
        return self.coord_to_index(rows, cols)

    # @jit
    def active_vertex_index_to_coord(self, v):
        """Get (i,j) coordinates of active vertex index `v`."""
        passive_index = self.list_active_vertices()[v]
        return jnp.column_stack(self.index_to_coord(passive_index))
    
    # @jit
    def node_values_to_array(self, values):
        """Reshapes the 1D array values of active vertices to the underlying 2D grid."""
        canvas = jnp.full((self.height, self.width), jnp.nan)
        vertices_coord = self.active_vertex_index_to_coord(jnp.arange(self.nb_active))
        canvas = canvas.at[vertices_coord[:,0], vertices_coord[:,1]].set(values)
        return canvas
    
    def array_to_node_values(self, array):
        """Reshapes the 1D array values of active vertices to the underlying 2D grid."""
        active_ij = self.active_vertex_index_to_coord(jnp.arange(self.nb_active))
        q = array[active_ij[:,0], active_ij[:,1]]
        return q
    
    # @jit
    def get_active_vertices_weights(self):
        """Get a 1D array of active vertices weights."""
        return self.array_to_node_values(self.vertex_weights)
    
    # @jit
    def coord_to_active_vertex_index(self, i, j):
        """Get (i,j) coordinates of active vertex index `v`."""
        # TODO: error raising is not compatible with jit
        # if ~jnp.all(self.activities[i,j]):
        #     raise IndexError(f"Vertices at i = {i}, j = {j} is not active")
        num_nodes = self.nb_active
        source_xy_coord = self.active_vertex_index_to_coord(jnp.arange(num_nodes))
        active_map = jnp.zeros_like(self.activities, dtype=int) - 1
        active_map = active_map.at[source_xy_coord[:,0], source_xy_coord[:,1]].set(jnp.arange(num_nodes))  # -1 if not an active vertex
        return active_map[i, j]
    
    @eqx.filter_jit
    def get_adjacency_matrix(self, fun = lambda x, y: y, neighbors=ROOK_CONTIGUITY):
        """
        Create an adjacency matrix from the vertices weights of the `GridGraph`
        object. `fun` is a function applied to define the edge weigh based the
        source and target vertex weights. It takes two arrays and returns an
        array of the same size. Defaults to assigning the target vertex weight.
        `neighbors` defines the contiguity pattern, and can be either
        `ROOK_CONTIGUITY` or `QUEEN_CONTIGUITY`.
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
        target_xy_coord = jnp.where(edge_validity[..., None], candidate_target_xy_coord, 0)

        # Get target node indices
        target_node_indices = active_map[target_xy_coord[..., 0], target_xy_coord[..., 1]]
        target_node_indices = jnp.where(edge_validity, target_node_indices, 0)

        # Source node indices
        source_node_indices = jnp.broadcast_to(jnp.arange(num_nodes)[:, None], (num_nodes, num_neighbors))
        source_node_indices = jnp.where(edge_validity, source_node_indices, 0)

        # Get values (edge weights)
        values = fun(permeability_raster[source_xy_coord[:, None, 0], source_xy_coord[:, None, 1]],
                     permeability_raster[target_xy_coord[..., 0], target_xy_coord[..., 1]])
        values = jnp.where(edge_validity, values, 0.0)

        # Flatten arrays
        source_node_indices = source_node_indices.ravel()
        target_node_indices = target_node_indices.ravel()
        values = values.ravel()

        # Construct indices array
        row_col_indices = jnp.stack([source_node_indices, target_node_indices], axis=1)

        # Create BCOO matrix
        A = BCOO((values, row_col_indices), shape=(num_nodes, num_nodes)) # not compatible with jit .sum_duplicates()

        return A
    
    # @jit
    def equivalent_connected_habitat(self):
        q = self.get_active_vertices_weights()
        K = self.get_adjacency_matrix()
        return jnp.sqrt(q @ (K @ q))
    
# We need to avoid calling super().__init__, and rather define an AbstractGridGraph object, see https://docs.kidger.site/equinox/pattern/
# class AbstractGridGraph(eqx.Module):
#     @abc.abstractmethod
#     def init(self, activities, vertex_weights, nb_active):
#         raise NotImplementedError

class ExplicitGridGraph(GridGraph):
    adjacency_matrix: jnp.ndarray

    def __init__(self, *args, adjacency_matrix, **kwargs):
        """
        A grid graph with adjacency matrix explicitly defined.
        """
        # assert int(activities.sum()) == proximity.shape[0], "The number of nodes in the graph defined by `proximity` should correspond to the number of active vertices defined in `activities`"
        super().__init__(*args, **kwargs)
        self.adjacency_matrix = adjacency_matrix
        
    def get_adjacency_matrix(self):
        return self.adjacency_matrix