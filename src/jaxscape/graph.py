import abc
from collections.abc import Callable

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.experimental.sparse import BCOO


class AbstractGraph(eqx.Module):
    """
    Abstract base class for graphs.
    """

    @property
    @abc.abstractmethod
    def nv(self) -> int:
        """Get the number of vertices."""
        pass

    @abc.abstractmethod
    def get_adjacency_matrix(self) -> BCOO:
        """Get the adjacency matrix of the graph."""
        pass


class Graph(AbstractGraph):
    """
    A simple graph defined by an adjacency matrix.

    **Arguments:**

    - `adjacency_matrix`: A `jax.experimental.sparse.BCOO` adjacency matrix.
    """

    adjacency_matrix: BCOO

    def __init__(self, adjacency_matrix: BCOO):
        self.adjacency_matrix = adjacency_matrix

    @property
    def nv(self) -> int:
        return self.adjacency_matrix.shape[0]

    def get_adjacency_matrix(self) -> BCOO:
        return self.adjacency_matrix


# Neighboring indices for GridGraph
ROOK_CONTIGUITY = jnp.array(
    [
        (1, 0),  # down
        (-1, 0),  # up
        (0, 1),  # right
        (0, -1),  # left
    ]
)

# Neighboring indices for GridGraph
QUEEN_CONTIGUITY = jnp.array(
    [
        (1, 0),  # down
        (-1, 0),  # up
        (0, 1),  # right
        (0, -1),  # left
        (1, 1),  # down-right
        (1, -1),  # down-left
        (-1, 1),  # up-right
        (-1, -1),  # up-left
    ]
)


class GridGraph(AbstractGraph):
    """
    Grid graph where vertices are defined by a rectangular `grid`.

    **Arguments:**

    - `grid` is a 2D array of shape `(height, width)` used to define edge weights. When calculating distances, edge weights are assume to represent permeability (i.e., 1/resistance, higher values indicate easier movement).
    - `fun` is a function applied to the source and target node values to define the edge weight. It takes two arrays and returns an
    array of the same size. Defaults to assigning the target vertex weight (`fun = lambda x, y: y`).
    - `neighbors` defines the contiguity pattern, and can be either `ROOK_CONTIGUITY` or `QUEEN_CONTIGUITY`.
    """

    grid: Array
    neighbors: Array
    fun: Callable[[Array, Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        grid: Array,
        fun: Callable[[Array, Array], Array] = lambda x, y: y,
        neighbors: Array = ROOK_CONTIGUITY,
    ):
        assert grid.ndim == 2, "`grid` should be 2D array"
        self.grid = grid
        self.fun = fun
        self.neighbors = neighbors

    def __repr__(self) -> str:
        return f"GridGraph of size {self.height}x{self.width}"

    @property
    def height(self) -> int:
        """Get the height of the grid (number of rows)."""
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        """Get the width of the grid (number of columns)."""
        return self.grid.shape[1]

    @property
    def nv(self) -> int:
        """Get the number of vertices."""
        return self.width * self.height

    # @jit
    def coord_to_index(self, i: Array, j: Array) -> Array:
        """Convert (i, j) grid coordinates to the associated passive vertex index."""
        num_columns = self.grid.shape[1]  # Get the number of columns in the grid
        return i * num_columns + j

    # @jit
    def index_to_coord(self, v: Array) -> Array:
        """Convert passive vertex index `v` to (i, j) grid coordinates."""
        num_columns = self.grid.shape[1]  # Get the number of columns in the grid
        i = v // num_columns  # Row index
        j = v % num_columns  # Column index
        return jnp.column_stack((i, j))

    # @jit
    def node_values_to_array(self, values: Array) -> Array:
        """Reshapes the 1D array values of vertices to the underlying 2D grid."""
        canvas = values.reshape(*self.grid.shape)
        return canvas

    def array_to_node_values(self, array: Array) -> Array:
        """Reshapes the 1D array values of vertices to the underlying 2D grid."""
        q = array.ravel()
        return q

    @eqx.filter_jit
    def get_adjacency_matrix(self) -> BCOO:
        """
        Create an adjacency matrix from the vertices weights of the `GridGraph`
        object.
        """
        # Get shape of raster
        nrows, ncols = self.grid.shape
        num_nodes = self.nv
        permeability_raster = self.grid
        # Get coordinates of active nodes
        source_xy_coord = self.index_to_coord(jnp.arange(num_nodes))

        num_neighbors = self.neighbors.shape[0]

        # Compute candidate target coordinates
        candidate_target_xy_coord = (
            source_xy_coord[:, None, :] + self.neighbors[None, :, :]
        )  # Shape (num_nodes, num_neighbors, 2)

        # Compute edge validity
        in_bounds = (
            (candidate_target_xy_coord[..., 0] >= 0)
            & (candidate_target_xy_coord[..., 0] < nrows)
            & (candidate_target_xy_coord[..., 1] >= 0)
            & (candidate_target_xy_coord[..., 1] < ncols)
        )

        # For invalid edges, set indices to 0
        target_xy_coord = jnp.where(in_bounds[..., None], candidate_target_xy_coord, 0)

        # Get target node indices
        target_node_indices = self.coord_to_index(
            target_xy_coord[..., 0], target_xy_coord[..., 1]
        )
        target_node_indices = jnp.where(in_bounds, target_node_indices, 0)

        # Source node indices
        source_node_indices = jnp.broadcast_to(
            jnp.arange(num_nodes)[:, None], (num_nodes, num_neighbors)
        )
        source_node_indices = jnp.where(in_bounds, source_node_indices, 0)

        # Get values (edge weights)
        values = self.fun(
            permeability_raster[
                source_xy_coord[:, None, 0], source_xy_coord[:, None, 1]
            ],
            permeability_raster[target_xy_coord[..., 0], target_xy_coord[..., 1]],
        )
        values = jnp.where(in_bounds, values, 0.0)

        # Flatten arrays
        source_node_indices = source_node_indices.ravel()
        target_node_indices = target_node_indices.ravel()
        values = values.ravel()

        # Construct indices array
        row_col_indices = jnp.stack([source_node_indices, target_node_indices], axis=1)

        # Create BCOO matrix
        A = BCOO(
            (values, row_col_indices), shape=(num_nodes, num_nodes)
        )  # not compatible with jit .sum_duplicates()

        return A
