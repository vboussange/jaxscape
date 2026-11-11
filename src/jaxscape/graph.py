import abc
from collections.abc import Callable
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, errors as jax_errors
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


@eqx.filter_jit
def _grid_edges(
    shape: tuple[int, ...], neighbors: Array
) -> tuple[Array, Array, Array, Array]:
    """The edges of the grid graphs of `shape` and contiguity `neighbors`.

    Vertex `v` has `k` stored entries, one per neighbour, in rows `v * k` to
    `v * k + k - 1` of the sparse indices. Entries of neighbours off the raster
    are stored at row and column 0, and the coordinates of their target are
    those of pixel `(0, 0)`.

    **Returns:**

    The coordinates of each vertex, with shape `(n, 2)`, those of the target of
    each of its edges, with shape `(n, k, 2)`, whether that target lies on the
    raster, with shape `(n, k)`, and the sparse indices of the adjacency matrix,
    with shape `(n * k, 2)`.
    """
    nrows, ncols = shape
    num_nodes = nrows * ncols
    # Coordinates of the vertices
    vertices = jnp.arange(num_nodes)
    source_xy_coord = jnp.column_stack((vertices // ncols, vertices % ncols))

    num_neighbors = neighbors.shape[0]

    # Compute candidate target coordinates
    candidate_target_xy_coord = (
        source_xy_coord[:, None, :] + neighbors[None, :, :]
    )  # Shape (num_nodes, num_neighbors, 2)

    # Compute edge validity
    inside = (
        (candidate_target_xy_coord[..., 0] >= 0)
        & (candidate_target_xy_coord[..., 0] < nrows)
        & (candidate_target_xy_coord[..., 1] >= 0)
        & (candidate_target_xy_coord[..., 1] < ncols)
    )

    # For invalid edges, set indices to 0
    target_xy_coord = jnp.where(inside[..., None], candidate_target_xy_coord, 0)

    # Get target node indices
    target_node_indices = target_xy_coord[..., 0] * ncols + target_xy_coord[..., 1]
    target_node_indices = jnp.where(inside, target_node_indices, 0)

    # Source node indices
    source_node_indices = jnp.broadcast_to(
        jnp.arange(num_nodes)[:, None], (num_nodes, num_neighbors)
    )
    source_node_indices = jnp.where(inside, source_node_indices, 0)

    # Construct indices array
    row_col_indices = jnp.stack(
        [source_node_indices.ravel(), target_node_indices.ravel()], axis=1
    )
    return source_xy_coord, target_xy_coord, inside, row_col_indices


@final
class GridTopology(eqx.Module):
    """The edges of the grid graphs of one raster shape and contiguity.

    It holds what their adjacency matrices share whatever the conductances:
    the source and target of each edge, which set the sparsity pattern of the
    matrices. Build it once and pass it to each `GridGraph` of that shape and
    contiguity, as in a loop over conductance surfaces: the graph then computes
    its edge weights alone, and its adjacency matrix equals that of the same
    `GridGraph` without topology. Build it outside `jax.jit` and pass it in, so
    that compiled programs read its edges.

    **Arguments:**

    - `shape`: Raster shape `(height, width)`.
    - `neighbors`: Contiguity pattern, either `ROOK_CONTIGUITY` or
      `QUEEN_CONTIGUITY`, or another array of `(row, column)` offsets with
      shape `(k, 2)`. It must be concrete, outside of a trace.

    !!! example

        ```python
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        from jaxscape import GridGraph, GridTopology, ResistanceDistance

        def fun(x, y):
            return (x + y) / 2

        topology = GridTopology((50, 50))
        surfaces = jnp.exp(jax.random.normal(jax.random.key(0), (4, 50, 50)))
        sites = jnp.array([0, 1275, 2499])

        distance = ResistanceDistance()
        state = distance.init(GridGraph(surfaces[0], fun=fun, topology=topology))

        @eqx.filter_jit
        def resistances(surface, topology, state):
            graph = GridGraph(surface, fun=fun, topology=topology)
            state = distance.reinit(graph, state)
            return distance.nodes_to_nodes_distance(graph, sites, state), state

        for surface in surfaces:
            resistance, state = resistances(surface, topology, state)
        ```
    """

    indices: Array
    inside: Array
    shape: tuple[int, int] = eqx.field(static=True)
    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)

    def __init__(
        self, shape: tuple[int, ...], neighbors: Array = ROOK_CONTIGUITY
    ) -> None:
        try:
            offsets = np.asarray(neighbors, dtype=np.int64)
        except jax_errors.TracerArrayConversionError:
            raise ValueError(
                "GridTopology needs concrete `neighbors`: build it outside "
                "`jax.jit` and pass it in."
            ) from None
        if offsets.ndim != 2 or offsets.shape[0] == 0 or offsets.shape[1] != 2:
            raise ValueError("`neighbors` must have shape `(k, 2)`, with `k >= 1`.")
        self.shape = (int(shape[0]), int(shape[1]))
        self.offsets = tuple((int(i), int(j)) for i, j in offsets)
        _, _, self.inside, self.indices = _grid_edges(
            self.shape, jnp.asarray(neighbors)
        )

    def _adjacency_matrix(
        self, grid: Array, fun: Callable[[Array, Array], Array]
    ) -> BCOO:
        """The adjacency matrix of the grid graph of `grid` and `fun`.

        Each edge weight comes from the values of `grid` at its source and
        target, as in `GridGraph`. Shifted copies of `grid` give the values of
        the targets; edges off the raster take the value of pixel `(0, 0)` and
        weigh 0.
        """
        height, width = self.shape
        num_nodes = height * width
        reach = max(max(abs(i), abs(j)) for i, j in self.offsets)
        padded = jnp.pad(grid, reach, constant_values=grid[0, 0])
        targets = jnp.stack(
            [
                padded[reach + i : reach + i + height, reach + j : reach + j + width]
                for i, j in self.offsets
            ],
            axis=-1,
        )
        values = fun(grid.reshape(num_nodes, 1), targets.reshape(num_nodes, -1))
        values = jnp.where(self.inside, values, 0.0)
        return BCOO((values.ravel(), self.indices), shape=(num_nodes, num_nodes))


class GridGraph(AbstractGraph):
    """
    Grid graph where vertices are defined by a rectangular `grid`.

    **Arguments:**

    - `grid`: 2D array with shape `(height, width)` used to define edge weights.
      When calculating distances, edge weights are assumed to represent
      permeability (1/resistance; higher values indicate easier movement).
    - `fun`: Function applied to the source and target node values to define
      the edge weight. Takes two arrays and returns an array of the same size.
      Defaults to the target vertex weight (`fun = lambda x, y: y`).
    - `neighbors`: Contiguity pattern, either `ROOK_CONTIGUITY` or
      `QUEEN_CONTIGUITY`. Defaults to `ROOK_CONTIGUITY`, or to the contiguity of
      `topology`; pass one or the other.
    - `topology`: Optional `GridTopology` of the shape of `grid`, built once
      for the graphs of one raster shape and contiguity, as in a loop over
      conductance surfaces. The adjacency matrix then reads its edges and
      computes the edge weights alone; it is the same with and without a
      topology.
    """

    grid: Array
    # The contiguity of a graph without topology; a topology holds its own.
    _neighbors: Array | None
    fun: Callable[[Array, Array], Array] = eqx.field(static=True)
    topology: GridTopology | None

    def __init__(
        self,
        grid: Array,
        fun: Callable[[Array, Array], Array] = lambda x, y: y,
        neighbors: Array | None = None,
        topology: GridTopology | None = None,
    ):
        assert grid.ndim == 2, "`grid` should be 2D array"
        if topology is not None:
            if neighbors is not None:
                raise ValueError(
                    "Pass the contiguity of a graph with a topology to the "
                    "topology alone: `GridTopology(shape, neighbors)`."
                )
            if tuple(grid.shape) != topology.shape:
                raise ValueError(
                    f"The topology has shape {topology.shape} and `grid` "
                    f"{tuple(grid.shape)}: build a `GridTopology` of the shape "
                    "of the raster."
                )
        elif neighbors is None:
            neighbors = ROOK_CONTIGUITY
        self.grid = grid
        self._neighbors = neighbors
        self.fun = fun
        self.topology = topology

    def __repr__(self) -> str:
        return f"GridGraph of size {self.height}x{self.width}"

    @property
    def neighbors(self) -> Array:
        """The contiguity pattern, as `(row, column)` offsets with shape `(k, 2)`."""
        if self.topology is not None:
            return jnp.asarray(self.topology.offsets)
        assert self._neighbors is not None
        return self._neighbors

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

        With a `topology`, it reads the edges from the topology and computes
        the edge weights alone.
        """
        if self.topology is not None:
            return self.topology._adjacency_matrix(self.grid, self.fun)
        num_nodes = self.nv
        permeability_raster = self.grid
        source_xy_coord, target_xy_coord, inside, row_col_indices = _grid_edges(
            self.grid.shape, self.neighbors
        )

        # Get values (edge weights)
        values = self.fun(
            permeability_raster[
                source_xy_coord[:, None, 0], source_xy_coord[:, None, 1]
            ],
            permeability_raster[target_xy_coord[..., 0], target_xy_coord[..., 1]],
        )
        values = jnp.where(inside, values, 0.0)

        # Create BCOO matrix
        A = BCOO(
            (values.ravel(), row_col_indices), shape=(num_nodes, num_nodes)
        )  # not compatible with jit .sum_duplicates()

        return A
