import warnings
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.core import Tracer
from jax.experimental.sparse import BCOO

from .graph import AbstractGraph, GridGraph, QUEEN_CONTIGUITY, ROOK_CONTIGUITY
from .solvers.operator import _solver_state


class AbstractDistance(eqx.Module):
    """Abstract base class for distance computations on graphs.

    Provides a unified interface for computing distances with automatic handling of
    coordinate-based (for `GridGraph`) or index-based node specification.

    **Arguments:**

    - `graph`: Graph on which to compute distances.
    - `sources`: Source nodes as vertex indices (1D array) or coordinates
      (Nx2 array for `GridGraph`).
    - `targets`: Target nodes as vertex indices (1D array) or coordinates
      (Nx2 array for `GridGraph`).
    - `nodes`: Nodes for pairwise distances as vertex indices (1D) or
      coordinates (Nx2).
    - `state`: Optional solver state. Defaults to automatic initialization.

    Specify either: `nodes` alone, `sources` and/or `targets`, or neither
    (for all-pairs).

    **Attributes:**

    - `sources_share_work`: Whether a query from several sources costs much less
      than one query per source, as when the linear solves of a query serve
      every source with one solver state, such as a factorization or a
      multigrid hierarchy. Windowed analyses then compute the distances of
      many sources on one tile graph. Defaults to `False`.
    - `symmetric`: Whether the distance from `i` to `j` equals the distance
      from `j` to `i` on graphs whose adjacency matrix is symmetric, as those
      of windowed analyses are. Their quality sensitivity then needs no
      gradient. Defaults to `False`.
    - `averages_sources`: Whether window queries take `source_weights` and
      return the mean of the distances from the pixels around each source,
      with these weights, at the cost of one query. Coarsened windowed
      analyses then take the permeability sensitivity of a block centre from
      its block source. Defaults to `False`.

    **Returns:**

    Distance array with shape depending on the inputs.

    !!! example

        ```python
        from jaxscape import LCPDistance, GridGraph
        import jax.numpy as jnp

        distance = LCPDistance()
        permeability = jnp.ones((12, 12))
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)

        # All-pairs distance
        D = distance(grid)  # Shape: (n_nodes, n_nodes)

        # Using vertex indices
        D = distance(
            grid,
            sources=jnp.array([0, 1]),
            targets=jnp.array([10, 20]),
        )

        # Using coordinates (for GridGraph)
        D = distance(
            grid,
            sources=jnp.array([[0, 0], [1, 1]]),
            targets=jnp.array([[10, 10]]),
        )

        # Pairwise among subset
        D = distance(grid, nodes=jnp.array([0, 5, 10]))  # Shape: (3, 3)
        ```
    """

    sources_share_work: ClassVar[bool] = False
    symmetric: ClassVar[bool] = False
    averages_sources: ClassVar[bool] = False

    def __call__(
        self,
        graph: AbstractGraph,
        sources: Array | None = None,
        targets: Array | None = None,
        nodes: Array | None = None,
        state: Any = None,
    ) -> Array:
        if state is None:
            # Solver state is not part of the differentiable solve.
            # Stop gradients here so auto-initialized state matches Lineax's
            # `state=None` semantics and never carries tangents into custom VJPs.
            state = self.init(jax.tree.map(jax.lax.stop_gradient, graph))

        if nodes is not None:
            assert sources is None and targets is None, (
                "Specify either `nodes`, `sources`, `targets`, or both "
                "`sources` and `targets`."
            )
            if nodes.ndim == 2:
                if not isinstance(graph, GridGraph):
                    raise ValueError(
                        "Coordinate input (2D array) requires a GridGraph. "
                        "Use vertex indices (1D array) for Graph objects."
                    )
                nodes = graph.coord_to_index(nodes[:, 0], nodes[:, 1])
            return self.nodes_to_nodes_distance(graph, nodes, state)

        elif sources is not None or targets is not None:
            sources = jnp.arange(graph.nv) if sources is None else sources
            targets = jnp.arange(graph.nv) if targets is None else targets

            if sources.ndim == 2:
                if not isinstance(graph, GridGraph):
                    raise ValueError(
                        "Coordinate input (2D array) for sources requires a GridGraph. "
                        "Use vertex indices (1D array) for Graph objects."
                    )
                sources = graph.coord_to_index(sources[:, 0], sources[:, 1])

            if targets.ndim == 2:
                if not isinstance(graph, GridGraph):
                    raise ValueError(
                        "Coordinate input (2D array) for targets requires a GridGraph. "
                        "Use vertex indices (1D array) for Graph objects."
                    )
                targets = graph.coord_to_index(targets[:, 0], targets[:, 1])

            return self.sources_to_targets_distance(graph, sources, targets, state)

        else:
            return self.all_pairs_distance(graph, state)

    @abstractmethod
    def init(self, graph: AbstractGraph) -> Any:
        pass

    def exact(self) -> "AbstractDistance":
        """Return the distance whose values the queries of this one estimate.

        Distances that queries compute exactly return themselves. Their solver
        states serve each other.

        **Returns:**

        The distance that this one estimates, or this one.
        """
        return self

    def reseeded(self, index: Array) -> "AbstractDistance":
        """Return the distance whose random draws depend on `index` too.

        The estimates of randomized distances for different indices are then
        independent. Distances that queries compute exactly return themselves.

        **Arguments:**

        - `index`: Integer, which may be traced.

        **Returns:**

        A distance of the same kind.
        """
        del index
        return self

    def debiased_proximity(
        self, proximity: Callable[[Array], Array], graph: AbstractGraph
    ) -> Callable[[Array], Array]:
        """Return the proximity to apply to the distances of queries on `graph`.

        Applied to randomized distances, it has the expectation of `proximity`
        applied to exact distances. Distances that queries compute exactly
        return `proximity`.

        **Arguments:**

        - `proximity`: Function mapping distances to proximities.
        - `graph`: Graph of the queries.

        **Returns:**

        Function mapping the distances of queries on `graph` to proximities.
        """
        del graph
        return proximity

    def reinit(self, graph: AbstractGraph, state: Any) -> Any:
        """Return the state for `graph`, reusing what the solver can from `state`.

        Queries on `graph` take the returned state, and so does the next `reinit`.

        **Arguments:**

        - `graph`: Graph of the next queries.
        - `state`: State from `init` or an earlier `reinit`.

        **Returns:**

        State for `graph`, without gradient.
        """
        del state
        return self.init(graph)

    def sources_to_windows_distance(
        self, graph: GridGraph, sources: Array, size: int, state: Any = None
    ) -> Array:
        """Return the distances from each source to the pixels of its window.

        The window of a source is the square of `size` pixels a side centred on
        it, and it must lie inside the grid: the query raises otherwise.
        Windowed analyses query the distances of their block centres through
        this method, and the profile of their default proximity through a
        sources-to-targets query. By default, it computes the distances from
        the sources to every vertex and slices their windows: it holds
        `m * n` distances for `m` sources and `n` vertices, besides the memory
        of the query. A distance may override it to hold fewer values, as
        `ResistanceDistance` does with `SpielmanApproximation`, or for speed,
        as `LCPDistance` does on grid graphs.

        **Arguments:**

        - `graph`: Grid graph on which to compute distances.
        - `sources`: Source vertex indices with shape `(m,)`.
        - `size`: Odd number of pixels along a side of the windows.
        - `state`: Optional solver state. Defaults to automatic initialization.

        **Returns:**

        Distances with shape `(m, size, size)`. For source `i` at row `r` and
        column `c`, entry `[i, a, b]` is the distance to the pixel at row
        `r - size // 2 + a` and column `c - size // 2 + b`.
        """
        if size % 2 == 0:
            raise ValueError("`size` must be odd.")
        sources = _inside_windows(graph, sources, size)
        distances = self(graph, sources=sources, state=state)
        return _windows(distances.reshape(-1, graph.height, graph.width), sources, size)

    @abstractmethod
    def nodes_to_nodes_distance(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> Array:
        pass

    @abstractmethod
    def sources_to_targets_distance(
        self,
        graph: AbstractGraph,
        sources: Array,
        targets: Array,
        state: Any = None,
    ) -> Array:
        pass

    @abstractmethod
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        pass


_WINDOW_OUT_OF_GRID = (
    "sources_to_windows_distance: the window of a source leaves the grid. The "
    "sources of windows of `size` pixels a side lie at least `size // 2` pixels "
    "from the edges of the grid."
)


def _windows_out_of_grid(graph: GridGraph, sources: Array, size: int) -> Array:
    """Mask of the `sources` whose window of `size` pixels a side leaves `graph`."""
    half = size // 2
    rows, columns = sources // graph.width, sources % graph.width
    return (
        (rows < half)
        | (rows >= graph.height - half)
        | (columns < half)
        | (columns >= graph.width - half)
    )


def _inside_windows(graph: GridGraph, sources: Array, size: int) -> Array:
    """`sources`, which raises if the window of `size` pixels a side of one
    leaves `graph`: slices of windows would wrap or clamp silently."""
    outside = jnp.any(_windows_out_of_grid(graph, sources, size))
    return eqx.error_if(sources, outside, _WINDOW_OUT_OF_GRID)


def _windows(distances: Array, sources: Array, size: int) -> Array:
    """The windows of `size` pixels a side centred on each source.

    `distances` holds the distances from each source to every pixel, with
    shape `(m, height, width)`; the result has shape `(m, size, size)`.
    """
    half, width = size // 2, distances.shape[-1]

    def window(distances, source):
        start = (source // width - half, source % width - half)
        return jax.lax.dynamic_slice(distances, start, (size, size))

    return jax.vmap(window)(distances, sources)


class AbstractSolverDistance(AbstractDistance):
    """Distance whose queries solve linear systems with one sparse matrix per graph.

    That matrix is the system matrix of the graph: sparse, symmetric and positive
    definite, such as the grounded Laplacian of `ResistanceDistance`. Subclass it
    with `solver` and `system_matrix` to get the solver state lifecycle of `init`
    and `reinit`.

    **Attributes:**

    - `solver`: Lineax solver for the system matrix, or `None` for a dense solve.
    """

    solver: eqx.AbstractVar[lx.AbstractLinearSolver | None]

    @abstractmethod
    def system_matrix(self, graph: AbstractGraph) -> BCOO:
        """Return the system matrix of `graph`.

        **Arguments:**

        - `graph`: Graph of the queries.

        **Returns:**

        The system matrix, as a `BCOO` matrix.
        """

    def init(self, graph: AbstractGraph) -> Any:
        """Build the solver state for `graph`.

        Some solvers read concrete sparse indices or build a preconditioner on
        the host, such as `GridMultigridSolver` and `AMJaxCGSolver`, so call it
        outside `jax.jit`. `ResistanceDistance` also builds the
        `GridMultigridSolver` state of a `GridGraph` inside `jax.jit`.

        **Arguments:**

        - `graph`: Graph whose system matrix the state is built for.

        **Returns:**

        State to pass to queries and to `reinit`, or `None` without a solver.
        """
        return self.reinit(graph, None)

    def reinit(self, graph: AbstractGraph, state: Any) -> Any:
        if self.solver is None:
            return None
        return _solver_state(self.system_matrix(graph), self.solver, None, state)

    def init_preconditioner(self, graph: AbstractGraph) -> Any:
        """Deprecated alias of `init`, which queries now reinitialize for any graph."""
        warnings.warn(
            "`init_preconditioner` is deprecated; use `init`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.init(graph)


def _has_tracers(tree: Any) -> bool:
    """Whether a leaf of `tree` is a tracer, as inside `jax.jit` or `jax.grad`."""
    return any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(tree))


# Contiguities by number of neighbours, for grid graphs whose neighbours are
# traced.
_CONTIGUITIES = {4: ROOK_CONTIGUITY, 8: QUEEN_CONTIGUITY}


def _grid_multigrid_state(distance: AbstractSolverDistance, graph: GridGraph) -> Any:
    """The `GridMultigridSolver` state of `distance` for `graph`, also in a trace.

    The solver reads the grid of the state from concrete sparse indices, outside
    the trace, with the exact distance (see `AbstractDistance.exact`), whose
    states serve `distance` and which holds no traced seed. Those of a traced
    graph are those of a graph of its shape and contiguity, of permeability 1:
    traced neighbours are those of rook or queen contiguity, by their number.
    The state of that graph is then reinitialized for `graph`, whose sparse
    indices the solver checks.
    """
    concrete = graph
    # A traced graph may lack concrete sparse indices.
    if _has_tracers(graph):
        neighbors = graph.neighbors
        # `eqx.filter_jit` traces the neighbours of a graph it takes whole.
        if _has_tracers(neighbors):
            neighbors = _CONTIGUITIES.get(neighbors.shape[0])
        if neighbors is None:
            # The solver raises.
            return distance.reinit(graph, None)
        with jax.ensure_compile_time_eval():
            permeability = jnp.ones(graph.grid.shape, graph.grid.dtype)
            concrete = GridGraph(permeability, neighbors=neighbors)
    with jax.ensure_compile_time_eval():
        state = distance.exact().reinit(concrete, None)
    return state if concrete is graph else distance.reinit(graph, state)
