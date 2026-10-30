import warnings
from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO

from .graph import AbstractGraph
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
    - `state`: Optional solver setup state. Defaults to automatic initialization.

    Specify either: `nodes` alone, `sources` and/or `targets`, or neither
    (for all-pairs).

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

    def __call__(
        self,
        graph: AbstractGraph,
        sources: Array | None = None,
        targets: Array | None = None,
        nodes: Array | None = None,
        state: Any = None,
    ) -> Array:
        if state is None:
            # Solver state is setup data, not part of the differentiable solve.
            # Stop gradients here so auto-initialized state matches Lineax's
            # `state=None` semantics and never carries tangents into custom VJPs.
            state = self.init(jax.tree.map(jax.lax.stop_gradient, graph))

        if nodes is not None:
            assert (
                sources is None and targets is None
            ), (
                "Specify either `nodes`, `sources`, `targets`, or both "
                "`sources` and `targets`."
            )
            if nodes.ndim == 2:
                if not hasattr(graph, "coord_to_index"):
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
                if not hasattr(graph, "coord_to_index"):
                    raise ValueError(
                        "Coordinate input (2D array) for sources requires a GridGraph. "
                        "Use vertex indices (1D array) for Graph objects."
                    )
                sources = graph.coord_to_index(sources[:, 0], sources[:, 1])

            if targets.ndim == 2:
                if not hasattr(graph, "coord_to_index"):
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


class AbstractSolverDistance(AbstractDistance):
    """Distance whose queries solve linear systems with one sparse matrix per graph.

    Subclass it with `solver` and `system_matrix` to get the solver state lifecycle
    of `init` and `reinit`.

    **Attributes:**

    - `solver`: Lineax solver for the system matrix, or `None` for a dense solve.
    """

    solver: eqx.AbstractVar[lx.AbstractLinearSolver | None]

    @abstractmethod
    def system_matrix(self, graph: AbstractGraph) -> BCOO:
        """Return the sparse matrix that queries on `graph` solve with.

        **Arguments:**

        - `graph`: Graph of the queries.

        **Returns:**

        Sparse symmetric positive definite matrix.
        """

    def init(self, graph: AbstractGraph) -> Any:
        """Build the solver state for `graph`.

        Some solvers read concrete sparse indices or build a preconditioner on the
        host, so call it outside `jax.jit`.

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
