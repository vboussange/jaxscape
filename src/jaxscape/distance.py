from abc import abstractmethod
from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .graph import AbstractGraph


class AbstractDistance(eqx.Module):
    """Abstract base class for distance computations on graphs.

    Provides a unified interface for computing distances with automatic handling of
    coordinate-based (for `GridGraph`) or index-based node specification.

    **Arguments:**

    - `graph`: Graph on which to compute distances.
    - `sources`: Source nodes as vertex indices (1D array) or coordinates (Nx2 array for `GridGraph`).
    - `targets`: Target nodes as vertex indices (1D array) or coordinates (Nx2 array for `GridGraph`).
    - `nodes`: Nodes for pairwise distances as vertex indices (1D) or coordinates (Nx2).

    Specify either: `nodes` alone, `sources` and/or `targets`, or neither (for all-pairs).

    **Returns:**

    Distance array with shape depending on the inputs.

    !!! example

        ```python
        from jaxscape import LCPDistance, GridGraph
        import jax.numpy as jnp

        distance = LCPDistance()
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)

        # All-pairs distance
        D = distance(grid)  # Shape: (n_nodes, n_nodes)

        # Using vertex indices
        D = distance(grid, sources=jnp.array([0, 1]), targets=jnp.array([10, 20]))  # Shape: (2, 2)

        # Using coordinates (for GridGraph)
        D = distance(grid, sources=jnp.array([[0, 0], [1, 1]]), targets=jnp.array([[10, 10]]))  # Shape: (2, 1)

        # Pairwise among subset
        D = distance(grid, nodes=jnp.array([0, 5, 10]))  # Shape: (3, 3)
        ```
    """

    def __call__(
        self,
        graph: AbstractGraph,
        sources: Optional[Array] = None,
        targets: Optional[Array] = None,
        nodes: Optional[Array] = None,
    ) -> Array:
        if nodes is not None:
            assert (
                sources is None and targets is None
            ), "Specify either `nodes`, `sources`, `targets`, or both `sources` and `targets`."
            if nodes.ndim == 2:
                if not hasattr(graph, "coord_to_index"):
                    raise ValueError(
                        "Coordinate input (2D array) requires a GridGraph. "
                        "Use vertex indices (1D array) for Graph objects."
                    )
                nodes = graph.coord_to_index(nodes[:, 0], nodes[:, 1])
            return self.nodes_to_nodes_distance(graph, nodes)

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

            return self.sources_to_targets_distance(graph, sources, targets)

        else:
            return self.all_pairs_distance(graph)

    @abstractmethod
    def nodes_to_nodes_distance(self, graph: AbstractGraph, nodes: Array) -> Array:
        pass

    @abstractmethod
    def sources_to_targets_distance(
        self, graph: AbstractGraph, sources: Array, targets: Array
    ) -> Array:
        pass

    @abstractmethod
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        pass
