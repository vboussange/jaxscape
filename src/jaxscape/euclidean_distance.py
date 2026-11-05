from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph, GridGraph


class EuclideanDistance(AbstractDistance):
    """
    Straight-line distance in grid coordinates. Only works with `GridGraph`.

    !!! example

        ```python
        from jaxscape import EuclideanDistance

        distance = EuclideanDistance()
        dist = distance(grid, sources=source_coords, targets=target_coords)
        ```
    """

    symmetric: ClassVar[bool] = True

    def init(self, graph: AbstractGraph) -> None:
        del graph
        return None

    @eqx.filter_jit
    def nodes_to_nodes_distance(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> Array:
        del state
        coords = _coordinates(graph, nodes)
        return euclidean_distance(coords, coords)

    @eqx.filter_jit
    def sources_to_targets_distance(
        self, graph: AbstractGraph, sources: Array, targets: Array, state: Any = None
    ) -> Array:
        del state
        source_coords = _coordinates(graph, sources)
        target_coords = _coordinates(graph, targets)
        return euclidean_distance(source_coords, target_coords)

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        del state
        coords = _coordinates(graph, jnp.arange(graph.nv))
        return euclidean_distance(coords, coords)


def _coordinates(graph: AbstractGraph, nodes: Array) -> Array:
    """The grid coordinates of `nodes`."""
    if not isinstance(graph, GridGraph):
        raise ValueError("`EuclideanDistance` needs a `GridGraph`.")
    return graph.index_to_coord(nodes)


@eqx.filter_jit
def euclidean_distance(
    coordinate_list_sources: Array, coordinate_list_target: Array
) -> Array:
    """Computes the euclidean distance between two sets of coordinates.
    `coordinate_list_sources` must be of size (N, 2) and
    `coordinate_list_target` must be of size (M, 2)."""
    Xmat = coordinate_list_sources[:, 0, None] - coordinate_list_target[None, :, 0]
    Ymat = coordinate_list_sources[:, 1, None] - coordinate_list_target[None, :, 1]
    euclidean_distance = jnp.hypot(Xmat, Ymat)
    return euclidean_distance
