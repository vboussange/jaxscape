import equinox as eqx
import jax.numpy as jnp
from jax import Array

from jaxscape.distance import AbstractDistance
from jaxscape.graph import GridGraph


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

    @eqx.filter_jit
    def nodes_to_nodes_distance(self, graph: GridGraph, nodes: Array) -> Array:
        coords = graph.index_to_coord(nodes)
        return euclidean_distance(coords, coords)

    @eqx.filter_jit
    def sources_to_targets_distance(
        self, graph: GridGraph, sources: Array, targets: Array
    ) -> Array:
        source_coords = graph.index_to_coord(sources)
        target_coords = graph.index_to_coord(targets)
        return euclidean_distance(source_coords, target_coords)

    @eqx.filter_jit
    def all_pairs_distance(self, graph: GridGraph) -> Array:
        coords = graph.index_to_coord(jnp.arange(graph.nv))
        return euclidean_distance(coords, coords)


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
