import jax.numpy as jnp
import jax.random as jr
from jaxscape import GridGraph
from jaxscape.euclidean_distance import EuclideanDistance


def test_euclidean_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))

    grid = GridGraph(grid=permeability_raster)
    distance = EuclideanDistance()
    dist = distance(grid)
    assert dist[0, 0] == 0
    source_idx = 0
    target_idx = 4
    source_xy_coord = grid.index_to_coord(jnp.array([source_idx, target_idx]))
    assert dist[source_idx, target_idx] == jnp.sqrt(
        jnp.sum((source_xy_coord[0, :] - source_xy_coord[1, :]) ** 2)
    )
