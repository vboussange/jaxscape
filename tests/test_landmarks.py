import pytest
import jax.numpy as jnp
from jax import random
from jaxscape.gridgraph import GridGraph
from jaxscape.landmarks import sum_neighborhood, coarse_graining
import jax.random as jr
import equinox

def test_sum_neighborhood():
    # Grid setup
    activities = jnp.array([[True, True, True],
                            [True, True, True],
                            [True, True, True]])

    vertex_weights = jnp.array([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0],
                                [7.0, 8.0, 9.0]])

    grid = GridGraph(activities, vertex_weights)

    # Test cases
    assert sum_neighborhood(grid, jnp.array([[1, 1]]), 3) ==  vertex_weights.sum() # Full grid sum
    assert sum_neighborhood(grid, jnp.array([[0, 0]]), 1) == 12.0  # Top-left corner

    # Test with NaN
    vertex_weights_nan = vertex_weights.at[2, 2].set(jnp.nan)
    grid_nan = GridGraph(activities, vertex_weights_nan)
    assert sum_neighborhood(grid_nan, jnp.array([[1, 1]]), 3) == pytest.approx(36.0)  # Ignore NaN

    # jit test
    jit_sum_neighborhood = equinox.filter_jit(sum_neighborhood)
    assert jit_sum_neighborhood(grid_nan, jnp.array([[1, 1]]), 3) == pytest.approx(36.0)  # Ignore NaN

    
def test_coarse_graining():
    N = 20
    vertex_weights = jnp.ones((N, N), dtype=jnp.float32)
    activities = jnp.ones_like(vertex_weights, dtype=bool)
    grid = GridGraph(activities, vertex_weights)
    buffer_size = 2
    
    # jit test
    jit_coarse_graining = equinox.filter_jit(coarse_graining)
    coarse_matrix = jit_coarse_graining(grid, buffer_size).todense()
    assert coarse_matrix[2,2] == 1.0