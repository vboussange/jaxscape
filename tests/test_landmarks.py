import pytest
import jax.numpy as jnp
from jax import random
from jaxscape.gridgraph import GridGraph
from jaxscape.landmarks import sum_neighborhood, coarse_graining
import jax.random as jr

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
    assert sum_neighborhood(grid, jnp.array([[0, 0]]), 3) == 12.0  # Top-left corner
    # TODO: we stopped here
    # assert sum_neighborhood(grid, jnp.array([[2, 2]]), 3) == pytest.approx(24.0)  # Bottom-right corner
    assert sum_neighborhood(grid, (1, 1), 1) == pytest.approx(5.0)   # Center cell only

    # Test with NaN
    vertex_weights_nan = vertex_weights.at[2, 2].set(jnp.nan)
    grid_nan = GridGraph(activities, vertex_weights_nan)
    assert sum_neighborhood(grid_nan, (1, 1), 3) == pytest.approx(36.0)  # Ignore NaN


    # custom test
    key = jr.key(0)
    permeability_raster = jr.uniform(key, (20, 20))
    activities = permeability_raster>0
    grid = GridGraph(activities, permeability_raster)
    jnp.meshgrid()

def test_coarse_graining():
    # Grid setup
    activities = jnp.array([[True, True, True, True],
                            [True, True, True, True],
                            [True, True, True, True],
                            [True, True, True, True]])

    vertex_weights = jnp.array([[1.0, 2.0, 3.0, 4.0],
                                [5.0, 6.0, 7.0, 8.0],
                                [9.0, 10.0, 11.0, 12.0],
                                [13.0, 14.0, 15.0, 16.0]])

    grid = GridGraph(activities, vertex_weights)

    # Test cases
    # TODO: failed here
    coarse_matrix_2x2 = coarse_graining(grid, 2)
    expected_2x2 = jnp.array([[14.0, 22.0],
                              [46.0, 54.0]])
    assert jnp.allclose(coarse_matrix_2x2[1:4:2, 1:4:2].todense(), expected_2x2)

    coarse_matrix_3x3 = coarse_graining(grid, 3)
    expected_3x3 = jnp.array([[45.0]])
    assert jnp.allclose(coarse_matrix_3x3[1:3, 1:3], expected_3x3)

    # Test with NaN
    vertex_weights_nan = vertex_weights.at[1, 1].set(jnp.nan)
    grid_nan = GridGraph(activities, vertex_weights_nan)
    coarse_matrix_2x2_nan = coarse_graining(grid_nan, 2)
    expected_2x2_nan = jnp.array([[8.0, 22.0],
                                   [46.0, 54.0]])
    assert jnp.allclose(coarse_matrix_2x2_nan[1:4:2, 1:4:2], expected_2x2_nan)

