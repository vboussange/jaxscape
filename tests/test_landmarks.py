import equinox
import jax.numpy as jnp
import pytest
from jaxscape import GridGraph
from jaxscape.landmarks import coarse_graining, sum_neighborhood


def test_sum_neighborhood():
    # Grid setup
    grid = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    mygrid = GridGraph(grid)

    # Test cases
    assert (
        sum_neighborhood(mygrid, jnp.array([[1, 1]]), 3) == grid.sum()
    )  # Full grid sum
    assert sum_neighborhood(mygrid, jnp.array([[0, 0]]), 1) == 12.0  # Top-left corner
    # Test with NaN
    vertex_weights_nan = grid.at[2, 2].set(jnp.nan)
    grid_nan = GridGraph(vertex_weights_nan)
    assert sum_neighborhood(grid_nan, jnp.array([[1, 1]]), 3) == pytest.approx(
        36.0
    )  # Ignore NaN

    # jit test
    jit_sum_neighborhood = equinox.filter_jit(sum_neighborhood)
    assert jit_sum_neighborhood(grid_nan, jnp.array([[1, 1]]), 3) == pytest.approx(
        36.0
    )  # Ignore NaN


def test_coarse_graining():
    N = 20
    grid = jnp.ones((N, N), dtype=jnp.float32)
    activities = jnp.ones_like(grid, dtype=bool)
    grid = GridGraph(grid)
    buffer_size = 2

    # jit test
    jit_coarse_graining = equinox.filter_jit(coarse_graining)
    coarse_matrix = jit_coarse_graining(grid, buffer_size).todense()
    assert coarse_matrix[2, 2] == 1.0
