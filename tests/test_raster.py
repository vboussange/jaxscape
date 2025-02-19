from jaxscape.rastergraph import RasterGraph
from jaxscape.gridgraph import GridGraph
import jax.numpy as jnp
import jax.random as jr
import pytest


@pytest.fixture
def sample_rastergraph():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    # Define a habitat suitability raster
    habitat_suitability = jr.uniform(key, (10, 10))
    grid = GridGraph(vertex_weights=habitat_suitability)
    x_coords = jnp.linspace(10, 100, 10)  # Example longitudes
    y_coords = jnp.linspace(-100, -10, 10)    # Example latitudes
    rastergraph = RasterGraph(grid, x_coords=x_coords, y_coords=y_coords)
    return rastergraph

def test_index_nearest_index(sample_rastergraph):
    # Test for the `index` method's accuracy in finding nearest indices
    lon, lat = 10, -90  # center coordinates
    i, j = sample_rastergraph.index(lon, lat)
    assert (i, j) == (0, 1)

    # Test for the `index` method's accuracy in finding nearest indices
    lon, lat = jnp.array([10, 20]), jnp.array([-100, -90])  # center coordinates
    i, j = sample_rastergraph.index(lon, lat)
    assert jnp.array_equal(i, jnp.array([0, 1]))
    assert jnp.array_equal(j, jnp.array([0, 1]))
    
    # Test with longitude out of bounds
    with pytest.raises(ValueError, match="Longitude .* is out of bounds"):
        sample_rastergraph.index(0., -90) # Longitude less than min longitude

    with pytest.raises(ValueError, match="Latitude .* is out of bounds"):
        sample_rastergraph.index(10., 0.) # Longitude less than min longitude