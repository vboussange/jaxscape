import jax.numpy as jnp
from jasxscape.run import batch_run_calculation, padding, run

def test_batch_run_calculation():
    # to implement
    pass

def test_padding():
    raster = jnp.ones((2300, 3600))
    buffer_size = 50
    window_size = 3
    padded_raster = padding(raster, buffer_size, window_size)
    
    for i in range(2):
        assert (padded_raster.shape[i] - 2 * buffer_size) % window_size == 0
    
    # other test
    raster = jnp.ones((230, 360))
    buffer_size = 3
    window_size = 27
    padded_raster = padding(raster, buffer_size, window_size)

    for i in range(2):
        assert (padded_raster.shape[i] - 2 * buffer_size) % window_size == 0
    
def test_run():
    # to implement
    pass