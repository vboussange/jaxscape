"""
Running sensitivity analysis of equivalent connected habitat for euclidean distance.
This script copies the behavior of omniscape.
"""
from jax import lax
import jax.numpy as jnp
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.euclidean_distance import EuclideanDistance
from tqdm import tqdm
import equinox as eqx

def make_raster(N=1010):
    key = jr.PRNGKey(1)
    return jr.uniform(key, (N, N), minval=0.1, maxval=0.9, dtype="float32")

@eqx.filter_jit
def batch_run_calculation(window_op, xy, hab_qual, raster_buffer):
    res = hab_qual
    def scan_fn(raster_buffer, x):
        _xy, _rast = x
        raster_buffer = window_op.update_raster_with_window(_xy, raster_buffer, _rast)
        return raster_buffer, None
    raster_buffer, _ = lax.scan(scan_fn, raster_buffer, (xy, res))
    return raster_buffer

if __name__ == "__main__":
    N = 100
    window_size = 1 # must be odd to be placed at the center
    buffer_size = 10
    batch_size = 20 
    
    permeability = make_raster(N)

    distance = EuclideanDistance()

    batch_op = WindowOperation(
        shape=permeability.shape, 
        window_size=batch_size, 
        buffer_size=buffer_size
    )

    output = jnp.zeros_like(permeability)

    for (xy_batch, permeability_batch) in tqdm(batch_op.lazy_iterator(permeability), desc="Batch progress"):
        window_op = WindowOperation(shape=permeability_batch.shape, 
                                    window_size=window_size, 
                                    buffer_size=buffer_size)
        xy, hab_qual = window_op.eager_iterator(permeability_batch)
        activities = jnp.ones_like(hab_qual, dtype="bool")
        raster_buffer = jnp.zeros_like(permeability_batch)
        res = batch_run_calculation(window_op, xy, hab_qual, raster_buffer)
        output = batch_op.update_raster_with_window(xy_batch, output, raster_buffer)
    
    
    # TODO: fix those tests
    assert jnp.allclose(permeability[1:-1, 1:-1], output[1:-1, 1:-1])