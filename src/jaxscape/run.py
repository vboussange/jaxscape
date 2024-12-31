import jax.numpy as jnp
import equinox as eqx
from jaxscape.moving_window import WindowOperation
from jax import lax
from tqdm import tqdm

@eqx.filter_jit
def batch_run_calculation(batch_op, window_op, xy, fun, *args):
    """
    Perform a batch run calculation using JAX for just-in-time compilation and scanning.
    Parameters:
        batch_op: An object containing batch operation parameters, including total window size.
        window_op: An object containing window operation parameters and methods.
        xy: Coordinates for the calculation.
        fun: A function to apply to the window center and additional arguments.
        *args: Additional arguments to pass to the function `fun`.
    Returns:
        A JAX array representing the raster buffer after applying the batch run calculation.
        
    # TODO: not sure how to deal with `args` that may contain more than one raster together with `activities`

    """
    raster_buffer = jnp.zeros((batch_op.total_window_size, batch_op.total_window_size))
    
    window_center = jnp.array([[window_op.total_window_size//2+1, window_op.total_window_size//2+1]])
    res = fun(window_center, *args)
    
    def scan_fn(raster_buffer, x):
        _xy, _rast = x
        raster_buffer = window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
        return raster_buffer, None
    raster_buffer, _ = lax.scan(scan_fn, raster_buffer, (xy, res))
    return raster_buffer

def padding(raster, buffer_size, window_size):
    """
    Pads the given raster array to ensure its dimensions are compatible with the
    specified window size, i.e. assert (raster.shape[i] - 2 * buffer_size) %
    window_size == 0
    """
    inner_height = raster.shape[0] - 2 * buffer_size
    inner_width = raster.shape[1] - 2 * buffer_size

    pad_height = (window_size - (inner_height % window_size)) % window_size
    pad_width = (window_size - (inner_width % window_size)) % window_size


    padded_raster = jnp.pad(
        raster,
        ((0,pad_height),(0,pad_width)),
        mode='constant'
    )
    return padded_raster

def run(fun, rasters, coarsening_factor, dependency_range, batch_size):
    """
    Run the specified function over the given rasters with coarsening and batching.
    
    Parameters:
        fun (callable): Function to apply, should be of the form fun(target, *rasters).
        rasters (list): List of rasters used by the function for calculations.
        coarsening_factor (float): Factor to determine the coarsening level, must be between 0 and 1.
        dependency_range (int): Range of dependency for the calculations.
        batch_size (int): Size of the batch for processing.
    Returns:
        jnp.ndarray: The resulting raster after applying the function.
    """
    # fun should be of the form fun(target, *rasters)
    
    assert 0 <= coarsening_factor <= 1
    
    # rasters should be a list of rasters used by fun for calculations
    assert len(rasters) >= 1
    for raster in rasters[1:]:
        assert raster.shape == rasters[0].shape
        
    # you should assert that `dependency_range` and `batch_size` are integers
        
    # `coarsening`  is the number of pixels -1 to skip per iteration and that will not be considered as landmarks
    coarsening = int(jnp.ceil(dependency_range * coarsening_factor))
    if coarsening % 2 == 0:
        coarsening += 1
    
    # buffer size should be of the order of the dispersal range - half that of the window operation size
    # size distance is calculated from the center pixel of the window
    buffer_size = int(dependency_range - (coarsening - 1)/2)
    if buffer_size < 1:
        raise ValueError("Buffer size is too small. Consider decreasing the coarsening factor or decreasing the raster resolution.")
    
    batch_window_size = batch_size * coarsening
    rasters_padded = [padding(raster, buffer_size, batch_size) for raster in rasters]

    batch_op = WindowOperation(
            shape=rasters_padded.shape, 
            window_size=batch_window_size, 
            buffer_size=buffer_size)
        
    output = jnp.zeros_like(rasters_padded) # initialize raster
    window_op = WindowOperation(shape=(batch_op.total_window_size, batch_op.total_window_size), 
                                window_size=coarsening, 
                                buffer_size=buffer_size)
    
    # TODO: not sure how to deal with `rasters` that may contain more than one raster and should replace `quality_padded`
    # TODO: how to treate `activities` is also unclear
    for (xy_batch, permeability_batch) in tqdm(batch_op.lazy_iterator(quality_padded), 
                                               desc="Batch progress",
                                               total=batch_op.nb_steps):
        
        xy, rast = window_op.eager_iterator(permeability_batch)
        activities = jnp.ones_like(hab_qual, dtype="bool")
        res = batch_run_calculation(batch_op, window_op, xy, fun, rast, activities)
        output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
    
    # unpadding
    output = output[:rasters[0].shape[0], :rasters[0].shape[1]]
    return output
        

