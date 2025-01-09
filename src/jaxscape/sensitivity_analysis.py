import jax.numpy as jnp
import equinox as eqx
from jaxscape.moving_window import WindowOperation
from jax import lax
from tqdm import tqdm
from jaxscape.connectivity_analysis import WindowedAnalysis
from jaxscape.gridgraph import GridGraph
from jaxscape.connectivity_analysis import connectivity
import numpy as np

d_quality = eqx.filter_jit(eqx.filter_grad(connectivity))
d_quality_vmap = eqx.filter_vmap(d_quality, in_axes=(0, 0, 0, None, None, None))

@eqx.filter_jit
@eqx.filter_grad
def d_permeability(permeability_raster, quality_raster, *args, **kwargs):
    return  connectivity(quality_raster, permeability_raster, *args, **kwargs)
d_permeability_vmap = eqx.filter_vmap(d_permeability, in_axes=(0, 0, 0, None, None, None))

class SensitivityAnalysis(WindowedAnalysis):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) > 0:
            self.original_shape = args[0].shape
        else:
            self.original_shape = np.array(kwargs.get('quality_raster', None).shape)
        
    def _scan_fn(self, raster_buffer, x):
            _xy, _rast = x
            raster_buffer = self.window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
            return raster_buffer, None
        
    @eqx.filter_jit
    def run(self, fun):
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
        
        output = jnp.zeros_like(self.quality_raster)
        
        for (xy_batch, quality_batch) in tqdm(
            self.batch_op.lazy_iterator(self.quality_raster),
            desc="Batch progress",
            total=self.batch_op.nb_steps
        ):
            permeability_batch = self.batch_op.extract_total_window(xy_batch, self.permeability_raster)
            xy, quality_windows = self.window_op.eager_iterator(quality_batch)
            _, permeability_windows = self.window_op.eager_iterator(permeability_batch)
            activities = jnp.ones_like(quality_windows, dtype="bool")
            
            # print(xy_batch + xy + self.window_op.total_window_size)

            raster_buffer = jnp.zeros((self.batch_op.total_window_size, self.batch_op.total_window_size))
            res = fun(quality_windows, permeability_windows, activities, self.window_op, self.distance, self.proximity)
            
            # handling padding
            padding = jnp.all(xy_batch + xy + self.window_op.total_window_size <= self.original_shape, axis=1)[:, None, None]
            res = res *  padding
            
            raster_buffer, _ = lax.scan(self._scan_fn, raster_buffer, (xy, res))
            output = self.batch_op.update_raster_with_window(xy_batch, output, raster_buffer, fun=jnp.add)

        output = output[:self.original_shape[0], :self.original_shape[1]]
        return output
        

