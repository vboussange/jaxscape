import jax.numpy as jnp
from jax import Array
import equinox as eqx
from jaxscape.window_operation import WindowOperation
from jax import lax
from tqdm import tqdm
from jaxscape.connectivity_analysis import WindowedAnalysis
from jaxscape.graph import GridGraph
from jaxscape.connectivity_analysis import connectivity
import numpy as np

d_quality = eqx.filter_jit(eqx.filter_grad(connectivity))
d_quality_vmap = eqx.filter_vmap(d_quality, in_axes=(0, 0, 0, None, None, None))

@eqx.filter_jit
@eqx.filter_grad
def d_permeability(permeability_raster: Array, quality_raster: Array, *args, **kwargs) -> float:
    return  connectivity(quality_raster, permeability_raster, *args, **kwargs)
d_permeability_vmap = eqx.filter_vmap(d_permeability, in_axes=(0, 0, 0, None, None, None))

class SensitivityAnalysis(WindowedAnalysis):
    """
    Compute gradients of connectivity with respect to landscape parameters:

    !!! example
    
        ```python
        from jaxscape import SensitivityAnalysis

        sens = SensitivityAnalysis(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=distance,
            proximity=proximity,
            dependency_range=D,
            batch_size=20
        )

        # Sensitivity to permeability changes
        sensitivity_perm = sens.run("permeability", q_weighted=True)

        # Sensitivity to quality changes  
        sensitivity_qual = sens.run("quality", q_weighted=True)
        ```

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) > 0:
            self.original_shape = args[0].shape
        else:
            self.original_shape = np.array(kwargs.get('quality_raster', None).shape) #todo: do we need np.array here?
        
    def _scan_fn(self, raster_buffer: Array, x: tuple) -> tuple:
            _xy, _rast = x
            raster_buffer = self.window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
            return raster_buffer, None
        
    def run(self, var: str = "quality", q_weighted: bool = True) -> Array:
        """
        Runs a sensitivity analysis by calculating the derivative of the connectivity with respect to `var`. `var` can be either "quality" or "permeability".
        """
        if var == "quality":
            sensitivity_fun = d_quality_vmap
        elif var == "permeability":
            sensitivity_fun = d_permeability_vmap
        else:
            raise ValueError("`var` must be either 'quality' or 'permeability'")
        
        output = jnp.zeros_like(self.quality_raster)
        
        for (xy_batch, quality_batch) in tqdm(
            self.batch_op.lazy_iterator(self.quality_raster),
            desc="Batch progress",
            total=self.batch_op.nb_steps,
            miniters=max(1, self.batch_op.nb_steps // 100)
        ):
            permeability_batch = self.batch_op.extract_total_window(xy_batch, self.permeability_raster)
            xy, quality_windows = self.window_op.eager_iterator(quality_batch)
            _, permeability_windows = self.window_op.eager_iterator(permeability_batch)
            
            raster_buffer = jnp.zeros((self.batch_op.total_window_size, self.batch_op.total_window_size))
            res = sensitivity_fun(quality_windows, permeability_windows, self.window_op, self.distance, self.proximity, q_weighted)
            
            # handling padding
            padding = jnp.all(xy_batch + xy + self.window_op.total_window_size <= self.original_shape, axis=1)[:, None, None]
            res = res *  padding
            
            raster_buffer, _ = lax.scan(self._scan_fn, raster_buffer, (xy, res))
            output = self.batch_op.update_raster_with_window(xy_batch, output, raster_buffer, fun=jnp.add)

        output = output[:self.original_shape[0], :self.original_shape[1]]
        return output
        

