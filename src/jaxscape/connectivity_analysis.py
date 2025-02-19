import jax.numpy as jnp
import equinox as eqx
from jaxscape.moving_window import WindowOperation
from jaxscape.utils import padding
from jax import lax
from tqdm import tqdm
from jaxscape.gridgraph import GridGraph

def connectivity(quality_raster, permeability_raster, window_op, distance, proximity, q_weighted):
    """
    Calculate connectivity of landscape characterised by `quality` and `permeability`.

    **Arguments:**
    
    - quality_raster : A 2D array representing the quality of each cell in the raster.
    - permeability_raster : A 2D array representing the permeability of each cell in the raster.
    - window_op : an object containing window operation parameters such as buffer_size and window_size.
    - distance : A function to calculate the distance between grid points.
    - proximity : A function to calculate the proximity based on distance.
        
    **Returns:**

    If `q_weighted=False`, returns the sum of the proximities, `jnp.sum(K)`. Otherwise, returns the sum of the proximities weighted by the qualities,  `q @ K @ q.T`.
    """
    grid = GridGraph(vertex_weights=permeability_raster,
                    fun= lambda x, y: (x + y)/2)
    window_center = jnp.array([[permeability_raster.shape[0]//2, permeability_raster.shape[1]//2]])
    window_center_index = grid.coord_to_index(
                window_center[:, 0], window_center[:, 1]
            )
    
    x_core_window, y_core_window = jnp.meshgrid(jnp.arange(window_op.buffer_size, 
                                            window_op.window_size+window_op.buffer_size), 
                                            jnp.arange(window_op.buffer_size, 
                                                        window_op.window_size+window_op.buffer_size))
    window_core_indices = grid.coord_to_index(
                x_core_window, y_core_window
            )

    dist = distance(grid, sources=window_center_index).flatten()
    K = proximity(dist)

    K = K.at[window_core_indices].set(0)
    
    if q_weighted:
    
        core_window_qual = lax.dynamic_slice(quality_raster, 
                                        start_indices=(window_op.buffer_size, window_op.buffer_size), 
                                        slice_sizes=(window_op.window_size, window_op.window_size))
        
        q = grid.array_to_node_values(quality_raster)

        qKqT = jnp.sum(core_window_qual) * (K @ q.T)

        return  qKqT
    else:
        return jnp.sum(K)

connectivity_vmap = eqx.filter_vmap(connectivity, in_axes=(0, 0, 0, None, None, None))


class WindowedAnalysis:
    def __init__(self, quality_raster, permeability_raster, distance, proximity, coarsening_factor, dependency_range, batch_size):
        """
        Base class for connectivity windowed analyses. By default, the underlying GridGraph has a `ROOK_CONTIGUITY`, and edges are defined by `fun = lambda x, y: (x + y)/2`.
        
        **Attributes:**
        
        - `quality_raster`: the quality raster data.
        
        - `permeability_raster` : the permeability raster data or a callable that generates it from `quality_raster`.
        
        - `distance` : a graph distance inheriting from `AbstractDistance`, used for calculating the ecological proximity (e.g., `LCPDistance()`).
        
        - `proximity` : a function that transforms a distance into an ecological proximity. Consider e.g. `lambda x: jnp.exp(-x)`.
       
        - `coarsening_factor` : the factor by which to coarsen the sensitivity analysis, based on the dependency range. Must be between 0 and 1.
            
        - `dependency_range` : the range of dependency for the analysis, in terms of pixels.
        
        - `batch_size` : the size of the batch for the analysis.
        """
        
        assert 0 <= coarsening_factor <= 1
        assert isinstance(dependency_range, int)
        assert isinstance(batch_size, int)
        
        coarsening = int(jnp.ceil(dependency_range * coarsening_factor))
        if coarsening % 2 == 0:
            coarsening += 1

        buffer_size = int(dependency_range - (coarsening - 1)/2)
        if buffer_size < 1:
            raise ValueError("Buffer size too small.")
        
        batch_window_size = batch_size * coarsening
        
        if callable(permeability_raster):
            permeability_raster = permeability_raster(quality_raster)
        else:
            permeability_raster = permeability_raster
        assert quality_raster.shape == permeability_raster.shape
        
        quality_padded = padding(quality_raster, buffer_size, batch_window_size)
        permeability_padded = padding(permeability_raster, buffer_size, batch_window_size)
        
        batch_op = WindowOperation(
            shape=quality_padded.shape,
            window_size=batch_window_size,
            buffer_size=buffer_size
        )
        window_op = WindowOperation(
            shape=(batch_op.total_window_size, batch_op.total_window_size),
            window_size=coarsening,
            buffer_size=buffer_size
        )
        
        self.quality_raster = quality_padded
        self.permeability_raster = permeability_padded
        self.distance = distance
        self.proximity = proximity
        self.batch_op = batch_op
        self.window_op = window_op
        # self.shape = quality_raster.shape
    
class ConnectivityAnalysis(WindowedAnalysis):
    def run(self, q_weighted=True):
        output = jnp.array(0.)
        for (xy_batch, quality_batch) in tqdm(
            self.batch_op.lazy_iterator(self.quality_raster),
            desc="Batch progress",
            total=self.batch_op.nb_steps,
            miniters=max(1, self.batch_op.nb_steps // 100)
        ):
            permeability_batch = self.batch_op.extract_total_window(xy_batch, self.permeability_raster)
            _, quality_windows = self.window_op.eager_iterator(quality_batch)
            _, permeability_windows = self.window_op.eager_iterator(permeability_batch)
            output += jnp.sum(connectivity_vmap(quality_windows, permeability_windows, self.window_op, self.distance, self.proximity, q_weighted))
        return output
