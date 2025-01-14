import jax.numpy as jnp
import equinox as eqx
from jaxscape.moving_window import WindowOperation
from jaxscape.utils import padding
from jax import lax
from tqdm import tqdm
from jaxscape.gridgraph import GridGraph

def connectivity(quality_raster, permeability_raster, activity, window_op, distance, proximity):
        grid = GridGraph(activities=activity, 
                        vertex_weights=permeability_raster,
                        nb_active=activity.size,
                        fun= lambda x, y: (x + y)/2)
        window_center = jnp.array([[activity.shape[0]//2, activity.shape[1]//2]])
        window_center_index = grid.coord_to_active_vertex_index(
                    window_center[:, 0], window_center[:, 1]
                )
        
        x_core_window, y_core_window = jnp.meshgrid(jnp.arange(window_op.buffer_size, 
                                                window_op.window_size+window_op.buffer_size), 
                                    jnp.arange(window_op.buffer_size, 
                                                window_op.window_size+window_op.buffer_size))
        window_core_indices = grid.coord_to_active_vertex_index(
                    x_core_window, y_core_window
                )

        dist = distance(grid, sources=window_center_index).flatten()
        K = proximity(dist)
        # TODO: this is a dirty fix, to remove artifacts from the diagonal
        K = K.at[window_core_indices].set(0)
        
        core_window_qual = lax.dynamic_slice(quality_raster, 
                                        start_indices=(window_op.buffer_size, window_op.buffer_size), 
                                        slice_sizes=(window_op.window_size, window_op.window_size))
        
        # center = jnp.array([core_window_qual.shape[0] // 2, core_window_qual.shape[1] // 2])
        # indices = jnp.indices(core_window_qual.shape).transpose(1, 2, 0)
        # distances = jnp.sqrt(jnp.sum((indices - center) ** 2, axis=-1))
        # correction_factor = 1 + distances # Adding a small value to avoid division by zero
        # quality_raster = quality_raster.at[x_core_window, y_core_window].set(quality_raster[x_core_window, y_core_window] * correction_factor)
        # quality_raster = quality_raster.at[x_core_window, y_core_window].set(0)
        q = grid.array_to_node_values(quality_raster)

        qKqT = jnp.sum(core_window_qual) * (K @ q.T)
        # qKqT = jnp.array(0.)
        return  qKqT

connectivity_vmap = eqx.filter_vmap(connectivity, in_axes=(0, 0, 0, None, None, None))


class WindowedAnalysis:
    def __init__(self, quality_raster, permeability_raster, distance, proximity, coarsening_factor, dependency_range, batch_size):
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
    def run(self):
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
            activities = jnp.ones_like(quality_windows, dtype="bool")
            output += jnp.sum(connectivity_vmap(quality_windows, permeability_windows, activities, self.window_op, self.distance, self.proximity))
        return output
