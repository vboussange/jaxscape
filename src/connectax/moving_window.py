import jax.numpy as jnp
from jax import grad, jit
from math import ceil
import numpy as np  # for NaN handling, not used in heavy computations
import matplotlib.pyplot as plt
from connectax.gridgraph import GridGraph
from connectax.utils import BCOO_to_sparse, get_largest_component_label
from connectax.landscape import Landscape
import jax
from jax.experimental.sparse import BCOO
from tqdm import tqdm

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

class WindowOperation:
    """Handles window-based operations on raster data."""
    def __init__(self, raster_data, window_size, buffer_size):
        assert isinstance(raster_data, jax.Array)
        assert isinstance(window_size, int)
        assert isinstance(buffer_size, int)
        
        self.raster_data = raster_data
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.total_window_size = self.window_size + 2 * self.buffer_size
        self.output_array = jnp.full(self.raster_data.shape, jnp.nan)

    def replace_missing(self, array, value=jnp.nan):
        """Replace missing data in the array with specified value."""
        return jnp.where(jnp.isnan(array), value, array)

    def extract_window(self, x_start, y_start):
        """Extract a buffered window from the raster data."""
        window = self.raster_data[
            x_start:x_start + self.total_window_size,
            y_start:y_start + self.total_window_size
        ]
        return self.replace_missing(window)

    def iterate_windows(self):
        """Yield buffered windows for computation, skipping empty areas."""
        width, height = self.raster_data.shape
        x_steps = int((width - 2 * self.buffer_size) // self.window_size)
        y_steps = int((height - 2 * self.buffer_size) // self.window_size)

        for i in range(1, x_steps - 1):
            for j in range(1, y_steps - 1):
                x_start, y_start = i * self.window_size, j * self.window_size
                window = self.extract_window(x_start, y_start)
                
                if jnp.any(~jnp.isnan(window)):
                    yield x_start, y_start, window
                    
def get_valid_activities(hab_qual, activities):
    # TODO: the best would be to avoid transfer between numpy and jax array
    grid = GridGraph(activities, hab_qual)
    A = grid.get_adjacency_matrix()
    Anp = BCOO_to_sparse(A)
    _, labels = connected_components(Anp, directed=True, connection="strong")
    label = get_largest_component_label(labels)
    vertex_belongs_to_largest_component_node = labels == label
    activities_pruned = grid.node_values_to_raster(vertex_belongs_to_largest_component_node)
    activities_pruned = activities_pruned == True
    return activities_pruned

# TODO: for now, we hardcode the distance to `rsp_distance`, but in the future, we should allow for arbitraty distance functions
def run_analysis(window_op, D, gridgraph, **kwargs):
    """Performs the sensitivity analysis on each valid window.
    `D` must be expressed in the unit of habitat quality in `window_op`.
    """
    for x_start, y_start, hab_qual in tqdm(window_op.iterate_windows(), total=window_op.total_window_size, desc="Running Analysis"):
        # Build grid graph and calculate Euclidean distances
        activities = hab_qual > 0
        valid_activities = get_valid_activities(hab_qual, activities)

        def connectivity(hab_qual):
            # TODO: need to iterate through the connected components
            # for now, we only take the largest component, but we could build a loop here
            grid = gridgraph(activities=valid_activities, 
                                    vertex_weights=hab_qual)
            dist = grid.get_distance_matrix(**kwargs)
            proximity = jnp.exp(-dist / D)
            landscape = Landscape(hab_qual, proximity, valid_activities)
            func = landscape.functional_habitat()
            return func
    
        grad_connectivity = grad(connectivity)
        dcon = grad_connectivity(hab_qual)

        # Store results into the core window area of the output array
        x_core_start = x_start + window_op.buffer_size
        x_core_end = x_core_start + window_op.window_size
        y_core_start = y_start + window_op.buffer_size
        y_core_end = y_core_start + window_op.window_size

        # Update the output array within the specified core region
        window_op.output_array = window_op.output_array.at[
            x_core_start:x_core_end, y_core_start:y_core_end
        ].set(dcon[window_op.buffer_size:window_op.buffer_size + window_op.window_size,
                window_op.buffer_size:window_op.buffer_size + window_op.window_size])


    return window_op.output_array