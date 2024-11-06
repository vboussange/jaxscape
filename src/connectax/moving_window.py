import jax.numpy as jnp
from jax import grad, jit
from math import ceil
import numpy as np  # for NaN handling, not used in heavy computations
import matplotlib.pyplot as plt
from connectax.gridgraph import GridGraph
from connectax.connectivity import BCOO_to_sparse, get_largest_component, functional_habitat
from connectax.rsp_distance import rsp_distance
import jax
from jax.experimental.sparse import BCOO

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
                    
def _get_vertices_largest_component(A):
    # Anp = BCOO_to_sparse(A)
    Anp = csr_matrix(np.array(A))
    # TODO: the best would be to avoid transfer between numpy and jax array
    _, labels = connected_components(Anp, directed=True, connection="strong")
    return get_largest_component(labels)

# TODO: for now, we hardcode the distance to `rsp_distance`, but in the future, we should allow for arbitraty distance functions
def run_analysis(window_op, theta, D):
    """Performs the sensitivity analysis on each valid window.
    `D` must be expressed in the unit of habitat quality in `window_op`.
    """
    for x_start, y_start, hab_qual in window_op.iterate_windows():
        # Build grid graph and calculate Euclidean distances
        activities = hab_qual > 0
        
        def connectivity(hab_qual):
            grid = GridGraph(activities, hab_qual)
            A = grid.adjacency_matrix().todense()
            # TODO: need to iterate through the connected components
            # for now, we only take the largest component
            vertices = jax.lax.stop_gradient(_get_vertices_largest_component(A))
            # A = BCSR.from_bcoo(A)
            _A = BCOO.fromdense(A[vertices, :][:, vertices]) # this operation is very slow, we should change it
            dist = rsp_distance(_A, theta)
            K = jnp.exp(-dist / D)
            active_ij = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active()))
            q = hab_qual[active_ij[:,0], active_ij[:,1]]
            func = functional_habitat(q, K)
            return func
    
        # TODO: Victor, you stopped here
        grad_connectivity = jit(grad(connectivity))
        dcon = grad_connectivity(hab_qual)

        # Store results into the core window area of the output array
        core_range = slice(window_op.buffer_size, window_op.buffer_size + window_op.window_size)
        window_op.output_array = window_op.output_array.at[
            x_start + core_range, y_start + core_range
        ].set(dcon[core_range, core_range])

    return window_op.output_array