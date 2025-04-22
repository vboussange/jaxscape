import jax
from jax import lax, vmap
import jax.numpy as jnp
import equinox as eqx

class WindowOperation(eqx.Module):
    shape: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    total_window_size: int = eqx.field(static=True)
    x_steps: int = eqx.field(static=True)
    y_steps: int = eqx.field(static=True)

    """Handles window-based operations on raster data."""
    def __init__(self, shape, window_size, buffer_size):
        assert isinstance(shape, tuple)
        assert isinstance(window_size, int)
        assert isinstance(buffer_size, int)
        for i in range(2):
            assert (shape[i] - 2 * buffer_size) % window_size == 0, f"`(shape[{i}] - 2 * buffer_size)`  must be divisible by `window_size`, consider padding the raster data."

        self.shape = shape
        # TODO: window size should be an arbitrary shape
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.total_window_size = self.window_size + 2 * self.buffer_size
        self.x_steps = int((shape[0] - 2 * buffer_size) // window_size)
        self.y_steps = int((shape[1] - 2 * buffer_size) // window_size)
        assert self.x_steps > 0, "`window_size` or `buffer_size` are too large for the raster data."
        assert self.y_steps > 0, "`window_size` or `buffer_size` are too large for the raster data."

    def extract_total_window(self, xy, raster):
        """Extract a buffered window from the raster data."""
        x_start, y_start = xy
        
        slice_shape = (self.total_window_size, self.total_window_size)

        window = lax.dynamic_slice(
            raster, 
            start_indices=(x_start, y_start), 
            slice_sizes=slice_shape
        )
        
        return window
    
    def extract_core_window(self, xy, raster):
        """Extract the core window from the raster data based on `xy` of total window."""
        x_start, y_start = xy
        
        x_core_start = x_start + self.buffer_size
        y_core_start = y_start + self.buffer_size
        
        window = lax.dynamic_slice(raster, 
                                    start_indices=(x_core_start, y_core_start), 
                                    slice_sizes=(self.window_size, self.window_size))
        
        return window
    
    @eqx.filter_jit
    def update_raster_with_focal_window(self, xy, raster, raster_window, fun=lambda current_raster_focal_window, raster_window_focal_window: raster_window_focal_window):
        """Updates `raster` with the inner core (focal pixels) of `raster_window`."""
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy
        
        x_core_start = x_start + self.buffer_size
        y_core_start = y_start + self.buffer_size
        
        raster_focal_window = self.extract_core_window(xy, raster)
        
        raster_window_focal_window = lax.dynamic_slice(raster_window, 
                                                        start_indices=(self.buffer_size, self.buffer_size), 
                                                        slice_sizes=(self.window_size, self.window_size))
        
        updated_slice = fun(raster_focal_window, raster_window_focal_window)


        # Update the output array within the specified core region
        return lax.dynamic_update_slice(raster, 
                                        updated_slice, 
                                        start_indices=(x_core_start, y_core_start))

    @eqx.filter_jit
    def update_raster_with_window(self, xy, raster, raster_window, fun=lambda current_raster_slice, raster_window: raster_window):
        """Modify `raster` by adding `raster_window`."""
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy

        current_slice = self.extract_total_window(xy, raster)

        # use it for update, if needed
        updated_slice = fun(current_slice, raster_window)

        updated_raster = lax.dynamic_update_slice(
            raster, 
            updated_slice, 
            start_indices=(x_start, y_start)
        )

        return updated_raster
    
    @property
    def nb_steps(self):
        return (self.x_steps) * (self.y_steps)

    def lazy_iterator(self, raster):
        """Yield buffered windows for computations."""
        x_steps = self.x_steps
        y_steps = self.y_steps
        for i in range(x_steps):
            for j in range(y_steps):
                x_start, y_start = i * self.window_size, j * self.window_size
                window = self.extract_total_window([x_start, y_start], raster)
                yield jnp.array([x_start, y_start]), window

    @eqx.filter_jit
    def eager_iterator(self, matrix):
        """Compute all windows and their coordinates at once.

        Args:
            matrix (jnp.ndarray): The 2D input array (e.g., raster data).
            window_shape (tuple): (window_height, window_width).

        Returns:
            (jnp.ndarray, jnp.ndarray, jnp.ndarray):
                x_starts: 1D array of x-coordinates of each window start
                y_starts: 1D array of y-coordinates of each window start
                windows: 3D array of shape (num_windows, window_height, window_width)
        """
        # TODO: to change
        window_height, window_width = self.total_window_size, self.total_window_size

        # Compute the valid start ranges
        startsx = jnp.arange(self.x_steps)
        startsy = jnp.arange(self.y_steps)

        # Create a grid of (x, y) start points
        ys, xs = jnp.meshgrid(startsy, startsx, indexing='xy')
        xs_flat = xs.ravel() * self.window_size
        ys_flat = ys.ravel() * self.window_size
        xy = jnp.stack([xs_flat, ys_flat], axis=-1)

        # Use vmap to extract each window
        windows = vmap(lambda x, y: jax.lax.dynamic_slice(matrix, (x, y), (window_height, window_width)))(xs_flat, ys_flat)

        return xy, windows
             