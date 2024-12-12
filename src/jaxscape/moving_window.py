import jax
from jax import vmap
import jax.numpy as jnp
import equinox

class WindowOperation:
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

    # def replace_missing(self, array, value=jnp.nan):
    #     """Replace missing data in the array with specified value."""
    #     return jnp.where(jnp.isnan(array), value, array)

    def extract_window(self, xy, raster):
        """Extract a buffered window from the raster data."""
        x_start, y_start = xy
        window = raster[
            x_start:x_start + self.total_window_size,
            y_start:y_start + self.total_window_size
        ]
        return window
    
    def update_raster_with_focal_window(self, xy, raster, raster_window):
        """Updates `raster` with the inner core (focal pixels) of `raster_window`."""
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy
        
        x_core_start = x_start + self.buffer_size
        x_core_end = x_core_start + self.window_size
        y_core_start = y_start + self.buffer_size
        y_core_end = y_core_start + self.window_size
        
        focal_window = raster_window[self.buffer_size:self.buffer_size + self.window_size,
                                self.buffer_size:self.buffer_size + self.window_size]

        # Update the output array within the specified core region
        return raster.at[
            x_core_start:x_core_end, y_core_start:y_core_end
        ].set(focal_window)
        
    
    def update_raster_with_window(self, xy, raster, raster_window):
        """Updates `raster` with `raster_window`."""
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy
        # Update the output array within the specified core region
        return raster.at[
            x_start:x_start+self.total_window_size, 
            y_start:y_start+self.total_window_size
        ].set(raster_window)

    def add_window_to_raster(self, xy, raster, raster_window):
        """Modify `raster` by adding `raster_window`."""
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy

        # Update the output array within the specified core region
        return raster.at[
            x_start:x_start+self.total_window_size, 
            y_start:y_start+self.total_window_size
        ].add(raster_window)
    
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
                window = self.extract_window([x_start, y_start], raster)
                yield jnp.array([x_start, y_start]), window

    @equinox.filter_jit
    
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
        xs_flat = xs.ravel()
        ys_flat = ys.ravel()
        xy = jnp.stack([xs_flat, ys_flat], axis=-1)

        # Use vmap to extract each window
        windows = vmap(lambda x, y: jax.lax.dynamic_slice(matrix, (x, y), (window_height, window_width)))(xs_flat, ys_flat)

        return xy, windows
             
    # TODO: this may not be needed when combining both lazy and eager iterators   
    # def iterate_window_batches(self, raster, batch_size):
    #     """Yield batches of buffered windows with a batch dimension."""
    #     batch_x_starts = []
    #     batch_y_starts = []
    #     batch_windows = []

    #     for i in range(self.x_steps):
    #         for j in range(self.y_steps):
    #             x_start = i * self.window_size
    #             y_start = j * self.window_size
    #             window = self.extract_window(x_start, y_start, raster)

    #             batch_x_starts.append(x_start)
    #             batch_y_starts.append(y_start)
    #             batch_windows.append(window)

    #             if len(batch_windows) == batch_size:
    #                 window_batch = jnp.stack(batch_windows, axis=0)
    #                 yield batch_x_starts, batch_y_starts, window_batch
    #                 batch_x_starts = []
    #                 batch_y_starts = []
    #                 batch_windows = []

    #     if batch_windows:
    #         window_batch = jnp.stack(batch_windows, axis=0)
    #         yield batch_x_starts, batch_y_starts, window_batch