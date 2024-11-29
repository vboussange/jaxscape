import jax
import jax.numpy as jnp


class WindowOperation:
    """Handles window-based operations on raster data."""
    def __init__(self, shape, window_size, buffer_size):
        assert isinstance(shape, tuple)
        assert isinstance(window_size, int)
        assert isinstance(buffer_size, int)
        
        self.shape = shape
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.total_window_size = self.window_size + 2 * self.buffer_size
        self.x_steps = int((shape[0] - 2 * buffer_size) // window_size)
        self.y_steps = int((shape[1] - 2 * buffer_size) // window_size)

    # def replace_missing(self, array, value=jnp.nan):
    #     """Replace missing data in the array with specified value."""
    #     return jnp.where(jnp.isnan(array), value, array)

    def extract_window(self, x_start, y_start, raster):
        """Extract a buffered window from the raster data."""
        window = raster[
            x_start:x_start + self.total_window_size,
            y_start:y_start + self.total_window_size
        ]
        return window
    
    def update_raster_from_window(self, x_start, y_start, raster, raster_window):
        """Extract a buffered window from the raster data."""
        assert isinstance(raster, jax.Array)

        # Store results into the core window area of the output array
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

    
    @property
    def nb_steps(self):
        return (self.x_steps) * (self.y_steps)

    def iterate_windows(self, raster):
        """Yield buffered windows for computation, skipping empty areas."""
        x_steps = self.x_steps
        y_steps = self.y_steps
        for i in range(x_steps):
            for j in range(y_steps):
                x_start, y_start = i * self.window_size, j * self.window_size
                window = self.extract_window(x_start, y_start, raster)
                
                yield x_start, y_start, window