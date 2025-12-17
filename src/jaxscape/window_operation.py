from collections.abc import Callable, Generator

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax, vmap


class WindowOperation(eqx.Module):
    """Manages window-based operations on raster data with buffering.

    Used for processing large rasters by dividing them into smaller windows with
    overlapping buffer regions. Ensures each window has sufficient context for
    operations that depend on neighboring pixels.

    **Attributes:**

    - `shape`: Raster dimensions `(height, width)`.
    - `window_size`: Core window size in pixels.
    - `buffer_size`: Buffer region size around each core window.
    - `total_window_size`: Total window size including buffers `(window_size + 2 * buffer_size)`.
    - `x_steps`, `y_steps`: Number of windows in each dimension.

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import WindowOperation

        raster = jnp.ones((100, 100))
        window_op = WindowOperation(
            shape=raster.shape,
            window_size=20,
            buffer_size=10
        )
        ```

    !!! warning
        You must ensure that `(shape[i] - 2 * buffer_size)` is divisible by
        `window_size` for both dimensions `i = 0, 1`. Consider using [`jaxscape.utils.padding`][jaxscape.utils.padding]
        to pad your raster data automatically.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    total_window_size: int = eqx.field(static=True)
    x_steps: int = eqx.field(static=True)
    y_steps: int = eqx.field(static=True)

    def __init__(self, shape: tuple[int, int], window_size: int, buffer_size: int):
        assert isinstance(shape, tuple)
        assert isinstance(window_size, int)
        assert isinstance(buffer_size, int)
        for i in range(2):
            assert (
                (shape[i] - 2 * buffer_size) % window_size == 0
            ), f"`(shape[{i}] - 2 * buffer_size)`  must be divisible by `window_size`, consider padding the raster data."

        self.shape = shape
        # TODO: window size should be an arbitrary shape
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.total_window_size = self.window_size + 2 * self.buffer_size
        self.x_steps = int((shape[0] - 2 * buffer_size) // window_size)
        self.y_steps = int((shape[1] - 2 * buffer_size) // window_size)
        assert (
            self.x_steps > 0
        ), "`window_size` or `buffer_size` are too large for the raster data."
        assert (
            self.y_steps > 0
        ), "`window_size` or `buffer_size` are too large for the raster data."

    def extract_total_window(self, xy: Array, raster: Array) -> Array:
        """Extract a window including buffer regions from the raster.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the window.
        - `raster`: 2D raster array.

        **Returns:**

        Window of shape `(total_window_size, total_window_size)`.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            raster = jnp.ones((100, 100))
            window = window_op.extract_total_window(jnp.array([0, 0]), raster)
            # window.shape = (40, 40)
            ```
        """
        x_start, y_start = xy

        slice_shape = (self.total_window_size, self.total_window_size)

        window = lax.dynamic_slice(
            raster, start_indices=(x_start, y_start), slice_sizes=slice_shape
        )

        return window

    def extract_core_window(self, xy: Array, raster: Array) -> Array:
        """Extract the core window without buffers from the raster.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the total window.
        - `raster`: 2D raster array.

        **Returns:**

        Core window of shape `(window_size, window_size)`.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            raster = jnp.ones((100, 100))
            core = window_op.extract_core_window(jnp.array([0, 0]), raster)
            # core.shape = (20, 20)
            ```
        """
        x_start, y_start = xy

        x_core_start = x_start + self.buffer_size
        y_core_start = y_start + self.buffer_size

        window = lax.dynamic_slice(
            raster,
            start_indices=(x_core_start, y_core_start),
            slice_sizes=(self.window_size, self.window_size),
        )

        return window

    @eqx.filter_jit
    def update_raster_with_core_window(
        self,
        xy: Array,
        raster: Array,
        raster_window: Array,
        fun: Callable[[Array, Array], Array] = lambda current_raster_core_window,
        raster_window_core_window: raster_window_core_window,
    ) -> Array:
        """Update raster by merging the core region of a processed window.

        Extracts the core (non-buffer) region from `raster_window` and updates the
        corresponding region in `raster` using the provided function.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the total window.
        - `raster`: Full raster array to update.
        - `raster_window`: Processed window including buffers.
        - `fun`: Function to combine current and new values. Defaults to replacement.

        **Returns:**

        Updated raster array.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            raster = jnp.zeros((100, 100))

            for xy, window in window_op.lazy_iterator(raster):
                processed = compute_distance(window)
                raster = window_op.update_raster_with_core_window(xy, raster, processed)
            ```
        """
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy

        x_core_start = x_start + self.buffer_size
        y_core_start = y_start + self.buffer_size

        raster_core_window = self.extract_core_window(xy, raster)

        raster_window_core_window = lax.dynamic_slice(
            raster_window,
            start_indices=(self.buffer_size, self.buffer_size),
            slice_sizes=(self.window_size, self.window_size),
        )

        updated_slice = fun(raster_core_window, raster_window_core_window)

        # Update the output array within the specified core region
        return lax.dynamic_update_slice(
            raster, updated_slice, start_indices=(x_core_start, y_core_start)
        )

    @eqx.filter_jit
    def update_raster_with_window(
        self,
        xy: Array,
        raster: Array,
        raster_window: Array,
        fun: Callable[[Array, Array], Array] = lambda current_raster_slice,
        raster_window: raster_window,
    ) -> Array:
        """Update raster with the entire window including buffers.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the window.
        - `raster`: Full raster array to update.
        - `raster_window`: Processed window to merge.
        - `fun`: Function to combine current and new values. Defaults to replacement.

        **Returns:**

        Updated raster array.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            raster = jnp.zeros((100, 100))
            window_data = jnp.ones((40, 40))

            # Replace window region
            raster = window_op.update_raster_with_window(
                jnp.array([0, 0]), raster, window_data
            )

            # Accumulate with custom function
            raster = window_op.update_raster_with_window(
                jnp.array([0, 0]), raster, window_data, fun=jnp.add
            )
            ```
        """
        assert isinstance(raster, jax.Array)
        x_start, y_start = xy

        current_slice = self.extract_total_window(xy, raster)

        # use it for update, if needed
        updated_slice = fun(current_slice, raster_window)

        updated_raster = lax.dynamic_update_slice(
            raster, updated_slice, start_indices=(x_start, y_start)
        )

        return updated_raster

    @property
    def nb_steps(self) -> int:
        """Total number of windows in the raster.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            print(window_op.nb_steps)  # 25 (5x5 grid of windows)
            ```
        """
        return (self.x_steps) * (self.y_steps)

    def lazy_iterator(
        self, raster: Array
    ) -> Generator[tuple[Array, Array], None, None]:
        """Iterate over windows one at a time.

        Memory-efficient iteration that yields windows sequentially without
        pre-computing all windows.

        **Arguments:**

        - `raster`: 2D raster array to iterate over.

        **Yields:**

        Tuples of `(xy, window)` where `xy` are start coordinates and `window`
        is the extracted window with buffers.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            raster = jnp.ones((100, 100))

            for xy, window in window_op.lazy_iterator(raster):
                # Process each window sequentially
                result = compute(window)
                # window.shape = (40, 40)
            ```
        """
        x_steps = self.x_steps
        y_steps = self.y_steps
        for i in range(x_steps):
            for j in range(y_steps):
                x_start, y_start = i * self.window_size, j * self.window_size
                window = self.extract_total_window([x_start, y_start], raster)
                yield jnp.array([x_start, y_start]), window

    @eqx.filter_jit
    def eager_iterator(self, matrix: Array) -> tuple[Array, Array]:
        """Extract all windows at once for parallel processing.

        Pre-computes all windows in a single operation using `vmap`, enabling
        efficient batch processing and parallelization.

        **Arguments:**

        - `matrix`: 2D input raster array.

        **Returns:**

        Tuple `(xy, windows)` where `xy` has shape `(num_windows, 2)` containing
        start coordinates, and `windows` has shape `(num_windows, window_height, window_width)`.

        !!! example

            ```python
            window_op = WindowOperation(shape=(100, 100), window_size=20, buffer_size=10)
            raster = jnp.ones((100, 100))

            xy, windows = window_op.eager_iterator(raster)
            # xy.shape = (25, 2), windows.shape = (25, 40, 40)

            # Process all windows in parallel
            results = jax.vmap(compute)(windows)
            ```
        """
        # TODO: to change
        window_height, window_width = self.total_window_size, self.total_window_size

        # Compute the valid start ranges
        startsx = jnp.arange(self.x_steps)
        startsy = jnp.arange(self.y_steps)

        # Create a grid of (x, y) start points
        ys, xs = jnp.meshgrid(startsy, startsx, indexing="xy")
        xs_flat = xs.ravel() * self.window_size
        ys_flat = ys.ravel() * self.window_size
        xy = jnp.stack([xs_flat, ys_flat], axis=-1)

        # Use vmap to extract each window
        windows = vmap(
            lambda x, y: jax.lax.dynamic_slice(
                matrix, (x, y), (window_height, window_width)
            )
        )(xs_flat, ys_flat)

        return xy, windows
