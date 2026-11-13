from collections.abc import Callable, Generator

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax, vmap


class WindowOperation(eqx.Module):
    """Division of a raster into square core windows, each with a buffer.

    A large raster is processed one total window at a time. A total window is
    a core window with its buffer, the band of `buffer_size` pixels around it,
    which overlaps the neighbouring total windows: it holds the pixels that a
    computation on the core window depends on.

    **Attributes:**

    - `shape`: Raster shape `(height, width)`.
    - `window_size`: Side of a core window, in pixels.
    - `buffer_size`: Width of the buffer around each core window, in pixels.
    - `total_window_size`: Side of a total window, a core window with its
      buffer, `window_size + 2 * buffer_size`.
    - `x_steps`, `y_steps`: Number of total windows along axis 0, the height,
      and along axis 1, the width.

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
        `shape[i] - 2 * buffer_size` must be divisible by `window_size` along
        both axes `i = 0, 1`.
        [`jaxscape.utils.padding`][jaxscape.utils.padding] pads a raster to such
        a shape.
    """

    shape: tuple[int, int] = eqx.field(static=True)
    window_size: int = eqx.field(static=True)
    buffer_size: int = eqx.field(static=True)
    total_window_size: int = eqx.field(static=True)
    x_steps: int = eqx.field(static=True)
    y_steps: int = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], window_size: int, buffer_size: int):
        assert isinstance(shape, tuple) and len(shape) == 2, (
            f"`shape` must be (height, width), got {shape}."
        )
        assert isinstance(window_size, int)
        assert isinstance(buffer_size, int)
        for i in range(2):
            assert (shape[i] - 2 * buffer_size) % window_size == 0, (
                f"`(shape[{i}] - 2 * buffer_size)`  must be divisible by "
                "`window_size`, consider padding the raster data."
            )

        self.shape = shape
        # TODO: window size should be an arbitrary shape
        self.window_size = window_size
        self.buffer_size = buffer_size
        self.total_window_size = self.window_size + 2 * self.buffer_size
        self.x_steps = int((shape[0] - 2 * buffer_size) // window_size)
        self.y_steps = int((shape[1] - 2 * buffer_size) // window_size)
        assert self.x_steps > 0, (
            "`window_size` or `buffer_size` are too large for the raster data."
        )
        assert self.y_steps > 0, (
            "`window_size` or `buffer_size` are too large for the raster data."
        )

    def extract_total_window(self, xy: Array, raster: Array) -> Array:
        """Extract a total window, a core window with its buffer, from the raster.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the total window.
        - `raster`: 2D raster array.

        **Returns:**

        Total window of shape `(total_window_size, total_window_size)`.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
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
        """Extract a core window, without its buffer, from the raster.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the total window.
        - `raster`: 2D raster array.

        **Returns:**

        Core window of shape `(window_size, window_size)`.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
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
        fun: Callable[
            [Array, Array], Array
        ] = lambda current_raster_core_window, raster_window_core_window: (
            raster_window_core_window
        ),
    ) -> Array:
        """Update a raster with the core window of a processed total window.

        Extracts the core window from `raster_window` and updates the
        corresponding core window of `raster` using the provided function.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the total window.
        - `raster`: Full raster array to update.
        - `raster_window`: Processed total window.
        - `fun`: Function to combine current and new values. Defaults to replacement.

        **Returns:**

        Updated raster array.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
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
        fun: Callable[
            [Array, Array], Array
        ] = lambda current_raster_slice, raster_window: raster_window,
    ) -> Array:
        """Update a raster with a whole processed total window.

        **Arguments:**

        - `xy`: Start coordinates `[x, y]` of the total window.
        - `raster`: Full raster array to update.
        - `raster_window`: Processed total window to merge.
        - `fun`: Function to combine current and new values. Defaults to replacement.

        **Returns:**

        Updated raster array.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
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
        """Number of total windows in the raster.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
            print(window_op.nb_steps)  # 16 (4x4 grid of total windows)
            ```
        """
        return (self.x_steps) * (self.y_steps)

    def lazy_iterator(
        self, raster: Array
    ) -> Generator[tuple[Array, Array], None, None]:
        """Iterate over the total windows one at a time.

        Yields the total windows one after another, so that only one is
        extracted at a time.

        **Arguments:**

        - `raster`: 2D raster array to iterate over.

        **Yields:**

        Tuples of `(xy, window)` where `xy` are start coordinates and `window`
        is the extracted total window.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
            raster = jnp.ones((100, 100))

            for xy, window in window_op.lazy_iterator(raster):
                # Process each total window sequentially
                result = compute(window)
                # window.shape = (40, 40)
            ```
        """
        x_steps = self.x_steps
        y_steps = self.y_steps
        for i in range(x_steps):
            for j in range(y_steps):
                xy = jnp.array([i * self.window_size, j * self.window_size])
                yield xy, self.extract_total_window(xy, raster)

    @eqx.filter_jit
    def eager_iterator(self, matrix: Array) -> tuple[Array, Array]:
        """Extract all total windows at once, for batched processing.

        Extracts the total windows in one operation with
        `jax.vmap`, so that a computation can then run on all of them at once.

        **Arguments:**

        - `matrix`: 2D input raster array.

        **Returns:**

        Tuple `(xy, windows)` where `xy` has shape `(num_windows, 2)` containing
        start coordinates, and `windows` has shape
        `(num_windows, total_window_size, total_window_size)`.

        !!! example

            ```python
            window_op = WindowOperation(
                shape=(100, 100), window_size=20, buffer_size=10
            )
            raster = jnp.ones((100, 100))

            xy, windows = window_op.eager_iterator(raster)
            # xy.shape = (16, 2), windows.shape = (16, 40, 40)

            # Process all total windows in parallel
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
