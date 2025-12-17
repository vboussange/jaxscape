import jax.numpy as jnp

# Adjust as necessary for your module path
import jax.random as jr
import numpy as np
import pytest
from jaxscape.window_operation import (
    WindowOperation,
)


@pytest.fixture
def sample_raster_data():
    """Provides a sample raster data array for testing."""
    return jnp.array(
        [
            [1.0, jnp.nan, 3.0, 4.0],
            [5.0, 6.0, jnp.nan, 8.0],
            [9.0, 10.0, 11.0, jnp.nan],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )


@pytest.fixture
def window_op(sample_raster_data):
    """Creates a WindowOperation instance with sample data."""
    return WindowOperation(shape=sample_raster_data.shape, window_size=2, buffer_size=1)


def test_initialization(sample_raster_data):
    """Test proper initialization of attributes."""
    window_size = 2
    buffer_size = 1
    op = WindowOperation(
        shape=sample_raster_data.shape, window_size=window_size, buffer_size=buffer_size
    )

    assert op.window_size == window_size
    assert op.buffer_size == buffer_size
    assert op.total_window_size == window_size + 2 * buffer_size
    assert op.x_steps == 1  # Derived from data shape
    assert op.y_steps == 1


def test_extract_total_window(window_op, sample_raster_data):
    """Test window extraction with buffering and NaN replacement."""
    xy = np.array([0, 0])
    result = window_op.extract_total_window(xy, sample_raster_data)

    expected = jnp.array(
        [
            [1.0, jnp.nan, 3.0, 4.0],
            [5.0, 6.0, jnp.nan, 8.0],
            [9.0, 10.0, 11.0, jnp.nan],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    assert jnp.array_equal(
        result[~jnp.isnan(result)], expected[~jnp.isnan(expected)]
    ), "extract_total_window did not extract the correct window with buffering"


def test_extract_core_window(window_op, sample_raster_data):
    """Test window extraction with buffering and NaN replacement."""
    xy = np.array([0, 0])
    result = window_op.extract_core_window(xy, sample_raster_data)

    expected = jnp.array([[6.0, jnp.nan], [10.0, 11.0]])
    assert jnp.array_equal(
        result[~jnp.isnan(result)], expected[~jnp.isnan(expected)]
    ), "extract_core_window did not extract the correct window with buffering"


def test_nb_steps(window_op):
    """Test that the number of steps is calculated correctly."""
    assert window_op.nb_steps == 1, "nb_steps calculation is incorrect"


def test_iterate_windows(window_op, sample_raster_data):
    """Test iterate_windows method yields correct windows."""
    windows = list(window_op.lazy_iterator(sample_raster_data))

    assert (
        len(windows) == 1
    ), "iterate_windows did not yield the expected number of windows"

    xy, _ = windows[0]
    assert xy[0] == 0
    assert xy[1] == 0


# def test_iterate_windows_large():
#     raster = jnp.ones((500, 500))
#     window_size = 2
#     buffer_size = 10
#     window_op = WindowOperation(shape=raster.shape,
#                          window_size=window_size,
#                          buffer_size=buffer_size)
#     it = lambda: jax.block_until_ready(list(window_op.iterate_windows(sample_raster_data)))
#     it()
#     time = timeit.repeat(it, number=1, repeat=10)


#     """Test iterate_windows method yields correct windows."""
#     windows = list(window_op.iterate_windows(sample_raster_data))

#     assert len(windows) == 1, "iterate_windows did not yield the expected number of windows"

#     x_start, y_start, _ = windows[0]
#     assert x_start == 0
#     assert y_start == 0


def test_update_raster_with_core_window():
    """Test update_raster method updates raster correctly. This also tests the lazy_iterator method."""

    def run_calculation(window):
        return window

    def make_raster(N=102):
        key = jr.PRNGKey(0)  # Random seed is explicit in JAX
        return jr.uniform(
            key, (N, N), dtype="float32"
        )  # Start with a uniform permeability

    window_size = 20
    buffer_size = 1

    permeability = make_raster()

    window_op = WindowOperation(
        shape=permeability.shape, window_size=window_size, buffer_size=buffer_size
    )

    # use pmap distribute the computation to multiple devices
    raster = jnp.full_like(permeability, jnp.nan)
    for window in window_op.lazy_iterator(permeability):
        xy, res = run_calculation(window)
        raster = window_op.update_raster_with_core_window(xy, raster, res)

    assert jnp.allclose(raster[1:-1, 1:-1], permeability[1:-1, 1:-1])
    assert jnp.isnan(raster[0, 0])  # buffer region


# TODO: this is not needed
# def test_iterate_window_batches():
#     """Test that iterate_window_batches yields correct batches of windows."""
#     sample_raster_data = jnp.arange(100).reshape((10, 10)).astype(float)

#     # Define window operation parameters
#     window_size = 2
#     buffer_size = 1
#     batch_size = 3

#     # Initialize WindowOperation
#     window_op = WindowOperation(
#         shape=sample_raster_data.shape,
#         window_size=window_size,
#         buffer_size=buffer_size
#     )

#     # Collect the batches
#     batches = list(window_op.iterate_window_batches(sample_raster_data, batch_size))

#     # Compute expected number of windows and batches
#     x_steps = window_op.x_steps
#     y_steps = window_op.y_steps
#     total_windows = x_steps * y_steps
#     expected_batches = (total_windows + batch_size - 1) // batch_size  # Ceiling division

#     assert len(batches) == expected_batches, f"Expected {expected_batches} batches, got {len(batches)}"

#     raster = jnp.full_like(sample_raster_data, jnp.nan)

#     for batch_x_starts, batch_y_starts, window_batch in batches:
#         batch_length = len(batch_x_starts)
#         assert batch_length == window_batch.shape[0], "Inconsistent batch sizes"

#         for i in range(batch_length):
#             x_start = batch_x_starts[i]
#             y_start = batch_y_starts[i]
#             window = window_batch[i]

#             raster = window_op.update_raster_with_window(x_start, y_start, raster, window)

#     assert jnp.allclose(raster[1:-1,1:-1], sample_raster_data[1:-1,1:-1])


def test_eager_iterator():
    """Test update_raster method updates raster correctly."""

    def make_raster(N=10):
        key = jr.PRNGKey(0)  # Random seed is explicit in JAX
        return jr.uniform(
            key, (N, N), dtype="float32"
        )  # Start with a uniform permeability

    window_size = 3
    buffer_size = 2

    permeability = make_raster()

    window_op = WindowOperation(
        shape=permeability.shape, window_size=window_size, buffer_size=buffer_size
    )

    windows = window_op.eager_iterator(permeability)

    for i, (xy, w) in enumerate(window_op.lazy_iterator(permeability)):
        assert jnp.allclose(w, windows[1][i])
        assert jnp.allclose(xy, windows[0][i])


def test_update_raster_with_window():
    """Test add_window_to_raster method adds window values correctly."""

    def make_raster(N=10):
        key = jr.PRNGKey(0)  # Random seed is explicit in JAX
        return jr.uniform(
            key, (N, N), dtype="float32"
        )  # Start with a uniform permeability

    window_size = 2
    buffer_size = 1

    raster = make_raster()
    window_op = WindowOperation(
        shape=raster.shape, window_size=window_size, buffer_size=buffer_size
    )

    xy = np.array([0, 0])
    window = jnp.ones(
        (window_op.total_window_size, window_op.total_window_size), dtype="float32"
    )

    updated_raster = window_op.update_raster_with_window(
        xy, raster, window, fun=jnp.add
    )

    expected_raster = raster.at[
        xy[0] : xy[0] + window_op.total_window_size,
        xy[1] : xy[1] + window_op.total_window_size,
    ].add(window)

    assert jnp.allclose(
        updated_raster, expected_raster
    ), "add_window_to_raster did not add the window values correctly"
