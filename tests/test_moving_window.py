import pytest
import jax.numpy as jnp
from jaxscape.moving_window import WindowOperation  # Adjust as necessary for your module path

@pytest.fixture
def sample_raster_data():
    """Provides a sample raster data array for testing."""
    return jnp.array([
        [1.0, jnp.nan, 3.0, 4.0],
        [5.0, 6.0, jnp.nan, 8.0],
        [9.0, 10.0, 11.0, jnp.nan],
        [13.0, 14.0, 15.0, 16.0]
    ])

@pytest.fixture
def window_op(sample_raster_data):
    """Creates a WindowOperation instance with sample data."""
    return WindowOperation(shape=sample_raster_data.shape, window_size=2, buffer_size=1)


def test_initialization(sample_raster_data):
    """Test proper initialization of attributes."""
    window_size = 2
    buffer_size = 1
    op = WindowOperation(shape=sample_raster_data.shape, window_size=window_size, buffer_size=buffer_size)
    
    assert op.window_size == window_size
    assert op.buffer_size == buffer_size
    assert op.total_window_size == window_size + 2 * buffer_size
    assert op.output_array.shape == sample_raster_data.shape
    assert jnp.isnan(op.output_array).all()
    assert op.x_steps == 1  # Derived from data shape
    assert op.y_steps == 1


def test_extract_window(window_op, sample_raster_data):
    """Test window extraction with buffering and NaN replacement."""
    x_start, y_start = 0, 0
    result = window_op.extract_window(x_start, y_start, sample_raster_data)
    
    expected = jnp.array([
         [1.0, jnp.nan, 3.0, 4.0],
        [5.0, 6.0, jnp.nan, 8.0],
        [9.0, 10.0, 11.0, jnp.nan],
        [13.0, 14.0, 15.0, 16.0]
    ])
    assert jnp.array_equal(result[~jnp.isnan(result)], expected[~jnp.isnan(expected)]), "extract_window did not extract the correct window with buffering"


def test_nb_steps(window_op):
    """Test that the number of steps is calculated correctly."""
    assert window_op.nb_steps == 1, "nb_steps calculation is incorrect"


def test_iterate_windows(window_op, sample_raster_data):
    """Test iterate_windows method yields correct windows."""
    windows = list(window_op.iterate_windows(sample_raster_data))
    
    assert len(windows) == 1, "iterate_windows did not yield the expected number of windows"
    
    x_start, y_start, _ = windows[0]
    assert x_start == 0
    assert y_start == 0