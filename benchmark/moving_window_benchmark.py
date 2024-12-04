"""
Running sensitivity analysis for Rupicapra rupicapra.
"""
import jax
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.gridgraph import GridGraph
from jaxscape.lcp_distance import _bellman_ford
import equinox

def make_raster(N=1002):
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    return jr.uniform(key, (N, N), dtype="float32")  # Start with a uniform permeability

def calculate_ech_scan(habitat_permability):
    activities = jnp.ones_like(habitat_permability, dtype="bool")
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=habitat_permability.size)
    
    ech = jnp.array(0, dtype=habitat_permability.dtype)
    A = grid.get_adjacency_matrix()
    
    @equinox.filter_checkpoint
    def body_fun(ech, source):
        dist = _bellman_ford(A, source)
        return ech + dist.sum(), None

    ech, _ = lax.scan(body_fun, ech, jnp.arange(1))
    # todo: modify to use landmarks instead of 1
    return ech

calculate_d_ech_dp_scan = equinox.filter_jit(equinox.filter_grad(calculate_ech_scan)) # sensitivity to permeability

# Parallelize over devices, processing one window per device
@jax.pmap
def run_calculation_pmap(hab_qual, x_start, y_start):
    res = calculate_d_ech_dp_scan(hab_qual)
    return x_start, y_start, res

def update_raster(raster, x_starts, y_starts, res_array):
    for x_start, y_start, res in zip(x_starts.tolist(), y_starts.tolist(), res_array):
        x0 = x_start
        y0 = y_start
        x1 = x0 + res.shape[0]
        y1 = y0 + res.shape[1]
        raster = raster.at[x0:x1, y0:y1].set(res)
    return raster

if __name__ == "__main__":
    window_size = 200
    buffer_size = 1
    threshold = 0.1 # todo: to use
    result_path = Path("./results")
    device = "cpu"

    permeability = make_raster()

    window_op = WindowOperation(
        shape=permeability.shape, window_size=window_size, buffer_size=buffer_size
    )

    # Collect windows
    x_starts = []
    y_starts = []
    hab_quals = []

    for window in window_op.iterate_windows(permeability):
        x_start, y_start, hab_qual = window
        x_starts.append(x_start)
        y_starts.append(y_start)
        hab_quals.append(hab_qual)

    x_starts = jnp.array(x_starts)
    y_starts = jnp.array(y_starts)
    hab_quals = jnp.stack(hab_quals)

    num_devices = jax.local_device_count()
    print("Number of devices:", num_devices)

    # Initialize the raster with NaNs
    raster = jnp.full_like(permeability, jnp.nan)

    num_windows = hab_quals.shape[0]

    # Process windows in batches of num_devices
    for i in range(0, num_windows, num_devices):
        # Extract the current batch
        hab_quals_batch = hab_quals[i:i+num_devices]
        x_starts_batch = x_starts[i:i+num_devices]
        y_starts_batch = y_starts[i:i+num_devices]

        actual_batch_size = hab_quals_batch.shape[0]

        # Pad the batch if it's smaller than num_devices
        if actual_batch_size < num_devices:
            pad_size = num_devices - actual_batch_size
            # Pad hab_quals_batch
            pad_shape = [(0, pad_size)] + [(0, 0)] * (hab_quals_batch.ndim - 1)
            hab_quals_batch = jnp.pad(hab_quals_batch, pad_shape, mode='constant')
            x_starts_batch = jnp.pad(x_starts_batch, [(0, pad_size)], mode='constant')
            y_starts_batch = jnp.pad(y_starts_batch, [(0, pad_size)], mode='constant')

        # Run the calculation in parallel across devices
        x_starts_pmap, y_starts_pmap, res_pmap = run_calculation_pmap(
            hab_quals_batch, x_starts_batch, y_starts_batch
        )

        # Remove padding if any
        if actual_batch_size < num_devices:
            x_starts_pmap = x_starts_pmap[:actual_batch_size]
            y_starts_pmap = y_starts_pmap[:actual_batch_size]
            res_pmap = res_pmap[:actual_batch_size]

        # Update the raster with the results from this batch
        raster = update_raster(raster, x_starts_pmap, y_starts_pmap, res_pmap)

    # Display the resulting raster
    cbar = plt.imshow(raster, vmin=-1000)
    plt.colorbar(cbar)
    plt.show()
