"""
Running sensitivity analysis for Rupicapra rupicapra.
"""
import jax
import numpy as np
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.gridgraph import GridGraph
from jaxscape.lcp_distance import bellman_ford
import equinox
from tqdm import tqdm

def make_raster(N=1010):
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    return jr.uniform(key, (N, N), minval=0.1, maxval=0.9, dtype="float32")  # Start with a uniform permeability

def calculate_connectivity_scan(habitat_permability, sources):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""
    activities = jnp.ones_like(habitat_permability, dtype="bool")
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=habitat_permability.size)
    
    A = grid.get_adjacency_matrix()
    connectivity = jnp.array(0, dtype=habitat_permability.dtype)
    @equinox.filter_checkpoint
    def body_fun(connectivity, source):
        dist = bellman_ford(A, source)
        connectivity = connectivity + jnp.sum(jnp.exp(-dist/dist.size/2))
        return connectivity, None

    connectivity, _ = lax.scan(body_fun, connectivity, sources)
    return connectivity

calculate_d_connectivity_dp_scan = equinox.filter_jit(equinox.filter_grad(calculate_connectivity_scan)) # sensitivity to permeability

# Parallelize over devices, processing one window per device
@equinox.filter_pmap
def run_calculation_pmap(hab_qual, x_start, y_start, sources):
    res = calculate_d_connectivity_dp_scan(hab_qual, sources)
    return x_start, y_start, res

if __name__ == "__main__":
    N = 610
    window_size = 100
    buffer_size = 5
    n_landmarks = 20 # number of lamdmarks for each window, that we place uniformaly across
    result_path = Path("./results/WindowOperation")
    result_path.mkdir(parents=True, exist_ok=True)

    permeability = make_raster(N)

    window_op = WindowOperation(
        shape=permeability.shape, 
        window_size=window_size, 
        buffer_size=buffer_size
    )
    sources = jnp.linspace(0, window_op.window_size**2, n_landmarks, dtype="int32")
    
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
    for i in tqdm(range(0, num_windows, num_devices), desc="Batch progress"):
        # Extract the current batch
        hab_quals_batch = hab_quals[i:i+num_devices]
        x_starts_batch = x_starts[i:i+num_devices]
        y_starts_batch = y_starts[i:i+num_devices]
        sources_batch = jnp.broadcast_to(sources, (num_devices,) + sources.shape)

        # Run the calculation in parallel across devices
        x_starts_pmap, y_starts_pmap, res_pmap = run_calculation_pmap(
            hab_quals_batch, x_starts_batch, y_starts_batch, sources_batch
        )

        for x_start, y_start, res in zip(x_starts_pmap, y_starts_pmap, res_pmap):
            raster = window_op.update_raster(x_start, y_start, raster, res)


    fig, ax = plt.subplots()
    cbar = ax.imshow(raster, vmax=1.)
    fig.colorbar(cbar, ax=ax)
    plt.show()
    fig.savefig(result_path / "lcp_moving_window.png", dpi=400)