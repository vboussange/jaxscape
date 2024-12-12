"""
Running sensitivity analysis for Rupicapra rupicapra.
This script copies the behavior of omniscape.
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
from jaxscape.lcp_distance import LCPDistance
import equinox
from tqdm import tqdm

def make_raster(N=1010):
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    return jr.uniform(key, (N, N), minval=0.1, maxval=0.9, dtype="float32")  # Start with a uniform permeability

def calculate_connectivity(habitat_permability, distance):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""
    # TODO: this is a hacky fix
    # print(habitat_permability.shape)
    activities = jnp.ones_like(habitat_permability, dtype="bool")
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=habitat_permability.size)
    # TODO: this is a hacky fix
    window_center = jnp.array([[habitat_permability.size//2+1, habitat_permability.size//2+1]])
    dist = distance(grid, window_center)
    connectivity = jnp.sum(jnp.exp(-dist/dist.size/2))
    return connectivity

connectivity_grad = equinox.filter_jit(equinox.filter_grad(calculate_connectivity)) # sensitivity to permeability

# TODO: not used
run_calculation_vmap = equinox.filter_vmap(connectivity_grad)

def batch_run_calculation(window_op, distance, raster, hab_qual, x_start, y_start):
    res = run_calculation_vmap(hab_qual, distance)
    
    # TODO: replace by a lax.scan loop
    for i, (xx, yy) in enumerate(zip(x_start, y_start)):
        raster = window_op.update_raster_with_window(xx, yy, raster, res[i, ...])

    return raster

if __name__ == "__main__":
    N = 100
    window_size = 1
    buffer_size = 20
    result_path = Path("./results/WindowOperation2")
    result_path.mkdir(parents=True, exist_ok=True)

    permeability = make_raster(N)
    distance = LCPDistance()

    window_op = WindowOperation(
        shape=permeability.shape, 
        window_size=window_size, 
        buffer_size=buffer_size
    )
    # TODO: implement batch processing
    batch_size = 10 # for now, we process one window at a time

    raster = jnp.full_like(permeability, jnp.nan)

    for window in tqdm(window_op.iterate_windows(permeability), desc="Batch progress"):
        x_start, y_start, hab_qual = window
        # TODO: to be replaced when using proper batches
        hab_qual = jnp.expand_dims(hab_qual, axis=0)
        x_start = [x_start]
        y_start = [y_start]
        
        raster = batch_run_calculation(window_op, distance, raster, hab_qual, x_start, y_start)

    fig, ax = plt.subplots()
    cbar = ax.imshow(raster, vmax=1.)
    fig.colorbar(cbar, ax=ax)
    plt.show()
    fig.savefig(result_path / "lcp_moving_window.png", dpi=400)