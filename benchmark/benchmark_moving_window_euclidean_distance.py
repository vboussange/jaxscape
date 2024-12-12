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
from jaxscape.euclidean_distance import EuclideanDistance
import equinox
from tqdm import tqdm

def make_raster(N=1010):
    key = jr.PRNGKey(1)
    return jr.uniform(key, (N, N), minval=0.1, maxval=0.9, dtype="float32")

def calculate_connectivity(habitat_permability, activities, distance):
    """For simplicitity, we calculate connectivity as the sum of the inverse of the exp of distances."""

    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=habitat_permability.size)

    window_center = jnp.array([[habitat_permability.shape[0]//2+1, habitat_permability.shape[1]//2+1]])
    
    q = grid.array_to_node_values(habitat_permability)
    dist = distance(grid, targets = window_center)
    K = jnp.exp(-dist/dist.size/2) # calculating proximity matrix
    
    qKqT = habitat_permability[window_center[0, 0], window_center[0, 1]] * (K.T @ q)
    connectivity = jnp.sqrt(jnp.sum(qKqT))
    return connectivity

connectivity_grad = equinox.filter_jit(equinox.filter_grad(calculate_connectivity))

run_calculation_vmap = equinox.filter_vmap(connectivity_grad)

def batch_run_calculation(window_op, distance, raster, hab_qual, x_start, y_start, activities):
    res = run_calculation_vmap(hab_qual, activities, distance)
    
    # TODO: replace by a lax.scan loop
    # but if you do that, you need to define xx and yy as static arguments
    for i, (xx, yy) in enumerate(zip(x_start, y_start)):
        raster = window_op.add_window_to_raster(xx, yy, raster, res[i, ...])
    return raster

if __name__ == "__main__":
    N = 100
    window_size = 1 # must be odd to be placed at the center
    buffer_size = 10
    batch_size = 100 
    
    result_path = Path("./results/WindowOperation2")
    result_path.mkdir(parents=True, exist_ok=True)

    permeability = make_raster(N)
    quality = make_raster(N)

    distance = EuclideanDistance()

    window_op = WindowOperation(
        shape=permeability.shape, 
        window_size=window_size, 
        buffer_size=buffer_size
    )

    raster = jnp.zeros_like(permeability)

    for (x_start, y_start, hab_qual) in tqdm(window_op.iterate_window_batches(permeability, batch_size), desc="Batch progress"):
        activities = jnp.ones((hab_qual.shape[0], 2*buffer_size+window_size, 2*buffer_size+window_size), dtype="bool")
        raster = batch_run_calculation(window_op, distance, raster, hab_qual, x_start, y_start, activities)

    fig, ax = plt.subplots()
    cbar = ax.imshow(raster)
    fig.colorbar(cbar, ax=ax)
    plt.show()
    # fig.savefig(result_path / "lcp_moving_window.png", dpi=400)
    
    # On RTX 3090:

    # N = 1000
    # window_size = 5 # must be odd to be placed at the center
    # buffer_size = 20
    # batch_size = 1000 
    # takes about 02:54min to run

    # N = 1120
    # window_size = 9 # must be odd
    # buffer_size = 20
    # batch_size = 500 
    # Takes about 01:28min to run