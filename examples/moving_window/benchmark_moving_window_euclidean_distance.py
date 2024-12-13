"""
Running sensitivity analysis of equivalent connected habitat for euclidean distance.
This script copies the behavior of omniscape.
"""
from jax import lax
import jax.numpy as jnp
from jaxscape.moving_window import WindowOperation
import jax.random as jr
from jaxscape.euclidean_distance import EuclideanDistance
# from jaxscape.lcp_distance import LCPDistance
from tqdm import tqdm
import equinox as eqx
from jaxscape.gridgraph import GridGraph
import matplotlib.pyplot as plt


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

connectivity_grad = eqx.filter_jit(eqx.filter_grad(calculate_connectivity))

run_calculation_vmap = eqx.filter_vmap(connectivity_grad)

@eqx.filter_jit
def batch_run_calculation(window_op, xy, hab_qual, activities, distance, raster_buffer):
    res = run_calculation_vmap(hab_qual, activities, distance)
    def scan_fn(raster_buffer, x):
        _xy, _rast = x
        raster_buffer = window_op.update_raster_with_window(_xy, raster_buffer, _rast, fun=jnp.add)
        return raster_buffer, None
    raster_buffer, _ = lax.scan(scan_fn, raster_buffer, (xy, res))
    return raster_buffer

if __name__ == "__main__":
    N = 2000
    window_size = 1 # must be odd to be placed at the center
    buffer_size = 10
    batch_size = 660
    
    permeability = make_raster(N)

    batch_op = WindowOperation(
        shape=permeability.shape, 
        window_size=batch_size, 
        buffer_size=buffer_size
    )

    permeability = make_raster(N)
    distance = EuclideanDistance()
    
    output = jnp.zeros_like(permeability) # initialize raster

    for (xy_batch, permeability_batch) in tqdm(batch_op.lazy_iterator(permeability), desc="Batch progress"):
        window_op = WindowOperation(shape=permeability_batch.shape, 
                                    window_size=window_size, 
                                    buffer_size=buffer_size)
        xy, hab_qual = window_op.eager_iterator(permeability_batch)
        activities = jnp.ones_like(hab_qual, dtype="bool")
        raster_buffer = jnp.zeros_like(permeability_batch)
        res = batch_run_calculation(window_op, xy, hab_qual, activities, distance, raster_buffer)
        output = batch_op.update_raster_with_window(xy_batch, output, res, fun=jnp.add)
    
    fig, ax = plt.subplots()
    cbar = ax.imshow(output)
    fig.colorbar(cbar, ax=ax)
    plt.show()