"""
We represent euclidean distance and ECH sensitivity in a simple setting
# TODO: you want to add the scale here
"""
import jax
import jax.numpy as jnp
from jaxscape.lcp_distance import LCPDistance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
import matplotlib.pyplot as plt
from pathlib import Path
import equinox

path_results = Path("results/ResistanceDistance")
path_results.mkdir(parents=True, exist_ok=True)

# Define a habitat permeability raster
N = 20
corridor_length = 6
corridor_width = 4
habitat_permability = jnp.ones((N, N), dtype="float32")
activities = habitat_permability > 0
habitat_quality = jnp.ones((N, N), dtype="float32") * 0
habitat_quality = habitat_quality.at[1, 1].set(1.)
habitat_quality = habitat_quality.at[N-2, N-2].set(1.)
nb_active = int(activities.sum())
plt.imshow(habitat_quality)
distance = LCPDistance()

def calculate_ech(habitat_permability, habitat_quality, activities, D, nb_active):
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=nb_active)
    dist = distance(grid)
    # scaling
    # dist = dist / dist.max()
    proximity = jnp.exp(-dist / D)
    landscape = ExplicitGridGraph(activities=activities, 
                                  vertex_weights=habitat_quality, 
                                  adjacency_matrix=proximity,
                                  nb_active=nb_active)

    ech = landscape.equivalent_connected_habitat()
    return ech

calculate_ech(habitat_permability, habitat_quality, activities, D, nb_active).shape

calculate_d_ech_dp = equinox.filter_jit(equinox.filter_grad(calculate_ech)) # sensitivity to permeability
calculate_d_ech_dq = equinox.filter_jit(jax.grad(calculate_ech, argnums=1)) # sensitivity to quality