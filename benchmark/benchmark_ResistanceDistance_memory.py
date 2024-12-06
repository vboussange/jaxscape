"""
Benchmarking memory usage of sensitivity calculation based on
different implementations of the ECH using resistance distance
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1"
from jax import lax
import jax.numpy as jnp
from jaxscape.resistance_distance import resistance_distance
from jaxscape.gridgraph import GridGraph
import matplotlib.pyplot as plt
from pathlib import Path
import equinox
import jaxscape


path_results = Path("results/benchmarks/")
path_results.mkdir(parents=True, exist_ok=True)

def create_landscape(size):
    habitat_permability = jnp.ones((size, size), dtype="float32")
    activities = jnp.ones((size, size), dtype="bool")
    nb_active = int(activities.sum())
    return habitat_permability, activities, nb_active

# calculating equivalent connected habitat using scan
# this is memory efficient, but slow
@equinox.filter_jit
def calculate_ech_scan(habitat_permability, activities, nb_active):
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=nb_active)
    
    A = grid.get_adjacency_matrix()
    dist = resistance_distance(A)
    return dist.sum()

calculate_d_ech_dp_scan = equinox.filter_jit(equinox.filter_grad(calculate_ech_scan)) # sensitivity to permeability

size = 100
habitat_permability, activities, nb_active = create_landscape(size)

# forward pass
calculate_ech_scan(habitat_permability, activities, nb_active)
# 200 nodes requires 23.84GiB, which could just fit in the GPU

# backward pass requires same memory
sensitivity = calculate_d_ech_dp_scan(habitat_permability, activities, nb_active)
cbar = plt.imshow(sensitivity)
plt.colorbar(cbar)

A = grid.get_adjacency_matrix()
L = jaxscape.resistance_distance.graph_laplacian(A)
# V = pinv(L.todense(), hermitian=True)  # TODO: in the future, we want to only permit undirected graphs for resistance distance calculation, hence enforce `hermitian`
V = jnp.linalg.pinv(L)  # Moore-Penrose pseudoinverse of Laplacian

