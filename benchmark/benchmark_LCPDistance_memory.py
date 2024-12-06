"""
Benchmarking memory usage of sensitivity calculation based on
different implementations of the ECH using LCP distance
"""
import jax
from jax import lax
import jax.numpy as jnp
from jaxscape.lcp_distance import bellman_ford
from jaxscape.gridgraph import GridGraph
import matplotlib.pyplot as plt
from pathlib import Path
import equinox


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
    
    ech = jnp.array(0, dtype=habitat_permability.dtype)
    A = grid.get_adjacency_matrix()
    
    @equinox.filter_checkpoint
    def body_fun(ech, source):
        dist = bellman_ford(A, source)
        return ech + dist.sum(), None

    ech, _ = lax.scan(body_fun, ech, jnp.arange(nb_active))
    return ech

calculate_d_ech_dp_scan = equinox.filter_jit(equinox.filter_grad(calculate_ech_scan)) # sensitivity to permeability

size = 50
habitat_permability, activities, nb_active = create_landscape(size)

# forward pass
calculate_ech_scan(habitat_permability, activities, nb_active)

# backward pass
sensitivity = calculate_d_ech_dp_scan(habitat_permability, activities, nb_active)
plt.imshow(sensitivity)


# # calculating equivalent connected habitat with inner vmap
# @equinox.filter_jit
# def calculate_ech_inner_vmap(habitat_permability, activities, nb_active):
#     grid = GridGraph(activities=activities, 
#                      vertex_weights=habitat_permability,
#                      nb_active=nb_active)
    
#     A = grid.get_adjacency_matrix()
    
#     sources = jnp.arange(nb_active)
    
#     @equinox.filter_checkpoint
#     def compute_dist(source):
#         dist = bellman_ford(A, source)
#         return dist.sum()

#     distances = jax.vmap(compute_dist)(sources)
#     return distances.sum()

# calculate_d_ech_dp_inner_vmap = equinox.filter_jit(equinox.filter_grad(calculate_ech_inner_vmap)) # sensitivity to permeability

# size = 60
# habitat_permability, activities, nb_active = create_landscape(size)

# # forward pass
# calculate_ech_inner_vmap(habitat_permability, activities, nb_active)

# # backward pass
# sensitivity = calculate_d_ech_dp_inner_vmap(habitat_permability, activities, nb_active)
# plt.imshow(sensitivity)



# # calculating equivalent connected habitat with outer vmap
# @equinox.filter_jit
# def calculate_ech_source(habitat_permability, activities, nb_active, source):
#     grid = GridGraph(activities=activities, 
#                      vertex_weights=habitat_permability,
#                      nb_active=nb_active)
    
#     A = grid.get_adjacency_matrix()
    
#     dist = bellman_ford(A, source)
#     return dist.sum()

# calculate_d_ech_dp_source = equinox.filter_jit(equinox.filter_grad(calculate_ech_source)) # sensitivity to permeability

# size = 60
# habitat_permability, activities, nb_active = create_landscape(size)

# # forward pass
# calculate_ech_source(habitat_permability, activities, nb_active, 1)

# # backward pass
# sensitivity = calculate_d_ech_dp_source(habitat_permability, activities, nb_active, 1)
# plt.imshow(sensitivity)


# calculating memory footprint of adjaceny matrix
# this is very fast with low footprint
@equinox.filter_jit
def calculate_sum_A(habitat_permability, activities, nb_active):
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=nb_active)
    
    A = grid.get_adjacency_matrix()
    
    return A.sum()

calculate_d_sum_A = equinox.filter_jit(equinox.filter_grad(calculate_sum_A)) # sensitivity to permeability

size = 1000
habitat_permability, activities, nb_active = create_landscape(size)

# forward pass
calculate_sum_A(habitat_permability, activities, nb_active)

# backward pass
sensitivity = calculate_d_sum_A(habitat_permability, activities, nb_active)
plt.imshow(sensitivity)