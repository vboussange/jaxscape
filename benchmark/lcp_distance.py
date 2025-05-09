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

path_results = Path("results/LCPDistance")
path_results.mkdir(parents=True, exist_ok=True)

# Define a habitat permeability raster
N = 20
corridor_length = 6
corridor_width = 4
habitat_permability = jnp.zeros((N, N), dtype="float32")
habitat_permability = habitat_permability.at[:, :int((N-corridor_length)/2)].set(1)
habitat_permability = habitat_permability.at[int((N-corridor_width)/2):int((N-corridor_width)/2) + corridor_width, int((N-corridor_length)/2):int((N-corridor_length)/2) + corridor_length].set(1)
habitat_permability = habitat_permability.at[:, int((N-corridor_length)/2) + corridor_length:].set(1)

fig, ax = plt.subplots()
ax.imshow(habitat_permability)
activities = habitat_permability > 0

# Distance calculation from focal pixel
focal_pixel_coord = (N-2, N-2)
ax.scatter([focal_pixel_coord[0]], [focal_pixel_coord[1]])

grid = GridGraph(activities=activities, vertex_weights=habitat_permability)
distance = LCPDistance()
mat = distance(grid)
vertex_index = grid.coord_to_active_vertex_index(*focal_pixel_coord)
distance_to_node = grid.node_values_to_array(mat[:, vertex_index])
fig, ax = plt.subplots()
ax.imshow(distance_to_node)
ax.scatter([focal_pixel_coord[0]], [focal_pixel_coord[1]], c="tab:red")
ax.set_axis_off()
fig.savefig(path_results / "distance_to_node.png", dpi=300)

habitat_quality = jnp.ones((N, N), dtype="float32") * 0.0
habitat_quality = habitat_quality.at[N-2, 1].set(1.)
habitat_quality = habitat_quality.at[N-2, N-2].set(1.)
fig, ax = plt.subplots()
ax.imshow(habitat_quality)
ax.axis('off')
fig.savefig(path_results / "habitat_suitability.png", dpi=300)

# `equivalent_connected_habitat` calculation
# We first need to calculate a distance, 
# that we transform into an ecological proximity
def calculate_ech(habitat_permability, habitat_quality, activities, D):
    grid = GridGraph(activities=activities, vertex_weights=habitat_permability)
    dist = distance(grid, landmarks = jnp.arange(grid.nb_active))

    proximity = jnp.exp(- dist / D)
    landscape = ExplicitGridGraph(activities=activities, 
                                  vertex_weights=habitat_quality, 
                                  adjacency_matrix=proximity)
    ech = landscape.equivalent_connected_habitat()
    return ech

calculate_d_ech_dp = jax.grad(calculate_ech) # sensitivity to permeability
calculate_d_ech_dq = jax.grad(calculate_ech, argnums=1) # sensitivity to quality


fig, axs = plt.subplots(2, 3)
for i, D in enumerate(jnp.array([5e-1, 5e0, 5e1])):
    # sensitivity to quality
    d_ech_dq = calculate_d_ech_dq(habitat_permability, habitat_quality, activities, D)
    axs[0,i].imshow(d_ech_dq)
    axs[0,i].text(0.5, -0.1, f"D={D:0.1e}", ha='center', va='center', transform=axs[0,i].transAxes)
    
    # sensitivity to permeability
    d_ech_dp = calculate_d_ech_dp(habitat_permability, habitat_quality, activities, D)
    cbar = axs[1,i].imshow(d_ech_dp)
    cbar = fig.colorbar(cbar, ax=axs[1,i], shrink=0.4)
    cbar.ax.tick_params(labelsize=8)
for ax in axs.flat:
    ax.axis('off')
axs[0,1].set_title("$\\frac{\\partial \\text{ECH}}{\\partial q}$")
axs[1,1].set_title("$\\frac{\\partial \\text{ECH}}{\\partial p}$")
fig.suptitle("LCPDistance")
fig.tight_layout()
fig.savefig(path_results / "sensitivities.png", dpi=300)

