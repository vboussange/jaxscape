import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
from jaxscape.rsp_distance import RSPDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.lcp_distance import LCPDistance
from jaxscape.gridgraph import GridGraph
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx


# loading jax array representing permeability
permeability = jnp.array(np.loadtxt("permeability.csv", delimiter=","))

# we discard pixels with permeability equal to 0
activities = permeability > 0
plt.imshow(permeability, cmap="gray")
plt.axis("off")

grid = GridGraph(activities=activities, vertex_weights=permeability)

# Calculating distances of all pixels to top left pixel
source = grid.coord_to_active_vertex_index([0], [0])

distances = {
    "LCP distance": LCPDistance(),
    "Resistance distance": ResistanceDistance(),
    "RSP distance": RSPDistance(theta=0.01, cost=lambda x: 1 / x)
}

fig, axs = plt.subplots(1, 3, figsize=(10, 4))
for ax, (title, distance) in zip(axs, distances.items()):
    dist_to_node = distance(grid, source)
    cbar = ax.imshow(grid.node_values_to_array(dist_to_node.ravel()), cmap="magma")
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(cbar, ax=ax, shrink=0.2)

fig.suptitle("Distance to top left pixel")
plt.tight_layout()
plt.show()
fig.savefig("distances.png", dpi=300, transparent=True)

# gradient distance! let's calculate the gradient of average path length w.r.t pixel permeability
# we need to provide the number of active vertices, for jit compilation
@eqx.filter_jit
def average_path_length(permeability, activities, nb_active, distance):
    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability,
                     nb_active=nb_active)
    dist = distance(grid)
    return dist.sum() / nb_active**2

grad_connectivity = jax.grad(average_path_length)
nb_active = int(activities.sum())


distance = LCPDistance()
average_path_length(permeability, activities, nb_active, distance)


sensitivities = grad_connectivity(permeability, activities, nb_active, distance)
fig = plt.figure()
cbar = plt.imshow(sensitivities, cmap = "magma")
plt.title("Gradient of APL w.r.t pixel's permeability")
plt.colorbar(cbar)
plt.axis("off")
fig.savefig("sensitivities.png", dpi=300, transparent=True)
# For a more advanced example with windowed sensitivity analysis and dispatch on multipled GPUs
# see benchmark/moving_window_*.py