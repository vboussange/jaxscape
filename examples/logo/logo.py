import jax
import jax.numpy as jnp
from jaxscape.rsp_distance import RSPDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.lcp_distance import LCPDistance
from jaxscape.gridgraph import GridGraph
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
import jax.random as jr

# loading jax array representing permeability
permeability = jnp.array(plt.imread("logo_raw.tiff")[20:-20,20:-20, 0])
permeability = (permeability - permeability.min()) / (permeability.max() - permeability.min())
threshold=0.00
key = jr.PRNGKey(0)
permeability = jnp.where(permeability > threshold, jr.uniform(key, permeability.shape, minval=0.9, maxval=1.1), permeability)
permeability = jnp.where(permeability <= threshold, 0.05, permeability)
activities = jnp.ones_like(permeability, dtype="bool")
cbar = plt.imshow(permeability, cmap="gray")
plt.colorbar(cbar)  
plt.axis("off")



distance=LCPDistance()
@eqx.filter_jit
def average_path_length(permeability, activities, nb_active, distance):
    grid = GridGraph(activities=activities, 
                     vertex_weights=permeability,
                     nb_active=nb_active)
    sources = jnp.linspace(0, nb_active, 50, dtype="int32")
    dist = distance(grid, sources)
    return dist.sum() / len(sources)

grad_connectivity = jax.grad(average_path_length)
nb_active = int(activities.sum())
average_path_length(permeability, activities, nb_active, distance)
sensitivities = grad_connectivity(permeability, activities, nb_active, distance)


fig = plt.figure(figsize=(10, 4))
cbar = plt.imshow(sensitivities,
                # jnp.exp(-sensitivities[1:,1:]), 
                  cmap="Spectral", 
                #   vmax=sensitivities.max() * 0.5,
                vmin=-3e2
                  )
plt.axis("off")
fig.tight_layout()
fig.savefig("logo.png", dpi=300)
