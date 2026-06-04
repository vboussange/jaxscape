import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import CholmodSolver
from matplotlib.colors import LinearSegmentedColormap


COLORS_BR = [
    "#2d00f7",
    "#6a00f4",
    "#8900f2",
    "#a100f2",
    "#b100e8",
    "#bc00dd",
    "#d100d1",
    "#db00b6",
    "#e500a4",
    "#f20089",
]
# check https://coolors.co/palettes/popular/gradient
CMAP_BR = LinearSegmentedColormap.from_list("species_richness", COLORS_BR, N=256)

# loading jax array representing permeability
mask = jnp.array(plt.imread("mask.png"))
mask = jnp.pad(mask, ((10, 10), (10, 10), (0, 0)), mode="constant", constant_values=0)
permeability = 1 - (mask[:, :, 3] > 0)
permeability += 1e-2  # avoid zero permeability

fig, ax = plt.subplots()
ax.imshow(permeability, cmap="gray")
# plt.axis("off")
node_coords = jnp.array(
    [
        [10, permeability.shape[0] - 100],
        [130, permeability.shape[0] - 90],
        [200, permeability.shape[0] - 100],
        [250, permeability.shape[0] - 150],
        [250, permeability.shape[0] - 30],
        [320, permeability.shape[0] - 70],
        [430, permeability.shape[0] - 80],
        # [490, permeability.shape[0]-65],
        [570, permeability.shape[0] - 110],
        [permeability.shape[1] - 50, permeability.shape[0] - 65],
    ]
)
ax.scatter(x=node_coords[:, 0], y=node_coords[:, 1], color="red", s=10)
plt.colorbar(ax.imshow(permeability, cmap="gray"), ax=ax)

grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
nodes = grid.coord_to_index(node_coords[:, 1], node_coords[:, 0])

distance = ResistanceDistance(solver=CholmodSolver())


# distance=LCPDistance()
@eqx.filter_jit
def calculate_dist(permeability):
    grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
    dist = distance(grid, nodes=nodes)
    return dist.sum() ** 2


grad_dist = eqx.filter_grad(calculate_dist)
calculate_dist(permeability)
sensitivities = grad_dist(permeability)

fig, ax = plt.subplots(figsize=(10, 4))
toprint = jnp.log(-sensitivities * (jnp.where(mask[:, :, 3] == 0, jnp.nan, 1)))
cbar = ax.imshow(
    toprint,
    cmap=CMAP_BR,
    vmax=jnp.nanquantile(toprint, 0.9),
    vmin=jnp.nanquantile(toprint, 0.1),
)
# plt.colorbar(cbar)
for coord in node_coords:
    circle = plt.Circle(
        (coord[0], coord[1]), radius=6, fill=False, edgecolor="#f72585", linewidth=2
    )
    ax.add_patch(circle)
plt.axis("off")
fig.tight_layout()
fig.savefig("logo.png", dpi=300, transparent=True)
