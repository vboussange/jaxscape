import rasterio
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import jax
import numpy as np
import time

from flax import nnx
import optax
from jax.nn import one_hot
import time

from jaxscape import LCPDistance, GridGraph
with rasterio.open("landcover.tif") as src:
  # We resample the raster to a smaller size to speed up the computation
  raster = src.read(1, masked=True, out_shape=(150, 150), resampling=rasterio.enums.Resampling.mode)  # Read the first band with masking
  lc_raster = jnp.array(raster.filled(0))   # Replace no data values with 0

category_color_dict = {
    11: "#476BA0",
    21: "#DDC9C9",
    22: "#D89382",
    23: "#ED0000",
    24: "#AA0000",
    31: "#b2b2b2",
    41: "#68AA63",
    42: "#1C6330",
    43: "#B5C98E",
    52: "#CCBA7C",
    71: "#E2E2C1",
    81: "#DBD83D",
    82: "#AA7028",
    90: "#BAD8EA",
    95: "#70A3BA"
}
category_labels = {
    11: "Water",
    21: "Developed, open space",
    22: "Developed, low intensity",
    23: "Developed, medium intensity",
    24: "Developed, high intensity",
    31: "Barren land",
    41: "Deciduous forest",
    42: "Evergreen forest",
    43: "Mixed forest",
    52: "Shrub/scrub",
    71: "Grassland/herbaceous",
    81: "Pasture/hay",
    82: "Cultivated crops",
    90: "Woody wetlands",
    95: "Emergent herbaceous wetlands"
}
cmap = plt.cm.colors.ListedColormap([col for cat, col in category_color_dict.items()])
norm = plt.cm.colors.BoundaryNorm(boundaries=[cat - 0.5 for cat in category_color_dict.keys()] + [list(category_color_dict.keys())[-1] + 0.5], ncolors=cmap.N)

# coordinates of the two populations
pop1 = (20, 10)
pop2 = (110, 110)
fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.imshow(lc_raster, cmap=cmap, norm=norm)
categories = list(category_color_dict.keys())
cbar = fig.colorbar(cax, 
          ticks=categories, 
          boundaries=[cat - 0.5 for cat in categories] + [categories[-1] + 0.5],
          shrink=0.7)
cbar.ax.set_yticklabels([category_labels[cat] for cat in categories])
ax.axis("off")

# Draw circles around cell (2, 2) and cell (99, 99)
circle1 = plt.Circle(pop1, 5, color='black', fill=False, linewidth=2)
circle2 = plt.Circle(pop2, 5, color='blue', fill=False, linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)

# Add text annotations
ax.text(pop1[0], pop1[1]+10, 'pop 1', color='black', fontsize=12, ha='center')
ax.text(pop2[0], pop2[1]+10, 'pop 2', color='blue', fontsize=12, ha='center')

plt.show()
fig.savefig("land_cover_raster.png", dpi=300, bbox_inches="tight", transparent=True)

# ground truth permeability
reclass_dict = {
    11: 2e-1,  # Water
    21: 2e-1,  # Developed, open space
    22: 3e-1,  # Developed, low intensity
    23: 1e-1,  # Developed, medium intensity (missing)
    24: 1e-1,  # Developed, high intensity (missing)
    31: 5e-1,  # Barren land
    41: 1.,  # Deciduous forest
    42: 1.,  # Evergreen forest
    43: 1.,  # Mixed forest
    52: 5e-1,  # Shrub/scrub
    71: 9e-1,  # Grassland/herbaceous
    81: 4e-1,  # Pasture/hay
    82: 4e-1,  # Cultivated crops
    90: 9e-1,  # Woody wetlands
    95: 6e-1  # Emergent herbaceous wetlands
}
permeability_raster = jnp.array(np.vectorize(reclass_dict.get)(lc_raster))
ref_grid = GridGraph(lc_raster)
source = ref_grid.coord_to_index(jnp.array([2]), jnp.array([2]))
target = ref_grid.coord_to_index(jnp.array([99]), jnp.array([99]))

distance = LCPDistance()
grid = GridGraph(permeability_raster)
start_time = time.time()
source_to_target_dist = distance(grid, source)[1,target]
print(f"Time taken for distance calculation: {time.time() - start_time:.2f} seconds")
print(f"Genetic distance between populations: {source_to_target_dist[0]:.2f}")

# fig, ax = plt.subplots()
# cbar = ax.imshow(true_dist_to_source, cmap="RdYlGn_r",  norm=plt.cm.colors.LogNorm())
# ax.axis("off")
# ax.set_title("Distance to source")
# fig.colorbar(cbar, ax=ax, shrink=0.5,)

category_to_index = {cat: i for i, cat in enumerate(reclass_dict.keys())}  # Map to indices

# Replace categories in lc_raster with indices
indexed_raster = jnp.array(np.vectorize(category_to_index.get)(lc_raster))


class Model(nnx.Module):
  def __init__(self, num_categories, rngs: nnx.Rngs):
    self.num_categories = num_categories
    self.linear = nnx.Linear(num_categories, 1, rngs=rngs)

  def __call__(self, x):
    x = self.linear(one_hot(x, num_classes=self.num_categories))
    x = jnp.clip(x, 1e-2, 1e0)
    return x
  
  
model = Model(len(category_to_index.keys()), rngs=nnx.Rngs(0)) 
optimizer = nnx.Optimizer(model, optax.adamw(5e-2)) 

x = indexed_raster.ravel()
@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer):
  def loss_fn(model):
    permeability = model(indexed_raster).ravel()
    permeability = ref_grid.node_values_to_array(permeability)
    grid = GridGraph(permeability)
    dist_to_node_hat = distance(grid, source).ravel()[target]
    return ((dist_to_node_hat - source_to_target_dist) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # in-place updates

  return loss


train_steps = 100
for step in range(train_steps):
  l = train_step(model, optimizer)
  if step % 10 == 0:
    print(f"Step {step}, loss: {l:.2f}")
  
# Plot final prediction of resistance against ground truth resistance
final_permeability = model(indexed_raster)[..., 1]
fig, ax = plt.subplots(1, 2, figsize=(8, 5))

ax[0].imshow(permeability_raster, cmap="RdYlGn")
ax[0].set_title("Ground truth permeability")
ax[0].axis("off")

cbar = ax[1].imshow(final_permeability, cmap="RdYlGn")
ax[1].set_title("Predicted permeability")
ax[1].axis("off")

fig.colorbar(cbar, ax=ax, shrink=0.5, location='right')
plt.show()
fig.savefig("inverse_problem.png", dpi=300, bbox_inches="tight", transparent=True)