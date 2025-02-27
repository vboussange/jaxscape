import rasterio
import jax.numpy as jnp 
import matplotlib.pyplot as plt
import jax
import numpy as np
import time

from jaxscape import LCPDistance, GridGraph

with rasterio.open("landcover.tif") as src:
  # We resample the raster to a smaller size to speed up the computation
  raster = src.read(1, masked=True, out_shape=(100, 100), resampling=rasterio.enums.Resampling.mode)  # Read the first band with masking
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
# Create a color map based on the category_color_dict
cmap = plt.cm.colors.ListedColormap([col for cat, col in category_color_dict.items()])

# Normalize the raster values to the range of the colormap
norm = plt.cm.colors.BoundaryNorm(boundaries=[cat - 0.5 for cat in category_color_dict.keys()] + [list(category_color_dict.keys())[-1] + 0.5], ncolors=cmap.N)

# Plot the raster
# plt.figure()
# plt.imshow(lc_raster, cmap=cmap, norm=norm)
# categories = list(category_color_dict.keys())
# plt.colorbar(ticks=categories, boundaries=[cat - 0.5 for cat in categories] + [categories[-1] + 0.5])
# plt.title("Land Cover Raster")
# plt.show()


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
    52: 8e-1,  # Shrub/scrub
    71: 9e-1,  # Grassland/herbaceous
    81: 4e-1,  # Pasture/hay
    82: 4e-1,  # Cultivated crops
    90: 9e-1,  # Woody wetlands
    95: 8e-1  # Emergent herbaceous wetlands
}
# Transform lc_raster into a resistance raster with reclass_dict
permeability_raster = jnp.array(np.vectorize(reclass_dict.get)(lc_raster))
plt.figure()
plt.imshow(permeability_raster, cmap="RdYlGn", norm=plt.cm.colors.LogNorm())
cbar = plt.colorbar()
cbar.set_label('Permeability')

distance = LCPDistance()
def calculate_dist_to_source(permeability_raster):
  grid = GridGraph(permeability_raster)
  source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
  dist_to_node = distance(grid, source)
  dist_to_node = grid.node_values_to_array(dist_to_node.ravel())
  return dist_to_node
  
start_time = time.time()
true_dist_to_source = calculate_dist_to_source(permeability_raster)
print(f"Time taken for distance calculation: {time.time() - start_time} seconds")

fig, ax = plt.subplots()
cbar = ax.imshow(true_dist_to_source, cmap="RdYlGn_r",  norm=plt.cm.colors.LogNorm())
ax.axis("off")
ax.set_title("Distance to source")
fig.colorbar(cbar, ax=ax, shrink=0.5,)

category_to_index = {cat: i for i, cat in enumerate(reclass_dict.keys())}  # Map to indices

# Replace categories in lc_raster with indices
import numpy as np
indexed_raster = jnp.array(np.vectorize(category_to_index.get)(lc_raster))

# build flax model
from flax import nnx
import optax
from jax.nn import one_hot
import time

class Model(nnx.Module):
  def __init__(self, dmid, num_categories, rngs: nnx.Rngs):
    self.num_categories = num_categories
    self.linear = nnx.Linear(num_categories, dmid, rngs=rngs)

  def __call__(self, x):
    x = self.linear(one_hot(x, num_classes=self.num_categories))
    return nnx.relu(x) + 1e-5
  
  
model = Model(64, len(category_to_index.keys()), rngs=nnx.Rngs(0)) 
optimizer = nnx.Optimizer(model, optax.adamw(1e-2)) 

permeability = model(indexed_raster)[..., 1]
plt.figure()
plt.imshow(permeability, cmap="RdYlGn", norm=plt.cm.colors.LogNorm())
cbar = plt.colorbar()
cbar.set_label('Permeability, first guess')

dist_to_source_hat = calculate_dist_to_source(permeability)
fig, ax = plt.subplots()
cbar = ax.imshow(dist_to_source_hat, cmap="RdYlGn_r",  norm=plt.cm.colors.LogNorm())
ax.axis("off")
ax.set_title("Distance to source, first guess")
fig.colorbar(cbar, ax=ax, shrink=0.5,)

def calculate_dist_to_source(permeability_raster):
  grid = GridGraph(permeability_raster)
  source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
  dist_to_node = distance(grid, source)
  dist_to_node = grid.node_values_to_array(dist_to_node.ravel())
  return dist_to_node


true_dist_to_node = true_dist_to_source.ravel()[1:2]
@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer):
  def loss_fn(model):
    permeability = model(indexed_raster)[..., 1]
    grid = GridGraph(permeability)
    source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))
    dist_to_node_hat = distance(grid, source)[1,1:2]
    return ((jnp.log(dist_to_node_hat) - jnp.log(true_dist_to_node)) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # in-place updates

  return loss


train_step(model, optimizer)

train_steps = 2000
for step in range(train_steps):
  l = train_step(model, optimizer)
  print(f"Step {step}, loss: {l}")
  
# Plot final prediction of resistance against ground truth resistance
final_permeability = model(indexed_raster)[..., 1]
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(permeability_raster, cmap="viridis", norm=plt.cm.colors.LogNorm())
ax[0].set_title("Ground truth permeability")
ax[0].axis("off")

cbar = ax[1].imshow(final_permeability, cmap="viridis", norm=plt.cm.colors.LogNorm())
ax[1].set_title("Predicted permeability")
ax[1].axis("off")

fig.colorbar(cbar, ax=ax, shrink=0.5, location='right')
plt.show()