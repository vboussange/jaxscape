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
  raster = src.read(1, masked=True, out_shape=(100, 100), resampling=rasterio.enums.Resampling.mode)  # Read the first band with masking
  lc_raster = jnp.array(raster.filled(0))   # Replace no data values with 0

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
# plt.figure()
# plt.imshow(permeability_raster, cmap="RdYlGn", norm=plt.cm.colors.LogNorm())
# cbar = plt.colorbar()
# cbar.set_label('Permeability')

category_to_index = {cat: i for i, cat in enumerate(reclass_dict.keys())}  # Map to indices

# Replace categories in lc_raster with indices
indexed_raster = jnp.array(np.vectorize(category_to_index.get)(lc_raster))

# flax model
class Model(nnx.Module):
  def __init__(self, num_categories, rngs: nnx.Rngs):
    self.num_categories = num_categories
    self.linear = nnx.Linear(num_categories, 1, rngs=rngs)

  def __call__(self, x):
    x = self.linear(one_hot(x, num_classes=self.num_categories))
    return x
  
  
model = Model(len(category_to_index.keys()), rngs=nnx.Rngs(0)) 
optimizer = nnx.Optimizer(model, optax.adam(1e-2)) 

x = jnp.array(list(category_to_index.values()))
y = jnp.array([reclass_dict[k] for k in category_to_index.keys()])
    
    
@nnx.jit  # automatic state management for JAX transforms
def train_step(model, optimizer):
  def loss_fn(model):
    permeability = model(x).ravel()
    permeability = jnp.clip(permeability, 1e-5, 1e0)
    return ((permeability - y) ** 2).mean()

  loss, grads = nnx.value_and_grad(loss_fn)(model)
  optimizer.update(grads)  # in-place updates

  return loss


train_step(model, optimizer)

train_steps = 2000
for step in range(train_steps):
  l = train_step(model, optimizer)
  print(f"Step {step}, loss: {l}")
  
model(x).ravel()
y