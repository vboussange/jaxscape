import rasterio
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxscape import WindowOperation, padding

with rasterio.open("../suitability.tif") as src:
    raster = src.read(1, masked=True)  # Read the first band with masking
    quality = jnp.array(raster.filled(0), dtype="float32")  # Replace no data values with 0

plt.imshow(quality)
plt.axis("off")
plt.savefig("quality_raster.png", dpi=300, bbox_inches="tight")

buffer_size=10
window_size=50
quality_padded = padding(quality, buffer_size, window_size)

window_op = WindowOperation(shape=quality_padded.shape, 
                            window_size=window_size, 
                            buffer_size=buffer_size)

window_op.nb_steps # 8

# lazy iterator
fig, axs = plt.subplots(2, 4, figsize=(10, 5))
for i, (xy, w) in enumerate(window_op.lazy_iterator(quality_padded)):
    ax = axs.flatten()[i]
    ax.set_title(f"xy = {xy}")
    ax.imshow(w)
    ax.axis("off")
plt.savefig("windows.png", dpi=300, bbox_inches="tight")

# eager iterator
xy, windows = window_op.eager_iterator(quality)
xy.shape # (8, 2)
windows.shape # (8, 10+50+10, 10+50+10)

plt.figure()
new_window = jnp.ones(windows[2, ...].shape, dtype="float32")
new_raster = window_op.update_raster_with_focal_window(xy[2], quality_padded, new_window)
plt.imshow(new_raster)
plt.axis("off")
plt.savefig("new_raster.png", dpi=300, bbox_inches="tight")