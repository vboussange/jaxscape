# xarray-backed grids

JAXScape's graphs, distances and analyses take and return plain JAX arrays. They
do not accept or return labelled arrays from xarray,
[`xarray_jax`](https://github.com/google-deepmind/xarray_jax) or
[`coordax`](https://github.com/neuralgcm/coordax).

## Why this is out of scope

JAXScape is a minimal JAX library. `GridGraph`, the distances and the windowed
analyses are Equinox modules whose fields are plain arrays, and every
computation is jitted, vmapped and differentiated through them, including
custom gradient rules and host callbacks to sparse direct solvers. Labelled
arrays would add a dependency to that layer and would have to survive all of
these transformations, for coordinates that no computation reads.

A raster's coordinates and CRS matter only when reading inputs and writing
outputs, and user code handles that in a few lines with rioxarray, as the
examples do:

```python
import jax.numpy as jnp
import numpy as np
import rioxarray
from jaxscape import GridGraph

raster = rioxarray.open_rasterio("permeability.tif", masked=True).sel(band=1)
graph = GridGraph(grid=jnp.asarray(raster.fillna(0), dtype="float32"))

# ... compute a map `result` with the raster's shape ...

out = raster.copy(data=np.asarray(result))  # keeps the coordinates and the CRS
out.rio.to_raster("result.tif")
```

## Prior requests

- #3: "Add support for `xarray_jax`"
