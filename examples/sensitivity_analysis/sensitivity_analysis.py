import jax.numpy as jnp
from jaxscape.lcp_distance import LCPDistance
import matplotlib.pyplot as plt
import jax.random as jr
import rasterio


with rasterio.open("../suitability.tif") as src:
    raster = src.read(1, masked=True)  # Read the first band with masking
    quality_raster = jnp.array(raster.filled(0), dtype="float32")  # Replace no data values with 0


D = 20

distance = LCPDistance()
proximity = lambda dist: jnp.exp(-dist) * (dist < D)

from jaxscape import ConnectivityAnalysis
prob = ConnectivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=quality_raster,
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=D,
                            batch_size=20)
connectivity = prob.run(q_weighted=False)

# Batch progress: 100%|██████████| 4/4 [00:02<00:00,  1.52it/s]
# Array(553.1566, dtype=float32)

from jaxscape import SensitivityAnalysis
# estimation via tiled connectivity analysis
prob = SensitivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=quality_raster,
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=D,
                            batch_size=20)

sensitivity_permeability = prob.run("permeability")

# Batch progress: 100%|██████████| 36/36 [00:01<00:00, 24.16it/s]

plt.imshow(sensitivity_permeability)
plt.savefig("sensitivity_permeability.png")