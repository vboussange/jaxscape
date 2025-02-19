import jax.numpy as jnp
from jaxscape.sensitivity_analysis import SensitivityAnalysis, d_quality_vmap, d_permeability_vmap
from jaxscape.gridgraph import GridGraph
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.lcp_distance import LCPDistance
import matplotlib.pyplot as plt
import jax.random as jr
import equinox as eqx

D = 11
quality_raster = jr.uniform(jr.PRNGKey(0), (51, 51))
plt.imshow(quality_raster)
plt.axis("off")
plt.savefig("quality_raster_sensitivity.png")

distance = LCPDistance()
proximity = lambda dist: jnp.exp(-dist) * (dist < D)
dependency_range=D

# estimation via tiled connectivity analysis
prob = SensitivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=quality_raster,
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=dependency_range,
                            batch_size=20)

sensitivity_permeability = prob.run("permeability")

# Batch progress: 100%|██████████| 36/36 [00:01<00:00, 24.16it/s]

plt.imshow(sensitivity_permeability)
plt.savefig("sensitivity_permeability.png")