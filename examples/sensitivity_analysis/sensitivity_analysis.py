import jax.numpy as jnp
from jaxscape import LCPDistance
import matplotlib.pyplot as plt
import jax.random as jr
import rasterio
import jax
import matplotlib


with rasterio.open("../suitability.tif") as src:
    raster = src.read(1, masked=True)  # Read the first band with masking
    quality_raster = jnp.array(raster.filled(0), dtype="float32") / 100  # Replace no data values with 0

plt.imshow(quality_raster)
plt.axis("off")

D = 20

distance = LCPDistance()

def proximity(dist):
    # return jax.scipy.special.softmax(-dist)
    return jnp.exp(-dist/D)

from jaxscape import ConnectivityAnalysis
connectivity_prob = ConnectivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=quality_raster,
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=D,
                            batch_size=50)
connectivity = connectivity_prob.run(q_weighted=False) # scalar value

# Batch progress: 100%|██████████| 6/6 [00:06<00:00,  1.14s/it]
# Array(10223621., dtype=float32)

from jaxscape import SensitivityAnalysis
sensitivity_prob = SensitivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=quality_raster,
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=D,
                            batch_size=20)

sensitivity_permeability = sensitivity_prob.run("permeability", q_weighted=True) # raster

# Batch progress: 100%|██████████| 36/36 [00:01<00:00, 24.16it/s]

elasticity = sensitivity_permeability * quality_raster
elasticity = jnp.nan_to_num(elasticity, nan=0.0)
plt.imshow(elasticity + 1e-2, 
           cmap="plasma", 
           norm=matplotlib.colors.LogNorm(vmin=1e0)
           )
plt.axis("off")
cbar = plt.colorbar(shrink=0.5)
cbar.set_label('Elasticity w.r.t permeability')
plt.savefig("elasticity_permeability.png")

# To spot bottleneck, a good idea is to compare the elasticity to a perfect landscape
# with no resistance to movement
sensitivity_prob = SensitivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=jnp.ones_like(quality_raster),
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=D,
                            batch_size=20)

sensitivity_permeability_ideal = sensitivity_prob.run("permeability", q_weighted=True) # raster
elasticity_ideal = sensitivity_permeability_ideal * quality_raster
elasticity_ideal = jnp.nan_to_num(elasticity_ideal, nan=0.0)

plt.imshow(elasticity_ideal - elasticity + 1e-2, 
           cmap="plasma", 
        #    vmax=1e2,
           norm=matplotlib.colors.LogNorm(vmin=1e0)
           )
plt.axis("off")
cbar = plt.colorbar(shrink=0.5)
cbar.set_label('Bottlenecks')
plt.savefig("bottlenecks.png")

# want to prioritize the landscape?
improved_permeability = 0.4
# Add 0.1 to coordinates in quality_raster which have highest values in sensitivity_permeability
threshold = jnp.percentile(elasticity, 95)  # Get the 99th percentile value excluding NaNs
high_sensitivity_coords = jnp.where(elasticity >= threshold)  # Get coordinates with high sensitivity
improved_quality_raster = quality_raster.at[high_sensitivity_coords].add(improved_permeability)

# Add 0.1 to 100 random cells of quality_raster
key = jr.PRNGKey(0)
random_indices = jr.choice(key, jnp.arange(elasticity.size), shape=(high_sensitivity_coords[0].size,), replace=False)
random_coords = jnp.unravel_index(random_indices, quality_raster.shape)
modified_quality_raster = quality_raster.at[random_coords].add(improved_permeability)

def run_connectivity_analysis(raster):
    connectivity_prob = ConnectivityAnalysis(quality_raster=quality_raster,
                            permeability_raster=raster,
                            distance=distance,
                            proximity=proximity,
                            coarsening_factor=0.,
                            dependency_range=D,
                            batch_size=50)
    return connectivity_prob.run(q_weighted=True)

base_connectivity = run_connectivity_analysis(quality_raster)
connectivity_improved = run_connectivity_analysis(improved_quality_raster)
connectivity_improved_randomly = run_connectivity_analysis(modified_quality_raster)

print("Landscape connectivity gain")
print(f"- based on priorization with elasticity: {(connectivity_improved - base_connectivity) / base_connectivity * 100:.2f}%")
print(f"- based on random priorization: {((connectivity_improved_randomly - base_connectivity) / base_connectivity * 100):.2f}%")
