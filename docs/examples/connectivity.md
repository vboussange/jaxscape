# Connectivity Analysis

Ecological connectivity analysis assesses how landscape structure facilitates or impedes movement and gene flow. JAXScape provides tools to quantify connectivity and identify critical areas for conservation.

## Connectivity Metrics

JAXScape implements connectivity analysis through the `ConnectivityAnalysis` class, which computes quality-weighted proximity indices similar to the [Harary index](https://mathworld.wolfram.com/HararyIndex.html) and [Equivalent Connected Area](https://www.sciencedirect.com/science/article/pii/S0006320721000604).

## Basic Workflow

### 1. Prepare Landscape Data

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
import rasterio

# Load habitat suitability raster
with rasterio.open("suitability.tif") as src:
    raster = src.read(1, masked=True)
    quality = jnp.array(raster.filled(0), dtype="float32")

# Visualize
plt.imshow(quality, cmap="Greens")
plt.colorbar(label="Habitat Quality")
plt.title("Landscape Quality")
plt.axis("off")
```

### 2. Define Species Parameters

```python
from jaxscape import ResistanceDistance
from jaxscape.solvers import CholmodSolver

# Dispersal range (maximum distance in pixels)
D = 20  # pixels

# Distance metric
distance = ResistanceDistance(solver=CholmodSolver())

# Proximity function: converts distance to proximity
def proximity(dist):
    """Negative exponential dispersal kernel."""
    return jnp.exp(-dist / D)
```

The proximity function defines how distance translates to ecological connectivity. Common forms:

- **Negative exponential**: `exp(-d/D)` - smooth decay
- **Threshold**: `(d < D).astype(float)` - binary connectivity
- **Power law**: `d**(-α)` - fat-tailed dispersal

### 3. Compute Connectivity

```python
from jaxscape import ConnectivityAnalysis

connectivity_analyzer = ConnectivityAnalysis(
    quality_raster=quality,
    permeability_raster=quality,  # Assuming quality = permeability
    distance=distance,
    proximity=proximity,
    coarsening_factor=0.0,
    dependency_range=D,
    batch_size=50
)

# Compute quality-weighted connectivity
connectivity = connectivity_analyzer.run(q_weighted=True)
print(f"Landscape connectivity index: {connectivity:.2f}")
```

### Parameters Explained

- `quality_raster`: Habitat quality or patch value (e.g., species density)
- `permeability_raster`: Movement cost surface (higher = easier movement)
- `distance`: Distance metric (LCP, Resistance, RSP)
- `proximity`: Function converting distance to connectivity
- `dependency_range`: Maximum relevant distance (optimization parameter)
- `batch_size`: Parallel processing batch size (tune for available memory)
- `coarsening_factor`: Experimental spatial aggregation (0.0 = no aggregation)

## Comparing Connectivity Scenarios

Compare connectivity under different landscape configurations:

```python
# Baseline landscape
connectivity_baseline = ConnectivityAnalysis(
    quality_raster=quality,
    permeability_raster=quality,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=50
).run(q_weighted=True)

# After habitat loss (50% quality reduction in a region)
quality_degraded = quality.at[20:40, 20:40].multiply(0.5)
connectivity_degraded = ConnectivityAnalysis(
    quality_raster=quality_degraded,
    permeability_raster=quality_degraded,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=50
).run(q_weighted=True)

# Impact assessment
impact = (connectivity_degraded - connectivity_baseline) / connectivity_baseline * 100
print(f"Connectivity change: {impact:.2f}%")
```

## Corridor Identification

Identify important corridors by computing connectivity contribution:

```python
from jaxscape import SensitivityAnalysis

# Compute sensitivity (connectivity contribution of each pixel)
sensitivity_analyzer = SensitivityAnalysis(
    quality_raster=quality,
    permeability_raster=quality,
    distance=distance,
    proximity=proximity,
    coarsening_factor=0.0,
    dependency_range=D,
    batch_size=20
)

# Sensitivity to permeability changes
corridor_importance = sensitivity_analyzer.run("permeability", q_weighted=True)

# Visualize corridors
import matplotlib.colors as colors

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.imshow(quality, cmap="Greens", alpha=0.7)
plt.title("Habitat Quality")
plt.axis("off")

plt.subplot(122)
plt.imshow(quality, cmap="Greens", alpha=0.3)
im = plt.imshow(corridor_importance, cmap="Reds", alpha=0.7,
                norm=colors.LogNorm(vmin=1e-2))
plt.title("Corridor Importance")
plt.colorbar(im, label="Connectivity Contribution")
plt.axis("off")

plt.tight_layout()
```

## Multi-Scale Analysis

Analyze connectivity at different dispersal scales:

```python
dispersal_ranges = [10, 20, 50, 100]  # pixels
connectivities = []

for D in dispersal_ranges:
    distance = ResistanceDistance(solver=CholmodSolver())
    proximity_func = lambda dist: jnp.exp(-dist / D)
    
    conn = ConnectivityAnalysis(
        quality_raster=quality,
        permeability_raster=quality,
        distance=distance,
        proximity=proximity_func,
        dependency_range=D,
        batch_size=50
    ).run(q_weighted=True)
    
    connectivities.append(conn)

# Plot scale-dependency
plt.figure()
plt.plot(dispersal_ranges, connectivities, 'o-', linewidth=2)
plt.xlabel("Dispersal Range (pixels)")
plt.ylabel("Connectivity Index")
plt.title("Scale-Dependent Connectivity")
plt.grid(True, alpha=0.3)
```

## Distance Metric Comparison

Compare how different distance metrics affect connectivity:

```python
from jaxscape import LCPDistance, RSPDistance

metrics = {
    "LCP": LCPDistance(),
    "Resistance": ResistanceDistance(solver=CholmodSolver()),
    "RSP (θ=0.01)": RSPDistance(theta=0.01, cost=lambda x: 1/x),
    "RSP (θ=0.1)": RSPDistance(theta=0.1, cost=lambda x: 1/x)
}

D = 20
results = {}

for name, distance in metrics.items():
    conn = ConnectivityAnalysis(
        quality_raster=quality,
        permeability_raster=quality,
        distance=distance,
        proximity=lambda dist: jnp.exp(-dist / D),
        dependency_range=D,
        batch_size=50
    ).run(q_weighted=True)
    results[name] = conn
    print(f"{name}: {conn:.2f}")

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), results.values())
plt.ylabel("Connectivity Index")
plt.title("Connectivity by Distance Metric")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
```

## Patch-Based Connectivity

Analyze connectivity between discrete habitat patches:

```python
from scipy import ndimage

# Identify patches (connected components)
patches = ndimage.label(quality > 0.5)[0]
n_patches = patches.max()

# Extract patch centers
patch_coords = []
for i in range(1, n_patches + 1):
    coords = jnp.argwhere(patches == i)
    center = coords.mean(axis=0).astype(int)
    patch_coords.append(center)

patch_coords = jnp.array(patch_coords)

# Compute inter-patch distances
from jaxscape import GridGraph

grid = GridGraph(grid=quality)
patch_indices = grid.coord_to_index(patch_coords[:, 0], patch_coords[:, 1])

distance_metric = ResistanceDistance()
patch_distances = distance_metric(grid, nodes=patch_indices)

# Compute patch connectivity matrix
patch_connectivity = proximity(patch_distances)

# Visualize patch network
plt.figure(figsize=(10, 8))
plt.imshow(patches, cmap="tab20", alpha=0.3)

# Draw connections
threshold = 0.1  # Only show strong connections
for i in range(n_patches):
    for j in range(i+1, n_patches):
        if patch_connectivity[i, j] > threshold:
            plt.plot([patch_coords[i, 1], patch_coords[j, 1]],
                    [patch_coords[i, 0], patch_coords[j, 0]],
                    'k-', alpha=patch_connectivity[i, j], linewidth=2)

plt.scatter(patch_coords[:, 1], patch_coords[:, 0], 
           c='red', s=100, zorder=5, edgecolors='black')
plt.title("Patch Connectivity Network")
plt.axis("off")
```

## Prioritization Analysis

Identify priority areas for conservation or restoration:

```python
# Compute current connectivity
baseline = ConnectivityAnalysis(
    quality_raster=quality,
    permeability_raster=quality,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=50
).run(q_weighted=True)

# Compute sensitivity
sensitivity = SensitivityAnalysis(
    quality_raster=quality,
    permeability_raster=quality,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=20
).run("quality", q_weighted=True)

# Restoration potential: sensitivity in degraded areas
degraded_mask = quality < 0.3
restoration_potential = sensitivity * degraded_mask

# Rank pixels by restoration value
flat_potential = restoration_potential.ravel()
top_pixels = jnp.argsort(flat_potential)[-100:]  # Top 100 pixels

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(131)
plt.imshow(quality, cmap="Greens")
plt.title("Current Quality")
plt.axis("off")

plt.subplot(132)
plt.imshow(restoration_potential, cmap="Reds")
plt.title("Restoration Potential")
plt.axis("off")

plt.subplot(133)
priority_map = jnp.zeros_like(quality.ravel())
priority_map = priority_map.at[top_pixels].set(1)
plt.imshow(priority_map.reshape(quality.shape), cmap="Reds")
plt.title("Top Priority Areas")
plt.axis("off")

plt.tight_layout()
```

## Best Practices

1. **Choose appropriate dispersal range**: Base `D` on species biology
2. **Validate distance metrics**: Compare with empirical movement data when available
3. **Scale analysis appropriately**: Use `dependency_range` to limit computation to relevant scales
4. **Consider multiple scenarios**: Test robustness to parameter uncertainty
5. **Interpret in context**: Combine quantitative metrics with ecological knowledge

## Performance Optimization

For large landscapes:

```python
# Use coarser resolution for initial analysis
quality_coarse = quality[::2, ::2]

# Use efficient solvers
from jaxscape.solvers import PyAMGSolver
distance = ResistanceDistance(solver=PyAMGSolver())

# Adjust batch size based on available memory
# Larger batch_size = faster but more memory
connectivity_analyzer = ConnectivityAnalysis(
    quality_raster=quality_coarse,
    permeability_raster=quality_coarse,
    distance=distance,
    proximity=proximity,
    dependency_range=D // 2,  # Scale with resolution
    batch_size=100  # Increase for more memory
)
```
