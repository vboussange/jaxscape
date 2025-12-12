# Sensitivity Analysis

Sensitivity analysis reveals how changes in landscape parameters affect connectivity metrics. JAXScape leverages JAX's automatic differentiation to compute gradients efficiently.

## Basic Sensitivity Analysis

### Computing Gradients

The simplest form of sensitivity analysis computes the gradient of a connectivity metric with respect to landscape permeability:

```python
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxscape import GridGraph, LCPDistance
import matplotlib.pyplot as plt

# Define landscape
permeability = jnp.array(np.loadtxt("permeability.csv", delimiter=","))

# Define objective function
@eqx.filter_jit
def average_path_length(permeability, distance):
    grid = GridGraph(grid=permeability)
    dist = distance(grid)
    return dist.sum() / (grid.nv ** 2)

# Compute gradient
grad_connectivity = jax.grad(average_path_length)
distance = LCPDistance()

sensitivities = grad_connectivity(permeability, distance)

# Visualize
plt.figure(figsize=(10, 8))
plt.imshow(sensitivities, cmap="RdBu_r", vmin=-1, vmax=1)
plt.title("Sensitivity of Average Path Length to Permeability")
plt.colorbar(label="∂APL/∂permeability")
plt.axis("off")
plt.show()
```

### Understanding Sensitivities

- **Positive values**: Increasing permeability at this location increases the objective (longer paths)
- **Negative values**: Increasing permeability at this location decreases the objective (shorter paths)
- **Large magnitude**: The location has high influence on connectivity

## Elasticity Analysis

Elasticity measures the proportional change in output for a proportional change in input, providing a scale-free interpretation:

```python
def compute_elasticity(sensitivities, permeability, objective_value):
    """Convert sensitivities to elasticities."""
    elasticity = (sensitivities * permeability) / objective_value
    return jnp.nan_to_num(elasticity, nan=0.0)

# Compute objective and sensitivity
objective = average_path_length(permeability, distance)
sensitivity = grad_connectivity(permeability, distance)

# Compute elasticity
elasticity = compute_elasticity(sensitivity, permeability, objective)

# Visualize
plt.figure(figsize=(10, 8))
im = plt.imshow(elasticity, cmap="plasma", 
                norm=matplotlib.colors.LogNorm(vmin=1e-3, vmax=1))
plt.title("Elasticity of Connectivity")
plt.colorbar(im, label="Elasticity")
plt.axis("off")
```

## The `SensitivityAnalysis` Class

For large landscapes, JAXScape provides a specialized class that efficiently computes sensitivities using windowed operations:

```python
from jaxscape import SensitivityAnalysis, ResistanceDistance
from jaxscape.solvers import CholmodSolver

# Load raster data
quality_raster = jnp.array(...)  # Habitat quality
permeability_raster = jnp.array(...)  # Movement permeability

# Define distance and proximity
D = 20  # Dispersal range in pixels
distance = ResistanceDistance(solver=CholmodSolver())

def proximity(dist):
    return jnp.exp(-dist / D)

# Create sensitivity analyzer
sensitivity_analyzer = SensitivityAnalysis(
    quality_raster=quality_raster,
    permeability_raster=permeability_raster,
    distance=distance,
    proximity=proximity,
    coarsening_factor=0.0,
    dependency_range=D,
    batch_size=20
)

# Compute sensitivity with respect to permeability
sensitivity_perm = sensitivity_analyzer.run("permeability", q_weighted=True)

# Compute sensitivity with respect to quality
sensitivity_qual = sensitivity_analyzer.run("quality", q_weighted=True)
```

### Parameters

- `quality_raster`: Habitat quality values (e.g., species density, patch quality)
- `permeability_raster`: Movement permeability (affects distance calculations)
- `distance`: Distance metric to use
- `proximity`: Function converting distance to proximity
- `dependency_range`: Maximum distance for pixel interactions
- `batch_size`: Number of pixels to process in parallel (tune for memory)
- `coarsening_factor`: Experimental feature for acceleration (0.0 = no coarsening)

## Computing Elasticities

```python
# Compute elasticity from sensitivities
elasticity_perm = sensitivity_perm * permeability_raster
elasticity_perm = jnp.nan_to_num(elasticity_perm, nan=0.0)

# Visualize on log scale
import matplotlib.colors as colors

plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.imshow(sensitivity_perm, cmap="RdBu_r")
plt.title("Sensitivity to Permeability")
plt.colorbar()
plt.axis("off")

plt.subplot(122)
plt.imshow(elasticity_perm + 1e-2, 
           cmap="plasma",
           norm=colors.LogNorm(vmin=1e0, vmax=elasticity_perm.max()))
plt.title("Elasticity to Permeability (log scale)")
plt.colorbar()
plt.axis("off")

plt.tight_layout()
```

## Connectivity Analysis

Before computing sensitivities, you might want to compute the baseline connectivity:

```python
from jaxscape import ConnectivityAnalysis

connectivity_analyzer = ConnectivityAnalysis(
    quality_raster=quality_raster,
    permeability_raster=permeability_raster,
    distance=distance,
    proximity=proximity,
    coarsening_factor=0.0,
    dependency_range=D,
    batch_size=50
)

# Compute quality-weighted connectivity
connectivity = connectivity_analyzer.run(q_weighted=True)
print(f"Landscape connectivity: {connectivity}")
```

This computes a scalar connectivity index similar to the [Harary index](https://mathworld.wolfram.com/HararyIndex.html) or [Equivalent Connected Area](https://www.sciencedirect.com/science/article/pii/S0006320721000604).

## Advanced: Custom Objective Functions

You can define custom objective functions and compute their gradients:

```python
@eqx.filter_jit
def connectivity_to_target(permeability, target_coords):
    """Compute average distance to a target location."""
    grid = GridGraph(grid=permeability)
    target_idx = grid.coord_to_index(target_coords[0], target_coords[1])
    
    distance = ResistanceDistance()
    distances = distance(grid, targets=jnp.array([target_idx]))
    
    return jnp.mean(distances)

# Compute sensitivity to target connectivity
target = jnp.array([25, 25])
grad_fn = jax.grad(connectivity_to_target)
sensitivity = grad_fn(permeability, target)

# Visualize: where to improve habitat to increase connectivity to target
plt.imshow(sensitivity, cmap="RdBu_r")
plt.plot(target[1], target[0], 'k*', markersize=20, label="Target")
plt.legend()
plt.title("Sensitivity: Improving Connectivity to Target")
plt.colorbar()
```

## Vectorized Multiple Scenarios

Compute sensitivities for multiple scenarios efficiently:

```python
import jax

def compute_sensitivity_scenario(permeability, distance_metric):
    """Compute sensitivity for one scenario."""
    grid = GridGraph(grid=permeability)
    dist = distance_metric(grid)
    return dist.sum()

grad_fn = jax.grad(compute_sensitivity_scenario)

# Multiple distance metrics
metrics = [
    LCPDistance(),
    ResistanceDistance(),
    RSPDistance(theta=0.01, cost=lambda x: 1/x)
]

# Compute all sensitivities
sensitivities = [grad_fn(permeability, metric) for metric in metrics]

# Visualize comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
titles = ["LCP", "Resistance", "RSP"]
for ax, sens, title in zip(axes, sensitivities, titles):
    im = ax.imshow(sens, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title(f"{title} Sensitivity")
    ax.axis("off")
    plt.colorbar(im, ax=ax)
plt.tight_layout()
```

## Performance Tips

1. **JIT Compilation**: Always use `@eqx.filter_jit` for repeated calculations
2. **Batch Processing**: Use the `batch_size` parameter to balance speed and memory
3. **Dependency Range**: Smaller ranges = faster computation (only analyze relevant pixels)
4. **Solver Selection**: For resistance distance, use iterative solvers on large grids
5. **Precision**: Consider `float32` if memory is constrained

```python
# Example of optimized workflow
@eqx.filter_jit
def optimized_sensitivity(permeability):
    grid = GridGraph(grid=permeability)
    distance = ResistanceDistance(solver=CholmodSolver())
    dist = distance(grid)
    return dist.sum() / (grid.nv ** 2)

# First call: compilation
sensitivity = jax.grad(optimized_sensitivity)(permeability)

# Subsequent calls: fast execution
for modified_perm in permeability_variants:
    sens = jax.grad(optimized_sensitivity)(modified_perm)
```

## Interpreting Results

When interpreting sensitivity and elasticity results:

- **Bottlenecks**: High sensitivity areas are bottlenecks where connectivity is most sensitive to change
- **Restoration priorities**: Areas with high elasticity are efficient restoration targets
- **Robust regions**: Low sensitivity indicates regions where connectivity is robust to disturbance
- **Corridors**: Linear patterns of high sensitivity often indicate important corridors

Use these insights to:

- Prioritize conservation efforts
- Design habitat restoration projects
- Assess vulnerability to land-use change
- Optimize corridor placement
