# Window Operations

JAXScape provides windowed operations for efficient processing of large rasters and connectivity analysis at scale.

## Overview

| Class | Purpose | Output | Use Case |
|-------|---------|--------|----------|
| **WindowOperation** | Generic windowed processing | Raster | Custom analyses on large landscapes |
| **ConnectivityAnalysis** | Landscape connectivity | Scalar | Quantify overall connectivity (Harary index) |
| **SensitivityAnalysis** | Connectivity gradients | Raster | Identify important pixels for connectivity |

## Connectivity Analysis

::: jaxscape.connectivity_analysis.ConnectivityAnalysis
    options:
      show_source: false
      heading_level: 3

Compute landscape connectivity as a quality-weighted sum of proximities:

```python
from jaxscape import ConnectivityAnalysis, ResistanceDistance

# Define parameters
D = 20  # Dispersal range
distance = ResistanceDistance()
proximity = lambda dist: jnp.exp(-dist / D)

# Analyze connectivity
conn = ConnectivityAnalysis(
    quality_raster=quality,
    permeability_raster=permeability,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=50
)

connectivity_index = conn.run(q_weighted=True)
```

**Key parameters:**
- `dependency_range`: Window size (should match species dispersal range)
- `batch_size`: Parallel processing batch (tune for memory)
- `q_weighted`: Weight by quality values (default: True)

## Sensitivity Analysis

::: jaxscape.sensitivity_analysis.SensitivityAnalysis
    options:
      show_source: false
      heading_level: 3

Compute gradients of connectivity with respect to landscape parameters:

```python
from jaxscape import SensitivityAnalysis

sens = SensitivityAnalysis(
    quality_raster=quality,
    permeability_raster=permeability,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=20
)

# Sensitivity to permeability changes
sensitivity_perm = sens.run("permeability", q_weighted=True)

# Sensitivity to quality changes  
sensitivity_qual = sens.run("quality", q_weighted=True)
```

Results show the impact of changing each pixel's value on overall connectivity.

## Custom Window Operations

::: jaxscape.window_operation.WindowOperation
    options:
      show_source: false
      heading_level: 3

Build custom analyses with `WindowOperation`:

```python
from jaxscape import WindowOperation

def custom_function(window_data):
    """Your custom analysis per window."""
    # Process local window
    return result

window_op = WindowOperation(
    raster=data,
    dependency_range=20,
    fun=custom_function,
    batch_size=30,
    overlap_mode='mean'  # How to combine overlapping results
)

result = window_op.run()
```

**Overlap modes:** `'mean'`, `'max'`, `'min'`, `'sum'`

## Performance Tuning

### Batch Size

Trade-off between speed and memory:

```python
# Limited memory
ConnectivityAnalysis(..., batch_size=10)

# Balanced
ConnectivityAnalysis(..., batch_size=50)

# Maximum speed (if memory allows)
ConnectivityAnalysis(..., batch_size=200)
```

### Coarsening (Experimental)

Speed up computation by spatial aggregation:

```python
# Full resolution (default)
ConnectivityAnalysis(..., coarsening_factor=0.0)

# 2x faster, approximate
ConnectivityAnalysis(..., coarsening_factor=0.3)

# Much faster, coarser approximation
ConnectivityAnalysis(..., coarsening_factor=0.5)
```

### Dependency Range

Limit computation to ecologically relevant distances:

```python
# Based on species dispersal
ConnectivityAnalysis(..., dependency_range=max_dispersal_distance)

# Smaller range = faster
ConnectivityAnalysis(..., dependency_range=10)

# Larger range = more context but slower
ConnectivityAnalysis(..., dependency_range=50)
```

## Practical Example

Complete workflow for landscape analysis:

```python
from jaxscape import ConnectivityAnalysis, SensitivityAnalysis
from jaxscape import ResistanceDistance
from jaxscape.solvers import PyAMGSolver

# Setup
D = 20
distance = ResistanceDistance(solver=PyAMGSolver())
proximity = lambda d: jnp.exp(-d / D)

# 1. Baseline connectivity
conn = ConnectivityAnalysis(
    quality_raster=quality,
    permeability_raster=permeability,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=50
).run(q_weighted=True)

print(f"Landscape connectivity: {conn:.2f}")

# 2. Identify important areas
sens = SensitivityAnalysis(
    quality_raster=quality,
    permeability_raster=permeability,
    distance=distance,
    proximity=proximity,
    dependency_range=D,
    batch_size=20
)

importance = sens.run("permeability", q_weighted=True)

# 3. Visualize
import matplotlib.pyplot as plt
plt.imshow(importance, cmap='Reds')
plt.title('Pixel Importance for Connectivity')
plt.colorbar(label='Sensitivity')
```