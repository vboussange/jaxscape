# Distance Metrics

JAXScape implements differentiable distance metrics for connectivity analysis on grid graphs.

## Overview

| Metric | Algorithm | Best For | Complexity | Notes |
|--------|-----------|----------|------------|-------|
| **Euclidean** | Direct | Baselines | O(N) | Ignores landscape structure |
| **LCP** | Bellman-Ford | Optimal paths | O(V×E×S) | Single best route |
| **Resistance** | Laplacian solver | Multi-path flow | O(N³) or O(N log N) | All paths weighted |
| **RSP** | Matrix exponential | Stochastic movement | O(N³) | Temperature-controlled |

**S** = number of sources, **V** = vertices, **E** = edges, **N** = total nodes

## AbstractDistance

::: jaxscape.distance.AbstractDistance
    options:
      show_source: false
      heading_level: 3

All distance classes share a unified interface with three usage patterns:

```python
# All-pairs distance
dist_matrix = distance(grid)  # Shape: (n_nodes, n_nodes)

# Source-target distance
dist_matrix = distance(grid, sources=[0, 1], targets=[10, 20])  # Shape: (2, 2)

# Pairwise among subset
dist_matrix = distance(grid, nodes=[0, 5, 10])  # Shape: (3, 3)
```

## Euclidean Distance

::: jaxscape.euclidean_distance.EuclideanDistance
    options:
      show_source: false
      heading_level: 3

Straight-line distance in grid coordinates:

```python
from jaxscape import EuclideanDistance

distance = EuclideanDistance()
dist = distance(grid, sources=source_coords, targets=target_coords)
```

**Use for:** Null models, baseline comparisons

## LCP Distance

::: jaxscape.lcp_distance.LCPDistance
    options:
      show_source: false
      heading_level: 3

Least-cost path using Bellman-Ford algorithm:

```python
from jaxscape import LCPDistance

distance = LCPDistance()
dist = distance(grid, sources=sources, targets=targets)
```

**Key features:**
- Edge cost = 1 / edge_weight
- Finds single optimal path
- Efficient for sparse sources

**Use for:** Optimal routing, corridor identification

## Resistance Distance

::: jaxscape.resistance_distance.ResistanceDistance
    options:
      show_source: false
      heading_level: 3

Effective electrical resistance (accounts for all paths):

```python
from jaxscape import ResistanceDistance
from jaxscape.solvers import PyAMGSolver

# Default: pseudo-inverse (small graphs)
distance = ResistanceDistance()

# With solver (large graphs)
distance = ResistanceDistance(solver=PyAMGSolver())

dist = distance(grid)
```

**Key features:**
- Symmetric metric
- Multi-path connectivity
- Requires solver for large graphs (>5k nodes)

**Use for:** Gene flow, ecological connectivity, current flow

## RSP Distance

::: jaxscape.rsp_distance.RSPDistance
    options:
      show_source: false
      heading_level: 3

Randomized shortest path with temperature parameter:

```python
from jaxscape import RSPDistance

# Low temperature (deterministic, like LCP)
distance = RSPDistance(theta=0.001, cost=lambda x: -jnp.log(x))

# High temperature (more path diversity)
distance = RSPDistance(theta=1.0, cost=lambda x: -jnp.log(x))

dist = distance(grid)
```

**Key features:**
- `theta`: Temperature parameter (lower = more deterministic)
- Custom cost functions supported
- Accounts for behavioral uncertainty

**Use for:** Stochastic movement, uncertain dispersal

## Usage Patterns

### Computing Specific Distances

```python
# All nodes to single target (efficient)
target = jnp.array([100])
distances = distance(grid, targets=target)

# Subset to subset
sources = jnp.array([0, 10, 20])
targets = jnp.array([100, 200, 300])
dist_matrix = distance(grid, sources=sources, targets=targets)

# Using coordinates instead of indices
source_coords = jnp.array([[0, 0], [10, 10]])
dist = distance(grid, sources=source_coords)
```

### Comparing Metrics

```python
metrics = {
    "Euclidean": EuclideanDistance(),
    "LCP": LCPDistance(),
    "Resistance": ResistanceDistance(),
    "RSP": RSPDistance(theta=0.01, cost=lambda x: 1/x)
}

results = {name: metric(grid, sources=source) 
           for name, metric in metrics.items()}
```

## Choosing a Distance Metric

**Ecological applications:**
- **Gene flow / dispersal**: Resistance or RSP (multi-path)
- **Corridor planning**: LCP or Resistance
- **Individual movement**: LCP (deterministic) or RSP (stochastic)

**Graph properties:**
- **Small graphs (<5k nodes)**: Any metric works
- **Large graphs (>10k nodes)**: Use LCP or Resistance with solver
- **Need symmetry**: Euclidean or Resistance

**Computational resources:**
- **Limited memory**: LCP or Euclidean
- **GPU available**: All metrics benefit
- **Many distance queries**: Use solvers for Resistance
