# Distance Calculations

This guide demonstrates how to compute various distance metrics on grid graphs using JAXScape.

## Distance Metrics Overview

JAXScape implements several distance metrics commonly used in connectivity analysis:

| Metric | Description | Use Case |
|--------|-------------|----------|
| LCP Distance | Least-cost path distance | Finding optimal routes through landscapes |
| Resistance Distance | Effective electrical resistance | Modeling flow and connectivity with multiple paths |
| RSP Distance | Randomized shortest paths | Accounting for path uncertainty and behavioral variation |
| Euclidean Distance | Straight-line distance | Baseline comparisons |

## Basic Distance Computation

### Setting Up

```python
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxscape import GridGraph, LCPDistance, ResistanceDistance, RSPDistance

# Create a simple permeability landscape
permeability = jnp.ones((50, 50))
permeability = permeability.at[20:30, 20:30].set(0.1)  # Low permeability region

grid = GridGraph(grid=permeability)
```

### Computing Distances from a Single Source

```python
# Define source node (top-left corner)
source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))

# Compute distances
lcp_dist = LCPDistance()(grid, sources=source)
res_dist = ResistanceDistance()(grid, sources=source)
rsp_dist = RSPDistance(theta=0.01, cost=lambda x: 1/x)(grid, sources=source)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, dist, title in zip(axes, 
                           [lcp_dist, res_dist, rsp_dist],
                           ['LCP', 'Resistance', 'RSP']):
    im = ax.imshow(grid.node_values_to_array(dist.ravel()), cmap='viridis')
    ax.set_title(f'{title} Distance')
    plt.colorbar(im, ax=ax)
plt.tight_layout()
```

## All-Pairs Distance

Compute distances between all pairs of nodes:

```python
# This computes the full distance matrix
distance = LCPDistance()
dist_matrix = distance(grid)

print(f"Shape: {dist_matrix.shape}")  # (n_nodes, n_nodes)
print(f"Distance from node 0 to node 100: {dist_matrix[0, 100]}")
```

!!! warning "Memory Usage"
    All-pairs distance computation creates an `n × n` matrix. For large grids, consider computing distances for subsets of nodes.

## Source-Target Distances

Compute distances from specific sources to specific targets:

```python
# Define sources and targets
sources = jnp.array([0, 100, 200])
targets = jnp.array([1000, 1500, 2000])

# Compute distances
distance = LCPDistance()
dist_matrix = distance(grid, sources=sources, targets=targets)

print(f"Shape: {dist_matrix.shape}")  # (3, 3)
```

This is particularly efficient when you need distances from all nodes to a single target (e.g., distance to a habitat patch):

```python
# Distance from all nodes to a single target
target = jnp.array([1000])
distances_to_target = LCPDistance()(grid, targets=target)

# Visualize
plt.imshow(grid.node_values_to_array(distances_to_target.ravel()), cmap='magma')
plt.title('Distance to Target')
plt.colorbar()
```

## Pairwise Distances Among Nodes

Compute distances among a subset of nodes:

```python
# Select specific nodes of interest
nodes = jnp.array([0, 500, 1000, 1500, 2000])

# Compute pairwise distances
distance = ResistanceDistance()
dist_matrix = distance(grid, nodes=nodes)

print(f"Shape: {dist_matrix.shape}")  # (5, 5)
```

## Using Coordinates

You can also work with grid coordinates instead of node indices:

```python
# Define sources by (i, j) coordinates
source_coords = jnp.array([[0, 0], [10, 10], [20, 20]])

distance = LCPDistance()
distances = distance(grid, sources=source_coords)

# JAXScape automatically converts coordinates to node indices
```

## Advanced: Using Solvers for Resistance Distance

For large graphs, resistance distance computation can be accelerated using iterative solvers:

```python
from jaxscape.solvers import PyAMGSolver, CholmodSolver

# Using PyAMG (requires pyamg package)
try:
    solver = PyAMGSolver(rtol=1e-6, maxiter=100_000)
    distance = ResistanceDistance(solver=solver)
    dist = distance(grid)
except ImportError:
    print("PyAMG not available")

# Using Cholesky (requires cholespy package)
try:
    solver = CholmodSolver()
    distance = ResistanceDistance(solver=solver)
    dist = distance(grid)
except ImportError:
    print("Cholespy not available")
```

Solvers are particularly useful when:

- Computing distances on large graphs (> 10,000 nodes)
- Running many distance computations in a loop
- Memory is constrained

## Customizing Distance Behavior

### LCP Distance

LCP distance uses the Bellman-Ford algorithm with edge costs defined as the inverse of permeability:

```python
# Custom edge weight function
def custom_edge_weight(source_perm, target_perm):
    return 1 / jnp.sqrt(source_perm * target_perm)

grid = GridGraph(grid=permeability, fun=custom_edge_weight)
distance = LCPDistance()
dist = distance(grid)
```

### RSP Distance

RSP distance has a temperature parameter controlling path diversity:

```python
# Low temperature: similar to LCP (deterministic)
rsp_low = RSPDistance(theta=0.001, cost=lambda x: -jnp.log(x))

# High temperature: more path diversity
rsp_high = RSPDistance(theta=1.0, cost=lambda x: -jnp.log(x))

dist_low = rsp_low(grid, sources=source)
dist_high = rsp_high(grid, sources=source)

# Visualize difference
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(grid.node_values_to_array(dist_low.ravel()), cmap='viridis')
axes[0].set_title('Low Temperature (θ=0.001)')
axes[1].imshow(grid.node_values_to_array(dist_high.ravel()), cmap='viridis')
axes[1].set_title('High Temperature (θ=1.0)')
```

### Custom Cost Functions

You can define custom cost functions for RSP distance:

```python
# Exponential cost
rsp = RSPDistance(theta=0.1, cost=lambda x: jnp.exp(-x))

# Quadratic cost
rsp = RSPDistance(theta=0.1, cost=lambda x: (1 - x)**2)

# Or provide a pre-computed cost matrix
cost_matrix = compute_custom_costs(grid)
rsp = RSPDistance(theta=0.1, cost=cost_matrix)
```

## Performance Considerations

For optimal performance:

1. **JIT Compilation**: Use `@eqx.filter_jit` for repeated computations
2. **Batch Operations**: Process multiple distance queries together
3. **Solver Selection**: Use iterative solvers for large resistance distance problems
4. **Precision**: Consider using lower precision (`float32`) for memory-constrained scenarios

```python
import equinox as eqx

@eqx.filter_jit
def compute_distances(permeability):
    grid = GridGraph(grid=permeability)
    distance = LCPDistance()
    return distance(grid)

# First call compiles
dist1 = compute_distances(permeability)

# Subsequent calls are fast
dist2 = compute_distances(permeability * 1.5)
```
