# Solvers

JAXScape wraps custom solvers into Lineax solvers for efficient resistance distance computations on large graphs.

## Solver Overview

| Solver | Type | Best For | Speed | Memory | Notes |
|--------|------|----------|-------|--------|-------|
| **None** (default) | Pseudo-inverse | Small graphs (<5k nodes) | Slow | High | Always available, maximum accuracy |
| **PyAMGSolver** | Iterative AMG | Large sparse graphs | Fast | Low | Requires `pyamg` package |
| **CholmodSolver** | Direct Cholesky | Medium graphs | Very Fast | Medium | Requires `cholespy` package |
| **Any Lineax solver** | Various | Custom needs | Varies | Varies | Use any `lineax.AbstractLinearSolver` |

## Using Solvers

### Default (Pseudo-Inverse)

The default method uses pseudo-inverse and works out of the box:

```python
from jaxscape import ResistanceDistance

# No solver specified - uses pseudo-inverse
distance = ResistanceDistance()
dist = distance(grid)
```

**Pros:** Always available, no dependencies, exact solution  
**Cons:** Memory-intensive, slow for large graphs (>5,000 nodes)

### Custom JAXScape Solvers

#### PyAMG Solver

::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      show_source: false
      heading_level: 3



```python
from jaxscape.solvers import PyAMGSolver

solver = PyAMGSolver(rtol=1e-6, maxiter=100_000)
distance = ResistanceDistance(solver=solver)
dist = distance(grid)
```

**Tuning parameters:**
```python
# Faster, less accurate
solver = PyAMGSolver(rtol=1e-4, maxiter=10_000)

# Custom AMG method
import pyamg
solver = PyAMGSolver(pyamg_method=pyamg.ruge_stuben_solver)
```

#### Cholesky Solver

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      show_source: false
      heading_level: 3

```python
from jaxscape.solvers import CholmodSolver

solver = CholmodSolver()
distance = ResistanceDistance(solver=solver)
dist = distance(grid)
```

### Using Other Lineax Solvers

You can use any Lineax solver with `ResistanceDistance`:

```python
import lineax as lx

# Conjugate Gradient
solver = lx.CG(rtol=1e-6, atol=1e-6)
distance = ResistanceDistance(solver=solver)
```

See the [Lineax documentation](https://docs.kidger.site/lineax/) for available solvers and their options.

### Direct Usage Example

```python
from jaxscape.solvers import linear_solve, batched_linear_solve, PyAMGSolver
import jax.numpy as jnp

# Get adjacency matrix
A = grid.get_adjacency_matrix()

# Single solve: Ax = b
b = jnp.ones(A.shape[0])
solver = PyAMGSolver()
x = linear_solve(A, b, solver)

# Batched solve: AX = B
B = jnp.stack([b, 2*b, 3*b], axis=-1)
X = batched_linear_solve(A, B, solver)
```

## Choosing a Solver

```python
# Small graph: default is fine
if grid.nv < 5000:
    distance = ResistanceDistance()

# Large graph: use iterative solver
elif grid.nv < 50000:
    distance = ResistanceDistance(solver=CholmodSolver())

# Very large graph: AMG is best
else:
    distance = ResistanceDistance(solver=PyAMGSolver(rtol=1e-5))
```

## Low-Level Solver methods and classes

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      show_source: false
      heading_level: 3
      
::: jaxscape.solvers.operator.linear_solve
    options:
      show_source: true
      heading_level: 3

::: jaxscape.solvers.operator.batched_linear_solve
    options:
      show_source: true
      heading_level: 3
