<h1 align='center'>JAXScape</h1>

-----

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/jaxscape.svg)](https://pypi.org/project/jaxscape)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxscape.svg)](https://pypi.org/project/jaxscape) -->

JAXScape is a minimal JAX library for connectivity analysis at scales. In the spirit of PyTorch, Equinox, BlackJAX or Flux, JAXScape aims to provide key utilities to build your own connectivity analysis workflow, including

- differentiable graph builds
- differentiable graph distance metrics
- moving window utilities
- graph to raster mappings

JAXScape leverages JAX's capabilities to accelerate distance computations on CPUs/GPUs/TPUs, while ensuring differentiability of all implemented classes and methods for awesome sensitivity analysis and optimization.


## Features 🚀
- [x] `GridGraph` with differentiable adjacency matrix method
- [x] Euclidean distance
- [x] Randomized shortest path distance ([REF](https://arxiv.org/pdf/1212.1666))
- [x] Moving window generator
- [ ] Linear solver choices, based on [Lineax](https://github.com/patrick-kidger/lineax)
- [ ] Preconditioners for Krylov-based solvers (GMRES, Conjugate Gradient)
  - See [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/)  or [PyAMG](https://github.com/pyamg/pyamg)
- [ ] Landmark implementation (see [ConScape](https://conscape.org/notebooks/nbk_landmarks.html))
- [ ] Differentiable connected component algorithm (see https://github.com/jax-ml/jax/issues/24737)
- [ ] Least-cost path (see https://github.com/srush/torch-queue/)
- [ ] Resistance distance (see [CircuitScape.jl](https://github.com/Circuitscape/Circuitscape.jl/blob/master/src/core.jl) implementation)
- [ ] Benchmark
  - scaling with number of nodes, CPU/GPU (python based)
  - benchmark against CircuitScape and ConScape (Julia based)

## Installation

```console
pip install jaxscape
```

## Documentation

To build.

## Quick example

Here we calculate the contribution of each pixel to the functional habitat connectivity of the landscape.

```python
import jax
import jax.numpy as jnp
from jaxscape.rsp_distance import RSPDistance
from jaxscape.gridgraph import GridGraph
import matplotlib.pyplot as plt

D = 1.0  # dispersal distance
theta = jnp.array(1.)
distance = RSPDistance(theta)

# Define a habitat suitability raster
habitat_suitability = jnp.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype="float32",
)
plt.imshow(habitat_suitability)
plt.show()
```
<div align="center"><img src="habitat_quality.png" alt="Sensitivities"></div>


```python
activities = habitat_suitability > 0

# Equivalent connected habitat calculation.
# We first need to calculate a distance, 
# that we transform into an ecological proximity
def calculate_ech(habitat_quality):
    grid = GridGraph(activities=activities, vertex_weights=habitat_quality)
    dist = distance(grid)
    K = jnp.exp(-dist / D)
    q = grid.get_active_vertices_weights()
    ech = jnp.sqrt(q @ (K @ q)) 
    return ech

# derivative of w.r.t pixel habitat suitability 
# represents pixel contribution to landscape connectivity
grad_ech = jax.grad(calculate_ech)
sensitivities = grad_ech(habitat_suitability)

plt.imshow(sensitivities)
plt.show()
```
<div align="center"><img src="sensitivities.png" alt="Sensitivities"></div>


## License

`jaxscape` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
