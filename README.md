<h1 align='center'>JAXScape</h1>

<!-- [![PyPI - Version](https://img.shields.io/pypi/v/jaxscape.svg)](https://pypi.org/project/jaxscape)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxscape.svg)](https://pypi.org/project/jaxscape) -->

JAXScape is a JAX library for ecological connectivity analysis at scales. It leverages JAX's capabilities to accelerates heavy computations on CPUs/GPUs/TPUs, and enables sensitivity analysis at low cost by lifting automatic differentiation.

-----

JAXScape implements:
- differentiable methods to derive permeability maps based on rasters of habitat suitability,
- differentiable distance metrics, including euclidean distance, randomized shortest path (and soon to be, least-cost-path and resistance distance),
- parallelizable utilities for moving window calculations.

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
from jaxscape.landscape import Landscape
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

# `equivalent_connected_habitat` calculation
# We first need to calculate a distance, 
# that we transform into an ecological proximity
def calculate_ech(habitat_quality):
    grid = GridGraph(activities=activities, vertex_weights=habitat_quality)
    dist = distance(grid)
    proximity = jnp.exp(-dist / D)
    landscape = Landscape(habitat_quality, proximity)
    ech = landscape.equivalent_connected_habitat()
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
