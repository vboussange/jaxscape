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
import jaxscape


D = 1. # dispersal distance
theta = jnp.array(0.01)
distance = jaxscape.RSPDistance(theta)

def calculate_ech(habitat_quality):
    grid = jaxscape.GridGraph(activities=activities, vertex_weights=habitat_quality)
    dist = distance(grid)
    proximity = jnp.exp(-dist / D)
    landscape = jaxscape.Landscape(habitat_quality, proximity)
    ech = landscape.equivalent_connected_habitat()
    return ech
    
grad_ech = grad(calculate_ech)
sensitivities = grad_ech(habitat_quality)

plt.imshow(sensitivities)
```

## License

`jaxscape` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
