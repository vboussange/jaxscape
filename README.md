[![tests](https://github.com/vboussange/jaxscape/actions/workflows/run_tests.yml/badge.svg)](https://github.com/vboussange/jaxscape/actions/workflows/run_tests.yml)
[![Build documentation](https://github.com/vboussange/jaxscape/actions/workflows/build_docs.yml/badge.svg)](https://github.com/vboussange/jaxscape/actions/workflows/build_docs.yml)
[![DOI](https://zenodo.org/badge/883274673.svg)](https://doi.org/10.5281/zenodo.15267703)

<div align="center">
  <img src="docs/examples/logo/logo.png" alt="JAXScape Logo" width="400">
</div>
<!-- [![PyPI - Version](https://img.shields.io/pypi/v/jaxscape.svg)](https://pypi.org/project/jaxscape)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/jaxscape.svg)](https://pypi.org/project/jaxscape) -->

---

JAXScape is a minimal JAX library for connectivity analysis at scales. It provide key utilities to build your own connectivity analysis workflow, including

- differentiable and GPU-accelerated graph distance metrics
- differentiable raster to graph and graph to raster mappings
- moving window utilities for implementing large-scale connectivity analysis pipelines

JAXScape implements custom methods and leverages JAX's capabilities to accelerate distance computations on CPUs/GPUs/TPUs, while ensuring differentiability across the codebase for awesome sensitivity analysis and optimization.

## Why JAXScape?

We report below some key runtimes from a comprehensive benchmark testing JAXScape on a RTX 3090 GPU against other related CPU-based software in Python, R, and Julia backends. 
See the [benchmark documentation](https://vboussange.github.io/jaxscape/benchmark/)
for the full setup, raw scorecards, convergence status, and fit-quality metrics.

### Resistance distance calculation
On a `1000 x 1000` spatial grid, JAXScape is `74x` faster than [`Circuitscape.jl`](https://github.com/Circuitscape/Circuitscape.jl) with `cg+amg` solver and `17x` faster than [`Circuitscape.jl`](https://github.com/Circuitscape/Circuitscape.jl) with `cholmod` solver. 
<!-- TODO: [`gdistance`](https://cran.r-project.org/package=gdistance) `commuteDistance` result. -->

<!-- ### Least-cost path
On a `1000 x 1000` spatial grid, the current scorecard does not report a successful JAXScape GPU run, so no GPU speed-up against [`gdistance`](https://cran.r-project.org/package=gdistance) `costDistance` is reported there. -->

### Sensitivity analysis
On a `175 x 175` spatial grid, JAXScape is `12x` faster than [`gdistance`](https://cran.r-project.org/package=gdistance) `shortestPath` for shortest-path sensitivity and `5.0x` faster than [`gdistance`](https://cran.r-project.org/package=gdistance) `passage` for resistance sensitivity. 
<!-- Cosine similarity to the `gdistance` reference surfaces is `0.130` and `0.173`, respectively. -->

### Inverse landscape genetics 
On a `100 x 100` spatial grid, JAXScape is `32x` faster than [`ResistanceGA`](https://github.com/wpeterman/ResistanceGA) and reaches a final relative RMSE about `16x` lower.


## Installation

```console
uv add jaxscape
```

For GPU compatibility, install JAX following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html). JAXScape will automatically use the JAX backend you have configured.

You may be required to install optional linear solvers for large-scale resistance distance computations. `CholmodSolver`, `PyAMGSolver`, and `AMJaxCGSolver` are documented on the [linear solver page](https://vboussange.github.io/jaxscape/api/linear_solvers).

## Quick start

**1. Download sample data**

```bash
curl -sL https://github.com/vboussange/jaxscape/releases/download/v0.0.6/data.zip -o data.zip && unzip -q data.zip && rm data.zip
```

**2. Run the analysis**

```python
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jaxscape import GridGraph
from jaxscape import LCPDistance, ResistanceDistance, RSPDistance

permeability = jnp.array(np.loadtxt("data/permeability.csv", delimiter=",")) + 0.001
barriers = np.asarray(permeability <= 0.1)

# Create a grid graph where edge weights are the average permeability of the two nodes
grid = GridGraph(grid=permeability, fun=lambda x, y: (x + y) / 2)

# We set the source to the top left pixel, and compute distances to all other pixels with three different distance metrics
source = grid.coord_to_index(jnp.array([0]), jnp.array([0]))

distances = {
    "LCP distance": LCPDistance(),
    "Resistance distance": ResistanceDistance(),
    "RSP distance": RSPDistance(theta=0.01, cost=lambda x: -jnp.log(x)),
}

distance_cmap = plt.cm.magma.copy()
distance_cmap.set_bad(color="lightgray")

fig, axs = plt.subplots(1, 4, figsize=(12, 4))

permeability_im = axs[0].imshow(permeability, cmap="gray", vmin=0, vmax=1)
axs[0].set_xticks([])
axs[0].set_yticks([])
for spine in axs[0].spines.values():
    spine.set_visible(True)
axs[0].set_title("Permeability")
fig.colorbar(permeability_im, ax=axs[0], shrink=0.2)

for ax, (title, distance) in zip(axs[1:], distances.items()):
    # Compute distances from all nodes to the source
    dist_to_node = distance(grid, source)

    # Convert from node values to a 2D array and hide barrier cells
    dist_array = np.asarray(grid.node_values_to_array(dist_to_node.ravel()))
    dist_array = np.ma.masked_where(barriers, dist_array)
    
    # Plotting
    im = ax.imshow(dist_array, cmap=distance_cmap)
    ax.axis("off")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.2)

fig.suptitle("Permeability and distance to the top-left pixel")
plt.tight_layout()
plt.show()
```
<div align="center"><img src="docs/examples/distance_calculation/distances.png" alt="Distances"  width="600"></div>

But what's really cool about `JAXscape` is that you can autodiff through thoses distances! Check out the [documentation](#documentation) to learn about applications and more!


## Documentation

Comprehensive documentation is available at [https://vboussange.github.io/jaxscape](https://vboussange.github.io/jaxscape/)

## Benchmarking

JAXScape includes a benchmark workspace that profiles internal JAXScape
implementations and compares them with external landscape-connectivity tools.
Published results and task definitions are available in the
[benchmark documentation](https://vboussange.github.io/jaxscape/benchmark/).
To reproduce the suite or run it on a local CPU/GPU device, follow the
instructions in [`benchmark/README.md`](benchmark/README.md).

## License

`jaxscape` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Citation

If you use JAXScape in your research, please cite:

```bibtex
@software{jaxscape2024,
  author = {Boussange, Victor},
  title = {JAXScape: A minimal JAX library for connectivity modelling at scale},
  year = {2025},
  doi = {10.5281/zenodo.15267703},
  url = {https://github.com/vboussange/jaxscape}
}
```
