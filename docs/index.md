# Getting started

JAXScape is a minimal JAX library for connectivity analysis at scales. It provide key utilities to build your own connectivity analysis workflow, including

- differentiable and GPU-accelerated graph distance metrics
- differentiable raster to graph and graph to raster mappings
- moving window utilities for implementing large-scale connectivity analysis pipelines

JAXScape leverages JAX's capabilities to accelerate distance computations on CPUs/GPUs/TPUs, while ensuring differentiability of all implemented classes and methods for awesome sensitivity analysis and optimization.


## Installation

```console
uv add jaxscape
```

The default installation is CPU-only. For NVIDIA GPUs, use `uv add 'jaxscape[cuda]'`, or install JAX following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html). JAXScape will automatically use the JAX backend you have configured.

You may be required to install optional linear solvers for large-scale resistance and free-energy distance computations. `CholmodSolver`, `PyAMGSolver`, and `AMJaxCGSolver` are documented on the [linear solver page](https://vboussange.github.io/jaxscape/api/linear_solvers), including the one-time `init_preconditioner(...)` workflow used by `AMJaxCGSolver` for JIT-compiled and differentiable repeated solves.

## Resistance distance methods

`ResistanceDistance()` computes exact resistance distances by default. For
larger graphs, use the Spielman-Srivastava randomized approximation by selecting
an explicit method object:

```python
from jaxscape import ResistanceDistance, SpielmanApproximation
from jaxscape.solvers import PyAMGSolver

distance = ResistanceDistance(
    method=SpielmanApproximation(epsilon=0.05, seed=0),
    solver=PyAMGSolver(),
)
```

The `epsilon` parameter belongs to `SpielmanApproximation`: smaller values use
more random projections and typically improve accuracy at higher memory and
solve cost. The approximate method remains compatible with `jax.grad` via a
custom implicit-differentiation rule.

## Quick start

This small example uses positive, symmetric conductances and the dense default
solvers. Edge costs for least-cost and free-energy distances are reciprocals of
conductance. Larger landscapes need an explicit
[linear solver](https://vboussange.github.io/jaxscape/api/linear_solvers/).

```python
import jax

# Float64 gives the free-energy calculation a wider usable numerical range.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxscape import FreeEnergyDistance, GridGraph, LCPDistance, ResistanceDistance

permeability = jnp.array(
    [[1.0, 1.0, 0.5, 1.0],
     [1.0, 0.5, 0.5, 1.0],
     [1.0, 1.0, 1.0, 1.0]],
    dtype=jnp.float64,
)
# The default GridGraph weight function can be directed. Use a symmetric one.
grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
source = jnp.array([[0, 0]])
distances = {
    "Least-cost distance": LCPDistance(),
    "Free-energy distance": FreeEnergyDistance(theta=0.2),
    "Resistance distance": ResistanceDistance(),
}

fig, axs = plt.subplots(1, 4, figsize=(12, 3))
im = axs[0].imshow(permeability, cmap="gray", vmin=0, vmax=1)
axs[0].set_title("Permeability")
fig.colorbar(im, ax=axs[0], shrink=0.6)
for ax, (title, distance) in zip(axs[1:], distances.items()):
    values = distance(grid, sources=source)
    im = ax.imshow(np.asarray(values).reshape(permeability.shape), cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.6)
for ax in axs:
    ax.axis("off")
fig.suptitle("Distances from the top-left pixel")
plt.tight_layout()
plt.show()
```

`theta` is **inverse temperature**: increasing it favors least-cost paths.
As it decreases toward zero, free-energy distance approaches a scaled effective
resistance; on a loop-free graph with `m` undirected edges, the limit is `m`
times the resistance distance. These limits may be numerically unresolved at
extreme parameter values; the calculation then raises an error.

<div align="center"><img src="examples/distance_calculation/distances.png" alt="Distances"  width="600"></div>

But what's really cool about jaxscape is that you can autodiff through thoses distances! Check out the Examples section to learn about applications and more!

## Features and roadmap 🚀
See issues; most notably:

- [ ] Support for direct and iterative linear sparse solvers on GPU (cf [spineax](https://github.com/johnviljoen/spineax))
- [ ] Benchmark against `CircuitScape`, `ConScape.jl` and [`radish`](https://github.com/nspope/radish).

## License

`jaxscape` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## Related packages
- gdistance
- ConScape
- Circuitscape
- graphhab
- conefor
- resistanceGA
- landscapemetrics
- radish

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
