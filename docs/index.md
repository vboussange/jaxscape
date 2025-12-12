# JAXScape

[![tests](https://github.com/vboussange/jaxscape/actions/workflows/run_tests.yml/badge.svg)](https://github.com/vboussange/jaxscape/actions/workflows/run_tests.yml)
[![DOI](https://zenodo.org/badge/883274673.svg)](https://doi.org/10.5281/zenodo.15267703)

<div align="center">
  <img src="assets/logo.png" alt="JAXScape Logo" width="500">
</div>

JAXScape is a minimal JAX library for connectivity analysis at scale. It provides key utilities to build your own connectivity analysis workflow, including:

- Differentiable raster to graph and graph to raster mappings
- Differentiable graph distance metrics
- Moving window utilities

JAXScape leverages JAX's capabilities to accelerate distance computations on CPUs/GPUs/TPUs, while ensuring differentiability of all implemented classes and methods for sensitivity analysis and optimization.

## Key Features

- **Automatic Differentiation**: All distance metrics are fully differentiable through JAX's autodiff
- **Hardware Acceleration**: Seamlessly run computations on CPUs, GPUs, or TPUs
- **Scalable**: Efficient implementations for large-scale connectivity analysis
- **Flexible**: Build custom workflows with modular components
- **Ecological Focus**: Designed for landscape connectivity and ecological network analysis

## Distance Metrics

JAXScape implements several distance metrics commonly used in connectivity analysis:

- **LCP Distance** (Least-Cost Path): Shortest path distance using Bellman-Ford algorithm
- **Resistance Distance**: Effective electrical resistance treating the graph as a resistor network
- **RSP Distance** (Randomized Shortest Path): Distance accounting for path randomization and temperature parameters
- **Euclidean Distance**: Straight-line distance between grid coordinates

## Getting Started

Get started with the [Installation Guide](getting-started/installation.md) or dive into the [Quick Start Tutorial](getting-started/quickstart.md).

## Citation

If you use JAXScape in your research, please cite:

```bibtex
@software{jaxscape2024,
  author = {Boussange, Victor},
  title = {JAXScape: A minimal JAX library for connectivity modelling at scale},
  year = {2024},
  doi = {10.5281/zenodo.15267703},
  url = {https://github.com/vboussange/jaxscape}
}
```

## License

JAXScape is licensed under the MIT License.
