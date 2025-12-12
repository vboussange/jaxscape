# Installation

## Basic Installation

Install JAXScape using `uv`:

```console
uv add git+https://github.com/vboussange/jaxscape.git
```

Or `pip`:

```console
pip install git+https://github.com/vboussange/jaxscape.git
```

## GPU Support

For GPU compatibility, install JAX following the [official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html). JAXScape will automatically use the JAX backend you have configured.

## Optional Solvers

Linear solvers improve performance for large-scale resistance distance computations. Install via `uv`:

```console
# PyAMG solver (algebraic multigrid)
uv add git+https://github.com/vboussange/jaxscape.git --extra pyamg

# Cholesky solver
uv add git+https://github.com/vboussange/jaxscape.git --extra cholespy

# All solvers
uv add git+https://github.com/vboussange/jaxscape.git --extra solvers
```

Or manually with `pip`:

```console
pip install pyamg scikit-sparse
```

!!! warning "Python Version"
    Solver packages require Python 3.10-3.12 only.