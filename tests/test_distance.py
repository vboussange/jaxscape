import jax
import jax.numpy as jnp
import jax.random as jr
from equinox import filter_grad, filter_jit
from jaxscape import (
    EuclideanDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    RSPDistance,
)


DISTANCES = [
    EuclideanDistance(),
    LCPDistance(),
    ResistanceDistance(),
    RSPDistance(theta=0.1),
]


def test_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (3, 3))  # Start with a uniform permeability

    for distance in DISTANCES:
        distance_jitted = filter_jit(distance)
        grid = GridGraph(permeability_raster, fun=lambda x, y: (x + y) / 2)
        dist = distance_jitted(grid)
        assert dist.shape == (grid.nv, grid.nv)

        dist = distance_jitted(grid, sources=jnp.array([0, 1]), targets=jnp.array([2]))
        assert dist.shape == (2, 1)

        dist = distance_jitted(grid, nodes=jnp.array([0, 1]))
        assert dist.shape == (2, 2)


def test_differentiability():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (2, 2))  # Start with a uniform permeability

    for distance in DISTANCES:

        def objective(permeability_raster):
            grid = GridGraph(permeability_raster, fun=lambda x, y: (x + y) / 2)
            dist = distance(grid)
            return jnp.sum(dist)

        grad_objective = filter_jit(filter_grad(objective))
        # %timeit grad_objective(permeability_raster) # 71.2 ms ± 16.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
        dobj = grad_objective(permeability_raster)
        assert isinstance(dobj, jax.Array)
