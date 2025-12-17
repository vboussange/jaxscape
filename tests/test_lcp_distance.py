import jax
import jax.numpy as jnp
import jax.random as jr
import scipy.sparse.csgraph as sp
from jax import grad, jit
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph
from jaxscape.lcp_distance import (
    bellman_ford,
    bellman_ford_multi_sources,
    floyd_warshall,
    LCPDistance,
)
from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford


def test_floyd_warshall():
    # direct `floyd_warshall` test
    D = jnp.array(
        [
            [0, 3, jnp.inf, 7],
            [8, 0, 2, jnp.inf],
            [5, jnp.inf, 0, 1],
            [2, jnp.inf, jnp.inf, 0],
        ]
    )

    A = BCOO.fromdense(1 / D)

    shortest_paths_jax = floyd_warshall(A)
    shortest_paths_scipy = sp.floyd_warshall(D, directed=True)
    assert jnp.allclose(shortest_paths_jax, shortest_paths_scipy)


def test_bellman_ford():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    grid = GridGraph(grid=permeability_raster)

    A = grid.get_adjacency_matrix()
    distances_jax = bellman_ford(A, 1)

    D = 1 / A.todense()
    distances_scipy = scipy_bellman_ford(D, indices=1, return_predecessors=False)

    assert jnp.allclose(distances_scipy, distances_jax, rtol=1e-5)

    # many sources
    sources = jnp.array([0, 5, 10])  # Replace with your landmark nodes
    distance_mat_jaxscape = bellman_ford_multi_sources(A, sources)
    distance_mat_scipy = scipy_bellman_ford(
        D, indices=sources, return_predecessors=False
    )
    assert jnp.allclose(distance_mat_jaxscape, distance_mat_scipy, rtol=1e-5)


def test_bellman_ford_floyd_warshall_differentiability():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability

    def sum_bellman_ford(permeability_raster):
        grid = GridGraph(permeability_raster)
        A = grid.get_adjacency_matrix()
        distances_jax = bellman_ford_multi_sources(
            A, jnp.arange(permeability_raster.size)
        )
        return jnp.sum(distances_jax)

    grad_sum_bellman_ford = jit(grad(sum_bellman_ford))
    sensitivity_bellman_ford = grad_sum_bellman_ford(permeability_raster)

    def sum_floyd_warshall(permeability_raster):
        grid = GridGraph(grid=permeability_raster)
        A = grid.get_adjacency_matrix()
        distances_jax = floyd_warshall(A)
        return jnp.sum(distances_jax)

    grad_sum_floyd_warshall = jit(grad(sum_floyd_warshall))
    sensitivity_floyd_warshall = grad_sum_floyd_warshall(permeability_raster)

    # import matplotlib.pyplot as plt
    # plt.imshow(sensitivity_bellman_ford)
    # plt.imshow(sensitivity_floyd_warshall)

    assert jnp.allclose(sensitivity_bellman_ford, sensitivity_floyd_warshall, rtol=1e-1)


def test_LCPDistance_sources_sparse():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    grid = GridGraph(grid=permeability_raster)
    distance = LCPDistance()

    sources = jnp.array([1, 3, 5])
    D = 1 / grid.get_adjacency_matrix().todense()
    lcp_scipy_sources = sp.floyd_warshall(D, directed=True)[sources, :]
    lcp_jax_sources = distance(grid, sources=sources)
    assert jnp.allclose(lcp_scipy_sources, lcp_jax_sources, rtol=1e-2)


def test_differentiability_lcp_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    distance = LCPDistance()

    def objective(permeability_raster):
        grid = GridGraph(grid=permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)

    grad_objective = grad(objective)
    dobj = grad_objective(
        permeability_raster
    )  # 1.26 ms ± 13.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    assert isinstance(dobj, jax.Array)

    # test with sources
    sources = jnp.array([0, 1, 2])

    def objective(permeability_raster):
        grid = GridGraph(grid=permeability_raster)
        dist = distance(grid, sources)

        return jnp.sum(dist)

    grad_objective = grad(objective)
    dobj = grad_objective(
        permeability_raster
    )  # 1.26 ms ± 13.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    assert isinstance(dobj, jax.Array)
