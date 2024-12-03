import jax
import pytest
import networkx as nx
from jax import grad, jit
import jax.numpy as jnp
from jaxscape.smooth_lcp_distance import bellman_ford_smoothmin, segment_smoothmin, segment_logsumexp
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
import scipy.sparse.csgraph as sp
import jax.random as jr
from jaxscape.utils import mapnz
from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford
from scipy.sparse import csr_matrix
from jax.scipy.special import logsumexp
from jax import lax, ops

def test_segment_logsumexp():
    data = jnp.array([0., 1., 1., 2., 2., 2.])
    segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    result = segment_logsumexp(data, segment_ids, 3)
    expected_result = jnp.array([logsumexp(data[i:i+2]) for i in range(0, len(data), 2)])
    assert jnp.allclose(result, expected_result)
    
def test_segment_logsumexp_differentiability():
    # Data and segment IDs
    data = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    segment_ids = jnp.array([0, 0, 1, 1, 2, 2])
    num_segments = 3

    # Define a simple function wrapping segment_logsumexp
    def func_segment_logsumexp(data):
        return segment_logsumexp(data, segment_ids, num_segments).sum()

    # Define a similar function using jnp.logsumexp
    def func_logsumexp(data):
        unique_segments = jnp.unique(segment_ids)
        result = 0
        for seg in unique_segments:
            mask = segment_ids == seg
            result += logsumexp(data[mask])
        return result

    # Compute gradients
    grad_segment_logsumexp = grad(func_segment_logsumexp)(data)
    grad_logsumexp = grad(func_logsumexp)(data)

    # Check if gradients are close
    assert jnp.allclose(grad_segment_logsumexp, grad_logsumexp), (
        "Gradients of segment_logsumexp do not match those of jnp.logsumexp"
    )
    
def test_segment_smoothmin_differentiability():
    # Data and segment IDs
    data = jnp.array([0.5, 0.5, 1.0, 1.5, 2.0, 2.5, jnp.inf])
    segment_ids = jnp.array([0, 0, 0, 1, 1, 2, 2])
    num_segments = 3
    
    result = segment_smoothmin(data, segment_ids, num_segments, 1e-10)
    assert jnp.allclose(result, jnp.array([0.5, 1.5, 2.5]))

    def func_segment_soothmin(data):
        return segment_smoothmin(data, segment_ids, num_segments, 1e-10).sum()
    grad_segment_logsumexp = grad(func_segment_soothmin)(data)
    assert jnp.allclose(grad_segment_logsumexp, jnp.array([0.5, 0.5, 0. , 1. , 0. , 1. , 0. ]))
    
def test_bellman_ford_smoothmin():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities,
                vertex_weights = permeability_raster)

    # many sources
    A = grid.get_adjacency_matrix()
    sources = jnp.arange(grid.nb_active)  # Replace with your landmark nodes
    distance_mat_jaxscape = bellman_ford_smoothmin(A.data, A.indices, grid.nb_active, sources, 1e-10)
    A_scipy = A.todense()
    distance_mat_scipy = scipy_bellman_ford(A_scipy, indices=sources, return_predecessors=False)
    assert jnp.allclose(distance_mat_jaxscape, distance_mat_scipy, atol=1e-5)
    
def test_bellman_ford_smoothmin_differentiability():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    def sum_bellman_ford_softmin(permeability_raster):
        grid = GridGraph(activities=activities,
                    vertex_weights = permeability_raster,
                    nb_active = permeability_raster.size)
        A = grid.get_adjacency_matrix()
        distances_jax = bellman_ford_smoothmin(A.data, A.indices, grid.nb_active, jnp.arange(grid.nb_active), 1e-20)
        return jnp.sum(distances_jax)
    
    grad_sum_bellman_ford = jit(grad(sum_bellman_ford_softmin))
    sensitivity_bellman_ford = grad_sum_bellman_ford(permeability_raster)
    
    assert isinstance(sensitivity_bellman_ford, jax.Array)