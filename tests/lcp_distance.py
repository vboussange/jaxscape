import jax
import pytest
import networkx as nx
from jax import grad, jit
import jax.numpy as jnp
from jaxscape.lcp_distance import floyd_warshall, bellman_ford, LCPDistance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
import scipy.sparse.csgraph as sp
import jax.random as jr
from jaxscape.utils import mapnz
from scipy.sparse.csgraph import bellman_ford as scipy_bellman_ford
from scipy.sparse import csr_matrix

def test_floyd_warshall():
    # direct `floyd_warshall` test
    D = jnp.array([
        [0,     3,   jnp.inf, 7],
        [8,     0,     2,     jnp.inf],
        [5,     jnp.inf, 0,     1],
        [2,     jnp.inf, jnp.inf, 0]
    ])

    shortest_paths_jax = floyd_warshall(D)
    shortest_paths_scipy = sp.floyd_warshall(D, directed=True)
    assert jnp.allclose(shortest_paths_jax, shortest_paths_scipy) # pass 

def test_floyd_warshall_differentiability():
    # direct `floyd_warshall` test
    D = jnp.array([
        [0,     3,   jnp.inf, 7],
        [8,     0,     2,     jnp.inf],
        [5,     jnp.inf, 0,     1],
        [2,     jnp.inf, jnp.inf, 0]
    ])
    
    def myfun(D):
        return jnp.sum(floyd_warshall(D))

    grad_myfun = grad(myfun)
    grad_myfun(D) # pass


def test_bellman_ford():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities,
                vertex_weights = permeability_raster)

    A = grid.get_adjacency_matrix()
    distances_jax = bellman_ford(A, jnp.array([1]))

    A_scipy = A.todense()
    distances_scipy = scipy_bellman_ford(A_scipy, indices=1, return_predecessors=False)

    assert jnp.allclose(distances_scipy, distances_jax)
    
    # many sources
    sources = jnp.array([0, 5, 10])  # Replace with your landmark nodes
    distance_mat_jaxscape = bellman_ford(A, sources)
    distance_mat_scipy = scipy_bellman_ford(A_scipy, indices=sources, return_predecessors=False)
    assert jnp.allclose(distance_mat_jaxscape, distance_mat_scipy)
    
def test_bellman_ford_differentiability():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    def myfun(permeability_raster):
        grid = GridGraph(activities=activities,
                    vertex_weights = permeability_raster,
                    nb_active = permeability_raster.size)
        A = grid.get_adjacency_matrix()
        distances_jax = bellman_ford(A, jnp.array([1]))
        return jnp.sum(distances_jax)
    
    grad_myfun = jit(grad(myfun))
    sensitivity = grad_myfun(permeability_raster)
    assert isinstance(sensitivity, jax.Array)

def test_LCPDistance_landmarks_sparse():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities,
                vertex_weights = permeability_raster)
    distance = LCPDistance()
    
    landmarks = jnp.array([1, 3, 5])
    D = csr_matrix(grid.get_adjacency_matrix().todense())
    lcp_scipy_landmarks = sp.floyd_warshall(D, directed=True)[landmarks, :]
    lcp_jax_landmarks = distance(grid, landmarks=landmarks)
    assert jnp.allclose(lcp_scipy_landmarks, lcp_jax_landmarks, rtol=1e-2)


def test_differentiability_lcp_distance():    
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    D = 1.
    distance = LCPDistance()

    def objective(permeability_raster):
        grid = GridGraph(activities=activities,
                        vertex_weights = permeability_raster)
        dist = distance(grid)
        proximity = jnp.exp(-dist / D)
        landscape = ExplicitGridGraph(activities=activities, 
                                      vertex_weights = permeability_raster,
                                      adjacency_matrix = proximity)
        func = landscape.equivalent_connected_habitat()
        return func
        
    grad_objective = grad(objective)
    dobj = grad_objective(permeability_raster) # 1.26 ms ± 13.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    assert isinstance(dobj, jax.Array)
    
    # test with landmarks
    landmarks = jnp.array([0, 1, 2])
    def objective(permeability_raster):
        grid = GridGraph(activities=activities,
                        vertex_weights = permeability_raster)
        dist = distance(grid, landmarks)

        return jnp.sum(dist)
        
    grad_objective = grad(objective)
    dobj = grad_objective(permeability_raster) # 1.26 ms ± 13.8 μs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
    assert isinstance(dobj, jax.Array)
    
    