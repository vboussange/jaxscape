import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from jax import grad, jit
from jax.experimental.sparse import BCOO
from jaxscape import GridGraph
from jaxscape.rsp_distance import rsp_distance, RSPDistance
from jaxscape.utils import mapnz


def test_rsp_distance_matrix():
    # This is the base example taken from ConScape
    expected_cost_conscape = jnp.array(
        [
            [0.0, 1.01848, 1.01848, 2.01848],
            [1.01848, 0.0, 2.01848, 1.01848],
            [1.01848, 2.01848, 0.0, 1.01848],
            [2.01848, 1.01848, 1.01848, 0.0],
        ]
    )
    permeability_raster = jnp.ones((2, 2))
    grid = GridGraph(grid=permeability_raster)
    theta = jnp.array(2.0)
    distance = RSPDistance(theta, cost=lambda x: x)
    mat = distance(grid)
    assert jnp.allclose(mat, expected_cost_conscape, atol=1e-4)


def test_rsp_distance():
    A = BCOO.fromdense(
        jnp.array(
            [
                [0, 1.2, 1.2, 0, 0, 0],
                [1.2, 0, 0, 1.2, 0, 0],
                [1.2, 0, 0, 0, 1.2, 0],
                [0, 1.5, 0, 0, 0, 1.5],
                [0, 0, 1.5, 0, 0, 1.5],
                [0, 0, 0, 1.5, 1.5, 0],
            ],
            dtype="float32",
        )
    )
    theta = jnp.array(1.0)
    C = mapnz(A, lambda x: -jnp.log(x))
    dist = rsp_distance(theta, A, C)
    assert isinstance(dist, jax.Array)


# test with true raster
# TODO: this test is broken, as we cannot prune a GridGraph with false `activities`.
# def test_rsp_distance_matrix():

#     raster_path = Path(__file__).parent / "data/habitat_suitability.csv"
#     habitat_suitability = jnp.array(np.loadtxt(raster_path, delimiter=","))

#     conscape_dist_path = Path(__file__).parent /  "data/conscape_rsp_distance_to_i=19_j=6.csv"
#     expected_cost_conscape = jnp.array(np.loadtxt(conscape_dist_path, delimiter=","))
#     grid = GridGraph(habitat_suitability)


#     # calculating distance to vertex 19, 6 in julia coordinates (corresponding to vertex 18, 5 in python coordinate)
#     theta = jnp.array(0.01)
#     distance = RSPDistance(theta)
#     mat = distance(grid)
#     vertex_index = grid.coord_to_index(18, 5)
#     expected_cost = grid.node_values_to_array(mat[:, vertex_index])

#     assert jnp.allclose(expected_cost[~jnp.isnan(expected_cost)], expected_cost_conscape[~jnp.isnan(expected_cost_conscape)], rtol = 1e-6)


def test_differentiability_rsp_distance_matrix():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    theta = jnp.array(0.01)
    distance = RSPDistance(theta)

    def objective(permeability_raster):
        grid = GridGraph(permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)

    grad_objective = grad(objective)
    # %timeit grad_objective(permeability_raster) # 71.2 ms ± 16.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)
    # TODO: implement proper test


def test_jit_differentiability_rsp_distance():
    key = jr.PRNGKey(0)  # Random seed is explicit in JAX
    permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
    theta = jnp.array(0.01)
    distance = RSPDistance(theta=theta)

    def objective(permeability_raster):
        grid = GridGraph(permeability_raster)
        dist = distance(grid)
        return jnp.sum(dist)

    grad_objective = jit(grad(objective))
    # %timeit grad_objective(permeability_raster) # 13 μs ± 4.18 μs per loop (mean ± std. dev. of 7 runs, 1 loop each)
    dobj = grad_objective(permeability_raster)
    assert isinstance(dobj, jax.Array)


if __name__ == "__main__":
    pytest.main()
