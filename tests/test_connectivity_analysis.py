import jax.numpy as jnp
from jaxscape.connectivity_analysis import ConnectivityAnalysis
from jaxscape.gridgraph import GridGraph
from jaxscape.euclidean_distance import EuclideanDistance
import matplotlib.pyplot as plt
import jax.random as jr


def test_connectivity_analysis():
    D = 10
    quality_raster = jr.uniform(jr.PRNGKey(0), (103, 103))
    distance = EuclideanDistance()
    proximity = lambda dist: jnp.exp(-dist / D) * (dist < D)
    dependency_range=D

    # we manually pad the quality raster because tiled connectivity analysis
    # perfpormed by `ConnectivityAnalysis` skips pixels in 
    # buffer zone of the window operation
    quality_raster = jnp.pad(
        quality_raster,
        ((dependency_range, dependency_range),
        (dependency_range, dependency_range)),
        mode="constant",
        constant_values=0
    )

    # estimation via tiled connectivity analysis
    prob = ConnectivityAnalysis(quality_raster=quality_raster,
                                permeability_raster=quality_raster,
                                distance=distance,
                                proximity=proximity,
                                coarsening_factor=0.,
                                dependency_range=dependency_range,
                                batch_size=16)
    print("Coarsening:", prob.window_op.window_size-1)
    print("Total window size", prob.window_op.total_window_size)
    ech_tiled = prob.run()
    print(ech_tiled)

    # NOTES:
    # - batch size does not seem to change ech_tiled, which is expected
    # - dependency does seem to change ech_tiled, which is expected (modifies perimeter buffer zone)

    # estimation via direct connectivity analysis
    grid = GridGraph(vertex_weights=quality_raster, fun= lambda x, y: (x + y)/2)
    q = grid.array_to_node_values(quality_raster)
    dist = distance(grid)
    K = proximity(dist)
    K = K.at[jnp.diag_indices_from(K)].set(0)
    ech = q @ K @ q.T
    print(ech)

    assert jnp.allclose(ech, ech_tiled, rtol=1e-2)
    # TODO: we do not get an exact match, which should be investigated further
