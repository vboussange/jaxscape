import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxscape import (
    ConnectivityAnalysis,
    EuclideanDistance,
    GridGraph,
    SensitivityAnalysis,
)


def test_connectivity_analysis():
    D = 10
    quality_raster = jr.uniform(jr.PRNGKey(0), (23, 23))
    distance = EuclideanDistance()
    proximity = lambda dist: jnp.exp(-dist / D) * (dist < D)

    # we manually pad the quality raster because tiled connectivity analysis
    # perfpormed by `ConnectivityAnalysis` skips pixels in
    # buffer zone of the window operation
    quality_raster = jnp.pad(
        quality_raster, ((D, D), (D, D)), mode="constant", constant_values=0
    )

    # estimation via tiled connectivity analysis
    prob = ConnectivityAnalysis(
        quality_raster=quality_raster,
        permeability_raster=quality_raster,
        distance=distance,
        proximity=proximity,
        coarsening_factor=0.0,
        dependency_range=D,
        batch_size=16,
    )
    print("Coarsening:", prob.window_op.window_size - 1)
    print("Total window size", prob.window_op.total_window_size)
    ech_tiled = prob.run()
    print(ech_tiled)

    # estimation via direct connectivity analysis
    grid = GridGraph(grid=quality_raster, fun=lambda x, y: (x + y) / 2)
    q = grid.array_to_node_values(quality_raster)
    dist = distance(grid)
    K = proximity(dist)
    K = K.at[jnp.diag_indices_from(K)].set(0)
    ech = q @ K @ q.T
    print(ech)

    assert jnp.allclose(ech, ech_tiled, rtol=1e-2)
    # TODO: we do not get an exact match, which should be investigated further


def test_sensitivity_analysis():
    D = 11
    quality_raster = jr.uniform(jr.PRNGKey(0), (21, 21))
    distance = EuclideanDistance()
    proximity = lambda dist: jnp.exp(-dist / D) * (dist < D)
    dependency_range = D

    # we manually pad the quality raster because tiled connectivity analysis
    # performed by `ConnectivityAnalysis` skips pixels in
    # buffer zone of the window operation
    quality_raster_padded = jnp.pad(
        quality_raster,
        ((dependency_range, dependency_range), (dependency_range, dependency_range)),
        mode="constant",
        constant_values=0,
    )

    # estimation via tiled sensitivity analysis
    prob = SensitivityAnalysis(
        quality_raster=quality_raster_padded,
        permeability_raster=quality_raster_padded,
        distance=distance,
        proximity=proximity,
        coarsening_factor=0.0,
        dependency_range=dependency_range,
        batch_size=10,
    )
    print("Raster original shape:", prob.original_shape)
    print("Raster padded shape:", prob.quality_raster.shape)
    print("Coarsening:", prob.window_op.window_size - 1)
    print("Total window size", prob.window_op.total_window_size)
    d_quality_tiled = prob.run("quality") * (quality_raster_padded > 0)

    # estimation via direct sensitivity analysis
    @eqx.filter_jit
    @eqx.filter_grad
    def grad_quality(quality_raster, permeability_raster):
        grid = GridGraph(permeability_raster)
        q = grid.array_to_node_values(quality_raster)
        dist = distance(grid)
        K = proximity(dist) - jnp.eye(dist.shape[0])
        ech = q @ (K @ q.T)
        return ech

    d_quality = grad_quality(
        quality_raster_padded, permeability_raster=quality_raster_padded
    ) * (quality_raster_padded > 0)

    assert jnp.allclose(d_quality_tiled, d_quality, rtol=1e-3)
    # plt.imshow(d_quality_tiled)
    # plt.imshow(d_quality)


# TODO: we should test sensitivity w.r.t permeability as well, for LCPDistance/ResistanceDistance/ResistanceDistance
# TODO: we should also test with coarsening factor > 0
