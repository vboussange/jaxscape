from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.signal import convolve2d
from jaxscape import (
    ConnectivityAnalysis,
    EuclideanDistance,
    FreeEnergyDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    SensitivityAnalysis,
    SpielmanApproximation,
)
from jaxscape.distance import AbstractDistance
from jaxscape.solvers import CuDSSSolver


D = 3
RNG = np.random.default_rng(0)
QUALITY = jnp.asarray(RNG.uniform(0, 1, (6, 7)) * (RNG.uniform(0, 1, (6, 7)) > 0.2))
# A no-data column cuts the raster in two, and two more no-data pixels.
PERMEABILITY = np.asarray(10 ** RNG.uniform(-1, 0, (6, 7)))
PERMEABILITY[:, 3] = 0
PERMEABILITY[[0, 4], [5, 1]] = 0
PERMEABILITY = jnp.asarray(PERMEABILITY)
# Documented: permeability is raised to at least 1e-4 times its maximum.
FLOOR = 1e-4 * float(jnp.max(PERMEABILITY))


def average(x, y):
    return (x + y) / 2


def proximity(dist):
    return jnp.exp(-dist / D)


def make(kind, distance, quality=QUALITY, permeability=PERMEABILITY, **kwargs):
    # 42 block centres: six batches of 7 windows, or six tiles of 2 x 4 block
    # centres for distances whose sources share work.
    kwargs = {"proximity": proximity, "batch_size": 8, "dependency_range": D, **kwargs}
    return kind(
        quality_raster=quality,
        permeability_raster=permeability,
        distance=distance,
        **kwargs,
    )


def assert_close(actual, expected, rtol):
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * float(np.max(np.abs(expected)))
    )


def pearson(a, b):
    return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]


def smooth(seed, shape):
    """Gaussian-smoothed noise rescaled to [0.05, 1]."""
    x = np.arange(-12, 13)
    kernel = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 32)
    noise = np.random.default_rng(seed).normal(size=shape)
    field = np.asarray(convolve2d(noise, kernel / kernel.sum(), mode="same"))
    return jnp.asarray(0.05 + 0.95 * (field - field.min()) / np.ptp(field))


def with_exact_gradient(randomized, distance, grid):
    """The `randomized` distances, with the gradient of the exact ones on `grid`.

    The gradient of window queries of randomized distances estimates it.
    """
    exact = distance.exact()(grid)
    return exact + jax.lax.stop_gradient(randomized - exact)


def own_windows(quality, permeability, distance, exact=False):
    """Connectivity summed over each pixel's own window, one pixel at a time.

    Each window is centred on its pixel and padded beyond the raster with quality
    0 and the permeability floor, and the distances come from the all-pairs
    query of `distance`, reseeded with the raster index of the pixel, and with
    the gradient of exact distances if `exact`.
    """
    size = 2 * D + 1
    centre = D * size + D
    permeability = jnp.where(permeability > FLOOR, permeability, FLOOR)
    weight = jnp.pad(quality, D)
    permeability = jnp.pad(permeability, D, constant_values=FLOOR)

    def window(i, j):
        w = jax.lax.dynamic_slice(weight, (i, j), (size, size)).ravel()
        p = jax.lax.dynamic_slice(permeability, (i, j), (size, size))
        grid = GridGraph(p, fun=average)
        distances = distance.reseeded(i * quality.shape[1] + j)(grid)
        if exact:
            distances = with_exact_gradient(distances, distance, grid)
        K = distance.debiased_proximity(proximity, grid)(distances[centre])
        return w[centre] * (K.at[centre].set(0) @ w)

    rows, columns = np.indices(quality.shape).reshape(2, -1)
    return jnp.sum(jax.vmap(window)(rows, columns))


def cudss(request, on_cpu):
    if on_cpu:
        request.getfixturevalue("stub_spineax")
        return CuDSSSolver()
    try:
        return CuDSSSolver()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")


# A randomized resistance: its all-pairs query on the same graph is the
# reference.
SPIELMAN = SpielmanApproximation(epsilon=0.3, seed=1)
# Each distance, its dense all-pairs reference, and the tolerance it reaches.
CASES = {
    "spielman": (
        lambda _: ResistanceDistance(method=SPIELMAN),
        ResistanceDistance(solver=None, method=SPIELMAN),
        1e-10,
    ),
    "euclidean": (lambda _: EuclideanDistance(), EuclideanDistance(), 1e-12),
    "lcp": (lambda _: LCPDistance(), LCPDistance(), 1e-12),
    # Larger `theta` cannot resolve paths through the floored pixels.
    "free-energy": (
        lambda _: FreeEnergyDistance(theta=1e-3),
        FreeEnergyDistance(theta=1e-3),
        1e-10,
    ),
    "resistance-dense": (
        lambda _: ResistanceDistance(solver=None),
        ResistanceDistance(solver=None),
        1e-10,
    ),
    "resistance-default": (
        lambda _: ResistanceDistance(),
        ResistanceDistance(solver=None),
        1e-10,
    ),
    "resistance-cudss-on-cpu": (
        lambda request: ResistanceDistance(cudss(request, on_cpu=True)),
        ResistanceDistance(solver=None),
        1e-8,
    ),
    "resistance-cudss": (
        lambda request: ResistanceDistance(cudss(request, on_cpu=False)),
        ResistanceDistance(solver=None),
        1e-8,
    ),
}


def assert_matches(analyses, expected, rtol, min_pearson=None):
    """Compare the runs of `analyses`, by kind, with the connectivity and its
    quality and permeability gradients `expected`, and with `min_pearson`, the
    permeability sensitivity by its Pearson correlation only."""
    connectivity, (d_quality, d_permeability) = expected
    assert_close(analyses[ConnectivityAnalysis].run(), connectivity, rtol)
    sensitivity = analyses[SensitivityAnalysis]
    assert_close(sensitivity.run("quality"), d_quality, rtol)
    if min_pearson:
        assert pearson(sensitivity.run("permeability"), d_permeability) > min_pearson
    else:
        assert_close(sensitivity.run("permeability"), d_permeability, rtol)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.gpu) if name.endswith("cudss") else name
        for name in CASES
    ],
)
def test_analyses_match_each_pixels_own_window(name, request):
    make_distance, reference, rtol = CASES[name]
    distance = make_distance(request)
    # Batches of one source compute each pixel in its own window, as distances
    # whose sources do not share work do in batches of any size: here ten
    # batches of 4 windows, and a last one of 2 that two dummy windows
    # complete, which add nothing.
    batch_size = 1 if distance.sources_share_work else 4
    kinds = (ConnectivityAnalysis, SensitivityAnalysis)
    analyses = {kind: make(kind, distance, batch_size=batch_size) for kind in kinds}
    assert analyses[ConnectivityAnalysis]._tiles_per_batch == batch_size

    # The permeability gradient of randomized distances is estimated from that
    # of exact distances. Measured: Pearson 0.998.
    randomized = name == "spielman"
    expected = jax.value_and_grad(own_windows, (0, 1))(
        QUALITY, PERMEABILITY, reference, randomized
    )
    assert_matches(analyses, expected, rtol, 0.99 if randomized else None)


def padded_raster(quality, permeability, distance, exact=False):
    """Connectivity with the distances on the graph of the padded raster.

    The raster is padded by `D` on all sides with quality 0 and the permeability
    floor, 1e-4 times its maximum permeability, and each pixel counts the pixels
    of its own window. Randomized distances are reseeded with the index 0 of the
    first pixel, and have the gradient of exact distances if `exact`.
    """
    height, width = quality.shape
    size = 2 * D + 1
    floor = 1e-4 * jax.lax.stop_gradient(jnp.max(permeability))
    permeability = jnp.where(permeability > floor, permeability, floor)
    permeability = jnp.pad(permeability, D, constant_values=floor)
    weight = jnp.pad(quality, D)
    grid = GridGraph(permeability, fun=average)
    K = distance.debiased_proximity(proximity, grid)
    distances = distance.reseeded(0)(grid)
    if exact:
        distances = with_exact_gradient(distances, distance, grid)
    distances = distances.reshape(height + 2 * D, width + 2 * D, -1)

    def pixel(i, j):
        row = distances[i + D, j + D].reshape(height + 2 * D, width + 2 * D)
        d = jax.lax.dynamic_slice(row, (i, j), (size, size)).ravel()
        w = jax.lax.dynamic_slice(weight, (i, j), (size, size)).ravel()
        return w[D * size + D] * (K(d).at[D * size + D].set(0) @ w)

    rows, columns = np.indices(quality.shape).reshape(2, -1)
    return jnp.sum(jax.vmap(pixel)(rows, columns))


@pytest.mark.parametrize("name", ["resistance-default", "spielman"])
def test_one_tile_computes_distances_on_the_padded_raster(name, request):
    # Resistance queries share their work between sources: a batch of as many
    # sources as pixels takes the distances of one tile of 5 x 7 pixels, the
    # whole raster.
    make_distance, reference, rtol = CASES[name]
    quality, permeability = QUALITY[:5], PERMEABILITY[:5]
    analyses = {
        kind: make(kind, make_distance(request), quality, permeability, batch_size=35)
        for kind in (ConnectivityAnalysis, SensitivityAnalysis)
    }

    # The permeability gradient of randomized distances is estimated from that
    # of exact distances. Measured: Pearson 0.982.
    randomized = name == "spielman"
    expected = jax.value_and_grad(padded_raster, (0, 1))(
        quality, permeability, reference, randomized
    )
    assert_matches(analyses, expected, rtol, 0.97 if randomized else None)


def test_tiles_approximate_each_pixels_own_window():
    # Batches of 16 pixels take tiles of 4 x 4 pixels. The distances of a pixel
    # are those of a larger graph than its window, and lower. Measured:
    # connectivity +2.7 %, Pearson 0.9997 for the quality sensitivity and
    # 0.9998 for the permeability sensitivity.
    quality, permeability = smooth(1, (20, 20)), smooth(2, (20, 20))

    def run(batch_size):
        def analysis(kind):
            return kind(
                quality_raster=quality,
                permeability_raster=permeability,
                distance=ResistanceDistance(solver=None),
                dependency_range=6,
                batch_size=batch_size,
            )

        sensitivity = analysis(SensitivityAnalysis)
        return sensitivity.tile_sides, (
            analysis(ConnectivityAnalysis).run(),
            sensitivity.run("quality"),
            sensitivity.run("permeability"),
        )

    (sides, windows), (tile_sides, tiles) = run(1), run(16)
    assert sides == (13, 13)
    assert tile_sides == (3 + 13, 3 + 13)
    np.testing.assert_allclose(tiles[0], windows[0], rtol=0.05)
    for tile, window in zip(tiles[1:], windows[1:]):
        assert pearson(tile, window) >= 0.999
    # On a raster of 3 columns, a batch of 16 takes tiles of 5 x 3 block
    # centres.
    narrow = jnp.ones((40, 3))
    windowed = make(
        ConnectivityAnalysis,
        ResistanceDistance(solver=None),
        narrow,
        narrow,
        batch_size=16,
        dependency_range=40,
    )
    assert windowed.tile_sides == (4 + 81, 2 + 81)


def test_float32_tile_sensitivity_matches_float64():
    # Tiles of 2 x 4 pixels and a buffer of 16 floored pixels around them,
    # larger than the coarsest grid of the default solver: resistance queries
    # ground a pixel of near-maximum permeability next to the sources, away
    # from the floored pixels.
    def sensitivity(dtype):
        return make(
            SensitivityAnalysis,
            ResistanceDistance(),
            QUALITY.astype(dtype),
            PERMEABILITY.astype(dtype),
            dependency_range=16,
            proximity=lambda dist: jnp.exp(-dist / 2),
        ).run("permeability")

    expected, actual = sensitivity(jnp.float64), sensitivity(jnp.float32)
    error = jnp.linalg.norm(actual - expected) / jnp.linalg.norm(expected)
    assert error < 1e-3


@pytest.mark.parametrize("q_weighted", [True, False])
def test_euclidean_connectivity_matches_all_pairs(q_weighted):
    def cut_off(dist):
        return jnp.exp(-dist / D) * (dist < D)

    def all_pairs(quality):
        grid = GridGraph(quality, fun=average)
        q = quality.ravel() if q_weighted else jnp.ones(grid.nv)
        K = cut_off(EuclideanDistance()(grid))
        return q @ (K - jnp.eye(grid.nv)) @ q

    def run(kind, *args):
        return make(
            kind, EuclideanDistance(), permeability=QUALITY, proximity=cut_off
        ).run(*args, q_weighted=q_weighted)

    assert_close(run(ConnectivityAnalysis), all_pairs(QUALITY), 1e-12)
    expected = jax.grad(all_pairs)(QUALITY)
    assert_close(run(SensitivityAnalysis, "quality"), expected, 1e-12)


# JAX warns of a sum into an array of another dtype, and will raise.
@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize(
    "quality_dtype, permeability_dtype, distance",
    [
        ("int32", "float64", LCPDistance()),
        ("bool", "float32", LCPDistance()),
        ("float64", "float32", LCPDistance()),
        # Its distances come from integer pixel coordinates.
        ("float32", "float32", EuclideanDistance()),
    ],
)
def test_rasters_of_any_dtypes_give_the_float64_results_in_their_common_dtype(
    quality_dtype, permeability_dtype, distance
):
    # Qualities 0, 1 and 2, or False and True.
    quality = (3 * QUALITY).astype(quality_dtype)
    permeability = PERMEABILITY.astype(permeability_dtype)

    def results(quality, permeability):
        def coarsened(kind):
            return make(kind, distance, quality, permeability, coarsening_factor=0.5)

        sensitivity = coarsened(SensitivityAnalysis)
        return [
            coarsened(ConnectivityAnalysis).run(),
            sensitivity.run("quality"),
            sensitivity.run("permeability"),
        ]

    dtype = "float64" if "float64" in (quality_dtype, permeability_dtype) else "float32"
    expected = results(quality.astype("float64"), permeability.astype("float64"))
    for actual, expected in zip(results(quality, permeability), expected):
        assert actual.dtype == dtype
        assert_close(actual, expected, 1e-5 if dtype == "float32" else 1e-12)


def test_sensitivity_is_the_gradient_of_connectivity_inside_jit():
    # The default solver and the default proximity: creating an analysis
    # builds the solver state and the profile of the proximity, here from
    # traced rasters.
    def analysis(kind, quality=QUALITY, permeability=PERMEABILITY):
        distance = ResistanceDistance()
        return make(kind, distance, quality, permeability, proximity=None)

    def connectivity(quality, permeability):
        return analysis(ConnectivityAnalysis, quality, permeability).run()

    sensitivity = analysis(SensitivityAnalysis)
    gradients = jax.jit(jax.grad(connectivity, (0, 1)))(QUALITY, PERMEABILITY)
    for var, expected in zip(["quality", "permeability"], gradients):
        actual = sensitivity.run(var)
        assert jnp.all(jnp.isfinite(actual))
        assert_close(actual, expected, 1e-12)


def test_default_proximity_is_exponential_in_equivalent_pixels():
    alpha = D / 4

    def kernel(distance):
        return make(ConnectivityAnalysis, distance, proximity=None).proximity

    # LCP distances at permeability 1 are whole pixels, past the window edge
    # too.
    distances = jnp.linspace(0, 2 * D, 61)
    assert_close(kernel(LCPDistance())(distances), jnp.exp(-distances / alpha), 1e-12)
    # A profile read off one randomized query is noisy: with this seed, the
    # randomized resistance drops from 0.80 two pixels from the centre of a
    # window of permeability 1 to 0.55 at its edge.
    randomized = ResistanceDistance(method=SpielmanApproximation(0.3, seed=3))
    assert randomized.exact() == ResistanceDistance()
    expected = kernel(ResistanceDistance())(distances / 4)
    assert_close(kernel(randomized)(distances / 4), expected, 1e-8)
    # Three habitat pixels 1, 2 and 3 pixels apart along a row, each at the
    # centre of its own window of permeability 1, where resistance grows like
    # the logarithm of the distance in pixels.
    quality = jnp.zeros((2 * D + 1, 2 * D + 4)).at[D, D + jnp.array([0, 1, 3])].set(1)
    connectivity = make(
        ConnectivityAnalysis,
        ResistanceDistance(),
        quality,
        jnp.ones_like(quality),
        proximity=None,
        batch_size=1,
    ).run()
    # Each pixel counts the others.
    expected = 2 * sum(np.exp(-r / alpha) for r in (1, 2, 3))
    np.testing.assert_allclose(connectivity, expected, rtol=1e-10)


def test_sensitivity_is_finite_for_proximity_singular_at_zero():
    def inverse(dist):
        return 1 / dist

    sensitivity = make(SensitivityAnalysis, LCPDistance(), proximity=inverse)
    for var in ["quality", "permeability"]:
        assert jnp.all(jnp.isfinite(sensitivity.run(var)))


def test_coarsened_permeability_sensitivity_ignores_permeability_out_of_reach():
    # Documented: the block correction is calibrated at the raster's maximum
    # permeability. Lowering permeability below the maximum in columns 35 and
    # beyond leaves alone the windows that reach columns 0 to 24, each of one
    # block centre.
    quality = smooth(1, (25, 50))
    permeability = jnp.ones_like(quality)
    lowered = permeability.at[:, 35:].set(0.2)

    def sensitivity(permeability):
        return make(
            SensitivityAnalysis,
            LCPDistance(),
            quality,
            permeability,
            batch_size=1,
            coarsening_factor=0.5,
        ).run("permeability")

    near = (slice(None), slice(0, 25))
    assert_close(sensitivity(lowered)[near], sensitivity(permeability)[near], 1e-12)


def test_coarsened_permeability_sensitivity_is_finite_where_proximities_underflow():
    # Habitat of permeability 1e-3 on two neighbouring blocks of 3 x 3 pixels
    # in an impassable matrix. Even at the maximum permeability, a neighbour is
    # hundreds of equivalent pixels away: the float32 proximities of the
    # default kernel underflow, and the block correction has no far-field
    # gradient to divide by.
    habitat = jnp.zeros((40, 40), "float32").at[12:15, 12:18].set(1)

    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            LCPDistance(),
            habitat,
            1e-3 * habitat,
            proximity=None,
            coarsening_factor=coarsening_factor,
        ).run("permeability")

    coarsened = sensitivity(0.5)
    assert jnp.all(jnp.isfinite(coarsened))
    np.testing.assert_array_equal(coarsened, sensitivity(0.0))


def test_coarsening_runs_on_a_raster_thinner_than_a_block():
    # Blocks of 3 x 3 pixels, on a raster of one row: every centre is in it.
    def coarsened(kind):
        return make(
            kind, LCPDistance(), QUALITY[:1], PERMEABILITY[:1], coarsening_factor=0.5
        )

    assert coarsened(ConnectivityAnalysis).run() > 0
    sensitivity = coarsened(SensitivityAnalysis).run("permeability")
    assert jnp.all(jnp.isfinite(sensitivity)) and jnp.any(sensitivity != 0)


def test_coarsened_quality_sensitivity_is_exact_inside_a_uniform_landscape():
    uniform = jnp.full((20, 20), 0.7)

    # Batches of one block centre compute it in its own window.
    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            LCPDistance(),
            uniform,
            uniform,
            batch_size=1,
            coarsening_factor=coarsening_factor,
        ).run("quality")

    # Blocks of 3 x 3 pixels centred on rows and columns 0, 1, 4, 7, ...: the
    # pixel windows and block windows that reach pixels 7 to 12 are all inside
    # the raster.
    interior = (slice(7, 13), slice(7, 13))
    assert_close(sensitivity(0.5)[interior], sensitivity(0.0)[interior], 1e-10)


def test_coarsened_resistance_sensitivity_is_exact_for_a_linear_proximity():
    # Documented: with block sources and a proximity linear in the distance,
    # each pixel counts its pairs in the windows of its block centres. Here
    # every window holds the raster, and one tile computes them all on the
    # graph of the padded raster, with blocks of 3 x 3 pixels.
    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            ResistanceDistance(solver=None),
            dependency_range=6,
            proximity=lambda dist: 1 - dist,
            batch_size=64,
            coarsening_factor=coarsening_factor,
        ).run("permeability")

    assert_close(sensitivity(0.5), sensitivity(0.0), 1e-10)


@pytest.mark.parametrize("var", ["quality", "permeability"])
def test_coarsening_approximates_pixel_windows_at_the_edges_of_a_uniform_landscape(
    var,
):
    # Measured largest errors, relative to the map maximum: 0.076 and 0.072
    # (0.37 and 0.30 with block centres inside the raster only, the nearest
    # taking all the interpolation weight).
    uniform = jnp.full((24, 25), 0.7)

    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            LCPDistance(),
            uniform,
            uniform,
            dependency_range=8,
            proximity=None,
            batch_size=1024,
            coarsening_factor=coarsening_factor,
        ).run(var)

    # Blocks of 3 x 3 pixels, centred on rows 0, 1, 4, ..., 22 and 23, and
    # columns 0, 1, 4, ..., 22 and 24; the pixels within 8 of an edge.
    edges = np.ones(uniform.shape, bool)
    edges[8:-8, 8:-8] = False
    assert_close(sensitivity(0.25)[edges], sensitivity(0.0)[edges], 0.08)


class Opaque(AbstractDistance):
    """A distance that the analyses cannot know to be symmetric."""

    distance: AbstractDistance

    def init(self, graph):
        return self.distance.init(graph)

    def nodes_to_nodes_distance(self, graph, nodes, state=None):
        return self.distance.nodes_to_nodes_distance(graph, nodes, state)

    def sources_to_targets_distance(self, graph, sources, targets, state=None):
        return self.distance.sources_to_targets_distance(graph, sources, targets, state)

    def all_pairs_distance(self, graph, state=None):
        return self.distance.all_pairs_distance(graph, state)


class Symmetric(Opaque):
    """A distance that declares itself symmetric."""

    symmetric: ClassVar[bool] = True


def test_coarsened_quality_sensitivity_reads_the_symmetry_of_the_distance():
    # Symmetric distances take twice the proximity-weighted quality seen from
    # each block centre, interpolated, and other distances the gradient of the
    # coarsened connectivity.
    def analysis(kind, distance, quality=QUALITY):
        return make(kind, distance, quality, coarsening_factor=0.5)

    def connectivity(quality):
        return analysis(ConnectivityAnalysis, Opaque(LCPDistance()), quality).run()

    def sensitivity(distance):
        return analysis(SensitivityAnalysis, distance).run("quality")

    expected = sensitivity(LCPDistance())
    np.testing.assert_array_equal(sensitivity(Symmetric(LCPDistance())), expected)
    gradient = sensitivity(Opaque(LCPDistance()))
    assert_close(gradient, jax.grad(connectivity)(QUALITY), 1e-12)
    assert not np.allclose(gradient, expected)


@pytest.mark.parametrize(
    "name, min_permeability_pearson",
    [("lcp", 0.8), pytest.param("resistance-dense", 0.95, marks=pytest.mark.slow)],
)
def test_coarsening_approximates_pixel_windows_on_a_smooth_landscape(
    name, min_permeability_pearson, request
):
    distance = CASES[name][0](request)
    quality, permeability = smooth(1, (64, 64)), smooth(2, (64, 64))

    def run(kind, coarsening_factor, *args):
        return kind(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=distance,
            dependency_range=8,
            batch_size=64,
            coarsening_factor=coarsening_factor,
        ).run(*args)

    # Blocks of 5 x 5 pixels: 25 times fewer windows.
    connectivity = run(ConnectivityAnalysis, 0.5)
    np.testing.assert_allclose(connectivity, run(ConnectivityAnalysis, 0.0), rtol=0.05)

    d_quality = run(SensitivityAnalysis, 0.5, "quality")
    expected = run(SensitivityAnalysis, 0.0, "quality")
    assert pearson(d_quality, expected) >= 0.98
    np.testing.assert_allclose(jnp.sum(d_quality), jnp.sum(expected), rtol=0.05)

    d_permeability = run(SensitivityAnalysis, 0.5, "permeability")
    expected = run(SensitivityAnalysis, 0.0, "permeability")
    assert pearson(d_permeability, expected) >= min_permeability_pearson
