import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.signal import convolve2d
from jaxscape import (
    ConnectivityAnalysis,
    EuclideanDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    SensitivityAnalysis,
    SpielmanApproximation,
)
from jaxscape.distance import AbstractDistance
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver, CuDSSSolver, cudsssolver


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
    # 42 windows: five full batches and a partial one.
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


def own_windows(quality, permeability, distance, q_weighted=True):
    """Connectivity summed over each pixel's own window, one pixel at a time.

    Each window is centred on its pixel and padded beyond the raster with quality
    0 and the permeability floor, and the distances come from the all-pairs
    query of `distance`.
    """
    size = 2 * D + 1
    centre = D * size + D
    permeability = jnp.where(permeability > FLOOR, permeability, FLOOR)
    weight = quality if q_weighted else jnp.ones_like(quality)
    weight = jnp.pad(weight, D)
    permeability = jnp.pad(permeability, D, constant_values=FLOOR)

    def window(i, j):
        w = jax.lax.dynamic_slice(weight, (i, j), (size, size)).ravel()
        p = jax.lax.dynamic_slice(permeability, (i, j), (size, size))
        grid = GridGraph(p, fun=average)
        K = distance.debiased_proximity(proximity, grid)(distance(grid)[centre])
        return w[centre] * (K.at[centre].set(0) @ w)

    rows, columns = np.indices(quality.shape).reshape(2, -1)
    return jnp.sum(jax.vmap(window)(rows, columns))


def make_cudss():
    try:
        return CuDSSSolver()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")


def solver(name, request):
    if name == "cholmod":
        pytest.importorskip("cholespy")
        return CholmodSolver()
    if name == "amjax":
        pytest.importorskip("amjax")
        return AMJaxCGSolver(rtol=1e-10, atol=1e-10, max_steps=2000)
    if name == "cudss-on-cpu":
        request.getfixturevalue("stub_spineax")
        return CuDSSSolver()
    return make_cudss()


# A randomized resistance: its all-pairs query on the same graph is the
# reference.
SPIELMAN = SpielmanApproximation(epsilon=0.3, seed=1)
# Each distance, its dense all-pairs reference, and the tolerance it reaches.
CASES = {
    "spielman": (
        lambda _: ResistanceDistance(method=SPIELMAN),
        ResistanceDistance(method=SPIELMAN),
        1e-10,
    ),
    "spielman-cholmod": (
        lambda request: ResistanceDistance(solver("cholmod", request), method=SPIELMAN),
        ResistanceDistance(method=SPIELMAN),
        1e-8,
    ),
    "euclidean": (lambda _: EuclideanDistance(), EuclideanDistance(), 1e-12),
    "lcp": (lambda _: LCPDistance(), LCPDistance(), 1e-12),
    "resistance-dense": (lambda _: ResistanceDistance(), ResistanceDistance(), 1e-10),
    **{
        f"resistance-{name}": (
            lambda request, name=name: ResistanceDistance(solver(name, request)),
            ResistanceDistance(),
            1e-8,
        )
        for name in ("cholmod", "amjax", "cudss-on-cpu", "cudss")
    },
}


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
    # whose sources do not share work do in batches of any size.
    batch_size = 1 if distance.sources_share_work else 8

    connectivity = make(ConnectivityAnalysis, distance, batch_size=batch_size).run()
    assert_close(connectivity, own_windows(QUALITY, PERMEABILITY, reference), rtol)

    sensitivity = make(SensitivityAnalysis, distance, batch_size=batch_size)
    for argnums, var in enumerate(["quality", "permeability"]):
        expected = jax.grad(own_windows, argnums)(QUALITY, PERMEABILITY, reference)
        assert_close(sensitivity.run(var), expected, rtol)


def padded_raster(quality, permeability, distance):
    """Connectivity with the distances on the graph of the padded raster.

    The raster is padded by `D` on all sides with quality 0 and the
    permeability floor, and each pixel counts the pixels of its own window.
    """
    height, width = quality.shape
    size = 2 * D + 1
    permeability = jnp.where(permeability > FLOOR, permeability, FLOOR)
    permeability = jnp.pad(permeability, D, constant_values=FLOOR)
    weight = jnp.pad(quality, D)
    grid = GridGraph(permeability, fun=average)
    K = distance.debiased_proximity(proximity, grid)
    distances = distance(grid).reshape(height + 2 * D, width + 2 * D, -1)

    def pixel(i, j):
        row = distances[i + D, j + D].reshape(height + 2 * D, width + 2 * D)
        d = jax.lax.dynamic_slice(row, (i, j), (size, size)).ravel()
        w = jax.lax.dynamic_slice(weight, (i, j), (size, size)).ravel()
        return w[D * size + D] * (K(d).at[D * size + D].set(0) @ w)

    rows, columns = np.indices(quality.shape).reshape(2, -1)
    return jnp.sum(jax.vmap(pixel)(rows, columns))


@pytest.mark.parametrize(
    "name", ["resistance-dense", "resistance-cholmod", "spielman", "spielman-cholmod"]
)
def test_one_tile_computes_distances_on_the_padded_raster(name, request):
    # Resistance queries share their work between sources: a batch of them
    # takes the distances of a tile of 5 x 7 pixels, the whole raster.
    distance = CASES[name][0](request)
    assert distance.sources_share_work
    quality, permeability = QUALITY[:5], PERMEABILITY[:5]

    def run(kind, *args):
        return make(kind, distance, quality, permeability, batch_size=35).run(*args)

    reference = CASES[name][1]
    assert_close(
        run(ConnectivityAnalysis),
        padded_raster(quality, permeability, reference),
        CASES[name][2],
    )
    for argnums, var in enumerate(["quality", "permeability"]):
        expected = jax.grad(padded_raster, argnums)(quality, permeability, reference)
        assert_close(run(SensitivityAnalysis, var), expected, CASES[name][2])


def test_tiles_approximate_each_pixels_own_window():
    # Tiles of 8 x 8 pixels: the distances of a pixel are those of a larger
    # graph than its window, and slightly lower. Measured: connectivity +6.7 %,
    # Pearson 0.9999 and 0.9995 for the quality and permeability sensitivity.
    quality, permeability = smooth(1, (32, 32)), smooth(2, (32, 32))

    def run(batch_size):
        def analysis(kind):
            return kind(
                quality_raster=quality,
                permeability_raster=permeability,
                distance=ResistanceDistance(),
                dependency_range=5,
                batch_size=batch_size,
            )

        sensitivity = analysis(SensitivityAnalysis)
        return (
            analysis(ConnectivityAnalysis).run(),
            sensitivity.run("quality"),
            sensitivity.run("permeability"),
        )

    tiles, windows = run(64), run(1)
    np.testing.assert_allclose(tiles[0], windows[0], rtol=0.1)
    for tile, window in zip(tiles[1:], windows[1:]):
        assert pearson(tile, window) >= 0.999


def test_float32_tile_sensitivity_matches_float64():
    # Tiles of 2 x 4 pixels and a buffer of 30 floored pixels around them:
    # resistance queries ground a pixel of near-maximum permeability next to
    # the sources, away from the floored pixels.
    def sensitivity(dtype):
        return make(
            SensitivityAnalysis,
            ResistanceDistance(),
            QUALITY.astype(dtype),
            PERMEABILITY.astype(dtype),
            dependency_range=30,
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


@pytest.mark.parametrize(
    "name, kernel",
    [
        ("lcp", proximity),
        ("resistance-cholmod", proximity),
        ("resistance-cholmod", None),
    ],
    ids=["lcp", "resistance", "resistance-default-proximity"],
)
def test_sensitivity_is_the_gradient_of_connectivity(name, kernel, request):
    distance = CASES[name][0](request)

    def connectivity(quality, permeability):
        return make(
            ConnectivityAnalysis, distance, quality, permeability, proximity=kernel
        ).run()

    sensitivity = make(SensitivityAnalysis, distance, proximity=kernel)
    for argnums, var in enumerate(["quality", "permeability"]):
        actual = sensitivity.run(var)
        assert jnp.all(jnp.isfinite(actual))
        expected = jax.grad(connectivity, argnums)(QUALITY, PERMEABILITY)
        assert_close(actual, expected, 1e-12)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("kind", [ConnectivityAnalysis, SensitivityAnalysis])
def test_default_lcp_proximity_is_exponential_in_pixels(kind, uniform):
    def exponential(dist):
        return jnp.exp(-dist / (D / 4))

    # At permeability 1, LCP distances are whole pixels, on the edge of the
    # window too.
    permeability = jnp.ones_like(QUALITY) if uniform else PERMEABILITY
    args = ("permeability",) if kind is SensitivityAnalysis else ()

    def run(proximity):
        return make(
            kind, LCPDistance(), permeability=permeability, proximity=proximity
        ).run(*args)

    assert_close(run(None), run(exponential), 1e-12)


@pytest.mark.parametrize("r", [1, 2, D])
def test_default_resistance_proximity_is_exponential_in_pixels(r):
    # Two habitat pixels r apart along a row, each at the centre of a window of
    # permeability 1, where resistance grows like the logarithm of r. Batches
    # of one source compute each pixel in its own window.
    quality = jnp.zeros((2 * D + 1, 2 * D + 1 + r)).at[D, jnp.array([D, D + r])].set(1)
    permeability = jnp.ones_like(quality)
    connectivity = make(
        ConnectivityAnalysis,
        ResistanceDistance(),
        quality,
        permeability,
        proximity=None,
        batch_size=1,
    ).run()
    # Each pixel counts the other.
    np.testing.assert_allclose(connectivity, 2 * np.exp(-r / (D / 4)), rtol=1e-10)


@pytest.mark.parametrize("name", ["spielman", "spielman-cholmod"])
def test_randomized_distances_take_the_default_proximity_of_exact_ones(name, request):
    # A profile read off one randomized query is noisy: with this seed, the
    # randomized resistance drops from 0.80 two pixels from the centre of a
    # window of permeability 1 to 0.55 at its edge.
    method = SpielmanApproximation(epsilon=0.3, seed=3)
    randomized = CASES[name][0](request)
    randomized = eqx.tree_at(lambda d: d.method, randomized, method)
    assert randomized.exact() == ResistanceDistance(randomized.solver)

    def kernel(distance):
        return make(ConnectivityAnalysis, distance, proximity=None).proximity

    distances = jnp.linspace(0, 3, 301)
    expected = kernel(ResistanceDistance())(distances)
    assert_close(kernel(randomized)(distances), expected, 1e-8)


@pytest.mark.gpu
def test_cudss_windows_carry_one_analysis():
    distance = ResistanceDistance(solver=make_cudss())
    spineax = cudsssolver._load_cudss()

    rebuilds = spineax.rebuild_count()
    jax.block_until_ready(make(ConnectivityAnalysis, distance).run())
    jax.block_until_ready(make(SensitivityAnalysis, distance).run("permeability"))
    # With coarsening, the windows carry on from the state of the block
    # correction's calibration.
    coarsened = make(SensitivityAnalysis, distance, coarsening_factor=0.5)
    jax.block_until_ready(coarsened.run("permeability"))
    # Each window refactorizes the state of the previous one: no handle is
    # used up twice, so no analysis repeats.
    assert spineax.rebuild_count() == rebuilds


def test_sensitivity_is_finite_for_proximity_singular_at_zero():
    def inverse(dist):
        return 1 / dist

    sensitivity = make(SensitivityAnalysis, LCPDistance(), proximity=inverse)
    for var in ["quality", "permeability"]:
        assert jnp.all(jnp.isfinite(sensitivity.run(var)))


@pytest.mark.parametrize(
    "name", ["lcp", "resistance-cholmod", "resistance-cudss-on-cpu"]
)
def test_coarsened_analyses_cover_the_raster(name, request):
    distance = CASES[name][0](request)

    # Blocks of 3 x 3 pixels, centred on rows 0, 1, 4 and 5, columns 0, 1, 4, 6.
    def coarsened(kind):
        return make(kind, distance, coarsening_factor=0.5)

    connectivity = coarsened(ConnectivityAnalysis).run()
    assert jnp.isfinite(connectivity) and connectivity > 0
    for var in ["quality", "permeability"]:
        sensitivity = coarsened(SensitivityAnalysis).run(var)
        assert sensitivity.shape == QUALITY.shape
        assert jnp.all(jnp.isfinite(sensitivity)) and jnp.any(sensitivity != 0)


@pytest.mark.parametrize("name", ["lcp", "resistance-dense"])
def test_coarsened_permeability_sensitivity_ignores_permeability_out_of_reach(
    name, request
):
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
            CASES[name][0](request),
            quality,
            permeability,
            batch_size=1,
            coarsening_factor=0.5,
        ).run("permeability")

    near = (slice(None), slice(0, 25))
    assert_close(sensitivity(lowered)[near], sensitivity(permeability)[near], 1e-12)


@pytest.mark.parametrize("name", ["lcp", "resistance-dense"])
def test_coarsened_permeability_sensitivity_is_finite_where_proximities_underflow(
    name, request
):
    # Habitat of permeability 1e-3 on two neighbouring blocks of 3 x 3 pixels
    # in an impassable matrix. Even at the maximum permeability, a neighbour is
    # hundreds of equivalent pixels away: the float32 proximities of the
    # default kernel underflow, and the block correction has no far-field
    # gradient to divide by.
    habitat = jnp.zeros((40, 40), "float32").at[12:15, 12:18].set(1)

    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            CASES[name][0](request),
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


@pytest.mark.parametrize("name", ["lcp", "resistance-dense"])
def test_coarsened_quality_sensitivity_is_exact_inside_a_uniform_landscape(
    name, request
):
    distance = CASES[name][0](request)
    uniform = jnp.full((20, 20), 0.7)

    # Batches of one block centre compute it in its own window.
    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            distance,
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


@pytest.mark.parametrize(
    "name, var, tolerance",
    [
        ("lcp", "quality", 0.08),
        ("lcp", "permeability", 0.08),
        ("resistance-dense", "quality", 0.1),
        ("resistance-dense", "permeability", 0.2),
    ],
)
def test_coarsening_approximates_pixel_windows_at_the_edges_of_a_uniform_landscape(
    name, var, tolerance, request
):
    # Measured largest errors, relative to the map maximum: 0.076, 0.072,
    # 0.098 and 0.192 (0.37, 0.30, 0.44 and 0.19 with block centres inside
    # the raster only, the nearest taking all the interpolation weight).
    distance = CASES[name][0](request)
    uniform = jnp.full((24, 25), 0.7)

    def sensitivity(coarsening_factor):
        return make(
            SensitivityAnalysis,
            distance,
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
    assert_close(sensitivity(0.25)[edges], sensitivity(0.0)[edges], tolerance)


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


def test_coarsened_quality_sensitivity_of_other_distances_is_a_gradient():
    distance = Opaque(LCPDistance())

    def connectivity(quality):
        return make(
            ConnectivityAnalysis, distance, quality, coarsening_factor=0.5
        ).run()

    sensitivity = make(SensitivityAnalysis, distance, coarsening_factor=0.5)
    expected = jax.grad(connectivity)(QUALITY)
    assert_close(sensitivity.run("quality"), expected, 1e-12)


def pearson(a, b):
    return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]


def smooth(seed, shape):
    """Gaussian-smoothed noise rescaled to [0.05, 1]."""
    x = np.arange(-12, 13)
    kernel = np.exp(-(x[:, None] ** 2 + x[None, :] ** 2) / 32)
    noise = np.random.default_rng(seed).normal(size=shape)
    field = np.asarray(convolve2d(noise, kernel / kernel.sum(), mode="same"))
    return jnp.asarray(0.05 + 0.95 * (field - field.min()) / np.ptp(field))


@pytest.mark.parametrize(
    "name, min_permeability_pearson", [("lcp", 0.8), ("resistance-dense", 0.95)]
)
def test_coarsening_approximates_pixel_windows_on_a_smooth_landscape(
    name, min_permeability_pearson, request
):
    distance = CASES[name][0](request)
    quality, permeability = smooth(1, (64, 64)), smooth(2, (64, 64))

    def run(kind, coarsening_factor, *args, q_weighted=True):
        return kind(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=distance,
            dependency_range=8,
            batch_size=512,
            coarsening_factor=coarsening_factor,
        ).run(*args, q_weighted=q_weighted)

    # Blocks of 5 x 5 pixels: 25 times fewer windows.
    for q_weighted in [True, False]:
        connectivity = run(ConnectivityAnalysis, 0.5, q_weighted=q_weighted)
        expected = run(ConnectivityAnalysis, 0.0, q_weighted=q_weighted)
        np.testing.assert_allclose(connectivity, expected, rtol=0.05)

    d_quality = run(SensitivityAnalysis, 0.5, "quality")
    expected = run(SensitivityAnalysis, 0.0, "quality")
    assert pearson(d_quality, expected) >= 0.98
    np.testing.assert_allclose(jnp.sum(d_quality), jnp.sum(expected), rtol=0.05)

    d_permeability = run(SensitivityAnalysis, 0.5, "permeability")
    expected = run(SensitivityAnalysis, 0.0, "permeability")
    assert pearson(d_permeability, expected) >= min_permeability_pearson


@pytest.fixture
def compilations():
    """Name the programs that XLA compiles."""
    events = []

    def listener(event, duration_secs, **kwargs):
        del duration_secs
        if event == "/jax/core/compile/backend_compile_duration":
            events.append(kwargs.get("fun_name"))

    jax.monitoring.register_event_duration_secs_listener(listener)
    yield events
    jax.monitoring.unregister_event_duration_listener(listener)


@pytest.mark.parametrize("name", ["lcp", "resistance-cholmod"])
@pytest.mark.parametrize(
    "kind, args",
    [
        (ConnectivityAnalysis, ()),
        (SensitivityAnalysis, ("quality",)),
        (SensitivityAnalysis, ("permeability",)),
    ],
)
@pytest.mark.parametrize("coarsening_factor", [0.0, 0.5])
def test_all_batches_and_runs_share_one_compiled_program(
    name, kind, args, coarsening_factor, compilations, request
):
    distance = CASES[name][0](request)
    # Batches of three windows, or of one tile of three pixels for resistance.
    windowed = make(
        kind,
        distance,
        QUALITY[:5],
        PERMEABILITY[:5],
        batch_size=3,
        coarsening_factor=coarsening_factor,
    )
    compilations.clear()
    windowed.run(*args)
    assert compilations.count("jit(_batch)") == 1

    # The analysis keeps its solver state: another run compiles nothing.
    compilations.clear()
    windowed.run(*args)
    assert compilations == []


@pytest.mark.parametrize("kind", [ConnectivityAnalysis, SensitivityAnalysis])
def test_default_proximity_compiles_once_for_all_analyses(kind, compilations):
    args = ("permeability",) if kind is SensitivityAnalysis else ()
    make(kind, LCPDistance(), proximity=None).run(*args)
    other = make(kind, LCPDistance(), 2 * QUALITY, 3 * PERMEABILITY, proximity=None)

    compilations.clear()
    other.run(*args)
    assert len(compilations) == 0
