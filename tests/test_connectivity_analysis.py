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


def analysis(kind, distance, quality=QUALITY, permeability=PERMEABILITY, **kwargs):
    # 42 windows: five full batches and a partial one.
    kwargs = {"proximity": proximity, "batch_size": 8, **kwargs}
    return kind(
        quality_raster=quality,
        permeability_raster=permeability,
        distance=distance,
        dependency_range=D,
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
        K = proximity(distance(GridGraph(p, fun=average))[centre])
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


# Each distance, its dense all-pairs reference, and the tolerance it reaches.
CASES = {
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

    connectivity = analysis(ConnectivityAnalysis, distance).run()
    assert_close(connectivity, own_windows(QUALITY, PERMEABILITY, reference), rtol)

    sensitivity = analysis(SensitivityAnalysis, distance)
    for argnums, var in enumerate(["quality", "permeability"]):
        expected = jax.grad(own_windows, argnums)(QUALITY, PERMEABILITY, reference)
        assert_close(sensitivity.run(var), expected, rtol)


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
        return analysis(
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
        return analysis(
            ConnectivityAnalysis, distance, quality, permeability, proximity=kernel
        ).run()

    sensitivity = analysis(SensitivityAnalysis, distance, proximity=kernel)
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
        return analysis(
            kind, LCPDistance(), permeability=permeability, proximity=proximity
        ).run(*args)

    assert_close(run(None), run(exponential), 1e-12)


@pytest.mark.parametrize("r", [1, 2, D])
def test_default_resistance_proximity_is_exponential_in_pixels(r):
    # Two habitat pixels r apart along a row, each at the centre of a window of
    # permeability 1, where resistance grows like the logarithm of r.
    quality = jnp.zeros((2 * D + 1, 2 * D + 1 + r)).at[D, jnp.array([D, D + r])].set(1)
    permeability = jnp.ones_like(quality)
    connectivity = analysis(
        ConnectivityAnalysis,
        ResistanceDistance(),
        quality,
        permeability,
        proximity=None,
    ).run()
    # Each pixel counts the other.
    np.testing.assert_allclose(connectivity, 2 * np.exp(-r / (D / 4)), rtol=1e-10)


def test_default_proximity_decreases_for_randomized_distances():
    # With this seed, the randomized resistance drops from 0.80 two pixels
    # from the centre of a window of permeability 1 to 0.55 at its edge.
    method = SpielmanApproximation(epsilon=0.3, seed=3)
    kernel = analysis(
        ConnectivityAnalysis, ResistanceDistance(method=method), proximity=None
    ).proximity

    distances = jnp.linspace(0, 3, 301)
    proximities = kernel(distances)
    assert jnp.all(jnp.diff(proximities) <= 0) and jnp.all(proximities <= 1)
    assert jnp.all(jnp.isfinite(jax.vmap(jax.grad(kernel))(distances)))


@pytest.mark.gpu
def test_cudss_windows_carry_one_analysis():
    distance = ResistanceDistance(solver=make_cudss())
    spineax = cudsssolver._load_cudss()

    rebuilds = spineax.rebuild_count()
    jax.block_until_ready(analysis(ConnectivityAnalysis, distance).run())
    jax.block_until_ready(analysis(SensitivityAnalysis, distance).run("permeability"))
    # Each window refactorizes the state of the previous one: no handle is
    # used up twice, so no analysis repeats.
    assert spineax.rebuild_count() == rebuilds


def test_sensitivity_is_finite_for_proximity_singular_at_zero():
    def inverse(dist):
        return 1 / dist

    sensitivity = analysis(SensitivityAnalysis, LCPDistance(), proximity=inverse)
    for var in ["quality", "permeability"]:
        assert jnp.all(jnp.isfinite(sensitivity.run(var)))


@pytest.mark.parametrize("name", ["lcp", "resistance-cholmod"])
def test_coarsened_analyses_cover_the_raster(name, request):
    distance = CASES[name][0](request)

    # Blocks of 3 x 3 pixels; the last column of blocks extends past the raster.
    def coarsened(kind):
        return analysis(kind, distance, coarsening_factor=0.5)

    connectivity = coarsened(ConnectivityAnalysis).run()
    assert jnp.isfinite(connectivity) and connectivity > 0
    for var in ["quality", "permeability"]:
        sensitivity = coarsened(SensitivityAnalysis).run(var)
        assert sensitivity.shape == QUALITY.shape
        assert jnp.all(jnp.isfinite(sensitivity)) and jnp.any(sensitivity != 0)


@pytest.mark.parametrize("name", ["lcp", "resistance-dense"])
def test_coarsened_permeability_sensitivity_is_finite_for_sparse_habitat(name, request):
    # Habitat on two neighbouring blocks of 3 x 3 pixels, centred on (13, 13)
    # and (13, 16), in an impassable matrix: at the raster's mean permeability,
    # the float32 proximities of the default kernel underflow.
    habitat = jnp.zeros((40, 40), "float32").at[12:15, 12:18].set(1)
    sensitivity = analysis(
        SensitivityAnalysis,
        CASES[name][0](request),
        habitat,
        habitat,
        proximity=None,
        coarsening_factor=0.5,
    ).run("permeability")
    assert jnp.all(jnp.isfinite(sensitivity)) and jnp.any(sensitivity != 0)


def test_coarsening_needs_a_block_centre_inside_the_raster():
    # Blocks of 3 x 3 pixels have their centre in the second row.
    with pytest.raises(ValueError, match="too small"):
        analysis(
            ConnectivityAnalysis,
            LCPDistance(),
            QUALITY[:1],
            PERMEABILITY[:1],
            coarsening_factor=0.5,
        )


@pytest.mark.parametrize("name", ["lcp", "resistance-dense"])
def test_coarsened_quality_sensitivity_is_exact_inside_a_uniform_landscape(
    name, request
):
    distance = CASES[name][0](request)
    uniform = jnp.full((20, 20), 0.7)

    def sensitivity(coarsening_factor):
        return analysis(
            SensitivityAnalysis,
            distance,
            uniform,
            uniform,
            coarsening_factor=coarsening_factor,
        ).run("quality")

    # Blocks of 3 x 3 pixels centred on rows and columns 1, 4, 7, ...: the
    # pixel windows and block windows that reach pixels 7 to 12 are all inside
    # the raster.
    interior = (slice(7, 13), slice(7, 13))
    assert_close(sensitivity(0.5)[interior], sensitivity(0.0)[interior], 1e-10)


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
        return analysis(
            ConnectivityAnalysis, distance, quality, coarsening_factor=0.5
        ).run()

    sensitivity = analysis(SensitivityAnalysis, distance, coarsening_factor=0.5)
    expected = jax.grad(connectivity)(QUALITY)
    assert_close(sensitivity.run("quality"), expected, 1e-12)


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

    def pearson(a, b):
        return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]

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
def test_all_batches_run_one_compiled_program(
    name, kind, args, coarsening_factor, compilations, request
):
    distance = CASES[name][0](request)
    quality, permeability = QUALITY[:5], PERMEABILITY[:5]

    def run(batch_size):
        analysis(
            kind,
            distance,
            quality,
            permeability,
            batch_size=batch_size,
            coarsening_factor=coarsening_factor,
        ).run(*args)

    # One batch compiles everything that does not depend on the batch count.
    run(35)
    compilations.clear()
    run(3)
    # Each new Cholmod state also recompiles the permeability correction.
    correction = "jit(_permeability_correction)"
    assert [name for name in compilations if name != correction] == ["jit(_batch)"]


@pytest.mark.parametrize("kind", [ConnectivityAnalysis, SensitivityAnalysis])
def test_default_proximity_compiles_once_for_all_analyses(kind, compilations):
    args = ("permeability",) if kind is SensitivityAnalysis else ()
    analysis(kind, LCPDistance(), proximity=None).run(*args)
    other = analysis(kind, LCPDistance(), 2 * QUALITY, 3 * PERMEABILITY, proximity=None)

    compilations.clear()
    other.run(*args)
    assert len(compilations) == 0
