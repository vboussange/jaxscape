import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxscape import (
    ConnectivityAnalysis,
    EuclideanDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    SensitivityAnalysis,
)
from jaxscape.solvers import AMJaxCGSolver, CholmodSolver, CuDSSSolver


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


@pytest.fixture
def compilations():
    """Count the programs that XLA compiles."""
    events = []

    def listener(event, duration_secs, **kwargs):
        del duration_secs, kwargs
        if event == "/jax/core/compile/backend_compile_duration":
            events.append(event)

    jax.monitoring.register_event_duration_secs_listener(listener)
    yield events
    jax.monitoring.unregister_event_duration_listener(listener)


@pytest.mark.parametrize("name", ["lcp", "resistance-cholmod"])
@pytest.mark.parametrize("kind", [ConnectivityAnalysis, SensitivityAnalysis])
def test_all_batches_run_one_compiled_program(name, kind, compilations, request):
    distance = CASES[name][0](request)
    args = ("permeability",) if kind is SensitivityAnalysis else ()
    quality, permeability = QUALITY[:5], PERMEABILITY[:5]
    # One batch compiles everything that does not depend on the batch count.
    analysis(kind, distance, quality, permeability, batch_size=35).run(*args)

    compilations.clear()
    analysis(kind, distance, quality, permeability, batch_size=3).run(*args)
    assert len(compilations) == 1
