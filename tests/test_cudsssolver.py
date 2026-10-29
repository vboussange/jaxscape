"""`CuDSSSolver` at its public boundaries.

Numerical tests run twice: against spineax and cuDSS on an NVIDIA GPU, which
skip explicitly elsewhere, and against a CPU double of spineax's phase API
(`fake_cudss.py`), so that the solver's plumbing also runs in CI.
"""

import importlib.machinery
import importlib.util
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fake_cudss import FAKE_CUDSS
from jax.experimental.sparse import BCOO
from jaxscape import Graph, GridGraph, ResistanceDistance, SpielmanApproximation
from jaxscape.resistance_distance import p_inv_resistance_distance
from jaxscape.solvers import (
    BCOOLinearOperator,
    CholmodSolver,
    CuDSSSolver,
    cudsssolver,
)
from jaxscape.solvers.operator import _batched_linear_solve, _linear_solve
from packaging.requirements import Requirement


@pytest.fixture(params=["cpu-double", "spineax-gpu"])
def cudss(request, monkeypatch):
    """The spineax phase API that `CuDSSSolver` runs on."""
    return use_cudss(request.param, monkeypatch)


@pytest.fixture(
    params=["cpu-double-nan-pivots", "cpu-double-perturbed-pivots", "spineax-gpu"]
)
def failing_cudss(request, monkeypatch):
    """The spineax phase API, for factorizations that fail.

    spineax reports no factorization status. A failed cuDSS factorization
    leaves NaN or, with perturbed pivots, finite factors; the CPU double
    reproduces both.
    """
    if request.param == "spineax-gpu":
        return use_cudss(request.param, monkeypatch)
    cudss = use_cudss("cpu-double", monkeypatch)
    cudss.failed_pivots = request.param.removeprefix("cpu-double-").split("-")[0]
    return cudss


def use_cudss(kind, monkeypatch):
    if kind == "spineax-gpu":
        try:
            return cudsssolver._load_cudss()
        except (ImportError, RuntimeError) as error:
            pytest.skip(
                f"GPU numerical tests need an NVIDIA GPU, spineax and cuDSS: {error}"
            )
    FAKE_CUDSS.reset()
    monkeypatch.setattr(cudsssolver, "_load_cudss", lambda: FAKE_CUDSS)
    return FAKE_CUDSS


def test_cudss_extra_pins_spineax_and_cudss_below_1():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    project = tomllib.loads(pyproject.read_text())["project"]
    extra = {
        requirement.name: requirement
        for requirement in map(Requirement, project["optional-dependencies"]["cudss"])
    }
    spineax, cudss = extra["spineax"], extra["nvidia-cudss-cu13"]

    assert spineax.specifier.contains("0.0.5")
    assert cudss.specifier.contains("0.8.0.10")
    assert not cudss.specifier.contains("1.0.0.35")
    # spineax publishes a single wheel: Linux x86-64, Python 3.12 or newer.
    for requirement in (spineax, cudss):
        assert requirement.marker is not None
        linux = {"sys_platform": "linux", "platform_machine": "x86_64"}
        assert requirement.marker.evaluate({**linux, "python_version": "3.12"})
        assert not requirement.marker.evaluate({**linux, "python_version": "3.11"})
        assert not requirement.marker.evaluate(
            {
                "sys_platform": "darwin",
                "platform_machine": "arm64",
                "python_version": "3.13",
            }
        )
    base = " ".join(project["dependencies"]).lower()
    assert "cuda" not in base and "nvidia" not in base and "spineax" not in base


def test_imports_and_defaults_are_unchanged_without_spineax():
    code = """
import sys
sys.modules["spineax"] = None  # an environment without the cudss extra
import jaxscape
from jaxscape import ResistanceDistance
from jaxscape.solvers import CuDSSSolver
assert ResistanceDistance().solver is None
try:
    CuDSSSolver()
except ImportError as error:
    print(error)
else:
    raise AssertionError("CuDSSSolver() must report the missing extra")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "spineax" in result.stdout
    assert "jaxscape[cudss]" in result.stdout


def installed_spineax_without_runtime(monkeypatch):
    """spineax is installed, but its cuDSS module cannot load."""
    spineax = types.ModuleType("spineax")
    spineax.__spec__ = importlib.machinery.ModuleSpec("spineax", None)
    monkeypatch.setitem(sys.modules, "spineax", spineax)
    monkeypatch.setitem(sys.modules, "spineax.cudss", None)


def jax_backend(monkeypatch, backend, platform_version):
    devices = [
        types.SimpleNamespace(
            platform=backend,
            client=types.SimpleNamespace(platform_version=platform_version),
        )
    ]
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    monkeypatch.setattr(jax, "devices", lambda *args, **kwargs: devices)


def test_missing_spineax_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "spineax", None)
    with pytest.raises(ImportError, match=r"spineax.*jaxscape\[cudss\]"):
        CuDSSSolver()


@pytest.mark.parametrize(
    "backend, platform_version", [("cpu", "cpu"), ("gpu", "PJRT C API\nrocm 60300")]
)
def test_backends_other_than_nvidia_gpus_are_rejected(
    monkeypatch, backend, platform_version
):
    installed_spineax_without_runtime(monkeypatch)
    jax_backend(monkeypatch, backend, platform_version)
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        CuDSSSolver()


def test_missing_cudss_runtime_is_reported(monkeypatch):
    installed_spineax_without_runtime(monkeypatch)
    jax_backend(monkeypatch, "gpu", "PJRT C API\ncuda 13000")
    with pytest.raises(ImportError, match=r"cuDSS runtime.*nvidia-cudss-cu13"):
        CuDSSSolver()


def cudss_calls(cudss, run):
    """Count the numeric factorizations and solve calls that `run` makes.

    The CPU double records them as they run; with spineax, they are the cuDSS
    calls in the compiled program.
    """
    if cudss is FAKE_CUDSS:
        factorizations = cudss.factorizations
        cudss.solve_calls.clear()
        jax.block_until_ready(run())
        return cudss.factorizations - factorizations, len(cudss.solve_calls)
    program = jax.jit(run).lower().as_text()
    numeric = program.count("spineax_token_factorize") + program.count(
        "spineax_token_refactorize"
    )
    return numeric, program.count("spineax_token_solve")


# Relative tolerance of float64 distances against the dense pseudo-inverse,
# which is itself accurate to about 1e-10 on these surfaces.
FLOAT64_RTOL = 1e-8


def distance_rtol(dtype, surface):
    """Relative tolerance of distances against the float64 dense reference."""
    if dtype == jnp.float64:
        return FLOAT64_RTOL
    # A 1e-6 floor across a two-cell barrier costs float32 about three of its
    # seven digits; the other surfaces keep about five.
    return 1e-2 if surface == "barrier_with_floor" else 1e-4


def conductance_grid(conductance):
    return GridGraph(conductance, fun=lambda x, y: (x + y) / 2)


def conductance_surface(name, shape=(6, 7)):
    """Small conductance rasters, from smooth to six orders of contrast."""
    rng = np.random.default_rng(0)
    if name == "smooth":
        surface = np.linspace(0.5, 2.0, shape[0] * shape[1]).reshape(shape)
    elif name == "six_decades":
        surface = 10.0 ** rng.uniform(-6.0, 0.0, shape)
    elif name == "barrier_with_floor":
        # A two-cell-wide straight barrier of zero conductance, raised to a
        # 1e-6 floor, so that edges crossing it conduct 1e-6.
        surface = 10.0 ** rng.uniform(-2.0, 0.0, shape)
        surface[:, 3:5] = 0.0
        surface = np.maximum(surface, 1e-6)
    else:
        raise ValueError(name)
    return jnp.asarray(surface, dtype=jnp.float64)


QUERIES = {
    "all_pairs": {},
    "nodes_to_nodes": {"nodes": jnp.array([0, 9, 41, 20, 9], dtype=jnp.int32)},
    "sources_to_targets": {
        "sources": jnp.array([0, 17], dtype=jnp.int32),
        "targets": jnp.array([41, 3, 25], dtype=jnp.int32),
    },
}


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("surface", ["smooth", "six_decades", "barrier_with_floor"])
def test_exact_distances_match_dense_reference(cudss, surface, query, compiled, dtype):
    conductance = conductance_surface(surface)
    grid = conductance_grid(conductance.astype(dtype))
    distance = ResistanceDistance(solver=CuDSSSolver())
    state = distance.init(grid)

    def distances(grid, state):
        return distance(grid, state=state, **QUERIES[query])

    if compiled:
        distances = jax.jit(distances)
    actual = distances(grid, state)

    expected = ResistanceDistance()(conductance_grid(conductance), **QUERIES[query])
    assert actual.dtype == dtype
    rtol = distance_rtol(dtype, surface)
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol * float(np.max(expected))
    )


TINY_GRAPHS = {
    # A single edge of conductance 4: the grounded Laplacian is 1 x 1.
    "edge": ([(0, 1, 4.0)], [[0.0, 0.25], [0.25, 0.0]]),
    # Conductances 2 and 4 in series.
    "path": (
        [(0, 1, 2.0), (1, 2, 4.0)],
        [[0, 0.5, 0.75], [0.5, 0, 0.25], [0.75, 0.25, 0]],
    ),
    # A unit triangle: one edge in parallel with two in series.
    "triangle": (
        [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)],
        np.full((3, 3), 2 / 3) - np.diag(np.full(3, 2 / 3)),
    ),
    # A unit square with one diagonal of conductance 2. By symmetry, no
    # current crosses the diagonal between vertices 1 and 3.
    "square": (
        [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 2.0)],
        np.array(
            [
                [0, 7 / 12, 1 / 3, 7 / 12],
                [7 / 12, 0, 7 / 12, 1],
                [1 / 3, 7 / 12, 0, 7 / 12],
                [7 / 12, 1, 7 / 12, 0],
            ]
        ),
    ),
}


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("graph", TINY_GRAPHS)
def test_tiny_graphs_give_exact_distances(cudss, graph, dtype):
    edges, expected = TINY_GRAPHS[graph]
    n = len(expected)
    adjacency = np.zeros((n, n))
    for u, v, w in edges:
        adjacency[u, v] = adjacency[v, u] = w
    graph = Graph(BCOO.fromdense(jnp.asarray(adjacency, dtype=dtype)))
    distance = ResistanceDistance(solver=CuDSSSolver())

    actual = distance(graph, state=distance.init(graph))

    assert actual.dtype == dtype
    rtol = distance_rtol(dtype, "smooth")
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=rtol, atol=rtol)


def test_compiled_queries_need_a_state_from_initialization(cudss):
    grid = conductance_grid(conductance_surface("smooth"))
    distance = ResistanceDistance(solver=CuDSSSolver())

    # Eager calls initialize their own state from the concrete graph.
    np.testing.assert_allclose(
        distance(grid), ResistanceDistance()(grid), rtol=FLOAT64_RTOL
    )
    with pytest.raises(ValueError, match=r"outside `jax\.jit`.*distance\.init"):
        jax.jit(lambda grid: distance(grid))(grid)


@pytest.mark.parametrize("query", QUERIES)
def test_all_right_hand_sides_of_a_query_share_one_factorization(cudss, query):
    grid = conductance_grid(conductance_surface("six_decades"))
    distance = ResistanceDistance(solver=CuDSSSolver())
    state = distance.init(grid)

    calls = cudss_calls(cudss, lambda: distance(grid, state=state, **QUERIES[query]))

    assert calls == (1, 1)


def sparse_layout(dense):
    """Store a symmetric matrix as JAXScape's sparse code paths do: unsorted,
    with duplicate entries splitting values, and padding entries interspersed.
    """
    n = dense.shape[0]
    rows, cols = np.nonzero(np.asarray(dense))
    values = np.asarray(dense)[rows, cols]
    split = np.arange(rows.size) % 3 == 0
    rows = np.concatenate([rows, rows[split], [n, n, n]])
    cols = np.concatenate([cols, cols[split], [n, n, n]])
    values = np.concatenate([values, values[split], [0.0, 0.0, 0.0]])
    values[: split.size][split] /= 2
    values[split.size : split.size + split.sum()] /= 2
    order = np.random.default_rng(1).permutation(rows.size)
    indices = np.stack([rows[order], cols[order]], axis=1)
    return BCOO(
        (jnp.asarray(values[order]), jnp.asarray(indices, dtype=jnp.int32)),
        shape=(n, n),
    )


def test_unsorted_duplicate_and_padded_graphs_give_exact_distances(cudss):
    # A ring with chords, large enough for duplicates to survive in the
    # grounded Laplacian; one edge conducts 1e-6.
    n = 10
    weights = np.geomspace(1e-2, 1e2, n)
    weights[3] = 1e-6
    edges = [(i, (i + 1) % n, w) for i, w in enumerate(weights)]
    edges += [(0, 5, 0.7), (2, 7, 1.3), (4, 9, 2.1)]
    adjacency = np.zeros((n, n))
    for u, v, w in edges:
        adjacency[u, v] = adjacency[v, u] = w
    graph = Graph(sparse_layout(adjacency))
    distance = ResistanceDistance(solver=CuDSSSolver())

    actual = distance(graph, state=distance.init(graph))

    expected = p_inv_resistance_distance(BCOO.fromdense(jnp.asarray(adjacency)))
    np.testing.assert_allclose(actual, expected, rtol=FLOAT64_RTOL, atol=1e-12)


@pytest.mark.parametrize("n_rhs", [None, 1, 4])
def test_sparse_solve_of_single_and_multiple_right_hand_sides(cudss, n_rhs):
    n = 9
    matrix = (
        np.diag(np.linspace(2.5, 4.0, n))
        - np.diag(np.full(n - 1, 1.0), 1)
        - np.diag(np.full(n - 1, 1.0), -1)
    )
    A = sparse_layout(matrix)
    solver = CuDSSSolver()
    state = solver.init(BCOOLinearOperator(A), {})
    rng = np.random.default_rng(2)
    if n_rhs is None:
        b = jnp.asarray(rng.standard_normal(n))

        def solve():
            return _linear_solve(A, b, solver, state=state)
    else:
        b = jnp.asarray(rng.standard_normal((n, n_rhs)))

        def solve():
            return _batched_linear_solve(A, b, solver, state=state)

    x = solve()

    np.testing.assert_allclose(x, np.linalg.solve(matrix, b), rtol=1e-12)
    np.testing.assert_allclose(A @ x - b, 0.0, atol=1e-12)
    assert cudss_calls(cudss, solve) == (1, 1)


@pytest.mark.parametrize("n_rhs", [None, 1, 4])
def test_sparse_solve_gradients_match_dense_reference(cudss, n_rhs):
    """Gradients with respect to the stored matrix values and right-hand sides,
    for a single right-hand side and for vmapped batches of them."""
    n = 9
    matrix = (
        np.diag(np.linspace(2.5, 4.0, n))
        - np.diag(np.full(n - 1, 1.0), 1)
        - np.diag(np.full(n - 1, 1.0), -1)
    )
    A = sparse_layout(matrix)
    solver = CuDSSSolver()
    state = solver.init(BCOOLinearOperator(A), {})
    rng = np.random.default_rng(3)
    rhs_shape = (n,) if n_rhs is None else (n, n_rhs)
    b = jnp.asarray(rng.standard_normal(rhs_shape))
    weights = jnp.asarray(rng.standard_normal(rhs_shape))

    def sparse_solve(A, b):
        if n_rhs is None:
            return _linear_solve(A, b, solver, state=state)
        return _batched_linear_solve(A, b, solver, state=state)

    def dense_solve(A, b):
        return jnp.linalg.solve(A.todense(), b)

    def gradients(solve):
        def objective(data, b):
            x = solve(BCOO((data, A.indices), shape=A.shape), b)
            return jnp.sum(weights * x)

        return jax.grad(objective, argnums=(0, 1))(A.data, b)

    data_gradient, rhs_gradient = gradients(sparse_solve)

    expected_data_gradient, expected_rhs_gradient = gradients(dense_solve)
    np.testing.assert_allclose(
        data_gradient, expected_data_gradient, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(
        rhs_gradient, expected_rhs_gradient, rtol=1e-10, atol=1e-12
    )


def test_queries_allocate_no_dense_matrix(cudss):
    n = 900
    conductance = jnp.linspace(0.5, 2.0, n).reshape(30, 30)
    distance = ResistanceDistance(solver=CuDSSSolver())
    state = distance.init(conductance_grid(conductance))
    nodes = jnp.array([0, 100, n - 1], dtype=jnp.int32)

    def objective(conductance, state):
        return jnp.sum(
            distance(conductance_grid(conductance), nodes=nodes, state=state)
        )

    # XLA's temporary buffers of the compiled query and of its gradient stay
    # far below one dense float64 grounded Laplacian.
    dense_grounded_laplacian = (n - 1) ** 2 * 8
    for program in (jax.jit(objective), jax.jit(jax.grad(objective))):
        memory = program.lower(conductance, state).compile().memory_analysis()
        assert memory is not None
        assert memory.temp_size_in_bytes < dense_grounded_laplacian / 4


def test_inverse_objective_gradients_match_dense_reference(cudss):
    """Fit land-cover conductances to sampled-site distances."""
    land_cover = jnp.array(
        [
            [0, 0, 1, 1, 2, 2, 0],
            [0, 2, 2, 1, 2, 0, 0],
            [2, 2, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 2, 1, 1],
            [1, 1, 2, 0, 0, 2, 1],
            [0, 1, 2, 2, 0, 0, 1],
        ]
    )
    features = jax.nn.one_hot(land_cover, num_classes=3, dtype=jnp.float64)
    nodes = jnp.array([0, 6, 17, 24, 35, 41], dtype=jnp.int32)
    pairs = jnp.triu_indices(nodes.size, k=1)
    distance = ResistanceDistance(solver=CuDSSSolver())
    reference = ResistanceDistance()

    def landscape(log_conductance):
        return conductance_grid(jnp.exp(features @ log_conductance))

    target = reference(landscape(jnp.array([0.3, -0.4, 0.2])), nodes=nodes)[pairs]

    def objective(log_conductance, metric, state):
        predicted = metric(landscape(log_conductance), nodes=nodes, state=state)
        return jnp.mean((predicted[pairs] - target) ** 2)

    initial = jnp.array([-0.2, 0.25, -0.1])
    state = distance.init(landscape(initial))
    evaluate = jax.jit(jax.value_and_grad(lambda p, s: objective(p, distance, s)))
    reference_evaluate = jax.value_and_grad(lambda p: objective(p, reference, None))
    # The same state from initialization, while conductances move away from
    # it, up to six orders of magnitude of contrast between land covers.
    for log_conductance in (
        initial,
        jnp.array([0.15, -0.2, 0.1]),
        jnp.array([0.0, -6.9, -13.8]),
    ):
        loss, gradient = evaluate(log_conductance, state)
        expected_loss, expected_gradient = reference_evaluate(log_conductance)
        np.testing.assert_allclose(loss, expected_loss, rtol=FLOAT64_RTOL)
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=FLOAT64_RTOL, atol=1e-12
        )


def query_and_state(distance, query):
    """The state-returning form of `query`, called with `QUERIES[query]`."""
    method = getattr(distance, f"{query}_distance_and_state")
    return lambda graph, state: method(graph, state=state, **QUERIES[query])


def compiled_step(distance, query):
    """`query` on the grid of a conductance raster, compiled, with the state."""
    distances_and_state = query_and_state(distance, query)

    @jax.jit
    def step(conductance, state):
        return distances_and_state(conductance_grid(conductance), state)

    return step


def compiled_executables(function):
    """The number of entries in the compilation cache of a `jax.jit` function."""
    return function._cache_size()  # type: ignore[attr-defined]


@pytest.mark.parametrize("query", QUERIES)
def test_compiled_steps_carry_the_returned_state(cudss, query):
    distance = ResistanceDistance(solver=CuDSSSolver())
    state = distance.init(conductance_grid(conductance_surface("smooth")))
    step = compiled_step(distance, query)

    rebuilds = cudss.rebuild_count()
    # New conductances at every step, on one raster geometry.
    for scale, surface in enumerate(
        ["smooth", "six_decades", "barrier_with_floor", "smooth"], start=1
    ):
        conductance = scale * conductance_surface(surface)
        distances, state = step(conductance, state)

        expected = ResistanceDistance()(conductance_grid(conductance), **QUERIES[query])
        np.testing.assert_allclose(distances, expected, rtol=FLOAT64_RTOL, atol=1e-12)
    # One executable for every step, and no step repeats the analysis.
    assert compiled_executables(step) == 1
    assert cudss.rebuild_count() == rebuilds


@pytest.mark.parametrize("query", QUERIES)
def test_compiled_gradient_steps_carry_the_returned_state(cudss, query):
    """Fit land-cover conductances, passing the state from step to step."""
    land_cover = np.random.default_rng(3).integers(0, 3, (6, 7))
    features = jax.nn.one_hot(land_cover, num_classes=3, dtype=jnp.float64)

    def landscape(log_conductance):
        return conductance_grid(jnp.exp(features @ log_conductance))

    distance = ResistanceDistance(solver=CuDSSSolver())
    distances_and_state = query_and_state(distance, query)
    reference = query_and_state(ResistanceDistance(), query)
    target, _ = reference(landscape(jnp.array([0.3, -0.4, 0.2])), None)

    def objective(log_conductance, distances_and_state, state):
        predicted, state = distances_and_state(landscape(log_conductance), state)
        return jnp.mean((predicted - target) ** 2), state

    step = jax.jit(
        jax.value_and_grad(
            lambda p, state: objective(p, distances_and_state, state), has_aux=True
        )
    )
    reference_step = jax.value_and_grad(
        lambda p: objective(p, reference, None), has_aux=True
    )
    initial = jnp.array([-0.2, 0.25, -0.1])
    state = distance.init(landscape(initial))

    rebuilds = cudss.rebuild_count()
    # Up to six orders of magnitude of contrast between land covers.
    for log_conductance in (
        initial,
        jnp.array([0.15, -0.2, 0.1]),
        jnp.array([0.0, -6.9, -13.8]),
        jnp.array([-0.3, 0.4, -2.0]),
    ):
        (loss, state), gradient = step(log_conductance, state)

        (expected_loss, _), expected_gradient = reference_step(log_conductance)
        np.testing.assert_allclose(loss, expected_loss, rtol=FLOAT64_RTOL)
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=FLOAT64_RTOL, atol=1e-12
        )
    assert compiled_executables(step) == 1
    assert cudss.rebuild_count() == rebuilds


@pytest.mark.parametrize("query", QUERIES)
def test_an_outdated_state_still_gives_exact_distances(cudss, query):
    distance = ResistanceDistance(solver=CuDSSSolver())
    outdated = distance.init(conductance_grid(conductance_surface("smooth")))
    step = compiled_step(distance, query)
    # This step uses up the factorization of the initial state.
    step(conductance_surface("six_decades"), outdated)

    rebuilds = cudss.rebuild_count()
    for surface in ("barrier_with_floor", "six_decades"):
        conductance = conductance_surface(surface)
        distances, _ = step(conductance, outdated)

        expected = ResistanceDistance()(conductance_grid(conductance), **QUERIES[query])
        np.testing.assert_allclose(distances, expected, rtol=FLOAT64_RTOL, atol=1e-12)
    # spineax repeated the analysis for the outdated state.
    assert cudss.rebuild_count() > rebuilds


needs_cholmod = pytest.mark.skipif(
    importlib.util.find_spec("cholespy") is None,
    reason="The reference CholmodSolver needs cholespy.",
)
SPIELMAN = SpielmanApproximation(epsilon=0.3, seed=7)


def spielman_atol(expected):
    """Absolute tolerance of float64 Spielman distances.

    A distance is a difference of squared feature norms, so near-zero
    distances, such as those of a vertex to itself, carry rounding errors
    relative to the largest distance.
    """
    return 1e-13 * np.max(np.abs(expected))


@needs_cholmod
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("surface", ["smooth", "six_decades", "barrier_with_floor"])
def test_spielman_distances_match_cholmod(cudss, surface, query, compiled):
    grid = conductance_grid(conductance_surface(surface))
    distance = ResistanceDistance(solver=CuDSSSolver(), method=SPIELMAN)
    state = distance.init(grid)

    def distances(grid, state):
        return distance(grid, state=state, **QUERIES[query])

    if compiled:
        distances = jax.jit(distances)
    actual = distances(grid, state)

    reference = ResistanceDistance(solver=CholmodSolver(), method=SPIELMAN)
    expected = reference(grid, **QUERIES[query])
    assert actual.dtype == jnp.float64
    np.testing.assert_allclose(
        actual, expected, rtol=FLOAT64_RTOL, atol=spielman_atol(expected)
    )


@needs_cholmod
@pytest.mark.parametrize("query", QUERIES)
def test_spielman_gradients_match_cholmod(cudss, query):
    """Fit land-cover conductances to approximate sampled-site distances."""
    land_cover = np.random.default_rng(4).integers(0, 3, (6, 7))
    features = jax.nn.one_hot(land_cover, num_classes=3, dtype=jnp.float64)

    def landscape(log_conductance):
        return conductance_grid(jnp.exp(features @ log_conductance))

    distance = ResistanceDistance(solver=CuDSSSolver(), method=SPIELMAN)
    reference = ResistanceDistance(solver=CholmodSolver(), method=SPIELMAN)
    distances_and_state = query_and_state(distance, query)
    reference_distances = query_and_state(reference, query)
    target, _ = reference_distances(landscape(jnp.array([0.3, -0.4, 0.2])), None)

    def objective(log_conductance, distances_and_state, state):
        predicted, state = distances_and_state(landscape(log_conductance), state)
        return jnp.mean((predicted - target) ** 2), state

    step = jax.jit(
        jax.value_and_grad(
            lambda p, state: objective(p, distances_and_state, state), has_aux=True
        )
    )
    reference_step = jax.value_and_grad(
        lambda p: objective(p, reference_distances, None), has_aux=True
    )
    initial = jnp.array([-0.2, 0.25, -0.1])
    state = distance.init(landscape(initial))

    # Up to six orders of magnitude of contrast between land covers.
    for log_conductance in (
        initial,
        jnp.array([0.15, -0.2, 0.1]),
        jnp.array([0.0, -6.9, -13.8]),
    ):
        (loss, state), gradient = step(log_conductance, state)

        (expected_loss, _), expected_gradient = reference_step(log_conductance)
        np.testing.assert_allclose(loss, expected_loss, rtol=FLOAT64_RTOL)
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=FLOAT64_RTOL, atol=1e-12
        )


@pytest.mark.parametrize("query", QUERIES)
def test_spielman_adjoint_solves_reuse_the_forward_factorization(cudss, query):
    conductance = conductance_surface("six_decades")
    distance = ResistanceDistance(solver=CuDSSSolver(), method=SPIELMAN)
    state = distance.init(conductance_grid(conductance))
    distances_and_state = query_and_state(distance, query)

    def distances(conductance):
        return distances_and_state(conductance_grid(conductance), state)[0]

    # The forward pass refactorizes the state and solves.
    forward, pullback = jax.vjp(distances, conductance)
    cotangent = jnp.asarray(np.random.default_rng(5).standard_normal(forward.shape))

    rebuilds = cudss.rebuild_count()
    jax.block_until_ready(pullback(cotangent))
    # The backward pass makes one multi-RHS adjoint solve with the forward
    # factorization: no new factorization, and no repeated analysis.
    assert cudss.rebuild_count() == rebuilds
    assert cudss_calls(cudss, lambda: pullback(cotangent)) == (0, 1)


def invalid_surface(name):
    """6 x 7 conductance rasters whose grounded Laplacian is not positive
    definite."""
    surface = np.array(conductance_surface("barrier_with_floor"))
    if name == "disconnected":
        # The barrier without its floor conducts nothing: the vertices left of
        # it float, and the grounded Laplacian is singular.
        surface[:, 3:5] = 0.0
    elif name == "negative":
        # The edges around this cell have negative conductances.
        surface[2, 2] = -10.0
    else:
        raise ValueError(name)
    return jnp.asarray(surface)


def raises_failed_solve():
    """Expect Lineax's error for a failed solve.

    Its exception type depends on the compilation wrapper and the JAX version.
    """
    return pytest.raises(Exception, match="linear solver returned non-finite")


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("surface", ["disconnected", "negative"])
def test_a_failed_factorization_raises(failing_cudss, surface, compiled, dtype):
    grid = conductance_grid(invalid_surface(surface).astype(dtype))
    distance = ResistanceDistance(solver=CuDSSSolver())
    # spineax reports nothing from analysis and factorization.
    state = distance.init(grid)

    def distances(grid, state):
        return distance(grid, state=state)

    if compiled:
        distances = jax.jit(distances)
    with raises_failed_solve():
        jax.block_until_ready(distances(grid, state))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("query", QUERIES)
@pytest.mark.parametrize("surface", ["disconnected", "negative"])
def test_a_failed_refactorization_raises_mid_optimization(
    failing_cudss, surface, query, dtype
):
    # Pivots of this surface stay above cuDSS's float32 pivot epsilon.
    valid = conductance_surface("smooth").astype(dtype)
    distance = ResistanceDistance(solver=CuDSSSolver())
    step = compiled_step(distance, query)
    _, state = step(valid, distance.init(conductance_grid(valid)))

    with raises_failed_solve():
        jax.block_until_ready(step(invalid_surface(surface).astype(dtype), state))
