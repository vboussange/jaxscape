"""What `CuDSSSolver` adds to `AbstractSparseDirectSolver`: its optional dependency,
its errors, and spineax's behaviour on an NVIDIA GPU, under the `gpu` marker.

`test_sparsedirectsolver.py` and `test_distance.py` also run `CuDSSSolver` on a GPU.
"""

import importlib.util
import subprocess
import sys
import tomllib
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest
from jaxscape import GridGraph, ResistanceDistance, SpielmanApproximation
from jaxscape.solvers import BCOOLinearOperator, CuDSSSolver, cudsssolver
from jaxscape.utils import graph_laplacian
from packaging.requirements import Requirement


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


def test_imports_and_defaults_load_no_cuda_library():
    """Importing JAXScape imports no spineax module, so imports and defaults
    are unchanged without the `cudss` extra."""
    code = """
import sys
import jaxscape
from jaxscape import ResistanceDistance
from jaxscape.solvers import CuDSSSolver
from jaxscape.solvers.cudsssolver import CUDSS_AVAILABLE
assert ResistanceDistance().solver is None
assert not any(name.startswith("spineax") for name in sys.modules)
print(CUDSS_AVAILABLE)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    installed = importlib.util.find_spec("spineax") is not None
    assert result.stdout.strip() == str(installed)


def installed_spineax_without_runtime(monkeypatch):
    """spineax is installed, but its cuDSS module cannot load."""
    monkeypatch.setattr(cudsssolver, "CUDSS_AVAILABLE", True)
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
    monkeypatch.setattr(cudsssolver, "CUDSS_AVAILABLE", False)
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


def test_the_solver_runs_on_spineax_phases(stub_spineax):
    del stub_spineax
    conductance = jnp.linspace(0.5, 2.0, 42, dtype=jnp.float64).reshape(6, 7)
    nodes = jnp.array([0, 9, 41])
    distance = ResistanceDistance(solver=CuDSSSolver())

    expected = ResistanceDistance()(grid_of(conductance), nodes=nodes)
    for dtype, rtol in ((jnp.float64, 1e-10), (jnp.float32, 1e-4)):
        grid = grid_of(conductance.astype(dtype))
        new_grid = grid_of(2 * conductance.astype(dtype))
        state = distance.reinit(new_grid, distance.init(grid))
        actual = distance(new_grid, nodes=nodes, state=state)
        np.testing.assert_allclose(actual, expected / 2, rtol=rtol)


@pytest.fixture
def cudss():
    try:
        return cudsssolver._load_cudss()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")


def grid_of(conductance):
    return GridGraph(conductance, fun=lambda x, y: (x + y) / 2)


def surface(name, shape=(6, 7)):
    rng = np.random.default_rng(0)
    if name == "smooth":
        values = np.linspace(0.5, 2.0, shape[0] * shape[1]).reshape(shape)
    elif name == "six_decades":
        values = 10.0 ** rng.uniform(-6.0, 0.0, shape)
    else:
        # A two-cell-wide barrier of zero conductance: the vertices left of it
        # float, and the grounded Laplacian is singular.
        values = 10.0 ** rng.uniform(-2.0, 0.0, shape)
        values[:, 3:5] = 0.0
    return jnp.asarray(values, dtype=jnp.float64)


QUERIES = {
    "all_pairs": {},
    "nodes_to_nodes": {"nodes": jnp.array([0, 9, 41, 20, 9], dtype=jnp.int32)},
    "sources_to_targets": {
        "sources": jnp.array([0, 17], dtype=jnp.int32),
        "targets": jnp.array([41, 3, 25], dtype=jnp.int32),
    },
}
SPIELMAN = SpielmanApproximation(epsilon=0.3, seed=7)


def compiled_calls(run):
    """Count the numeric factorizations and solve calls in the program of `run`.

    A `reinit` also holds branches that run only after a failed check, each with
    one analysis, one factorization and one check solve, so each analysis discounts
    one factorization and one solve.
    """
    program = jax.jit(run).lower().as_text()
    analyses = program.count("spineax_token_analyze")
    numeric = (
        program.count("spineax_token_factorize")
        + program.count("spineax_token_refactorize")
        - analyses
    )
    return numeric, program.count("spineax_token_solve") - analyses


@pytest.mark.gpu
def test_all_right_hand_sides_of_a_query_share_one_factorization(cudss):
    del cudss
    grid = grid_of(surface("six_decades"))
    distance = ResistanceDistance(solver=CuDSSSolver())
    state = distance.init(grid)
    nodes = QUERIES["nodes_to_nodes"]["nodes"]

    calls = compiled_calls(lambda: distance(grid, state=state, nodes=nodes))

    # One refactorization, its check (a solve of one right-hand side), and one
    # solve of all the right-hand sides of the query.
    assert calls == (1, 2)


@pytest.mark.gpu
def test_queries_allocate_no_dense_matrix(cudss):
    del cudss
    n = 900
    conductance = jnp.linspace(0.5, 2.0, n).reshape(30, 30)
    distance = ResistanceDistance(solver=CuDSSSolver())
    state = distance.init(grid_of(conductance))
    nodes = jnp.array([0, 100, n - 1], dtype=jnp.int32)

    def objective(conductance, state):
        return jnp.sum(distance(grid_of(conductance), nodes=nodes, state=state))

    # XLA's temporary buffers of the compiled query and of its gradient stay
    # far below one dense float64 grounded Laplacian.
    dense_grounded_laplacian = (n - 1) ** 2 * 8
    for program in (jax.jit(objective), jax.jit(jax.grad(objective))):
        memory = program.lower(conductance, state).compile().memory_analysis()
        assert memory is not None
        assert memory.temp_size_in_bytes < dense_grounded_laplacian / 4


@pytest.mark.gpu
@pytest.mark.parametrize("query", QUERIES)
def test_carrying_the_state_never_repeats_the_analysis(cudss, query):
    distance = ResistanceDistance(solver=CuDSSSolver())
    reference = ResistanceDistance()
    state = distance.init(grid_of(surface("smooth")))

    @jax.jit
    def step(log_conductance, state):
        def loss(log_conductance, state):
            grid = grid_of(jnp.exp(log_conductance))
            state = distance.reinit(grid, state)
            return jnp.mean(distance(grid, state=state, **QUERIES[query]) ** 2), state

        return jax.value_and_grad(loss, has_aux=True)(log_conductance, state)

    rebuilds = cudss.rebuild_count()
    for scale, name in enumerate(["smooth", "six_decades", "smooth"], start=1):
        log_conductance = jnp.log(scale * surface(name))
        (value, state), gradient = step(log_conductance, state)

        expected, expected_gradient = jax.value_and_grad(
            lambda c: jnp.mean(reference(grid_of(jnp.exp(c)), **QUERIES[query]) ** 2)
        )(log_conductance)
        np.testing.assert_allclose(value, expected, rtol=1e-8)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-10)
    assert step._cache_size() == 1  # type: ignore[attr-defined]
    assert cudss.rebuild_count() == rebuilds


@pytest.mark.gpu
def test_queries_with_one_state_share_its_factorization(cudss):
    distance = ResistanceDistance(solver=CuDSSSolver())
    grid = grid_of(surface("smooth"))
    state = distance.init(grid)

    rebuilds = cudss.rebuild_count()
    for query in QUERIES.values():
        jax.block_until_ready(distance(grid, state=state, **query))

    assert cudss.rebuild_count() == rebuilds


@pytest.mark.gpu
def test_a_used_up_state_repeats_the_analysis_and_stays_exact(cudss):
    distance = ResistanceDistance(solver=CuDSSSolver())
    nodes = QUERIES["nodes_to_nodes"]["nodes"]
    used_up = distance.init(grid_of(surface("smooth")))
    jax.block_until_ready(distance.reinit(grid_of(surface("six_decades")), used_up))

    rebuilds = cudss.rebuild_count()
    conductance = 2 * surface("smooth")
    actual = distance(grid_of(conductance), nodes=nodes, state=used_up)

    expected = ResistanceDistance()(grid_of(conductance), nodes=nodes)
    np.testing.assert_allclose(actual, expected, rtol=1e-8)
    assert cudss.rebuild_count() > rebuilds


@pytest.mark.gpu
@pytest.mark.parametrize("query", ["all_pairs", "sources_to_targets"])
def test_spielman_adjoint_solves_reuse_the_forward_factorization(cudss, query):
    conductance = surface("six_decades")
    distance = ResistanceDistance(solver=CuDSSSolver(), method=SPIELMAN)
    state = distance.init(grid_of(conductance))

    def distances(conductance):
        return distance(grid_of(conductance), state=state, **QUERIES[query])

    forward, pullback = jax.vjp(distances, conductance)
    reference = ResistanceDistance(method=SPIELMAN)
    expected = reference(grid_of(conductance), **QUERIES[query])
    np.testing.assert_allclose(
        forward, expected, rtol=1e-6, atol=1e-10 * float(np.max(np.abs(expected)))
    )
    cotangent = jnp.asarray(np.random.default_rng(5).standard_normal(forward.shape))

    rebuilds = cudss.rebuild_count()
    jax.block_until_ready(pullback(cotangent))
    # The backward pass makes one multi-RHS adjoint solve with the forward
    # factorization: no new factorization, and no repeated analysis.
    assert cudss.rebuild_count() == rebuilds
    assert compiled_calls(lambda: pullback(cotangent)) == (0, 1)


@pytest.mark.gpu
def test_a_refactorization_after_a_cut_off_graph_is_retried(cudss):
    """The factorization of a graph with vertices cut off from the ground vertex
    can pass its check and still leave its cuDSS handle unusable, as it did on a
    100 x 100 grid of an RTX 3090. The next refactorization is retried."""
    del cudss
    solver = CuDSSSolver()
    laplacians = [
        graph_laplacian(grid_of(c).get_adjacency_matrix())[:-1, :-1]
        for c in (surface("smooth"), surface("cut_off"), 3 * surface("smooth"))
    ]
    operators = [BCOOLinearOperator(laplacian) for laplacian in laplacians]
    b = jnp.ones(operators[0].in_size())

    @jax.jit
    def step(operator, state):
        state = solver.reinit(operator, {}, state)
        solution = lx.linear_solve(operator, b, solver, state=state, throw=False)
        return solution.value, solution.result, state

    _, _, state = step(operators[1], solver.init(operators[0], {}))
    value, result, _ = step(operators[2], state)

    assert result == lx.RESULTS.successful
    expected = np.linalg.solve(np.asarray(laplacians[2].todense()), b)
    np.testing.assert_allclose(value, expected, rtol=1e-8)
