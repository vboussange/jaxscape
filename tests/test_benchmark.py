"""Benchmark report metadata and a small task-runner smoke check."""

from dataclasses import replace
from importlib.metadata import PackageNotFoundError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmark import benchmark_distances
from benchmark.benchmark_distances import (
    build_case,
    CASES,
    CASES_BY_TASK,
    default_jaxscape_config,
    environment_payload,
)
from benchmark.benchmark_registry import DISPLAY_ORDER_BY_TASK
from benchmark.generate_benchmark_docs import environment_table
from benchmark.jaxscape import (
    inverse_landscape_genetics,
    resistance_distance,
    resistance_profile_support,
)
from jaxscape.solvers import cudsssolver


CUDSS_RUNNERS = {
    "resistance_distance": resistance_distance,
    "inverse_landscape_genetics": inverse_landscape_genetics,
}
CUDSS_CASES = {
    "resistance_distance": "synthetic_resistance_small",
    "inverse_landscape_genetics": "synthetic_inverse_medium",
}
CUDSS_PROFILE_KEYS = ["cudss_f32", "cudss_f64"]


def test_report_environment_includes_dependency_versions():
    environment = environment_payload(default_jaxscape_config())
    versions = environment["dependency_versions"]
    assert isinstance(versions, dict)
    assert versions["jax"]
    assert {"spineax", "nvidia-cudss-cu13", "nvidia-cudss-cu12"} <= versions.keys()
    rendered = environment_table({"environment": environment})
    assert "Dependency versions" in rendered
    assert "amjax" in rendered
    assert "spineax=" in rendered


def test_free_energy_benchmark_worker():
    from benchmark.jaxscape.free_energy_distance import run_worker

    record = run_worker(
        {
            "case_name": "synthetic_free_energy_small",
            "backend": "cpu",
            "repeats": 1,
        }
    )
    assert record.status == "ok", record.note
    assert len(record.timings_seconds) == 1
    assert record.tool == "JAXScape / CG / f64 (CPU)"
    assert record.metrics["theta"] == 0.1
    assert record.metrics["dtype"] == "float64"


@pytest.mark.parametrize("task", CUDSS_RUNNERS)
def test_cudss_profiles_are_registered_for_gpus_in_both_precisions(task):
    profiles = CUDSS_RUNNERS[task].selected_profiles(CUDSS_PROFILE_KEYS)

    assert [jnp.dtype(profile.dtype) for profile in profiles] == [
        jnp.float32,
        jnp.float64,
    ]
    for profile in profiles:
        assert profile.method_factory is None
        assert f"{profile.tool} (GPU)" in DISPLAY_ORDER_BY_TASK[task]
        assert f"{profile.tool} (CPU)" not in DISPLAY_ORDER_BY_TASK[task]


@pytest.mark.parametrize("spineax_installed", [False, True])
@pytest.mark.parametrize("task", CUDSS_RUNNERS)
def test_cudss_workers_skip_without_an_nvidia_gpu(task, spineax_installed, monkeypatch):
    if jax.default_backend() == "gpu":
        pytest.skip("This availability check needs a JAX process without a GPU.")
    if spineax_installed:
        monkeypatch.setattr(cudsssolver, "CUDSS_AVAILABLE", True)
    record = CUDSS_RUNNERS[task].run_worker(
        {
            "case_name": CUDSS_CASES[task],
            "profile_key": "cudss_f64",
            "backend": "gpu",
            "repeats": 3,
        }
    )

    assert record.status == "skipped"
    assert record.tool == "JAXScape / CuDSSSolver / f64 (GPU)"
    assert record.timings_seconds == []
    assert "NVIDIA GPU" in record.note


@pytest.mark.parametrize("task", CUDSS_RUNNERS)
def test_cudss_collection_without_a_gpu_records_only_skipped_series(task):
    runner = CUDSS_RUNNERS[task]
    collect = (
        runner.collect_jaxscape_resistance_results
        if task == "resistance_distance"
        else runner.collect_jaxscape_inverse_results
    )
    config = replace(default_jaxscape_config(), gpu_device=None)

    records = collect(CASES_BY_TASK[task][0], config, profile_keys=CUDSS_PROFILE_KEYS)

    assert [record.tool for record in records] == [
        "JAXScape / CuDSSSolver / f32 (GPU)",
        "JAXScape / CuDSSSolver / f64 (GPU)",
    ]
    assert all(record.status == "skipped" for record in records)


@pytest.mark.parametrize("cudss_package", ["nvidia-cudss-cu13", "nvidia-cudss-cu12"])
def test_cudss_records_carry_spineax_and_cudss_versions(cudss_package, monkeypatch):
    installed = {"spineax": "0.0.5", cudss_package: "0.8.0.10"}

    def installed_version(package):
        if package not in installed:
            raise PackageNotFoundError(package)
        return installed[package]

    monkeypatch.setattr(benchmark_distances, "version", installed_version)

    profile = resistance_distance.PROFILE_BY_KEY["cudss_f64"]
    metadata = resistance_profile_support.profile_metadata(profile, "gpu")
    assert metadata["spineax_version"] == "0.0.5"
    assert metadata["cudss_version"] == f"{cudss_package}==0.8.0.10"


def test_cudss_resistance_profile_passes_the_state_from_call_to_call(stub_spineax):
    del stub_spineax
    profile = resistance_distance.PROFILE_BY_KEY["cudss_f64"]

    record = resistance_distance._run_resistance_profile_direct(
        CASES["resistance"][0],
        f"{profile.tool} (GPU)",
        device=jax.devices("cpu")[0],
        repeats=3,
        solver_factory=profile.solver_factory,
        passes_state=profile.passes_state,
        dtype=profile.dtype,
    )

    assert record.status == "ok", record.note
    assert len(record.timings_seconds) == 3
    assert record.metrics["spineax_rebuilds"] == 0


def tiny_inverse_case():
    return build_case(
        "tiny_inverse", "inverse_landscape_genetics", "small", size=6, seed=30
    )


@pytest.fixture(params=["stub", pytest.param("spineax", marks=pytest.mark.gpu)])
def cudss(request):
    if request.param == "stub":
        return request.getfixturevalue("stub_spineax")
    try:
        return cudsssolver._load_cudss()
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"Needs an NVIDIA GPU, spineax and cuDSS: {error}")


def test_the_inverse_loop_takes_the_steps_of_optimistix_minimise(cudss, request):
    """It re-analyses once early, and once for each rejected line-search step."""
    optx = pytest.importorskip("optimistix")
    from benchmark.benchmark_distances import as_point_array
    from jaxscape import ResistanceDistance
    from jaxscape.solvers import CuDSSSolver

    runner = inverse_landscape_genetics
    case = tiny_inverse_case()
    base = runner._base_resistance(jnp.asarray(case.raster, dtype=jnp.float64))
    coords = as_point_array(case.points)
    start = runner._initial_inverse_params(base)

    def grid(params):
        return runner._inverse_grid(
            runner._permeability_from_inverse_params(params, base)
        )

    # Targets far from the start, so that the line search rejects steps.
    targets = ResistanceDistance()(grid(start + 4.0), nodes=coords)
    args = (base, coords, targets)
    distance = ResistanceDistance(solver=CuDSSSolver())
    optimizer = optx.LBFGS(rtol=1e-5, atol=1e-5)

    def objective(params, args_and_state):
        (base, coords, targets), state = args_and_state
        return runner._inverse_loss_and_state(
            params, base, coords, targets, state, distance=distance
        )

    state = distance.init(grid(start))
    rebuilds = cudss.rebuild_count()
    solution = runner._minimise_carrying_state(
        objective, optimizer, start, args, state, max_steps=50
    )
    rejected_steps = int(solution.stats["rejected_steps"])
    rebuilds = cudss.rebuild_count() - rebuilds

    expected = optx.minimise(
        lambda params, args: runner._inverse_loss(
            params, *args, distance=ResistanceDistance()
        ),
        optimizer,
        start,
        args,
        max_steps=50,
        throw=False,
    )
    assert int(solution.stats["num_steps"]) == int(expected.stats["num_steps"]) > 1
    assert solution.result == expected.result
    # Equal to rounding: GPU scatter-adds sum in varying order.
    np.testing.assert_allclose(solution.value, expected.value, rtol=1e-7)
    assert rejected_steps > 0
    if request.node.get_closest_marker("gpu"):
        assert rebuilds == rejected_steps + 1
