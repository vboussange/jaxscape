"""cuDSS benchmark profiles, checked without running performance workloads."""


import jax
import jax.numpy as jnp
import pytest
from benchmark.benchmark_registry import DISPLAY_ORDER_BY_TASK
from benchmark.jaxscape import inverse_landscape_genetics, resistance_distance
from fake_cudss import FAKE_CUDSS
from jaxscape.solvers import cudsssolver


RUNNERS = {
    "resistance_distance": resistance_distance,
    "inverse_landscape_genetics": inverse_landscape_genetics,
}
CUDSS_PROFILE_KEYS = ["cudss_f32", "cudss_f64"]


@pytest.mark.parametrize("task", RUNNERS)
def test_cudss_profiles_are_registered_for_gpus_in_both_precisions(task):
    profiles = RUNNERS[task].selected_profiles(CUDSS_PROFILE_KEYS)

    assert [profile.tool for profile in profiles] == [
        "JAXScape / CuDSSSolver / f32",
        "JAXScape / CuDSSSolver / f64",
    ]
    assert [jnp.dtype(profile.dtype) for profile in profiles] == [
        jnp.float32,
        jnp.float64,
    ]
    # Exact resistance distances, on NVIDIA GPUs only.
    assert all(profile.method_factory is None for profile in profiles)
    for profile in profiles:
        assert f"{profile.tool} (GPU)" in DISPLAY_ORDER_BY_TASK[task]
        assert f"{profile.tool} (CPU)" not in DISPLAY_ORDER_BY_TASK[task]


CASE_BY_TASK = {
    "resistance_distance": "synthetic_resistance_small",
    "inverse_landscape_genetics": "synthetic_inverse_medium",
}


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("key", CUDSS_PROFILE_KEYS)
@pytest.mark.parametrize("task", RUNNERS)
def test_cudss_worker_skips_without_an_nvidia_gpu_and_runs_nothing(task, key, backend):
    if backend == "gpu" and jax.default_backend() == "gpu":
        pytest.skip("This availability check needs a JAX process without a GPU.")
    runner = RUNNERS[task]
    record = runner.run_worker(
        {
            "case_name": CASE_BY_TASK[task],
            "profile_key": key,
            "backend": backend,
            "repeats": 3,
        }
    )

    assert record.status == "skipped"
    assert record.tool.startswith(runner.PROFILE_BY_KEY[key].tool)
    assert record.median_seconds is None
    assert record.timings_seconds == []
    assert record.metrics == {}
    assert "NVIDIA GPU" in record.note


@pytest.mark.parametrize("task", RUNNERS)
def test_cudss_worker_skips_when_spineax_is_installed_without_a_gpu(task, monkeypatch):
    if jax.default_backend() == "gpu":
        pytest.skip("This availability check needs a JAX process without a GPU.")
    monkeypatch.setattr(cudsssolver, "CUDSS_AVAILABLE", True)

    record = RUNNERS[task].run_worker(
        {
            "case_name": CASE_BY_TASK[task],
            "profile_key": "cudss_f64",
            "backend": "gpu",
            "repeats": 3,
        }
    )

    assert record.status == "skipped"
    assert record.tool == "JAXScape / CuDSSSolver / f64 (GPU)"
    assert record.timings_seconds == []
    assert "NVIDIA GPU" in record.note


@pytest.mark.parametrize("task", RUNNERS)
def test_cudss_collection_without_a_gpu_records_only_skipped_gpu_series(task):
    from dataclasses import replace

    from benchmark.benchmark_distances import CASES_BY_TASK, default_jaxscape_config

    runner = RUNNERS[task]
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
    assert all(record.timings_seconds == [] for record in records)


def test_report_environment_lists_spineax_and_cudss_versions():
    from benchmark.benchmark_distances import (
        default_jaxscape_config,
        environment_payload,
    )
    from benchmark.generate_benchmark_docs import environment_table

    environment = environment_payload(default_jaxscape_config())
    versions = environment["dependency_versions"]

    assert isinstance(versions, dict)
    assert {"spineax", "nvidia-cudss-cu12", "nvidia-cudss-cu13"} <= versions.keys()
    rendered = environment_table({"environment": environment})
    assert "spineax=" in rendered
    assert "nvidia-cudss-cu13=" in rendered


@pytest.mark.parametrize(
    "installed, cudss_version",
    [
        (
            {"spineax": "0.0.5", "nvidia-cudss-cu13": "0.8.0.10"},
            "nvidia-cudss-cu13==0.8.0.10",
        ),
        (
            {"spineax": "0.0.5", "nvidia-cudss-cu12": "0.8.0.10"},
            "nvidia-cudss-cu12==0.8.0.10",
        ),
    ],
)
def test_cudss_records_carry_spineax_and_cudss_versions(
    installed, cudss_version, monkeypatch
):
    from importlib.metadata import PackageNotFoundError

    from benchmark import benchmark_distances
    from benchmark.jaxscape import resistance_profile_support

    def installed_version(package):
        if package not in installed:
            raise PackageNotFoundError(package)
        return installed[package]

    monkeypatch.setattr(benchmark_distances, "version", installed_version)

    for key in CUDSS_PROFILE_KEYS:
        profile = resistance_distance.PROFILE_BY_KEY[key]
        metadata = resistance_profile_support.profile_metadata(profile, "gpu")
        assert metadata["spineax_version"] == "0.0.5"
        assert metadata["cudss_version"] == cudss_version


@pytest.fixture
def cpu_double_of_cudss(monkeypatch):
    """Run `CuDSSSolver` on the CPU double of spineax's phase API."""
    FAKE_CUDSS.reset()
    monkeypatch.setattr(cudsssolver, "_load_cudss", lambda: FAKE_CUDSS)
    return FAKE_CUDSS


def test_cudss_resistance_profile_passes_the_state_from_call_to_call(
    cpu_double_of_cudss,
):
    from benchmark.benchmark_distances import CASES

    case = CASES["resistance"][0]
    profile = resistance_distance.PROFILE_BY_KEY["cudss_f64"]

    record = resistance_distance._run_resistance_profile_direct(
        case,
        f"{profile.tool} (GPU)",
        device=jax.devices("cpu")[0],
        repeats=3,
        solver_factory=profile.solver_factory,
        passes_state=profile.passes_state,
        dtype=profile.dtype,
    )

    assert record.status == "ok", record.note
    assert len(record.timings_seconds) == 3
    # Analysis runs once, before timing. Every call refactorizes the state the
    # previous call returned, so spineax never repeats the analysis.
    assert cpu_double_of_cudss.analyses == 1
    assert cpu_double_of_cudss.rebuild_count() == 0
    assert record.metrics["spineax_rebuilds"] == 0


def tiny_inverse_case():
    from benchmark.benchmark_distances import build_case

    return build_case(
        "tiny_inverse", "inverse_landscape_genetics", "small", size=6, seed=30
    )


def test_cudss_inverse_profile_records_the_analyses_spineax_repeats(
    cpu_double_of_cudss,
):
    pytest.importorskip("cholespy", reason="target distances use CholmodSolver")
    pytest.importorskip("optimistix")
    profile = inverse_landscape_genetics.PROFILE_BY_KEY["cudss_f32"]

    record = inverse_landscape_genetics._run_inverse_profile_direct(
        tiny_inverse_case(),
        profile,
        device=jax.devices("cpu")[0],
        repeats=2,
        tool_label=f"{profile.tool} (GPU)",
    )

    assert record.status == "ok", record.note
    assert len(record.timings_seconds) == 2
    assert record.metrics["iteration_count"] > 1
    # Each optimization starts from the state the previous one returned, and
    # its objective evaluations share that state, so spineax repeats the
    # analysis for evaluations after the first; the record counts them.
    assert record.metrics["spineax_rebuilds"] == cpu_double_of_cudss.rebuild_count()
