"""cuDSS benchmark profiles, checked without running performance workloads."""

import importlib.machinery
import sys
import types

import jax
import jax.numpy as jnp
import pytest
from benchmark.benchmark_registry import DISPLAY_ORDER_BY_TASK
from benchmark.jaxscape import inverse_landscape_genetics, resistance_distance


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
    spineax = types.ModuleType("spineax")
    spineax.__spec__ = importlib.machinery.ModuleSpec("spineax", None)
    monkeypatch.setitem(sys.modules, "spineax", spineax)

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
