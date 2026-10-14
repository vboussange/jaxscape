"""Benchmark wiring correctness, without running performance workloads."""

import jax.numpy as jnp
import pytest
from benchmark.benchmark_registry import DISPLAY_ORDER_BY_TASK, TOOL_SPECS_BY_TASK
from benchmark.jaxscape import inverse_landscape_genetics, resistance_distance


@pytest.mark.parametrize("runner", [resistance_distance, inverse_landscape_genetics])
def test_jaxamg_profiles_are_exact_gpu_only_in_both_precisions(runner):
    profiles = runner.selected_profiles(["jaxamgcg_f32", "jaxamgcg_f64"])
    assert [str(jnp.dtype(profile.dtype)) for profile in profiles] == [
        "float32",
        "float64",
    ]
    assert all(profile.gpu_only and profile.gpu_capable for profile in profiles)
    assert all(profile.method_factory is None for profile in profiles)
    assert all(not profile.requires_preparation for profile in profiles)
    task = (
        "resistance_distance"
        if runner is resistance_distance
        else "inverse_landscape_genetics"
    )
    for profile in profiles:
        assert f"{profile.tool} (GPU)" in DISPLAY_ORDER_BY_TASK[task]
        assert f"{profile.tool} (CPU)" not in DISPLAY_ORDER_BY_TASK[task]
    assert not any(
        "JAXAMG" in tool.label for tool in TOOL_SPECS_BY_TASK["sensitivity_analysis"]
    )


@pytest.mark.parametrize(
    "runner,case",
    [
        (resistance_distance, "synthetic_resistance_small"),
        (inverse_landscape_genetics, "synthetic_inverse_medium"),
    ],
)
@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_jaxamg_worker_rejects_unavailable_backend_without_workload(
    runner, case, backend
):
    import jax

    if backend == "gpu" and jax.default_backend() == "gpu":
        pytest.skip("This availability check requires a CPU-only JAX process")
    record = runner.run_worker(
        {
            "case_name": case,
            "profile_key": "jaxamgcg_f32",
            "backend": backend,
            "repeats": 3,
        }
    )
    assert record.status == "skipped"
    assert record.median_seconds is None
    assert "NVIDIA" in record.note


@pytest.mark.parametrize(
    "runner,group",
    [(resistance_distance, "resistance"), (inverse_landscape_genetics, "inverse")],
)
def test_gpu_only_collection_reports_unavailable_without_cpu_fallback(runner, group):
    from dataclasses import replace

    from benchmark.benchmark_distances import CASES, default_jaxscape_config

    config = replace(default_jaxscape_config(), gpu_device=None)
    collect = (
        runner.collect_jaxscape_resistance_results
        if group == "resistance"
        else runner.collect_jaxscape_inverse_results
    )
    records = collect(
        CASES[group][0], config, profile_keys=["jaxamgcg_f32", "jaxamgcg_f64"]
    )
    assert len(records) == 2
    assert all(record.tool.endswith("(GPU)") for record in records)
    assert all(
        record.status == "skipped" and record.median_seconds is None
        for record in records
    )


def test_missing_optional_package_has_actionable_availability_error(monkeypatch):
    import sys

    from benchmark.jaxscape.resistance_profile_support import make_jaxamgcg_solver

    monkeypatch.setitem(sys.modules, "jaxamg", None)
    with pytest.raises(ImportError, match="jaxscape\\[jaxamg\\].*CUDA/AmgX"):
        make_jaxamgcg_solver()


def test_report_environment_includes_dependency_versions():
    from benchmark.benchmark_distances import (
        default_jaxscape_config,
        environment_payload,
    )
    from benchmark.generate_benchmark_docs import environment_table

    environment = environment_payload(default_jaxscape_config())
    assert environment["dependency_versions"]["jax"]
    rendered = environment_table({"environment": environment})
    assert "Dependency versions" in rendered
    assert "jaxamg" in rendered


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_compiled_inverse_benchmark_uses_current_values_and_gradients(dtype):
    import equinox as eqx
    import jax
    import numpy as np
    from benchmark.jaxscape.resistance_profile_support import make_jaxamgcg_solver
    from jaxscape import ResistanceDistance

    if jax.default_backend() != "gpu":
        pytest.skip("JAX-AMG benchmark objective requires an NVIDIA GPU")
    pytest.importorskip("jaxamg")
    pytest.importorskip("jaxamg._amgx", reason="JAX-AMG native AmgX backend missing")
    if "cuda" not in jax.devices()[0].client.platform_version.lower():
        pytest.skip("JAX-AMG requires CUDA")

    runner = inverse_landscape_genetics
    base = jnp.asarray([[1.1, 1.4, 2.2], [2.6, 1.8, 1.3], [1.5, 2.1, 2.8]], dtype=dtype)
    coords = jnp.asarray([[0, 0], [1, 2], [2, 1]], dtype=jnp.int32)
    reference = ResistanceDistance()
    initial = runner._initial_inverse_params(base)
    targets = reference(
        runner._inverse_grid(runner._permeability_from_inverse_params(initial, base)),
        nodes=coords,
    )

    def loss(params, distance):
        return runner._inverse_loss(params, base, coords, targets, distance=distance)

    compiled = eqx.filter_jit(eqx.filter_value_and_grad(loss))
    distance = ResistanceDistance(solver=make_jaxamgcg_solver())
    losses = []
    for offset in [(0.8, -0.6), (-0.4, 0.7)]:
        params = initial + jnp.asarray(offset, dtype=dtype)
        expected = compiled(params, reference)
        actual = compiled(params, distance)
        for got, want in zip(actual, expected, strict=True):
            assert bool(jnp.all(jnp.isfinite(got)))
            np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-5)
        losses.append(float(actual[0]))
    assert not np.isclose(*losses)
