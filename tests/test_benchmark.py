"""Benchmark report metadata, without running performance workloads."""


def test_report_environment_includes_dependency_versions():
    from benchmark.benchmark_distances import (
        default_jaxscape_config,
        environment_payload,
    )
    from benchmark.generate_benchmark_docs import environment_table

    environment = environment_payload(default_jaxscape_config())
    versions = environment["dependency_versions"]
    assert isinstance(versions, dict)
    assert versions["jax"]
    rendered = environment_table({"environment": environment})
    assert "Dependency versions" in rendered
    assert "amjax" in rendered


def test_free_energy_benchmark_reports_independent_accuracy():
    from benchmark.jaxscape.free_energy_distance import run_worker

    record = run_worker(
        {
            "case_name": "synthetic_free_energy_small",
            "theta": 0.1,
            "backend": "cpu",
            "repeats": 1,
        }
    )
    assert record.status == "ok", record.note
    assert len(record.timings_seconds) == 1
    assert record.metrics["off_diagonal_entries"] == 380
    assert record.metrics["max_scaled_error"] < 1e-7
    assert len(record.metrics["derivative_checks"]) == 4
    assert max(x["scaled_error"] for x in record.metrics["derivative_checks"]) < 1e-6
