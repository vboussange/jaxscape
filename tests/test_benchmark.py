"""Benchmark report metadata and a small task-runner smoke check."""


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
