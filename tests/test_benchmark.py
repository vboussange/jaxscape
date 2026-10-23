"""Benchmark report metadata, without running performance workloads."""


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
    assert "amjax" in rendered
