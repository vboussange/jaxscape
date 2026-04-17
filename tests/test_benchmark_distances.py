import json
import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "benchmark" / "benchmark_distances.py"
)
SPEC = importlib.util.spec_from_file_location("jaxscape_benchmark_distances", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

BenchmarkConfig = MODULE.BenchmarkConfig
run_benchmarks = MODULE.run_benchmarks


def test_run_benchmarks_connectivity_and_sensitivity(tmp_path):
    output_path = tmp_path / "benchmark-results.json"
    results = run_benchmarks(
        BenchmarkConfig(
            grid_sizes=(3,),
            repeats=1,
            distance_name="euclidean",
            device_names=("cpu",),
            scenario_names=("connectivity", "sensitivity"),
            dispersal_radius=2.0,
            output_path=str(output_path),
        )
    )

    assert output_path.exists()
    assert json.loads(output_path.read_text()) == results
    assert {run["scenario"] for run in results["runs"]} == {
        "connectivity",
        "sensitivity",
    }

    for run in results["runs"]:
        assert run["device"] == "cpu"
        assert run["grid_shape"] == [3, 3]
        assert len(run["timings_seconds"]) == 1
        assert run["median_seconds"] >= 0


def test_run_benchmarks_inverse_handles_optional_dependency(tmp_path):
    output_path = tmp_path / "inverse-benchmark-results.json"
    results = run_benchmarks(
        BenchmarkConfig(
            grid_sizes=(3,),
            repeats=1,
            device_names=("cpu",),
            scenario_names=("inverse",),
            inverse_max_steps=1,
            output_path=str(output_path),
        )
    )

    run = results["runs"][0]
    assert run["scenario"] == "inverse"
    assert run["device"] == "cpu"
    assert len(run["timings_seconds"]) == 1
    assert run["median_seconds"] >= 0

    if run["status"] == "skipped":
        assert "optimistix" in run["reason"]
    else:
        assert run["final_loss"] >= 0
        assert run["num_steps"] >= 0
