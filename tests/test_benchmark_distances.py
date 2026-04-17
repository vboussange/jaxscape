import json
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "benchmark" / "benchmark_distances.py"


def test_run_benchmarks_connectivity_and_sensitivity(tmp_path):
    output_path = tmp_path / "benchmark-results.json"
    subprocess.run(
        [
            "python",
            str(SCRIPT_PATH),
            "--scenario",
            "connectivity",
            "--device",
            "cpu",
            "--distance",
            "euclidean",
            "--grid-sizes",
            "3",
            "--repeats",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
    )
    connectivity_results = json.loads(output_path.read_text())

    assert output_path.exists()
    assert {run["scenario"] for run in connectivity_results["runs"]} == {"connectivity"}

    for run in connectivity_results["runs"]:
        assert run["device"] == "cpu"
        assert run["grid_shape"] == [3, 3]
        assert len(run["timings_seconds"]) == 1
        assert run["median_seconds"] >= 0

    subprocess.run(
        [
            "python",
            str(SCRIPT_PATH),
            "--scenario",
            "sensitivity",
            "--device",
            "cpu",
            "--distance",
            "euclidean",
            "--grid-sizes",
            "3",
            "--repeats",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
    )
    sensitivity_results = json.loads(output_path.read_text())
    assert {run["scenario"] for run in sensitivity_results["runs"]} == {"sensitivity"}


def test_run_benchmarks_inverse_handles_optional_dependency(tmp_path):
    output_path = tmp_path / "inverse-benchmark-results.json"
    subprocess.run(
        [
            "python",
            str(SCRIPT_PATH),
            "--scenario",
            "inverse",
            "--device",
            "cpu",
            "--grid-sizes",
            "3",
            "--repeats",
            "1",
            "--inverse-max-steps",
            "1",
            "--output",
            str(output_path),
        ],
        check=True,
    )
    results = json.loads(output_path.read_text())

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
