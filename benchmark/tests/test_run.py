import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


BENCHMARK = Path(__file__).resolve().parents[1]
CORES = ",".join(str(core) for core in sorted(os.sched_getaffinity(0))[:2])
CONFIG = f"""
[run]
timeout = 10
repeats = 3
single_call_after = 300
cores = "{CORES}"
gpu = ""
smoke_side = 16
smoke_dependency_range = 1

[landscape]
correlation_length = 4
habitat_threshold = 0.7
seed = 1

[E1]
sites = 4
seed = 2

[distances.resistance]
sides = [8]

[distances.lcp]
sides = [8, 16, 32]

[distances.free_energy]
sides = [8, 16]
theta = 0.01

[windows]
dependency_range = 2
coarsening_factor = 0.5

[maryland]
raster = "LAND_COVER"
questions = ["E2"]
resistance = {{ 1 = 1, 2 = 10 }}

[[tools]]
tool = "Fake"
question = "E1"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 1

[tools.distances.resistance]
dtype = "float64"

[tools.distances.lcp]
dtype = "float64"
fail_from = 16

[tools.distances.free_energy]
dtype = "float64"
sleep_from = 16

[[tools]]
tool = "Fake per device"
question = "E1"
adapter = "tests/fake_adapter.py"
devices = ["gpu", "cpu"]

[tools.distances.resistance]
solver = {{ gpu = ["Iterative", "Direct"], cpu = "Direct" }}

[[tools]]
tool = "Fake E2"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]

[tools.distances.resistance]
dtype = "float32"

[[tools]]
tool = "JAXScape"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]

[tools.distances.lcp]

[[tools]]
tool = "Fake map"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 3
compared_with = "lcp"

[tools.distances.free_energy]
off_graph = true

[[tools]]
tool = "Fake map without JAXScape"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 3

[tools.distances.resistance]

[solver_view]
tool = "Fake solvers"
adapter = "tests/fake_adapter.py"
device = "gpu"
dtypes = ["float32", "float64"]
reference_solver = "Direct"
timeout = 2

[solver_view.questions.E1]
resistance = {{}}
free_energy = {{ sides = [8] }}

[solver_view.questions.E2]
resistance = {{ batch_size = 4, sides = [8] }}

[solver_view.solvers]
Direct = {{}}
Iterative = {{ error = 0.5 }}
Approximate = {{ error = 0.1, questions = ["E2"], dtypes = ["float32"] }}
Slow = {{ sleep_from = 8, questions = ["E2"], dtypes = ["float64"] }}
"""
# A land cover of 20 x 24 pixels, of classes 1 (resistance 1), 2 (resistance 10)
# and 3 (a barrier).
LAND_COVER = np.resize(np.array([1, 1, 2, 3], np.uint8), (20, 24))


def run(tmp_path, *arguments, config_text=CONFIG):
    land_cover = tmp_path / "land_cover.tif"
    Image.fromarray(LAND_COVER).save(land_cover)
    config = tmp_path / "config.toml"
    config.write_text(config_text.replace("LAND_COVER", str(land_cover)))
    return subprocess.run(
        [sys.executable, "run.py", *map(str, arguments), "--config", str(config)],
        cwd=BENCHMARK,
        capture_output=True,
        text=True,
    )


def records(run_dir, tool="Fake"):
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    return {
        (r["distance"], r["case"]): r
        for r in map(json.loads, lines)
        if r["tool"] == tool
    }


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("run")
    run(tmp_path, tmp_path / "run")
    return tmp_path / "run"


def test_runner_records_the_status_of_each_cell(run_dir):
    statuses = {cell: record["status"] for cell, record in records(run_dir).items()}

    assert statuses == {
        ("resistance", "synthetic-8"): "ok",
        ("lcp", "synthetic-8"): "ok",
        ("free_energy", "synthetic-8"): "ok",
        ("lcp", "synthetic-16"): "failed",
        ("free_energy", "synthetic-16"): "timeout",
        ("lcp", "synthetic-32"): "skipped",
    }


def test_runner_records_timings_version_and_agreement_of_a_cell(run_dir):
    record = records(run_dir)[("resistance", "synthetic-8")]

    assert record["tool"] == "Fake"
    assert record["device"] == "cpu"
    assert record["dtype"] == "float64"
    assert record["cells"] == 64
    assert record["warmup_s"] > 0
    assert len(record["times_s"]) == 3
    assert min(record["times_s"]) <= record["median_s"] <= max(record["times_s"])
    assert record["version"] == "1.0"
    assert record["files"] is False
    assert record["tier"] == 1
    assert record["measure"] == "largest relative error"
    assert record["agreement"] > 0


def test_runner_resolves_settings_per_device_and_runs_each_solver_of_a_list(run_dir):
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    solvers = {
        (r["device"], r["solver"])
        for r in map(json.loads, lines)
        if r["tool"] == "Fake per device"
    }

    assert solvers == {("gpu", "Iterative"), ("gpu", "Direct"), ("cpu", "Direct")}


def test_runner_runs_the_maryland_case_for_its_questions(run_dir):
    cells = {
        case: record["cells"]
        for (_, case), record in records(run_dir, "Fake E2").items()
    }

    assert cells == {"synthetic-8": 64, "maryland": 480}


def solver_records(run_dir):
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    return {
        (r["question"], r["distance"], r["dtype"], r["solver"], r["case"]): r
        for r in map(json.loads, lines)
        if r["tool"] == "Fake solvers"
    }


def test_solver_view_runs_each_solver_in_each_dtype_on_the_gpu(run_dir):
    cells = solver_records(run_dir)

    def each(question, distance, cases, solvers=("Direct", "Iterative")):
        dtypes = ("float32", "float64")
        return {
            (question, distance, dtype, solver, case)
            for case in cases
            for dtype in dtypes
            for solver in solvers
        }

    assert set(cells) == (
        each("E1", "resistance", ["synthetic-8"])
        | each("E1", "free_energy", ["synthetic-8"])
        | each("E2", "resistance", ["synthetic-8"])
        | {("E2", "resistance", "float32", "Approximate", "synthetic-8")}
        | {("E2", "resistance", "float64", "Slow", "synthetic-8")}
    )
    assert {(r["view"], r["device"]) for r in cells.values()} == {("solver", "gpu")}
    assert {r["view"] for r in records(run_dir).values()} == {"comparison"}


def test_solver_view_records_the_error_against_the_float64_reference(run_dir):
    cells = solver_records(run_dir)
    e2 = {
        solver: cells[("E2", "resistance", "float32", solver, "synthetic-8")]
        for solver in ("Direct", "Iterative", "Approximate")
    }

    assert {r["measure"] for r in e2.values()} == {"relative L2 error"}
    assert e2["Direct"]["agreement"] == 0
    assert e2["Iterative"]["agreement"] == pytest.approx(0.5)
    assert e2["Approximate"]["agreement"] == pytest.approx(0.1)
    e1 = cells[("E1", "resistance", "float64", "Direct", "synthetic-8")]
    assert e1["measure"] == "largest relative error"
    assert (
        e1["agreement"] == records(run_dir)[("resistance", "synthetic-8")]["agreement"]
    )


def test_solver_view_stops_a_call_after_its_own_timeout(run_dir):
    cells = solver_records(run_dir)
    slow = cells.pop(("E2", "resistance", "float64", "Slow", "synthetic-8"))

    assert (slow["status"], slow["reason"]) == ("timeout", "a call exceeded 2 s")
    assert {r["status"] for r in cells.values()} == {"ok"}


def test_runner_records_no_agreement_for_a_tool_without_tier(run_dir):
    record = records(run_dir, "Fake E2")[("resistance", "maryland")]

    assert record["status"] == "ok"
    assert record["tier"] is None
    assert record["measure"] is None
    assert record["agreement"] is None


def test_runner_records_the_rank_correlation_with_the_jaxscape_map_it_is_compared_with(
    run_dir,
):
    cells = records(run_dir, "Fake map")

    assert set(cells) == {
        ("free_energy", "synthetic-8"),
        ("free_energy", "synthetic-16"),
        ("free_energy", "maryland"),
    }
    for record in cells.values():
        assert record["status"] == "ok"
        assert record["tier"] == 3
        assert record["measure"] == "Spearman's ρ"
        # The map is a monotone function of JAXScape's map, apart from a pixel
        # off its graph.
        assert record["agreement"] == pytest.approx(1)


def test_runner_records_no_rank_correlation_without_a_jaxscape_map(run_dir):
    record = records(run_dir, "Fake map without JAXScape")[
        ("resistance", "synthetic-8")
    ]

    assert record["status"] == "ok"
    assert record["measure"] == "Spearman's ρ"
    assert record["agreement"] is None


def test_runner_keeps_the_maryland_map_of_a_tool_with_a_tier(run_dir):
    kept = np.load(
        run_dir / "maps" / "E2-Fake map-free_energy-cpu-float64-maryland.npy"
    )

    # The fake map of Maryland, NaN on its first pixel.
    assert kept.shape == (20, 24)
    assert np.isnan(kept[0, 0])


def test_runner_records_why_a_cell_failed_or_was_skipped(run_dir):
    cells = records(run_dir)

    assert (
        "fake failure on a raster of side 16"
        in cells[("lcp", "synthetic-16")]["reason"]
    )
    assert cells[("lcp", "synthetic-32")]["reason"] == "after failed at synthetic-16"


def test_runner_snapshots_the_environment(run_dir):
    environment = json.loads((run_dir / "environment.json").read_text())

    assert environment["cores"] == CORES
    assert environment["threads"]["OMP_NUM_THREADS"] == "2"
    assert environment["packages"]["jaxscape"] == "0.0.16"
    assert environment["packages"]["Circuitscape"] == "6.1.0"
    assert environment["packages"]["ConScape"] == "0.3.0"
    assert environment["packages"]["Omniscape"] == "0.6.4"
    assert environment["packages"]["R"] == "4.1.2"
    assert environment["packages"]["gdistance"] == "1.6.5"
    assert environment["packages"]["samc"] == "4.2.1"
    assert environment["packages"]["ecoscape-connectivity"] == "0.2.5"
    assert environment["packages"]["eco-connectivity"] == "2.5.0"
    assert environment["packages"]["Graphab"] == "3.0.8"
    assert environment["packages"]["Java"] == "21.0.13"


def test_restart_skips_the_recorded_cells(run_dir, tmp_path):
    before = (run_dir / "records.jsonl").read_text()

    run(tmp_path, run_dir)

    assert (run_dir / "records.jsonl").read_text() == before


def test_smoke_mode_exits_with_an_error_when_a_cell_fails(tmp_path):
    result = run(tmp_path, tmp_path / "smoke", "--smoke")

    assert result.returncode != 0
    assert {case for _, case in records(tmp_path / "smoke")} == {"synthetic-16"}
    cells = {
        case: record["cells"]
        for (_, case), record in records(tmp_path / "smoke", "Fake E2").items()
    }
    assert cells == {"synthetic-16": 256, "maryland": 256}


def test_smoke_mode_passes_a_solver_that_fails_a_check_of_jaxscape(tmp_path):
    config = (
        CONFIG.split("[[tools]]")[0]
        + """
[[tools]]
tool = "Fake E2"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]

[tools.distances.resistance]

[solver_view]
tool = "Fake solvers"
adapter = "tests/fake_adapter.py"
device = "gpu"
dtypes = ["float32"]
reference_solver = "Direct"

[solver_view.questions.E2]
resistance = { sides = [8] }

[solver_view.solvers]
Direct = {}
Unresolved = { check_from = 16 }
"""
    )

    result = run(tmp_path, tmp_path / "smoke", "--smoke", config_text=config)

    assert result.returncode == 0, result.stdout
    unresolved = solver_records(tmp_path / "smoke")[
        ("E2", "resistance", "float32", "Unresolved", "synthetic-16")
    ]
    assert unresolved["status"] == "failed"
    assert "EquinoxRuntimeError" in unresolved["reason"]
