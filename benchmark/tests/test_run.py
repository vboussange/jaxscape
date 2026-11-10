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
"""
# A land cover of 20 x 24 pixels, of classes 1 (resistance 1), 2 (resistance 10)
# and 3 (a barrier).
LAND_COVER = np.resize(np.array([1, 1, 2, 3], np.uint8), (20, 24))


def run(tmp_path, *arguments):
    land_cover = tmp_path / "land_cover.tif"
    Image.fromarray(LAND_COVER).save(land_cover)
    config = tmp_path / "config.toml"
    config.write_text(CONFIG.replace("LAND_COVER", str(land_cover)))
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


def test_runner_records_no_agreement_for_a_tool_without_tier(run_dir):
    record = records(run_dir, "Fake E2")[("resistance", "maryland")]

    assert record["status"] == "ok"
    assert record["tier"] is None
    assert record["measure"] is None
    assert record["agreement"] is None


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
    assert environment["packages"]["R"] == "4.1.2"
    assert environment["packages"]["gdistance"] == "1.6.5"


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
