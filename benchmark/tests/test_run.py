import json
import os
from importlib.metadata import version

import numpy as np
import pytest
from PIL import Image


CORES = ",".join(str(core) for core in sorted(os.sched_getaffinity(0))[:2])
# The configuration of the tests. Its E1 reference is the fake tool, so that a
# fake tool's error is its `error`. Its tools of E2 come before JAXScape, whose
# CPU device comes before its GPU, and whose GPU map is the one compared.
CONFIG = f"""
[run]
timeout = 10
repeats = 3
single_call_after = 300
cores = "{CORES}"
gpu = ""
smoke_side = 16
smoke_dependency_range = 1
smoke_tolerance = 1e-3

[landscape]
correlation_length = 4
habitat_threshold = 0.7
seed = 1

[E1]
sites = 4
seed = 2
reference = "tests/fake_adapter.py"

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
package = "numpy"
question = "E1"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 1

[tools.distances.resistance]
dtype = "float64"
error = 1e-4

[tools.distances.lcp]
dtype = "float64"
fail_on = ["synthetic-16"]

[tools.distances.free_energy]
dtype = "float64"
sleep_on = ["synthetic-16"]

[[tools]]
tool = "Fake per device"
package = "numpy"
question = "E1"
adapter = "tests/fake_adapter.py"
devices = ["gpu", "cpu"]

[tools.distances.resistance]
solver = {{ gpu = ["Iterative", "Direct"], cpu = "Direct" }}

[[tools]]
tool = "Fake E2"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]

[tools.distances.resistance]
dtype = "float32"
total = 42.0

[[tools]]
tool = "Fake map"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 3
compared_with = "lcp"

[tools.distances.free_energy]
off_graph = true

[[tools]]
tool = "Fake map of a failed JAXScape cell"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 3

[tools.distances.resistance]

[[tools]]
tool = "Fake chain"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]

[tools.distances.lcp]
fail_on = ["synthetic-16"]

[[tools]]
tool = "Fake Maryland failure"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]

[tools.distances.lcp]
fail_on = ["maryland"]

[[tools]]
tool = "Fake map paired with a label"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 3
compared_with = "lcp, reversed"

[tools.distances.lcp]

[[tools]]
tool = "JAXScape"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu", "gpu"]

[tools.distances.lcp]
reversed = {{ cpu = true, gpu = false }}

[tools.distances.resistance]
fail_on = {{ cpu = [], gpu = ["synthetic-8", "maryland"] }}

[tools.distances."lcp, reversed"]
distance = "lcp"
reversed = true

[solver_view]
tool = "Fake solvers"
package = "scipy"
adapter = "tests/fake_adapter.py"
device = "gpu"
dtypes = ["float32", "float64"]
reference_solver = "Direct"
timeout = 2
fails_by_design = [
  {{ distance = "free_energy", dtype = "float32" }},
  {{ solver = "Iterative", question = ["E2"], dtype = "float64" }},
]

[solver_view.questions.E1]
resistance = {{}}
free_energy = {{}}

[solver_view.questions.E2]
resistance = {{ batch_size = 4, sides = [8] }}

[solver_view.solvers]
Direct = {{}}
Iterative = {{ error = 0.5 }}
Approximate = {{ error = 0.1, questions = ["E2"], dtypes = ["float32"] }}
Slow = {{ sleep_on = ["synthetic-8"], questions = ["E2"], dtypes = ["float64"] }}
"""
# A land cover of 20 x 24 pixels, of classes 1 (resistance 1), 2 (resistance 10)
# and 3 (a barrier).
LAND_COVER = np.resize(np.array([1, 1, 2, 3], np.uint8), (20, 24))
# A configuration that passes smoke mode: every cell `ok`, with its agreement.
PASSING = (
    CONFIG.split("[[tools]]")[0]
    + """
[[tools]]
tool = "Fake"
package = "numpy"
question = "E1"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 1

[tools.distances.resistance]
error = ERROR

[[tools]]
tool = "JAXScape"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["gpu"]

[tools.distances.lcp]

[[tools]]
tool = "Fake map"
package = "numpy"
question = "E2"
adapter = "tests/fake_adapter.py"
devices = ["cpu"]
tier = 3

[tools.distances.lcp]
nan = NAN
"""
)


@pytest.fixture(scope="session")
def run(script):
    """Return a function that runs the runner on a configuration in `tmp_path`."""

    def run(tmp_path, *arguments, config_text=CONFIG):
        land_cover = tmp_path / "land_cover.tif"
        Image.fromarray(LAND_COVER).save(land_cover)
        config = tmp_path / "config.toml"
        config.write_text(config_text.replace("LAND_COVER", str(land_cover)))
        return script("run.py", *arguments, "--config", config)

    return run


def records(run_dir, tool="Fake"):
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    return {
        (r["distance"], r["case"]): r
        for r in map(json.loads, lines)
        if r["tool"] == tool and r["view"] == "comparison"
    }


@pytest.fixture(scope="module")
def run_dir(run, tmp_path_factory):
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
    assert record["version"] == version("numpy")
    assert record["files"] is False
    assert record["tier"] == 1
    assert record["measure"] == "largest relative error"
    assert record["agreement"] == pytest.approx(1e-4)


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


def test_runner_records_the_landscape_total_that_a_tool_reports(run_dir):
    totals = {case: r["total"] for (_, case), r in records(run_dir, "Fake E2").items()}
    others = records(run_dir, "Fake map").values()

    assert totals == {"synthetic-8": 42.0, "maryland": 42.0}
    assert {record["total"] for record in others} == {None}


def test_maryland_runs_after_a_failure_and_its_failure_skips_nothing(run_dir):
    chain = {
        case: r["status"] for (_, case), r in records(run_dir, "Fake chain").items()
    }
    maryland = {
        case: r["status"]
        for (_, case), r in records(run_dir, "Fake Maryland failure").items()
    }

    assert chain == {
        "synthetic-8": "ok",
        "synthetic-16": "failed",
        "maryland": "ok",
        "synthetic-32": "skipped",
    }
    assert maryland == {
        "synthetic-8": "ok",
        "synthetic-16": "ok",
        "maryland": "failed",
        "synthetic-32": "ok",
    }


def solver_records(run_dir):
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    return {
        (r["question"], r["distance"], r["dtype"], r["solver"], r["case"]): r
        for r in map(json.loads, lines)
        if r["view"] == "solver"
    }


def test_solver_view_runs_each_solver_in_each_dtype_on_the_gpu(run_dir):
    cells = solver_records(run_dir)

    def each(question, distance, cases, dtypes, solvers=("Direct", "Iterative")):
        return {
            (question, distance, dtype, solver, case)
            for case in cases
            for dtype in dtypes
            for solver in solvers
        }

    # E1 runs on every raster of its distance, E2 on the sides of its entry,
    # and the combinations that fail by design are left out.
    assert set(cells) == (
        each("E1", "resistance", ["synthetic-8"], ["float32", "float64"])
        | each("E1", "free_energy", ["synthetic-8", "synthetic-16"], ["float64"])
        | each("E2", "resistance", ["synthetic-8"], ["float32"])
        | {("E2", "resistance", "float32", "Approximate", "synthetic-8")}
        | {("E2", "resistance", "float64", "Direct", "synthetic-8")}
        | {("E2", "resistance", "float64", "Slow", "synthetic-8")}
    )
    assert {(r["tool"], r["device"]) for r in cells.values()} == {
        ("Fake solvers", "gpu")
    }
    assert {r["version"] for r in cells.values()} == {version("scipy")}


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
    e1 = cells[("E1", "resistance", "float64", "Iterative", "synthetic-8")]
    assert e1["measure"] == "largest relative error"
    assert e1["agreement"] == pytest.approx(0.5)


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


def test_runner_compares_a_map_with_the_jaxscape_gpu_map_of_its_paired_distance(
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
        # The map is a monotone function of JAXScape's GPU map, apart from a
        # pixel off its graph, and reverses its CPU map.
        assert record["agreement"] == pytest.approx(1)


def test_a_table_labelled_with_its_distance_runs_as_a_configuration_of_its_own(
    run_dir,
):
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    labelled = {
        (r["device"], r["case"]): r["status"]
        for r in map(json.loads, lines)
        if (r["tool"], r["distance"]) == ("JAXScape", "lcp, reversed")
    }

    # It runs on the cases of the distance that it names, on each device.
    cases = ("synthetic-8", "synthetic-16", "synthetic-32", "maryland")
    assert labelled == {(d, case): "ok" for d in ("cpu", "gpu") for case in cases}


def test_runner_compares_a_map_with_the_jaxscape_gpu_map_of_the_label_it_names(
    run_dir,
):
    cells = records(run_dir, "Fake map paired with a label")

    assert len(cells) == 4
    for record in cells.values():
        assert record["compared_with"] == "lcp, reversed"
        # The map is the quality, which JAXScape's GPU map under the label
        # reverses, and its GPU map under least-cost distance equals.
        assert record["agreement"] == pytest.approx(-1)
    assert {r["compared_with"] for r in records(run_dir, "Fake map").values()} == {
        "lcp"
    }
    assert {r["compared_with"] for r in records(run_dir, "JAXScape").values()} == {None}


def test_runner_records_no_rank_correlation_when_the_jaxscape_gpu_cell_failed(
    run_dir,
):
    jaxscape = records(run_dir, "JAXScape")
    cells = records(run_dir, "Fake map of a failed JAXScape cell")

    assert jaxscape[("resistance", "synthetic-8")]["status"] == "failed"
    assert {(r["status"], r["agreement"]) for r in cells.values()} == {("ok", None)}


def test_runner_keeps_the_maryland_map_of_a_tool_with_a_tier(run_dir):
    [kept] = (run_dir / "maps").glob("*Fake map-free_energy-cpu-*-maryland.npy")
    map_ = np.load(kept)

    # The fake map of Maryland, NaN on its first pixel.
    assert map_.shape == (20, 24)
    assert np.isnan(map_[0, 0])


def test_runner_records_why_a_cell_failed_or_was_skipped(run_dir):
    cells = records(run_dir)

    assert "fake failure on synthetic-16" in cells[("lcp", "synthetic-16")]["reason"]
    assert cells[("lcp", "synthetic-32")]["reason"] == "after failed at synthetic-16"


def test_runner_snapshots_the_environment(run_dir):
    environment = json.loads((run_dir / "environment.json").read_text())

    assert environment["gpu"] is None
    assert environment["cores"] == CORES
    assert environment["threads"]["OMP_NUM_THREADS"] == "2"
    assert environment["runtimes"]["Julia"] == "1.13.3"
    assert environment["runtimes"]["R"] == "4.1.2"
    assert environment["runtimes"]["Java"] == "21.0.13"
    assert environment["tools"]["Fake"] == version("numpy")
    assert environment["tools"]["Fake solvers"] == version("scipy")


def test_runner_records_the_configuration_of_its_run(run_dir):
    config = json.loads((run_dir / "config.json").read_text())

    assert config["run"]["repeats"] == 3
    assert config["windows"] == {
        "dependency_range": 2,
        "coarsening_factor": 0.5,
        "block_size": 1,
    }


def test_restart_skips_the_recorded_cells(run, run_dir, tmp_path):
    before = (run_dir / "records.jsonl").read_text()

    run(tmp_path, run_dir)

    assert (run_dir / "records.jsonl").read_text() == before


def test_smoke_mode_exits_with_an_error_when_a_cell_fails(run, tmp_path):
    result = run(tmp_path, tmp_path / "smoke", "--smoke")

    assert result.returncode != 0
    assert {case for _, case in records(tmp_path / "smoke")} == {"synthetic-16"}
    cells = {
        case: record["cells"]
        for (_, case), record in records(tmp_path / "smoke", "Fake E2").items()
    }
    assert cells == {"synthetic-16": 256, "maryland": 256}


def test_smoke_mode_records_the_smoke_settings_as_the_configuration_of_its_run(
    run, tmp_path
):
    run(tmp_path, tmp_path / "smoke", "--smoke")
    config = json.loads((tmp_path / "smoke" / "config.json").read_text())

    assert config["run"]["repeats"] == 1
    assert config["windows"]["dependency_range"] == 1
    assert {d: v["sides"] for d, v in config["distances"].items()} == {
        "resistance": [16],
        "lcp": [16],
        "free_energy": [16],
    }
    assert config["solver_view"]["questions"]["E2"]["resistance"]["sides"] == [16]


def passing(error="1e-4", nan="false"):
    """Return the passing configuration with an E1 error and a NaN flag.

    `error` is the error of its E1 tool, and `nan` whether its map is NaN.
    """
    return PASSING.replace("ERROR", error).replace("NAN", nan)


def test_smoke_mode_passes_when_every_cell_is_ok_with_its_agreement(run, tmp_path):
    result = run(tmp_path, tmp_path / "smoke", "--smoke", config_text=passing())

    assert result.returncode == 0, result.stdout


def test_smoke_mode_exits_with_an_error_when_a_tier_1_error_exceeds_its_tolerance(
    run, tmp_path
):
    result = run(tmp_path, tmp_path / "smoke", "--smoke", config_text=passing("1e-2"))

    assert result.returncode != 0
    assert "E1" in result.stdout
    assert "tolerance 0.001" in result.stdout


def test_an_undefined_agreement_is_missing_and_fails_smoke_mode(run, tmp_path):
    config = PASSING.replace("error = ERROR", "nan = true").replace("NAN", "true")
    result = run(tmp_path, tmp_path / "smoke", "--smoke", config_text=config)

    assert result.returncode != 0
    agreements = {
        tool: record["agreement"]
        for tool in ("Fake", "Fake map")
        for record in records(tmp_path / "smoke", tool).values()
    }
    assert agreements == {"Fake": None, "Fake map": None}
