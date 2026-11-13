"""Run the benchmark: every cell of the configuration, each in its own process.

    uv run python run.py RUN_DIR            # run, or resume, the benchmark
    uv run python run.py RUN_DIR --smoke    # every cell once, on a small raster

A cell is one question, tool, distance, device and case, in the comparison or in
the solver view, which runs JAXScape with each solver. The runner writes the
configuration of the run to `RUN_DIR/config.json` and the environment snapshot
to `RUN_DIR/environment.json`, appends the record of each cell to
`RUN_DIR/records.csv` as it finishes, and skips the cells recorded there when
it restarts, which it does only with the configuration and environment of the
run. It deletes the work files of each cell when the cell ends, and keeps in
`RUN_DIR/maps` the maps of JAXScape's GPU cells of the connectivity and
sensitivity analyses in the comparison, with which it compares the other tools'
maps, and the other tools' maps of Maryland, for the report's Maryland figure.
Smoke mode runs every cell on a raster of `smoke_side` pixels a side, with
windows of `smoke_dependency_range` and one timed call, into a new run
directory, and exits with an error unless every cell is `ok` with its
agreement, and every Tier 1 error is at most `smoke_tolerance`.
"""

import argparse
import copy
import csv
import json
import math
import os
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from datetime import datetime, UTC
from importlib.metadata import version
from pathlib import Path

import numpy as np
from landscape import maryland, sites, synthetic
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
# The fields of a cell record, the contract between the runner and the report,
# and the type of their values; the first ones, up to `case`, are the key of
# the cell. `times_s` holds the times of the timed calls.
KEY = ("view", "question", "tool", "distance", "device", "dtype", "solver", "case")
FIELDS = {
    **dict.fromkeys(KEY, str),
    **{"cells": int, "status": str, "reason": str, "warmup_s": float},
    **{"times_s": list, "median_s": float, "measure": str, "agreement": float},
    **{"tier": int, "compared_with": str, "total": float, "version": str},
    **{"files": bool, "adapter": str},
}
INTERPRETERS = {".jl": ["julia", f"--project={HERE / 'julia'}"], ".R": ["Rscript"]}
# Out-of-memory errors; cuDSS's status 2 is `CUDSS_STATUS_ALLOC_FAILED`.
OOM = re.compile(
    r"RESOURCE_EXHAUSTED|out of memory|OutOfMemoryError|MemoryError|bad_alloc|"
    r"cannot allocate|cuDSS call failed \(status 2\)",
    re.IGNORECASE,
)
ERROR = re.compile(r"^([\w.]*(Error|Exception)\b|ERROR:)")


def main():
    """Run the benchmark from the command line, as the module docstring shows."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--config", type=Path, default=HERE / "config.toml")
    arguments = parser.parse_args()
    config = tomllib.loads(arguments.config.read_text())
    records = arguments.run_dir / "records.csv"
    if arguments.smoke and records.exists():
        sys.exit(f"smoke mode needs a new run directory, and {records} exists")

    run(run_config(config, arguments.smoke), arguments.run_dir)

    if arguments.smoke:
        tolerance = config["run"]["smoke_tolerance"]
        failures = [
            record for record in read_records(records) if not passes(record, tolerance)
        ]
        for record in failures:
            print(
                f"smoke: {label(record)} is {record['status']}, "
                f"{record['measure']} {record['agreement']} "
                f"(Tier 1 tolerance {tolerance}): {record['reason']}"
            )
        sys.exit(1 if failures else 0)


def passes(record, tolerance):
    """Return whether a record passes smoke mode.

    A record passes when it is `ok` with the agreement of its measure, and when
    its Tier 1 error is at most `tolerance`.
    """
    if record["status"] != "ok":
        return False
    if record["measure"] is None:
        return True
    if record["agreement"] is None:
        return False
    return record["tier"] != 1 or record["agreement"] <= tolerance


def run_config(config, smoke):
    """Return the configuration of a run.

    **Arguments:**

    - `config`: The configuration, as `config.toml` gives it.
    - `smoke`: Whether the run is in smoke mode, which runs every distance, and
      every question of the solver view, on the raster of `smoke_side` in
      place of the synthetic rasters, takes the top-left square of Maryland of
      that side and windows of `smoke_dependency_range`, and makes one timed
      call.

    **Returns:**

    A copy of the configuration, with the smoke settings applied in smoke mode,
    and with `block_size`, the side of JAXScape's blocks, added to the windows:
    `dependency_range * coarsening_factor` rounded up to an odd integer.
    """
    config = copy.deepcopy(config)
    if smoke:
        side = config["run"]["smoke_side"]
        config["run"]["repeats"] = 1
        for parameters in config["distances"].values():
            parameters["sides"] = [side]
        view = config.get("solver_view", {})
        restricted = [
            *(s for d in view.get("questions", {}).values() for s in d.values()),
            *view.get("solvers", {}).values(),
        ]
        for settings in restricted:
            if "cases" in settings:
                smoke_cases = (
                    case if case == "maryland" else f"synthetic-{side}"
                    for case in settings["cases"]
                )
                settings["cases"] = list(dict.fromkeys(smoke_cases))
        if "maryland" in config:
            config["maryland"]["side"] = side
        if "windows" in config:
            config["windows"]["dependency_range"] = config["run"][
                "smoke_dependency_range"
            ]
    if "windows" in config:
        windows = config["windows"]
        block = math.ceil(windows["dependency_range"] * windows["coarsening_factor"])
        windows["block_size"] = block + 1 - block % 2
    return config


def run(config, run_dir):
    """Run every cell of a configuration that a run directory has no record of.

    Within a configuration, synthetic rasters run from the smallest up, and
    after a failure, a timeout or running out of memory, the configuration
    skips its larger synthetic rasters. Maryland is outside this rule: it
    always runs, and its failure skips nothing.

    **Arguments:**

    - `config`: The configuration of the run (see `run_config`).
    - `run_dir`: The run directory, into which the run writes `config.json`
      and `environment.json` when they are missing, and appends to
      `records.csv`. When they exist, the run resumes only if they equal the
      configuration and the environment snapshot of this run, but for the
      date of the snapshot, and exits with their differences otherwise.
    """
    for directory in ("logs", "references", "maps"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    snapshots = {"config": config, "environment": environment(config)}
    differ = []
    for name, snapshot in snapshots.items():
        path = run_dir / f"{name}.json"
        if not path.exists():
            path.write_text(json.dumps(snapshot, indent=2) + "\n")
        saved, current = json.loads(path.read_text()), json.loads(json.dumps(snapshot))
        saved.pop("date", None)
        current.pop("date", None)
        differ += [f"{name}.json: {where}" for where in differences(saved, current)]
    if differ:
        sys.exit(
            f"{run_dir} holds a run of another configuration or environment, "
            "which this run does not resume (saved value, then value of this "
            "run):\n" + "\n".join(differ)
        )
    versions = snapshots["environment"]["tools"]
    records_path = run_dir / "records.csv"
    recorded = {}
    if records_path.exists():
        for record in read_records(records_path):
            recorded[tuple(record[field] for field in KEY)] = record
    # JAXScape runs first, so that its maps exist when other tools need them.
    ordered = sorted(configurations(config), key=lambda c: c["tool"] != "JAXScape")

    stopped_at = {}  # configuration key -> its failure on a synthetic raster
    shm = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(dir=shm, prefix="jaxscape-benchmark-") as tmp:
        for case in cases(config):
            case_dir = Path(tmp) / case["case"]
            for configuration in ordered:
                if not runs_on(configuration, case):
                    continue
                cell = {
                    **configuration,
                    "case": case["case"],
                    "cells": case["cells"],
                    "version": versions.get(configuration["tool"]),
                }
                reference_cases = configuration["reference_cases"]
                if reference_cases is not None and case["case"] not in reference_cases:
                    cell["measure"] = None  # no reference to measure against
                key = tuple(cell[field] for field in KEY)
                configuration_key = key[:-1]
                synthetic_case = case["side"] is not None
                if key in recorded:
                    record = recorded[key]
                else:
                    if synthetic_case and configuration_key in stopped_at:
                        failure = stopped_at[configuration_key]
                        reason = f"after {failure['status']} at {failure['case']}"
                        record = new_record(cell, "skipped", reason)
                    else:
                        if not case_dir.exists():
                            write_case(config, case, case_dir)
                        if cell["device"] == "gpu":
                            wait_for_gpu(config, cell)
                        record = run_cell(config, cell, case_dir, run_dir)
                    append_record(records_path, record)
                    outcome = record["median_s"] or record["reason"]
                    print(f"{label(record)}: {record['status']} {outcome}", flush=True)
                if synthetic_case and record["status"] in ("failed", "timeout", "oom"):
                    stopped_at[configuration_key] = record
            shutil.rmtree(case_dir, ignore_errors=True)


def differences(saved, current, path=""):
    """Return where two snapshots differ, such as `run.repeats: 5, 3`.

    Each difference names the key, joined by dots, the saved value and the
    current one.
    """
    if isinstance(saved, dict) and isinstance(current, dict):
        return [
            difference
            for key in sorted(saved.keys() | current.keys())
            for difference in differences(
                saved.get(key), current.get(key), f"{path}.{key}" if path else key
            )
        ]
    return [] if saved == current else [f"{path}: {saved!r}, {current!r}"]


def append_record(path, record):
    """Append a record to a records file, which starts with the header of `FIELDS`.

    A list of times is one value, its times joined by `;`, and a missing value
    is empty.
    """
    new = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(
            {
                field: ";".join(map(str, value)) if isinstance(value, list) else value
                for field, value in record.items()
            }
        )


def read_records(path):
    """Return the records of a records file (see `append_record`)."""

    def parse(kind, text):
        if text == "":
            return None
        if kind is list:
            return [float(time) for time in text.split(";")]
        return text == "True" if kind is bool else kind(text)

    with open(path, newline="") as file:
        return [
            {field: parse(FIELDS[field], text) for field, text in row.items()}
            for row in csv.DictReader(file)
        ]


def cases(config):
    """Return the cases of a run, from the fewest cells up.

    A case holds its `side`, or `None` for Maryland, and the distances and
    questions that run on it: a synthetic raster runs every question with the
    distances whose sides include it, and Maryland every distance with its
    questions.
    """
    sides = {
        distance: values["sides"] for distance, values in config["distances"].items()
    }
    questions = {entry["question"] for entry in config["tools"]}
    questions |= set(config.get("solver_view", {}).get("questions", {}))
    found = [
        {
            "case": f"synthetic-{side}",
            "side": side,
            "cells": side * side,
            "distances": {d for d, values in sides.items() if side in values},
            "questions": questions,
        }
        for side in {side for values in sides.values() for side in values}
    ]
    if "maryland" in config:
        quality, _ = rasters(config, None)
        found.append(
            {
                "case": "maryland",
                "side": None,
                "cells": quality.size,
                "distances": set(sides),
                "questions": set(config["maryland"]["questions"]),
            }
        )
    return sorted(found, key=lambda case: (case["cells"], case["case"]))


def rasters(config, side):
    """Return the quality and permeability rasters of a case.

    The case is the synthetic raster of `side` pixels a side, or Maryland for
    `None`.
    """
    if side is None:
        case = config["maryland"]
        return maryland(HERE / case["raster"], case["resistance"], case.get("side"))
    landscape = config["landscape"]
    return synthetic(
        side,
        landscape["correlation_length"],
        landscape["habitat_threshold"],
        landscape["seed"],
    )


def configurations(config):
    """Yield the configurations of both views: what a cell runs, but its case.

    The comparison gives one configuration per tool entry, distance, device and
    solver: a setting given as a table keyed by device takes its value for the
    device, and a list of solvers gives one configuration per solver. The solver
    view gives one per distance of each of its questions, dtype and solver, but
    the combinations of its `fails_by_design`: a solver's entry gives the label
    of its records and the settings that it adds to those of the distance, or
    replaces, its `questions` and `dtypes` restrict where it runs, and its
    `timeout` replaces that of the view. The reference of a connectivity or
    sensitivity analysis runs on the `cases` of the settings of its distance
    alone.
    """
    for entry in config["tools"]:
        for distance, settings in entry["distances"].items():
            for device in entry["devices"]:
                resolved = {
                    name: value[device] if isinstance(value, dict) else value
                    for name, value in settings.items()
                }
                solvers = resolved.get("solver", "")
                for solver in solvers if isinstance(solvers, list) else [solvers]:
                    with_solver = {**resolved, "solver": solver} if solver else resolved
                    yield configuration(
                        "comparison", entry, distance, device, solver, with_solver
                    )
    view = config.get("solver_view")
    if view is None:
        return
    for question, distances in view["questions"].items():
        for distance, shared in distances.items():
            for dtype in view["dtypes"]:
                for solver, added in view["solvers"].items():
                    added = dict(added)
                    combination = dict(
                        question=question, distance=distance, dtype=dtype, solver=solver
                    )
                    if (
                        question not in added.pop("questions", [question])
                        or dtype not in added.pop("dtypes", [dtype])
                        or fails_by_design(view, combination)
                    ):
                        continue
                    entry = {
                        **view,
                        "question": question,
                        "reference_cases": shared.get("cases"),
                        "timeout": added.pop("timeout", view.get("timeout")),
                    }
                    settings = {**shared, "dtype": dtype, "solver": solver, **added}
                    yield configuration(
                        "solver", entry, distance, view["device"], solver, settings
                    )


def configuration(view, entry, distance, device, solver, settings):
    """Return the configuration of a cell of a view, from its entry.

    **Arguments:**

    - `view`: `comparison` or `solver`.
    - `entry`: The tool entry of the comparison, or the solver view with its
      `question`.
    - `distance`, `device`: What the cell computes, and where. `distance` is
      the key of the settings' table, which labels the records; the cell
      computes under the settings' `distance`, by default this key.
    - `solver`: The label of the solver in the records, or `""` for none.
    - `settings`: The settings that the adapter reads; their `cases`, such as
      `["synthetic-256", "maryland"]`, restrict the cases on which the
      configuration runs.

    **Returns:**

    The configuration, with the fields of the record that do not depend on the
    case and what the runner needs to run its cells: `computed_with`, the
    distance under which it computes, for a tool compared by Spearman's ρ,
    `compared_with`, the label of the JAXScape map compared, and the
    `reference_cases` of the entry, the cases on which its reference runs, or
    `None` for every case.
    """
    settings = dict(settings)
    cases = settings.pop("cases", None)
    computed_with = settings.pop("distance", distance)
    tier = entry.get("tier")
    measured = view == "solver" or tier is not None
    measure = MEASURES[view, entry["question"]][0] if measured else None
    return {
        "view": view,
        "question": entry["question"],
        "tool": entry["tool"],
        "distance": distance,
        "device": device,
        "dtype": settings.get("dtype", "float64"),
        "solver": solver or None,
        "tier": tier,
        "measure": measure,
        "compared_with": (
            entry.get("compared_with", distance) if measure == RHO else None
        ),
        "computed_with": computed_with,
        "adapter": entry["adapter"],
        "settings": settings,
        "cases": cases,
        "reference_cases": entry.get("reference_cases"),
        "timeout": entry.get("timeout"),
    }


def fails_by_design(view, combination):
    """Return whether a combination of the solver view fails by design.

    It does when it matches an entry of the view's `fails_by_design`: each
    field of the entry holds the combination's value, or a list that holds it.
    """
    return any(
        all(
            combination[field] in (wanted if isinstance(wanted, list) else [wanted])
            for field, wanted in entry.items()
        )
        for entry in view.get("fails_by_design", [])
    )


def runs_on(configuration, case):
    """Return whether a configuration runs on a case.

    The distance under which it computes and its question run on the case, and
    a configuration with `cases` runs on these cases alone.
    """
    if configuration["computed_with"] not in case["distances"]:
        return False
    if configuration["question"] not in case["questions"]:
        return False
    return configuration["cases"] is None or case["case"] in configuration["cases"]


def label(cell):
    """Return the name of a cell in messages and file names.

    **Arguments:**

    - `cell`: A cell, or its record.

    **Returns:**

    The values of its key that are set, joined by hyphens.
    """
    return "-".join(str(cell[field]) for field in KEY if cell[field])


def write_case(config, case, case_dir):
    """Write the rasters of a case and the sites of the distance calculation."""
    quality, permeability = rasters(config, case["side"])
    case_dir.mkdir(parents=True)
    np.save(case_dir / "quality.npy", quality)
    np.save(case_dir / "permeability.npy", permeability)
    np.save(
        case_dir / "sites.npy",
        sites(quality, config["distance"]["sites"], config["distance"]["seed"]),
    )


def new_record(cell, status, reason):
    """Return the record of a cell with a status and its reason, but no output."""
    return {field: cell.get(field) for field in FIELDS} | {
        "status": status,
        "reason": reason,
    }


def run_cell(config, cell, case_dir, run_dir):
    """Run a cell, and return its record.

    A cell whose adapter succeeds but whose output cannot be measured is
    `failed`, with the error. The work files of the cell are deleted when it
    ends.
    """
    name = label(cell)
    work = case_dir.parent / "cells" / name
    try:
        status, reason = execute(
            config,
            cell["adapter"],
            cell["device"],
            cell_file(config, cell, case_dir, work),
            work,
            run_dir / "logs" / f"{name}.log",
            cell["timeout"],
        )
        if status != "ok":
            return new_record(cell, status, reason)
        try:
            return measured(config, cell, work, case_dir, run_dir)
        except Exception as error:
            return new_record(cell, "failed", f"{type(error).__name__}: {error}"[:300])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def measured(config, cell, work, case_dir, run_dir):
    """Return the record of a cell whose adapter succeeded, from its work files.

    A non-finite output is `failed`: an output of the distance calculation or
    of JAXScape with a value that is NaN or infinite, or the map of another
    tool without a finite pixel, since such a tool gives NaN to the pixels off
    its graph.
    """
    output = np.load(work / "output.npy")
    finite = np.isfinite(output)
    strict = cell["question"] == "distance" or cell["tool"] == "JAXScape"
    if not (finite.all() if strict else finite.any()):
        count = output.size - np.count_nonzero(finite)
        reason = f"{count} of {output.size} values are NaN or infinite"
        return new_record(cell, "failed", f"non-finite output: {reason}")
    record = new_record(cell, "ok", None)
    result = json.loads((work / "result.json").read_text())
    record.update(
        warmup_s=result["warmup_s"],
        times_s=result["times_s"],
        median_s=statistics.median(result["times_s"]),
        files=result["files"],
        total=result.get("total"),
    )
    jaxscape_gpu = cell["tool"] == "JAXScape" and cell["device"] == "gpu"
    windowed = cell["question"] != "distance"
    if cell["view"] == "comparison" and windowed and jaxscape_gpu:
        np.save(map_path(run_dir, cell), output)
    if cell["measure"] is None:
        return record
    if cell["measure"] == RHO:
        paired = {**cell, "tool": "JAXScape", "distance": cell["compared_with"]}
        path = map_path(run_dir, paired)
        reference = np.load(path) if path.exists() else None
        if cell["case"] == "maryland":  # for the Maryland figure of the report
            np.save(map_path(run_dir, cell), output)
    else:
        reference = reference_of(config, cell, case_dir, run_dir)
    if reference is not None:
        agreement = MEASURES[cell["view"], cell["question"]][1](output, reference)
        record["agreement"] = None if math.isnan(agreement) else agreement
    return record


def map_path(run_dir, cell):
    """Return where the runner keeps the map of a connectivity or sensitivity cell.

    **Arguments:**

    - `run_dir`: The run directory.
    - `cell`: A cell of the comparison, or its record.

    **Returns:**

    For JAXScape, the path of its map under the question, case and distance of
    the cell: the map of its GPU cell of the comparison, with which the maps of
    the other tools of that question and case paired with that distance are
    compared. For another tool, the path of its map under its label.
    """
    if cell["tool"] == "JAXScape":
        name = f"{cell['question']}-{cell['case']}-{cell['distance']}"
    else:
        name = label(cell)
    return run_dir / "maps" / f"{name}.npy"


def cell_file(config, cell, case_dir, work):
    """Return the cell file that an adapter reads (see `adapters/protocol.py`)."""
    parameters = config["distances"][cell["computed_with"]]
    return {
        "question": cell["question"],
        "distance": cell["computed_with"],
        "device": cell["device"],
        "settings": cell["settings"],
        "parameters": {name: v for name, v in parameters.items() if name != "sides"},
        "windows": config.get("windows", {}),
        "case": str(case_dir),
        "repeats": config["run"]["repeats"],
        "single_call_after": config["run"]["single_call_after"],
        "progress": str(work / "progress"),
        "output": str(work / "output.npy"),
        "result": str(work / "result.json"),
    }


def reference_of(config, cell, case_dir, run_dir):
    """Return the float64 reference of a cell.

    The reference is computed once per case and distance, and for the
    connectivity and sensitivity analyses per question. The distance
    calculation takes the `reference` adapter of its configuration on the CPU,
    and the analyses the adapter of the solver view on its device, in float64
    with its `reference_solver` and the settings of the distance in the
    question. A reference that failed is recorded next to where it would be and
    never retried; the function then returns `None`.
    """
    if cell["question"] == "distance":
        name = f"{cell['case']}-{cell['computed_with']}"
        adapter, device = config["distance"]["reference"], "cpu"
        settings = {"dtype": "float64"}
    else:
        name = f"{cell['case']}-{cell['question']}-{cell['computed_with']}"
        view = config["solver_view"]
        adapter, device = view["adapter"], view["device"]
        shared = view["questions"][cell["question"]][cell["computed_with"]]
        settings = {k: v for k, v in shared.items() if k != "cases"}
        settings.update(dtype="float64", solver=view["reference_solver"])
    path = run_dir / "references" / f"{name}.npy"
    failure = path.with_suffix(".txt")
    if path.exists():
        return np.load(path)
    if failure.exists():
        return None
    work = case_dir.parent / "references" / name
    reference = {**cell, "settings": settings, "device": device}
    contents = {**cell_file(config, reference, case_dir, work), "repeats": 0}
    log = run_dir / "logs" / f"reference-{name}.log"
    try:
        status, reason = execute(config, adapter, device, contents, work, log)
        if status != "ok":
            failure.write_text(f"{status}: {reason}\n")
            return None
        shutil.copy(work / "output.npy", path)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return np.load(path)


def largest_relative_error(output, reference):
    """Return the largest relative error over the pairs of sites."""
    pairs = np.triu_indices(len(reference), 1)
    error = np.abs(output[pairs] - reference[pairs]) / np.abs(reference[pairs])
    return float(np.max(error))


def relative_l2_error(output, reference):
    """Return the relative L2 error of a map against its reference."""
    return float(np.linalg.norm(output - reference) / np.linalg.norm(reference))


def rank_correlation(output, reference):
    """Return Spearman's ρ between two maps, over the pixels finite in both."""
    finite = np.isfinite(output) & np.isfinite(reference)
    return float(spearmanr(output[finite], reference[finite]).statistic)


RHO = "Spearman's ρ"
# The agreement measure of each view and question, and the function that
# measures it from an output and its reference. A measure that is undefined,
# NaN, is recorded as missing.
MEASURES = {
    ("comparison", "distance"): ("largest relative error", largest_relative_error),
    ("comparison", "connectivity"): (RHO, rank_correlation),
    ("comparison", "sensitivity"): (RHO, rank_correlation),
    ("solver", "distance"): ("largest relative error", largest_relative_error),
    ("solver", "connectivity"): ("relative L2 error", relative_l2_error),
    ("solver", "sensitivity"): ("relative L2 error", relative_l2_error),
}


def execute(config, adapter, device, contents, work, log, timeout=None):
    """Run an adapter on a cell file, and return the status and its reason.

    The adapter runs pinned to the cores of the configuration, and a call that
    runs longer than `timeout` seconds, by default that of the configuration,
    stops it.
    """
    work.mkdir(parents=True)
    (work / "cell.json").write_text(json.dumps(contents))
    path = Path(adapter)
    if path.suffix == ".py":
        interpreter = [sys.executable, "-m", ".".join(path.with_suffix("").parts)]
    else:
        interpreter = [*INTERPRETERS[path.suffix], str(HERE / path)]
    command = [
        "taskset",
        "-c",
        config["run"]["cores"],
        *interpreter,
        str(work / "cell.json"),
    ]
    timeout = timeout or config["run"]["timeout"]
    progress = work / "progress"
    with open(log, "w") as output:
        process = subprocess.Popen(
            command,
            cwd=HERE,
            env=variables(config, device),
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        start = time.time()
        while process.poll() is None:
            call_start = progress.stat().st_mtime if progress.exists() else start
            if time.time() - max(start, call_start) > timeout:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return "timeout", f"a call exceeded {timeout} s"
            time.sleep(0.2)
    if process.returncode == 0 and (work / "result.json").exists():
        return "ok", None
    lines = [line.strip() for line in log.read_text(errors="replace").splitlines()]
    lines = [line for line in lines if line]
    errors = [index for index, line in enumerate(lines) if ERROR.match(line)]
    # The last error, or the last line, with the line that follows an error
    # line ending in ":", such as the message of R's `Error in f(x) :`.
    reason = ""
    if lines:
        last = (errors or [len(lines) - 1])[-1]
        reason = " ".join(lines[last : last + 1 + lines[last].endswith(":")])[:300]
    reason = reason or f"exit status {process.returncode}"
    if process.returncode == -signal.SIGKILL or OOM.search(reason):
        return "oom", reason
    return "failed", reason


def threads(cores):
    """Return the thread settings matched to a `taskset` list of cores.

    XLA sizes its CPU thread pools to the cores that `taskset` leaves it.
    """
    names = (
        "OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS JULIA_NUM_THREADS "
        "RCPP_PARALLEL_NUM_THREADS"
    ).split()
    return {name: str(core_count(cores)) for name in names}


def core_count(cores):
    """Return the number of cores of a `taskset` list, such as 16 for `80-95`."""
    count = 0
    for part in cores.split(","):
        first, _, last = part.partition("-")
        count += int(last or first) - int(first) + 1
    return count


def variables(config, device):
    """Return the environment variables of a cell's process.

    The process inherits the variables of the runner, but those of JAX and XLA,
    such as `JAX_ENABLE_X64`, which would change what a cell computes. GPU
    cells see the GPU of the configuration, by its index in the order of the
    PCI bus, as `nvidia-smi` numbers it. They allocate memory as they need it,
    and keep the CPU platform, on which Equinox raises the errors of its
    checks.
    """
    inherited = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("JAX_", "XLA_"))
    }
    env = {**inherited, **threads(config["run"]["cores"])}
    if device == "gpu":
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=str(config["run"]["gpu"]),
            JAX_PLATFORMS="cuda,cpu",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
    else:
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    return env


def wait_for_gpu(config, cell):
    """Wait for the GPU of the configuration to run no other compute process.

    The run stops, to resume later, when another process still computes on
    the GPU after a minute, since it would share the GPU with the cell. A
    configuration without a GPU waits for none.
    """
    if config["run"]["gpu"] == "":
        return
    query = [
        "nvidia-smi",
        f"--id={config['run']['gpu']}",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader",
    ]
    for _ in range(60):
        processes = subprocess.run(query, capture_output=True, text=True, check=True)
        if not processes.stdout.strip():
            return
        time.sleep(1)
    sys.exit(
        f"GPU {config['run']['gpu']} runs other processes (pid, memory): "
        f"{'; '.join(processes.stdout.strip().splitlines())}. The run stops before "
        f"{label(cell)}, and resumes from it when restarted."
    )


def environment(config):
    """Return the snapshot of the run's hardware, threads and versions.

    The GPU is the model of that of the configuration, with its driver, the
    CUDA version of the driver and its memory, or `None` without one. `cores`
    is the number of cores of the configuration. The versions are those of the
    operating system, the runtimes, Julia and R as their interpreters print
    them, and each tool of the configuration (see `tool_versions`).
    `benchmark` is the commit of this directory, and whether its tracked files
    outside `results` differ from it.
    """
    gpu = None
    if config["run"]["gpu"] != "":
        query = [
            "nvidia-smi",
            f"--id={config['run']['gpu']}",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        values = subprocess.run(query, capture_output=True, text=True, check=True)
        gpu = dict(zip(("name", "driver", "memory"), values.stdout.strip().split(", ")))
        header = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        gpu["cuda"] = re.search(r"CUDA Version: (\S+)", header).group(1)
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    julia = tomllib.loads((HERE / "julia" / "Manifest.toml").read_text())
    renv = json.loads((HERE / "renv.lock").read_text())
    lines = (HERE / "java" / "jre" / "release").read_text().splitlines()
    java = dict(line.split("=", 1) for line in lines if "=" in line)

    def printed(*command):
        output = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
        return output.stdout + output.stderr  # Rscript prints its version to stderr

    def interpreter(command):
        return re.search(r"version (\S+)", printed(command, "--version")).group(1)

    changes = printed(
        "git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":!results"
    )
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "benchmark": {
            "commit": printed("git", "rev-parse", "HEAD").strip(),
            "modified": bool(changes.strip()),
        },
        "gpu": gpu,
        "cpu": cpu,
        "cores": core_count(config["run"]["cores"]),
        "threads": threads(config["run"]["cores"]),
        "os": platform.freedesktop_os_release()["PRETTY_NAME"],
        "runtimes": {
            "Python": platform.python_version(),
            "JAX": version("jax"),
            "jaxlib": version("jaxlib"),
            "PyTorch": version("torch"),
            "Julia": interpreter("julia"),
            "R": interpreter("Rscript"),
            "Java": java["JAVA_VERSION"].strip('"'),
        },
        "tools": tool_versions(config, julia, renv),
    }


def tool_versions(config, julia, renv):
    """Return the version of each tool of the configuration, from its pins.

    **Arguments:**

    - `config`: The configuration, whose tool entries and solver view name the
      tools. An entry's `package`, by default its tool's name without `.jl`,
      names the tool in the pins of its adapter's language.
    - `julia`: `julia/Manifest.toml`, which pins the tools of Julia adapters.
    - `renv`: `renv.lock`, which pins the tools of R adapters.

    **Returns:**

    The version of each tool. The tool of a Python adapter is a distribution of
    this project, or a jar of `java/` named `<package>-<version>.jar`, which
    install.sh pins.
    """
    jars = dict(jar.stem.rsplit("-", 1) for jar in (HERE / "java").glob("*.jar"))

    def pinned(entry):
        package = entry.get("package", entry["tool"].removesuffix(".jl"))
        language = Path(entry["adapter"]).suffix
        if language == ".jl":
            return julia["deps"][package][0]["version"]
        if language == ".R":
            return renv["Packages"][package]["Version"]
        return jars[package] if package in jars else version(package)

    entries = config["tools"] + (
        [config["solver_view"]] if "solver_view" in config else []
    )
    return {entry["tool"]: pinned(entry) for entry in entries}


if __name__ == "__main__":
    main()
