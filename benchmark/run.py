"""Run the benchmark: every cell of the configuration, each in its own process.

    uv run python run.py RUN_DIR            # run, or resume, the benchmark
    uv run python run.py RUN_DIR --smoke    # every cell once, on a small raster

A cell is one question, tool, distance, device and case, in the comparison or in
the solver view, which runs JAXScape with each solver. The runner writes the
configuration of the run to `RUN_DIR/config.json` and the environment snapshot
to `RUN_DIR/environment.json`, appends the record of each cell to
`RUN_DIR/records.jsonl` as it finishes, and skips the cells recorded there when
it restarts. It keeps in `RUN_DIR/maps` the maps of JAXScape's E2 and E3 cells
of the comparison on the GPU, with which it compares the other tools' maps, and
the other tools' maps of Maryland, for the report's Maryland figure. Smoke mode
runs every cell on a raster of `smoke_side` pixels a side, with windows of
`smoke_dependency_range` and one timed call, into a new run directory, and
exits with an error unless every cell is `ok` with its agreement, and every
Tier 1 error is at most `smoke_tolerance`.
"""

import argparse
import copy
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
# The fields of a cell record, the contract between the runner and the report;
# the first ones, up to `case`, are the key of the cell.
KEY = ("view", "question", "tool", "distance", "device", "dtype", "solver", "case")
FIELDS = (
    *KEY,
    *("cells", "status", "reason", "warmup_s", "times_s", "median_s"),
    *("measure", "agreement", "tier", "compared_with", "total", "version"),
    *("files", "adapter"),
)
INTERPRETERS = {".jl": ["julia", f"--project={HERE / 'julia'}"], ".R": ["Rscript"]}
OOM = re.compile(
    r"RESOURCE_EXHAUSTED|out of memory|OutOfMemoryError|MemoryError|bad_alloc|"
    r"cannot allocate",
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
    records = arguments.run_dir / "records.jsonl"
    if arguments.smoke and records.exists():
        sys.exit(f"smoke mode needs a new run directory, and {records} exists")

    run(run_config(config, arguments.smoke), arguments.run_dir)

    if arguments.smoke:
        tolerance = config["run"]["smoke_tolerance"]
        failures = [
            record
            for record in map(json.loads, records.read_text().splitlines())
            if not passes(record, tolerance)
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
      every question of the solver view, on the raster of `smoke_side`, takes
      the top-left square of Maryland of that side and windows of
      `smoke_dependency_range`, and makes one timed call.

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
        for distances in config.get("solver_view", {}).get("questions", {}).values():
            for settings in distances.values():
                if "sides" in settings:
                    settings["sides"] = [side]
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
      `records.jsonl`.
    """
    for directory in ("logs", "references", "maps"):
        (run_dir / directory).mkdir(parents=True, exist_ok=True)
    snapshots = {"config": lambda: config, "environment": lambda: environment(config)}
    for name, snapshot in snapshots.items():
        path = run_dir / f"{name}.json"
        if not path.exists():
            path.write_text(json.dumps(snapshot(), indent=2) + "\n")
    versions = json.loads((run_dir / "environment.json").read_text())["tools"]
    records_path = run_dir / "records.jsonl"
    recorded = {}
    if records_path.exists():
        for record in map(json.loads, records_path.read_text().splitlines()):
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
                        record = run_cell(config, cell, case_dir, run_dir)
                    with open(records_path, "a") as file:
                        file.write(json.dumps(record) + "\n")
                    outcome = record["median_s"] or record["reason"]
                    print(f"{label(record)}: {record['status']} {outcome}", flush=True)
                if synthetic_case and record["status"] in ("failed", "timeout", "oom"):
                    stopped_at[configuration_key] = record
            shutil.rmtree(case_dir, ignore_errors=True)


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
    of its records and the settings that it adds to those of the distance, and
    its `questions` and `dtypes` restrict where it runs.
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
                    settings = {**shared, "dtype": dtype, "solver": solver, **added}
                    entry = {**view, "question": question}
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
    - `solver`: The label of the solver in the records, or `""`.
    - `settings`: The settings that the adapter reads; their `sides` restrict
      the synthetic rasters on which the configuration runs.

    **Returns:**

    The configuration, with the fields of the record that do not depend on the
    case and what the runner needs to run its cells: `computed_with`, the
    distance under which it computes, and, for a tool compared by Spearman's
    ρ, `compared_with`, the label of the JAXScape map compared.
    """
    settings = dict(settings)
    sides = settings.pop("sides", None)
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
        "solver": solver,
        "tier": tier,
        "measure": measure,
        "compared_with": (
            entry.get("compared_with", distance) if measure == RHO else None
        ),
        "computed_with": computed_with,
        "adapter": entry["adapter"],
        "settings": settings,
        "sides": sides,
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
    a configuration with `sides` runs on the synthetic rasters of these sides
    alone.
    """
    if configuration["computed_with"] not in case["distances"]:
        return False
    if configuration["question"] not in case["questions"]:
        return False
    return configuration["sides"] is None or case["side"] in configuration["sides"]


def label(cell):
    """Return the name of a cell in messages and file names.

    **Arguments:**

    - `cell`: A cell, or its record.

    **Returns:**

    The values of its key that are set, joined by hyphens.
    """
    return "-".join(str(cell[field]) for field in KEY if cell[field])


def write_case(config, case, case_dir):
    """Write the rasters of a case and the sites of E1 to a directory."""
    quality, permeability = rasters(config, case["side"])
    case_dir.mkdir(parents=True)
    np.save(case_dir / "quality.npy", quality)
    np.save(case_dir / "permeability.npy", permeability)
    np.save(
        case_dir / "sites.npy",
        sites(quality, config["E1"]["sites"], config["E1"]["seed"]),
    )


def new_record(cell, status, reason):
    """Return the record of a cell with a status and its reason, but no output."""
    return {field: cell.get(field) for field in FIELDS} | {
        "status": status,
        "reason": reason,
    }


def run_cell(config, cell, case_dir, run_dir):
    """Run a cell, and return its record."""
    name = label(cell)
    work = case_dir.parent / "cells" / name
    status, reason = execute(
        config,
        cell["adapter"],
        cell["device"],
        cell_file(config, cell, case_dir, work),
        work,
        run_dir / "logs" / f"{name}.log",
        cell["timeout"],
    )
    record = new_record(cell, status, reason)
    if status != "ok":
        return record
    result = json.loads((work / "result.json").read_text())
    record.update(
        warmup_s=result["warmup_s"],
        times_s=result["times_s"],
        median_s=statistics.median(result["times_s"]),
        files=result["files"],
        total=result.get("total"),
    )
    output = np.load(work / "output.npy")
    jaxscape_gpu = cell["tool"] == "JAXScape" and cell["device"] == "gpu"
    if cell["view"] == "comparison" and cell["question"] != "E1" and jaxscape_gpu:
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
    """Return where the runner keeps the map of a cell of E2 or E3.

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

    The reference is computed once per case and distance, and for E2 and E3 per
    question. E1 takes the `reference` adapter of its configuration on the CPU.
    E2 and E3 take the adapter of the solver view on its device, in float64
    with its `reference_solver` and the settings of the distance in the
    question. A reference that failed is recorded next to where it would be and
    never retried; the function then returns `None`.
    """
    if cell["question"] == "E1":
        name = f"{cell['case']}-{cell['computed_with']}"
        adapter, device = config["E1"]["reference"], "cpu"
        settings = {"dtype": "float64"}
    else:
        name = f"{cell['case']}-{cell['question']}-{cell['computed_with']}"
        view = config["solver_view"]
        adapter, device = view["adapter"], view["device"]
        shared = view["questions"][cell["question"]][cell["computed_with"]]
        settings = {k: v for k, v in shared.items() if k != "sides"}
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
    status, reason = execute(config, adapter, device, contents, work, log)
    if status != "ok":
        failure.write_text(f"{status}: {reason}\n")
        return None
    shutil.copy(work / "output.npy", path)
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
    ("comparison", "E1"): ("largest relative error", largest_relative_error),
    ("comparison", "E2"): (RHO, rank_correlation),
    ("comparison", "E3"): (RHO, rank_correlation),
    ("solver", "E1"): ("largest relative error", largest_relative_error),
    ("solver", "E2"): ("relative L2 error", relative_l2_error),
    ("solver", "E3"): ("relative L2 error", relative_l2_error),
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
    errors = [line for line in lines if ERROR.match(line)]
    reason = (errors or [line for line in lines if line] or [""])[-1][:300]
    reason = reason or f"exit status {process.returncode}"
    if process.returncode == -signal.SIGKILL or OOM.search(reason):
        return "oom", reason
    return "failed", reason


def threads(cores):
    """Return the thread settings matched to a `taskset` list of cores.

    XLA sizes its CPU thread pools to the cores that `taskset` leaves it.
    """
    count = 0
    for part in cores.split(","):
        first, _, last = part.partition("-")
        count += int(last or first) - int(first) + 1
    names = (
        "OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS JULIA_NUM_THREADS "
        "RCPP_PARALLEL_NUM_THREADS"
    ).split()
    return {name: str(count) for name in names}


def variables(config, device):
    """Return the environment variables of a cell's process.

    GPU cells see the GPU of the configuration, by its index in the order of
    the PCI bus, as `nvidia-smi` numbers it. They allocate memory as they need
    it, so that JAX never fails to preallocate on a GPU that another process
    uses, and keep the CPU platform, on which Equinox raises the errors of its
    checks.
    """
    env = {**os.environ, **threads(config["run"]["cores"])}
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


def environment(config):
    """Return the snapshot of the run's hardware, threads and versions.

    The GPU is that of the configuration, with its index and UUID, or `None`
    without one. The versions are those of the runtimes, and of each tool of
    the configuration (see `tool_versions`).
    """
    gpu = None
    if config["run"]["gpu"] != "":
        query = [
            "nvidia-smi",
            f"--id={config['run']['gpu']}",
            "--query-gpu=index,uuid,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        values = subprocess.run(query, capture_output=True, text=True, check=True)
        names = ("index", "uuid", "name", "driver", "memory")
        gpu = dict(zip(names, values.stdout.strip().split(", ")))
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    julia = tomllib.loads((HERE / "julia" / "Manifest.toml").read_text())
    renv = json.loads((HERE / "renv.lock").read_text())
    lines = (HERE / "java" / "jre" / "release").read_text().splitlines()
    java = dict(line.split("=", 1) for line in lines if "=" in line)
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": gpu,
        "cpu": cpu,
        "cores": config["run"]["cores"],
        "threads": threads(config["run"]["cores"]),
        "runtimes": {
            "Python": platform.python_version(),
            "JAX": version("jax"),
            "jaxlib": version("jaxlib"),
            "PyTorch": version("torch"),
            "Julia": julia["julia_version"],
            "R": renv["R"]["Version"],
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
