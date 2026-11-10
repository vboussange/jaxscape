"""Run the benchmark: every cell of the configuration, each in its own process.

    uv run python run.py RUN_DIR            # run, or resume, the benchmark
    uv run python run.py RUN_DIR --smoke    # every cell once, on a small raster

A cell is one question, tool, distance, device and case, in the comparison or in
the solver view, which runs JAXScape with each solver. The runner appends the
record of each cell to `RUN_DIR/records.jsonl` as it finishes, and skips the
cells recorded there when it restarts. It keeps the maps of JAXScape's E2 and
E3 cells of the comparison in `RUN_DIR/maps`, against which it measures the
other tools' maps, and the other tools' maps of Maryland, for the report's
Maryland figure. Smoke mode runs every cell on a raster of `smoke_side` pixels a
side, with windows of `smoke_dependency_range` and one timed call, into a new
run directory, and exits with an error when a cell fails or lacks its
agreement; a cell of the solver view may fail a check of JAXScape.
"""

import argparse
import json
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
KEY = ("view", "question", "tool", "distance", "device", "dtype", "solver", "case")
MEASURES = {
    "E1": "largest relative error",
    "E2": "Spearman's ρ",
    "E3": "Spearman's ρ",
}
SOLVER_MEASURES = {
    "E1": "largest relative error",
    "E2": "relative L2 error",
    "E3": "relative L2 error",
}
REFERENCE = "adapters/reference.py"
INTERPRETERS = {".jl": ["julia", f"--project={HERE / 'julia'}"], ".R": ["Rscript"]}
OOM = re.compile(
    r"RESOURCE_EXHAUSTED|out of memory|OutOfMemoryError|MemoryError|bad_alloc|"
    r"cannot allocate",
    re.IGNORECASE,
)
ERROR = re.compile(r"^([\w.]*(Error|Exception)\b|ERROR:)")
CHECK = re.compile(r"^(equinox\.)?EquinoxRuntimeError\b")  # a check of JAXScape
R_PACKAGES = ["gdistance", "samc"]  # the R tools, whose versions each run records


def main():
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

    run(config, arguments.run_dir, arguments.smoke)

    if arguments.smoke:
        incomplete = [
            record
            for record in map(json.loads, records.read_text().splitlines())
            if not passes(record)
        ]
        for record in incomplete:
            print(
                f"smoke: {cell_name(record)} is {record['status']}, "
                f"agreement {record['agreement']}: {record['reason']}"
            )
        sys.exit(1 if incomplete else 0)


def passes(record):
    """Return whether a record passes smoke mode: `ok` with its agreement, or,
    in the solver view, failed by a check of JAXScape, such as a solver's
    non-finite output, which is how the solver fares."""
    if record["status"] == "ok":
        return record["measure"] is None or record["agreement"] is not None
    return (
        record["view"] == "solver"
        and record["status"] == "failed"
        and bool(CHECK.match(record["reason"]))
    )


def run(config, run_dir, smoke):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    (run_dir / "references").mkdir(exist_ok=True)
    (run_dir / "maps").mkdir(exist_ok=True)
    if not (run_dir / "environment.json").exists():
        snapshot = json.dumps(environment(config), indent=2)
        (run_dir / "environment.json").write_text(snapshot + "\n")
    records_path = run_dir / "records.jsonl"
    recorded = {}
    if records_path.exists():
        for record in map(json.loads, records_path.read_text().splitlines()):
            recorded[tuple(record[field] for field in KEY)] = record

    stopped = {}  # configuration -> case of its failure, timeout or oom
    shm = "/dev/shm" if Path("/dev/shm").is_dir() else None
    with tempfile.TemporaryDirectory(dir=shm, prefix="jaxscape-benchmark-") as tmp:
        for case in cases(config, smoke):
            case_dir = Path(tmp) / case["case"]
            for configuration in configurations(config):
                if not (
                    configuration["distance"] in case["distances"]
                    and configuration["question"] in case["questions"]
                    and runs_on(configuration, case, smoke)
                ):
                    continue
                cell = {**configuration, "case": case["case"], "cells": case["cells"]}
                key = tuple(cell[field] for field in KEY)
                stop = key[:-1]
                if key in recorded:
                    record = recorded[key]
                else:
                    if stop in stopped:
                        record = blank(cell, "skipped", f"after {stopped[stop]}")
                    else:
                        if not case_dir.exists():
                            write_case(config, case["case"], case_dir, smoke)
                        record = run_cell(config, cell, case_dir, run_dir, smoke)
                    with open(records_path, "a") as file:
                        file.write(json.dumps(record) + "\n")
                    outcome = record["median_s"] or record["reason"]
                    print(
                        f"{cell_name(record)}: {record['status']} {outcome}", flush=True
                    )
                if record["status"] in ("failed", "timeout", "oom"):
                    stopped[stop] = f"{record['status']} at {record['case']}"
            shutil.rmtree(case_dir, ignore_errors=True)


def cases(config, smoke):
    """Return the cases of the run, from the fewest cells up.

    A case holds the distances and the questions that run on it: a synthetic
    raster runs every question with the distances whose sides include it, and
    the Maryland case every distance with its questions.
    """
    sides = {
        distance: [config["run"]["smoke_side"]] if smoke else parameters["sides"]
        for distance, parameters in config["distances"].items()
    }
    found = [
        {
            "case": f"synthetic-{side}",
            "cells": side * side,
            "distances": {d for d, values in sides.items() if side in values},
            "questions": {entry["question"] for entry in config["tools"]}
            | set(config.get("solver_view", {}).get("questions", {})),
        }
        for side in {side for values in sides.values() for side in values}
    ]
    if "maryland" in config:
        quality, _ = rasters(config, "maryland", smoke)
        found.append(
            {
                "case": "maryland",
                "cells": quality.size,
                "distances": set(sides),
                "questions": set(config["maryland"]["questions"]),
            }
        )
    return sorted(found, key=lambda case: (case["cells"], case["case"]))


def rasters(config, case, smoke):
    """Return the quality and permeability rasters of a case.

    Smoke mode takes the top-left square of Maryland of `smoke_side` pixels a
    side.
    """
    if case == "maryland":
        maryland_ = config["maryland"]
        side = config["run"]["smoke_side"] if smoke else None
        return maryland(HERE / maryland_["raster"], maryland_["resistance"], side)
    landscape = config["landscape"]
    return synthetic(
        int(case.removeprefix("synthetic-")),
        landscape["correlation_length"],
        landscape["habitat_threshold"],
        landscape["seed"],
    )


def configurations(config):
    """Yield one configuration per tool entry, distance, device and solver.

    A setting given as a table keyed by device takes its value for the device,
    and a list of solvers gives one configuration per solver.
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
                    yield {
                        "view": "comparison",
                        "question": entry["question"],
                        "tool": entry["tool"],
                        "distance": distance,
                        "device": device,
                        "dtype": resolved.get("dtype", "float64"),
                        "solver": solver,
                        "tier": entry.get("tier"),
                        "measure": MEASURES.get(entry["question"])
                        if entry.get("tier")
                        else None,
                        "compared_with": entry.get("compared_with", distance),
                        "adapter": entry["adapter"],
                        "settings": {**resolved, "solver": solver}
                        if solver
                        else resolved,
                    }
    yield from solver_configurations(config)


def solver_configurations(config):
    """Yield the configurations of the solver view: each solver in each dtype,
    for each distance of each question.

    A solver's entry gives the label of its records and the settings it adds
    to those of the distance; its `questions` and `dtypes` restrict where it
    runs.
    """
    view = config.get("solver_view")
    if view is None:
        return
    for question, distances in view["questions"].items():
        for distance, settings in distances.items():
            settings = dict(settings)
            sides = settings.pop("sides", None)
            for dtype in view["dtypes"]:
                for solver, added in view["solvers"].items():
                    added = dict(added)
                    if question not in added.pop("questions", [question]):
                        continue
                    if dtype not in added.pop("dtypes", [dtype]):
                        continue
                    yield {
                        "view": "solver",
                        "question": question,
                        "tool": view["tool"],
                        "distance": distance,
                        "device": view["device"],
                        "dtype": dtype,
                        "solver": solver,
                        "tier": None,
                        "measure": SOLVER_MEASURES[question],
                        "adapter": view["adapter"],
                        "settings": {**settings, "dtype": dtype, "solver": solver}
                        | added,
                        "sides": sides,
                        "timeout": view.get("timeout"),
                    }


def runs_on(configuration, case, smoke):
    """Return whether a configuration runs on a case.

    A configuration with `sides` runs on the synthetic rasters of these sides,
    or on the raster of smoke mode, instead of every raster of its distance
    and Maryland.
    """
    sides = configuration.get("sides")
    if sides is None:
        return True
    if case["case"] == "maryland":
        return False
    return smoke or int(case["case"].removeprefix("synthetic-")) in sides


def cell_name(cell):
    return " ".join(str(cell[field]) for field in KEY if cell[field])


def write_case(config, case, case_dir, smoke):
    """Write the quality and permeability rasters of a case, and the sites of
    E1."""
    quality, permeability = rasters(config, case, smoke)
    case_dir.mkdir(parents=True)
    np.save(case_dir / "quality.npy", quality)
    np.save(case_dir / "permeability.npy", permeability)
    np.save(
        case_dir / "sites.npy",
        sites(quality, config["E1"]["sites"], config["E1"]["seed"]),
    )


def blank(cell, status, reason):
    """Return the record of a cell without output."""
    return {
        **{field: cell[field] for field in KEY},
        "cells": cell["cells"],
        "status": status,
        "reason": reason,
        "warmup_s": None,
        "times_s": None,
        "median_s": None,
        "measure": cell["measure"],
        "agreement": None,
        "tier": cell["tier"],
        "version": None,
        "files": None,
        "adapter": cell["adapter"],
    }


def cell_id(cell):
    """Return the name of a cell's log, and of its map of Maryland: its fields,
    led by `solver` for a cell of the solver view."""
    fields = KEY if cell.get("view") == "solver" else KEY[1:]
    return "-".join(str(cell[field]) for field in fields if cell[field])


def run_cell(config, cell, case_dir, run_dir, smoke):
    name = cell_id(cell)
    work = case_dir.parent / "cells" / name
    log = run_dir / "logs" / f"{name}.log"
    cell_file = specification(config, cell, case_dir, work, smoke)
    timeout = cell.get("timeout")
    status, reason = execute(
        config, cell["adapter"], cell["device"], cell_file, work, log, timeout
    )
    record = blank(cell, status, reason)
    if status != "ok":
        return record
    result = json.loads((work / "result.json").read_text())
    record.update(
        warmup_s=result["warmup_s"],
        times_s=result["times_s"],
        median_s=statistics.median(result["times_s"]),
        version=result["version"],
        files=result["files"],
    )
    output = np.load(work / "output.npy")
    comparison = cell["view"] == "comparison"
    if comparison and cell["tool"] == "JAXScape" and cell["question"] != "E1":
        path = map_path(run_dir, cell, cell["distance"])
        if not path.exists():
            np.save(path, output)
    if cell["tier"] == 1 or not comparison:
        reference = reference_of(config, cell, case_dir, run_dir, smoke)
        if reference is not None:
            record["agreement"] = AGREEMENTS[cell["measure"]](output, reference)
    elif cell["tier"]:
        path = map_path(run_dir, cell, cell["compared_with"])
        if path.exists():
            record["agreement"] = rank_correlation(output, np.load(path))
        if cell["case"] == "maryland":  # for the Maryland figure of the report
            np.save(run_dir / "maps" / f"{name}.npy", output)
    return record


def map_path(run_dir, cell, distance):
    """Return where the JAXScape map of a question, case and distance is kept.

    The first JAXScape cell of the comparison to succeed, in the order of the
    configuration, keeps its map: the map that the tools of E2 and E3 compared
    with this distance are compared with.
    """
    return run_dir / "maps" / f"{cell['question']}-{cell['case']}-{distance}.npy"


def rank_correlation(output, reference):
    """Return Spearman's ρ between two maps, over the pixels finite in both."""
    finite = np.isfinite(output) & np.isfinite(reference)
    return float(spearmanr(output[finite], reference[finite]).statistic)


def specification(config, cell, case_dir, work, smoke):
    """Return the cell file that an adapter reads (see `adapters/protocol.py`)."""
    parameters = dict(config["distances"][cell["distance"]])
    parameters.pop("sides")
    windows = dict(config.get("windows", {}))
    if smoke and windows:
        windows["dependency_range"] = config["run"]["smoke_dependency_range"]
    return {
        "question": cell["question"],
        "distance": cell["distance"],
        "device": cell["device"],
        "settings": cell["settings"],
        "parameters": parameters,
        "windows": windows,
        "case": str(case_dir),
        "repeats": 1 if smoke else config["run"]["repeats"],
        "single_call_after": config["run"]["single_call_after"],
        "progress": str(work / "progress"),
        "output": str(work / "output.npy"),
        "result": str(work / "result.json"),
    }


def reference_of(config, cell, case_dir, run_dir, smoke):
    """Return the float64 reference of a cell, computed once per case and
    distance, and for E2 and E3 per question.

    E1 takes `REFERENCE` on the CPU. E2 and E3 take the adapter of the solver
    view on its device, in float64 with its `reference_solver` and the
    settings of the distance in the question. A reference that failed is
    recorded next to where it would be and never retried; the function then
    returns `None`.
    """
    if cell["question"] == "E1":
        name = f"{cell['case']}-{cell['distance']}"
        adapter, device, settings = REFERENCE, "cpu", {"dtype": "float64"}
    else:
        name = f"{cell['case']}-{cell['question']}-{cell['distance']}"
        view = config["solver_view"]
        adapter, device = view["adapter"], view["device"]
        settings = dict(view["questions"][cell["question"]][cell["distance"]])
        settings.pop("sides", None)
        settings.update(dtype="float64", solver=view["reference_solver"])
    path = run_dir / "references" / f"{name}.npy"
    failure = path.with_suffix(".txt")
    if path.exists():
        return np.load(path)
    if failure.exists():
        return None
    work = case_dir.parent / "references" / name
    reference = {**cell, "settings": settings, "device": device}
    specification_ = specification(config, reference, case_dir, work, smoke)
    specification_["repeats"] = 0
    log = run_dir / "logs" / f"reference-{name}.log"
    status, reason = execute(config, adapter, device, specification_, work, log)
    if status != "ok":
        failure.write_text(f"{status}: {reason}\n")
        return None
    shutil.copy(work / "output.npy", path)
    return np.load(path)


def largest_relative_error(output, reference):
    pairs = np.triu_indices(len(reference), 1)
    error = np.abs(output[pairs] - reference[pairs]) / np.abs(reference[pairs])
    return float(np.max(error))


def relative_l2_error(output, reference):
    return float(np.linalg.norm(output - reference) / np.linalg.norm(reference))


AGREEMENTS = {
    "largest relative error": largest_relative_error,
    "relative L2 error": relative_l2_error,
}


def execute(config, adapter, device, specification, work, log, timeout=None):
    """Run an adapter on a cell file, and return the status and its reason.

    The adapter runs pinned to the cores of the configuration, and a call that
    runs longer than `timeout` seconds, by default that of the configuration,
    stops it.
    """
    work.mkdir(parents=True)
    (work / "cell.json").write_text(json.dumps(specification))
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

    GPU cells allocate memory as they need it, so that JAX never fails to
    preallocate on a GPU that another process uses, and keep the CPU platform,
    on which Equinox raises the errors of their checks.
    """
    env = {**os.environ, **threads(config["run"]["cores"])}
    if device == "gpu":
        env.update(
            CUDA_VISIBLE_DEVICES=str(config["run"]["gpu"]),
            JAX_PLATFORMS="cuda,cpu",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
        )
    else:
        env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
    return env


def environment(config):
    """Return the snapshot of the run's hardware, threads and versions."""
    gpu = "none"
    if config["run"]["gpu"] != "" and shutil.which("nvidia-smi"):
        query = [
            "nvidia-smi",
            f"--id={config['run']['gpu']}",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
        name, driver, memory = (
            subprocess.run(query, capture_output=True, text=True, check=True)
            .stdout.strip()
            .split(", ")
        )
        gpu = f"{name}, driver {driver}, {memory}"
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    return {
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "host": platform.node(),
        "gpu": gpu,
        "cpu": cpu,
        "cores": config["run"]["cores"],
        "threads": threads(config["run"]["cores"]),
        "python": platform.python_version(),
        "packages": {
            **{
                name: version(name)
                for name in (
                    "jaxscape",
                    "jax",
                    "jaxlib",
                    "numpy",
                    "scipy",
                    "scikit-image",
                    "ecoscape-connectivity",
                    "torch",
                    "eco-connectivity",
                )
            },
            **julia_packages(),
            **r_packages(),
            **java_packages(),
        },
    }


def julia_packages():
    """Return the versions of Julia and of the packages of its project, as
    pinned by `julia/Manifest.toml`."""
    project = tomllib.loads((HERE / "julia" / "Project.toml").read_text())
    manifest = tomllib.loads((HERE / "julia" / "Manifest.toml").read_text())
    return {
        "Julia": manifest["julia_version"],
        **{name: manifest["deps"][name][0]["version"] for name in project["deps"]},
    }


def r_packages():
    """Return the versions of R and of `R_PACKAGES`, as pinned by `renv.lock`."""
    lock = json.loads((HERE / "renv.lock").read_text())
    return {
        "R": lock["R"]["Version"],
        **{name: lock["Packages"][name]["Version"] for name in R_PACKAGES},
    }


def java_packages():
    """Return the versions of Graphab and of its Java runtime, which install.sh
    installs in `java/`."""
    jar = next((HERE / "java").glob("graphab-*.jar"))
    lines = (HERE / "java" / "jre" / "release").read_text().splitlines()
    release = dict(line.split("=", 1) for line in lines if "=" in line)
    return {
        "Graphab": jar.stem.removeprefix("graphab-"),
        "Java": release["JAVA_VERSION"].strip('"'),
    }


if __name__ == "__main__":
    main()
