"""Cross-tool benchmark orchestration for JAXScape distance workloads.

This script intentionally avoids a large CLI surface. Edit the constants below or
run `benchmark/run_benchmarks.sh` to regenerate the benchmark artifacts that are
committed to the repository and visualised in the documentation.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from jaxscape import (
    EuclideanDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
)

try:
    import optimistix as optx
except ImportError:
    optx = None
    _optimistix_import_error = "optimistix is not installed"
else:
    _optimistix_import_error = None


BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
RESULTS_CSV = RESULTS_DIR / "benchmark_results.csv"
CIRCUITSCAPE_SCRIPT = BENCHMARK_DIR / "external" / "circuitscape_resistance.jl"
SAMC_SCRIPT = BENCHMARK_DIR / "external" / "samc_sensitivity.R"
RESISTANCE_GA_SCRIPT = BENCHMARK_DIR / "external" / "resistancega_inverse.R"
CONEFOR_SCRIPT = BENCHMARK_DIR / "external" / "conefor_runner.sh"
REPEATS = 3
CPU_DEVICE = jax.devices("cpu")[0]
MIN_PERMEABILITY = 1e-3


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    task: str
    raster: list[list[float]]
    points: list[tuple[int, int]]
    repeats: int = REPEATS


@dataclass(frozen=True)
class BenchmarkRecord:
    task: str
    tool: str
    scenario: str
    status: str
    median_seconds: float | None
    timings_seconds: list[float]
    note: str = ""


def create_landscape(seed: int = 0, size: int = 6) -> jax.Array:
    key = jr.PRNGKey(seed)
    base = jr.uniform(key, (size, size), minval=0.3, maxval=1.0)
    ridge = jnp.exp(-3 * jnp.linspace(-1.0, 1.0, size) ** 2)
    barrier = 0.4 * jnp.outer(ridge, jnp.ones(size))
    return jnp.clip(base - barrier, 0.15, 1.0)


def benchmark_cases() -> dict[str, BenchmarkCase]:
    raster = create_landscape()
    raster_as_lists = [[float(value) for value in row] for row in raster.tolist()]
    points = [(0, 0), (0, 5), (5, 0), (5, 5), (3, 3)]
    return {
        "resistance": BenchmarkCase(
            name="synthetic_resistance",
            task="resistance_distance",
            raster=raster_as_lists,
            points=points,
        ),
        "lcp": BenchmarkCase(
            name="synthetic_lcp",
            task="least_cost_path",
            raster=raster_as_lists,
            points=points,
        ),
        "sensitivity": BenchmarkCase(
            name="synthetic_sensitivity",
            task="sensitivity_analysis",
            raster=raster_as_lists,
            points=points,
        ),
        "inverse": BenchmarkCase(
            name="synthetic_inverse",
            task="inverse_landscape_genetics",
            raster=raster_as_lists,
            points=points,
        ),
    }


CASES = benchmark_cases()


def edge_weight(x: jax.Array, y: jax.Array) -> jax.Array:
    """Average adjacent node values to obtain an undirected edge weight."""
    return (x + y) / 2


def as_array(raster: list[list[float]]) -> jax.Array:
    return jnp.asarray(raster, dtype=jnp.float32)


def as_point_array(points: list[tuple[int, int]]) -> jax.Array:
    return jnp.asarray(points, dtype=jnp.int32)


def measure_runtime(fun, *args, repeats: int) -> tuple[list[float], Any]:
    result = fun(*args)
    jax.block_until_ready(result)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fun(*args)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)
    return timings, result


@eqx.filter_jit
def _jaxscape_resistance(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=edge_weight)
    return ResistanceDistance()(grid, nodes=nodes)


@eqx.filter_jit
def _jaxscape_lcp(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=edge_weight)
    return LCPDistance()(grid, nodes=nodes)


def run_jaxscape_resistance(case: BenchmarkCase) -> BenchmarkRecord:
    permeability = jax.device_put(as_array(case.raster), CPU_DEVICE)
    nodes = jax.device_put(as_point_array(case.points), CPU_DEVICE)
    timings, _ = measure_runtime(_jaxscape_resistance, permeability, nodes, repeats=case.repeats)
    return BenchmarkRecord(case.task, "JAXScape", case.name, "ok", statistics.median(timings), timings)


def run_jaxscape_lcp(case: BenchmarkCase) -> BenchmarkRecord:
    permeability = jax.device_put(as_array(case.raster), CPU_DEVICE)
    nodes = jax.device_put(as_point_array(case.points), CPU_DEVICE)
    timings, _ = measure_runtime(_jaxscape_lcp, permeability, nodes, repeats=case.repeats)
    return BenchmarkRecord(case.task, "JAXScape", case.name, "ok", statistics.median(timings), timings)


def _sensitivity_objective(quality_raster: jax.Array) -> jax.Array:
    distance = EuclideanDistance()
    grid = GridGraph(quality_raster, fun=edge_weight)
    quality = grid.array_to_node_values(quality_raster)
    dist = distance(grid)
    kernel = jnp.exp(-dist / 3.0)
    kernel = kernel.at[jnp.diag_indices_from(kernel)].set(0)
    return quality @ kernel @ quality.T


run_sensitivity_problem = eqx.filter_jit(eqx.filter_grad(_sensitivity_objective))


def run_jaxscape_sensitivity(case: BenchmarkCase) -> BenchmarkRecord:
    quality = jax.device_put(as_array(case.raster), CPU_DEVICE)
    timings, _ = measure_runtime(run_sensitivity_problem, quality, repeats=case.repeats)
    return BenchmarkRecord(case.task, "JAXScape", case.name, "ok", statistics.median(timings), timings)


def _sample_coordinates(size: int) -> jax.Array:
    candidates = jnp.array(
        [[0, 0], [0, size - 1], [size - 1, 0], [size - 1, size - 1], [size // 2, size // 2]],
        dtype=jnp.int32,
    )
    return jnp.unique(candidates, axis=0)


def _inverse_loss(logits: jax.Array, sample_coords: jax.Array, target_distances: jax.Array) -> jax.Array:
    permeability = jnn.sigmoid(logits) + jnp.array(MIN_PERMEABILITY, dtype=logits.dtype)
    grid = GridGraph(grid=permeability, fun=edge_weight)
    predicted = ResistanceDistance()(grid, nodes=sample_coords)
    return jnp.mean((predicted - target_distances) ** 2)


def run_jaxscape_inverse(case: BenchmarkCase) -> BenchmarkRecord:
    if optx is None:
        return BenchmarkRecord(case.task, "JAXScape + Optimistix", case.name, "skipped", None, [], _optimistix_import_error or "optimistix missing")

    landscape = jax.device_put(as_array(case.raster), CPU_DEVICE)
    sample_coords = jax.device_put(_sample_coordinates(landscape.shape[0]), CPU_DEVICE)
    target_distances = ResistanceDistance()(GridGraph(landscape, fun=edge_weight), nodes=sample_coords)
    solver = optx.LBFGS(rtol=1e-5, atol=1e-5)

    def objective(logits: jax.Array, args: tuple[jax.Array, jax.Array]) -> jax.Array:
        coords, targets = args
        return _inverse_loss(logits, coords, targets)

    def solve(problem_landscape: jax.Array) -> jax.Array:
        init_logits = jnp.zeros_like(problem_landscape)
        solution = optx.minimise(
            objective,
            solver,
            init_logits,
            args=(sample_coords, target_distances),
            max_steps=8,
            throw=False,
        )
        return _inverse_loss(solution.value, sample_coords, target_distances)

    timings, _ = measure_runtime(solve, landscape, repeats=case.repeats)
    return BenchmarkRecord(case.task, "JAXScape + Optimistix", case.name, "ok", statistics.median(timings), timings)


def ascii_grid_from_raster(raster: list[list[float]], nodata: float = -9999.0) -> str:
    rows = [" ".join(f"{value:.6f}" for value in row) for row in raster]
    return "\n".join(
        [
            f"ncols         {len(raster[0])}",
            f"nrows         {len(raster)}",
            "xllcorner     0",
            "yllcorner     0",
            "cellsize      1",
            f"NODATA_value  {nodata:g}",
            *rows,
        ]
    )


def ascii_points_from_points(shape: tuple[int, int], points: list[tuple[int, int]], nodata: int = -9999) -> str:
    grid = [[nodata for _ in range(shape[1])] for _ in range(shape[0])]
    for index, (i, j) in enumerate(points, start=1):
        grid[i][j] = index
    rows = [" ".join(str(value) for value in row) for row in grid]
    return "\n".join(
        [
            f"ncols         {shape[1]}",
            f"nrows         {shape[0]}",
            "xllcorner     0",
            "yllcorner     0",
            "cellsize      1",
            f"NODATA_value  {nodata}",
            *rows,
        ]
    )


def circuitscape_ini(cellmap: Path, points: Path, output_prefix: Path) -> str:
    return textwrap.dedent(
        f"""
        [Options for advanced mode]
        ground_file_is_resistances = False
        source_file = None
        remove_src_or_gnd = keepall
        ground_file = None
        use_unit_currents = False
        use_direct_grounds = False

        [Calculation options]
        low_memory_mode = False
        solver = cg+amg
        print_timings = False

        [Options for pairwise and one-to-all and all-to-one modes]
        included_pairs_file = None
        use_included_pairs = False
        point_file = {points}

        [Output options]
        write_cum_cur_map_only = False
        log_transform_maps = False
        output_file = {output_prefix}
        write_max_cur_maps = False
        write_volt_maps = False
        set_null_currents_to_nodata = False
        set_null_voltages_to_nodata = False
        compress_grids = False
        write_cur_maps = False

        [Short circuit regions (aka polygons)]
        use_polygons = False
        polygon_file = None

        [Connection scheme for raster habitat data]
        connect_four_neighbors_only = True
        connect_using_avg_resistances = True

        [Habitat raster or graph]
        habitat_file = {cellmap}
        habitat_map_is_resistances = True

        [Options for one-to-all and all-to-one modes]
        use_variable_source_strengths = False
        variable_source_file = None

        [Version]
        version = unknown

        [Mask file]
        use_mask = False
        mask_file = None

        [Circuitscape mode]
        data_type = raster
        scenario = pairwise
        """
    ).strip() + "\n"


def run_circuitscape_resistance(case: BenchmarkCase) -> BenchmarkRecord:
    if shutil.which("julia") is None:
        return BenchmarkRecord(case.task, "Circuitscape.jl", case.name, "skipped", None, [], "Julia is not available")

    with tempfile.TemporaryDirectory(prefix="jaxscape-circuitscape-") as tmpdir:
        workspace = Path(tmpdir)
        cellmap = workspace / "cellmap.asc"
        points = workspace / "points.asc"
        output_prefix = workspace / "result"
        ini = workspace / "config.ini"
        output = workspace / "circuitscape.json"
        cellmap.write_text(ascii_grid_from_raster(case.raster))
        points.write_text(ascii_points_from_points((len(case.raster), len(case.raster[0])), case.points))
        ini.write_text(circuitscape_ini(cellmap, points, output_prefix))
        subprocess.run(
            ["julia", str(CIRCUITSCAPE_SCRIPT), str(ini), str(case.repeats), str(output)],
            check=True,
        )
        payload = json.loads(output.read_text())
    return BenchmarkRecord(case.task, "Circuitscape.jl", case.name, payload["status"], payload.get("median_seconds"), payload.get("timings_seconds", []), payload.get("note", ""))


def skip_record(task: str, tool: str, scenario: str, note: str) -> BenchmarkRecord:
    return BenchmarkRecord(task, tool, scenario, "skipped", None, [], note)


def run_conefor_placeholder(case: BenchmarkCase, task_label: str) -> BenchmarkRecord:
    conefor_bin = os.environ.get("CONEFOR_BIN")
    if not conefor_bin:
        return skip_record(task_label, "Conefor", case.name, "Set CONEFOR_BIN to enable the Conefor adapter.")
    output = Path(tempfile.mkdtemp(prefix="jaxscape-conefor-")) / "conefor.json"
    subprocess.run([str(CONEFOR_SCRIPT), conefor_bin, task_label, str(output)], check=True)
    payload = json.loads(output.read_text())
    return BenchmarkRecord(task_label, "Conefor", case.name, payload["status"], payload.get("median_seconds"), payload.get("timings_seconds", []), payload.get("note", ""))


def run_samc_sensitivity(case: BenchmarkCase) -> BenchmarkRecord:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return skip_record(case.task, "samc", case.name, "Rscript is not installed in this environment.")
    output = Path(tempfile.mkdtemp(prefix="jaxscape-samc-")) / "samc.json"
    raster_json = Path(tempfile.mkdtemp(prefix="jaxscape-samc-input-")) / "raster.json"
    raster_json.write_text(json.dumps(case.raster))
    subprocess.run([rscript, str(SAMC_SCRIPT), str(raster_json), str(case.repeats), str(output)], check=True)
    payload = json.loads(output.read_text())
    return BenchmarkRecord(case.task, "samc", case.name, payload["status"], payload.get("median_seconds"), payload.get("timings_seconds", []), payload.get("note", ""))


def run_resistancega_inverse(case: BenchmarkCase) -> BenchmarkRecord:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return skip_record(case.task, "ResistanceGA", case.name, "Rscript is not installed in this environment.")
    output = Path(tempfile.mkdtemp(prefix="jaxscape-resistancega-")) / "resistancega.json"
    raster_json = Path(tempfile.mkdtemp(prefix="jaxscape-resistancega-input-")) / "raster.json"
    raster_json.write_text(json.dumps(case.raster))
    subprocess.run([rscript, str(RESISTANCE_GA_SCRIPT), str(raster_json), str(case.repeats), str(output)], check=True)
    payload = json.loads(output.read_text())
    return BenchmarkRecord(case.task, "ResistanceGA", case.name, payload["status"], payload.get("median_seconds"), payload.get("timings_seconds", []), payload.get("note", ""))


def collect_results() -> list[BenchmarkRecord]:
    resistance_case = CASES["resistance"]
    lcp_case = CASES["lcp"]
    sensitivity_case = CASES["sensitivity"]
    inverse_case = CASES["inverse"]
    return [
        run_jaxscape_resistance(resistance_case),
        run_circuitscape_resistance(resistance_case),
        run_conefor_placeholder(resistance_case, resistance_case.task),
        run_jaxscape_lcp(lcp_case),
        run_conefor_placeholder(lcp_case, lcp_case.task),
        run_jaxscape_sensitivity(sensitivity_case),
        run_samc_sensitivity(sensitivity_case),
        run_jaxscape_inverse(inverse_case),
        run_resistancega_inverse(inverse_case),
    ]


def write_results(records: list[BenchmarkRecord]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "python": shutil.which("python") or "python",
            "julia": shutil.which("julia"),
            "rscript": shutil.which("Rscript"),
            "repeats": REPEATS,
        },
        "records": [asdict(record) for record in records],
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    with RESULTS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "tool", "scenario", "status", "median_seconds", "timings_seconds", "note"],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["timings_seconds"] = json.dumps(row["timings_seconds"])
            writer.writerow(row)


def main() -> None:
    records = collect_results()
    write_results(records)
    print(json.dumps({"records": [asdict(record) for record in records]}, indent=2))


if __name__ == "__main__":
    main()
