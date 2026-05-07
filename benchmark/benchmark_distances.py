"""Cross-tool benchmark orchestration for JAXScape distance workloads."""

from __future__ import annotations

import argparse
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
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "JULIA_NUM_THREADS",
    "RCPP_PARALLEL_NUM_THREADS",
    "RCPPTHREAD_NUM_THREADS",
)
DEFAULT_BENCHMARK_THREADS = 4


def normalise_thread_count(raw_value: str | None) -> int:
    if raw_value is None:
        return DEFAULT_BENCHMARK_THREADS
    try:
        return max(1, int(raw_value))
    except ValueError as error:
        raise RuntimeError(f"BENCHMARK_THREADS must be an integer, received {raw_value!r}.") from error


def configure_thread_environment(thread_count: int) -> None:
    thread_value = str(thread_count)
    os.environ.setdefault("BENCHMARK_THREADS", thread_value)
    for env_name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ.setdefault(env_name, thread_value)

    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    extra_flags: list[str] = []
    if "--xla_cpu_multi_thread_eigen=" not in xla_flags:
        extra_flags.append("--xla_cpu_multi_thread_eigen=false" if thread_count == 1 else "--xla_cpu_multi_thread_eigen=true")
    if "intra_op_parallelism_threads=" not in xla_flags:
        extra_flags.append(f"intra_op_parallelism_threads={thread_count}")
    if extra_flags:
        os.environ["XLA_FLAGS"] = " ".join(filter(None, [xla_flags, *extra_flags]))


BENCHMARK_THREADS = normalise_thread_count(os.environ.get("BENCHMARK_THREADS"))
configure_thread_environment(BENCHMARK_THREADS)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr

from jaxscape import GridGraph, LCPDistance, ResistanceDistance

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
JULIA_PROJECT_DIR = BENCHMARK_DIR / "julia"
JULIA_DEPOT_DIR = BENCHMARK_DIR / ".julia"
R_LIBS_DIR = BENCHMARK_DIR / ".r-lib"
CIRCUITSCAPE_SCRIPT = BENCHMARK_DIR / "external" / "circuitscape_resistance.jl"
GDISTANCE_SCRIPT = BENCHMARK_DIR / "external" / "gdistance_runner.R"
RESISTANCE_GA_SCRIPT = BENCHMARK_DIR / "external" / "resistancega_inverse.R"
CONEFOR_SCRIPT = BENCHMARK_DIR / "external" / "conefor_runner.sh"
REPEATS = 3
MIN_PERMEABILITY = 1e-3
INVERSE_OPTIMISTIX_MAX_STEPS = 8
INVERSE_RESISTANCEGA_MAXITER = 3
SIZE_LABELS = ("small", "medium", "large")
CASE_GROUP_SPECS = {
    "resistance": {
        "task": "resistance_distance",
        "include_offset": True,
        "size_by_label": {"small": 12, "medium": 18, "large": 24},
        "seed_base": 0,
    },
    "lcp": {
        "task": "least_cost_path",
        "include_offset": True,
        "size_by_label": {"small": 16, "medium": 24, "large": 32},
        "seed_base": 10,
    },
    "sensitivity": {
        "task": "sensitivity_analysis",
        "include_offset": False,
        "size_by_label": {"small": 6, "medium": 8, "large": 10},
        "seed_base": 20,
    },
    "inverse": {
        "task": "inverse_landscape_genetics",
        "include_offset": False,
        "size_by_label": {"small": 6, "medium": 8, "large": 10},
        "seed_base": 30,
    },
}
REQUIRED_BASE_TOOL_LABELS_BY_TASK = {
    "resistance_distance": {
        "JAXScape / pinv",
        "JAXScape / PyAMG",
        "gdistance / commuteDistance",
        "Circuitscape.jl / cg+amg",
        "Circuitscape.jl / cholmod",
    },
    "least_cost_path": {"JAXScape", "gdistance / costDistance"},
    "sensitivity_analysis": {
        "JAXScape / shortest-path gradient",
        "gdistance / shortestPath",
        "JAXScape / resistance gradient",
        "gdistance / passage",
    },
    "inverse_landscape_genetics": {"JAXScape + Optimistix", "ResistanceGA"},
}
GPU_CAPABLE_TOOL_LABELS_BY_TASK = {
    "resistance_distance": {"JAXScape / pinv"},
    "least_cost_path": {"JAXScape"},
    "sensitivity_analysis": {"JAXScape / shortest-path gradient", "JAXScape / resistance gradient"},
    "inverse_landscape_genetics": {"JAXScape + Optimistix"},
}
GPU_PLACEHOLDER_NOTE = "GPU backend unavailable on this machine; placeholder emitted for the GPU-capable JAX profile."


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    task: str
    size_label: str
    grid_size: int
    seed: int
    include_offset: bool
    raster: list[list[float]]
    points: list[tuple[int, int]]


@dataclass(frozen=True)
class BenchmarkRecord:
    task: str
    tool: str
    software: str
    scenario: str
    status: str
    median_seconds: float | None
    timings_seconds: list[float]
    metrics: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass(frozen=True)
class BenchmarkConfig:
    repeats: int
    requested_device_platform: str
    cpu_device: jax.Device
    gpu_device: jax.Device | None
    include_conefor: bool
    require_complete: bool
    results_json: Path
    results_csv: Path
    benchmark_threads: int
    available_device_platforms: tuple[str, ...]


def create_landscape(seed: int = 0, size: int = 6) -> jax.Array:
    key = jr.PRNGKey(seed)
    base = jr.uniform(key, (size, size), minval=0.3, maxval=1.0)
    ridge = jnp.exp(-3 * jnp.linspace(-1.0, 1.0, size) ** 2)
    barrier = 0.4 * jnp.outer(ridge, jnp.ones(size))
    return jnp.clip(base - barrier, 0.15, 1.0)


def unique_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []
    for point in points:
        normalised = (int(point[0]), int(point[1]))
        if normalised in seen:
            continue
        seen.add(normalised)
        ordered.append(normalised)
    return ordered


def benchmark_points(size: int, include_offset: bool = True) -> list[tuple[int, int]]:
    margin = 1 if size > 4 else 0
    lower = margin
    upper = size - 1 - margin
    candidates = [
        (lower, lower),
        (lower, upper),
        (upper, lower),
        (upper, upper),
        (size // 2, size // 2),
    ]
    if include_offset and size > 4:
        candidates.extend(
            [
                (size // 3, min(size - 1, (2 * size) // 3)),
                (min(size - 1, (2 * size) // 3), size // 3),
            ]
        )
    return unique_points(candidates)


def build_case(
    name: str,
    task: str,
    size_label: str,
    size: int,
    seed: int,
    *,
    include_offset: bool = True,
) -> BenchmarkCase:
    raster = create_landscape(seed=seed, size=size)
    raster_as_lists = [[float(value) for value in row] for row in raster.tolist()]
    return BenchmarkCase(
        name=name,
        task=task,
        size_label=size_label,
        grid_size=size,
        seed=seed,
        include_offset=include_offset,
        raster=raster_as_lists,
        points=benchmark_points(size, include_offset=include_offset),
    )


def benchmark_cases() -> dict[str, list[BenchmarkCase]]:
    cases: dict[str, list[BenchmarkCase]] = {}
    for group_name, spec in CASE_GROUP_SPECS.items():
        group_cases: list[BenchmarkCase] = []
        for offset, size_label in enumerate(SIZE_LABELS):
            group_cases.append(
                build_case(
                    name=f"synthetic_{group_name}_{size_label}",
                    task=spec["task"],
                    size_label=size_label,
                    size=spec["size_by_label"][size_label],
                    seed=spec["seed_base"] + offset,
                    include_offset=bool(spec["include_offset"]),
                )
            )
        cases[group_name] = group_cases
    return cases


def cases_by_task(cases: dict[str, list[BenchmarkCase]]) -> dict[str, list[BenchmarkCase]]:
    grouped: dict[str, list[BenchmarkCase]] = {}
    for group_cases in cases.values():
        for case in group_cases:
            grouped.setdefault(case.task, []).append(case)
    return grouped


def iter_cases(cases: dict[str, list[BenchmarkCase]]) -> list[BenchmarkCase]:
    return [case for group_cases in cases.values() for case in group_cases]


CASES = benchmark_cases()
CASES_BY_TASK = cases_by_task(CASES)


def edge_weight(x: jax.Array, y: jax.Array) -> jax.Array:
    """Average adjacent node values to obtain an undirected edge weight."""
    return (x + y) / 2


def cost_conductance(x: jax.Array, y: jax.Array) -> jax.Array:
    """Map adjacent cell costs to a shared conductance parameterization."""
    return jnp.reciprocal(jnp.maximum((x + y) / 2, jnp.array(1e-6, dtype=x.dtype)))


def as_array(raster: list[list[float]]) -> jax.Array:
    return jnp.asarray(raster, dtype=jnp.float32)


def as_point_array(points: list[tuple[int, int]]) -> jax.Array:
    return jnp.asarray(points, dtype=jnp.int32)


def available_device_platforms() -> tuple[str, ...]:
    return tuple(sorted({device.platform for device in jax.devices()}))


def resolve_required_device(platform: str) -> jax.Device:
    try:
        return jax.devices(platform)[0]
    except RuntimeError as error:
        available = ", ".join(sorted({available_device.platform for available_device in jax.devices()}))
        raise RuntimeError(f"JAX device platform '{platform}' is unavailable. Detected platforms: {available}.") from error


def resolve_optional_device(platform: str) -> jax.Device | None:
    try:
        return jax.devices(platform)[0]
    except RuntimeError:
        return None


def thread_environment_snapshot() -> dict[str, str]:
    env_names = ("BENCHMARK_THREADS", *THREAD_ENVIRONMENT_VARIABLES, "XLA_FLAGS")
    return {env_name: os.environ[env_name] for env_name in env_names if os.environ.get(env_name)}


def benchmark_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("JULIA_DEPOT_PATH", str(JULIA_DEPOT_DIR))
    environment.setdefault("R_LIBS_USER", str(R_LIBS_DIR))
    environment.setdefault("BENCHMARK_THREADS", str(BENCHMARK_THREADS))
    for env_name in THREAD_ENVIRONMENT_VARIABLES:
        environment.setdefault(env_name, str(BENCHMARK_THREADS))
    return environment


def write_case_payload(case: BenchmarkCase, directory: Path) -> Path:
    payload = {
        "name": case.name,
        "task": case.task,
        "size_label": case.size_label,
        "grid_size": case.grid_size,
        "seed": case.seed,
        "include_offset": case.include_offset,
        "raster": case.raster,
        "points": case.points,
    }
    payload_path = directory / f"{case.name}.json"
    payload_path.write_text(json.dumps(payload, indent=2))
    return payload_path


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


def ok_record(task: str, tool: str, software: str, scenario: str, timings: list[float], note: str = "") -> BenchmarkRecord:
    return BenchmarkRecord(task, tool, software, scenario, "ok", statistics.median(timings), timings, {}, note)


def ok_record_with_metrics(
    task: str,
    tool: str,
    software: str,
    scenario: str,
    timings: list[float],
    *,
    metrics: dict[str, Any],
    note: str = "",
) -> BenchmarkRecord:
    return BenchmarkRecord(task, tool, software, scenario, "ok", statistics.median(timings), timings, metrics, note)


def skip_record(task: str, tool: str, software: str, scenario: str, note: str) -> BenchmarkRecord:
    return BenchmarkRecord(task, tool, software, scenario, "skipped", None, [], {}, note)


def failed_record(task: str, tool: str, software: str, scenario: str, note: str) -> BenchmarkRecord:
    return BenchmarkRecord(task, tool, software, scenario, "failed", None, [], {}, note)


def backend_tool_label(tool: str, backend: str) -> str:
    return f"{tool} ({backend.upper()})"


def gpu_placeholder_record(case: BenchmarkCase, tool: str, software: str) -> BenchmarkRecord:
    return skip_record(case.task, backend_tool_label(tool, "gpu"), software, case.name, GPU_PLACEHOLDER_NOTE)


def collect_dual_backend_records(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    base_tool: str,
    software: str,
    runner: Callable[..., BenchmarkRecord],
) -> list[BenchmarkRecord]:
    records = [runner(case, config, device=config.cpu_device, tool_label=backend_tool_label(base_tool, "cpu"))]
    if config.gpu_device is None:
        records.append(gpu_placeholder_record(case, base_tool, software))
    else:
        records.append(runner(case, config, device=config.gpu_device, tool_label=backend_tool_label(base_tool, "gpu")))
    return records


def lower_triangle_values(matrix: jax.Array) -> jax.Array:
    indices = jnp.tril_indices(matrix.shape[0], k=-1)
    return matrix[indices]


def fit_error_metrics(predicted_values: jax.Array, target_values: jax.Array) -> dict[str, float]:
    residual = predicted_values - target_values
    rmse = jnp.sqrt(jnp.mean(jnp.square(residual)))
    reference_scale = jnp.sqrt(jnp.mean(jnp.square(target_values)))
    relative_rmse = rmse / jnp.maximum(reference_scale, jnp.array(1e-12, dtype=target_values.dtype))
    return {
        "rmse": float(rmse),
        "relative_rmse": float(relative_rmse),
    }


def normalise_nonnegative(array: np.ndarray) -> np.ndarray:
    clipped = np.nan_to_num(np.asarray(array, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.maximum(clipped, 0.0)
    total = clipped.sum()
    if total <= 0:
        return clipped
    return clipped / total


def centrality_alignment_metrics(candidate: jax.Array | np.ndarray, reference: jax.Array | np.ndarray) -> dict[str, float]:
    candidate_norm = normalise_nonnegative(np.asarray(candidate, dtype=float).ravel())
    reference_norm = normalise_nonnegative(np.asarray(reference, dtype=float).ravel())
    if not np.any(candidate_norm) or not np.any(reference_norm):
        return {
            "cosine_similarity": 0.0,
            "normalised_l1": 1.0,
            "normalised_rmse": 1.0,
        }

    cosine_similarity = float(np.dot(candidate_norm, reference_norm) / (np.linalg.norm(candidate_norm) * np.linalg.norm(reference_norm)))
    return {
        "cosine_similarity": cosine_similarity,
        "normalised_l1": float(np.abs(candidate_norm - reference_norm).sum()),
        "normalised_rmse": float(np.sqrt(np.mean(np.square(candidate_norm - reference_norm)))),
    }


def benchmark_graph_volume(cost_surface: jax.Array) -> jax.Array:
    horizontal = cost_conductance(cost_surface[:, :-1], cost_surface[:, 1:])
    vertical = cost_conductance(cost_surface[:-1, :], cost_surface[1:, :])
    return 2.0 * (jnp.sum(horizontal) + jnp.sum(vertical))


@eqx.filter_jit
def _jaxscape_resistance(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return ResistanceDistance()(grid, nodes=nodes)


@eqx.filter_jit
def _jaxscape_resistance_with_solver(permeability: jax.Array, nodes: jax.Array, solver: Any) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return ResistanceDistance(solver=solver)(grid, nodes=nodes)


@eqx.filter_jit
def _jaxscape_lcp(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return LCPDistance()(grid, nodes=nodes)


def make_pyamg_solver() -> Any:
    try:
        from jaxscape.solvers.pyamgsolver import PyAMGSolver
    except ImportError as error:
        raise ImportError("Install the benchmark Python extra to enable the PyAMG resistance profile.") from error
    return PyAMGSolver(rtol=1e-6, maxiter=50_000)


def make_cholmod_solver() -> Any:
    try:
        from jaxscape.solvers.cholmodsolver import CholmodSolver
    except ImportError as error:
        raise ImportError("Install the cholespy Python extra to enable the Cholmod resistance profile.") from error
    return CholmodSolver()


def run_jaxscape_resistance_profile(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    tool_label: str,
    *,
    device: jax.Device,
    solver_factory: Callable[[], Any] | None = None,
) -> BenchmarkRecord:
    try:
        solver = None if solver_factory is None else solver_factory()
    except ImportError as error:
        return skip_record(case.task, tool_label, "JAXScape", case.name, str(error))
    except Exception as error:  # pragma: no cover - defensive benchmark reporting
        return failed_record(case.task, tool_label, "JAXScape", case.name, str(error))

    permeability = jax.device_put(as_array(case.raster), device)
    nodes = jax.device_put(as_point_array(case.points), device)
    if solver is None:
        timings, _ = measure_runtime(_jaxscape_resistance, permeability, nodes, repeats=config.repeats)
    else:
        timings, _ = measure_runtime(_jaxscape_resistance_with_solver, permeability, nodes, solver, repeats=config.repeats)
    return ok_record(case.task, tool_label, "JAXScape", case.name, timings)


def run_jaxscape_lcp(case: BenchmarkCase, config: BenchmarkConfig, *, device: jax.Device, tool_label: str) -> BenchmarkRecord:
    permeability = jax.device_put(as_array(case.raster), device)
    nodes = jax.device_put(as_point_array(case.points), device)
    timings, _ = measure_runtime(_jaxscape_lcp, permeability, nodes, repeats=config.repeats)
    return ok_record(case.task, tool_label, "JAXScape", case.name, timings)


@eqx.filter_jit
@eqx.filter_grad
def run_shortest_path_centrality_problem(cost_raster: jax.Array, sample_nodes: jax.Array) -> jax.Array:
    grid = GridGraph(cost_raster, fun=cost_conductance)
    pairwise_distances = LCPDistance()(grid, nodes=sample_nodes)
    return jnp.sum(pairwise_distances[0, 1:])


@eqx.filter_jit
@eqx.filter_grad
def run_resistance_centrality_problem(cost_raster: jax.Array, sample_nodes: jax.Array) -> jax.Array:
    grid = GridGraph(cost_raster, fun=cost_conductance)
    pairwise_resistance = ResistanceDistance()(grid, nodes=sample_nodes)
    return benchmark_graph_volume(cost_raster) * jnp.sum(pairwise_resistance[0, 1:])


def run_jaxscape_centrality(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    device: jax.Device,
    tool_label: str,
    centrality_fun,
    note: str,
    reference_raster: np.ndarray | None = None,
) -> tuple[BenchmarkRecord, np.ndarray]:
    cost_surface = jax.device_put(as_array(case.raster), device)
    nodes = jax.device_put(as_point_array(case.points), device)
    timings, gradient = measure_runtime(centrality_fun, cost_surface, nodes, repeats=config.repeats)
    gradient_raster = np.asarray(jax.device_get(jnp.abs(gradient)))
    metrics: dict[str, Any] = {
        "centrality_mass": float(np.sum(gradient_raster)),
    }
    if reference_raster is not None:
        metrics.update(centrality_alignment_metrics(gradient_raster, reference_raster))
    return (
        ok_record_with_metrics(case.task, tool_label, "JAXScape", case.name, timings, metrics=metrics, note=note),
        gradient_raster,
    )


def _inverse_loss(logits: jax.Array, sample_coords: jax.Array, target_distances: jax.Array) -> jax.Array:
    permeability = jnn.sigmoid(logits) + jnp.array(MIN_PERMEABILITY, dtype=logits.dtype)
    grid = GridGraph(grid=permeability, fun=edge_weight)
    predicted = ResistanceDistance()(grid, nodes=sample_coords)
    return jnp.mean((predicted - target_distances) ** 2)


def run_jaxscape_inverse(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    device: jax.Device,
    tool_label: str,
) -> BenchmarkRecord:
    if optx is None:
        return skip_record(case.task, tool_label, "JAXScape + Optimistix", case.name, _optimistix_import_error or "optimistix missing")

    landscape = jax.device_put(as_array(case.raster), device)
    sample_coords = jax.device_put(as_point_array(case.points), device)
    target_distances = ResistanceDistance()(GridGraph(landscape, fun=edge_weight), nodes=sample_coords)
    target_values = lower_triangle_values(target_distances)
    solver = optx.LBFGS(rtol=1e-5, atol=1e-5)

    def objective(logits: jax.Array, args: tuple[jax.Array, jax.Array]) -> jax.Array:
        coords, targets = args
        return _inverse_loss(logits, coords, targets)

    def solve(problem_landscape: jax.Array) -> optx.Solution:
        init_logits = jnp.zeros_like(problem_landscape)
        return optx.minimise(
            objective,
            solver,
            init_logits,
            args=(sample_coords, target_distances),
            max_steps=INVERSE_OPTIMISTIX_MAX_STEPS,
            throw=False,
        )

    timings, solution = measure_runtime(solve, landscape, repeats=config.repeats)
    fitted_permeability = jnn.sigmoid(solution.value) + jnp.array(MIN_PERMEABILITY, dtype=solution.value.dtype)
    predicted_distances = ResistanceDistance()(GridGraph(fitted_permeability, fun=edge_weight), nodes=sample_coords)
    metrics = fit_error_metrics(lower_triangle_values(predicted_distances), target_values)
    metrics.update(
        {
            "final_mse": float(_inverse_loss(solution.value, sample_coords, target_distances)),
            "converged": bool(solution.result == optx.RESULTS.successful),
            "iteration_count": int(solution.stats["num_steps"]),
            "iteration_limit": INVERSE_OPTIMISTIX_MAX_STEPS,
        }
    )
    return ok_record_with_metrics(
        case.task,
        tool_label,
        "JAXScape + Optimistix",
        case.name,
        timings,
        metrics=metrics,
        note="Fixed-budget L-BFGS calibration against the synthetic resistance-distance target.",
    )


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


def circuitscape_ini(cellmap: Path, points: Path, output_prefix: Path, solver_name: str) -> str:
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
        solver = {solver_name}
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


def run_circuitscape_resistance(case: BenchmarkCase, config: BenchmarkConfig, solver_name: str, tool_label: str) -> BenchmarkRecord:
    if shutil.which("julia") is None:
        return skip_record(case.task, tool_label, "Circuitscape.jl", case.name, "Julia is not available in this environment.")

    with tempfile.TemporaryDirectory(prefix="jaxscape-circuitscape-") as tmpdir:
        workspace = Path(tmpdir)
        cellmap = workspace / "cellmap.asc"
        points = workspace / "points.asc"
        output_prefix = workspace / "result"
        ini = workspace / "config.ini"
        output = workspace / "circuitscape.json"
        cellmap.write_text(ascii_grid_from_raster(case.raster))
        points.write_text(ascii_points_from_points((len(case.raster), len(case.raster[0])), case.points))
        ini.write_text(circuitscape_ini(cellmap, points, output_prefix, solver_name))
        subprocess.run(
            ["julia", f"--project={JULIA_PROJECT_DIR}", str(CIRCUITSCAPE_SCRIPT), str(ini), str(config.repeats), str(output)],
            check=True,
            env=benchmark_environment(),
        )
        payload = json.loads(output.read_text())
    return BenchmarkRecord(
        case.task,
        tool_label,
        "Circuitscape.jl",
        case.name,
        payload["status"],
        payload.get("median_seconds"),
        payload.get("timings_seconds", []),
        payload.get("metrics", {}),
        payload.get("note", ""),
    )


def run_gdistance_payload(case: BenchmarkCase, config: BenchmarkConfig, mode: str) -> dict[str, Any]:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return {
            "status": "skipped",
            "timings_seconds": [],
            "median_seconds": None,
            "metrics": {},
            "note": "Rscript is not installed in this environment.",
        }

    with tempfile.TemporaryDirectory(prefix=f"jaxscape-gdistance-{mode}-") as tmpdir:
        workspace = Path(tmpdir)
        output = workspace / f"{mode}.json"
        payload_path = write_case_payload(case, workspace)
        subprocess.run(
            [rscript, str(GDISTANCE_SCRIPT), mode, str(payload_path), str(config.repeats), str(output)],
            check=True,
            env=benchmark_environment(),
        )
        return json.loads(output.read_text())


def gdistance_record_from_payload(case: BenchmarkCase, tool_label: str, payload: dict[str, Any], *, note_suffix: str = "") -> BenchmarkRecord:
    note = payload.get("note", "")
    if note_suffix:
        note = f"{note} {note_suffix}".strip()
    return BenchmarkRecord(
        case.task,
        tool_label,
        "gdistance",
        case.name,
        payload["status"],
        payload.get("median_seconds"),
        payload.get("timings_seconds", []),
        payload.get("metrics", {}),
        note,
    )


def gdistance_centrality_raster(payload: dict[str, Any]) -> np.ndarray | None:
    raster = payload.get("metrics", {}).get("centrality_raster")
    if raster is None:
        return None
    return np.asarray(raster, dtype=float)


def run_conefor_placeholder(case: BenchmarkCase, task_label: str) -> BenchmarkRecord:
    conefor_bin = os.environ.get("CONEFOR_BIN")
    if not conefor_bin:
        return skip_record(task_label, "Conefor", "Conefor", case.name, "Set CONEFOR_BIN to enable the Conefor adapter.")
    output = Path(tempfile.mkdtemp(prefix="jaxscape-conefor-")) / "conefor.json"
    subprocess.run([str(CONEFOR_SCRIPT), conefor_bin, task_label, str(output)], check=True)
    payload = json.loads(output.read_text())
    return BenchmarkRecord(
        task_label,
        "Conefor",
        "Conefor",
        case.name,
        payload["status"],
        payload.get("median_seconds"),
        payload.get("timings_seconds", []),
        payload.get("metrics", {}),
        payload.get("note", ""),
    )


def run_resistancega_inverse(case: BenchmarkCase, config: BenchmarkConfig) -> BenchmarkRecord:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return skip_record(case.task, "ResistanceGA", "ResistanceGA", case.name, "Rscript is not installed in this environment.")
    with tempfile.TemporaryDirectory(prefix="jaxscape-resistancega-") as tmpdir:
        workspace = Path(tmpdir)
        output = workspace / "resistancega.json"
        payload_path = write_case_payload(case, workspace)
        subprocess.run(
            [rscript, str(RESISTANCE_GA_SCRIPT), str(payload_path), str(config.repeats), str(output)],
            check=True,
            env=benchmark_environment(),
        )
        payload = json.loads(output.read_text())
    return BenchmarkRecord(
        case.task,
        "ResistanceGA",
        "ResistanceGA",
        case.name,
        payload["status"],
        payload.get("median_seconds"),
        payload.get("timings_seconds", []),
        payload.get("metrics", {}),
        payload.get("note", ""),
    )


def collect_resistance_results(case: BenchmarkCase, config: BenchmarkConfig) -> list[BenchmarkRecord]:
    gdistance_payload = run_gdistance_payload(case, config, "resistance_distance")
    graph_volume = float(jax.device_get(benchmark_graph_volume(as_array(case.raster))))
    if gdistance_payload["status"] == "ok" and gdistance_payload.get("metrics", {}).get("distance_matrix") is not None and graph_volume > 0:
        commute_matrix = np.asarray(gdistance_payload["metrics"]["distance_matrix"], dtype=float)
        gdistance_payload["metrics"] = {
            "graph_volume": graph_volume,
            "distance_matrix": (commute_matrix / graph_volume).tolist(),
        }

    records = [
        *collect_dual_backend_records(case, config, "JAXScape / pinv", "JAXScape", run_jaxscape_resistance_profile),
        run_jaxscape_resistance_profile(case, config, "JAXScape / PyAMG", device=config.cpu_device, solver_factory=make_pyamg_solver),
        run_jaxscape_resistance_profile(case, config, "JAXScape / CholmodSolver", device=config.cpu_device, solver_factory=make_cholmod_solver),
        gdistance_record_from_payload(
            case,
            "gdistance / commuteDistance",
            gdistance_payload,
            note_suffix="Reported values are rescaled from commute time to effective resistance by dividing by the graph volume.",
        ),
        run_circuitscape_resistance(case, config, "cg+amg", "Circuitscape.jl / cg+amg"),
        run_circuitscape_resistance(case, config, "cholmod", "Circuitscape.jl / cholmod"),
    ]
    if sys.platform == "darwin":
        records.append(run_circuitscape_resistance(case, config, "accelerate", "Circuitscape.jl / accelerate"))
    return records


def collect_least_cost_results(case: BenchmarkCase, config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records = collect_dual_backend_records(case, config, "JAXScape", "JAXScape", run_jaxscape_lcp)
    records.append(gdistance_record_from_payload(case, "gdistance / costDistance", run_gdistance_payload(case, config, "least_cost_path")))
    return records


def collect_sensitivity_results(case: BenchmarkCase, config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []

    shortest_path_payload = run_gdistance_payload(case, config, "least_cost_centrality")
    shortest_path_reference = gdistance_centrality_raster(shortest_path_payload) if shortest_path_payload.get("status") == "ok" else None
    jax_shortest_cpu_record, _ = run_jaxscape_centrality(
        case,
        config,
        device=config.cpu_device,
        tool_label=backend_tool_label("JAXScape / shortest-path gradient", "cpu"),
        centrality_fun=run_shortest_path_centrality_problem,
        note="Gradient of the summed single-origin least-cost objective on the shared cost surface.",
        reference_raster=shortest_path_reference,
    )
    records.append(jax_shortest_cpu_record)
    if config.gpu_device is None:
        records.append(gpu_placeholder_record(case, "JAXScape / shortest-path gradient", "JAXScape"))
    else:
        jax_shortest_gpu_record, _ = run_jaxscape_centrality(
            case,
            config,
            device=config.gpu_device,
            tool_label=backend_tool_label("JAXScape / shortest-path gradient", "gpu"),
            centrality_fun=run_shortest_path_centrality_problem,
            note="Gradient of the summed single-origin least-cost objective on the shared cost surface.",
            reference_raster=shortest_path_reference,
        )
        records.append(jax_shortest_gpu_record)
    records.append(gdistance_record_from_payload(case, "gdistance / shortestPath", shortest_path_payload))

    resistance_payload = run_gdistance_payload(case, config, "resistance_centrality")
    resistance_reference = gdistance_centrality_raster(resistance_payload) if resistance_payload.get("status") == "ok" else None
    jax_resistance_cpu_record, _ = run_jaxscape_centrality(
        case,
        config,
        device=config.cpu_device,
        tool_label=backend_tool_label("JAXScape / resistance gradient", "cpu"),
        centrality_fun=run_resistance_centrality_problem,
        note="Gradient of graph-volume-scaled effective resistance, matching commute-time passage counts on the shared cost surface.",
        reference_raster=resistance_reference,
    )
    records.append(jax_resistance_cpu_record)
    if config.gpu_device is None:
        records.append(gpu_placeholder_record(case, "JAXScape / resistance gradient", "JAXScape"))
    else:
        jax_resistance_gpu_record, _ = run_jaxscape_centrality(
            case,
            config,
            device=config.gpu_device,
            tool_label=backend_tool_label("JAXScape / resistance gradient", "gpu"),
            centrality_fun=run_resistance_centrality_problem,
            note="Gradient of graph-volume-scaled effective resistance, matching commute-time passage counts on the shared cost surface.",
            reference_raster=resistance_reference,
        )
        records.append(jax_resistance_gpu_record)
    records.append(gdistance_record_from_payload(case, "gdistance / passage", resistance_payload))

    return records


def collect_results(config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["resistance"]:
        records.extend(collect_resistance_results(case, config))
    for case in CASES["lcp"]:
        records.extend(collect_least_cost_results(case, config))
    for case in CASES["sensitivity"]:
        records.extend(collect_sensitivity_results(case, config))
    for case in CASES["inverse"]:
        records.extend(
            collect_dual_backend_records(
                case,
                config,
                "JAXScape + Optimistix",
                "JAXScape + Optimistix",
                run_jaxscape_inverse,
            )
        )
        records.append(run_resistancega_inverse(case, config))
    if config.include_conefor:
        for case in CASES["resistance"]:
            records.append(run_conefor_placeholder(case, case.task))
        for case in CASES["lcp"]:
            records.append(run_conefor_placeholder(case, case.task))
    return records


def write_results(records: list[BenchmarkRecord], config: BenchmarkConfig) -> None:
    config.results_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": {
            "python": shutil.which("python") or "python",
            "julia": shutil.which("julia"),
            "rscript": shutil.which("Rscript"),
            "platform": sys.platform,
            "jax_backend": jax.default_backend(),
            "requested_device_platform": config.requested_device_platform,
            "available_jax_platforms": list(config.available_device_platforms),
            "cpu_device": repr(config.cpu_device),
            "gpu_device": repr(config.gpu_device) if config.gpu_device is not None else None,
            "gpu_backend_available": config.gpu_device is not None,
            "julia_project": str(JULIA_PROJECT_DIR),
            "julia_depot": str(JULIA_DEPOT_DIR),
            "r_libs_user": str(R_LIBS_DIR),
            "repeats": config.repeats,
            "benchmark_threads": config.benchmark_threads,
            "thread_environment": thread_environment_snapshot(),
        },
        "cases": [
            {
                "name": case.name,
                "task": case.task,
                "size_label": case.size_label,
                "grid_size": case.grid_size,
                "seed": case.seed,
                "include_offset": case.include_offset,
                "point_count": len(case.points),
                "points": case.points,
            }
            for case in iter_cases(CASES)
        ],
        "records": [asdict(record) for record in records],
    }
    config.results_json.write_text(json.dumps(payload, indent=2))
    with config.results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "tool", "software", "scenario", "status", "median_seconds", "timings_seconds", "metrics", "note"],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["timings_seconds"] = json.dumps(row["timings_seconds"])
            row["metrics"] = json.dumps(row["metrics"])
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=["default", "cpu", "gpu", "tpu"],
        default=os.environ.get("JAXSCAPE_BENCHMARK_DEVICE", "default"),
        help="Preferred JAX backend hint. CPU cross-tool profiles always run, and GPU-capable JAX profiles emit a GPU series when available.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--include-conefor", action="store_true", help="Include the optional Conefor adapter in the output artifacts.")
    parser.add_argument("--require-complete", action="store_true", help="Exit with a non-zero status when a required benchmark profile is skipped or fails.")
    parser.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)
    return parser.parse_args()


def required_records(config: BenchmarkConfig) -> dict[tuple[str, str, str], set[str]]:
    required: dict[tuple[str, str, str], set[str]] = {}
    for task, labels in REQUIRED_BASE_TOOL_LABELS_BY_TASK.items():
        gpu_capable_labels = GPU_CAPABLE_TOOL_LABELS_BY_TASK.get(task, set())
        for case in CASES_BY_TASK[task]:
            for tool in labels:
                if tool in gpu_capable_labels:
                    required[(task, case.name, backend_tool_label(tool, "cpu"))] = {"ok"}
                    required[(task, case.name, backend_tool_label(tool, "gpu"))] = {"ok"} if config.gpu_device is not None else {"skipped"}
                else:
                    required[(task, case.name, tool)] = {"ok"}
    return required


def validate_records(records: list[BenchmarkRecord], config: BenchmarkConfig) -> None:
    if not config.require_complete:
        return

    required = required_records(config)
    observed = {(record.task, record.scenario, record.tool): record for record in records}
    missing = sorted(set(required) - set(observed))
    if missing:
        summary = ", ".join(f"{tool}/{task}/{scenario}" for task, scenario, tool in missing)
        raise RuntimeError(f"Benchmark suite is incomplete: missing records for {summary}")

    incomplete = [
        record
        for key, record in observed.items()
        if key in required and record.status not in required[key]
    ]
    if incomplete:
        summary = "; ".join(
            f"{record.tool}/{record.task}/{record.scenario}: {record.note or record.status}" for record in incomplete
        )
        raise RuntimeError(f"Benchmark suite is incomplete: {summary}")


def main() -> None:
    args = parse_args()
    cpu_device = resolve_required_device("cpu")
    gpu_device = resolve_optional_device("gpu")
    config = BenchmarkConfig(
        repeats=args.repeats,
        requested_device_platform=args.device,
        cpu_device=cpu_device,
        gpu_device=gpu_device,
        include_conefor=args.include_conefor,
        require_complete=args.require_complete,
        results_json=args.results_json,
        results_csv=args.results_csv,
        benchmark_threads=BENCHMARK_THREADS,
        available_device_platforms=available_device_platforms(),
    )
    records = collect_results(config)
    validate_records(records, config)
    write_results(records, config)
    print(json.dumps({"records": [asdict(record) for record in records]}, indent=2))


if __name__ == "__main__":
    main()
