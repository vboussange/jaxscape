"""Cross-tool benchmark orchestration for JAXScape distance workloads."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))
sys.modules.setdefault("benchmark.benchmark_distances", sys.modules[__name__])

from benchmark.benchmark_registry import (
    backend_tool_label,
    BENCHMARK_TASK_SPECS,
    GPU_CAPABLE_TOOL_LABELS_BY_TASK,
    REQUIRED_BASE_TOOL_LABELS_BY_TASK,
)
from benchmark.inverse_benchmark_settings import load_inverse_benchmark_settings
from benchmark.jaxscape.utils import (
    benchmark_walltime_seconds,
    configure_logging,
    format_walltime_seconds,
)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

LOGGER = logging.getLogger(__name__)

BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "benchmark_results.json"
RESULTS_CSV = RESULTS_DIR / "benchmark_results.csv"
JULIA_PROJECT_DIR = BENCHMARK_DIR / "julia"
JULIA_DEPOT_DIR = BENCHMARK_DIR / ".julia"
R_LIBS_DIR = BENCHMARK_DIR / ".r-lib"
EXTERNAL_DIR = BENCHMARK_DIR / "external"
CIRCUITSCAPE_SCRIPT = EXTERNAL_DIR / "circuitscape_resistance.jl"
GDISTANCE_SCRIPT = EXTERNAL_DIR / "gdistance_runner.R"
RESISTANCE_GA_SCRIPT = EXTERNAL_DIR / "resistancega_inverse.R"
CONEFOR_SCRIPT = EXTERNAL_DIR / "conefor_runner.sh"
LOCAL_TOOLCHAIN_BIN_DIRS = (
    Path.home() / ".juliaup" / "bin",
    Path.home() / ".local" / "bin",
)

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
REPEATS = 3
MIN_PERMEABILITY = 1e-3
DEFAULT_BENCHMARK_POINT_COUNT = 20
DEFAULT_EXTERNAL_WALLTIME_SECONDS = 900.0
EXTERNAL_WALLTIME_ENV = "JAXSCAPE_BENCHMARK_EXTERNAL_WALLTIME_SECONDS"
EXTERNAL_WALLTIME_ENV_BY_SOFTWARE = {
    "Circuitscape.jl": "JAXSCAPE_BENCHMARK_CIRCUITSCAPE_WALLTIME_SECONDS",
    "gdistance": "JAXSCAPE_BENCHMARK_GDISTANCE_WALLTIME_SECONDS",
    "ResistanceGA": "JAXSCAPE_BENCHMARK_RESISTANCEGA_WALLTIME_SECONDS",
    "Conefor": "JAXSCAPE_BENCHMARK_CONEFOR_WALLTIME_SECONDS",
}
GPU_PLACEHOLDER_NOTE = (
    "GPU backend unavailable on this machine; placeholder emitted for the "
    "GPU-capable JAX profile."
)


def configure_local_toolchain_path() -> None:
    current_path = os.environ.get("PATH", "")
    path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
    existing_entries = set(path_entries)
    prepended_entries: list[str] = []
    for candidate_dir in LOCAL_TOOLCHAIN_BIN_DIRS:
        candidate_string = str(candidate_dir)
        if candidate_dir.is_dir() and candidate_string not in existing_entries:
            prepended_entries.append(candidate_string)
            existing_entries.add(candidate_string)
    if prepended_entries:
        os.environ["PATH"] = os.pathsep.join(
            [*prepended_entries, *path_entries]
        )


def configure_thread_environment(thread_count: int) -> None:
    thread_value = str(thread_count)
    os.environ.setdefault("BENCHMARK_THREADS", thread_value)
    for env_name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ.setdefault(env_name, thread_value)

    xla_flags = os.environ.get("XLA_FLAGS", "").strip()
    extra_flags: list[str] = []
    if "--xla_cpu_multi_thread_eigen=" not in xla_flags:
        extra_flags.append(
            "--xla_cpu_multi_thread_eigen=false"
            if thread_count == 1
            else "--xla_cpu_multi_thread_eigen=true"
        )
    if "intra_op_parallelism_threads=" not in xla_flags:
        extra_flags.append(f"intra_op_parallelism_threads={thread_count}")
    if extra_flags:
        os.environ["XLA_FLAGS"] = " ".join(filter(None, [xla_flags, *extra_flags]))


try:
    BENCHMARK_THREADS = max(
        1, int(os.environ.get("BENCHMARK_THREADS", DEFAULT_BENCHMARK_THREADS))
    )
except ValueError as error:
    raw_value = os.environ.get("BENCHMARK_THREADS")
    message = f"BENCHMARK_THREADS must be an integer, received {raw_value!r}."
    raise RuntimeError(message) from error

configure_thread_environment(BENCHMARK_THREADS)
configure_local_toolchain_path()
try:
    BENCHMARK_POINT_COUNT = int(
        os.environ.get("JAXSCAPE_BENCHMARK_POINT_COUNT", DEFAULT_BENCHMARK_POINT_COUNT)
    )
except ValueError as error:
    raw_value = os.environ.get("JAXSCAPE_BENCHMARK_POINT_COUNT")
    message = (
        f"JAXSCAPE_BENCHMARK_POINT_COUNT must be an integer, received {raw_value!r}."
    )
    raise RuntimeError(message) from error

if BENCHMARK_POINT_COUNT < 1:
    raise RuntimeError(
        "JAXSCAPE_BENCHMARK_POINT_COUNT must be at least 1, received "
        f"{BENCHMARK_POINT_COUNT}."
    )

import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    task: str
    size_label: str
    grid_size: int
    seed: int
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


def benchmark_points(
    size: int,
    *,
    point_count: int | None = None,
    seed: int = 0,
) -> list[tuple[int, int]]:
    resolved_point_count = (
        BENCHMARK_POINT_COUNT if point_count is None else int(point_count)
    )
    if resolved_point_count < 1:
        raise RuntimeError(
            "JAXSCAPE_BENCHMARK_POINT_COUNT must be at least 1, received "
            f"{resolved_point_count}."
        )
    if size <= 0:
        return []

    margin = 1 if size > 4 else 0
    interior_points = [
        (i, j)
        for i in range(margin, size - margin)
        for j in range(margin, size - margin)
    ]
    if not interior_points:
        return []

    sample_size = min(resolved_point_count, len(interior_points))
    sample_indices = np.random.default_rng(seed).choice(
        len(interior_points), size=sample_size, replace=False
    )
    return [interior_points[int(index)] for index in np.atleast_1d(sample_indices)]


def build_case(
    name: str,
    task: str,
    size_label: str,
    size: int,
    seed: int,
) -> BenchmarkCase:
    raster = create_landscape(seed=seed, size=size)
    raster_as_lists = [[float(value) for value in row] for row in raster.tolist()]
    return BenchmarkCase(
        name=name,
        task=task,
        size_label=size_label,
        grid_size=size,
        seed=seed,
        raster=raster_as_lists,
        points=benchmark_points(size, seed=seed),
    )


def benchmark_cases() -> dict[str, list[BenchmarkCase]]:
    cases: dict[str, list[BenchmarkCase]] = {}
    for spec in BENCHMARK_TASK_SPECS:
        group_cases: list[BenchmarkCase] = []
        for offset, (size_label, size) in enumerate(spec.size_by_label.items()):
            group_cases.append(
                build_case(
                    name=f"synthetic_{spec.group_name}_{size_label}",
                    task=spec.task,
                    size_label=size_label,
                    size=size,
                    seed=spec.seed_base + offset,
                )
            )
        cases[spec.group_name] = group_cases
    return cases


def refresh_benchmark_cases() -> None:
    CASES.clear()
    CASES.update(benchmark_cases())
    CASES_BY_TASK.clear()
    CASES_BY_TASK.update(cases_by_task(CASES))


def configure_benchmark_point_count(point_count: int) -> int:
    global BENCHMARK_POINT_COUNT

    BENCHMARK_POINT_COUNT = int(point_count)
    if BENCHMARK_POINT_COUNT < 1:
        raise RuntimeError(
            "JAXSCAPE_BENCHMARK_POINT_COUNT must be at least 1, received "
            f"{BENCHMARK_POINT_COUNT}."
        )
    refresh_benchmark_cases()
    return BENCHMARK_POINT_COUNT


def cases_by_task(
    cases: dict[str, list[BenchmarkCase]],
) -> dict[str, list[BenchmarkCase]]:
    grouped: dict[str, list[BenchmarkCase]] = {}
    for group_cases in cases.values():
        for case in group_cases:
            grouped.setdefault(case.task, []).append(case)
    return grouped


def iter_cases(
    cases: dict[str, list[BenchmarkCase]] | None = None,
) -> list[BenchmarkCase]:
    selected_cases = CASES if cases is None else cases
    return [case for group_cases in selected_cases.values() for case in group_cases]


CASES = benchmark_cases()
CASES_BY_TASK = cases_by_task(CASES)


def available_device_platforms() -> tuple[str, ...]:
    return tuple(sorted({device.platform for device in jax.devices()}))


def resolve_required_device(platform: str) -> jax.Device:
    try:
        return jax.devices(platform)[0]
    except RuntimeError as error:
        available = ", ".join(
            sorted({available_device.platform for available_device in jax.devices()})
        )
        message = (
            f"JAX device platform '{platform}' is unavailable. "
            f"Detected platforms: {available}."
        )
        raise RuntimeError(message) from error


def resolve_optional_device(platform: str) -> jax.Device | None:
    try:
        return jax.devices(platform)[0]
    except RuntimeError:
        return None


def build_config(
    *,
    repeats: int,
    requested_device_platform: str,
    include_conefor: bool,
    require_complete: bool,
    results_json: Path,
    results_csv: Path,
) -> BenchmarkConfig:
    return BenchmarkConfig(
        repeats=repeats,
        requested_device_platform=requested_device_platform,
        cpu_device=resolve_required_device("cpu"),
        gpu_device=resolve_optional_device("gpu"),
        include_conefor=include_conefor,
        require_complete=require_complete,
        results_json=Path(results_json),
        results_csv=Path(results_csv),
        benchmark_threads=BENCHMARK_THREADS,
        available_device_platforms=available_device_platforms(),
    )


def standalone_task_result_paths(task: str) -> tuple[Path, Path]:
    task_results_dir = RESULTS_DIR / "tasks" / task
    return (
        task_results_dir / "benchmark_results.json",
        task_results_dir / "benchmark_results.csv",
    )


def config_for_standalone_task(
    config: BenchmarkConfig, cases: Sequence[BenchmarkCase]
) -> BenchmarkConfig:
    if not cases:
        return config
    if config.results_json != RESULTS_JSON or config.results_csv != RESULTS_CSV:
        return config

    task_names = {case.task for case in cases}
    if len(task_names) != 1:
        return config

    task = next(iter(task_names))
    results_json, results_csv = standalone_task_result_paths(task)
    return replace(config, results_json=results_json, results_csv=results_csv)


def thread_environment_snapshot() -> dict[str, str]:
    env_names = ("BENCHMARK_THREADS", *THREAD_ENVIRONMENT_VARIABLES, "XLA_FLAGS")
    return {
        env_name: os.environ[env_name]
        for env_name in env_names
        if os.environ.get(env_name)
    }


def benchmark_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("JULIA_DEPOT_PATH", str(JULIA_DEPOT_DIR))
    environment.setdefault("R_LIBS_USER", str(R_LIBS_DIR))
    environment.setdefault("BENCHMARK_THREADS", str(BENCHMARK_THREADS))
    for env_name in THREAD_ENVIRONMENT_VARIABLES:
        environment.setdefault(env_name, str(BENCHMARK_THREADS))
    return environment


def environment_payload(config: BenchmarkConfig) -> dict[str, object]:
    return {
        "python": shutil.which("python") or "python",
        "julia": shutil.which("julia"),
        "rscript": shutil.which("Rscript"),
        "platform": sys.platform,
        "jax_backend": jax.default_backend(),
        "requested_device_platform": config.requested_device_platform,
        "available_jax_platforms": list(config.available_device_platforms),
        "cpu_device": repr(config.cpu_device),
        "gpu_device": repr(config.gpu_device)
        if config.gpu_device is not None
        else None,
        "gpu_backend_available": config.gpu_device is not None,
        "julia_project": str(JULIA_PROJECT_DIR),
        "julia_depot": str(JULIA_DEPOT_DIR),
        "r_libs_user": str(R_LIBS_DIR),
        "repeats": config.repeats,
        "benchmark_threads": config.benchmark_threads,
        "thread_environment": thread_environment_snapshot(),
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        choices=["default", "cpu", "gpu", "tpu"],
        default=os.environ.get("JAXSCAPE_BENCHMARK_DEVICE", "default"),
        help=(
            "Preferred JAX backend hint. CPU cross-tool profiles always run, "
            "and GPU-capable JAX profiles emit a GPU series when available."
        ),
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--include-conefor",
        action="store_true",
        help="Include the optional Conefor adapter in the output artifacts.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Exit with a non-zero status when a required benchmark profile is "
            "skipped or fails."
        ),
    )
    parser.add_argument("--results-json", type=Path, default=RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=RESULTS_CSV)


def config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return build_config(
        repeats=args.repeats,
        requested_device_platform=args.device,
        include_conefor=getattr(args, "include_conefor", False),
        require_complete=getattr(args, "require_complete", False),
        results_json=args.results_json,
        results_csv=args.results_csv,
    )


def default_jaxscape_config() -> BenchmarkConfig:
    return build_config(
        repeats=REPEATS,
        requested_device_platform=os.environ.get(
            "JAXSCAPE_BENCHMARK_DEVICE", "default"
        ),
        include_conefor=False,
        require_complete=False,
        results_json=RESULTS_JSON,
        results_csv=RESULTS_CSV,
    )


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


def measure_runtime(
    fun: Callable[..., Any], *args: Any, repeats: int
) -> tuple[list[float], Any]:
    result = fun(*args)
    jax.block_until_ready(result)
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fun(*args)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)
    return timings, result


def ok_record(
    task: str,
    tool: str,
    software: str,
    scenario: str,
    timings: list[float],
    note: str = "",
) -> BenchmarkRecord:
    return BenchmarkRecord(
        task,
        tool,
        software,
        scenario,
        "ok",
        statistics.median(timings),
        timings,
        {},
        note,
    )


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
    return BenchmarkRecord(
        task,
        tool,
        software,
        scenario,
        "ok",
        statistics.median(timings),
        timings,
        metrics,
        note,
    )


def skip_record(
    task: str, tool: str, software: str, scenario: str, note: str
) -> BenchmarkRecord:
    return BenchmarkRecord(
        task, tool, software, scenario, "skipped", None, [], {}, note
    )


def failed_record(
    task: str, tool: str, software: str, scenario: str, note: str
) -> BenchmarkRecord:
    return BenchmarkRecord(task, tool, software, scenario, "failed", None, [], {}, note)


def gpu_placeholder_record(
    case: BenchmarkCase, tool: str, software: str
) -> BenchmarkRecord:
    return skip_record(
        case.task,
        backend_tool_label(tool, "gpu"),
        software,
        case.name,
        GPU_PLACEHOLDER_NOTE,
    )


def lower_triangle_values(matrix: jax.Array) -> jax.Array:
    indices = jnp.tril_indices(matrix.shape[0], k=-1)
    return matrix[indices]


def fit_error_metrics(
    predicted_values: jax.Array, target_values: jax.Array
) -> dict[str, float]:
    residual = predicted_values - target_values
    rmse = jnp.sqrt(jnp.mean(jnp.square(residual)))
    reference_scale = jnp.sqrt(jnp.mean(jnp.square(target_values)))
    relative_rmse = rmse / jnp.maximum(
        reference_scale, jnp.array(1e-12, dtype=target_values.dtype)
    )
    return {
        "rmse": float(rmse),
        "relative_rmse": float(relative_rmse),
    }


def normalise_nonnegative(array: np.ndarray) -> np.ndarray:
    clipped = np.nan_to_num(
        np.asarray(array, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    clipped = np.maximum(clipped, 0.0)
    total = clipped.sum()
    if total <= 0:
        return clipped
    return clipped / total


def centrality_alignment_metrics(
    candidate: jax.Array | np.ndarray, reference: jax.Array | np.ndarray
) -> dict[str, float]:
    candidate_norm = normalise_nonnegative(np.asarray(candidate, dtype=float).ravel())
    reference_norm = normalise_nonnegative(np.asarray(reference, dtype=float).ravel())
    if not np.any(candidate_norm) or not np.any(reference_norm):
        return {
            "cosine_similarity": 0.0,
            "normalised_l1": 1.0,
            "normalised_rmse": 1.0,
        }

    cosine_similarity = float(
        np.dot(candidate_norm, reference_norm)
        / (np.linalg.norm(candidate_norm) * np.linalg.norm(reference_norm))
    )
    return {
        "cosine_similarity": cosine_similarity,
        "normalised_l1": float(np.abs(candidate_norm - reference_norm).sum()),
        "normalised_rmse": float(
            np.sqrt(np.mean(np.square(candidate_norm - reference_norm)))
        ),
    }


def benchmark_graph_volume(cost_surface: jax.Array) -> jax.Array:
    horizontal = cost_conductance(cost_surface[:, :-1], cost_surface[:, 1:])
    vertical = cost_conductance(cost_surface[:-1, :], cost_surface[1:, :])
    return 2.0 * (jnp.sum(horizontal) + jnp.sum(vertical))


def write_case_payload(case: BenchmarkCase, directory: Path) -> Path:
    payload = {
        "name": case.name,
        "task": case.task,
        "size_label": case.size_label,
        "grid_size": case.grid_size,
        "seed": case.seed,
        "raster": case.raster,
        "points": case.points,
    }
    payload_path = directory / f"{case.name}.json"
    payload_path.write_text(json.dumps(payload, indent=2))
    return payload_path


def _walltime_seconds_from_env(env_name: str) -> float | None:
    raw_value = os.environ.get(env_name)
    if raw_value is None:
        return None
    try:
        return max(1.0, float(raw_value))
    except ValueError as error:
        message = (
            f"{env_name} must be numeric, received {raw_value!r}."
        )
        raise RuntimeError(message) from error


def external_tool_walltime_seconds(software: str) -> float:
    for env_name in (
        EXTERNAL_WALLTIME_ENV_BY_SOFTWARE.get(software),
        EXTERNAL_WALLTIME_ENV,
        "JAXSCAPE_BENCHMARK_WALLTIME_SECONDS",
    ):
        if env_name is None:
            continue
        resolved = _walltime_seconds_from_env(env_name)
        if resolved is not None:
            return resolved
    return DEFAULT_EXTERNAL_WALLTIME_SECONDS


def external_failure_note(
    tool_label: str,
    *,
    timeout_seconds: float | None = None,
    returncode: int | None = None,
) -> str:
    if timeout_seconds is not None:
        return (
            f"Exceeded walltime of {format_walltime_seconds(timeout_seconds)} "
            f"while running {tool_label}."
        )
    if returncode is not None and returncode < 0:
        if returncode == -9:
            return (
                f"Process was killed while running {tool_label}; this often "
                "indicates an out-of-memory condition."
            )
        return f"Process exited from signal {-returncode} while running {tool_label}."
    return f"External tool exited with status {returncode} while running {tool_label}."


def run_external_subprocess(
    *,
    args: Sequence[str],
    case: BenchmarkCase,
    tool_label: str,
    software: str,
    env: dict[str, str] | None = None,
) -> str | None:
    configure_logging()
    timeout_seconds = external_tool_walltime_seconds(software)
    LOGGER.info(
        "Starting external benchmark for case=%s tool=%s walltime=%s",
        case.name,
        tool_label,
        format_walltime_seconds(timeout_seconds),
    )
    start_time = time.perf_counter()
    try:
        completed = subprocess.run(
            list(args),
            check=False,
            env=env,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        LOGGER.error(
            "External benchmark timed out for case=%s tool=%s after %s",
            case.name,
            tool_label,
            format_walltime_seconds(timeout_seconds),
        )
        return external_failure_note(tool_label, timeout_seconds=timeout_seconds)

    elapsed_seconds = time.perf_counter() - start_time
    if completed.returncode != 0:
        note = external_failure_note(tool_label, returncode=completed.returncode)
        LOGGER.error(
            "External benchmark failed for case=%s tool=%s after %.3fs: %s",
            case.name,
            tool_label,
            elapsed_seconds,
            note,
        )
        return note

    LOGGER.info(
        "Completed external benchmark for case=%s tool=%s in %.3fs",
        case.name,
        tool_label,
        elapsed_seconds,
    )
    return None


def log_case_start(case: BenchmarkCase) -> None:
    configure_logging()
    LOGGER.info(
        "Starting benchmark case task=%s scenario=%s grid=%sx%s points=%s",
        case.task,
        case.name,
        case.grid_size,
        case.grid_size,
        len(case.points),
    )


def log_case_completion(case: BenchmarkCase, records: Sequence[BenchmarkRecord]) -> None:
    status_counts = Counter(record.status for record in records)
    counts = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    )
    LOGGER.info(
        "Completed benchmark case task=%s scenario=%s records=%s",
        case.task,
        case.name,
        counts or "none",
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


def ascii_points_from_points(
    shape: tuple[int, int], points: list[tuple[int, int]], nodata: int = -9999
) -> str:
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


def circuitscape_ini(
    cellmap: Path, points: Path, output_prefix: Path, solver_name: str
) -> str:
    return (
        textwrap.dedent(
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
        ).strip()
        + "\n"
    )


def run_circuitscape_resistance(
    case: BenchmarkCase, config: BenchmarkConfig, solver_name: str, tool_label: str
) -> BenchmarkRecord:
    if shutil.which("julia") is None:
        return skip_record(
            case.task,
            tool_label,
            "Circuitscape.jl",
            case.name,
            "Julia is not available in this environment.",
        )

    with tempfile.TemporaryDirectory(prefix="jaxscape-circuitscape-") as tmpdir:
        workspace = Path(tmpdir)
        cellmap = workspace / "cellmap.asc"
        points = workspace / "points.asc"
        output_prefix = workspace / "result"
        ini = workspace / "config.ini"
        output = workspace / "circuitscape.json"
        cellmap.write_text(ascii_grid_from_raster(case.raster))
        points.write_text(
            ascii_points_from_points(
                (len(case.raster), len(case.raster[0])), case.points
            )
        )
        ini.write_text(circuitscape_ini(cellmap, points, output_prefix, solver_name))
        note = run_external_subprocess(
            args=[
                "julia",
                f"--project={JULIA_PROJECT_DIR}",
                str(CIRCUITSCAPE_SCRIPT),
                str(ini),
                str(config.repeats),
                str(output),
            ],
            case=case,
            tool_label=tool_label,
            software="Circuitscape.jl",
            env=benchmark_environment(),
        )
        if note is not None:
            return failed_record(case.task, tool_label, "Circuitscape.jl", case.name, note)
        try:
            payload = json.loads(output.read_text())
        except Exception as error:
            return failed_record(
                case.task,
                tool_label,
                "Circuitscape.jl",
                case.name,
                f"Failed to decode Circuitscape output: {error}",
            )
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


def run_gdistance_payload(
    case: BenchmarkCase, config: BenchmarkConfig, mode: str
) -> dict[str, Any]:
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
        tool_label = {
            "resistance_distance": "gdistance / commuteDistance",
            "least_cost_path": "gdistance / costDistance",
            "least_cost_centrality": "gdistance / shortestPath",
            "resistance_centrality": "gdistance / passage",
        }.get(mode, f"gdistance / {mode}")
        note = run_external_subprocess(
            args=[
                rscript,
                str(GDISTANCE_SCRIPT),
                mode,
                str(payload_path),
                str(config.repeats),
                str(output),
            ],
            case=case,
            tool_label=tool_label,
            software="gdistance",
            env=benchmark_environment(),
        )
        if note is not None:
            return {
                "status": "failed",
                "timings_seconds": [],
                "median_seconds": None,
                "metrics": {},
                "note": note,
            }
        try:
            return json.loads(output.read_text())
        except Exception as error:
            return {
                "status": "failed",
                "timings_seconds": [],
                "median_seconds": None,
                "metrics": {},
                "note": f"Failed to decode gdistance output: {error}",
            }


def gdistance_record_from_payload(
    case: BenchmarkCase,
    tool_label: str,
    payload: dict[str, Any],
    *,
    note_suffix: str = "",
) -> BenchmarkRecord:
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
        return skip_record(
            task_label,
            "Conefor",
            "Conefor",
            case.name,
            "Set CONEFOR_BIN to enable the Conefor adapter.",
        )
    output = Path(tempfile.mkdtemp(prefix="jaxscape-conefor-")) / "conefor.json"
    note = run_external_subprocess(
        args=[str(CONEFOR_SCRIPT), conefor_bin, task_label, str(output)],
        case=case,
        tool_label="Conefor",
        software="Conefor",
    )
    if note is not None:
        return failed_record(task_label, "Conefor", "Conefor", case.name, note)
    try:
        payload = json.loads(output.read_text())
    except Exception as error:
        return failed_record(
            task_label,
            "Conefor",
            "Conefor",
            case.name,
            f"Failed to decode Conefor output: {error}",
        )
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


def run_resistancega_inverse(
    case: BenchmarkCase, config: BenchmarkConfig
) -> BenchmarkRecord:
    resistancega_settings = load_inverse_benchmark_settings().resistancega
    rscript = shutil.which("Rscript")
    if rscript is None:
        return skip_record(
            case.task,
            "ResistanceGA",
            "ResistanceGA",
            case.name,
            "Rscript is not installed in this environment.",
        )
    with tempfile.TemporaryDirectory(prefix="jaxscape-resistancega-") as tmpdir:
        workspace = Path(tmpdir)
        output = workspace / "resistancega.json"
        payload_path = write_case_payload(case, workspace)
        note = run_external_subprocess(
            args=[
                rscript,
                str(RESISTANCE_GA_SCRIPT),
                str(payload_path),
                str(config.repeats),
                str(output),
                resistancega_settings.method,
                resistancega_settings.transformation,
                str(resistancega_settings.pop_size),
                str(resistancega_settings.maxiter),
                str(resistancega_settings.run),
                str(resistancega_settings.max_cont),
                str(resistancega_settings.seed),
            ],
            case=case,
            tool_label="ResistanceGA",
            software="ResistanceGA",
            env=benchmark_environment(),
        )
        if note is not None:
            return failed_record(
                case.task,
                "ResistanceGA",
                "ResistanceGA",
                case.name,
                note,
            )
        try:
            payload = json.loads(output.read_text())
        except Exception as error:
            return failed_record(
                case.task,
                "ResistanceGA",
                "ResistanceGA",
                case.name,
                f"Failed to decode ResistanceGA output: {error}",
            )
    payload_metrics = payload.get("metrics", {})
    payload_metrics.update(
        {
            "ga_method": resistancega_settings.method,
            "ga_transformation": resistancega_settings.transformation,
            "ga_pop_size": resistancega_settings.pop_size,
            "ga_maxiter": resistancega_settings.maxiter,
            "ga_run": resistancega_settings.run,
            "ga_max_cont": resistancega_settings.max_cont,
            "ga_seed": resistancega_settings.seed,
            "budget_policy": "fixed_ga_iterations",
        }
    )
    return BenchmarkRecord(
        case.task,
        "ResistanceGA",
        "ResistanceGA",
        case.name,
        payload["status"],
        payload.get("median_seconds"),
        payload.get("timings_seconds", []),
        payload_metrics,
        payload.get("note", ""),
    )


def case_payload(cases: list[BenchmarkCase]) -> list[dict[str, object]]:
    return [
        {
            "name": case.name,
            "task": case.task,
            "size_label": case.size_label,
            "grid_size": case.grid_size,
            "seed": case.seed,
            "point_count": len(case.points),
            "points": case.points,
        }
        for case in cases
    ]


def write_results(
    records: list[BenchmarkRecord],
    config: BenchmarkConfig,
    *,
    cases: list[BenchmarkCase] | None = None,
) -> None:
    result_cases = iter_cases(CASES) if cases is None else cases
    config.results_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": environment_payload(config),
        "cases": case_payload(result_cases),
        "records": [asdict(record) for record in records],
    }
    config.results_json.write_text(json.dumps(payload, indent=2))
    with config.results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "tool",
                "software",
                "scenario",
                "status",
                "median_seconds",
                "timings_seconds",
                "metrics",
                "note",
            ],
        )
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["timings_seconds"] = json.dumps(row["timings_seconds"])
            row["metrics"] = json.dumps(row["metrics"])
            writer.writerow(row)


def required_records(config: BenchmarkConfig) -> dict[tuple[str, str, str], set[str]]:
    required: dict[tuple[str, str, str], set[str]] = {}
    for task, labels in REQUIRED_BASE_TOOL_LABELS_BY_TASK.items():
        gpu_capable_labels = GPU_CAPABLE_TOOL_LABELS_BY_TASK.get(task, set())
        for case in CASES_BY_TASK[task]:
            for tool in labels:
                if tool in gpu_capable_labels:
                    required[(task, case.name, backend_tool_label(tool, "cpu"))] = {
                        "ok"
                    }
                    required[(task, case.name, backend_tool_label(tool, "gpu"))] = (
                        {"ok"} if config.gpu_device is not None else {"skipped"}
                    )
                else:
                    required[(task, case.name, tool)] = {"ok"}
    return required


def validate_records(records: list[BenchmarkRecord], config: BenchmarkConfig) -> None:
    if not config.require_complete:
        return

    required = required_records(config)
    observed = {
        (record.task, record.scenario, record.tool): record for record in records
    }
    missing = sorted(set(required) - set(observed))
    if missing:
        summary = ", ".join(
            f"{tool}/{task}/{scenario}" for task, scenario, tool in missing
        )
        raise RuntimeError(
            f"Benchmark suite is incomplete: missing records for {summary}"
        )

    incomplete = [
        record
        for key, record in observed.items()
        if key in required and record.status not in required[key]
    ]
    if incomplete:
        summary = "; ".join(
            (
                f"{record.tool}/{record.task}/{record.scenario}: "
                f"{record.note or record.status}"
            )
            for record in incomplete
        )
        raise RuntimeError(f"Benchmark suite is incomplete: {summary}")


def collect_resistance_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    from benchmark.jaxscape import resistance_distance

    gdistance_payload = run_gdistance_payload(case, config, "resistance_distance")
    graph_volume = float(jax.device_get(benchmark_graph_volume(as_array(case.raster))))
    if (
        gdistance_payload["status"] == "ok"
        and gdistance_payload.get("metrics", {}).get("distance_matrix") is not None
        and graph_volume > 0
    ):
        commute_matrix = np.asarray(
            gdistance_payload["metrics"]["distance_matrix"], dtype=float
        )
        gdistance_payload["metrics"] = {
            "graph_volume": graph_volume,
            "distance_matrix": (commute_matrix / graph_volume).tolist(),
        }

    records = [
        *resistance_distance.collect_jaxscape_resistance_results(case, config),
        gdistance_record_from_payload(
            case,
            "gdistance / commuteDistance",
            gdistance_payload,
            note_suffix=(
                "Reported values are rescaled from commute time to effective "
                "resistance by dividing by the graph volume."
            ),
        ),
        run_circuitscape_resistance(case, config, "cg+amg", "Circuitscape.jl / cg+amg"),
        run_circuitscape_resistance(
            case, config, "cholmod", "Circuitscape.jl / cholmod"
        ),
    ]
    if config.include_conefor:
        records.append(run_conefor_placeholder(case, case.task))
    return records


def collect_least_cost_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    from benchmark.jaxscape import least_cost_path

    records = least_cost_path.collect_jaxscape_lcp_results(case, config)
    records.append(
        gdistance_record_from_payload(
            case,
            "gdistance / costDistance",
            run_gdistance_payload(case, config, "least_cost_path"),
        )
    )
    if config.include_conefor:
        records.append(run_conefor_placeholder(case, case.task))
    return records


def collect_sensitivity_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    from benchmark.jaxscape import sensitivity_analysis

    shortest_path_payload = run_gdistance_payload(case, config, "least_cost_centrality")
    shortest_path_reference = (
        gdistance_centrality_raster(shortest_path_payload)
        if shortest_path_payload.get("status") == "ok"
        else None
    )
    resistance_payload = run_gdistance_payload(case, config, "resistance_centrality")
    resistance_reference = (
        gdistance_centrality_raster(resistance_payload)
        if resistance_payload.get("status") == "ok"
        else None
    )

    records = sensitivity_analysis.collect_jaxscape_sensitivity_results(
        case,
        config,
        shortest_path_reference=shortest_path_reference,
        resistance_reference=resistance_reference,
    )
    records.append(
        gdistance_record_from_payload(
            case, "gdistance / shortestPath", shortest_path_payload
        )
    )
    records.append(
        gdistance_record_from_payload(case, "gdistance / passage", resistance_payload)
    )
    return records


def collect_inverse_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    from benchmark.jaxscape import inverse_landscape_genetics

    return [
        *inverse_landscape_genetics.collect_jaxscape_inverse_results(case, config),
        run_resistancega_inverse(case, config),
    ]


def collect_results(config: BenchmarkConfig) -> list[BenchmarkRecord]:
    configure_logging()
    records: list[BenchmarkRecord] = []
    for case in CASES["resistance"]:
        log_case_start(case)
        case_records = collect_resistance_results(case, config)
        records.extend(case_records)
        log_case_completion(case, case_records)
    for case in CASES["lcp"]:
        log_case_start(case)
        case_records = collect_least_cost_results(case, config)
        records.extend(case_records)
        log_case_completion(case, case_records)
    for case in CASES["sensitivity"]:
        log_case_start(case)
        case_records = collect_sensitivity_results(case, config)
        records.extend(case_records)
        log_case_completion(case, case_records)
    for case in CASES["inverse"]:
        log_case_start(case)
        case_records = collect_inverse_results(case, config)
        records.extend(case_records)
        log_case_completion(case, case_records)
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    config = config_from_args(args)
    records = collect_results(config)
    validate_records(records, config)
    write_results(records, config)
    print(json.dumps({"records": [asdict(record) for record in records]}, indent=2))


if __name__ == "__main__":
    main()
