"""JAXScape resistance-distance benchmark task runner."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

from benchmark.jaxscape.utils import (
    benchmark_walltime_seconds,
    case_by_name,
    configure_standalone_environment,
    device_for_backend,
    looks_like_oom,
    run_standalone_task,
    run_worker_subprocess,
)


configure_standalone_environment()

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxscape import GridGraph, ResistanceDistance

from benchmark.benchmark_distances import (
    as_point_array,
    backend_tool_label,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    CASES,
    cost_conductance,
    default_jaxscape_config,
    failed_record,
    gpu_placeholder_record,
    measure_runtime,
    ok_record,
    skip_record,
)
from benchmark.jaxscape.resistance_profile_support import (
    make_amjaxcg_solver,
    make_cholmod_solver,
    make_pyamg_solver,
    make_spielman_method,
    ResistanceProfile,
)


JAXSCAPE_RESISTANCE_PROFILES = (
    ResistanceProfile(
        "pinv_f32",
        "JAXScape / pinv / f32",
        gpu_capable=True,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "pinv_f64",
        "JAXScape / pinv / f64",
        gpu_capable=True,
        dtype=jnp.float64,
    ),
    ResistanceProfile("pyamg", "JAXScape / PyAMG", solver_factory=make_pyamg_solver),
    ResistanceProfile(
        "cholmod_f32",
        "JAXScape / CholmodSolver / f32",
        solver_factory=make_cholmod_solver,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "cholmod_f64",
        "JAXScape / CholmodSolver / f64",
        solver_factory=make_cholmod_solver,
        dtype=jnp.float64,
    ),
    ResistanceProfile(
        "amjaxcg_f32",
        "JAXScape / AMJaxCGSolver / f32",
        solver_factory=make_amjaxcg_solver,
        gpu_capable=True,
        requires_preparation=True,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "amjaxcg_f64",
        "JAXScape / AMJaxCGSolver / f64",
        solver_factory=make_amjaxcg_solver,
        gpu_capable=True,
        requires_preparation=True,
        dtype=jnp.float64,
    ),
    ResistanceProfile(
        "approx_pinv_f32",
        "JAXScape / approx pinv / f32",
        method_factory=make_spielman_method,
        gpu_capable=True,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "approx_pinv_f64",
        "JAXScape / approx pinv / f64",
        method_factory=make_spielman_method,
        gpu_capable=True,
        dtype=jnp.float64,
    ),
    ResistanceProfile(
        "approx_amjaxcg_f32",
        "JAXScape / approx AMJaxCGSolver / f32",
        solver_factory=make_amjaxcg_solver,
        method_factory=make_spielman_method,
        gpu_capable=True,
        requires_preparation=True,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "approx_amjaxcg_f64",
        "JAXScape / approx AMJaxCGSolver / f64",
        solver_factory=make_amjaxcg_solver,
        method_factory=make_spielman_method,
        gpu_capable=True,
        requires_preparation=True,
        dtype=jnp.float64,
    ),
    ResistanceProfile(
        "approx_cholmod_f32",
        "JAXScape / approx CholmodSolver / f32",
        solver_factory=make_cholmod_solver,
        method_factory=make_spielman_method,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "approx_cholmod_f64",
        "JAXScape / approx CholmodSolver / f64",
        solver_factory=make_cholmod_solver,
        method_factory=make_spielman_method,
        dtype=jnp.float64,
    ),
)
PROFILE_BY_KEY = {profile.key: profile for profile in JAXSCAPE_RESISTANCE_PROFILES}
LOGGER = logging.getLogger(__name__)


def _run_resistance_profile_direct(
    case: BenchmarkCase,
    tool_label: str,
    *,
    device: jax.Device,
    repeats: int,
    solver_factory: Callable[[], Any] | None = None,
    method_factory: Callable[[], Any] | None = None,
    requires_preparation: bool = False,
    dtype: Any = jnp.float32,
) -> BenchmarkRecord:
    LOGGER.info(
        "Starting resistance benchmark for case=%s tool=%s backend=%s repeats=%s",
        case.name,
        tool_label,
        device.platform,
        repeats,
    )
    try:
        solver = None if solver_factory is None else solver_factory()
        method = None if method_factory is None else method_factory()
    except ImportError as error:
        LOGGER.warning(
            "Skipping resistance benchmark for case=%s tool=%s: %s",
            case.name,
            tool_label,
            error,
        )
        return skip_record(case.task, tool_label, "JAXScape", case.name, str(error))
    except Exception as error:
        LOGGER.exception(
            "Solver construction failed for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, str(error))

    try:
        dtype_context = jax.enable_x64() if dtype == jnp.float64 else nullcontext()
        with dtype_context:
            permeability = jax.device_put(jnp.asarray(case.raster, dtype=dtype), device)
            nodes = jax.device_put(as_point_array(case.points), device)
            distance = (
                ResistanceDistance(solver=solver)
                if method is None
                else ResistanceDistance(solver=solver, method=method)
            )
            if requires_preparation:
                grid = GridGraph(permeability, fun=cost_conductance)
                state = distance.init(grid)
                timings, _ = measure_runtime(
                    _jaxscape_prepared_resistance,
                    distance,
                    grid,
                    nodes,
                    state,
                    repeats=repeats,
                )
            else:
                timings, _ = measure_runtime(
                    _jaxscape_resistance_with_distance,
                    distance,
                    permeability,
                    nodes,
                    repeats=repeats,
                )
    except MemoryError as error:
        note = f"Out-of-memory while running resistance benchmark: {error}"
        LOGGER.exception(
            "OOM in resistance benchmark for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)
    except Exception as error:
        note = str(error)
        if looks_like_oom(note):
            note = f"Out-of-memory while running resistance benchmark: {note}"
        LOGGER.exception(
            "Resistance benchmark failed for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)

    record = ok_record(case.task, tool_label, "JAXScape", case.name, timings)
    LOGGER.info(
        "Completed resistance benchmark for case=%s tool=%s median=%.6fs",
        case.name,
        tool_label,
        record.median_seconds,
    )
    return record


def profile_for_tool_label(tool_label: str) -> ResistanceProfile:
    for profile in JAXSCAPE_RESISTANCE_PROFILES:
        candidate_labels = {
            profile.tool,
            backend_tool_label(profile.tool, "cpu"),
            backend_tool_label(profile.tool, "gpu"),
        }
        if tool_label in candidate_labels:
            return profile
    raise KeyError(f"Unknown resistance tool label {tool_label!r}.")


def resistance_worker_payload(
    case: BenchmarkCase, profile: ResistanceProfile, backend: str, repeats: int
) -> dict[str, Any]:
    return {
        "case_name": case.name,
        "profile_key": profile.key,
        "backend": backend,
        "repeats": repeats,
    }


def run_worker(payload: dict[str, Any]) -> BenchmarkRecord:
    case = case_by_name(CASES["resistance"], payload["case_name"])
    profile = PROFILE_BY_KEY[payload["profile_key"]]
    backend = payload["backend"]
    repeats = int(payload["repeats"])
    tool_label = (
        profile.tool
        if backend == "cpu" and not profile.gpu_capable
        else backend_tool_label(profile.tool, backend)
    )
    device = device_for_backend(backend)
    return _run_resistance_profile_direct(
        case,
        tool_label,
        device=device,
        repeats=repeats,
        solver_factory=profile.solver_factory,
        method_factory=profile.method_factory,
        requires_preparation=profile.requires_preparation,
        dtype=profile.dtype,
    )


@eqx.filter_jit
def _jaxscape_resistance_with_distance(
    distance: ResistanceDistance, permeability: jax.Array, nodes: jax.Array
) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return distance(grid, nodes=nodes)


@eqx.filter_jit
def _jaxscape_prepared_resistance(
    distance: ResistanceDistance, grid: GridGraph, nodes: jax.Array, state: Any
) -> jax.Array:
    return distance(grid, nodes=nodes, state=state)


def run_jaxscape_resistance_profile(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    tool_label: str,
    *,
    device: jax.Device,
    solver_factory: Callable[[], Any] | None = None,
    requires_preparation: bool = False,
) -> BenchmarkRecord:
    del solver_factory, requires_preparation, device
    profile = profile_for_tool_label(tool_label)
    backend = "gpu" if "(GPU)" in tool_label else "cpu"
    return run_worker_subprocess(
        script_path=Path(__file__).resolve(),
        payload=resistance_worker_payload(case, profile, backend, config.repeats),
        case_task=case.task,
        case_name=case.name,
        tool_label=tool_label,
        software="JAXScape",
        walltime_seconds=benchmark_walltime_seconds(),
        logger=LOGGER,
    )


def selected_profiles(
    profile_keys: list[str] | None = None,
) -> tuple[ResistanceProfile, ...]:
    if profile_keys is None:
        return JAXSCAPE_RESISTANCE_PROFILES
    return tuple(PROFILE_BY_KEY[key] for key in profile_keys)


def collect_jaxscape_resistance_results(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    profile_keys: list[str] | None = None,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for profile in selected_profiles(profile_keys):
        if profile.gpu_capable:
            records.append(
                run_jaxscape_resistance_profile(
                    case,
                    config,
                    backend_tool_label(profile.tool, "cpu"),
                    device=config.cpu_device,
                    solver_factory=profile.solver_factory,
                    requires_preparation=profile.requires_preparation,
                )
            )
            if config.gpu_device is None:
                records.append(gpu_placeholder_record(case, profile.tool, "JAXScape"))
            else:
                records.append(
                    run_jaxscape_resistance_profile(
                        case,
                        config,
                        backend_tool_label(profile.tool, "gpu"),
                        device=config.gpu_device,
                        solver_factory=profile.solver_factory,
                        requires_preparation=profile.requires_preparation,
                    )
                )
        else:
            records.append(
                run_jaxscape_resistance_profile(
                    case,
                    config,
                    profile.tool,
                    device=config.cpu_device,
                    solver_factory=profile.solver_factory,
                    requires_preparation=profile.requires_preparation,
                )
            )
    return records


def collect_task_results(
    config: BenchmarkConfig, *, profile_keys: list[str] | None = None
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["resistance"]:
        records.extend(
            collect_jaxscape_resistance_results(case, config, profile_keys=profile_keys)
        )
    return records


def main() -> None:
    run_standalone_task(
        collect_task_results=collect_task_results,
        config_factory=default_jaxscape_config,
        cases=CASES["resistance"],
        worker_handler=run_worker,
    )


if __name__ == "__main__":
    main()
