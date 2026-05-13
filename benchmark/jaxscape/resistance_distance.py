"""JAXScape resistance-distance benchmark task runner."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
from jaxscape import GridGraph, ResistanceDistance

from benchmark.benchmark_distances import (
    as_array,
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


@dataclass(frozen=True)
class ResistanceProfile:
    key: str
    tool: str
    solver_factory: Callable[[], Any] | None = None
    gpu_capable: bool = False
    requires_preparation: bool = False


def make_pyamg_solver() -> Any:
    try:
        from jaxscape.solvers.pyamgsolver import PyAMGSolver
    except ImportError as error:
        raise ImportError(
            "Install the benchmark Python extra to enable the PyAMG resistance profile."
        ) from error
    return PyAMGSolver(rtol=1e-6, maxiter=50_000)


def make_cholmod_solver() -> Any:
    try:
        from jaxscape.solvers.cholmodsolver import CholmodSolver
    except ImportError as error:
        raise ImportError(
            "Install the cholespy Python extra to enable the Cholmod profile."
        ) from error
    return CholmodSolver()


def make_amjaxcg_solver() -> Any:
    try:
        from jaxscape.solvers.amjaxcgsolver import AMJaxCGSolver
    except ImportError as error:
        raise ImportError(
            "Install the amjax Python extra to enable the AMJaxCG resistance "
            "profile."
        ) from error
    return AMJaxCGSolver(rtol=1e-5, atol=1e-5, max_steps=500)


JAXSCAPE_RESISTANCE_PROFILES = (
    ResistanceProfile("pinv", "JAXScape / pinv", gpu_capable=True),
    ResistanceProfile("pyamg", "JAXScape / PyAMG", solver_factory=make_pyamg_solver),
    ResistanceProfile(
        "cholmod", "JAXScape / CholmodSolver", solver_factory=make_cholmod_solver
    ),
    ResistanceProfile(
        "amjaxcg",
        "JAXScape / AMJaxCGSolver",
        solver_factory=make_amjaxcg_solver,
        requires_preparation=True,
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
    requires_preparation: bool = False,
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
        permeability = jax.device_put(as_array(case.raster), device)
        nodes = jax.device_put(as_point_array(case.points), device)
        if solver is None:
            timings, _ = measure_runtime(
                _jaxscape_resistance, permeability, nodes, repeats=repeats
            )
        elif requires_preparation:
            grid = GridGraph(permeability, fun=cost_conductance)
            distance = ResistanceDistance(solver=solver)
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
                _jaxscape_resistance_with_solver,
                permeability,
                nodes,
                solver,
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
        requires_preparation=profile.requires_preparation,
    )


@eqx.filter_jit
def _jaxscape_resistance(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return ResistanceDistance()(grid, nodes=nodes)


@eqx.filter_jit
def _jaxscape_resistance_with_solver(
    permeability: jax.Array, nodes: jax.Array, solver: Any
) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return ResistanceDistance(solver=solver)(grid, nodes=nodes)


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
            collect_jaxscape_resistance_results(
                case, config, profile_keys=profile_keys
            )
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
