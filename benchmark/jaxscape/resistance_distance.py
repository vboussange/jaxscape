"""JAXScape resistance-distance benchmark task runner."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
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
    write_results,
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
    distance: ResistanceDistance, grid: GridGraph, nodes: jax.Array
) -> jax.Array:
    return distance(grid, nodes=nodes)


def run_jaxscape_resistance_profile(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    tool_label: str,
    *,
    device: jax.Device,
    solver_factory: Callable[[], Any] | None = None,
    requires_preparation: bool = False,
) -> BenchmarkRecord:
    try:
        solver = None if solver_factory is None else solver_factory()
    except ImportError as error:
        return skip_record(case.task, tool_label, "JAXScape", case.name, str(error))
    except Exception as error:
        return failed_record(case.task, tool_label, "JAXScape", case.name, str(error))

    permeability = jax.device_put(as_array(case.raster), device)
    nodes = jax.device_put(as_point_array(case.points), device)
    if solver is None:
        timings, _ = measure_runtime(
            _jaxscape_resistance, permeability, nodes, repeats=config.repeats
        )
    elif requires_preparation:
        grid = GridGraph(permeability, fun=cost_conductance)
        distance = ResistanceDistance(solver=solver).prepare_solver(grid)
        timings, _ = measure_runtime(
            _jaxscape_prepared_resistance,
            distance,
            grid,
            nodes,
            repeats=config.repeats,
        )
    else:
        timings, _ = measure_runtime(
            _jaxscape_resistance_with_solver,
            permeability,
            nodes,
            solver,
            repeats=config.repeats,
        )
    return ok_record(case.task, tool_label, "JAXScape", case.name, timings)


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
    config = default_jaxscape_config()
    records = collect_task_results(config)
    write_results(records, config, cases=CASES["resistance"])
    print(json.dumps({"records": [asdict(record) for record in records]}, indent=2))


if __name__ == "__main__":
    main()
