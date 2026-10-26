"""JAXScape free-energy distance benchmark task runner."""

from __future__ import annotations

import logging
import sys
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
import lineax as lx
from jaxscape import FreeEnergyDistance, GridGraph

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
    ok_record_with_metrics,
)


LOGGER = logging.getLogger(__name__)
TOOL = "JAXScape / CG / f64"
THETA = 0.1
CG_SETTINGS: dict[str, Any] = dict(
    rtol=1e-10, atol=1e-90, max_steps=10_000, stabilise_every=None
)


@eqx.filter_jit
def _jaxscape_free_energy(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return FreeEnergyDistance(THETA, solver=lx.CG(**CG_SETTINGS))(grid, nodes=nodes)


def _run_jaxscape_free_energy_direct(
    case: BenchmarkCase,
    *,
    device: jax.Device,
    repeats: int,
    tool_label: str,
) -> BenchmarkRecord:
    LOGGER.info(
        "Starting free-energy benchmark for case=%s tool=%s backend=%s repeats=%s",
        case.name,
        tool_label,
        device.platform,
        repeats,
    )
    try:
        with jax.enable_x64():
            permeability = jax.device_put(
                jnp.asarray(case.raster, dtype=jnp.float64), device
            )
            nodes = jax.device_put(as_point_array(case.points), device)
            timings, distances = measure_runtime(
                _jaxscape_free_energy, permeability, nodes, repeats=repeats
            )
            if not bool(jnp.all(jnp.isfinite(distances))):
                raise ValueError("Free-energy distances must be finite.")
    except MemoryError as error:
        note = f"Out-of-memory while running free-energy benchmark: {error}"
        LOGGER.exception(
            "OOM in free-energy benchmark for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)
    except Exception as error:
        note = str(error)
        if looks_like_oom(note):
            note = f"Out-of-memory while running free-energy benchmark: {note}"
        LOGGER.exception(
            "Free-energy benchmark failed for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)

    record = ok_record_with_metrics(
        case.task,
        tool_label,
        "JAXScape",
        case.name,
        timings,
        metrics=dict(
            theta=THETA,
            dtype="float64",
            backend=device.platform,
            solver_settings=CG_SETTINGS,
        ),
    )
    LOGGER.info(
        "Completed free-energy benchmark for case=%s tool=%s median=%.6fs",
        case.name,
        tool_label,
        record.median_seconds,
    )
    return record


def free_energy_worker_payload(
    case: BenchmarkCase, backend: str, repeats: int
) -> dict[str, Any]:
    return {
        "case_name": case.name,
        "backend": backend,
        "repeats": repeats,
    }


def run_worker(payload: dict[str, Any]) -> BenchmarkRecord:
    case = case_by_name(CASES["free_energy"], payload["case_name"])
    backend = payload["backend"]
    repeats = int(payload["repeats"])
    tool_label = backend_tool_label(TOOL, backend)
    return _run_jaxscape_free_energy_direct(
        case,
        device=device_for_backend(backend),
        repeats=repeats,
        tool_label=tool_label,
    )


def run_jaxscape_free_energy(
    case: BenchmarkCase, config: BenchmarkConfig, *, device: jax.Device, tool_label: str
) -> BenchmarkRecord:
    del device
    backend = "gpu" if "(GPU)" in tool_label else "cpu"
    return run_worker_subprocess(
        script_path=Path(__file__).resolve(),
        payload=free_energy_worker_payload(case, backend, config.repeats),
        case_task=case.task,
        case_name=case.name,
        tool_label=tool_label,
        software="JAXScape",
        walltime_seconds=benchmark_walltime_seconds(),
        logger=LOGGER,
    )


def collect_jaxscape_free_energy_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    records = [
        run_jaxscape_free_energy(
            case,
            config,
            device=config.cpu_device,
            tool_label=backend_tool_label(TOOL, "cpu"),
        )
    ]
    if config.gpu_device is None:
        records.append(gpu_placeholder_record(case, TOOL, "JAXScape"))
    else:
        records.append(
            run_jaxscape_free_energy(
                case,
                config,
                device=config.gpu_device,
                tool_label=backend_tool_label(TOOL, "gpu"),
            )
        )
    return records


def collect_task_results(config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["free_energy"]:
        records.extend(collect_jaxscape_free_energy_results(case, config))
    return records


def main() -> None:
    run_standalone_task(
        collect_task_results=collect_task_results,
        config_factory=default_jaxscape_config,
        cases=CASES["free_energy"],
        worker_handler=run_worker,
    )


if __name__ == "__main__":
    main()
