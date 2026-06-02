"""JAXScape least-cost path benchmark task runner."""

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
from jaxscape import GridGraph, LCPDistance

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
)


LOGGER = logging.getLogger(__name__)


@eqx.filter_jit
def _jaxscape_lcp(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return LCPDistance()(grid, nodes=nodes)


def _run_jaxscape_lcp_direct(
    case: BenchmarkCase,
    *,
    device: jax.Device,
    repeats: int,
    tool_label: str,
) -> BenchmarkRecord:
    LOGGER.info(
        "Starting least-cost benchmark for case=%s tool=%s backend=%s repeats=%s",
        case.name,
        tool_label,
        device.platform,
        repeats,
    )
    try:
        permeability = jax.device_put(as_array(case.raster), device)
        nodes = jax.device_put(as_point_array(case.points), device)
        timings, _ = measure_runtime(
            _jaxscape_lcp, permeability, nodes, repeats=repeats
        )
    except MemoryError as error:
        note = f"Out-of-memory while running least-cost benchmark: {error}"
        LOGGER.exception(
            "OOM in least-cost benchmark for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)
    except Exception as error:
        note = str(error)
        if looks_like_oom(note):
            note = f"Out-of-memory while running least-cost benchmark: {note}"
        LOGGER.exception(
            "Least-cost benchmark failed for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)

    record = ok_record(case.task, tool_label, "JAXScape", case.name, timings)
    LOGGER.info(
        "Completed least-cost benchmark for case=%s tool=%s median=%.6fs",
        case.name,
        tool_label,
        record.median_seconds,
    )
    return record


def lcp_worker_payload(
    case: BenchmarkCase, backend: str, repeats: int
) -> dict[str, Any]:
    return {
        "case_name": case.name,
        "backend": backend,
        "repeats": repeats,
    }


def run_worker(payload: dict[str, Any]) -> BenchmarkRecord:
    case = case_by_name(CASES["lcp"], payload["case_name"])
    backend = payload["backend"]
    repeats = int(payload["repeats"])
    tool_label = backend_tool_label("JAXScape", backend)
    return _run_jaxscape_lcp_direct(
        case,
        device=device_for_backend(backend),
        repeats=repeats,
        tool_label=tool_label,
    )


def run_jaxscape_lcp(
    case: BenchmarkCase, config: BenchmarkConfig, *, device: jax.Device, tool_label: str
) -> BenchmarkRecord:
    del device
    backend = "gpu" if "(GPU)" in tool_label else "cpu"
    return run_worker_subprocess(
        script_path=Path(__file__).resolve(),
        payload=lcp_worker_payload(case, backend, config.repeats),
        case_task=case.task,
        case_name=case.name,
        tool_label=tool_label,
        software="JAXScape",
        walltime_seconds=benchmark_walltime_seconds(),
        logger=LOGGER,
    )


def collect_jaxscape_lcp_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    records = [
        run_jaxscape_lcp(
            case,
            config,
            device=config.cpu_device,
            tool_label=backend_tool_label("JAXScape", "cpu"),
        )
    ]
    if config.gpu_device is None:
        records.append(gpu_placeholder_record(case, "JAXScape", "JAXScape"))
    else:
        records.append(
            run_jaxscape_lcp(
                case,
                config,
                device=config.gpu_device,
                tool_label=backend_tool_label("JAXScape", "gpu"),
            )
        )
    return records


def collect_task_results(config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["lcp"]:
        records.extend(collect_jaxscape_lcp_results(case, config))
    return records


def main() -> None:
    run_standalone_task(
        collect_task_results=collect_task_results,
        config_factory=default_jaxscape_config,
        cases=CASES["lcp"],
        worker_handler=run_worker,
    )


if __name__ == "__main__":
    main()
