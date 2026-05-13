"""JAXScape least-cost path benchmark task runner."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

from benchmark.jaxscape.utils import (
    configure_standalone_environment,
    run_standalone_task,
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
    gpu_placeholder_record,
    measure_runtime,
    ok_record,
)


@eqx.filter_jit
def _jaxscape_lcp(permeability: jax.Array, nodes: jax.Array) -> jax.Array:
    grid = GridGraph(permeability, fun=cost_conductance)
    return LCPDistance()(grid, nodes=nodes)


def run_jaxscape_lcp(
    case: BenchmarkCase, config: BenchmarkConfig, *, device: jax.Device, tool_label: str
) -> BenchmarkRecord:
    permeability = jax.device_put(as_array(case.raster), device)
    nodes = jax.device_put(as_point_array(case.points), device)
    timings, _ = measure_runtime(
        _jaxscape_lcp, permeability, nodes, repeats=config.repeats
    )
    return ok_record(case.task, tool_label, "JAXScape", case.name, timings)


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
    )


if __name__ == "__main__":
    main()
