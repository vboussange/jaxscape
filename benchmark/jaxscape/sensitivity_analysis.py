"""JAXScape sensitivity-analysis benchmark task runner."""

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
import jax.numpy as jnp
import numpy as np
from jaxscape import GridGraph, LCPDistance, ResistanceDistance

from benchmark.benchmark_distances import (
    as_array,
    as_point_array,
    backend_tool_label,
    benchmark_graph_volume,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    CASES,
    centrality_alignment_metrics,
    cost_conductance,
    default_jaxscape_config,
    gpu_placeholder_record,
    measure_runtime,
    ok_record_with_metrics,
)


SHORTEST_PATH_GRADIENT_NOTE = (
    "Gradient of the summed single-origin least-cost objective on the shared "
    "cost surface."
)
RESISTANCE_GRADIENT_NOTE = (
    "Gradient of graph-volume-scaled effective resistance, matching "
    "commute-time passage counts on the shared cost surface."
)


@eqx.filter_jit
@eqx.filter_grad
def run_shortest_path_centrality_problem(
    cost_raster: jax.Array, sample_nodes: jax.Array
) -> jax.Array:
    grid = GridGraph(cost_raster, fun=cost_conductance)
    pairwise_distances = LCPDistance()(grid, nodes=sample_nodes)
    return jnp.sum(pairwise_distances[0, 1:])


@eqx.filter_jit
@eqx.filter_grad
def run_resistance_centrality_problem(
    cost_raster: jax.Array, sample_nodes: jax.Array
) -> jax.Array:
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
    timings, gradient = measure_runtime(
        centrality_fun, cost_surface, nodes, repeats=config.repeats
    )
    gradient_raster = np.asarray(jax.device_get(jnp.abs(gradient)))
    metrics: dict[str, object] = {
        "centrality_mass": float(np.sum(gradient_raster)),
    }
    if reference_raster is not None:
        metrics.update(centrality_alignment_metrics(gradient_raster, reference_raster))
    return (
        ok_record_with_metrics(
            case.task,
            tool_label,
            "JAXScape",
            case.name,
            timings,
            metrics=metrics,
            note=note,
        ),
        gradient_raster,
    )


def collect_jaxscape_sensitivity_results(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    shortest_path_reference: np.ndarray | None = None,
    resistance_reference: np.ndarray | None = None,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    jax_shortest_cpu_record, _ = run_jaxscape_centrality(
        case,
        config,
        device=config.cpu_device,
        tool_label=backend_tool_label("JAXScape / shortest-path gradient", "cpu"),
        centrality_fun=run_shortest_path_centrality_problem,
        note=SHORTEST_PATH_GRADIENT_NOTE,
        reference_raster=shortest_path_reference,
    )
    records.append(jax_shortest_cpu_record)
    if config.gpu_device is None:
        records.append(
            gpu_placeholder_record(
                case, "JAXScape / shortest-path gradient", "JAXScape"
            )
        )
    else:
        jax_shortest_gpu_record, _ = run_jaxscape_centrality(
            case,
            config,
            device=config.gpu_device,
            tool_label=backend_tool_label("JAXScape / shortest-path gradient", "gpu"),
            centrality_fun=run_shortest_path_centrality_problem,
            note=SHORTEST_PATH_GRADIENT_NOTE,
            reference_raster=shortest_path_reference,
        )
        records.append(jax_shortest_gpu_record)

    jax_resistance_cpu_record, _ = run_jaxscape_centrality(
        case,
        config,
        device=config.cpu_device,
        tool_label=backend_tool_label("JAXScape / resistance gradient", "cpu"),
        centrality_fun=run_resistance_centrality_problem,
        note=RESISTANCE_GRADIENT_NOTE,
        reference_raster=resistance_reference,
    )
    records.append(jax_resistance_cpu_record)
    if config.gpu_device is None:
        records.append(
            gpu_placeholder_record(case, "JAXScape / resistance gradient", "JAXScape")
        )
    else:
        jax_resistance_gpu_record, _ = run_jaxscape_centrality(
            case,
            config,
            device=config.gpu_device,
            tool_label=backend_tool_label("JAXScape / resistance gradient", "gpu"),
            centrality_fun=run_resistance_centrality_problem,
            note=RESISTANCE_GRADIENT_NOTE,
            reference_raster=resistance_reference,
        )
        records.append(jax_resistance_gpu_record)
    return records


def collect_task_results(config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["sensitivity"]:
        records.extend(collect_jaxscape_sensitivity_results(case, config))
    return records


def main() -> None:
    run_standalone_task(
        collect_task_results=collect_task_results,
        config_factory=default_jaxscape_config,
        cases=CASES["sensitivity"],
    )


if __name__ == "__main__":
    main()
