"""JAXScape inverse landscape genetics benchmark task runner."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxscape import GridGraph, ResistanceDistance

from benchmark.benchmark_distances import (
    as_array,
    as_point_array,
    backend_tool_label,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    CASES,
    default_jaxscape_config,
    edge_weight,
    fit_error_metrics,
    gpu_placeholder_record,
    lower_triangle_values,
    measure_runtime,
    MIN_PERMEABILITY,
    ok_record_with_metrics,
    skip_record,
    write_results,
)


try:
    import optimistix as optx
except ImportError:
    optx = None
    _optimistix_import_error = "optimistix is not installed"
else:
    _optimistix_import_error = None


INVERSE_OPTIMISTIX_MAX_STEPS = 8


def _inverse_loss(
    logits: jax.Array, sample_coords: jax.Array, target_distances: jax.Array
) -> jax.Array:
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
        return skip_record(
            case.task,
            tool_label,
            "JAXScape + Optimistix",
            case.name,
            _optimistix_import_error or "optimistix missing",
        )

    landscape = jax.device_put(as_array(case.raster), device)
    sample_coords = jax.device_put(as_point_array(case.points), device)
    target_distances = ResistanceDistance()(
        GridGraph(landscape, fun=edge_weight), nodes=sample_coords
    )
    target_values = lower_triangle_values(target_distances)
    solver = optx.LBFGS(rtol=1e-5, atol=1e-5)

    def objective(logits: jax.Array, args: tuple[jax.Array, jax.Array]) -> jax.Array:
        coords, targets = args
        return _inverse_loss(logits, coords, targets)

    def solve(problem_landscape: jax.Array):
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
    fitted_permeability = jnn.sigmoid(solution.value) + jnp.array(
        MIN_PERMEABILITY, dtype=solution.value.dtype
    )
    predicted_distances = ResistanceDistance()(
        GridGraph(fitted_permeability, fun=edge_weight), nodes=sample_coords
    )
    metrics = fit_error_metrics(
        lower_triangle_values(predicted_distances), target_values
    )
    metrics.update(
        {
            "final_mse": float(
                _inverse_loss(solution.value, sample_coords, target_distances)
            ),
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
        note=(
            "Fixed-budget L-BFGS calibration against the synthetic "
            "resistance-distance target."
        ),
    )


def collect_jaxscape_inverse_results(
    case: BenchmarkCase, config: BenchmarkConfig
) -> list[BenchmarkRecord]:
    records = [
        run_jaxscape_inverse(
            case,
            config,
            device=config.cpu_device,
            tool_label=backend_tool_label("JAXScape + Optimistix", "cpu"),
        )
    ]
    if config.gpu_device is None:
        records.append(
            gpu_placeholder_record(
                case, "JAXScape + Optimistix", "JAXScape + Optimistix"
            )
        )
    else:
        records.append(
            run_jaxscape_inverse(
                case,
                config,
                device=config.gpu_device,
                tool_label=backend_tool_label("JAXScape + Optimistix", "gpu"),
            )
        )
    return records


def collect_task_results(config: BenchmarkConfig) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["inverse"]:
        records.extend(collect_jaxscape_inverse_results(case, config))
    return records


def main() -> None:
    config = default_jaxscape_config()
    records = collect_task_results(config)
    write_results(records, config, cases=CASES["inverse"])
    print(json.dumps({"records": [asdict(record) for record in records]}, indent=2))


if __name__ == "__main__":
    main()
