"""JAXScape sensitivity-analysis benchmark task runner."""

from __future__ import annotations

import logging
import os
import sys
import tempfile
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
import numpy as np
from jaxscape import GridGraph, LCPDistance, ResistanceDistance
from jaxscape.solvers import batched_linear_solve
from jaxscape.utils import graph_laplacian

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
    failed_record,
    gpu_placeholder_record,
    measure_runtime,
    ok_record_with_metrics,
)
from benchmark.jaxscape.resistance_profile_support import (
    make_amjaxcg_solver,
    make_pyamg_solver,
)


SHORTEST_PATH_GRADIENT_NOTE = (
    "Gradient of the same single-origin summed least-cost objective used by "
    "the `gdistance::shortestPath` reference on the shared cost surface and "
    "destination set."
)
RESISTANCE_GRADIENT_NOTE = (
    "Target-wise checkpointed closed-form gradient of graph-volume-scaled "
    "effective resistance with solver-backed grounded Laplacian solves, "
    "compared against `gdistance::passage(..., totalNet = \"total\")` on "
    "the same single-origin, shared-surface, shared-destination task."
)
LOGGER = logging.getLogger(__name__)
DEFAULT_SENSITIVITY_WALLTIME_SECONDS = 180.0
SENSITIVITY_WALLTIME_ENV = "JAXSCAPE_BENCHMARK_SENSITIVITY_WALLTIME_SECONDS"
SENSITIVITY_LARGE_REPEATS = 1
AMJAX_RESISTANCE_MAX_GRID_SIZE = 100


def sensitivity_walltime_seconds() -> float:
    raw_value = os.environ.get(SENSITIVITY_WALLTIME_ENV)
    if raw_value is None:
        return max(benchmark_walltime_seconds(), DEFAULT_SENSITIVITY_WALLTIME_SECONDS)
    try:
        walltime = float(raw_value)
    except ValueError as error:
        raise RuntimeError(
            f"{SENSITIVITY_WALLTIME_ENV} must be numeric, received {raw_value!r}."
        ) from error
    if walltime <= 0:
        raise RuntimeError(
            f"{SENSITIVITY_WALLTIME_ENV} must be positive, received {raw_value!r}."
        )
    return walltime


def sensitivity_repeats(case: BenchmarkCase, config: BenchmarkConfig) -> int:
    if case.size_label == "large":
        return min(config.repeats, SENSITIVITY_LARGE_REPEATS)
    return config.repeats


def _resistance_solver_candidates(
    case: BenchmarkCase,
) -> tuple[tuple[str, Any, bool, str], ...]:
    if case.grid_size > AMJAX_RESISTANCE_MAX_GRID_SIZE:
        return (("PyAMGSolver", make_pyamg_solver, False, "host_callback"),)
    return (
        ("AMJaxCGSolver", make_amjaxcg_solver, True, "jax_preconditioned_cg"),
        ("PyAMGSolver", make_pyamg_solver, False, "host_callback"),
    )


def _short_error_summary(error: Exception) -> str:
    return str(error).splitlines()[0][:180]


@eqx.filter_jit
@eqx.filter_grad
def run_shortest_path_centrality_problem(
    cost_raster: jax.Array, sample_nodes: jax.Array
) -> jax.Array:
    grid = GridGraph(cost_raster, fun=cost_conductance)
    source_distances = LCPDistance()(
        grid,
        sources=sample_nodes[:1],
        targets=sample_nodes[1:],
    )
    return jnp.sum(source_distances)


@eqx.filter_jit
def run_resistance_centrality_problem(
    cost_raster: jax.Array,
    sample_nodes: jax.Array,
    solver: Any,
    state: Any,
) -> jax.Array:
    grid = GridGraph(cost_raster, fun=cost_conductance)
    nodes = grid.coord_to_index(sample_nodes[:, 0], sample_nodes[:, 1])
    return _single_source_resistance_gradient(
        cost_raster,
        grid.get_adjacency_matrix(),
        nodes[0],
        nodes[1:],
        solver,
        state,
    )


def _grounded_basis(node: jax.Array, n_reduced: int, dtype: jnp.dtype) -> jax.Array:
    node = node.astype(jnp.int32)
    return jnp.where(
        node < n_reduced,
        jax.nn.one_hot(node, n_reduced, dtype=dtype),
        jnp.zeros((n_reduced,), dtype=dtype),
    )


def _grounded_values(
    values: jax.Array,
    nodes: jax.Array,
    n_reduced: int,
) -> jax.Array:
    clamped_nodes = jnp.minimum(nodes, n_reduced - 1)
    return jnp.where(nodes < n_reduced, values[clamped_nodes], 0)


def _single_source_resistance_gradient(
    cost_raster: jax.Array,
    adjacency_matrix,
    source: jax.Array,
    targets: jax.Array,
    solver: Any,
    state: Any,
) -> jax.Array:
    laplacian_reduced = graph_laplacian(adjacency_matrix)[:-1, :-1]
    n_reduced = laplacian_reduced.shape[0]
    source_basis = _grounded_basis(source, n_reduced, laplacian_reduced.dtype)
    rows = adjacency_matrix.indices[:, 0]
    cols = adjacency_matrix.indices[:, 1]
    undirected_edges = (rows < cols) & (adjacency_matrix.data > 0)

    @jax.checkpoint
    def resistance_to_target(target: jax.Array) -> tuple[jax.Array, jax.Array]:
        rhs = source_basis - _grounded_basis(
            target,
            n_reduced,
            laplacian_reduced.dtype,
        )
        potential = batched_linear_solve(
            laplacian_reduced,
            rhs[:, None],
            solver,
            state=state,
        )[:, 0]
        row_potentials = _grounded_values(potential, rows, n_reduced)
        col_potentials = _grounded_values(potential, cols, n_reduced)
        edge_delta_squares = jnp.where(
            undirected_edges,
            jnp.square(row_potentials - col_potentials),
            0,
        )
        return jnp.dot(rhs, potential), edge_delta_squares

    def accumulate(
        carry: tuple[jax.Array, jax.Array], target: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], None]:
        resistance_sum, edge_delta_squares_sum = carry
        resistance, edge_delta_squares = resistance_to_target(target)
        return (
            resistance_sum + resistance,
            edge_delta_squares_sum + edge_delta_squares,
        ), None

    initial = (
        jnp.array(0.0, dtype=laplacian_reduced.dtype),
        jnp.zeros_like(adjacency_matrix.data),
    )
    (resistance_sum, edge_delta_squares_sum), _ = jax.lax.scan(
        accumulate,
        initial,
        targets,
    )
    graph_volume = benchmark_graph_volume(cost_raster)
    edge_weight_sensitivity = jnp.where(
        undirected_edges,
        2.0 * resistance_sum - graph_volume * edge_delta_squares_sum,
        0,
    )

    flat_cost = cost_raster.ravel()
    adjacent_cost_sum = flat_cost[rows] + flat_cost[cols]
    weight_to_cost_derivative = -2.0 / jnp.square(
        jnp.maximum(adjacent_cost_sum, jnp.array(1e-6, dtype=cost_raster.dtype))
    )
    edge_cost_sensitivity = edge_weight_sensitivity * weight_to_cost_derivative
    flat_gradient = jnp.zeros_like(flat_cost)
    flat_gradient = flat_gradient.at[rows].add(edge_cost_sensitivity)
    flat_gradient = flat_gradient.at[cols].add(edge_cost_sensitivity)
    return flat_gradient.reshape(cost_raster.shape)


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


def _reference_raster_from_path(reference_raster_path: str | None) -> np.ndarray | None:
    if reference_raster_path is None:
        return None
    return np.load(reference_raster_path)


def _sensitivity_components(
    centrality_kind: str,
) -> tuple[Any, str, str]:
    if centrality_kind == "shortest_path":
        return (
            run_shortest_path_centrality_problem,
            SHORTEST_PATH_GRADIENT_NOTE,
            "JAXScape / shortest-path gradient",
        )
    if centrality_kind == "resistance":
        return (
            run_resistance_centrality_problem,
            RESISTANCE_GRADIENT_NOTE,
            "JAXScape / resistance gradient",
        )
    raise KeyError(f"Unknown sensitivity benchmark kind {centrality_kind!r}.")


def _run_jaxscape_centrality_direct(
    case: BenchmarkCase,
    *,
    backend: str,
    repeats: int,
    centrality_kind: str,
    reference_raster_path: str | None,
) -> BenchmarkRecord:
    device = device_for_backend(backend)
    centrality_fun, note, tool_base = _sensitivity_components(centrality_kind)
    tool_label = backend_tool_label(tool_base, backend)
    LOGGER.info(
        "Starting sensitivity benchmark for case=%s tool=%s backend=%s repeats=%s",
        case.name,
        tool_label,
        device.platform,
        repeats,
    )
    try:
        cost_surface = jax.device_put(as_array(case.raster), device)
        nodes = jax.device_put(as_point_array(case.points), device)
        if centrality_kind == "resistance":
            solver_name = ""
            solver_preconditioner_reused = False
            solver_backend = ""
            attempt_notes: list[str] = []
            last_error: Exception | None = None
            for (
                candidate_name,
                candidate_factory,
                candidate_preconditioner_reused,
                candidate_backend,
            ) in _resistance_solver_candidates(case):
                try:
                    solver = candidate_factory()
                    state = ResistanceDistance(solver=solver).init_preconditioner(
                        GridGraph(cost_surface, fun=cost_conductance)
                    )
                    timings, gradient = measure_runtime(
                        centrality_fun,
                        cost_surface,
                        nodes,
                        solver,
                        state,
                        repeats=repeats,
                    )
                    solver_name = candidate_name
                    solver_preconditioner_reused = candidate_preconditioner_reused
                    solver_backend = candidate_backend
                    break
                except Exception as error:
                    last_error = error
                    attempt_notes.append(
                        f"{candidate_name}: {_short_error_summary(error)}"
                    )
                    LOGGER.warning(
                        "Sensitivity solver candidate failed for case=%s tool=%s "
                        "solver=%s: %s",
                        case.name,
                        tool_label,
                        candidate_name,
                        error,
                    )
            else:
                assert last_error is not None
                raise last_error
        else:
            attempt_notes = []
            solver_name = ""
            solver_preconditioner_reused = False
            solver_backend = ""
            timings, gradient = measure_runtime(
                centrality_fun, cost_surface, nodes, repeats=repeats
            )
        gradient_raster = np.asarray(jax.device_get(jnp.abs(gradient)))
        metrics: dict[str, object] = {
            "centrality_mass": float(np.sum(gradient_raster)),
            "timed_repeats": len(timings),
        }
        if centrality_kind == "resistance":
            metrics.update(
                {
                    "solver": solver_name,
                    "solver_backend": solver_backend,
                    "solver_preconditioner_reused": solver_preconditioner_reused,
                    "checkpointing": "target_wise_single_rhs",
                    "gradient_rule": "closed_form_edge_sensitivity",
                }
            )
        if centrality_kind == "shortest_path":
            metrics.update({"checkpointing": "single_source_bellman_ford"})
        reference_raster = _reference_raster_from_path(reference_raster_path)
        if reference_raster is not None:
            metrics.update(
                centrality_alignment_metrics(gradient_raster, reference_raster)
            )
        record = ok_record_with_metrics(
            case.task,
            tool_label,
            "JAXScape",
            case.name,
            timings,
            metrics=metrics,
            note=(
                note
                if not attempt_notes
                else f"{note} Solver fallback notes: {'; '.join(attempt_notes)}"
            ),
        )
    except MemoryError as error:
        note = f"Out-of-memory while running sensitivity benchmark: {error}"
        LOGGER.exception(
            "OOM in sensitivity benchmark for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)
    except Exception as error:
        note = str(error)
        if looks_like_oom(note):
            note = f"Out-of-memory while running sensitivity benchmark: {note}"
        LOGGER.exception(
            "Sensitivity benchmark failed for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(case.task, tool_label, "JAXScape", case.name, note)

    LOGGER.info(
        "Completed sensitivity benchmark for case=%s tool=%s median=%.6fs",
        case.name,
        tool_label,
        record.median_seconds,
    )
    return record


def sensitivity_worker_payload(
    case: BenchmarkCase,
    *,
    backend: str,
    repeats: int,
    centrality_kind: str,
    reference_raster_path: str | None,
) -> dict[str, Any]:
    return {
        "case_name": case.name,
        "backend": backend,
        "repeats": repeats,
        "centrality_kind": centrality_kind,
        "reference_raster_path": reference_raster_path,
    }


def run_worker(payload: dict[str, Any]) -> BenchmarkRecord:
    case = case_by_name(CASES["sensitivity"], payload["case_name"])
    return _run_jaxscape_centrality_direct(
        case,
        backend=payload["backend"],
        repeats=int(payload["repeats"]),
        centrality_kind=payload["centrality_kind"],
        reference_raster_path=payload.get("reference_raster_path"),
    )


def run_jaxscape_centrality_profile(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    tool_base: str,
    backend: str,
    centrality_kind: str,
    reference_raster_path: str | None,
) -> BenchmarkRecord:
    return run_worker_subprocess(
        script_path=Path(__file__).resolve(),
        payload=sensitivity_worker_payload(
            case,
            backend=backend,
            repeats=sensitivity_repeats(case, config),
            centrality_kind=centrality_kind,
            reference_raster_path=reference_raster_path,
        ),
        case_task=case.task,
        case_name=case.name,
        tool_label=backend_tool_label(tool_base, backend),
        software="JAXScape",
        walltime_seconds=sensitivity_walltime_seconds(),
        logger=LOGGER,
    )


def _write_reference_raster(
    temp_dir: str, filename: str, reference_raster: np.ndarray | None
) -> str | None:
    if reference_raster is None:
        return None
    path = Path(temp_dir) / filename
    np.save(path, reference_raster)
    return str(path)


def collect_jaxscape_sensitivity_results(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    shortest_path_reference: np.ndarray | None = None,
    resistance_reference: np.ndarray | None = None,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    with tempfile.TemporaryDirectory(prefix="jaxscape-sensitivity-") as temp_dir:
        shortest_path_reference_path = _write_reference_raster(
            temp_dir,
            "shortest_path_reference.npy",
            shortest_path_reference,
        )
        resistance_reference_path = _write_reference_raster(
            temp_dir,
            "resistance_reference.npy",
            resistance_reference,
        )

        records.append(
            run_jaxscape_centrality_profile(
                case,
                config,
                tool_base="JAXScape / shortest-path gradient",
                backend="cpu",
                centrality_kind="shortest_path",
                reference_raster_path=shortest_path_reference_path,
            )
        )

        if config.gpu_device is None:
            records.append(
                gpu_placeholder_record(
                    case, "JAXScape / shortest-path gradient", "JAXScape"
                )
            )
        else:
            records.append(
                run_jaxscape_centrality_profile(
                    case,
                    config,
                    tool_base="JAXScape / shortest-path gradient",
                    backend="gpu",
                    centrality_kind="shortest_path",
                    reference_raster_path=shortest_path_reference_path,
                )
            )

        records.append(
            run_jaxscape_centrality_profile(
                case,
                config,
                tool_base="JAXScape / resistance gradient",
                backend="cpu",
                centrality_kind="resistance",
                reference_raster_path=resistance_reference_path,
            )
        )
        if config.gpu_device is None:
            records.append(
                gpu_placeholder_record(
                    case, "JAXScape / resistance gradient", "JAXScape"
                )
            )
        else:
            records.append(
                run_jaxscape_centrality_profile(
                    case,
                    config,
                    tool_base="JAXScape / resistance gradient",
                    backend="gpu",
                    centrality_kind="resistance",
                    reference_raster_path=resistance_reference_path,
                )
            )
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
        worker_handler=run_worker,
    )


if __name__ == "__main__":
    main()
