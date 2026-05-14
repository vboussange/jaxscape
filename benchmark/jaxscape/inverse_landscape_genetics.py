"""JAXScape inverse landscape genetics benchmark task runner."""

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

import jax
import jax.nn as jnn
import jax.numpy as jnp
from jaxscape import GridGraph, ResistanceDistance

from benchmark.benchmark_distances import (
    as_point_array,
    backend_tool_label,
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    CASES,
    default_jaxscape_config,
    edge_weight,
    failed_record,
    fit_error_metrics,
    gpu_placeholder_record,
    lower_triangle_values,
    measure_runtime,
    MIN_PERMEABILITY,
    ok_record_with_metrics,
    skip_record,
)
from benchmark.jaxscape.resistance_profile_support import (
    make_amjaxcg_solver,
    make_cholmod_solver,
    make_spielman_method,
    ResistanceProfile,
)


try:
    import optimistix as optx
except ImportError:
    optx = None
    _optimistix_import_error = "optimistix is not installed"
else:
    _optimistix_import_error = None


INVERSE_OPTIMISTIX_MAX_STEPS = 50
INVERSE_TARGET_DTYPE = jnp.dtype(jnp.float64)
INVERSE_TARGET_TOOL = "JAXScape / CholmodSolver"
LOGGER = logging.getLogger(__name__)

JAXSCAPE_INVERSE_PROFILES = (
    ResistanceProfile(
        "cholmod_f32",
        "JAXScape / CholmodSolver / f32",
        solver_factory=make_cholmod_solver,
        dtype=jnp.float32,
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
)
PROFILE_BY_KEY = {profile.key: profile for profile in JAXSCAPE_INVERSE_PROFILES}


def _permeability_from_logits(logits: jax.Array) -> jax.Array:
    return jnn.sigmoid(logits) + jnp.array(MIN_PERMEABILITY, dtype=logits.dtype)


def _inverse_grid(permeability: jax.Array) -> GridGraph:
    return GridGraph(grid=permeability, fun=edge_weight)


def _distance_for_profile(
    *,
    solver_factory: Callable[[], Any] | None = None,
    method_factory: Callable[[], Any] | None = None,
) -> ResistanceDistance:
    solver = None if solver_factory is None else solver_factory()
    method = None if method_factory is None else method_factory()
    if method is None:
        return ResistanceDistance(solver=solver)
    return ResistanceDistance(solver=solver, method=method)


def _target_distances(case: BenchmarkCase, dtype: Any) -> jax.Array:
    target_distance = _distance_for_profile(solver_factory=make_cholmod_solver)
    with jax.enable_x64():
        target_landscape = jnp.asarray(case.raster, dtype=INVERSE_TARGET_DTYPE)
        target_nodes = as_point_array(case.points)
        target_matrix = target_distance(
            _inverse_grid(target_landscape), nodes=target_nodes
        )
    return jnp.asarray(target_matrix, dtype=dtype)


def _distance_state(
    distance: ResistanceDistance,
    grid: GridGraph,
    *,
    state: Any = None,
) -> Any:
    if state is not None or distance.solver is None:
        return state
    return jax.tree.map(jax.lax.stop_gradient, distance.init(grid))


def _inverse_loss(
    logits: jax.Array,
    sample_coords: jax.Array,
    target_distances: jax.Array,
    *,
    distance: ResistanceDistance,
    state: Any = None,
) -> jax.Array:
    permeability = _permeability_from_logits(logits)
    grid = _inverse_grid(permeability)
    predicted = distance(
        grid,
        nodes=sample_coords,
        state=_distance_state(distance, grid, state=state),
    )
    return jnp.mean((predicted - target_distances) ** 2)


def _run_inverse_profile_direct(
    case: BenchmarkCase,
    profile: ResistanceProfile,
    *,
    device: Any,
    repeats: int,
    tool_label: str,
) -> BenchmarkRecord:
    LOGGER.info(
        "Starting inverse benchmark for case=%s tool=%s backend=%s repeats=%s",
        case.name,
        tool_label,
        device.platform,
        repeats,
    )
    if optx is None:
        return skip_record(
            case.task,
            tool_label,
            "JAXScape + Optimistix",
            case.name,
            _optimistix_import_error or "optimistix missing",
        )
    assert optx is not None
    optimistix = optx

    try:
        distance = _distance_for_profile(
            solver_factory=profile.solver_factory,
            method_factory=profile.method_factory,
        )
        dtype_context = (
            jax.enable_x64()
            if jnp.dtype(profile.dtype) == INVERSE_TARGET_DTYPE
            else nullcontext()
        )
        with dtype_context:
            landscape = jax.device_put(
                jnp.asarray(case.raster, dtype=profile.dtype), device
            )
            sample_coords = jax.device_put(as_point_array(case.points), device)
            target_distances = jax.device_put(
                _target_distances(case, profile.dtype), device
            )
            target_values = lower_triangle_values(target_distances)
            solver = optimistix.LBFGS(rtol=1e-5, atol=1e-5)
            init_logits = jnp.zeros_like(landscape)
            fixed_state = None
            if profile.requires_preparation:
                # Benchmark design choice: reuse the initial solver state across
                # optimisation steps to measure the impact of preparation reuse.
                # This is not the recommended pattern for real calibration runs.
                fixed_state = distance.init(
                    _inverse_grid(_permeability_from_logits(init_logits))
                )

            def objective(
                logits: jax.Array, args: tuple[jax.Array, jax.Array]
            ) -> jax.Array:
                coords, targets = args
                return _inverse_loss(
                    logits,
                    coords,
                    targets,
                    distance=distance,
                    state=fixed_state,
                )

            def solve(start_logits: jax.Array):
                return optimistix.minimise(
                    objective,
                    solver,
                    start_logits,
                    args=(sample_coords, target_distances),
                    max_steps=INVERSE_OPTIMISTIX_MAX_STEPS,
                    throw=False,
                )

            timings, solution = measure_runtime(solve, init_logits, repeats=repeats)
            fitted_permeability = _permeability_from_logits(solution.value)
            predicted_distances = distance(
                _inverse_grid(fitted_permeability),
                nodes=sample_coords,
                state=_distance_state(
                    distance,
                    _inverse_grid(fitted_permeability),
                    state=fixed_state,
                ),
            )
            final_mse = jnp.mean((predicted_distances - target_distances) ** 2)
            metrics: dict[str, Any] = fit_error_metrics(
                lower_triangle_values(predicted_distances), target_values
            )
            metrics.update(
                {
                    "final_mse": float(final_mse),
                    "converged": bool(
                        solution.result == optimistix.RESULTS.successful
                    ),
                    "iteration_count": int(solution.stats["num_steps"]),
                    "iteration_limit": INVERSE_OPTIMISTIX_MAX_STEPS,
                    "solver_state_reused": bool(profile.requires_preparation),
                    "dtype": str(jnp.dtype(profile.dtype)),
                    "backend": device.platform,
                    "target_distance_tool": INVERSE_TARGET_TOOL,
                }
            )
    except ImportError as error:
        LOGGER.warning(
            "Skipping inverse benchmark for case=%s tool=%s: %s",
            case.name,
            tool_label,
            error,
        )
        return skip_record(
            case.task, tool_label, "JAXScape + Optimistix", case.name, str(error)
        )
    except MemoryError as error:
        note = f"Out-of-memory while running inverse benchmark: {error}"
        LOGGER.exception(
            "OOM in inverse benchmark for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(
            case.task, tool_label, "JAXScape + Optimistix", case.name, note
        )
    except Exception as error:
        note = str(error)
        if looks_like_oom(note):
            note = f"Out-of-memory while running inverse benchmark: {note}"
        LOGGER.exception(
            "Inverse benchmark failed for case=%s tool=%s",
            case.name,
            tool_label,
        )
        return failed_record(
            case.task, tool_label, "JAXScape + Optimistix", case.name, note
        )

    record = ok_record_with_metrics(
        case.task,
        tool_label,
        "JAXScape + Optimistix",
        case.name,
        timings,
        metrics=metrics,
    )
    LOGGER.info(
        "Completed inverse benchmark for case=%s tool=%s median=%.6fs final_mse=%.6g",
        case.name,
        tool_label,
        record.median_seconds,
        metrics["final_mse"],
    )
    return record


def profile_for_tool_label(tool_label: str) -> ResistanceProfile:
    for profile in JAXSCAPE_INVERSE_PROFILES:
        candidate_labels = {
            profile.tool,
            backend_tool_label(profile.tool, "cpu"),
            backend_tool_label(profile.tool, "gpu"),
        }
        if tool_label in candidate_labels:
            return profile
    raise KeyError(f"Unknown inverse benchmark tool label {tool_label!r}.")


def inverse_worker_payload(
    case: BenchmarkCase, profile: ResistanceProfile, backend: str, repeats: int
) -> dict[str, Any]:
    return {
        "case_name": case.name,
        "profile_key": profile.key,
        "backend": backend,
        "repeats": repeats,
    }


def run_worker(payload: dict[str, Any]) -> BenchmarkRecord:
    case = case_by_name(CASES["inverse"], payload["case_name"])
    profile = PROFILE_BY_KEY[payload["profile_key"]]
    backend = payload["backend"]
    repeats = int(payload["repeats"])
    tool_label = (
        profile.tool
        if backend == "cpu" and not profile.gpu_capable
        else backend_tool_label(profile.tool, backend)
    )
    return _run_inverse_profile_direct(
        case,
        profile,
        device=device_for_backend(backend),
        repeats=repeats,
        tool_label=tool_label,
    )


def run_jaxscape_inverse_profile(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    tool_label: str,
    *,
    device: Any,
) -> BenchmarkRecord:
    del device
    profile = profile_for_tool_label(tool_label)
    backend = "gpu" if "(GPU)" in tool_label else "cpu"
    return run_worker_subprocess(
        script_path=Path(__file__).resolve(),
        payload=inverse_worker_payload(case, profile, backend, config.repeats),
        case_task=case.task,
        case_name=case.name,
        tool_label=tool_label,
        software="JAXScape + Optimistix",
        walltime_seconds=benchmark_walltime_seconds(),
        logger=LOGGER,
    )


def selected_profiles(
    profile_keys: list[str] | None = None,
) -> tuple[ResistanceProfile, ...]:
    if profile_keys is None:
        return JAXSCAPE_INVERSE_PROFILES
    return tuple(PROFILE_BY_KEY[key] for key in profile_keys)


def collect_jaxscape_inverse_results(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    *,
    profile_keys: list[str] | None = None,
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for profile in selected_profiles(profile_keys):
        if profile.gpu_capable:
            records.append(
                run_jaxscape_inverse_profile(
                    case,
                    config,
                    backend_tool_label(profile.tool, "cpu"),
                    device=config.cpu_device,
                )
            )
            if config.gpu_device is None:
                records.append(
                    gpu_placeholder_record(case, profile.tool, "JAXScape + Optimistix")
                )
            else:
                records.append(
                    run_jaxscape_inverse_profile(
                        case,
                        config,
                        backend_tool_label(profile.tool, "gpu"),
                        device=config.gpu_device,
                    )
                )
        else:
            records.append(
                run_jaxscape_inverse_profile(
                    case,
                    config,
                    profile.tool,
                    device=config.cpu_device,
                )
            )
    return records


def collect_task_results(
    config: BenchmarkConfig, *, profile_keys: list[str] | None = None
) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for case in CASES["inverse"]:
        records.extend(
            collect_jaxscape_inverse_results(case, config, profile_keys=profile_keys)
        )
    return records


def main() -> None:
    run_standalone_task(
        collect_task_results=collect_task_results,
        config_factory=default_jaxscape_config,
        cases=CASES["inverse"],
        worker_handler=run_worker,
    )


if __name__ == "__main__":
    main()
