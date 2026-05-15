"""JAXScape inverse landscape genetics benchmark task runner."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
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
from benchmark.inverse_benchmark_settings import load_inverse_benchmark_settings
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


INVERSE_TARGET_DTYPE = jnp.dtype(jnp.float64)
INVERSE_TARGET_TOOL = "JAXScape / CholmodSolver"
LOGGER = logging.getLogger(__name__)
MONOMOLECULAR_SCALE_MIN = 0.0
MONOMOLECULAR_SCALE_MAX = 10.0
MONOMOLECULAR_INIT_SHAPE = 5.0
MONOMOLECULAR_SHAPE_MIN = 1e-3
MONOMOLECULAR_MAX_MIN = 1e-3
MONOMOLECULAR_RESISTANCE_CAP = 1e6
INVERSE_AMJAX_CPU_WALLTIME_SECONDS = 180.0

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


def _base_resistance(permeability: jax.Array) -> jax.Array:
    minimum = jnp.array(MIN_PERMEABILITY, dtype=permeability.dtype)
    return 1.0 / jnp.maximum(permeability, minimum)


def _resistancega_scale(resistance: jax.Array) -> jax.Array:
    minimum = jnp.min(resistance)
    maximum = jnp.max(resistance)
    span = maximum - minimum
    midpoint = jnp.array(
        (MONOMOLECULAR_SCALE_MIN + MONOMOLECULAR_SCALE_MAX) / 2,
        dtype=resistance.dtype,
    )
    return jnp.where(
        span > 0,
        (MONOMOLECULAR_SCALE_MAX - MONOMOLECULAR_SCALE_MIN)
        / span
        * (resistance - maximum)
        + MONOMOLECULAR_SCALE_MAX,
        jnp.full_like(resistance, midpoint),
    )


def _positive_parameter(raw: jax.Array, minimum: float) -> jax.Array:
    return jnn.softplus(raw) + jnp.array(minimum, dtype=raw.dtype)


def _positive_parameter_inverse(value: jax.Array, minimum: float) -> jax.Array:
    shifted = jnp.maximum(
        value - jnp.array(minimum, dtype=value.dtype),
        jnp.array(1e-6, dtype=value.dtype),
    )
    return jnp.log(jnp.expm1(shifted))


def _permeability_from_inverse_params(
    params: jax.Array, base_resistance: jax.Array
) -> jax.Array:
    scaled_resistance = _resistancega_scale(base_resistance)
    shape = _positive_parameter(params[0], MONOMOLECULAR_SHAPE_MIN)
    max_resistance = _positive_parameter(params[1], MONOMOLECULAR_MAX_MIN)
    transformed_resistance = max_resistance * (
        1.0 - jnp.exp(-scaled_resistance / shape)
    ) + 1.0
    transformed_resistance = jnp.clip(
        transformed_resistance,
        1.0,
        jnp.array(MONOMOLECULAR_RESISTANCE_CAP, dtype=base_resistance.dtype),
    )
    return jnp.maximum(
        1.0 / transformed_resistance,
        jnp.array(MIN_PERMEABILITY, dtype=base_resistance.dtype),
    )


def _initial_inverse_params(base_resistance: jax.Array) -> jax.Array:
    dtype = base_resistance.dtype
    init_shape = jnp.array(MONOMOLECULAR_INIT_SHAPE, dtype=dtype)
    init_max = (
        jnp.maximum(jnp.max(base_resistance) - 1.0, jnp.array(1e-3, dtype=dtype))
        / (1.0 - jnp.exp(-MONOMOLECULAR_SCALE_MAX / init_shape))
    )
    return jnp.stack(
        [
            _positive_parameter_inverse(init_shape, MONOMOLECULAR_SHAPE_MIN),
            _positive_parameter_inverse(init_max, MONOMOLECULAR_MAX_MIN),
        ]
    )


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
    params: jax.Array,
    base_resistance: jax.Array,
    sample_coords: jax.Array,
    target_distances: jax.Array,
    *,
    distance: ResistanceDistance,
    state: Any = None,
) -> jax.Array:
    permeability = _permeability_from_inverse_params(params, base_resistance)
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
    inverse_settings = load_inverse_benchmark_settings().jaxscape

    try:
        distance = _distance_for_profile(
            solver_factory=profile.solver_factory,
            method_factory=profile.method_factory,
        )
        dtype_context = jax.enable_x64(jnp.dtype(profile.dtype) == INVERSE_TARGET_DTYPE)
        with dtype_context:
            landscape = jax.device_put(
                jnp.asarray(case.raster, dtype=profile.dtype), device
            )
            base_resistance = _base_resistance(landscape)
            sample_coords = jax.device_put(as_point_array(case.points), device)
            target_distances = jax.device_put(
                _target_distances(case, profile.dtype), device
            )
            target_values = lower_triangle_values(target_distances)
            if inverse_settings.optimizer.lower() != "lbfgs":
                raise ValueError(
                    "Only the LBFGS inverse optimizer is currently supported, "
                    f"received {inverse_settings.optimizer!r}."
                )
            solver = optimistix.LBFGS(
                rtol=inverse_settings.rtol,
                atol=inverse_settings.atol,
            )
            init_params = _initial_inverse_params(base_resistance)
            fixed_state = None
            if profile.requires_preparation:
                # Match the ResistanceGA benchmark setup: both tools optimise a
                # transformed version of the same base surface, so AMJax can
                # safely reuse a hierarchy built from the initial transform.
                fixed_state = distance.init_preconditioner(
                    _inverse_grid(
                        _permeability_from_inverse_params(init_params, base_resistance)
                    )
                )

            def objective(
                params: jax.Array,
                args: tuple[jax.Array, jax.Array, jax.Array],
            ) -> jax.Array:
                reference_resistance, coords, targets = args
                return _inverse_loss(
                    params,
                    reference_resistance,
                    coords,
                    targets,
                    distance=distance,
                    state=fixed_state,
                )

            def solve(start_params: jax.Array):
                return optimistix.minimise(
                    objective,
                    solver,
                    start_params,
                    args=(base_resistance, sample_coords, target_distances),
                    max_steps=inverse_settings.max_steps,
                    throw=False,
                )

            timings, solution = measure_runtime(solve, init_params, repeats=repeats)

            fitted_permeability = _permeability_from_inverse_params(
                solution.value,
                base_resistance,
            )
            final_grid = _inverse_grid(fitted_permeability)
            final_state = None
            if profile.requires_preparation:
                final_state = distance.init(final_grid)
            predicted_distances = distance(
                final_grid,
                nodes=sample_coords,
                state=_distance_state(distance, final_grid, state=final_state),
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
                    "iteration_limit": inverse_settings.max_steps,
                    "solver_preconditioner_reused": bool(
                        profile.requires_preparation
                    ),
                    "solver_cg_state_refreshed": bool(profile.requires_preparation),
                    "solver_preconditioner_refresh_count": 1
                    if profile.requires_preparation
                    else 0,
                    "parameterization": "monomolecular_base_resistance",
                    "parameter_count": 2,
                    "final_shape_parameter": float(
                        _positive_parameter(
                            solution.value[0], MONOMOLECULAR_SHAPE_MIN
                        )
                    ),
                    "final_max_parameter": float(
                        _positive_parameter(
                            solution.value[1], MONOMOLECULAR_MAX_MIN
                        )
                    ),
                    "dtype": str(jnp.dtype(profile.dtype)),
                    "backend": device.platform,
                    "target_distance_tool": INVERSE_TARGET_TOOL,
                    "optimizer": inverse_settings.optimizer,
                    "optimizer_rtol": inverse_settings.rtol,
                    "optimizer_atol": inverse_settings.atol,
                    "budget_policy": "fixed_max_steps",
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


def inverse_worker_walltime_seconds(
    profile: ResistanceProfile, backend: str
) -> float:
    walltime_seconds = benchmark_walltime_seconds()
    if profile.requires_preparation and backend == "cpu":
        return max(walltime_seconds, INVERSE_AMJAX_CPU_WALLTIME_SECONDS)
    return walltime_seconds


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
        walltime_seconds=inverse_worker_walltime_seconds(profile, backend),
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
