"""Selected-site FED profiles for the shared distance benchmark."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from benchmark.jaxscape.utils import (
    benchmark_walltime_seconds,
    case_by_name,
    configure_standalone_environment,
    device_for_backend,
    run_standalone_task,
    run_worker_subprocess,
)


configure_standalone_environment()

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
import scipy.sparse as sp
from jaxscape import FreeEnergyDistance, GridGraph
from scipy.sparse.linalg import splu

from benchmark.benchmark_distances import (
    as_point_array,
    backend_tool_label,
    BenchmarkCase,
    CASES,
    cost_conductance,
    default_jaxscape_config,
    dependency_versions,
    failed_record,
    gpu_placeholder_record,
    measure_runtime,
    ok_record_with_metrics,
    write_case_payload,
)


LOGGER = logging.getLogger(__name__)
THETAS = (0.01, 0.1, 1.0)
# Explicit float64 workload profile, not a change to the library default.
# Tiny atol resolves distant entries; the final guard checks residual drift
# when periodic CG residual replacement is disabled.
CG_SETTINGS: dict[str, Any] = dict(
    rtol=1e-10, atol=1e-90, max_steps=10_000, stabilise_every=None
)


def sparse_reference(raster, points, theta):
    """Independent nonsymmetric killed-walk solve, with reciprocal costs."""
    raster = np.asarray(raster, dtype=np.float64)
    vertices = np.arange(raster.size).reshape(raster.shape)
    source = np.concatenate((vertices[:-1].ravel(), vertices[:, :-1].ravel()))
    target = np.concatenate((vertices[1:].ravel(), vertices[:, 1:].ravel()))
    rows, cols = np.concatenate((source, target)), np.concatenate((target, source))
    costs = np.maximum((raster.ravel()[rows] + raster.ravel()[cols]) / 2, 1e-6)
    conductance = 1 / costs
    degree = np.bincount(rows, weights=conductance, minlength=raster.size)
    walk = sp.csc_matrix(
        (conductance / degree[rows] * np.exp(-theta * costs), (rows, cols)),
        shape=(raster.size, raster.size),
    )
    coordinates = np.asarray(points)
    nodes = coordinates[:, 0] * raster.shape[1] + coordinates[:, 1]
    rhs = np.zeros((raster.size, len(nodes)))
    rhs[nodes, np.arange(len(nodes))] = 1
    selected = splu(sp.eye(raster.size, format="csc") - walk).solve(rhs)[nodes]
    directed = (np.log(np.diag(selected))[None, :] - np.log(selected)) / theta
    result = (directed + directed.T) / 2
    np.fill_diagonal(result, 0)
    return result


def derivative_checks(raster, points, theta, coefficients, gradient, theta_gradient):
    """Compare three landscape directions and theta with independent differences."""
    rng = np.random.default_rng(129)
    checks = []
    for index in range(4):
        if index < 3:
            direction = rng.normal(size=raster.shape)
            direction /= np.max(np.abs(direction))
            step = 1e-4
            plus = sparse_reference(raster + step * direction, points, theta)
            minus = sparse_reference(raster - step * direction, points, theta)
            autodiff = float(np.sum(gradient * direction))
        else:
            step = theta * 1e-4
            plus = sparse_reference(raster, points, theta + step)
            minus = sparse_reference(raster, points, theta - step)
            autodiff = float(theta_gradient)
        difference = float(np.sum(coefficients * (plus - minus)) / (2 * step))
        checks.append(
            dict(
                parameter=f"landscape_direction_{index}" if index < 3 else "theta",
                step=step,
                finite_difference=difference,
                autodiff=autodiff,
                scaled_error=abs(difference - autodiff) / max(1, abs(difference)),
            )
        )
    return checks


def run_profile(case, *, theta, device, repeats, solver_settings=None):
    """Time public forward/gradient calls and validate the same realized inputs.

    Custom cases can reuse `build_case` for bounded larger-grid experiments.

    **Arguments:**

    - `case`: `BenchmarkCase` containing the raster and at least two selected
      sites. Reference and production calculations use the same realized input.
    - `theta`: Finite positive scalar inverse temperature.
    - `device`: JAX device on which forward and gradient computations run.
    - `repeats`: Positive number of synchronized warmed samples per operation.
    - `solver_settings`: Optional dictionary of `lineax.CG` keyword arguments,
      including tolerances, for explicit profile or negative-control runs.
      `None` uses `CG_SETTINGS`; production solver defaults remain unchanged.

    **Returns:**

    `BenchmarkRecord` with forward timings, setup/gradient measurements and
    independent accuracy checks. Computation or accuracy-check failures return
    a `failed` record with the error message and metadata collected so far.
    """
    settings = CG_SETTINGS if solver_settings is None else solver_settings
    tool = backend_tool_label(f"JAXScape / CG / theta={theta:g} / f64", device.platform)
    metrics: dict[str, Any] = dict(
        theta=theta,
        dtype="float64",
        backend=device.platform,
        device=str(device),
        device_kind=device.device_kind,
        dependency_versions=dependency_versions(),
        implementation_sha256=hashlib.sha256(
            (ROOT / "src/jaxscape/free_energy_distance.py").read_bytes()
        ).hexdigest(),
        solver_settings=settings,
        samples=repeats,
        point_count=len(case.points),
        seed=case.seed,
        timing_scope="Graph construction, state rebuild, selected solves and guard",
    )
    try:
        with jax.enable_x64(), jax.default_device(device):
            raster = np.asarray(case.raster, dtype=np.float64)
            metrics["input_sha256"] = hashlib.sha256(raster.tobytes()).hexdigest()
            values = jax.device_put(jnp.asarray(raster), device)
            nodes = jax.device_put(as_point_array(case.points), device)
            temperature = jax.device_put(jnp.asarray(theta), device)
            solver = lx.CG(**settings)
            coefficients = (
                np.random.default_rng(128).normal(
                    size=(len(case.points), len(case.points))
                )
                / len(case.points) ** 2
            )

            def forward(values, temperature):
                graph = GridGraph(values, fun=cost_conductance)
                return FreeEnergyDistance(temperature, solver)(graph, nodes=nodes)

            def setup(values, temperature):
                graph = GridGraph(values, fun=cost_conductance)
                return FreeEnergyDistance(temperature, solver).init(graph)

            def loss(values, temperature):
                return jnp.sum(jnp.asarray(coefficients) * forward(values, temperature))

            functions = dict(
                setup=setup,
                forward=forward,
                value_and_grad=jax.value_and_grad(loss, argnums=(0, 1)),
            )
            outputs = {}
            metrics["compile_seconds"] = {}
            for name, function in functions.items():
                start = time.perf_counter()
                compiled = jax.jit(function).lower(values, temperature).compile()
                metrics["compile_seconds"][name] = time.perf_counter() - start
                timings, outputs[name] = measure_runtime(
                    compiled, values, temperature, repeats=repeats
                )
                metrics[f"{name}_seconds"] = timings

            distances = np.asarray(outputs["forward"])
            objective, (gradient, theta_gradient) = outputs["value_and_grad"]
            reference = sparse_reference(raster, case.points, theta)
            off_diagonal = ~np.eye(len(case.points), dtype=bool)
            metrics["off_diagonal_entries"] = int(off_diagonal.sum())
            metrics["max_scaled_error"] = float(
                np.max(
                    np.abs(distances[off_diagonal] - reference[off_diagonal])
                    / np.maximum(1, np.abs(reference[off_diagonal]))
                )
            )
            metrics["all_finite"] = bool(
                np.isfinite(distances).all()
                and np.isfinite(reference).all()
                and np.isfinite(gradient).all()
                and np.isfinite(theta_gradient)
            )
            metrics["objective"] = float(objective)
            metrics["output_devices"] = [str(x) for x in outputs["forward"].devices()]
            metrics["gradient_devices"] = [str(x) for x in gradient.devices()]
            metrics["theta_gradient_devices"] = [
                str(x) for x in theta_gradient.devices()
            ]
            metrics["derivative_checks"] = derivative_checks(
                raster,
                case.points,
                theta,
                coefficients,
                np.asarray(gradient),
                float(theta_gradient),
            )
            # Float64 targets are looser than observed roundoff. Central
            # differences have truncation/cancellation error and only check
            # these four directions, not every gradient component.
            if not (
                metrics["all_finite"]
                and metrics["max_scaled_error"] < 1e-7
                and max(x["scaled_error"] for x in metrics["derivative_checks"]) < 1e-6
            ):
                raise ValueError("Independent value or gradient accuracy check failed.")
    except Exception as error:
        return replace(
            failed_record(case.task, tool, "JAXScape", case.name, str(error)),
            metrics=metrics,
        )
    return ok_record_with_metrics(
        case.task,
        tool,
        "JAXScape",
        case.name,
        metrics["forward_seconds"],
        metrics=metrics,
    )


def run_worker(payload):
    case = (
        BenchmarkCase(**json.loads(Path(payload["case_path"]).read_text()))
        if "case_path" in payload
        else case_by_name(CASES["free_energy"], payload["case_name"])
    )
    return run_profile(
        case,
        theta=float(payload["theta"]),
        device=device_for_backend(payload["backend"]),
        repeats=int(payload["repeats"]),
        solver_settings=payload.get("solver_settings"),
    )


def collect_jaxscape_free_energy_results(case, config, *, thetas=THETAS):
    records = []
    # Pass the realized case, not regenerated device-dependent random values.
    # A file also avoids the worker environment's argument-size limit.
    with tempfile.TemporaryDirectory(prefix="jaxscape-fed-") as directory:
        case_path = write_case_payload(case, Path(directory))
        for theta in thetas:
            tool = f"JAXScape / CG / theta={theta:g} / f64"
            for backend, device in (
                ("cpu", config.cpu_device),
                ("gpu", config.gpu_device),
            ):
                if device is None:
                    records.append(gpu_placeholder_record(case, tool, "JAXScape"))
                    continue
                records.append(
                    run_worker_subprocess(
                        script_path=Path(__file__).resolve(),
                        payload=dict(
                            case_path=str(case_path),
                            theta=theta,
                            backend=backend,
                            repeats=config.repeats,
                        ),
                        case_task=case.task,
                        case_name=case.name,
                        tool_label=backend_tool_label(tool, backend),
                        software="JAXScape",
                        walltime_seconds=benchmark_walltime_seconds(),
                        logger=LOGGER,
                    )
                )
    return records


def collect_task_results(config):
    return [
        record
        for case in CASES["free_energy"]
        for record in collect_jaxscape_free_energy_results(case, config)
    ]


if __name__ == "__main__":
    run_standalone_task(
        collect_task_results=collect_task_results,
        config_factory=default_jaxscape_config,
        cases=CASES["free_energy"],
        worker_handler=run_worker,
    )
