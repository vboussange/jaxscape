"""Reusable benchmark suite for JAXScape workloads.

The script benchmarks three workloads discussed in the project roadmap:

1. connectivity analysis (forward pass)
2. sensitivity analysis (gradient with respect to permeability)
3. inverse landscape genetics (optional, requires ``optimistix``)

Results are written to JSON so they can be compared across CPU/GPU runs and,
when desired, against external tools on shared benchmark landscapes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from jaxscape import EuclideanDistance, GridGraph, LCPDistance, ResistanceDistance, RSPDistance

try:
    import optimistix as optx
except ImportError as exc:  # pragma: no cover - exercised in tests through the skip path
    optx = None
    OPTIMISTIX_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on optional extra
    OPTIMISTIX_IMPORT_ERROR = None


RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_OUTPUT = RESULTS_DIR / "benchmark_results.json"
SCENARIOS = ("connectivity", "sensitivity", "inverse")


def edge_weight(x: Array, y: Array) -> Array:
    return (x + y) / 2


@dataclass(frozen=True)
class BenchmarkConfig:
    grid_sizes: tuple[int, ...] = (8, 12, 16)
    repeats: int = 3
    seed: int = 0
    dispersal_radius: float = 5.0
    distance_name: str = "resistance"
    device_names: tuple[str, ...] = ("cpu",)
    scenario_names: tuple[str, ...] = ("connectivity", "sensitivity", "inverse")
    inverse_max_steps: int = 25
    landscape_path: str | None = None
    output_path: str = str(DEFAULT_OUTPUT)


def available_devices() -> dict[str, list[jax.Device]]:
    devices: dict[str, list[jax.Device]] = {}
    for platform in ("cpu", "gpu", "tpu"):
        try:
            platform_devices = list(jax.devices(platform))
        except RuntimeError:
            platform_devices = []
        if platform_devices:
            devices[platform] = platform_devices
    return devices


def _select_devices(device_names: Iterable[str]) -> list[tuple[str, jax.Device]]:
    devices = available_devices()
    selected: list[tuple[str, jax.Device]] = []
    for device_name in device_names:
        if device_name not in devices:
            continue
        selected.append((device_name, devices[device_name][0]))
    if not selected:
        available = ", ".join(sorted(devices)) or "none"
        msg = f"No requested JAX devices are available. Requested={tuple(device_names)!r}; available={available}."
        raise RuntimeError(msg)
    return selected


def _load_landscape(path: str | Path) -> Array:
    landscape_path = Path(path)
    if landscape_path.suffix == ".npy":
        raster = np.load(landscape_path)
    else:
        raster = np.loadtxt(landscape_path, delimiter=",")
    if raster.ndim != 2:
        raise ValueError(f"Expected a 2D raster, got shape {raster.shape}.")
    return jnp.asarray(raster, dtype=jnp.float32)


def create_landscape(size: int, seed: int = 0) -> Array:
    key = jr.PRNGKey(seed)
    base = jr.uniform(key, (size, size), minval=0.2, maxval=1.0)
    x = jnp.linspace(-1.0, 1.0, size)
    ridge = jnp.exp(-4.0 * x**2)
    barrier = 0.35 * jnp.outer(ridge, jnp.ones_like(ridge))
    landscape = jnp.clip(base - barrier, 0.05, 1.0)
    return landscape.astype(jnp.float32)


def _create_quality(size: int, seed: int) -> Array:
    key = jr.PRNGKey(seed + 1)
    quality = jr.uniform(key, (size, size), minval=0.5, maxval=1.0)
    return quality.astype(jnp.float32)


def _distance_from_name(name: str):
    if name == "euclidean":
        return EuclideanDistance()
    if name == "lcp":
        return LCPDistance()
    if name == "resistance":
        return ResistanceDistance()
    if name == "rsp":
        return RSPDistance(theta=jnp.array(1e-3, dtype=jnp.float32))
    raise ValueError(
        f"Unknown distance {name!r}. Expected one of ('euclidean', 'lcp', 'resistance', 'rsp')."
    )


def _connectivity_score(
    permeability_raster: Array,
    quality_raster: Array,
    distance,
    dispersal_radius: Array,
) -> Array:
    grid = GridGraph(grid=permeability_raster, fun=edge_weight)
    quality = grid.array_to_node_values(quality_raster)
    dist = distance(grid)
    kernel = jnp.exp(-dist / dispersal_radius)
    kernel = kernel.at[jnp.diag_indices_from(kernel)].set(0)
    return quality @ kernel @ quality.T


connectivity_score = eqx.filter_jit(_connectivity_score)
sensitivity_score = eqx.filter_jit(eqx.filter_grad(_connectivity_score))


def _sample_coordinates(size: int) -> Array:
    candidates = jnp.array(
        [
            [0, 0],
            [0, size - 1],
            [size - 1, 0],
            [size - 1, size - 1],
            [size // 2, size // 2],
        ],
        dtype=jnp.int32,
    )
    return jnp.unique(candidates, axis=0)


def _inverse_loss(logits: Array, sample_coords: Array, target_distances: Array) -> Array:
    permeability = jnn.sigmoid(logits) + jnp.array(1e-3, dtype=logits.dtype)
    grid = GridGraph(grid=permeability, fun=edge_weight)
    predicted = ResistanceDistance()(grid, nodes=sample_coords)
    return jnp.mean((predicted - target_distances) ** 2)


def _run_inverse_problem(landscape: Array, max_steps: int) -> dict[str, Any]:
    if optx is None:
        return {
            "status": "skipped",
            "reason": (
                "Optional dependency 'optimistix' is not installed. "
                f"Original import error: {OPTIMISTIX_IMPORT_ERROR}"
            ),
        }

    sample_coords = _sample_coordinates(landscape.shape[0])
    target_distances = ResistanceDistance()(
        GridGraph(landscape, fun=edge_weight), nodes=sample_coords
    )
    initial_logits = jnp.zeros_like(landscape)
    solver = optx.LBFGS(rtol=1e-5, atol=1e-5)
    solution = optx.minimise(
        _inverse_loss,
        solver,
        initial_logits,
        args=(sample_coords, target_distances),
        max_steps=max_steps,
    )
    return {
        "status": "ok",
        "final_loss": float(
            _inverse_loss(solution.value, sample_coords, target_distances)
        ),
        "num_steps": int(solution.stats["num_steps"]),
    }


def _measure_runtime(
    fun: Callable[..., Any], *args: Any, repeats: int
) -> tuple[list[float], Any]:
    result = fun(*args)
    jax.block_until_ready(result)

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fun(*args)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)
    return timings, result


def _to_platform_arrays(device: jax.Device, *arrays: Array) -> list[Array]:
    return [jax.device_put(array, device) for array in arrays]


def _run_connectivity_case(
    landscape: Array, quality: Array, config: BenchmarkConfig, device_name: str, device: jax.Device
) -> dict[str, Any]:
    distance = _distance_from_name(config.distance_name)
    permeability_device, quality_device = _to_platform_arrays(device, landscape, quality)
    dispersal_radius = jax.device_put(
        jnp.array(config.dispersal_radius, dtype=jnp.float32), device
    )
    timings, result = _measure_runtime(
        connectivity_score,
        permeability_device,
        quality_device,
        distance,
        dispersal_radius,
        repeats=config.repeats,
    )
    return {
        "scenario": "connectivity",
        "device": device_name,
        "distance": config.distance_name,
        "grid_shape": list(map(int, landscape.shape)),
        "timings_seconds": timings,
        "median_seconds": statistics.median(timings),
        "connectivity": float(result),
    }


def _run_sensitivity_case(
    landscape: Array, quality: Array, config: BenchmarkConfig, device_name: str, device: jax.Device
) -> dict[str, Any]:
    distance = _distance_from_name(config.distance_name)
    permeability_device, quality_device = _to_platform_arrays(device, landscape, quality)
    dispersal_radius = jax.device_put(
        jnp.array(config.dispersal_radius, dtype=jnp.float32), device
    )
    timings, gradient = _measure_runtime(
        sensitivity_score,
        permeability_device,
        quality_device,
        distance,
        dispersal_radius,
        repeats=config.repeats,
    )
    return {
        "scenario": "sensitivity",
        "device": device_name,
        "distance": config.distance_name,
        "grid_shape": list(map(int, landscape.shape)),
        "timings_seconds": timings,
        "median_seconds": statistics.median(timings),
        "gradient_norm": float(jnp.linalg.norm(gradient)),
    }


def _run_inverse_case(
    landscape: Array, config: BenchmarkConfig, device_name: str, device: jax.Device
) -> dict[str, Any]:
    landscape_device = jax.device_put(landscape, device)
    timings, result = _measure_runtime(
        _run_inverse_problem, landscape_device, config.inverse_max_steps, repeats=config.repeats
    )
    payload = {
        "scenario": "inverse",
        "device": device_name,
        "distance": "resistance",
        "grid_shape": list(map(int, landscape.shape)),
        "timings_seconds": timings,
        "median_seconds": statistics.median(timings),
    }
    payload.update(result)
    return payload


def _scenario_runner(name: str) -> Callable[..., dict[str, Any]]:
    if name == "connectivity":
        return _run_connectivity_case
    if name == "sensitivity":
        return _run_sensitivity_case
    if name == "inverse":
        return _run_inverse_case
    raise ValueError(f"Unknown scenario {name!r}.")


def run_benchmarks(config: BenchmarkConfig) -> dict[str, Any]:
    selected_devices = _select_devices(config.device_names)
    if config.landscape_path is None:
        landscapes = [
            (
                create_landscape(size, seed=config.seed),
                _create_quality(size, seed=config.seed),
            )
            for size in config.grid_sizes
        ]
    else:
        landscape = _load_landscape(config.landscape_path)
        landscapes = [(landscape, _create_quality(int(landscape.shape[0]), seed=config.seed))]

    results: dict[str, Any] = {
        "config": json.loads(json.dumps(asdict(config))),
        "available_devices": {
            name: [str(device) for device in devices]
            for name, devices in available_devices().items()
        },
        "runs": [],
    }

    for scenario_name in config.scenario_names:
        runner = _scenario_runner(scenario_name)
        for landscape, quality in landscapes:
            for device_name, device in selected_devices:
                if scenario_name == "inverse":
                    run = runner(landscape, config, device_name, device)
                else:
                    run = runner(landscape, quality, config, device_name, device)
                results["runs"].append(run)

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    return results


def _parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=(*SCENARIOS, "all"),
        default="all",
        help="Benchmark workload to execute.",
    )
    parser.add_argument(
        "--distance",
        choices=("euclidean", "lcp", "resistance", "rsp"),
        default="resistance",
        help="Distance metric used for connectivity and sensitivity benchmarks.",
    )
    parser.add_argument(
        "--grid-sizes",
        nargs="+",
        type=int,
        default=list(BenchmarkConfig.grid_sizes),
        help="Synthetic square raster sizes to benchmark.",
    )
    parser.add_argument(
        "--device",
        dest="device_names",
        nargs="+",
        choices=("cpu", "gpu", "tpu"),
        default=list(BenchmarkConfig.device_names),
        help="JAX platforms to benchmark.",
    )
    parser.add_argument("--repeats", type=int, default=BenchmarkConfig.repeats)
    parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    parser.add_argument(
        "--dispersal-radius",
        type=float,
        default=BenchmarkConfig.dispersal_radius,
    )
    parser.add_argument(
        "--inverse-max-steps",
        type=int,
        default=BenchmarkConfig.inverse_max_steps,
    )
    parser.add_argument(
        "--landscape-path",
        type=str,
        default=None,
        help="Optional .npy or .csv raster to benchmark instead of synthetic landscapes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to the JSON results file.",
    )
    args = parser.parse_args()
    scenario_names = SCENARIOS if args.scenario == "all" else (args.scenario,)
    return BenchmarkConfig(
        grid_sizes=tuple(args.grid_sizes),
        repeats=args.repeats,
        seed=args.seed,
        dispersal_radius=args.dispersal_radius,
        distance_name=args.distance,
        device_names=tuple(args.device_names),
        scenario_names=scenario_names,
        inverse_max_steps=args.inverse_max_steps,
        landscape_path=args.landscape_path,
        output_path=args.output,
    )


def main() -> None:
    config = _parse_args()
    results = run_benchmarks(config)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
