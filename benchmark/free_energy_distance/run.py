"""Reproduce issue #31 subset timings and numerical evidence on this machine."""

import argparse
import hashlib
import importlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, UTC
from pathlib import Path
from typing import cast, TypedDict

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxscape import FreeEnergyDistance, GridGraph


# Enable the requested float64 cases in this benchmark process, never in the library.
jax.config.update("jax_enable_x64", True)
SEED = 20261026
CORE = importlib.import_module("jaxscape.free_energy_distance")


class Timing(TypedDict):
    median_ms: float
    samples_ms: list[float]


def symmetric_weight(x, y):
    return (x + y) / 2


def endpoints(side, count):
    center = (side // 2 - 1) * side + side // 2 - 1
    if count == 3:
        return jnp.array([center, center + 1, center + side]), None
    return (
        jnp.array([center, center + 1]),
        jnp.array([center + side, center + side + 1, center + 2 * side]),
    )


def loss_weights(count):
    if count == 3:
        return jnp.array([[0.0, 1.0, -0.4], [0.3, 0.0, 0.7], [-0.2, -0.6, 0.0]])
    return jnp.array([[1.0, -0.4, 0.7], [-0.2, 0.3, -1.1]])


def independent_reference(values, theta, sources, targets) -> jax.Array:
    """Dense nonsymmetric reference from P/W; no production assembly or VJP."""
    side = values.shape[0]
    n = values.size
    vertices = np.arange(n).reshape(side, side)
    left = np.concatenate([vertices[:, :-1].ravel(), vertices[:-1, :].ravel()])
    right = np.concatenate([vertices[:, 1:].ravel(), vertices[1:, :].ravel()])
    flat = values.ravel()
    conductances = (flat[left] + flat[right]) / 2
    adjacency = jnp.zeros((n, n), dtype=values.dtype)
    adjacency = adjacency.at[left, right].set(conductances)
    adjacency = adjacency.at[right, left].set(conductances)
    present = adjacency > 0
    costs = 1 / jnp.where(present, adjacency, 1)
    transition = adjacency / adjacency.sum(axis=1, keepdims=True)
    walk = jnp.where(present, transition * jnp.exp(-theta * costs), 0)
    nodes = sources if targets is None else jnp.concatenate([sources, targets])
    # Only selected columns of Z=(I-W)^-1 are needed even in the reference.
    basis = jax.nn.one_hot(nodes, n, dtype=values.dtype).T
    selected = jnp.linalg.solve(jnp.eye(n, dtype=values.dtype) - walk, basis)[nodes]
    diagonal = jnp.diag(selected)
    ns = sources.size
    rows = jnp.arange(ns)
    cols = rows if targets is None else jnp.arange(targets.size) + ns
    st = selected[rows[:, None], cols[None, :]]
    ts = selected[cols[:, None], rows[None, :]].T
    result = (
        jnp.log(diagonal[rows])[:, None]
        + jnp.log(diagonal[cols])[None, :]
        - jnp.log(st)
        - jnp.log(ts)
    ) / (2 * theta)
    return cast(
        jax.Array,
        jnp.where(
            sources[:, None] == (sources if targets is None else targets)[None, :],
            0,
            result,
        ),
    )


def public_distances(values, theta, sources, targets, solver) -> jax.Array:
    graph = GridGraph(values, fun=symmetric_weight)
    distance = FreeEnergyDistance(theta=theta, solver=solver)
    if targets is None:
        return cast(jax.Array, distance(graph, nodes=sources))
    return cast(jax.Array, distance(graph, sources=sources, targets=targets))


def synchronize(value):
    return jax.block_until_ready(value)


def timed_samples(function, arguments, samples) -> Timing:
    synchronize(function(*arguments))
    timings = []
    for _ in range(samples):
        start = time.perf_counter()
        synchronize(function(*arguments))
        timings.append((time.perf_counter() - start) * 1_000)
    return {"median_ms": statistics.median(timings), "samples_ms": timings}


def paired_timings(checked, unchecked, arguments, samples) -> tuple[Timing, ...]:
    """Interleave identical solves in seeded order to measure check overhead."""
    synchronize(checked(*arguments))
    synchronize(unchecked(*arguments))
    times = {"checked": [], "unchecked": []}
    rng = np.random.default_rng(SEED)
    for _ in range(samples):
        for name in rng.permutation(["checked", "unchecked"]).tolist():
            function = checked if name == "checked" else unchecked
            start = time.perf_counter()
            synchronize(function(*arguments))
            times[name].append((time.perf_counter() - start) * 1_000)
    return tuple(
        Timing(median_ms=statistics.median(times[name]), samples_ms=times[name])
        for name in ["checked", "unchecked"]
    )


def compile_function(function, arguments):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*arguments).compile()
    return compiled, (time.perf_counter() - start) * 1_000


def memory_metadata(compiled):
    memory = compiled.memory_analysis()
    keys = (
        "argument_size_in_bytes",
        "output_size_in_bytes",
        "alias_size_in_bytes",
        "temp_size_in_bytes",
        "generated_code_size_in_bytes",
    )
    return {key: int(getattr(memory, key)) for key in keys}


def trace_summary(closed, n):
    primitives = Counter()
    shapes = set()
    seen = set()

    def visit(value):
        if id(value) in seen:
            return
        if hasattr(value, "jaxpr"):
            seen.add(id(value))
            visit(value.jaxpr)
        elif hasattr(value, "eqns"):
            seen.add(id(value))
            for variable in [*value.constvars, *value.invars, *value.outvars]:
                shape = getattr(getattr(variable, "aval", None), "shape", None)
                if shape is not None:
                    shapes.add(tuple(shape))
            for equation in value.eqns:
                primitives[equation.primitive.name] += 1
                for variable in [*equation.invars, *equation.outvars]:
                    shape = getattr(getattr(variable, "aval", None), "shape", None)
                    if shape is not None:
                        shapes.add(tuple(shape))
                visit(equation.params)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(closed)
    solve_names = (
        "custom_linear_solve",
        "lu",
        "cholesky",
        "triangular_solve",
        "while",
        "pure_callback",
    )
    return {
        "primitive_counts": dict(sorted(primitives.items())),
        "solve_or_iteration_primitives": {
            name: primitives[name] for name in solve_names if primitives[name]
        },
        "has_n_by_n_array": (n, n) in shapes,
        "largest_array_elements": max((int(np.prod(s)) for s in shapes), default=0),
        "array_shapes": [list(shape) for shape in sorted(shapes)],
    }


@contextmanager
def bypass_acceptance_for_measurement():
    """Benchmark instrumentation only: no library/public accuracy switch exists."""
    original_entry_error = CORE._entry_error
    original_check = CORE._check_distances

    def zero_entry_error(operator, basis, potentials, killing):
        del operator, basis, killing
        return jnp.zeros(potentials.shape[1], dtype=potentials.dtype)

    setattr(CORE, "_entry_error", zero_entry_error)
    setattr(CORE, "_check_distances", lambda distances, *args: distances)
    # Decorated distance methods must be retraced after changing private globals.
    jax.clear_caches()
    eqx.clear_caches()
    try:
        yield
    finally:
        setattr(CORE, "_entry_error", original_entry_error)
        setattr(CORE, "_check_distances", original_check)
        jax.clear_caches()
        eqx.clear_caches()


def error_metrics(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return {
        "max_absolute": float(np.max(np.abs(actual - expected))),
        "relative_l2": float(
            np.linalg.norm(actual - expected)
            / max(float(np.linalg.norm(expected)), float(np.finfo(float).tiny))
        ),
    }


def case(side, count, dtype_name, backend, samples):
    dtype = jnp.dtype(dtype_name)
    rng = np.random.default_rng(SEED)
    values = jnp.asarray(rng.uniform(0.8, 1.4, (side, side)), dtype=dtype)
    theta = jnp.asarray(1.0, dtype=dtype)
    sources, targets = endpoints(side, count)
    weights = loss_weights(count).astype(dtype)
    tolerance = 1e-7 if dtype_name == "float32" else 1e-12
    solver = (
        None
        if backend == "dense"
        else lx.CG(rtol=tolerance, atol=tolerance * 0.01, max_steps=1_000)
    )

    def setup(values, theta):
        graph = GridGraph(values, fun=symmetric_weight)
        return FreeEnergyDistance(theta=theta, solver=solver).init(graph)

    compiled_setup, compile_setup_ms = compile_function(setup, (values, theta))
    setup_timing = timed_samples(compiled_setup, (values, theta), samples)

    def forward(values, theta):
        return public_distances(values, theta, sources, targets, solver)

    def objective(values, theta):
        return jnp.sum(forward(values, theta) * weights)

    arguments = values, theta
    compiled_forward, compile_forward_ms = compile_function(forward, arguments)
    gradient = jax.value_and_grad(objective, argnums=(0, 1))
    compiled_gradient, compile_gradient_ms = compile_function(gradient, arguments)
    actual = synchronize(compiled_forward(*arguments))
    actual_value, actual_gradient = synchronize(compiled_gradient(*arguments))
    gradient_timing = timed_samples(compiled_gradient, arguments, samples)

    def reference(values, theta):
        return independent_reference(values, theta, sources, targets)

    def reference_objective(values, theta):
        return jnp.sum(reference(values, theta) * weights.astype(jnp.float64))

    # Accuracy is compared with a float64 independent reference in both dtypes.
    reference_arguments = values.astype(jnp.float64), theta.astype(jnp.float64)
    expected = synchronize(jax.jit(reference)(*reference_arguments))
    expected_value, expected_gradient = synchronize(
        jax.jit(jax.value_and_grad(reference_objective, argnums=(0, 1)))(
            *reference_arguments
        )
    )
    _, pullback = jax.vjp(objective, *arguments)
    backward_jaxpr = jax.make_jaxpr(pullback)(jnp.ones_like(actual_value))

    def backward_from_saved_arrays(saved_arrays, cotangent):
        # Passing saved forward arrays as runtime arguments prevents XLA from
        # folding a fixed-input pullback into a precomputed gradient constant.
        return jax_core.eval_jaxpr(backward_jaxpr.jaxpr, saved_arrays, cotangent)

    backward_arguments = (tuple(backward_jaxpr.consts), jnp.ones_like(actual_value))
    backward_compiled, compile_backward_ms = compile_function(
        backward_from_saved_arrays, backward_arguments
    )
    backward_timing = timed_samples(backward_compiled, backward_arguments, samples)
    backward_result = synchronize(backward_compiled(*backward_arguments))
    for actual_leaf, backward_leaf in zip(actual_gradient, backward_result):
        np.testing.assert_allclose(backward_leaf, actual_leaf, rtol=1e-5, atol=1e-7)
    with bypass_acceptance_for_measurement():
        unchecked_forward, unchecked_compile_ms = compile_function(forward, arguments)
        unchecked_actual = synchronize(unchecked_forward(*arguments))
    forward_timing, unchecked_timing = paired_timings(
        compiled_forward, unchecked_forward, arguments, samples
    )
    np.testing.assert_allclose(unchecked_actual, actual, rtol=0, atol=0)
    return {
        "side": side,
        "vertices": side**2,
        "stored_adjacency_entries": 4 * side**2,
        "present_directed_edges": 4 * side * (side - 1),
        "selected_columns": count,
        "dtype": dtype_name,
        "backend": backend,
        "theta": float(theta),
        "seed": SEED,
        "solver": None
        if solver is None
        else {
            "name": "lineax.CG",
            "rtol": tolerance,
            "atol": tolerance * 0.01,
            "max_steps": 1_000,
        },
        "setup": setup_timing,
        "compile_ms": {
            "setup": compile_setup_ms,
            "forward": compile_forward_ms,
            "value_and_grad": compile_gradient_ms,
            "backward_only": compile_backward_ms,
            "without_acceptance": unchecked_compile_ms,
        },
        "forward": forward_timing,
        "value_and_grad": gradient_timing,
        "backward_only": backward_timing,
        "without_acceptance": unchecked_timing,
        "acceptance_overhead_ms": (
            forward_timing["median_ms"] - unchecked_timing["median_ms"]
        ),
        "acceptance_overhead_ratio": (
            forward_timing["median_ms"] / unchecked_timing["median_ms"] - 1
        ),
        "accuracy": {
            "distances": error_metrics(actual, expected),
            "objective": error_metrics(actual_value, expected_value),
            "raster_gradient": error_metrics(actual_gradient[0], expected_gradient[0]),
            "theta_gradient": error_metrics(actual_gradient[1], expected_gradient[1]),
        },
        "compiler_memory": {
            "forward": memory_metadata(compiled_forward),
            "value_and_grad": memory_metadata(compiled_gradient),
            "backward_only": memory_metadata(backward_compiled),
        },
        "forward_trace": trace_summary(jax.make_jaxpr(forward)(*arguments), side**2),
        "backward_trace": trace_summary(backward_jaxpr, side**2),
        "status": "accepted",
    }


def numerical_probes():
    """Record accepted values/gradients and explicit rejected finite precisions."""
    side = 8
    base = np.random.default_rng(SEED).uniform(0.8, 1.4, (side, side))
    sources, targets = endpoints(side, 3)
    records = []
    for dtype_name in ["float32", "float64"]:
        values = jnp.asarray(base, dtype=dtype_name)
        weights = loss_weights(3).astype(dtype_name)

        def forward(values, theta):
            return public_distances(values, theta, sources, targets, None)

        def objective(values, theta):
            return jnp.sum(forward(values, theta) * weights)

        def reference_objective(values, theta):
            return jnp.sum(
                independent_reference(values, theta, sources, targets)
                * weights.astype(jnp.float64)
            )

        function = jax.jit(forward)
        gradient = jax.jit(jax.grad(objective, argnums=(0, 1)))
        reference_gradient = jax.jit(jax.grad(reference_objective, argnums=(0, 1)))
        for theta_value in [1e-12, 1e-8, 1e-5, 0.001, 0.01, 0.1, 1.0, 5.0, 50.0, 500.0]:
            record = {
                "dtype": dtype_name,
                "theta": theta_value,
                "vertices": side**2,
                "selected_columns": 3,
                "backend": "dense",
            }
            theta = jnp.asarray(theta_value, dtype=dtype_name)
            try:
                actual = synchronize(function(values, theta))
                gradients = synchronize(gradient(values, theta))
                expected = synchronize(
                    jax.jit(independent_reference)(
                        values.astype(jnp.float64),
                        theta.astype(jnp.float64),
                        sources,
                        targets,
                    )
                )
                expected_gradients = synchronize(
                    reference_gradient(
                        values.astype(jnp.float64), theta.astype(jnp.float64)
                    )
                )
                record.update(
                    status="accepted",
                    accuracy={
                        "distances": error_metrics(actual, expected),
                        "raster_gradient": error_metrics(
                            gradients[0], expected_gradients[0]
                        ),
                        "theta_gradient": error_metrics(
                            gradients[1], expected_gradients[1]
                        ),
                    },
                )
            except Exception as error:
                record.update(status="rejected", error=failure_message(error))
            records.append(record)
            print("numerical", dtype_name, theta_value, record["status"], flush=True)
    return records


def failure_message(error):
    for line in str(error).splitlines():
        if "FreeEnergyDistance" in line:
            return line.strip()
    return str(error).split("\n\n")[0][-2_000:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("results.json")
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--columns", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--backends", nargs="+", default=["cg"])
    parser.add_argument("--numerical-only", action="store_true")
    arguments = parser.parse_args()
    source_file = CORE.__file__
    assert source_file is not None
    source = Path(source_file)
    source_root = source.parents[2]
    source_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--porcelain", "--", str(source)],
        text=True,
    ).strip()
    cpu_name = platform.processor()
    if platform.system() == "Darwin":
        cpu_query = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        if cpu_query.returncode == 0:
            cpu_name = cpu_query.stdout.strip()
    results = {
        "metadata": {
            "measured_at_utc": datetime.now(UTC).isoformat(),
            "implementation_revision": subprocess.check_output(
                ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
            ).strip(),
            "implementation_has_local_changes": bool(source_status),
            "implementation_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "thread_environment": {
                key: os.environ.get(key)
                for key in [
                    "XLA_FLAGS",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                ]
            },
            "base_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "platform": platform.platform(),
            "processor": cpu_name,
            "devices": [str(device) for device in jax.devices()],
            "jax": jax.__version__,
            "equinox": eqx.__version__,
            "lineax": lx.__version__,
            "numpy": np.__version__,
            "samples": arguments.samples,
            "memory_measurement": (
                "XLA compiled memory estimate, not measured peak memory"
            ),
            "timing_scope": (
                "Preplaced inputs; graph/adjacency + automatic state + solve + "
                "selection; synchronized; setup additionally reported; "
                "no claim of state-reuse speedup"
            ),
            "objective": (
                "Signed weighted sum of selected distances; jointly "
                "differentiate all raster cells and theta"
            ),
        },
        "cases": [],
    }
    if arguments.numerical_only:
        results["numerical_probes"] = numerical_probes()
        arguments.output.write_text(json.dumps(results, indent=2) + "\n")
        return
    for side in arguments.sizes:
        for count in arguments.columns:
            for dtype_name in arguments.dtypes:
                for backend in arguments.backends:
                    label = f"{side}x{side} K={count} {dtype_name} {backend}"
                    print(label, flush=True)
                    try:
                        record = case(
                            side, count, dtype_name, backend, arguments.samples
                        )
                        print(record["forward"]["median_ms"], "ms", flush=True)
                    except Exception as error:
                        record = {
                            "side": side,
                            "selected_columns": count,
                            "dtype": dtype_name,
                            "backend": backend,
                            "status": "rejected",
                            "error": failure_message(error),
                        }
                        print("rejected:", record["error"][-300:], flush=True)
                    results["cases"].append(record)
                    arguments.output.write_text(json.dumps(results, indent=2) + "\n")
                    jax.clear_caches()
                    eqx.clear_caches()


if __name__ == "__main__":
    main()
