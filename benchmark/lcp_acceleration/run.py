"""Compare public LCPDistance backends; see README.md for the measurement contract."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, UTC
from importlib.metadata import version
from pathlib import Path
from types import ModuleType

import equinox
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jaxscape import GridGraph, LCPDistance
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


METHODS = ("before", "after")
KINDS = ("forward", "value_and_grad")
ROOT = Path(__file__).resolve().parents[2]


def conductance(a, b):
    """Use the investigation's positive cell-cost parameterization."""
    return 2 / (a + b)


def make_case(size, source_count, seed, landscape, dtype):
    """Build deterministic rook-grid inputs; preserve investigation RNG order."""
    rng = np.random.default_rng(seed)
    shape = (1, size) if landscape == "corridor" else (size, size)
    raster = jnp.asarray(rng.uniform(0.2, 1.0, shape), dtype=dtype)
    if landscape == "clipped":
        base = jax.random.uniform(
            jax.random.PRNGKey(seed), shape, dtype=dtype, minval=0.3, maxval=1.0
        )
        ridge = jnp.exp(-3 * jnp.linspace(-1.0, 1.0, size, dtype=dtype) ** 2)
        raster = jnp.clip(base - 0.4 * jnp.outer(ridge, jnp.ones(size)), 0.15, 1.0)
    elif landscape == "uniform":
        raster = jnp.ones(shape, dtype=dtype)
    if landscape == "corridor":
        sources = jnp.array([0], dtype=jnp.int32)
        targets = jnp.array([size - 1], dtype=jnp.int32)
    else:
        sources = jnp.asarray(
            rng.choice(raster.size, size=source_count, replace=False), dtype=jnp.int32
        )
        targets = jnp.asarray(
            rng.choice(raster.size, size=min(19, raster.size), replace=False),
            dtype=jnp.int32,
        )
    return raster, sources, targets


def build_function(distance_class, sources, targets):
    """Keep graph construction in the measured and differentiated function."""
    metric = distance_class()

    def distances(raster):
        return metric(
            GridGraph(raster, fun=conductance), sources=sources, targets=targets
        )

    return distances


def error_metrics(value, reference):
    value, reference = np.asarray(value), np.asarray(reference)
    return {
        "max_abs": float(np.max(np.abs(value - reference), initial=0)),
        "relative_l2": float(
            np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-30)
        ),
        "finite": bool(np.isfinite(value).all()),
    }


def memory_estimate(compiled):
    """Return optional XLA estimates, never measured peak memory."""
    try:
        memory = compiled.memory_analysis()
    except (NotImplementedError, RuntimeError) as exc:
        return {"available": False, "reason": str(exc)}
    if memory is None:
        return {"available": False, "reason": "Executable exposes no memory analysis"}
    return {
        "available": True,
        "measurement_type": "XLA executable compiler estimate; not measured peak",
        "temporary_bytes": memory.temp_size_in_bytes,
        "argument_bytes": memory.argument_size_in_bytes,
        "output_bytes": memory.output_size_in_bytes,
        "alias_bytes": memory.alias_size_in_bytes,
    }


def benchmark_case(size, source_count, args, landscape, distance_classes):
    raster, sources, targets = make_case(
        size, source_count, args.seed, landscape, getattr(jnp, args.dtype)
    )
    jax.block_until_ready((raster, sources, targets))
    adjacency = GridGraph(raster, fun=conductance).get_adjacency_matrix()
    present = np.asarray(adjacency.data) > 0
    indices = np.asarray(adjacency.indices)[present]
    # These GridGraph workloads contain no positive duplicate coordinates.
    costs = np.asarray(1 / adjacency.data[present], dtype=np.float64)
    reference = dijkstra(
        csr_matrix((costs, indices.T), shape=(raster.size, raster.size)),
        directed=True,
        indices=np.asarray(sources),
    )[:, np.asarray(targets)]
    result = {
        "shape": list(raster.shape),
        "landscape": landscape,
        "seed": args.seed,
        "sources": np.asarray(sources).tolist(),
        "targets": np.asarray(targets).tolist(),
        "input_sha256": hashlib.sha256(np.asarray(raster).tobytes()).hexdigest(),
        "methods": {},
        "sample_order": [],
    }
    functions, outputs = {}, {}
    for algorithm in METHODS:
        distances = build_function(distance_classes[algorithm], sources, targets)
        variants = {
            "forward": distances,
            "value_and_grad": jax.value_and_grad(lambda x: distances(x).sum()),
        }
        result["methods"][algorithm] = {}
        for kind, function in variants.items():
            start = time.perf_counter()
            compiled = jax.jit(function).lower(raster).compile()
            compilation = time.perf_counter() - start
            output = jax.block_until_ready(compiled(raster))
            for _ in range(args.warmups - 1):
                output = jax.block_until_ready(compiled(raster))
            functions[algorithm, kind] = compiled
            outputs[algorithm, kind] = output
            result["methods"][algorithm][kind] = {
                "lower_and_compile_seconds": compilation,
                "xla_memory_estimate": memory_estimate(compiled),
                "samples_seconds": [],
            }
        tolerance = 3e-6 if args.dtype == "float32" else 1e-12
        np.testing.assert_allclose(
            outputs[algorithm, "forward"], reference, rtol=tolerance, atol=tolerance
        )
        np.testing.assert_allclose(
            outputs[algorithm, "value_and_grad"][0],
            reference.sum(),
            rtol=tolerance,
            atol=tolerance,
        )
        if not np.isfinite(np.asarray(outputs[algorithm, "value_and_grad"][1])).all():
            raise ValueError(f"Nonfinite gradient for {algorithm}, {landscape}")
        result["methods"][algorithm]["distance_error_scipy"] = error_metrics(
            outputs[algorithm, "forward"], reference
        )
    for algorithm in METHODS:
        # Descriptive only: tied inputs deliberately need not match finite-scan AD.
        result["methods"][algorithm][
            "gradient_error_historical_backend"
        ] = error_metrics(
            outputs[algorithm, "value_and_grad"][1],
            outputs["before", "value_and_grad"][1],
        )
    rng = np.random.default_rng(args.seed + 1)
    keys = list(functions)
    for _ in range(args.repeats):
        order = [keys[i] for i in rng.permutation(len(keys))]
        result["sample_order"].append([list(key) for key in order])
        for algorithm, kind in order:
            start = time.perf_counter()
            jax.block_until_ready(functions[algorithm, kind](raster))
            elapsed = time.perf_counter() - start
            result["methods"][algorithm][kind]["samples_seconds"].append(elapsed)
    for algorithm in METHODS:
        for kind in KINDS:
            metrics = result["methods"][algorithm][kind]
            metrics["median_seconds"] = float(np.median(metrics["samples_seconds"]))
            print(
                f"{landscape} {tuple(raster.shape)} S={sources.size} "
                f"{algorithm} {kind}: {1000 * metrics['median_seconds']:.3f} ms",
                flush=True,
            )
    return result


def device_inventory():
    inventory = {}
    for backend in ("cpu", "gpu"):
        try:
            inventory[backend] = {
                "available": True,
                "devices": [str(device) for device in jax.devices(backend)],
            }
        except RuntimeError as exc:
            inventory[backend] = {"available": False, "reason": str(exc)}
    return inventory


def git_output(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def hardware_description():
    if platform.system() == "Darwin":
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    return platform.processor() or platform.machine()


def load_baseline(ref):
    """Load the old public metric only for before/after benchmarking."""
    revision = git_output("rev-parse", f"{ref}^{{commit}}")
    source = git_output("show", f"{revision}:src/jaxscape/lcp_distance.py")
    module = ModuleType("_lcp_benchmark_baseline")
    sys.modules[module.__name__] = module
    exec(compile(source, f"{revision}:lcp_distance.py", "exec"), module.__dict__)
    return getattr(module, "LCPDistance"), revision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="33e609b")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--sources", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--corridor-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if min(args.repeats, args.warmups) < 1:
        parser.error("--repeats and --warmups must be positive")
    if min(args.sizes) < 2 or args.corridor_size < 2:
        parser.error("Grid and corridor sizes must be at least two")
    if min(args.sources) < 1 or max(args.sources) > min(args.sizes) ** 2:
        parser.error("Source counts must fit every random grid")
    jax.config.update("jax_enable_x64", args.dtype == "float64")
    jax.config.update("jax_enable_compilation_cache", False)
    inventory = device_inventory()
    if not inventory[args.device]["available"]:
        parser.error(f"Requested {args.device} unavailable: {inventory[args.device]}")
    device = jax.devices(args.device)[0]
    baseline, baseline_revision = load_baseline(args.baseline_ref)
    distance_classes = {"before": baseline, "after": LCPDistance}
    report = {
        "measurement": "public LCPDistance before/after default implementation",
        "baseline_revision": baseline_revision,
        "baseline_policy": "historical lcp_distance.py, current graph/API/dependencies",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "hardware": hardware_description(),
        "python": sys.version,
        "versions": {
            "jax": jax.__version__,
            "jaxlib": version("jaxlib"),
            "equinox": equinox.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "devices": inventory,
        "selected_device": str(device),
        "device_kind": device.device_kind,
        "dtype": args.dtype,
        "revision": git_output("rev-parse", "HEAD"),
        "tracked_worktree_changes": git_output(
            "status", "--short", "--untracked-files=no"
        ),
        "implementation_sha256": hashlib.sha256(
            (ROOT / "src/jaxscape/lcp_distance.py").read_bytes()
        ).hexdigest(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "command": sys.argv,
        "objective": "sum of selected distances; derivative w.r.t. raster",
        "edge_conductance": "2 / (cell_cost_u + cell_cost_v), rook neighborhoods",
        "policy": {
            "compilation": "lower/compile separate; persistent cache disabled",
            "warmups_per_executable": args.warmups,
            "repeats": args.repeats,
            "synchronization": "jax.block_until_ready on entire output for every call",
            "order": "randomized across methods and forward/gradient each repetition",
            "included": "graph/adjacency construction, distance/gradient",
            "excluded": "inputs, compilation, warmup, correctness checks",
            "memory": "XLA compiler estimate; no measured peak memory",
        },
        "cases": [],
    }
    cases = [(size, count, "random") for size in args.sizes for count in args.sources]
    cases += [(max(args.sizes), 1, "clipped"), (min(args.sizes), 1, "uniform")]
    cases += [(args.corridor_size, 1, "corridor")]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with jax.default_device(device):
        for size, count, landscape in cases:
            report["cases"].append(
                benchmark_case(size, count, args, landscape, distance_classes)
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            jax.clear_caches()
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
