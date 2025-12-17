"""
Benchmark the computation time of for forward and backward pass of equivalent
connected habitat (ECH) for different network sizes on CPU vs GPU.
"""
import json
import time
from pathlib import Path

import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxscape import GridGraph
from jaxscape.lcp_distance import LCPDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.rsp_distance import RSPDistance


path_results = Path("results/benchmarks/")
path_results.mkdir(parents=True, exist_ok=True)


def create_landscape(size):
    habitat_permability = jnp.ones((size, size), dtype="float32")
    return habitat_permability


def calculate_ech(habitat_permability, distance):
    grid = GridGraph(grid=habitat_permability)
    return jnp.sum(distance(grid))


calculate_d_ech_dp = equinox.filter_jit(
    equinox.filter_grad(calculate_ech)
)  # sensitivity to permeability
D = jnp.array(1.0, dtype="float32")


def benchmark(device, fun, distance, node_sizes):
    times = []
    for size in node_sizes:
        try:
            print(
                f"Calculating ECH for size {size} on {device} using {distance.__class__.__name__}"
            )
            habitat_permability, activities, nb_active = create_landscape(size)

            # Warm-up to avoid measuring compile time
            fun(
                jax.device_put(habitat_permability, device),
                jax.device_put(activities),
                nb_active,
                distance,
            ).block_until_ready()

            start_time = time.time()
            fun(
                jax.device_put(habitat_permability, device),
                jax.device_put(activities),
                nb_active,
                distance,
            ).block_until_ready()
            end_time = time.time()

            times.append(end_time - start_time)
        except Exception as e:
            print(
                f"Error occurred for size {size} on {device} using {distance.__class__.__name__}: {e}"
            )
            times.append(float("inf"))  # Use infinity to indicate failure
    return times


def plot_benchmark(node_sizes, cpu_times, gpu_times, distance_name):
    fig, ax = plt.subplots()
    ax.plot(node_sizes**2, cpu_times, marker="o", label="CPU")
    ax.plot(node_sizes**2, gpu_times, marker="o", label="GPU")
    ax.set_xlabel("Nb. of nodes.")
    ax.set_ylabel("Computation time (s)")
    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"Benchmark for {distance_name}")
    return fig, ax


def run_benchmark():
    distances = [
        ResistanceDistance(),
        LCPDistance(),
        #  SmoothLCPDistance(tau=1e-10), requires landmarks, similar to LCPDistance
        RSPDistance(theta=jnp.array(1e-3, dtype="float32")),
    ]

    # Forward pass
    node_sizes = jnp.arange(10, 50, 5)
    # Save results before plotting
    results = {
        "forward_pass_times": {},
        "backward_pass_times": {},
        "node_sizes": node_sizes.tolist(),
    }

    for distance in distances:
        results["forward_pass_times"][distance.__class__.__name__] = {
            "cpu": benchmark(
                jax.devices("cpu")[0], calculate_ech, distance, node_sizes
            ),
            "gpu": benchmark(
                jax.devices("gpu")[0], calculate_ech, distance, node_sizes
            ),
        }
        results["backward_pass_times"][distance.__class__.__name__] = {
            "cpu": benchmark(
                jax.devices("cpu")[0], calculate_d_ech_dp, distance, node_sizes
            ),
            "gpu": benchmark(
                jax.devices("gpu")[0], calculate_d_ech_dp, distance, node_sizes
            ),
        }

    with open(path_results / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    if True:
        results = run_benchmark()
    else:
        with open(path_results / "benchmark_results.json") as f:
            results = json.load(f)

    node_sizes = jnp.array(results["node_sizes"])
    # Plot all distances on the same graph for forward pass
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for res_key, res in list(results.items())[:2]:
        fig, ax = plt.subplots()
        for i, (distance_name, times) in enumerate(res.items()):
            color = colors[i % len(colors)]
            ax.plot(
                node_sizes**2,
                times["cpu"],
                marker="o",
                linestyle="-",
                color=color,
                label=f"CPU {distance_name}",
            )
            ax.plot(
                node_sizes**2,
                times["gpu"],
                marker="v",
                linestyle="--",
                color=color,
                label=f"GPU {distance_name}",
            )
        ax.set_xlabel("Nb. of nodes")
        ax.set_ylabel("Computation time (s)")
        ax.set_yscale("log")
        ax.grid(True)
        ax.legend()
        ax.set_title(f'{res_key.replace("_", " ").title()}')
        fig.savefig(path_results / f"benchmark_{res_key}_all_distances.png")
