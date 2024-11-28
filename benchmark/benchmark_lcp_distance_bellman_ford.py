"""
Benchmark the computation time of the equivalent connected habitat (ECH) for different network sizes on CPU vs GPU.
Using bellman_ford for LCP calculations.

NOTE: Forward pass working, but gradient computation overloads
"""
import jax
import jax.numpy as jnp
from jaxscape.lcp_distance import LCPDistance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
import matplotlib.pyplot as plt
from pathlib import Path
import equinox
import time
from jaxscape.lcp_distance import _bellman_ford

path_results = Path("results/ResistanceDistance")
path_results.mkdir(parents=True, exist_ok=True)

distance = LCPDistance()

def calc_Kq(A, q, D, source):
    dist = _bellman_ford(A, source)
    proximity = jnp.exp(-dist / D)
    return proximity @ q

def calculate_ech(habitat_permability, habitat_quality, activities, D, nb_active):
    grid = GridGraph(activities=activities, 
                     vertex_weights=habitat_permability,
                     nb_active=nb_active)
    q = habitat_quality.ravel()
    A = grid.get_adjacency_matrix()
    Kq = jax.vmap(calc_Kq, in_axes=(None, None, None, 0))(A, q, D, jnp.arange(nb_active))
    qKq = q @ Kq
    return qKq

calculate_d_ech_dp = equinox.filter_jit(equinox.filter_grad(calculate_ech)) # sensitivity to permeability
node_sizes = jnp.arange(20, 100, 20)
D = jnp.array(1.0, dtype="float32")

def benchmark(device):
    times = []
    for size in node_sizes:
        print(f"Calculating ECH for size {size} on {device}")
        habitat_permability = jnp.ones((size, size), dtype="float32")
        activities = habitat_permability > 0
        habitat_quality = jnp.ones((size, size), dtype="float32") * 0
        habitat_quality = habitat_quality.at[1, 1].set(1.)
        habitat_quality = habitat_quality.at[size-2, size-2].set(1.)
        nb_active = int(activities.sum())
        
        # Warm-up to avoid measuring compile time
        calculate_d_ech_dp(jax.device_put(habitat_permability, device), 
                           jax.device_put(habitat_quality), 
                           jax.device_put(activities), 
                           1.0, 
                           nb_active).block_until_ready()
        
        start_time = time.time()
        calculate_d_ech_dp(jax.device_put(habitat_permability, device), 
                           jax.device_put(habitat_quality), 
                           jax.device_put(activities), 
                           1.0, 
                           nb_active).block_until_ready()
        end_time = time.time()
        
        times.append(end_time - start_time)
    return times

def plot_benchmark(cpu_times, gpu_times):
    fig, ax = plt.subplots()
    ax.plot(node_sizes**2, cpu_times, marker='o', label='CPU')
    ax.plot(node_sizes**2, gpu_times, marker='o', label='GPU')
    ax.set_xlabel('Nb. of nodes.')
    ax.set_ylabel('Computation time (s)')
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.grid(True)
    ax.legend()
    return fig, ax


# Forward pass
gpu_times = benchmark(jax.devices("gpu")[0], calculate_ech)
cpu_times = benchmark(jax.devices("cpu")[0], calculate_ech)
fig, ax = plot_benchmark(cpu_times, gpu_times)
fig.savefig(path_results / "benchmark_lcp_distance_bellman_ford_forward_pass.png")


# Backward pass
gpu_times = benchmark(jax.devices("gpu")[0], calculate_d_ech_dp)
cpu_times = benchmark(jax.devices("cpu")[0], calculate_d_ech_dp)
fig, ax = plot_benchmark(cpu_times, gpu_times)
fig.savefig(path_results / "benchmark_lcp_distance_bellman_ford_backward_pass.png")
