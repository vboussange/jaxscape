# Benchmark

This page compares JAXScape with the software that ecologists use for connectivity analysis on three tasks, distance calculation, connectivity analysis and sensitivity analysis, and compares the linear solvers of JAXScape. The results were obtained with JAXScape 0.0.16; the page names its solvers as the current API does. [`records.csv`](records.csv) holds the record of every cell of the run, and [Setup](#setup) states the hardware, the timing protocol and the settings.

Each comparison carries a tier of comparability. Tier 1 computes the same quantity as JAXScape, and the largest relative error against a float64 reference measures its accuracy; only at Tier 1 do two runtimes measure the same computation. Tier 2 computes the same quantity on another graph or with another kernel, and Tier 3 a related quantity that answers the same question; both report Spearman's rank correlation ρ between the map of the tool and that of JAXScape.

In the figures of the three tasks, a line gives the warm runtime of a tool and distance against the number of raster cells, at each raster that of its fastest setting that succeeded: solid with filled markers on the GPU, dashed with hollow markers on the CPU, and a star on Maryland. The table under each figure lists every setting, and the status and recorded reason of each cell that did not succeed.

## Distance calculation

The effective distance $d_{ij}$ between two habitat sites accounts for the landscape between them, and replaces Euclidean distance in dispersal kernels and in models of isolation by distance. Each tool computes the distances among 32 sites drawn at random from the habitat pixels, across the whole raster: 496 pairs. Every tool builds the same graph: rook contiguity, with the mean resistance of two pixels as the resistance of their edge. Free energy takes $\theta = 0.01$. The error of a cell is the largest relative error over the pairs against a float64 reference: a sparse direct solve with `CholmodSolver` for resistance and free energy, and Dijkstra's algorithm in SciPy for least-cost paths.

![Distance calculation: warm runtime against raster cells](distance.png)

??? note "Warm runtime (s) · largest relative error against the float64 reference"

    Each cell gives the warm runtime and the largest relative error over the pairs of sites, `missing` without a reference.

    | Tool | Distance | Device | dtype | Solver | Tier | 256² | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | JAXScape | resistance | GPU | float32 | GridMultigridCGSolver | 1 | 0.0249 s · 6.8e-05 | 0.351 s · 3.7e-04 | 5.48 s · 3.2e-03 | oom (1) |
    | JAXScape | resistance | CPU | float32 | GridMultigridCGSolver | 1 | 0.437 s · 7.5e-05 | 5.1 s · 3.7e-04 | 87.4 s · 1.2e-03 | 308 s · missing |
    | JAXScape | least-cost path | GPU | float32 |  | 1 | 0.209 s · 5.9e-07 | 10.4 s · 1.5e-06 | oom (2) | skipped (3) |
    | JAXScape | least-cost path | CPU | float32 |  | 1 | 13.6 s · 5.9e-07 | 506 s · 1.5e-06 | timeout (4) | skipped (5) |
    | JAXScape | free energy | GPU | float64 | CuDSSSolver | 1 | 0.304 s · 1.6e-15 | 6.1 s · 1.3e-15 |  |  |
    | JAXScape | free energy | CPU | float64 | CholmodSolver | 1 | 2.42 s · 0.0e+00 | 38.7 s · 0.0e+00 |  |  |
    | Circuitscape.jl | resistance | CPU | float64 | cg+amg | 1 | 0.401 s · 6.2e-07 | 21.5 s · 1.3e-06 | 622 s · 5.6e-05 | 2.21e+03 s · missing |
    | Circuitscape.jl | resistance | CPU | float64 | cholmod | 1 | 0.271 s · 8.7e-11 | 5.04 s · 1.6e-09 | 130 s · 1.9e-08 | 646 s · missing |
    | ConScape.jl | free energy | CPU | float64 |  | 1 | 0.644 s · 1.3e-15 | 13.3 s · 9.0e-16 |  |  |
    | gdistance | resistance | CPU | float64 |  | 1 | 1.54 s · 7.1e-13 | 40.9 s · 1.5e-10 | 1.7e+03 s · 2.5e-09 | failed (6) |
    | gdistance | least-cost path | CPU | float64 |  | 1 | 1.42 s · 1.4e-15 | 28.1 s · 4.5e-15 | 522 s · 7.6e-15 | 2.48e+03 s · 1.1e-14 |
    | skimage.graph.MCP | least-cost path | CPU | float64 |  | 1 | 0.43 s · 3.3e-16 | 12.5 s · 3.4e-16 | 247 s · 3.1e-16 | 1.19e+03 s · 3.6e-16 |

    1. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 44.98GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error=''] [executable_name='jit_nodes_to_nodes_distance']`
    2. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 14.00GiB with allocator GPU_0_bfc on device 0. [executable_name='jit_nodes_to_nodes_distance'] [tf-allocator-allocation-error='']`
    3. `after oom at synthetic-4096`
    4. `a call exceeded 3600 s`
    5. `after timeout at synthetic-4096`
    6. `Error in h(simpleError(msg, call)) :`

## Connectivity analysis

Connectivity measures how much habitat each pixel reaches through the landscape: Hanski's connectivity $S_i = \sum_j K(d_{ij})\, q_j$, with the habitat quality $q_j$ and the proximity $K$, and its landscape total $C = \sum_i q_i S_i$, the quantity behind the equivalent connected area. In each call, JAXScape computes the map of $\partial C / \partial q_i$, which equals $2 S_i$ for symmetric distances, with `SensitivityAnalysis`, and the total $C$ with `ConnectivityAnalysis`. Both are moving-window analyses, on windows of `dependency_range` 100 pixels with `coarsening_factor` 0.2, which gives blocks of 21 pixels, with the proximity $e^{-r/25}$, $r$ in equivalent pixels; they build their graphs with rook contiguity, with the mean permeability of two pixels as the conductance of their edge. Resistance takes `SpielmanApproximation(epsilon=0.15)` with `GridMultigridCGSolver(rtol=3e-4)`, the setting of the [connectivity example](../examples/connectivity_analysis/connectivity_analysis.ipynb), on tiles of up to 2,601 block centres. The map of each other tool is compared with that of JAXScape's GPU cell under the distance of its row, over the pixels where both are defined; ConScape.jl, under free energy, is compared with JAXScape under least-cost path. Graphab gives the square of its equivalent connectivity, $\sum_I \sum_J a_I a_J e^{-d_{IJ}/25}$ over its habitat patches, as its total $C$.

![Connectivity analysis: warm runtime against raster cells](connectivity.png)

??? note "Warm runtime (s) · Spearman ρ with JAXScape's map · landscape total C"

    Each cell gives the warm runtime, Spearman's ρ between the map of the tool and that of JAXScape, `missing` without JAXScape's map or where ρ is undefined, and the landscape total $C$ of a tool that computes one.

    | Tool | Distance | Device | dtype | Solver | Tier | 256² | Maryland | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | JAXScape | resistance | GPU | float32 | GridMultigridCGSolver(rtol=3e-4) |  | 0.506 s · C = 1.69e+06 | 3.3 s · C = 5.83e+07 | 3.71 s · C = 3.5e+07 | 45.7 s · C = 6.67e+08 | 176 s · C = 2.73e+09 |
    | JAXScape | resistance | CPU | float32 | GridMultigridCGSolver(rtol=3e-4) |  | 36.6 s · C = 1.69e+06 | 254 s · C = 5.83e+07 | 295 s · C = 3.5e+07 | timeout (1) | skipped (2) |
    | JAXScape | least-cost path | GPU | float32 |  |  | 0.179 s · C = 7.89e+06 | 1.93 s · C = 1.03e+08 | 2.11 s · C = 1.33e+08 | 35.5 s · C = 2.3e+09 | 141 s · C = 9.25e+09 |
    | JAXScape | least-cost path | CPU | float32 |  |  | 9.18 s · C = 7.89e+06 | 67.3 s · C = 1.03e+08 | 119 s · C = 1.33e+08 | 1.79e+03 s · C = 2.3e+09 | timeout (1) |
    | ConScape.jl | free energy | CPU | float64 |  | 3 | 1.07 s · ρ = 0.93 | 52.5 s · ρ = 0.81 | 199 s · ρ = 0.91 |  |  |
    | eco-connectivity | least-cost path | CPU | float32 |  | 2 | 0.573 s · ρ = 0.98 | 4.83 s · ρ = 0.81 | 10.8 s · ρ = 0.97 | 180 s · ρ = 0.97 | 727 s · ρ = 0.97 |
    | Graphab | least-cost path | CPU | float64 |  | 3 | 3.63 s · ρ = -0.06 · C = 1.89e+08 | 6.65 s · ρ = 0.80 · C = 5.11e+08 | 6.12 s · ρ = 0.32 · C = 5.7e+09 | 25.5 s · ρ = 0.38 · C = 1.31e+11 | 80.1 s · ρ = 0.33 · C = 5.76e+11 |

    1. `a call exceeded 3600 s`
    2. `after timeout at synthetic-4096`

## Sensitivity analysis

The sensitivity $\partial C / \partial p_k$ of connectivity to the permeability $p_k$ of each pixel locates corridors and pinch points: the pixels whose loss of permeability reduces connectivity the most. JAXScape computes it with `SensitivityAnalysis`, with the windows and settings of the connectivity analysis; the other tools map a flow, such as current, betweenness or visits. JAXScape's normalized sensitivity (K = −R) is the map of the section [Comparison with Omniscape's normalized current](../examples/connectivity_analysis/connectivity_analysis.ipynb#comparison-with-omniscapes-normalized-current) of the connectivity example: under resistance with the proximity $K(R) = -R$, which weighs every pair of habitat pixels equally, $p_k^2\, \partial C / \partial p_k = -\partial C / \partial r_k$ divided by $\partial C / \partial p_k$ on a landscape of permeability 1 with the same quality, and 0 where the latter is not positive; each call computes both sensitivities. Omniscape.jl computes its normalized current with the configuration of its Maryland example on the same windows, on a graph of eight neighbours, and is compared with the normalized sensitivity (K = −R). ConScape.jl is compared with least-cost path, and each other tool with the distance of its row.

![Sensitivity analysis: warm runtime against raster cells](sensitivity.png)

??? note "Warm runtime (s) · Spearman ρ with JAXScape's map"

    Each cell gives the warm runtime and Spearman's ρ between the map of the tool and that of JAXScape, `missing` without JAXScape's map or where ρ is undefined.

    | Tool | Distance | Device | dtype | Solver | Tier | 256² | Maryland | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | JAXScape | resistance | GPU | float32 | GridMultigridCGSolver(rtol=3e-4) |  | 0.544 s | 5.33 s | 4.62 s | 67.3 s | 264 s |
    | JAXScape | resistance | CPU | float32 | GridMultigridCGSolver(rtol=3e-4) |  | 42 s | 353 s | 319 s | timeout (1) | skipped (2) |
    | JAXScape | least-cost path | GPU | float32 |  |  | 0.154 s | 1.82 s | 1.99 s | 32.1 s | 126 s |
    | JAXScape | least-cost path | CPU | float32 |  |  | 7.36 s | 67.9 s | 102 s | 1.51e+03 s | timeout (1) |
    | JAXScape | normalized sensitivity (K = −R) | GPU | float32 | GridMultigridCGSolver(rtol=3e-4) |  | 1.06 s | 9.15 s | 9.16 s | 133 s | 532 s |
    | JAXScape | normalized sensitivity (K = −R) | CPU | float32 | GridMultigridCGSolver(rtol=3e-4) |  | 78.9 s | 631 s | 629 s | timeout (1) | skipped (2) |
    | ConScape.jl | free energy | CPU | float64 |  | 3 | 1.75 s · ρ = 0.84 | 102 s · ρ missing | 414 s · ρ = 0.78 |  |  |
    | EcoScape | least-cost path | GPU | float32 |  | 3 | 7.78 s · ρ = 0.76 | 13.7 s · ρ = 0.93 | 42.7 s · ρ = 0.75 | 439 s · ρ = 0.76 | 1.61e+03 s · ρ = 0.75 |
    | samc | resistance | CPU | float64 | direct | 3 | 2.5 s · ρ = 0.74 | 11.9 s · ρ = 0.07 | 29.9 s · ρ = 0.59 | 1.01e+03 s · ρ = 0.55 | failed (3) |
    | Omniscape.jl | resistance | CPU | float64 | cg+amg | 3 | 2.94 s · ρ = 0.97 | 51.6 s · ρ = 0.80 | 56.8 s · ρ = 0.93 | 787 s · ρ = 0.93 | 2.74e+03 s · ρ = 0.94 |
    | Omniscape.jl | resistance | CPU | float64 | cholmod | 3 | 5.16 s · ρ = 0.97 | 76 s · ρ = 0.80 | 117 s · ρ = 0.93 | 1.72e+03 s · ρ = 0.93 | timeout (1) |

    1. `a call exceeded 3600 s`
    2. `after timeout at synthetic-4096`
    3. `Error in (function (cond)  :`

![Maryland: JAXScape's normalized sensitivity (K = −R) and Omniscape.jl's normalized current](maryland.png)

On Maryland, JAXScape's normalized sensitivity (K = −R) (GPU, 9.15 s) and Omniscape.jl's normalized current (CPU, `cg+amg`, 51.6 s), each coloured by the rank of its pixels, with barriers in gray: Spearman's ρ = 0.80.

## Solvers

JAXScape solves the linear systems of resistance and free-energy distances with one of several solvers, in float32 or float64. Each runs here on the GPU in float32 and float64: in the distance calculation on every raster of its distance, and in the connectivity and sensitivity analyses under exact resistance at 256², and under `SpielmanApproximation(epsilon=0.15)` with `GridMultigridCGSolver(rtol=3e-4)` and with `CuDSSSolver` on 256², 1024², 4096², 8192² and Maryland. `dense` is the dense solve, `solver=None`. The error of a cell is, in the distance calculation, the largest relative error against its float64 reference, and in the windowed analyses the relative L2 error of the map $m$, $\lVert m - m_{\mathrm{ref}} \rVert_2 / \lVert m_{\mathrm{ref}} \rVert_2$, against the map of exact resistance in float64 with `CuDSSSolver` on the same tiles, measured at 256² alone. A call stops after 15 min, and under `SpielmanApproximation(epsilon=0.15)` after 1 h. These combinations are not run: free energy in float32; free energy with `GridMultigridCGSolver`, `AMJaxCGSolver` or `PyAMGCGSolver`; `AMJaxCGSolver` in float32.

![Solvers: warm runtime against raster cells](solvers.png)

Warm runtime of each solver against the raster cells: solid with filled markers in float32, dashed with hollow markers in float64, and a star on Maryland.

![Solvers: warm runtime against error](solver-errors.png)

Warm runtime of each cell with a measured error against its error: the colour gives the solver, a filled marker float32 and a hollow one float64, and the shape the raster. Errors below $10^{-16}$ are drawn at $10^{-16}$.

??? note "Warm runtime (s) · error against the float64 reference"

    Each cell gives the warm runtime, the error where it is measured, `missing` without a reference, and, in the connectivity analysis, the landscape total $C$.

    | Task | Distance | dtype | Solver | 256² | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | distance | resistance | float32 | GridMultigridCGSolver | 0.0246 s · 6.8e-05 | 0.352 s · 3.7e-04 | 5.46 s · 3.2e-03 | oom (1) |
    | distance | resistance | float32 | CholmodSolver | 1.16 s · 1.0e-05 | 25.3 s · 1.8e-04 | 844 s · 2.6e-04 | oom (2) |
    | distance | resistance | float32 | CuDSSSolver | 0.292 s · 1.2e-04 | 5.81 s · 3.9e-03 | failed (3) | skipped (4) |
    | distance | resistance | float32 | AMJaxCGSolver | failed (5) | skipped (6) | skipped (6) | skipped (6) |
    | distance | resistance | float32 | PyAMGCGSolver | 2.73 s · 4.4e-05 | 59 s · 2.4e-04 | timeout (7) | skipped (8) |
    | distance | resistance | float32 | dense | oom (9) | skipped (10) | skipped (10) | skipped (10) |
    | distance | resistance | float64 | GridMultigridCGSolver | 0.051 s · 1.1e-07 | 0.465 s · 3.0e-07 | oom (11) | skipped (12) |
    | distance | resistance | float64 | CholmodSolver | 1.18 s · 0.0e+00 | 25.3 s · 0.0e+00 | oom (13) | skipped (12) |
    | distance | resistance | float64 | CuDSSSolver | 0.297 s · 2.6e-13 | 5.89 s · 2.9e-12 | oom (14) | skipped (12) |
    | distance | resistance | float64 | AMJaxCGSolver | 2.04 s · 5.5e-04 | 5.18 s · 5.4e-04 | oom (15) | skipped (12) |
    | distance | resistance | float64 | PyAMGCGSolver | 2.94 s · 1.5e-07 | 66.4 s · 2.6e-07 | timeout (7) | skipped (8) |
    | distance | resistance | float64 | dense | oom (16) | skipped (10) | skipped (10) | skipped (10) |
    | distance | free energy | float64 | GridMultigridCGSolver | failed (17) | skipped (6) |  |  |
    | distance | free energy | float64 | CholmodSolver | 3.93 s · 4.4e-16 | 47.5 s · 2.5e-16 |  |  |
    | distance | free energy | float64 | CuDSSSolver | 0.305 s · 1.8e-15 | 6.07 s · 1.3e-15 |  |  |
    | distance | free energy | float64 | dense | oom (18) | skipped (10) |  |  |
    | connectivity | resistance | float32 | GridMultigridCGSolver | 102 s · 1.1e-03 · C = 1.72e+06 |  |  |  |
    | connectivity | resistance | float32 | CholmodSolver | timeout (7) |  |  |  |
    | connectivity | resistance | float32 | CuDSSSolver | 108 s · 1.6e-03 · C = 1.72e+06 |  |  |  |
    | connectivity | resistance | float32 | PyAMGCGSolver | timeout (7) |  |  |  |
    | connectivity | resistance | float32 | dense | failed (19) |  |  |  |
    | connectivity | resistance | float32 | SpielmanApproximation(epsilon=0.15), GridMultigridCGSolver(rtol=3e-4) | 0.508 s · 6.5e-02 · C = 1.69e+06 |  |  |  |
    | connectivity | resistance | float64 | GridMultigridCGSolver | 279 s · 5.1e-07 · C = 1.72e+06 |  |  |  |
    | connectivity | resistance | float64 | CholmodSolver | timeout (7) |  |  |  |
    | connectivity | resistance | float64 | CuDSSSolver | 170 s · 8.8e-15 · C = 1.72e+06 |  |  |  |
    | connectivity | resistance | float64 | AMJaxCGSolver | oom (20) |  |  |  |
    | connectivity | resistance | float64 | PyAMGCGSolver | timeout (7) |  |  |  |
    | connectivity | resistance | float64 | dense | failed (21) |  |  |  |
    | sensitivity | resistance | float32 | GridMultigridCGSolver | 105 s · 1.3e-03 |  |  |  |
    | sensitivity | resistance | float32 | CholmodSolver | timeout (7) |  |  |  |
    | sensitivity | resistance | float32 | CuDSSSolver | 134 s · 2.3e-03 |  |  |  |
    | sensitivity | resistance | float32 | PyAMGCGSolver | timeout (7) |  |  |  |
    | sensitivity | resistance | float32 | dense | failed (19) |  |  |  |
    | sensitivity | resistance | float32 | SpielmanApproximation(epsilon=0.15), GridMultigridCGSolver(rtol=3e-4) | 0.542 s · 7.7e-02 |  |  |  |
    | sensitivity | resistance | float64 | GridMultigridCGSolver | 346 s · 3.2e-07 |  |  |  |
    | sensitivity | resistance | float64 | CholmodSolver | timeout (7) |  |  |  |
    | sensitivity | resistance | float64 | CuDSSSolver | 251 s · 1.5e-13 |  |  |  |
    | sensitivity | resistance | float64 | AMJaxCGSolver | oom (22) |  |  |  |
    | sensitivity | resistance | float64 | PyAMGCGSolver | timeout (7) |  |  |  |
    | sensitivity | resistance | float64 | dense | failed (21) |  |  |  |

    1. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 44.98GiB with allocator GPU_0_bfc on device 0. [executable_name='jit_nodes_to_nodes_distance'] [tf-allocator-allocation-error='']`
    2. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.50GiB with allocator GPU_0_bfc on device 0. [executable_name='jit_less_equal'] [tf-allocator-allocation-error='']`
    3. `jax.errors.JaxRuntimeError: INTERNAL: spineax pbatch token: cuDSS call failed (status 2): cudssExecute analysis [executable_name='jit__initial_state']`
    4. `after failed at synthetic-4096`
    5. `equinox.EquinoxRuntimeError: A linear solver returned non-finite (NaN or inf) output. This usually means that an`
    6. `after failed at synthetic-256`
    7. `a call exceeded 900 s`
    8. `after timeout at synthetic-4096`
    9. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 16.02GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error='']`
    10. `after oom at synthetic-256`
    11. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 22.50GiB with allocator GPU_0_bfc on device 0. [executable_name='jit_nodes_to_nodes_distance'] [tf-allocator-allocation-error='']`
    12. `after oom at synthetic-4096`
    13. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 9.88GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error=''] [executable_name='jit_nodes_to_nodes_distance']`
    14. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 13.62GiB with allocator GPU_0_bfc on device 0. [executable_name='jit_nodes_to_nodes_distance'] [tf-allocator-allocation-error='']`
    15. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Failed to get configs for: 8 out of 83 instructions. See logs for all failures. Example failure:`
    16. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 32.02GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error='']`
    17. `equinox.EquinoxRuntimeError: FreeEnergyDistance cannot resolve the requested logarithms accurately; choose a different theta, float64 precision, or tighter solver tolerances.`
    18. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 64.01GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error=''] [executable_name='jit_nodes_to_nodes_distance']`
    19. `jax.errors.JaxRuntimeError: NOT_FOUND: All configs failed during profiling or were excluded from selection.`
    20. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 16.17GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error=''] [executable_name='jit__batch']`
    21. `jax.errors.JaxRuntimeError: NOT_FOUND: Failed to get configs for: 2 out of 23 instructions. See logs for all failures. Example failure:`
    22. `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 17.40GiB with allocator GPU_0_bfc on device 0. [tf-allocator-allocation-error=''] [executable_name='jit__batch']`

## Setup

**Hardware.** The run takes one NVIDIA GeForce RTX 3090 with 24576 MiB, driver 580.178.04 and CUDA 13.0, and 16 cores of an AMD EPYC 7742 64-Core Processor, under Ubuntu 22.04.5 LTS. Each cell runs in its own process, pinned to these cores, with 16 threads for OpenMP, BLAS, Julia and R; XLA sizes its thread pools to the same cores.

**Timing.** A warm-up call comes first and compiles the code of tools that compile. The warm runtime is the median of the timed calls that follow it, 5 per cell, or the time of a single call when the warm-up takes more than 5 min. Every call redoes the work that depends on the landscape: graph assembly, factorization, multigrid hierarchy and solver state. A call stops after 1 h (`timeout`). Within a setting, synthetic rasters run from the smallest up, and after a failure, a timeout or running out of memory (`oom`) on one, the larger ones are `skipped`; Maryland always runs.

**Landscapes.** Each synthetic raster draws a Gaussian random field with a Gaussian covariance of correlation length 25 pixels, on a periodic grid with a fixed seed, and normalizes its ranks to [0, 1]. Habitat quality equals the field where it is at least 0.7, and 0 elsewhere: 30 % of the pixels are habitat. Permeability is $10^{-2(1 - f)}$ for the field $f$, from 0.01 to 1, and resistance is its inverse. A larger raster covers a larger area at the same grain. The connectivity analysis and sensitivity analysis also run on the Maryland case of Omniscape's example: the 2016 land cover of Frederick, Maryland, from the National Land Cover Database, 694 × 725 pixels of 30 m. Each class takes the resistance of Omniscape's reclassification table; medium- and high-intensity development are barriers, and forest, the classes of resistance 1, is habitat of quality 1.

**Versions.** Python 3.13.5, JAX 0.12.0.1, jaxlib 0.12.0.1, PyTorch 2.15.0, Julia 1.13.3, R 4.1.2, Java 21.0.13. The table gives the version of each tool, its tasks, and how its adapter maps the shared parameters onto it.

??? note "Tools: version, tasks and parameter mapping"

    | Tool | Version | Tasks | Mapping |
    | --- | --- | --- | --- |
    | JAXScape | 0.0.16 | distance, connectivity, sensitivity | JAXScape builds a `GridGraph` of the permeability raster with rook contiguity, the conductance of an edge being the harmonic mean of the permeabilities of its two pixels (`landscape.conductance`), so that its resistance is their mean resistance. Least-cost and free-energy distances take 1 / conductance as the edge cost, and the free energy takes the shared `theta`. The distance calculation is a nodes-to-nodes query among the sites. The connectivity analysis runs `SensitivityAnalysis(var="quality")` for the map and `ConnectivityAnalysis` for the landscape total, sum_i q_i S_i, and the sensitivity analysis runs `SensitivityAnalysis(var="permeability")`, with the shared `dependency_range` and `coarsening_factor`, the default proximity and the `batch_size` of the settings; these windowed analyses build their graphs with JAXScape's own edge rule, the mean permeability of the two pixels. A `proximity` setting, a function of the distance in Python such as `lambda d: -d`, replaces the default proximity. With `normalized`, the map of the sensitivity analysis is the normalized sensitivity to resistance of the connectivity example: p^2 dC/dp divided by dC/dp on a flat landscape of permeability 1 and the same quality, and 0 where the latter is not positive, both computed in each call. The distance runs with the solver, the resistance `method` and the dtype of its settings, such as `GridMultigridSolver(rtol=3e-4)` (`dense` is the dense solve, `solver=None`), and every call builds the graphs and the solver states. |
    | Circuitscape.jl | 6.1.0 | distance | Circuitscape.jl computes the pairwise resistances among the sites of the distance calculation, in its `pairwise` scenario, with the permeability raster as the conductance of each cell. `connect_four_neighbors_only` gives rook contiguity, and `connect_using_avg_resistances` makes the resistance of an edge the mean resistance of its two cells, as `landscape.conductance` does. The solver is `cg+amg` or `cholmod`, in double precision, with `parallelize` on the Julia threads of the cell. Circuitscape.jl has no public in-memory API for this scenario: each call builds its `RasterData` in memory and runs the internals that `compute` runs after reading files (`build_problem`, `single_ground_all_pairs`), and writes only its table of resistances, to /dev/shm. Every call rebuilds the graph, the Laplacian and the AMG hierarchy or Cholesky factor. |
    | ConScape.jl | 0.3.0 | distance, connectivity, sensitivity | ConScape.jl builds its graph with rook contiguity (`N4`), the affinity of an edge being the harmonic mean of the permeabilities of its two pixels (`AverageWeight`), as `landscape.conductance` gives, and the cost of an edge 1 / affinity (`Inv`). Its free energy takes the shared `theta`. The distance calculation computes the free-energy distances among the sites, the targets of a `GridRSP`. ConScape's free-energy distance is directed, and the adapter returns the mean of the two directions between two sites, as JAXScape's `FreeEnergyDistance` does. The connectivity and sensitivity analyses run on the whole raster, without windows, with targets at landmarks every b pixels (`coarse_graining`), each holding the quality of its block, where b is the `block_size` of the windows, JAXScape's block size: 21. The connectivity analysis returns Hanski's S_i, the connected habitat of each pixel at source quality 1 (`connected_habitat`): the sum over landmarks of their quality times exp(-d / (dependency_range / 4)), exp(-d / 25), for the free-energy distance d to the landmark. The sensitivity analysis returns the quality-weighted betweenness (`betweenness_qweighted`). JAXScape has no free-energy map, and the configuration compares these maps with JAXScape's under least-cost distance (`compared_with = "lcp"`). Free energy runs from a multiple of resistance distance, as theta falls to 0, to least-cost distance, as theta grows. On the synthetic raster of 256 pixels a side, for every theta from 1e-5 to 1.5e-2, it ranks the pairs of sites of the distance calculation as least-cost distance does (Spearman's rho 0.98 to 0.99, against 0.92 to 0.95 with resistance distance), and at the shared theta, 0.01, ConScape's maps of connectivity and sensitivity correlate more with the maps of JAXScape's GPU cells under least-cost distance (rho 0.93 and 0.84) than under resistance distance (0.87 and 0.72). Every call rebuilds the affinities, the `Grid`, the landmarks and the `GridRSP`, with its sparse LU factorization, which runs on the BLAS threads of the cell. |
    | gdistance | 1.6.5 | distance | gdistance builds a transition layer of the resistance raster, 1 / permeability, with 4 neighbours (rook contiguity) and the transition function 1 / mean(x): the conductance between two pixels is the reciprocal of their mean resistance, `landscape.conductance`. Resistance distances are `commuteDistance` divided by the volume of the graph, the sum of its conductances in both directions, and least-cost distances are `costDistance`, whose edge cost is 1 / conductance. Pixels have side 1, so `geoCorrection` leaves the layer unchanged and is left out. Every call builds the transition layer, whose conductances depend on the resistance. |
    | skimage.graph.MCP | 0.26.0 | distance | skimage.graph.MCP_Geometric finds least-cost paths on the resistance raster, 1 / permeability, with rook contiguity (`fully_connected=False`). The cost of a step is the mean resistance of its two pixels, 1 / `landscape.conductance`, the edge cost of JAXScape. Each call builds the MCP object, and searches from each site until it reaches all the others. |
    | EcoScape | 0.2.5 | sensitivity | EcoScape (`ecoscape-connectivity`) simulates the repopulation of the habitat, the pixels of quality > 0, from seeds drawn at random in it, and maps its flow: the gradient of the repopulation with respect to the permeability of each pixel, times that permeability. Its edge rule is its own: at each step, individuals spread to the 8 neighbours of a pixel, survive a coin toss, and are multiplied by the permeability of their destination. The dispersal distance is the shared `dependency_range`, in pixels, which is also the border of its tiles and their padding, so that its map covers the raster. Quality enters only as habitat. The other settings are EcoScape's defaults: 400 simulations, 4 seeds per square of side 2 `dependency_range` + 1, tiles of 1000 pixels and no gap crossing, with the `batch_size` of the settings and a fixed random seed. EcoScape tiles a raster through GeoTIFFs only: every call writes the habitat and permeability to /dev/shm, runs on the GPU, and reads back the flow, which EcoScape writes with its connectivity. Its map is compared with JAXScape's under least-cost distances: repopulation follows the most permeable neighbour, the maximum over neighbours, as a least-cost path does. |
    | samc | 4.2.1 | sensitivity | samc builds a spatial absorbing Markov chain on the resistance raster, 1 / permeability, with 4 neighbours (rook contiguity) and the transition function 1 / mean(x): an individual moves to a neighbour in proportion to the harmonic mean of the permeabilities of the two pixels, `landscape.conductance`, and barriers, of permeability 0, are left out of the chain. The absorption of every pixel is (4 / `dependency_range`)^2 per step, so that an individual moves about `dependency_range` / 4 pixels before it is absorbed, the length over which JAXScape's default proximity decays. Individuals are released in proportion to quality, and the map is `visitation`: the expected number of visits to each pixel before absorption. The solver is samc's `method`, whose `direct` method factorizes the chain with a single thread. Every call builds the chain, on which samc caches its factorization. Its map is compared with JAXScape's under resistance distances: an absorbing random walk is the random walk behind resistance distance. |
    | eco-connectivity | 2.5.0 | connectivity | eco-connectivity computes the connectedness of each pixel i, with `option=1`: the sum over pixels j != i of exp(-d(i, j) / 25) q_j, divided by the same sum with quality 1 and every weight 1. Its numerator is Hanski's S_i; its denominator, the reach of an intact landscape, is constant away from the raster's edges. d is a least-cost distance over 8 neighbours, in which a step costs its length times the weight of the pixel it enters, here its resistance, 1 / permeability: the resistance raster is (r - 1) / (r_max - 1), and `max_cost` is r_max, the largest resistance. Quality is the condition raster. Barriers, of permeability 0, are masked, and the map is NaN there. Pixels are 1 km, so `lambdas=[25]` gives the kernel exp(-d/25) in pixels. The multi-resolution graph keeps the default `window_size`, `outer_window` and `window_mode`, and the default `levels` up to the first whose reach, `outer_window` times the level in pixels, covers `dependency_range`. `n_threads` is the number of cores of the cell. The tool reads GeoTIFFs: each call writes the two rasters to /dev/shm and computes the map from them. |
    | Graphab | 3.0.8 | connectivity | Graphab builds a graph of habitat patches, the clusters of habitat pixels (quality > 0) connected through 4 neighbours. Its links are least-cost paths between patches over 8 neighbours, in which a step costs its length times the mean resistance, 1 / permeability, of its two pixels (`extcost`), and barriers, of permeability 0, are no data. Links and graph keep the paths of cost up to `dependency_range` (`maxcost`, `threshold`, `topo=complete`), where the kernel falls to exp(-4). `d=25` and `p=exp(-1)` give the kernel exp(-d/25) on the cost of the shortest path in the graph, which adds the costs of crossing patches. The capacity a_I of a patch is its area in pixels. Each call computes the equivalent connectivity EC of the graph, whose square, the sum over patches I and J of a_I a_J exp(-d_IJ/25), is the landscape total, and the flux F_I = sum over patches J != I of a_J exp(-d_IJ/25) (`beta=1`). The map is F_I + a_I, the patch analogue of Hanski's S_i, which adds the patch's own term that F leaves out, on the pixels of each patch, and NaN elsewhere. `-proc` is the number of cores of the cell. Graphab reads and writes files: each call writes the land and resistance rasters to /dev/shm, creates a new project from them, and reads the patches and EC back. |
    | Omniscape.jl | 0.6.4 | sensitivity | Omniscape.jl computes the normalized current of the sensitivity analysis with the configuration of its Maryland example: `radius` is the shared `dependency_range`, and `block_size` the side of JAXScape's blocks, `dependency_range * coarsening_factor` rounded up to an odd integer, 21 pixels. Resistance is 1 / permeability, with the barriers, of permeability 0, as NoData, and the source strength of a pixel is its quality; on Maryland, these are the resistances of the example's reclassification table and the forest sources of its `source_from_resistance` with `r_cutoff` 1. The map is the cumulative current divided by the flow potential (`calc_normalized_current`, `calc_flow_potential`), compared with JAXScape's normalized sensitivity to resistance, `resistance, K = -R, normalized`, which the connectivity example relates to it: resistance is the distance of Omniscape's circuits, the proximity K(R) = -R weighs every pair of habitat pixels equally, as Omniscape injects the current of every source of a window, and the flat landscape normalizes it, as the flow potential normalizes the current. Omniscape keeps its other defaults: eight neighbours, the conductance of an edge being the mean permeability of its two pixels, divided by √2 on diagonals; artifact correction; double precision; the solver of the settings, with the windows solved in parallel on the Julia threads of the cell. Each call runs `run_omniscape` on arrays in memory, which redoes the sources, the targets and every window solve; it reads and writes no file, and only creates an empty project directory in /dev/shm. |
