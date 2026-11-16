# Benchmark

This page compares JAXScape with the software that ecologists use for connectivity analysis on three questions: the distance calculation, the connectivity analysis and the sensitivity analysis. For each tool, it reports the warm runtime and the agreement of the results with a reference. The last section compares the linear solvers of JAXScape. [`records.csv`](records.csv) holds the record of every cell of the run: one setting of a tool on one raster.

## Setup

### Hardware and software

- **GPU**: one NVIDIA GeForce RTX 3090 with 24576 MiB, driver 580.178.04, CUDA 13.0.
- **CPU**: 16 cores of an AMD EPYC 7742 64-Core Processor, under Ubuntu 22.04.5 LTS. Each cell runs in its own process, pinned to these cores, with 16 threads for OpenMP, BLAS, Julia and R; XLA sizes its thread pools to the same cores.
- **Languages and libraries**: Python 3.13.5, JAX 0.12.0.1, jaxlib 0.12.0.1, PyTorch 2.15.0, Julia 1.13.3, R 4.1.2, Java 21.0.13.

### Tools

| Tool | Version | Questions |
| --- | --- | --- |
| JAXScape | 0.0.20 | distance calculation, connectivity analysis, sensitivity analysis |
| Circuitscape.jl | 6.1.0 | distance calculation |
| ConScape.jl | 0.3.0 | distance calculation, connectivity analysis, sensitivity analysis |
| gdistance | 1.6.5 | distance calculation |
| skimage.graph.MCP | 0.26.0 | distance calculation |
| EcoScape | 0.2.5 | sensitivity analysis |
| samc | 4.2.1 | sensitivity analysis |
| eco-connectivity | 2.5.0 | connectivity analysis |
| Graphab | 3.0.8 | connectivity analysis |
| Omniscape.jl | 0.6.4 | sensitivity analysis |

??? note "Parameter mapping of each tool"

    | Tool | Mapping |
    | --- | --- |
    | JAXScape | JAXScape builds a `GridGraph` of the permeability raster with rook contiguity, the conductance of an edge being the harmonic mean of the permeabilities of its two pixels (`landscape.conductance`), so that its resistance is their mean resistance. The edge cost of least-cost path and free-energy distances is 1 / conductance, and the free-energy distance has the shared `theta`. The distance calculation is a nodes-to-nodes query among the sites. The connectivity analysis runs `SensitivityAnalysis(...).run("quality")` for its map, and the sensitivity analysis runs `SensitivityAnalysis(...).run("permeability")`, with the shared `dependency_range` and `coarsening_factor`, the default proximity and the `batch_size` of the settings; these windowed analyses build their graphs with JAXScape's own edge rule, the mean permeability of the two pixels. A `proximity` setting, a function of the distance in Python such as `lambda d: -d`, replaces the default proximity. With `normalized`, the map of the sensitivity analysis is the normalized sensitivity to resistance of the connectivity example: p^2 dC/dp divided by dC/dp on a landscape of permeability 1 and the same quality, and 0 where the latter is not positive, both computed in each call. The distance runs with the solver, the resistance `method` and the dtype of its settings, such as `GeometricMultigridCGSolver(rtol=3e-4)` (`dense` is the dense solve, `solver=None`), and every call builds the graphs and the solver states. |
    | Circuitscape.jl | Circuitscape.jl computes the pairwise resistances among the sites of the distance calculation, in its `pairwise` scenario, with the permeability raster as the conductance of each pixel. `connect_four_neighbors_only` gives rook contiguity, and `connect_using_avg_resistances` makes the resistance of an edge the mean resistance of its two pixels, as `landscape.conductance` does. The solver is `cg+amg` or `cholmod`, in double precision, with `parallelize` on the Julia threads of the cell. Circuitscape.jl has no public in-memory API for this scenario: each call builds its `RasterData` in memory and runs the internals that `compute` runs after reading files (`build_problem`, `single_ground_all_pairs`), and writes only its table of resistances, to /dev/shm. Every call rebuilds the graph, the Laplacian and the AMG hierarchy or Cholesky factor. |
    | ConScape.jl | ConScape.jl builds its graph with rook contiguity (`N4`), the affinity of an edge being the harmonic mean of the permeabilities of its two pixels (`AverageWeight`), as `landscape.conductance` gives, and the cost of an edge 1 / affinity (`Inv`). `Grid` keeps the largest strongly connected component of the graph, through `largest_subgraph`, by its default `prune`, as in ConScape's examples. Its free-energy distance has the shared `theta`. The distance calculation computes the free-energy distances among the sites, the targets of a `GridRSP`. ConScape's free-energy distance is directed, and the adapter returns the mean of the two directions between two sites, as JAXScape's `FreeEnergyDistance` does. The connectivity and sensitivity analyses run on the whole raster, without windows, with targets at landmarks every b pixels (`coarse_graining`), each holding the quality of its block, where b is the `block_size` of the windows, JAXScape's block size: 21. The connectivity analysis returns Hanski's S_i, the connected habitat of each pixel at source quality 1 (`connected_habitat`): the sum over landmarks of their quality times exp(-d / (dependency_range / 4)), exp(-d / 25), for the free-energy distance d to the landmark. The sensitivity analysis returns the quality-weighted betweenness (`betweenness_qweighted`). On Maryland, whose resistances reach 1000, the free-energy distance d is larger than 7.4e4 for 45 % of the pairs of a pixel of the component and a landmark: the entry of ConScape's matrix Z for such a pair, proportional to exp(-theta d), is below the smallest positive float64, and is 0. `betweenness_qweighted` replaces the reciprocal of each such entry by the largest float64, its products with the landmark qualities larger than 1 are infinite, and its map is NaN on every pixel. `connected_habitat` weighs such a pair by exp(-d / 25), which is 0, and its map is finite. JAXScape has no free-energy map, and the configuration compares these maps with JAXScape's under least-cost path distance (`compared_with = "lcp"`). The free-energy distance tends to a multiple of the resistance distance as theta tends to 0, and to the least-cost path distance as theta grows. On the synthetic raster of 256 pixels a side, for every theta from 1e-5 to 1.5e-2, the ranking of the pairs of sites of the distance calculation by free-energy distance agrees with their ranking by least-cost path distance (Spearman's ρ 0.98 to 0.99, against 0.92 to 0.95 with resistance distance), and at the shared theta, 0.01, ConScape's maps of connectivity and sensitivity correlate more with the maps of JAXScape's GPU cells under least-cost path distance (ρ 0.93 and 0.84) than under resistance distance (0.87 and 0.72). Every call rebuilds the affinities, the `Grid`, the landmarks and the `GridRSP`, with its sparse LU factorization, which runs on the BLAS threads of the cell. |
    | gdistance | gdistance builds a transition layer of the resistance raster, 1 / permeability, with 4 neighbours (rook contiguity) and the transition function 1 / mean(x): the conductance between two pixels is the reciprocal of their mean resistance, `landscape.conductance`. Resistance distances are `commuteDistance` divided by the volume of the graph, the sum of its conductances in both directions, and least-cost path distances are `costDistance`, whose edge cost is 1 / conductance. Pixels have side 1, so `geoCorrection` leaves the layer unchanged and is left out. Every call builds the transition layer, whose conductances depend on the resistance. |
    | skimage.graph.MCP | skimage.graph.MCP_Geometric finds least-cost paths on the resistance raster, 1 / permeability, with rook contiguity (`fully_connected=False`). The cost of a step is the mean resistance of its two pixels, 1 / `landscape.conductance`, the edge cost of JAXScape. Each call builds the MCP object, and searches from each site until it reaches all the others. |
    | EcoScape | EcoScape (`ecoscape-connectivity`) simulates the repopulation of the habitat, the pixels of quality > 0, from seeds drawn at random in it, and maps its flow: the gradient of the repopulation with respect to the permeability of each pixel, times that permeability. Its edge rule is its own: at each step, individuals spread to the 8 neighbours of a pixel, survive a coin toss, and are multiplied by the permeability of their destination. The dispersal distance is the shared `dependency_range`, in pixels, which is also the border of its tiles and their padding, so that its map covers the raster. Quality enters only as habitat. The other settings are EcoScape's defaults: 400 simulations, 4 seeds per square of side 2 `dependency_range` + 1, tiles of 1000 pixels and no gap crossing, with the `batch_size` of the settings and a fixed random seed. EcoScape tiles a raster through GeoTIFFs only: every call writes the habitat and permeability to /dev/shm, runs on the GPU, and reads back the flow, which EcoScape writes with its connectivity. Its map is compared with JAXScape's under least-cost path distances: repopulation follows the most permeable neighbour, the maximum over neighbours, as a least-cost path does. |
    | samc | samc builds a spatial absorbing Markov chain on the resistance raster, 1 / permeability, with 4 neighbours (rook contiguity) and the transition function 1 / mean(x): an individual moves to a neighbour in proportion to the harmonic mean of the permeabilities of the two pixels, `landscape.conductance`, and barriers, of permeability 0, are left out of the chain. The absorption of every pixel is (4 / `dependency_range`)^2 per step, so that an individual moves about `dependency_range` / 4 pixels before it is absorbed, the length over which JAXScape's default proximity decays. Individuals are released in proportion to quality, and the map is `visitation`: the expected number of visits to each pixel before absorption. The solver is samc's `method`, whose `direct` method factorizes the chain with a single thread. Every call builds the chain, on which samc caches its factorization. Its map is compared with JAXScape's under resistance distances: an absorbing random walk is the random walk behind resistance distance. |
    | eco-connectivity | eco-connectivity computes the connectedness of each pixel i, with `option=1`: the sum over pixels j != i of exp(-d(i, j) / 25) q_j, divided by the same sum with quality 1 and every weight 1. Its numerator is Hanski's S_i; its denominator, the reach of an intact landscape, is constant away from the raster's edges. d is a least-cost path distance over 8 neighbours, in which a step costs its length times the weight of the pixel it enters, here its resistance, 1 / permeability: the resistance raster is (r - 1) / (r_max - 1), and `max_cost` is r_max, the largest resistance. Quality is the condition raster. Barriers, of permeability 0, are masked, and the map is NaN there. Pixels are 1 km, so `lambdas=[25]` gives the proximity exp(-d/25) in pixels. The multi-resolution graph keeps the default `window_size`, `outer_window` and `window_mode`, and the default `levels` up to the first whose reach, `outer_window` times the level in pixels, covers `dependency_range`. `n_threads` is the number of cores of the cell. The tool reads GeoTIFFs: each call writes the two rasters to /dev/shm and computes the map from them. |
    | Graphab | Graphab builds a graph of habitat patches, the clusters of habitat pixels (quality > 0) connected through 4 neighbours. Its links are least-cost paths between patches over 8 neighbours, in which a step costs its length times the mean resistance, 1 / permeability, of its two pixels (`extcost`), and barriers, of permeability 0, are no data. Links and graph keep the paths of cost up to `dependency_range` (`maxcost`, `threshold`, `topo=complete`), where the proximity falls to exp(-4). `d=25` and `p=exp(-1)` give the proximity exp(-d/25) on the cost of the shortest path in the graph, which adds the costs of crossing patches. The capacity a_I of a patch is its area in pixels. Each call computes the flux F_I = sum over patches J != I of a_J exp(-d_IJ/25) (`beta=1`). The map is F_I + a_I, the patch analogue of Hanski's S_i, which adds the patch's own term that F leaves out, on the pixels of each patch, and NaN elsewhere. `-proc` is the number of cores of the cell. Graphab reads and writes files: each call writes the land and resistance rasters to /dev/shm, creates a new project from them, and reads the patches back. |
    | Omniscape.jl | Omniscape.jl computes the normalized current of the sensitivity analysis with the configuration of its Maryland example: `radius` is the shared `dependency_range`, and `block_size` the side of JAXScape's blocks, `dependency_range * coarsening_factor` rounded up to an odd integer, 21 pixels. Resistance is 1 / permeability, with the barriers, of permeability 0, as NoData, and the source strength of a pixel is its quality; on Maryland, these are the resistances of the example's reclassification table and the forest sources of its `source_from_resistance` with `r_cutoff` 1. The map is the cumulative current divided by the flow potential (`calc_normalized_current`, `calc_flow_potential`), compared with JAXScape's normalized sensitivity to resistance, `resistance, K = -R, normalized`, which the connectivity example relates to it: resistance is the distance of Omniscape's circuits, every pair of habitat pixels has the same weight under the proximity K(R) = -R, as Omniscape injects the current of every source of a window, and the sensitivity is divided by its value on a landscape of permeability 1, as the current is divided by the flow potential. Omniscape keeps its other defaults: eight neighbours, the conductance of an edge being the mean permeability of its two pixels, divided by √2 on diagonals; artifact correction; double precision; the solver of the settings, with the windows solved in parallel on the Julia threads of the cell. Each call runs `run_omniscape` on arrays in memory, which redoes the sources, the targets and every window solve; it reads and writes no file, and only creates an empty project directory in /dev/shm. |

### Landscapes

- **Synthetic rasters** of 256², 1024², 4096² and 8192² pixels. The benchmark draws a Gaussian random field $f$ with a Gaussian covariance of correlation length 25 pixels, on a periodic grid with a fixed seed, and normalizes its ranks to [0, 1]. The habitat quality is $f$ where $f \geq 0.7$, and 0 elsewhere: 30 % of the pixels are habitat. The permeability is $10^{-2(1 - f)}$, from 0.01 to 1, and the resistance is its inverse. A larger raster covers a larger area at the same grain.
- **Maryland**, for the connectivity analysis and sensitivity analysis: the land cover of Omniscape's example, the 2016 National Land Cover Database of Frederick, Maryland, 694 × 725 pixels of 30 m. The resistance of each land-cover class is that of Omniscape's reclassification table; medium- and high-intensity development are barriers, and forest, the classes of resistance 1, is habitat of quality 1.

### Timing

- **Warm runtime**: the median time of the 5 timed calls of a cell, which follow a warm-up call that compiles the code of the tools that compile; the time of a single timed call when the warm-up call takes more than 5 min, as in 1 of the 56 cells that succeeded on 256², 2 of the 25 on Maryland, 4 of the 48 on 1024², 12 of the 31 on 4096² and 13 of the 22 on 8192².
- **Cells run more than once**, each run in a new process, with the warm runtime of each run, of which the page gives the smallest and [`reruns.csv`](reruns.csv) holds the records of the others: Circuitscape.jl, CPU, float64, cholmod, distance calculation on 4096²: 141 s, 157 s and 296 s; Circuitscape.jl, CPU, float64, cg+amg, distance calculation on 8192²: 1890 s and 2630 s; Omniscape.jl, CPU, float64, cg+amg, sensitivity analysis on 8192²: 2170 s and 2230 s.
- **Work of a call**: every call redoes the work that depends on the landscape: graph assembly, factorization, multigrid hierarchy and solver state.
- **Speed-up of JAXScape** over a tool under a distance: the warm runtime of the tool divided by that of JAXScape on the GPU, each in its fastest setting on the largest synthetic raster that both completed, with JAXScape under the same distance in the distance calculation, and under the distance or label of the column "Compared with JAXScape under" in the connectivity and sensitivity analyses.
- **Time limit**: a call stops after 1 h, and in the section of the linear solvers of JAXScape after the time limit stated in its Method.
- **Status of a cell that did not complete**: `timeout` when a call exceeds its time limit, `oom` when a call runs out of memory, and `failed` on another error. A setting runs on the synthetic rasters from the smallest up, and is `skipped` on the rasters larger than one where it stopped; Maryland always runs.

### Comparability tiers

| Tier | The tool computes | Agreement measure |
| --- | --- | --- |
| 1 | the quantity of JAXScape, on the same graph | largest relative error with respect to a float64 reference |
| 2 | the quantity of JAXScape, on another graph or with another proximity | Spearman's ρ between the map of the tool and that of JAXScape |
| 3 | a related quantity that answers the same question | Spearman's ρ between the map of the tool and that of JAXScape |

Only at Tier 1 do the tool and JAXScape perform the same computation.

### Figures and tables

- **Figures of runtime**: each line passes through the warm runtimes of the fastest setting of a tool, distance and device that succeeded on each synthetic raster, solid on the GPU and dashed on the CPU; in the section of the linear solvers, a line is a solver and dtype, solid in float32 and dashed in float64. A star marks Maryland.
- **Tables**: an entry is the warm runtime and the agreement of a cell that succeeded, or the status of a cell that did not.

## Distance calculation

On the largest synthetic raster that JAXScape on the GPU and the other tool both completed, JAXScape is faster than every other tool, by a factor of 1.2 (skimage.graph.MCP, least-cost path distance) to 248 (gdistance, resistance distance). The largest relative error with respect to the float64 reference is at most 1.3e-03 for JAXScape, under resistance distance in float32, and at most 5.6e-05 for the other tools. On the GPU, JAXScape ran out of memory under least-cost path distance on 4096² and under resistance distance on 8192².

### Results

<img src="distance.png" alt="Distance calculation: warm runtime" width="610">

| Tool | Distance | Raster | Warm runtime of the tool | Warm runtime of JAXScape | Speed-up | Error of the tool | Error of JAXScape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Circuitscape.jl | resistance | 4096² | 141 s | 6.81 s | 21 | 1.9e-08 | 1.3e-03 |
| ConScape.jl | free energy | 1024² | 13.0 s | 5.66 s | 2.3 | 9.0e-16 | 1.3e-15 |
| gdistance | resistance | 4096² | 1690 s | 6.81 s | 248 | 2.6e-09 | 1.3e-03 |
| gdistance | least-cost path | 1024² | 27.7 s | 10.6 s | 2.6 | 4.5e-15 | 1.5e-06 |
| skimage.graph.MCP | least-cost path | 1024² | 12.4 s | 10.6 s | 1.2 | 3.4e-16 | 1.5e-06 |

??? note "Warm runtime and error of every setting on every raster"

    An entry without an error has no reference.

    | Tool | Distance | Setting | 256² | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- |
    | JAXScape | resistance | GPU, float32, GeometricMultigridCGSolver | 0.0386 s, error 6.5e-05 | 0.419 s, error 2.7e-04 | 6.81 s, error 1.3e-03 | oom |
    | JAXScape | resistance | CPU, float32, GeometricMultigridCGSolver | 0.396 s, error 6.8e-05 | 4.99 s, error 2.3e-04 | 88.1 s, error 6.7e-04 | 350 s |
    | JAXScape | least-cost path | GPU, float32 | 0.211 s, error 5.9e-07 | 10.6 s, error 1.5e-06 | oom | skipped |
    | JAXScape | least-cost path | CPU, float32 | 12.9 s, error 5.9e-07 | 505 s, error 1.5e-06 | timeout | skipped |
    | JAXScape | free energy | GPU, float64, CuDSSSolver | 0.281 s, error 1.4e-15 | 5.66 s, error 1.3e-15 |  |  |
    | JAXScape | free energy | CPU, float64, CholmodSolver | 1.65 s, error 0.0e+00 | 37.9 s, error 0.0e+00 |  |  |
    | Circuitscape.jl | resistance | CPU, float64, cg+amg | 0.383 s, error 6.2e-07 | 21.0 s, error 1.3e-06 | 602 s, error 5.6e-05 | 1890 s |
    | Circuitscape.jl | resistance | CPU, float64, cholmod | 0.276 s, error 8.7e-11 | 4.93 s, error 1.6e-09 | 141 s, error 1.9e-08 | 642 s |
    | ConScape.jl | free energy | CPU, float64 | 0.620 s, error 1.3e-15 | 13.0 s, error 9.0e-16 |  |  |
    | gdistance | resistance | CPU, float64 | 1.53 s, error 6.8e-13 | 39.2 s, error 1.5e-10 | 1690 s, error 2.6e-09 | oom |
    | gdistance | least-cost path | CPU, float64 | 1.41 s, error 1.4e-15 | 27.7 s, error 4.5e-15 | 520 s, error 7.6e-15 | 2620 s, error 1.1e-14 |
    | skimage.graph.MCP | least-cost path | CPU, float64 | 0.429 s, error 3.3e-16 | 12.4 s, error 3.4e-16 | 244 s, error 3.1e-16 | 1180 s, error 3.6e-16 |

### Settings that did not complete

| Tool | Distance | Setting | Raster | Status | Recorded reason |
| --- | --- | --- | --- | --- | --- |
| JAXScape | resistance | GPU, float32, GeometricMultigridCGSolver | 8192² | `oom` | allocation of 44.98 GiB failed |
| JAXScape | least-cost path | GPU, float32 | 4096² | `oom` | allocation of 14.00 GiB failed |
| JAXScape | least-cost path | CPU, float32 | 4096² | `timeout` | `a call exceeded 3600 s` |
| gdistance | resistance | CPU, float64 | 8192² | `oom` | `Error in h(simpleError(msg, call)) : error in evaluating the argument 'x' in selecting a method for function 'nrow': error in evaluating the argument 'x' in selecting a method for function 'as.matrix': LU factorization of .gCMatrix failed: out of memory or near-singular` |

### Method

- **Quantity**: the distance $d_{ij}$ on a graph between habitat sites $i$ and $j$: resistance distance, least-cost path distance, and free-energy distance with $\theta = 0.01$. Ecologists use it in place of Euclidean distance in dispersal kernels and in models of isolation by distance.
- **Sites**: 32 sites drawn at random from the habitat pixels of the raster, 496 pairs.
- **Graph**: every tool builds the same graph: rook contiguity, with the mean resistance of two pixels as the resistance of their edge.
- **Rasters**: free-energy distance on 256² and 1024² alone.
- **Error**: the largest relative error over the pairs of sites with respect to a float64 reference computed on the CPU: the linear systems of resistance and free-energy distances are solved with JAXScape's `CholmodSolver`, and least-cost path distances are computed with Dijkstra's algorithm in SciPy.
- **Tier**: every tool is at Tier 1.

## Connectivity analysis

On the largest synthetic raster that JAXScape on the GPU and the other tool both completed, JAXScape is faster than every other tool, by a factor of 1.1 (Graphab) to 180 (ConScape.jl). On those rasters, Spearman's ρ between the map of a tool and that of JAXScape ranges from 0.33 (Graphab) to 0.97 (eco-connectivity).

### Results

<img src="connectivity.png" alt="Connectivity analysis: warm runtime" width="610">

| Tool | Tier | Compared with JAXScape under | Raster | Warm runtime of the tool | Warm runtime of JAXScape | Speed-up | Spearman's ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConScape.jl | 3 | least-cost path | 1024² | 197 s | 1.10 s | 180 | 0.91 |
| eco-connectivity | 2 | least-cost path | 8192² | 725 s | 74.8 s | 9.7 | 0.97 |
| Graphab | 3 | least-cost path | 8192² | 80.6 s | 74.8 s | 1.1 | 0.33 |

??? note "Warm runtime and Spearman's ρ of every setting on every raster"

    The maps of JAXScape are the references, and an entry of another tool without ρ has no map of JAXScape to compare with, or ρ is undefined.

    | Tool | Distance | Setting | 256² | Maryland | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | JAXScape | resistance | GPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 0.277 s | 1.74 s | 1.97 s | 23.3 s | 89.6 s |
    | JAXScape | resistance | CPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 17.8 s | 130 s | 148 s | 2310 s | timeout |
    | JAXScape | least-cost path | GPU, float32 | 0.0931 s | 0.652 s | 1.10 s | 18.8 s | 74.8 s |
    | JAXScape | least-cost path | CPU, float32 | 4.59 s | 33.4 s | 59.3 s | 895 s | timeout |
    | ConScape.jl | free energy | CPU, float64 | 1.06 s, ρ = 0.93 | 51.6 s, ρ = 0.81 | 197 s, ρ = 0.91 |  |  |
    | eco-connectivity | least-cost path | CPU, float32 | 0.579 s, ρ = 0.98 | 4.84 s, ρ = 0.81 | 10.8 s, ρ = 0.97 | 180 s, ρ = 0.97 | 725 s, ρ = 0.97 |
    | Graphab | least-cost path | CPU, float64 | 3.65 s, ρ = -0.06 | 6.72 s, ρ = 0.80 | 6.16 s, ρ = 0.32 | 25.1 s, ρ = 0.38 | 80.6 s, ρ = 0.33 |

### Settings that did not complete

| Tool | Distance | Setting | Raster | Status | Recorded reason |
| --- | --- | --- | --- | --- | --- |
| JAXScape | resistance | CPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 8192² | `timeout` | `a call exceeded 3600 s` |
| JAXScape | least-cost path | CPU, float32 | 8192² | `timeout` | `a call exceeded 3600 s` |

### Method

- **Quantity**: Hanski's connectivity $S_i = \sum_j K(d_{ij})\, q_j$ of each pixel $i$, with the habitat quality $q_j$ of pixel $j$ and the proximity $K$, and its landscape total $C = \sum_i q_i S_i$, the square of the equivalent connected area.
- **JAXScape**: in each call, `SensitivityAnalysis` computes the map of $\partial C / \partial q_i$, equal to $2 S_i$ for a symmetric distance.
- **Windows**: windowed analyses with `dependency_range` 100 pixels and `coarsening_factor` 0.2, that is, blocks of 21 × 21 pixels, under the proximity $K(d) = e^{-d/25}$, with $d$ in equivalent pixels.
- **Graph**: rook contiguity, with the mean permeability of two pixels as the conductance of their edge. JAXScape's windowed analyses raise permeability below the permeability floor, a small fraction of the largest permeability, to the floor. On Maryland, the other tools treat the barriers, of permeability 0, as impassable, and JAXScape's analyses still cross the barrier pixels: the edge between a barrier pixel and a pixel of permeability $p$ has a conductance of about $p/2$.
- **Resistance distance**: `SpielmanApproximation(epsilon=0.15)` with `GeometricMultigridCGSolver(rtol=3e-4)`, the setting of the [connectivity example](../examples/connectivity_analysis/connectivity_analysis.ipynb), on tiles of up to 2,601 block centres.
- **Agreement**: Spearman's ρ between the map of a tool and the map of JAXScape on the GPU under the distance or label of the column "Compared with JAXScape under", over the pixels where both maps are defined.

## Sensitivity analysis

On the largest synthetic raster that JAXScape on the GPU and the other tool both completed, JAXScape is faster than every other tool, by a factor of 4.0 (Omniscape.jl) to 202 (ConScape.jl). On those rasters, Spearman's ρ between the map of a tool and that of JAXScape ranges from 0.55 (samc) to 0.94 (Omniscape.jl).

### Results

<img src="sensitivity.png" alt="Sensitivity analysis: warm runtime" width="610">

| Tool | Tier | Compared with JAXScape under | Raster | Warm runtime of the tool | Warm runtime of JAXScape | Speed-up | Spearman's ρ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ConScape.jl | 3 | least-cost path | 1024² | 404 s | 2.00 s | 202 | 0.78 |
| EcoScape | 3 | least-cost path | 8192² | 1620 s | 128 s | 13 | 0.75 |
| samc | 3 | resistance | 4096² | 989 s | 68.5 s | 14 | 0.55 |
| Omniscape.jl | 3 | normalized sensitivity (K = −R) | 8192² | 2170 s | 538 s | 4.0 | 0.94 |

??? note "Warm runtime and Spearman's ρ of every setting on every raster"

    The maps of JAXScape are the references, and an entry of another tool without ρ has no map of JAXScape to compare with, or ρ is undefined.

    | Tool | Distance or label | Setting | 256² | Maryland | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- |
    | JAXScape | resistance | GPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 0.561 s | 5.21 s | 4.76 s | 68.5 s | 268 s |
    | JAXScape | resistance | CPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 39.9 s | 369 s | 317 s | timeout | skipped |
    | JAXScape | least-cost path | GPU, float32 | 0.154 s | 1.25 s | 2.00 s | 32.5 s | 128 s |
    | JAXScape | least-cost path | CPU, float32 | 7.56 s | 67.7 s | 101 s | 1510 s | timeout |
    | JAXScape | normalized sensitivity (K = −R) | GPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 1.10 s | 9.36 s | 9.39 s | 136 s | 538 s |
    | JAXScape | normalized sensitivity (K = −R) | CPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 78.8 s | 663 s | 628 s | timeout | skipped |
    | ConScape.jl | free energy | CPU, float64 | 1.74 s, ρ = 0.84 | failed | 404 s, ρ = 0.78 |  |  |
    | EcoScape | least-cost path | GPU, float32 | 7.72 s, ρ = 0.76 | 13.7 s, ρ = 0.93 | 42.6 s, ρ = 0.75 | 439 s, ρ = 0.76 | 1620 s, ρ = 0.75 |
    | samc | resistance | CPU, float64, direct | 2.47 s, ρ = 0.74 | 11.8 s, ρ = 0.07 | 29.3 s, ρ = 0.59 | 989 s, ρ = 0.55 | oom |
    | Omniscape.jl | resistance | CPU, float64, cg+amg | 2.83 s, ρ = 0.97 | 50.7 s, ρ = 0.80 | 56.7 s, ρ = 0.93 | 785 s, ρ = 0.93 | 2170 s, ρ = 0.94 |
    | Omniscape.jl | resistance | CPU, float64, cholmod | 5.22 s, ρ = 0.97 | 76.1 s, ρ = 0.80 | 116 s, ρ = 0.93 | 1690 s, ρ = 0.93 | timeout |

<img src="maryland.png" alt="Maryland: JAXScape's normalized sensitivity (K = −R) and Omniscape.jl's normalized current" width="610">

Rank of each pixel of Maryland in JAXScape's normalized sensitivity (K = −R) and in Omniscape.jl's normalized current (`cg+amg`), with the pixels without a value in gray. Spearman's ρ between the two maps is 0.80.

### Settings that did not complete

| Tool | Distance or label | Setting | Raster | Status | Recorded reason |
| --- | --- | --- | --- | --- | --- |
| JAXScape | resistance | CPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 4096² | `timeout` | `a call exceeded 3600 s` |
| JAXScape | least-cost path | CPU, float32 | 8192² | `timeout` | `a call exceeded 3600 s` |
| JAXScape | normalized sensitivity (K = −R) | CPU, float32, GeometricMultigridCGSolver(rtol=3e-4) | 4096² | `timeout` | `a call exceeded 3600 s` |
| ConScape.jl | free energy | CPU, float64 | Maryland | `failed` | `non-finite output: 503150 of 503150 values are NaN or infinite` |
| samc | resistance | CPU, float64, direct | 8192² | `oom` | `Error in (function (cond)  : error in evaluating the argument 'vec' in selecting a method for function 'map': std::bad_alloc` |
| Omniscape.jl | resistance | CPU, float64, cholmod | 8192² | `timeout` | `a call exceeded 3600 s` |

### Method

- **Quantity**: the sensitivity $\partial C / \partial p_k$ of $C$ with respect to the permeability $p_k$ of each pixel $k$. The pixels of largest sensitivity are corridors and pinch points: the pixels where a decrease in permeability gives the largest decrease in $C$.
- **JAXScape**: `SensitivityAnalysis` computes the sensitivity, with the windows and settings of the connectivity analysis. The other tools compute a flow: a current, a betweenness or a number of visits.
- **Normalized sensitivity (K = −R)**: the map of the section [Comparison with Omniscape's normalized current](../examples/connectivity_analysis/connectivity_analysis.ipynb#comparison-with-omniscapes-normalized-current) of the connectivity example. Under resistance distance with the proximity $K(R) = -R$, under which every pair of habitat pixels has the same weight, it is the ratio of $-\partial C / \partial r_k = p_k^2\, \partial C / \partial p_k$, the sensitivity of $C$ with respect to the resistance $r_k$ of pixel $k$, to its value on a landscape of permeability 1 with the same quality, and 0 where the latter is not positive. Each call computes both sensitivities.
- **Omniscape.jl**: computes its normalized current with the configuration of its Maryland example, with `radius` equal to `dependency_range`, the same block size, and a graph of eight neighbours.
- **Agreement**: as in the connectivity analysis.

## Linear solvers of JAXScape

The fastest solver of each question, distance and dtype, without an approximation of the distance, on the largest synthetic raster on which one succeeded:

| Question | Distance | dtype | Raster | Fastest solver | Warm runtime | Error | Next fastest solver | Speed-up over it |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| distance calculation | resistance | float32 | 4096² | GeometricMultigridCGSolver | 6.79 s | 1.3e-03 | CholmodSolver | 125 |
| distance calculation | resistance | float64 | 1024² | GeometricMultigridCGSolver | 0.568 s | 2.2e-07 | CuDSSSolver | 9.6 |
| distance calculation | free energy | float64 | 1024² | CuDSSSolver | 5.66 s | 1.3e-15 | CholmodSolver | 7.9 |
| connectivity analysis | resistance | float32 | 256² | CuDSSSolver | 51.3 s | 1.6e-03 | GeometricMultigridCGSolver | 1.0 |
| connectivity analysis | resistance | float64 | 256² | CuDSSSolver | 80.9 s | 5.8e-14 | GeometricMultigridCGSolver | 1.8 |
| sensitivity analysis | resistance | float32 | 256² | GeometricMultigridCGSolver | 106 s | 6.1e-04 | CuDSSSolver | 1.2 |
| sensitivity analysis | resistance | float64 | 256² | CuDSSSolver | 244 s | 1.4e-13 | GeometricMultigridCGSolver | 1.4 |

The approximations of resistance distance, each on the raster of its row above, or else on its largest synthetic raster:

| Question | Distance | dtype | Raster | Approximation | Solver | Warm runtime | Error | Speed-up over the fastest solver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| connectivity analysis | resistance | float32 | 256² | SpielmanApproximation(epsilon=0.15) | GeometricMultigridCGSolver(rtol=3e-4) | 0.279 s | 6.5e-02 | 184 |
| connectivity analysis | resistance | float32 | 256² | SpielmanApproximation(epsilon=0.15) | CuDSSSolver | 1.35 s | 6.5e-02 | 38 |
| connectivity analysis | resistance | float64 | 256² | SpielmanApproximation(epsilon=0.15) | GeometricMultigridCGSolver(rtol=3e-4) | 0.663 s | 6.5e-02 | 122 |
| connectivity analysis | resistance | float64 | 256² | SpielmanApproximation(epsilon=0.15) | CuDSSSolver | 1.56 s | 6.5e-02 | 52 |
| sensitivity analysis | resistance | float32 | 256² | SpielmanApproximation(epsilon=0.15) | GeometricMultigridCGSolver(rtol=3e-4) | 0.561 s | 7.7e-02 | 189 |
| sensitivity analysis | resistance | float32 | 256² | SpielmanApproximation(epsilon=0.15) | CuDSSSolver | 1.73 s | 7.7e-02 | 61 |
| sensitivity analysis | resistance | float64 | 256² | SpielmanApproximation(epsilon=0.15) | GeometricMultigridCGSolver(rtol=3e-4) | 2.03 s | 7.7e-02 | 120 |
| sensitivity analysis | resistance | float64 | 256² | SpielmanApproximation(epsilon=0.15) | CuDSSSolver | 2.73 s | 7.7e-02 | 89 |

### Results

<img src="solvers.png" alt="Linear solvers: warm runtime" width="732">

<img src="solver-errors.png" alt="Linear solvers: warm runtime and error" width="732">

??? note "Warm runtime and error of every solver on every raster"

    An entry without an error has no reference.

    | Question | Distance | dtype | Solver | 256² | Maryland | 1024² | 4096² | 8192² |
    | --- | --- | --- | --- | --- | --- | --- | --- | --- |
    | distance calculation | resistance | float32 | GeometricMultigridCGSolver | 0.0380 s, error 6.4e-05 |  | 0.423 s, error 2.7e-04 | 6.79 s, error 1.3e-03 | oom |
    | distance calculation | resistance | float32 | CholmodSolver | 0.621 s, error 9.5e-06 |  | 24.4 s, error 1.2e-04 | 851 s, error 2.7e-04 | oom |
    | distance calculation | resistance | float32 | CuDSSSolver | 0.271 s, error 1.0e-04 |  | 5.38 s, error 2.5e-03 | oom | skipped |
    | distance calculation | resistance | float32 | AMJaxCGSolver | 0.444 s, error 5.8e-05 |  | 6.88 s, error 2.1e-04 | oom | skipped |
    | distance calculation | resistance | float32 | dense | oom |  | skipped | skipped | skipped |
    | distance calculation | resistance | float64 | GeometricMultigridCGSolver | 0.0718 s, error 1.7e-07 |  | 0.568 s, error 2.2e-07 | oom | skipped |
    | distance calculation | resistance | float64 | CholmodSolver | 0.674 s, error 0.0e+00 |  | 25.0 s, error 0.0e+00 | oom | skipped |
    | distance calculation | resistance | float64 | CuDSSSolver | 0.276 s, error 2.2e-13 |  | 5.43 s, error 3.0e-12 | oom | skipped |
    | distance calculation | resistance | float64 | AMJaxCGSolver | 0.463 s, error 5.0e-07 |  | 7.23 s, error 9.4e-07 | oom | skipped |
    | distance calculation | resistance | float64 | dense | oom |  | skipped | skipped | skipped |
    | distance calculation | free energy | float64 | CholmodSolver | 1.11 s, error 4.4e-16 |  | 44.6 s, error 4.4e-16 |  |  |
    | distance calculation | free energy | float64 | CuDSSSolver | 0.282 s, error 1.4e-15 |  | 5.66 s, error 1.3e-15 |  |  |
    | distance calculation | free energy | float64 | dense | oom |  | skipped |  |  |
    | connectivity analysis | resistance | float32 | GeometricMultigridCGSolver | 51.4 s, error 1.1e-03 |  |  |  |  |
    | connectivity analysis | resistance | float32 | CholmodSolver | timeout |  |  |  |  |
    | connectivity analysis | resistance | float32 | CuDSSSolver | 51.3 s, error 1.6e-03 |  |  |  |  |
    | connectivity analysis | resistance | float32 | AMJaxCGSolver | timeout |  |  |  |  |
    | connectivity analysis | resistance | float32 | dense | failed |  |  |  |  |
    | connectivity analysis | resistance | float32 | SpielmanApproximation(epsilon=0.15), GeometricMultigridCGSolver(rtol=3e-4) | 0.279 s, error 6.5e-02 | 1.74 s | 1.97 s | 23.3 s | 89.7 s |
    | connectivity analysis | resistance | float32 | SpielmanApproximation(epsilon=0.15), CuDSSSolver | 1.35 s, error 6.5e-02 | 2.86 s | 4.21 s | 38.7 s | 168 s |
    | connectivity analysis | resistance | float64 | GeometricMultigridCGSolver | 142 s, error 5.6e-07 |  |  |  |  |
    | connectivity analysis | resistance | float64 | CholmodSolver | timeout |  |  |  |  |
    | connectivity analysis | resistance | float64 | CuDSSSolver | 80.9 s, error 5.8e-14 |  |  |  |  |
    | connectivity analysis | resistance | float64 | AMJaxCGSolver | oom |  |  |  |  |
    | connectivity analysis | resistance | float64 | dense | failed |  |  |  |  |
    | connectivity analysis | resistance | float64 | SpielmanApproximation(epsilon=0.15), GeometricMultigridCGSolver(rtol=3e-4) | 0.663 s, error 6.5e-02 | 4.50 s | 4.61 s | 64.5 s | 248 s |
    | connectivity analysis | resistance | float64 | SpielmanApproximation(epsilon=0.15), CuDSSSolver | 1.56 s, error 6.5e-02 | 4.11 s | 6.73 s | 74.2 s | 330 s |
    | sensitivity analysis | resistance | float32 | GeometricMultigridCGSolver | 106 s, error 6.1e-04 |  |  |  |  |
    | sensitivity analysis | resistance | float32 | CholmodSolver | timeout |  |  |  |  |
    | sensitivity analysis | resistance | float32 | CuDSSSolver | 129 s, error 2.3e-03 |  |  |  |  |
    | sensitivity analysis | resistance | float32 | AMJaxCGSolver | timeout |  |  |  |  |
    | sensitivity analysis | resistance | float32 | dense | failed |  |  |  |  |
    | sensitivity analysis | resistance | float32 | SpielmanApproximation(epsilon=0.15), GeometricMultigridCGSolver(rtol=3e-4) | 0.561 s, error 7.7e-02 | 5.22 s | 4.76 s | 68.4 s | 267 s |
    | sensitivity analysis | resistance | float32 | SpielmanApproximation(epsilon=0.15), CuDSSSolver | 1.73 s, error 7.7e-02 | 4.99 s | 8.62 s | 100 s | 447 s |
    | sensitivity analysis | resistance | float64 | GeometricMultigridCGSolver | 348 s, error 3.0e-07 |  |  |  |  |
    | sensitivity analysis | resistance | float64 | CholmodSolver | timeout |  |  |  |  |
    | sensitivity analysis | resistance | float64 | CuDSSSolver | 244 s, error 1.4e-13 |  |  |  |  |
    | sensitivity analysis | resistance | float64 | AMJaxCGSolver | oom |  |  |  |  |
    | sensitivity analysis | resistance | float64 | dense | failed |  |  |  |  |
    | sensitivity analysis | resistance | float64 | SpielmanApproximation(epsilon=0.15), GeometricMultigridCGSolver(rtol=3e-4) | 2.03 s, error 7.7e-02 | 16.6 s | 19.1 s | 293 s | 1120 s |
    | sensitivity analysis | resistance | float64 | SpielmanApproximation(epsilon=0.15), CuDSSSolver | 2.73 s, error 7.7e-02 | 9.64 s | 17.5 s | 237 s | 1050 s |

### Settings that did not complete

??? note "23 settings"

    | Question | Distance | dtype | Solver | Raster | Status | Recorded reason |
    | --- | --- | --- | --- | --- | --- | --- |
    | distance calculation | resistance | float32 | GeometricMultigridCGSolver | 8192² | `oom` | allocation of 44.98 GiB failed |
    | distance calculation | resistance | float32 | CholmodSolver | 8192² | `oom` | allocation of 2.50 GiB failed |
    | distance calculation | resistance | float32 | CuDSSSolver | 4096² | `oom` | `jax.errors.JaxRuntimeError: INTERNAL: spineax pbatch token: cuDSS call failed (status 2): cudssExecute analysis [executable_name='jit__analysed_state']` |
    | distance calculation | resistance | float32 | AMJaxCGSolver | 4096² | `oom` | allocation of 3.89 GiB failed |
    | distance calculation | resistance | float32 | dense | 256² | `oom` | allocation of 16.02 GiB failed |
    | distance calculation | resistance | float64 | GeometricMultigridCGSolver | 4096² | `oom` | allocation of 22.50 GiB failed |
    | distance calculation | resistance | float64 | CholmodSolver | 4096² | `oom` | allocation of 9.88 GiB failed |
    | distance calculation | resistance | float64 | CuDSSSolver | 4096² | `oom` | `jax.errors.JaxRuntimeError: INTERNAL: spineax pbatch token: cuDSS call failed (status 2): cudssExecute analysis [executable_name='jit__analysed_state']` |
    | distance calculation | resistance | float64 | AMJaxCGSolver | 4096² | `oom` | allocation of 3.89 GiB failed |
    | distance calculation | resistance | float64 | dense | 256² | `oom` | allocation of 32.02 GiB failed |
    | distance calculation | free energy | float64 | dense | 256² | `oom` | allocation of 64.01 GiB failed |
    | connectivity analysis | resistance | float32 | CholmodSolver | 256² | `timeout` | `a call exceeded 900 s` |
    | connectivity analysis | resistance | float32 | AMJaxCGSolver | 256² | `timeout` | `a call exceeded 900 s` |
    | connectivity analysis | resistance | float32 | dense | 256² | `failed` | `jax.errors.JaxRuntimeError: NOT_FOUND: All configs failed during profiling or were excluded from selection.` |
    | connectivity analysis | resistance | float64 | CholmodSolver | 256² | `timeout` | `a call exceeded 900 s` |
    | connectivity analysis | resistance | float64 | AMJaxCGSolver | 256² | `oom` | allocation of 11.42 GiB failed |
    | connectivity analysis | resistance | float64 | dense | 256² | `failed` | `jax.errors.JaxRuntimeError: NOT_FOUND: Failed to get configs for: 2 out of 24 instructions. See logs for all failures. Example failure: All configs failed during profiling or were excluded from selection.` |
    | sensitivity analysis | resistance | float32 | CholmodSolver | 256² | `timeout` | `a call exceeded 900 s` |
    | sensitivity analysis | resistance | float32 | AMJaxCGSolver | 256² | `timeout` | `a call exceeded 900 s` |
    | sensitivity analysis | resistance | float32 | dense | 256² | `failed` | `jax.errors.JaxRuntimeError: NOT_FOUND: All configs failed during profiling or were excluded from selection.` |
    | sensitivity analysis | resistance | float64 | CholmodSolver | 256² | `timeout` | `a call exceeded 900 s` |
    | sensitivity analysis | resistance | float64 | AMJaxCGSolver | 256² | `oom` | allocation of 11.70 GiB failed |
    | sensitivity analysis | resistance | float64 | dense | 256² | `failed` | `jax.errors.JaxRuntimeError: NOT_FOUND: Failed to get configs for: 2 out of 24 instructions. See logs for all failures. Example failure: All configs failed during profiling or were excluded from selection.` |

### Method

- **Solvers**: JAXScape solves the linear systems of resistance and free-energy distances with one of several solvers, on the GPU, in float32 and float64. `dense` is the dense solve, `solver=None`. `CholmodSolver` factorizes and solves on the CPU.
- **Rasters**: in the distance calculation, every raster of its distance; in the windowed analyses, 256² under exact resistance distance and 256², Maryland, 1024², 4096² and 8192² under `SpielmanApproximation(epsilon=0.15)` with `GeometricMultigridCGSolver(rtol=3e-4)` and with `CuDSSSolver`.
- **Error**: in the distance calculation, the largest relative error with respect to the float64 reference of the distance calculation; in the windowed analyses, the relative L2 error $\lVert m - m_{\mathrm{ref}} \rVert_2 / \lVert m_{\mathrm{ref}} \rVert_2$ of the map $m$ with respect to the map $m_{\mathrm{ref}}$ of exact resistance distance in float64 with `CuDSSSolver` on the same tiles, measured on 256² alone. In the figure of error, errors below $10^{-16}$ are drawn at $10^{-16}$.
- **Time limit**: a call stops after 15 min, and under `SpielmanApproximation(epsilon=0.15)` after 1 h.
- **Not run**: free-energy distance in float32; free-energy distance with `GeometricMultigridCGSolver` or `AMJaxCGSolver`. In float32, JAXScape resolves no free-energy distance between sites across the raster, at any theta. The iterative solvers `GeometricMultigridCGSolver` and `AMJaxCGSolver` resolve no free-energy distance in either dtype: as the docstring of `FreeEnergyDistance` states, they need tolerances far below their defaults to resolve the small entries of $Z$ between distant vertices.
