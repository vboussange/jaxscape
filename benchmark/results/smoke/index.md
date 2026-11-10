# Benchmark

This page compares JAXScape 0.0.16 with the software that ecologists use for connectivity analysis, on three questions from ecology. It is generated from the records of the run `smoke`, started on 2026-11-10 on gpunode05. [`results.csv`](results.csv) lists every cell of the run.

## Questions

- **E1. Isolation between populations.** How isolated are two populations once the landscape between them is taken into account? *Quantity:* the effective distance $d_{ij}$ between focal habitat sites, which replaces Euclidean distance in dispersal kernels and in models of isolation by distance.
- **E2. Habitat connectivity.** How much habitat can each location functionally reach, and how much connected habitat does the landscape hold in total? *Quantity:* Hanski's connectivity $S_i = \sum_j K(d_{ij}) q_j$ at each location, and its quality-weighted total $\sum_i q_i S_i$, the quantity behind the equivalent connected area (ECA).
- **E3. Corridors and pinch points.** Which locations carry dispersal between habitat areas, so that losing their permeability would reduce connectivity the most? *Quantity:* a map of how much each location contributes to connectivity: the marginal sensitivity $\partial C / \partial p_k$ in JAXScape, and a flow, such as current, betweenness or visits, in the other tools.

Each comparison carries a tier of comparability. Tier 1 compares the same quantity, so that the error of each tool against a float64 reference measures its accuracy. Tier 2 compares the same quantity computed on another graph or with another kernel, and Tier 3 a related quantity that answers the same question; both report Spearman's rank correlation between maps. A runtime gap measures a speed-up only at Tier 1.

## Landscapes

Each synthetic case draws a Gaussian random field with a Gaussian covariance of correlation length 25 pixels, on a periodic grid with a fixed seed, and normalizes its ranks to [0, 1]. Quality equals the field where the field is at least 0.7, and 0 elsewhere, so that 30 % of the cells are habitat. Permeability is $10^{-2(1 - f)}$ for the field $f$, from 0.01 to 1, and resistance is its inverse. A larger raster covers a larger area at the same grain.

## Comparison

Each cell runs in its own process, pinned to the cores of the environment table, with thread settings matched to them; XLA sizes its thread pools to these cores. A warm-up call comes first and compiles the code of tools that compile. The warm runtime is the median of the five timed calls that follow it, or the time of a single call when the warm-up takes more than five minutes. Every call redoes all work that depends on the landscape's values: graph assembly, factorization, multigrid hierarchy and solver state. A call stops after one hour. Within a configuration, rasters run from the smallest up, and after a failure, a timeout or running out of memory (`oom`), the configuration skips its larger rasters. A tool runs each setting of a small grid fixed in advance from its documentation: the tables list every setting, and the figures show, at each raster, the fastest setting that succeeds. In the figures, tools on the GPU are solid and on the CPU dashed, and rasters where no setting succeeds show a status along the top edge.

### E1. Isolation between populations

Each tool computes the distances among 32 sites drawn at random from the habitat cells with a fixed seed, across the whole raster: 496 pairs. Every tool builds the same graph: rook contiguity, and the resistance of an edge is the mean resistance of its two pixels. The agreement of a record is the largest relative error over the pairs against a float64 reference, computed once for each case and distance: a sparse direct solve with `CholmodSolver` for resistance and free energy, and Dijkstra's algorithm in SciPy for least-cost distances.

![E1: warm runtime against raster cells](e1.png)

Agreement (largest relative error) of each configuration and raster, or the status of the cell; `missing` marks a cell whose reference is unavailable.

| Tool | Distance | Device | dtype | Solver | Tier | 32² |
| --- | --- | --- | --- | --- | --- | --- |
| JAXScape | resistance | GPU | float32 | GridMultigridSolver | 1 | 1.1e-05 |
| JAXScape | resistance | CPU | float32 | GridMultigridSolver | 1 | 5.8e-06 |
| JAXScape | lcp | GPU | float32 |  | 1 | 2.5e-07 |
| JAXScape | lcp | CPU | float32 |  | 1 | 2.4e-07 |
| JAXScape | free_energy | GPU | float64 | CuDSSSolver | 1 | 2.0e-15 |
| JAXScape | free_energy | CPU | float64 | CholmodSolver | 1 | 0.0e+00 |
| Circuitscape.jl | resistance | CPU | float64 | cg+amg | 1 | 8.9e-07 |
| Circuitscape.jl | resistance | CPU | float64 | cholmod | 1 | 4.6e-12 |
| ConScape.jl | free_energy | CPU | float64 |  | 1 | 1.4e-15 |
| gdistance | resistance | CPU | float64 |  | 1 | 7.3e-14 |
| gdistance | lcp | CPU | float64 |  | 1 | 8.1e-16 |
| skimage.graph.MCP | lcp | CPU | float64 |  | 1 | 3.3e-16 |

## Environment

| Setting | Value |
| --- | --- |
| Host | gpunode05 |
| GPU | NVIDIA GeForce RTX 3090, driver 580.178.04, 24576 MiB |
| CPU | AMD EPYC 7742 64-Core Processor |
| Cores | 48-63 |
| Threads | `OMP_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=16`, `MKL_NUM_THREADS=16`, `JULIA_NUM_THREADS=16`, `RCPP_PARALLEL_NUM_THREADS=16` |
| Python | 3.13.5 |
| Packages | jaxscape 0.0.16, jax 0.12.0.1, jaxlib 0.12.0.1, numpy 2.5.4, scipy 1.18.1, scikit-image 0.26.0, Julia 1.13.3, Circuitscape 6.1.0, ConScape 0.3.0, JSON 1.13.4, NPZ 0.4.3, R 4.1.2, gdistance 1.6.5 |
| Tools | JAXScape 0.0.16, Circuitscape.jl 6.1.0, ConScape.jl 0.3.0, gdistance 1.6.5, skimage.graph.MCP 0.26.0 |

## Parameter mappings

The header comment of each adapter states how it maps the shared parameters onto its tool.

| Tool | Questions | Adapter | Mapping |
| --- | --- | --- | --- |
| JAXScape | E1 | `adapters/jaxscape.py` | JAXScape builds a `GridGraph` of the permeability raster with rook contiguity, the conductance of an edge being the harmonic mean of the permeabilities of its two pixels (`landscape.conductance`), so that its resistance is their mean resistance. Least-cost and free-energy distances take 1 / conductance as the edge cost, and the free energy takes the shared `theta`. E1 is a nodes-to-nodes query among the sites. The distance runs with the solver and dtype of its settings, and every call builds the graph and the solver state. |
| Circuitscape.jl | E1 | `adapters/circuitscape.jl` | Circuitscape.jl computes the pairwise resistances among the E1 sites, in its `pairwise` scenario, with the permeability raster as the conductance of each cell. `connect_four_neighbors_only` gives rook contiguity, and `connect_using_avg_resistances` makes the resistance of an edge the mean resistance of its two cells, as `landscape.conductance` does. The solver is `cg+amg` or `cholmod`, in double precision, with `parallelize` on the Julia threads of the cell. Circuitscape.jl has no public in-memory API for this scenario: each call builds its `RasterData` in memory and runs the internals that `compute` runs after reading files (`build_problem`, `single_ground_all_pairs`), and writes only its table of resistances, to /dev/shm. Every call rebuilds the graph, the Laplacian and the AMG hierarchy or Cholesky factor. |
| ConScape.jl | E1 | `adapters/conscape.jl` | ConScape.jl computes the free-energy distances among the E1 sites, with the shared `theta`. Its graph has rook contiguity (`N4`), the affinity of an edge being the harmonic mean of the permeabilities of its two pixels (`AverageWeight`), as `landscape.conductance` gives, and the cost of an edge 1 / affinity (`Inv`). The sites are the targets of a `GridRSP`. ConScape's free-energy distance is directed, and the adapter returns the mean of the two directions between two sites, as JAXScape's `FreeEnergyDistance` does. Every call rebuilds the affinities, the `Grid` and the `GridRSP`, with its sparse LU factorization, which runs on the BLAS threads of the cell. |
| gdistance | E1 | `adapters/gdistance.R` | gdistance builds a transition layer of the resistance raster, 1 / permeability, with 4 neighbours (rook contiguity) and the transition function 1 / mean(x): the conductance between two pixels is the reciprocal of their mean resistance, `landscape.conductance`. Resistance distances are `commuteDistance` divided by the volume of the graph, the sum of its conductances in both directions, and least-cost distances are `costDistance`, whose edge cost is 1 / conductance. Pixels have side 1, so `geoCorrection` leaves the layer unchanged and is left out. Every call builds the transition layer, whose conductances depend on the resistance. |
| skimage.graph.MCP | E1 | `adapters/skimage.py` | skimage.graph.MCP_Geometric finds least-cost paths on the resistance raster, 1 / permeability, with rook contiguity (`fully_connected=False`). The cost of a step is the mean resistance of its two pixels, 1 / `landscape.conductance`, the edge cost of JAXScape. Each call builds the MCP object, and searches from each site until it reaches all the others. |

## Excluded tools

| Tool | Reason |
| --- | --- |
| Linkage Mapper | Needs ArcGIS. |
| Circuitscape 4 | Superseded by Circuitscape.jl. |
| GFlow | Abandoned. |
| UNICOR | Has no release and no licence. |
| Conefor | Starts from a patch graph. |
| Makurhini | Starts from vector patches. |
| GRASS `r.cost` | Duplicates least-cost paths. |
| WhiteboxTools | Duplicates least-cost paths. |
| `terra` | Duplicates least-cost paths. |
| DiffiScape | Wraps JAXScape. |
| circuitscaper | Wraps Circuitscape.jl. |
| enerscape | Computes no connectivity from resistance. |
| Zonation | Computes no connectivity from resistance. |
| MSPA | Computes no connectivity from resistance. |
| ResistanceGA | Moves to `ilg-toolkit` with inverse landscape genetics. |
| radish | Moves to `ilg-toolkit` with inverse landscape genetics. |
