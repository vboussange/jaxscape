# JAXScape

JAXScape computes differentiable landscape connectivity distances on graphs. This glossary covers how those distances reuse linear-solver work across queries and graphs.

## Language

### Queries and solver state

**Query**:
One distance computation on a graph: between all vertices, among selected nodes, or from sources to targets.

**System matrix**:
The sparse symmetric positive definite matrix whose linear systems the queries on a graph solve, such as its grounded Laplacian.

**Solver state**:
What a linear solver has prepared for a system matrix, so that solves with that matrix skip the preparation.
_Avoid_: Setup data, cache

**Reinitialization**:
Producing the solver state of a new system matrix from a state of the same sparsity pattern, keeping what stays valid.
_Avoid_: Refresh, materialization, update

**Outdated state**:
A solver state prepared for other values than the system matrix of the current graph.
_Avoid_: Stale state

**Latest state**:
The solver state that windowed analyses left last for a distance, graph shape and dtype, which the next windowed analysis, run, calibration or default-proximity profile with them starts from and replaces. A process keeps those of the 8 most recently used combinations.
_Avoid_: Cached state, shared cache

### Sparse direct solvers

**Sparsity pattern**:
The positions of the stored entries of a system matrix. On a grid, the raster shape and contiguity determine it.
_Avoid_: Geometry, structure

**Analysis**:
The costly phase of a sparse direct solver, which depends on the sparsity pattern alone.
_Avoid_: Symbolic factorization, setup

**Factorization**:
The numeric Cholesky factors of a system matrix.

**Refactorization**:
A factorization of new values that reuses an existing analysis.
_Avoid_: Update, refresh

**Used-up state**:
A solver state whose factorization a refactorization has consumed. Solving with it repeats the analysis and stays correct.

### Windowed analyses

**Window**:
The pixels within `dependency_range` of a pixel along both axes. A pixel counts the pixels of its window.
_Avoid_: Neighbourhood

**Block centre**:
A pixel whose window an analysis computes. Without coarsening every pixel is one; with coarsening, values interpolate between them.

**Tile**:
The union of the windows of a grid of up to `batch_size` block centres; with as many as the raster has, the padded raster. One window query from all of them, on the graph of the tile, gives each its distances.
_Avoid_: Batch window, chunk

**Window query**:
The distances from each source to the pixels of its window (`AbstractDistance.sources_to_windows_distance`), the query of windowed analyses for their block centres. Their default proximity reads its profile off a sources-to-targets query.

**Probe**:
One of the random sign vectors from which the gradient of a Spielman window query estimates the gradient of exact resistances: it draws a sign per source and a sign per pixel, whose product weighs the pair of a source and a pixel of its window.

**Sources that share work**:
A distance whose query from several sources costs much less than one query per source, such as resistance, whose solves on a graph serve every source with one solver state, a factorization or a multigrid hierarchy. Windowed analyses run it on tiles, and other distances on windows.
