# JAXScape

JAXScape computes differentiable landscape connectivity distances on graphs. This glossary covers how those distances reuse linear-solver work across queries and graphs.

## Language

### Queries and solver state

**Query**:
One distance computation on a graph: between all vertices, among selected nodes, or from sources to targets.

**System matrix**:
The sparse symmetric positive definite matrix whose linear systems the queries on a graph solve, such as its grounded Laplacian.

**Solver state**:
What a linear solver has prepared for a system matrix, so that solves with that matrix skip the preparation. For a JAXScape sparse solver, the sparsity pattern and values of the system matrix, with a factorization or a preconditioner.
_Avoid_: Setup data, cache

**Reinitialization**:
Producing the solver state of a new system matrix from a state of the same sparsity pattern, keeping what stays valid.
_Avoid_: Refresh, materialization, update

**Outdated state**:
A solver state prepared for other values than the system matrix of the current graph.
_Avoid_: Stale state

### Sparse solvers

**Sparsity pattern**:
The positions of the stored entries of a system matrix. On a grid, the raster shape and contiguity determine it.
_Avoid_: Geometry, structure

**Grid topology**:
The edges of the grid graphs of one raster shape and contiguity, without their weights (`GridTopology`): their underlying graph. The conductance surfaces of one raster share it, so it is built once rather than with each graph, and it fixes the sparsity pattern of their system matrices.
_Avoid_: Grid geometry

### Sparse direct solvers

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

### Sparse iterative solvers

**Preconditioner**:
What the state of an iterative solver keeps across values, from which each solve applies an approximate inverse of the current system matrix: the grid of `GeometricMultigridCGSolver`, on which each solve builds a multigrid hierarchy, or the hierarchy of `AMJaxCGSolver`. Values other than those it was built from change the number of iterations, and a solve that then exceeds `max_steps` fails.
_Avoid_: Hierarchy (one kind of preconditioner), setup

### Windowed analyses

**Window**:
The pixels within `dependency_range` of a pixel along both axes. A pixel counts the pixels of its window.
_Avoid_: Neighbourhood

**Block centre**:
A pixel whose window an analysis computes. Without coarsening every pixel is one; with coarsening, values interpolate between them.

**Block source**:
The pixels that a block centre stands for, weighted by their interpolation weights times their quality, and summing to 1. A resistance window query from it gives the mean of the resistances from these pixels to each pixel of the window (`source_weights`).
_Avoid_: Block average, spread source

**Tile**:
The union of the windows of a grid of up to `batch_size` block centres; with as many as the raster has, the padded raster. One window query from all of them, on the graph of the tile, gives each its distances.
_Avoid_: Batch window, chunk

**Window query**:
The distances from each source to the pixels of its window (`AbstractDistance.sources_to_windows_distance`), the query of windowed analyses for their block centres. Their default proximity reads its profile off a sources-to-targets query.

**Probe**:
One of the random sign vectors from which the gradient of a Spielman window query estimates the gradient of exact resistances: it draws a sign per source and a sign per pixel, whose product weighs the pair of a source and a pixel of its window.

**Sources that share work**:
A distance whose query from several sources costs much less than one query per source, such as resistance, whose solves on a graph serve every source with one solver state, a factorization or a multigrid hierarchy. Windowed analyses run it on tiles, and other distances on windows.
