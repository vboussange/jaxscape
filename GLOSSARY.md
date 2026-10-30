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
