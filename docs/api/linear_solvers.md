# Linear solvers

Computing resistance and free-energy distances, and their gradients, requires
solving linear systems in a sparse symmetric positive definite matrix of the
graph, which
[`AbstractSolverDistance.system_matrix`][jaxscape.distance.AbstractSolverDistance.system_matrix]
returns. For
[`ResistanceDistance`][jaxscape.resistance_distance.ResistanceDistance], this
matrix is the graph Laplacian grounded at one vertex. For
[`FreeEnergyDistance`][jaxscape.free_energy_distance.FreeEnergyDistance], with
conductances $a_{ij}$ and inverse temperature $\theta$, it is the Laplacian of
the edge weights $a_{ij} e^{-\theta / a_{ij}}$ plus the positive diagonal
$\kappa_i = \sum_j a_{ij} (1 - e^{-\theta / a_{ij}})$. A linear solver, passed
as the `solver` argument of the distance, solves these systems; with
`solver=None`, the distances are computed with dense linear algebra instead. The solver sets the
runtime, the memory and the accuracy of the distances.

## Choosing a solver

The default solver of `ResistanceDistance` is `GeometricMultigridCGSolver()`:
conjugate gradients (CG) preconditioned by geometric multigrid, in JAX on CPU
and GPU, for the matrices of grid graphs such as those of
[`GridGraph`][jaxscape.graph.GridGraph]. On other graphs, the default solver
raises an error; pass `solver`. `FreeEnergyDistance` uses dense linear algebra
by default, `solver=None`.

| Distance and query | GPU | CPU |
|---|---|---|
| Resistance distances among a few vertices: `nodes_to_nodes_distance`, few sources and targets, gradient-based fitting of conductances to observed distances, and [`SpielmanApproximation`][jaxscape.resistance_distance.SpielmanApproximation] | `GeometricMultigridCGSolver`, the default | `GeometricMultigridCGSolver`, the default |
| Resistance distances from one or all sources to every vertex: `all_pairs_distance`, and windowed analyses with [`ExactResistance`][jaxscape.resistance_distance.ExactResistance] | `solver=None` for small graphs and tiles; `GeometricMultigridCGSolver` for larger ones | `solver=None` for small graphs and tiles; `CholmodSolver` for larger ones |
| Free-energy distances | `solver=None`, the default, for small graphs; `CuDSSSolver` for larger ones | `solver=None` for small graphs; `CholmodSolver` for larger ones |
| Resistance or free-energy distances on graphs other than grid graphs | `solver=None` for small graphs; `CuDSSSolver` for larger ones; `AMJaxCGSolver` when the factorization does not fit in memory | `solver=None` for small graphs; `CholmodSolver` for larger ones; `AMJaxCGSolver` when the factorization does not fit in memory |

In float64, on GPUs of low float64 throughput, `CuDSSSolver` can compute
resistance distances faster than `GeometricMultigridCGSolver`.

Memory restricts the choice first. Dense linear algebra stores matrices of
$n^2$ entries for a graph of $n$ vertices, and so fits small graphs only.
`GeometricMultigridCGSolver` stores the matrix on its grid and a few arrays of $n$
entries per right-hand side. Sparse direct solvers store a factorization of
the matrix; when it does not fit in memory, use `GeometricMultigridCGSolver` on
grid graphs and `AMJaxCGSolver` on other graphs.

The entries of the inverse of the free-energy system matrix decay
exponentially with path cost. Iterative solvers therefore need tolerances far
below their defaults to resolve the distances between distant vertices, and in
float32, the distances are accurate over a limited range of `theta` on a given
landscape.

With a solver, `ResistanceDistance` requires a connected graph: the grounded
Laplacian of a disconnected graph is singular, and queries raise an error. A
grid graph is disconnected when some of its pixels are joined to the others
only by edges of zero conductance, such as the edges that the weight function
`fun` of `GridGraph` gives between pixels of zero permeability. Windowed analyses raise permeability to a floor value, so that
their graphs are connected.

The tolerance at which a solver gives accurate distances depends on the query
and on the floating-point precision. A solver can converge and still give
inaccurate distances: check the distances of your queries, for example against
`solver=None` on a small graph. The measurements behind this page are in [issue #35](https://github.com/vboussange/jaxscape/issues/35#issuecomment-6760998974).

## Lineax solvers

Any [Lineax](https://docs.kidger.site/lineax/) solver of sparse symmetric
positive definite operators can be passed as `solver`, such as conjugate
gradients without preconditioner, `lineax.CG`:

```python
import lineax as lx
from jaxscape import FreeEnergyDistance, ResistanceDistance

solver = lx.CG(rtol=1e-6, atol=1e-6)
resistance = ResistanceDistance(solver=solver)
free_energy = FreeEnergyDistance(theta=0.2, solver=solver)
```

The distance passes its system matrix to the solver as a
[`BCOOLinearOperator`](#jaxscape.solvers.operator.BCOOLinearOperator). The
[Lineax documentation](https://docs.kidger.site/lineax/) lists the available
solvers and their options.

## JAXScape solvers

`jaxscape.solvers` provides four solvers. `GeometricMultigridCGSolver` needs no
optional dependency. Each of the three others needs one, installed with an
extra of `jaxscape`:

```console
uv add jaxscape --extra cholespy   # CholmodSolver
uv add jaxscape --extra amjax      # AMJaxCGSolver
uv add jaxscape --extra cudss      # CuDSSSolver (NVIDIA GPU, CUDA 13)
```

| Solver | Method | Execution | Graphs |
|---|---|---|---|
| `GeometricMultigridCGSolver` | Conjugate gradients preconditioned by geometric multigrid | JAX, on CPU or GPU | Grid graphs |
| `AMJaxCGSolver` | Conjugate gradients preconditioned by algebraic multigrid | Preconditioner built on the CPU; solves in JAX, on CPU or GPU | Any connected graph |
| `CholmodSolver` | Sparse Cholesky factorization | CHOLMOD on the CPU, through a host callback | Any connected graph |
| `CuDSSSolver` | Sparse Cholesky factorization | NVIDIA cuDSS on the GPU, inside `jax.jit` | Any connected graph |

A host callback runs Python code on the CPU from a compiled JAX program. Arrays
on a GPU then travel to the CPU and back.

### Geometric multigrid

`GeometricMultigridCGSolver` solves the matrices of grid graphs, such as the
grounded Laplacian of a `GridGraph` with rook or queen contiguity, by conjugate
gradients preconditioned by geometric multigrid. It is the default solver of
`ResistanceDistance`. It runs in JAX on CPU and GPU, in float32 and float64,
inside `jax.jit`, and needs no optional dependency.

??? info "Theory: multigrid preconditioning"

    The number of conjugate gradient iterations grows with the condition number
    of the matrix, which grows with the size of the grid and with the contrast
    of conductances. A preconditioner is an approximate inverse of the matrix,
    which the solver applies at each iteration, so that it reaches the
    tolerance in fewer iterations. In multigrid, the solver builds the
    preconditioner from a hierarchy of coarser grids: by relaxation on a grid,
    it lowers the components of the error that vary from pixel to pixel, and by
    corrections on the coarser grids, the components that vary smoothly, at a
    fraction of the cost. In geometric multigrid, the solver builds the coarser
    grids from the raster: each holds every other pixel along each axis. The
    cycle, the interpolation between grids and the solve on the coarsest grid
    are given in the API entry of `GeometricMultigridCGSolver` below.

The preconditioner of its state is the grid, a raster shape and contiguity.
`init` reads the grid from the sparse indices of the matrix;
`distance.init(graph)` reads it from the shape and contiguity of a
`GridGraph`, also inside `jax.jit`. `reinit` replaces the values of the matrix
with those of another matrix of the same grid. Each solve builds the multigrid
hierarchy of the matrix it solves, so that each tile of a windowed analysis is
solved with the hierarchy of its own conductances.

```python
from jaxscape import ResistanceDistance, SpielmanApproximation
from jaxscape.solvers import GeometricMultigridCGSolver

exact = ResistanceDistance()  # GeometricMultigridCGSolver(rtol=1e-5)
approximate = ResistanceDistance(
    solver=GeometricMultigridCGSolver(rtol=3e-4),
    method=SpielmanApproximation(epsilon=0.15),
)
```

`rtol` is the bound on the norm of the residual that conjugate gradients
update at each iteration, relative to the norm of the right-hand side. With a
larger `rtol`, the solves take fewer iterations. In float32, because of
rounding, the residual `b - A x` of the solution `x` stops decreasing at some
level; with an `rtol` below that level, the iterations continue without
improving the solution. The resistances to pixels behind barriers converge
last: with a large `rtol`, the gradient of the distances with respect to the
permeability of these pixels loses accuracy first.

In float32 on grid graphs, `GeometricMultigridCGSolver` gives the most accurate
resistance distances of the JAXScape solvers. In float32, sparse direct solvers
give inaccurate solutions around the ground vertex of large tiles (see the note on grounding of
[`ResistanceDistance`][jaxscape.resistance_distance.ResistanceDistance]): use
them in float64.

::: jaxscape.solvers.geometricmultigridcgsolver.GeometricMultigridCGSolver
    options:
      members: false
      inherited_members: false

### Algebraic multigrid

In algebraic multigrid, the coarser levels are built from the entries of the
matrix, without a grid, so that the matrix of any graph can be preconditioned. `AMJaxCGSolver`
solves by conjugate gradients preconditioned by an algebraic multigrid
hierarchy, built by smoothed aggregation by default. It suits graphs other than
grid graphs whose sparse Cholesky factorization does not fit in memory.

`AMJaxCGSolver` builds the hierarchy once, in `init`, on the CPU. It solves
by conjugate gradients, preconditioned by V-cycles of the hierarchy, in JAX on
CPU or GPU, inside `jax.jit` and `jax.grad`. `reinit` keeps the prolongation
and restriction operators of the hierarchy, and each solve computes the
matrices of its coarser levels from the current values, as Galerkin products
with these operators. The state also holds, for each level but the coarsest,
the map from the values of the level to those of the next: sparse matrices
several times the size of the hierarchy. The hierarchy has the dtype of the
matrix. For ill-conditioned graph Laplacians, enable 64-bit floats with
`JAX_ENABLE_X64=true` and build the matrix and the right-hand sides in
float64: enabling 64-bit floats does not convert existing float32 arrays. A
matrix whose hierarchy has a single level, as that of a small graph, is solved
by conjugate gradients without preconditioner.

::: jaxscape.solvers.amjaxcgsolver.AMJaxCGSolver
    options:
      members: false
      inherited_members: false

### Sparse direct solvers

A sparse Cholesky factorization of the matrix is a product $L L^\top$, with $L$
sparse and lower triangular, after a reordering of its rows and columns that
limits the number of entries of $L$. Each solve then takes two triangular
solves. The factorization costs far more than a solve, and many right-hand
sides share it. Its cost depends on the sparsity pattern of the matrix only, and
is the same whatever the contrast of conductances. `CholmodSolver` and
`CuDSSSolver` solve the matrix of any connected graph. In float32, their
solutions on large grids are inaccurate around the ground vertex of
`ResistanceDistance`.

`CholmodSolver` factorizes with CHOLMOD, through cholespy, on the CPU, through
a host callback. Solves reuse the last factorization while the values of the
matrix stay the same.

`CuDSSSolver` factorizes with NVIDIA cuDSS, through spineax, on the GPU, inside
`jax.jit`. Its state keeps the analysis of the sparsity pattern: the ordering
of the rows and columns and the structure of $L$. A matrix with new values is
factorized on this analysis. It needs an NVIDIA GPU and Linux x86-64.

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

::: jaxscape.solvers.cudsssolver.CuDSSSolver
    options:
      members: false
      inherited_members: false

### Reusing solver state

Building the state of a solver, as a sparse factorization or a multigrid
hierarchy, can cost far more than a solve. `ResistanceDistance` and
`FreeEnergyDistance` share one state lifecycle:

- `distance.init(graph)` builds the state for the system matrix of `graph`.
  JAXScape's sparse solvers, except `CholmodSolver`, read concrete sparse
  indices, and `AMJaxCGSolver` runs code on the CPU, so call it outside
  `jax.jit`. `ResistanceDistance` also builds the state of a `GridGraph` inside
  `jax.jit`, from the shape and contiguity of the graph. The compiled program
  then holds the arrays of the state as constants: on large grids, it compiles
  faster with the state passed as an argument.
- `distance.reinit(graph, state)` returns the state for another graph of the
  same sparsity pattern, and keeps what the solver can reuse. It runs inside
  `jax.jit`.
- Queries take the state as `state`, and the queries on one graph can share
  it. A query always solves the system matrix of the graph it receives: an
  outdated state can make the query slower, and leaves its distances
  unchanged.

An optimization loop over conductances passes the state from one step to the
next, whatever the solver. `distance.init` returns a state with Lineax's and
JAXScape's solvers, and `None` only with `solver=None`. For a solver that keeps
nothing from one matrix to the next, such as `lineax.CG`, `reinit` ignores the
state and returns `init(graph)`. The state has no gradient: with
`has_aux=True`, the differentiated function returns it alongside the loss.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxscape import GridGraph, ResistanceDistance


def landscape(log_conductance):
    return GridGraph(jnp.exp(log_conductance), fun=lambda x, y: (x + y) / 2)


sites = jnp.array([0, 45, 99])
true_log_conductance = jax.random.normal(jax.random.key(0), (10, 10))
target = ResistanceDistance()(landscape(true_log_conductance), nodes=sites)

distance = ResistanceDistance()  # GeometricMultigridCGSolver
log_conductance = jnp.zeros((10, 10))
state = distance.init(landscape(log_conductance))


def loss(log_conductance, state):
    grid = landscape(log_conductance)
    state = distance.reinit(grid, state)
    predicted = distance.nodes_to_nodes_distance(grid, sites, state)
    return jnp.mean((predicted - target) ** 2), state


@eqx.filter_jit
def step(log_conductance, state):
    (value, state), gradient = eqx.filter_value_and_grad(loss, has_aux=True)(
        log_conductance, state
    )
    return log_conductance - 0.1 * gradient, value, state


for _ in range(10):
    log_conductance, value, state = step(log_conductance, state)
```

What the state of each solver keeps from one graph to the next:

| Solver | Kept by the state |
|---|---|
| `GeometricMultigridCGSolver` | Its preconditioner, the grid; each solve builds the multigrid hierarchy of the new values |
| `AMJaxCGSolver` | Its preconditioner, the prolongation and restriction operators of the algebraic multigrid hierarchy of `init`; each solve computes the matrices of the coarser levels from the new values |
| `CuDSSSolver` | The analysis, and the factorization while the values stay the same |
| `CholmodSolver` | The last factorization, while the values stay the same |
| Lineax's iterative solvers, such as `lineax.CG` | Nothing: each query runs `init` |

The graphs of one raster shape and contiguity can also share a
[`GridTopology`][jaxscape.graph.GridTopology], built once outside `jax.jit`. It
holds their edges, so that each `GridGraph(surface, fun=fun, topology=topology)`
computes its edge weights only.

Windowed analyses build their own states and pass them from tile to tile; see
[`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis].

## Extending the solvers

`BCOOLinearOperator` presents a sparse matrix to Lineax. `AbstractSparseSolver`
is the parent of JAXScape's sparse solvers of symmetric positive definite
matrices. Its state holds the compressed sparse row (CSR) sparsity pattern of
the matrix, read once from concrete sparse indices, the values, which `reinit`
replaces inside `jax.jit`, and a factorization or a preconditioner. It scales
right-hand sides, checks the sparse indices of each new matrix, and reports
failed solves. A new solver subclasses one of its two families, with a final
class. The methods that it implements take the sparsity pattern as a
[`CSRPattern`][jaxscape.solvers.sparsesolver.CSRPattern]: its arrays `offsets`
and `columns` give the columns of each row, and its properties `n`, `nnz` and
`rows` the number of rows, the number of CSR values and the row of each CSR
value.

`AbstractSparseDirectSolver` turns the phases of a sparse factorization library
into a solver, and checks each factorization and solve; `CuDSSSolver`
subclasses it:

```python
from typing import final

import jax.numpy as jnp
from jaxscape.solvers import AbstractSparseDirectSolver


@final
class MySolver(AbstractSparseDirectSolver):
    def factorize(self, pattern, values):
        return my_library.factorize(pattern.offsets, pattern.columns, values)

    def refactorize(self, factorization, pattern, values):  # Optional.
        return my_library.refactorize(factorization, values)

    def solve(self, factorization, vector):
        return my_library.solve(factorization, vector)

    def residual_tolerance(self, dtype):
        return 1e-4 if dtype == jnp.float64 else 1e-1
```

The phases run inside `jax.jit`. Set `residual_tolerance` from the residuals the
library leaves on valid matrices of your workload.

`AbstractSparseIterativeSolver` runs conjugate gradients with a preconditioner;
`GeometricMultigridCGSolver` and `AMJaxCGSolver` subclass it. `preconditioner`
builds what the state keeps across values, in `init`, and `precondition`
returns the preconditioner of the current values at each solve, as a function
of residuals. `matvec` returns the product of the current matrix with vectors,
and `vector_shape` the shape in which both take vectors: `csr_matvec`
multiplies vectors of shape `(n,)` with the CSR values, and
`GeometricMultigridCGSolver` multiplies rasters with its stencil. Jacobi
preconditioning keeps nothing:

```python
from typing import final

import jax.numpy as jnp
from jaxscape.solvers import AbstractSparseIterativeSolver, csr_matvec


@final
class JacobiCGSolver(AbstractSparseIterativeSolver):
    rtol: float = 1e-5
    max_steps: int = 1_000

    def preconditioner(self, pattern, values):
        return None

    def precondition(self, preconditioner, pattern, values):
        on_diagonal = jnp.where(pattern.rows == pattern.columns, values, 0)
        diagonal = jnp.zeros(pattern.n, values.dtype).at[pattern.rows].add(on_diagonal)
        return lambda residuals: residuals / diagonal[:, None]

    def matvec(self, preconditioner, pattern, values):
        return csr_matvec(pattern, values)

    def vector_shape(self, preconditioner, pattern):
        return (pattern.n,)
```

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      heading_level: 3
      members: false

::: jaxscape.solvers.sparsesolver.CSRPattern
    options:
      heading_level: 3
      members: false

::: jaxscape.solvers.sparsesolver.AbstractSparseSolver
    options:
      heading_level: 3
      members:
        - init
        - reinit

::: jaxscape.solvers.sparsedirectsolver.AbstractSparseDirectSolver
    options:
      heading_level: 3
      members:
        - factorize
        - refactorize
        - solve
        - residual_tolerance

::: jaxscape.solvers.sparseiterativesolver.AbstractSparseIterativeSolver
    options:
      heading_level: 3
      members:
        - preconditioner
        - precondition
        - matvec
        - vector_shape

::: jaxscape.solvers.sparseiterativesolver.csr_matvec
    options:
      heading_level: 3
