# Linear solvers

Computing resistance and free-energy distances, and their gradients, requires
solving linear systems in a sparse symmetric positive definite matrix of the
graph, which `system_matrix` returns. For `ResistanceDistance`, this matrix is
the graph Laplacian grounded at one vertex. For `FreeEnergyDistance`, with
conductances $a_{ij}$ and inverse temperature $\theta$, it is the Laplacian of
the edge weights $a_{ij} e^{-\theta / a_{ij}}$ plus the positive diagonal
$\kappa_i = \sum_j a_{ij} (1 - e^{-\theta / a_{ij}})$. A linear solver, passed
as the `solver` argument of the distance, solves these systems; `solver=None`
computes the distances with dense linear algebra instead. The solver sets the
runtime, the memory and the accuracy of the distances.

## Choosing a solver

The default solver of `ResistanceDistance` is `GeometricMultigridCGSolver()`:
conjugate gradients (CG) preconditioned by geometric multigrid, in JAX on CPU
and GPU, for the matrices of grid graphs such as those of `GridGraph`.
`FreeEnergyDistance` uses dense linear algebra by default, `solver=None`.

| Distance and query | GPU | CPU |
|---|---|---|
| Resistance distances among a few vertices: `nodes_to_nodes_distance`, few sources and targets, gradient-based fitting of conductances to observed distances, and `SpielmanApproximation` | `GeometricMultigridCGSolver`, the default | `GeometricMultigridCGSolver`, the default |
| Resistance distances from one or all sources to every vertex: `all_pairs_distance`, and windowed analyses with `ExactResistance` | `solver=None` for small graphs and tiles; `GeometricMultigridCGSolver` for larger ones | `solver=None` for small graphs and tiles; `CholmodSolver` for larger ones |
| Free-energy distances | `solver=None`, the default, for small graphs; `CuDSSSolver` for larger ones | `solver=None` for small graphs; `CholmodSolver` for larger ones |
| Resistance or free-energy distances on graphs other than grid graphs | `CuDSSSolver` | `CholmodSolver` |

In float64, on GPUs of low float64 throughput, `CuDSSSolver` can compute
resistance distances faster than `GeometricMultigridCGSolver`.

Memory restricts the choice first. Dense linear algebra stores matrices of
$n^2$ entries for a graph of $n$ vertices, and so fits small graphs only.
`GeometricMultigridCGSolver` stores the matrix on its grid and a few arrays of $n$
entries per right-hand side. Sparse Cholesky solvers store a factorization of
the matrix; when it does not fit in memory, use `AMJaxCGSolver` on a GPU and
`PyAMGCGSolver` on a CPU.

The entries of the inverse of the free-energy system matrix decay
exponentially with path cost. Iterative solvers therefore need tolerances far
below their defaults to resolve the distances between distant vertices, and
float32 resolves a limited range of `theta` on a given landscape.

With a solver, `ResistanceDistance` requires a connected graph: the grounded
Laplacian of a disconnected graph is singular, and queries raise an error. A
grid graph is disconnected when edges of zero conductance separate it, as the
weight function `fun` of `GridGraph` gives between pixels of zero
permeability. Windowed analyses raise permeability to a floor value, so that
their graphs are connected.

The tolerance at which a solver gives accurate distances depends on the query
and on the floating-point precision. The convergence of a solver does not
guarantee accurate distances: check the distances of your queries, for example
against `solver=None` on a small graph. The measurements behind this guide are
in [issue #35](https://github.com/vboussange/jaxscape/issues/35#issuecomment-6760998974).

## `lineax` solvers

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

## `JAXScape` solvers

`jaxscape.solvers` provides five solvers. `GeometricMultigridCGSolver` needs no
optional dependency. Each of the four others needs one, installed with an
extra of `jaxscape`:

```console
uv add jaxscape --extra cholespy   # CholmodSolver
uv add jaxscape --extra pyamg      # PyAMGCGSolver
uv add jaxscape --extra amjax      # AMJaxCGSolver
uv add jaxscape --extra cudss      # CuDSSSolver (NVIDIA GPU, CUDA 13)
```

| Solver | Method | Execution | Graphs |
|---|---|---|---|
| `GeometricMultigridCGSolver` | Conjugate gradients preconditioned by geometric multigrid | JAX, on CPU or GPU | Grid graphs |
| `AMJaxCGSolver` | Conjugate gradients preconditioned by algebraic multigrid | Preconditioner built on the CPU; solves in JAX, on CPU or GPU | Any connected graph |
| `PyAMGCGSolver` | Conjugate gradients preconditioned by algebraic multigrid | SciPy and PyAMG on the CPU, through a host callback | Any connected graph |
| `CholmodSolver` | Sparse Cholesky factorization | CHOLMOD on the CPU, through a host callback | Any connected graph |
| `CuDSSSolver` | Sparse Cholesky factorization | NVIDIA cuDSS on the GPU, inside `jax.jit` | Any connected graph |

A host callback runs Python code on the CPU from a compiled JAX program. Arrays
on a GPU then travel to the CPU and back.

!!! info "Testing"

    The continuous integration tests `GeometricMultigridCGSolver`, `CholmodSolver`,
    `PyAMGCGSolver` and `AMJaxCGSolver`. Its machines have no NVIDIA GPU: they
    test the logic of `AbstractSparseDirectSolver` with a dense Cholesky
    factorization in place of cuDSS. The `CuDSSSolver` tests are marked `gpu`
    and run only on a machine with an NVIDIA GPU, spineax and cuDSS.

### Geometric multigrid

`GeometricMultigridCGSolver` solves the matrices of grid graphs, such as the
grounded Laplacian of a `GridGraph` with rook or queen contiguity, by conjugate
gradients preconditioned by geometric multigrid. It is the default solver of
`ResistanceDistance`. It runs in JAX on CPU and GPU, in float32 and float64,
inside `jax.jit`, and needs no optional dependency.

??? info "Theory: multigrid preconditioning"

    The number of conjugate gradient iterations grows with the condition number
    of the matrix, which grows with the size of the grid and with the contrast
    of conductances. A preconditioner applies an approximate inverse of the
    matrix at each iteration, so that fewer iterations reach the tolerance. A
    multigrid preconditioner builds this approximate inverse from a hierarchy
    of coarser grids: relaxation on a grid reduces the components of the error
    that vary from pixel to pixel, and the coarser grids reduce the components
    that vary smoothly, at a fraction of the cost. Geometric multigrid builds
    the coarser grids from the raster: each keeps every other pixel along each
    axis. The entry below details the cycle, the interpolation between grids
    and the solve on the coarsest grid.

The state of the solver holds the grid, a raster shape and contiguity, and the
matrix coefficients on it. `init` reads the grid from the sparse indices of
the matrix; `distance.init(graph)` reads it from the shape and contiguity of a
`GridGraph`, also inside `jax.jit`. `reinit` writes the values of another
matrix of the same grid into the state. Each solve builds the multigrid
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

`rtol` bounds the norm of the residual that conjugate gradients update at each
iteration, relative to the norm of the right-hand side. A larger `rtol` takes
fewer iterations. In float32, rounding stops the decrease of the residual
`b - A x` of the solution `x` at some level, and an `rtol` below that level
adds iterations without improving the solution. The resistances to pixels
behind barriers converge last: a large `rtol` first degrades the gradient of
the distances with respect to the permeability of these pixels.

In float32 on grid graphs, `GeometricMultigridCGSolver` computed the most accurate
resistance distances of the solvers measured in issue #35. Sparse Cholesky
solvers in float32 compute inaccurate solutions around the ground vertex of
large tiles (see the note on grounding of
[`ResistanceDistance`][jaxscape.resistance_distance.ResistanceDistance]): use
them in float64.

::: jaxscape.solvers.geometricmultigridcgsolver.GeometricMultigridCGSolver
    options:
      members: false

### Algebraic multigrid

Algebraic multigrid builds its coarser levels from the entries of the matrix,
without a grid, and so preconditions the matrix of any graph. `AMJaxCGSolver`
and `PyAMGCGSolver` solve by conjugate gradients preconditioned by an algebraic
multigrid hierarchy, which PyAMG builds, by smoothed aggregation by default.
They suit graphs other than grid graphs whose sparse Cholesky factorization
does not fit in memory.

`AMJaxCGSolver` builds the hierarchy once, in `init`, on the CPU. It solves
with Lineax conjugate gradients, preconditioned by AMJax multigrid V-cycles, in
JAX on CPU or GPU, inside `jax.jit` and `jax.grad`. `reinit` keeps the
hierarchy for a matrix of the same sparsity pattern with other values. The
hierarchy has the dtype of the matrix. For ill-conditioned graph Laplacians,
enable 64-bit floats with `JAX_ENABLE_X64=true` and build the matrix and the
right-hand sides in float64: enabling 64-bit floats does not convert existing
float32 arrays. A matrix whose hierarchy has a single level, as that of a small
graph, is solved by conjugate gradients without preconditioner.

`PyAMGCGSolver` builds the hierarchy at each solve and runs SciPy conjugate
gradients on the CPU, through a host callback, on one right-hand side after
another.

::: jaxscape.solvers.amjaxcgsolver.AMJaxCGSolver
    options:
      members: false

::: jaxscape.solvers.pyamgcgsolver.PyAMGCGSolver
    options:
      members: false

### Sparse Cholesky factorization

A sparse Cholesky factorization writes the matrix as $L L^\top$, with $L$
sparse and lower triangular, after a reordering of its rows and columns that
limits the number of entries of $L$. Each solve then takes two triangular
solves. The factorization costs far more than a solve, and many right-hand
sides share it. Its cost depends on the sparsity pattern of the matrix only, so
that the contrast of conductances does not change it. `CholmodSolver` and
`CuDSSSolver` solve the matrix of any connected graph. In float32, their
solutions on large grids are inaccurate around the ground vertex of
`ResistanceDistance`.

`CholmodSolver` factorizes with CHOLMOD, through cholespy, on the CPU, through
a host callback. The states of one sparsity pattern share one factorization,
which solves reuse while the matrix values stay the same.

`CuDSSSolver` factorizes with NVIDIA cuDSS, through spineax, on the GPU, inside
`jax.jit`. Its state keeps the symbolic analysis of the sparsity pattern, which
orders the rows and columns and sets the structure of $L$; a matrix with new
values is factorized on this analysis. It needs an NVIDIA GPU, Linux x86-64
and Python 3.12 or newer.

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
  Some solvers read concrete sparse indices or run code on the CPU, so call it
  outside `jax.jit`. With `GeometricMultigridCGSolver`, `distance.init` of a
  `GridGraph` also runs inside `jax.jit`: it reads the grid from the shape and
  contiguity of the graph.
- `distance.reinit(graph, state)` returns the state for another graph of the
  same sparsity pattern, and keeps what the solver can reuse. It runs inside
  `jax.jit`.
- Queries take the state as `state`, and the queries on one graph can share
  it. A query always solves the system matrix of the graph it receives: a state
  built for another graph makes the query slower and leaves its distances
  unchanged.

An optimization loop over conductances passes the state from one step to the
next. The state has no gradient: with `has_aux=True`, the differentiated
function returns it alongside the loss.

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
| `GeometricMultigridCGSolver` | The grid; each solve builds the multigrid hierarchy of the new values |
| `AMJaxCGSolver` | The algebraic multigrid hierarchy |
| `CuDSSSolver` | The symbolic analysis, and the factorization while the values stay the same |
| `CholmodSolver` | Its last factorization, while the values stay the same |
| `PyAMGCGSolver`, Lineax solvers | Nothing: each query runs `init`, which only records the matrix |

The graphs of one raster shape and contiguity can also share a
[`GridTopology`][jaxscape.graph.GridTopology], built once outside `jax.jit`. It
holds their edges, so that each `GridGraph(surface, fun=fun, topology=topology)`
computes its edge weights only.

Windowed analyses build their own states and pass them from tile to tile; see
[`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis].

## Extending the solvers

Three public classes underlie the solvers above and serve to write new ones.
`BCOOLinearOperator` presents a sparse matrix to Lineax.
`AbstractReusableStateSolver` is the base of the solvers whose state serves
another matrix of the same sparsity pattern. `AbstractSparseDirectSolver`
turns the phases of a sparse factorization library into such a solver;
`CuDSSSolver` subclasses it.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      heading_level: 3
      members: false

::: jaxscape.solvers.operator.AbstractReusableStateSolver
    options:
      heading_level: 3
      members:
        - reinit

::: jaxscape.solvers.sparsedirectsolver.AbstractSparseDirectSolver
    options:
      heading_level: 3
      members:
        - factorize
        - refactorize
        - solve
        - residual_tolerance
        - reinit
