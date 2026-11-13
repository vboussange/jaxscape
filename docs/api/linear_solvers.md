# Linear solvers

`ResistanceDistance` and `FreeEnergyDistance` solve linear systems. The solver sets their runtime, memory and accuracy.

## Choosing a solver

`ResistanceDistance()` solves with `GridMultigridSolver()`: conjugate gradients
preconditioned by geometric multigrid, in JAX on CPU and GPU, for the matrices
of grid graphs such as `GridGraph`. `FreeEnergyDistance()` takes dense solves,
`solver=None`.

| Distance and query | GPU | CPU |
|---|---|---|
| `ResistanceDistance`, few sources and targets: `nodes_to_nodes_distance`, inverse fitting, `SpielmanApproximation` | `GridMultigridSolver`, the default | `GridMultigridSolver`, the default |
| `ResistanceDistance`, one or all sources to every vertex: `all_pairs_distance`, exact windowed analyses | `solver=None` for small graphs or tiles, `GridMultigridSolver` above | `solver=None` for small graphs or tiles, `CholmodSolver` above |
| `FreeEnergyDistance` | `solver=None`, the default, for small graphs; `CuDSSSolver` above | `solver=None` for small graphs; `CholmodSolver` above |
| Graphs other than grid graphs | `CuDSSSolver` | `CholmodSolver` |

In float64, `CuDSSSolver` can be faster than the default on GPUs with low
float64 throughput.

Memory decides first. `solver=None` holds dense matrices of the size of the
graph squared, and so fits small graphs only. `GridMultigridSolver` holds a few
arrays of the size of the graph per right-hand side. Direct solvers hold a
sparse factorization; when it does not fit, use `GridMultigridSolver` on grid
graphs and `AMJaxCGSolver` on other graphs.

Free-energy distances decay exponentially with path cost: iterative solvers
need tolerances far below their defaults to resolve distant entries, and
float32 resolves a limited range of `theta` on a given landscape.

With a solver, the graph must be connected: the grounded Laplacian of a
disconnected graph is singular, and queries raise. Edges of zero conductance
disconnect a grid graph, as `fun` gives between pixels of zero permeability.
Windowed analyses raise permeability to a floor, so that their graphs are
connected.

Solver tolerances depend on the workload and precision. Check accuracy for your
queries; solver convergence alone does not establish accurate distances. The
measurements behind this guide are on
[issue #35](https://github.com/vboussange/jaxscape/issues/35#issuecomment-6760998974).

## `lineax` solvers

Use a [Lineax](https://docs.kidger.site/lineax/) solver compatible with the sparse symmetric positive-definite operator, such as Conjugate Gradient:

```python
import lineax as lx
from jaxscape import FreeEnergyDistance, ResistanceDistance

solver = lx.CG(rtol=1e-6, atol=1e-6)
resistance = ResistanceDistance(solver=solver)
free_energy = FreeEnergyDistance(theta=0.2, solver=solver)
```

See the [Lineax documentation](https://docs.kidger.site/lineax/) for the full list of available solvers and their options.

## `JAXScape` solvers

`JAXScape` provides a multigrid solver for grid graphs, the default of
`ResistanceDistance`, and three optional solvers as extras:

| Solver | Execution | Typical use and limits |
|---|---|---|
| `GridMultigridSolver` | JAX on CPU or GPU | Conjugate gradients with a geometric multigrid preconditioner for the matrices of `GridGraph`s. Each solve builds its hierarchy inside `jax.jit`. No host step and no extra dependency. |
| `CholmodSolver` | CPU host callback | Direct factorization for fixed sparse systems; efficient reuse when factorization memory fits. GPU arrays cross the host-device boundary. |
| `AMJaxCGSolver` | CPU setup; JAX solve on CPU or GPU | Conjugate gradients with an algebraic multigrid hierarchy, built once on the host, for graphs other than grid graphs whose factorization does not fit. No per-solve host callback. |
| `CuDSSSolver` | NVIDIA GPU, through cuDSS and spineax | Direct factorization inside `jax.jit`, with cost and accuracy independent of conductance contrast. Linux x86-64 and Python 3.12 or newer. |

**Installation:**

```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra amjax      # CG with an AMJax preconditioner
uv add jaxscape --extra cudss      # cuDSS direct solver (NVIDIA GPU, CUDA 13)
```

`PyAMGSolver` is deprecated: `AMJaxCGSolver` builds the same PyAMG hierarchy
once and runs its cycles in JAX.

!!! info "CI/CD coverage"
    The standard CI test suite exercises `GridMultigridSolver`, `CholmodSolver`
    and `AMJaxCGSolver`. CI has no NVIDIA GPU: it runs the sparse direct solver logic
    on a dense solver, and the `CuDSSSolver` tests marked `gpu` need an NVIDIA GPU
    with spineax and cuDSS.

## Algebraic multigrid solver

`AMJaxCGSolver` builds a multigrid hierarchy once, on the host, with PyAMG, and
solves with conjugate gradients preconditioned by its V-cycles in JAX, inside
`jax.jit` and `jax.grad`. The hierarchy keeps the dtype of the matrix. For
ill-conditioned graph Laplacians, enable `JAX_ENABLE_X64=true` and build the
matrix and right-hand sides in `float64`: enabling x64 does not promote
existing `float32` arrays. Tiny graphs, whose hierarchy has a single level,
fall back to plain conjugate gradients.

Build the state once for a matrix:

```python
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO
from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator

A = BCOO.from_scipy_sparse(...)
b = jnp.ones(A.shape[0], dtype=A.data.dtype)

solver = AMJaxCGSolver(rtol=1e-6, max_steps=1_000)
state = solver.init(BCOOLinearOperator(A), {})
x = lx.linear_solve(BCOOLinearOperator(A), b, solver, state=state).value
```

For a matrix with the same sparsity pattern and other values, `reinit` keeps the
hierarchy. Solves take more iterations the further the values are from those
of the hierarchy:

```python
operator = BCOOLinearOperator(A1)
state = solver.reinit(operator, {}, state)
x = lx.linear_solve(operator, b, solver, state=state).value
```

Distances do this for you, as [Reusing solver state](#reusing-solver-state) describes.

## Geometric multigrid solver

`GridMultigridSolver` solves the grounded Laplacians of grid graphs with
conjugate gradients, preconditioned by one V-cycle of a multigrid hierarchy per
iteration. `init` reads the grid, a raster shape and contiguity, from the sparse
indices; `distance.init(graph)` takes them from a `GridGraph`, also inside
`jax.jit`. `reinit` takes the current conductances, and each solve builds their
hierarchy on the grid, so every tile of a windowed analysis is solved with its
own hierarchy.

```python
from jaxscape import ResistanceDistance, SpielmanApproximation
from jaxscape.solvers import GridMultigridSolver

exact = ResistanceDistance()
approximate = ResistanceDistance(
    solver=GridMultigridSolver(rtol=3e-4),
    method=SpielmanApproximation(epsilon=0.15),
)
```

`rtol` bounds the residual that conjugate gradients update, relative to the
right-hand side, and sets the cost. In float32, the residual of the solution
stops decreasing below some level, so a tighter `rtol` adds iterations without
improving the solution. Resistances to pixels behind barriers converge last: a
loose `rtol` degrades the permeability sensitivity there first.

For float32 resistances and their sensitivity on grid graphs, the multigrid
keeps the accuracy of float64. Float32 direct solves are inaccurate around the
ground vertex of large tiles (see the note on grounding of
[`ResistanceDistance`][jaxscape.resistance_distance.ResistanceDistance]); use
`CuDSSSolver` for graphs other than grids, or in float64.

## Reusing solver state

Solver setup can cost far more than a solve, as a sparse analysis or a multigrid hierarchy does. `ResistanceDistance` and `FreeEnergyDistance` share one state lifecycle:

- `distance.init(graph)` builds the state of `graph`. Solvers read concrete
  sparse indices, and some run host code, so call it outside `jax.jit`, except
  for a `GridGraph`: its state builds inside `jax.jit` too, from its shape and
  contiguity.
- `distance.reinit(graph, state)` returns the state of another graph with the same
  sparsity pattern, reusing what the solver can. It runs inside `jax.jit`.
- Queries take that state, and all queries on a graph can share it. A query always
  solves the graph it is given: a state of another graph costs time and leaves the
  distances unchanged.

A loop over conductances carries the state from one step to the next, whatever the solver: `init` returns `None` only with `solver=None`, and `reinit` builds a new state for a solver that keeps nothing from one matrix to the next, such as `lineax.CG`. The state carries no gradient, and `has_aux=True` passes it through the gradient transformation:

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

# GridMultigridSolver: the table below lists what the state of each solver keeps.
distance = ResistanceDistance()
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

What a state keeps from one graph to the next:

| Solver | Kept by the state |
|---|---|
| `GridMultigridSolver` | Its preconditioner, the grid; each solve builds the hierarchy of the new values |
| `AMJaxCGSolver` | Its preconditioner, the multigrid hierarchy of the values of `init` |
| `CuDSSSolver` | The sparse analysis, and the factorization while the values stay the same |
| `CholmodSolver` | Its last factorization, while the values stay the same |
| Lineax solvers | Nothing: each query runs `init`, which only records the matrix |

The graphs of one raster shape and contiguity can also share a
[`GridTopology`][jaxscape.graph.GridTopology], built once outside `jax.jit`:
it holds their edges, and each `GridGraph(surface, fun=fun, topology=topology)`
then computes its edge weights alone.

Windowed analyses build their own states and carry them from tile to tile. See
[`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis].

## Writing a sparse solver

`AbstractSparseSolver` is the Lineax solver of the symmetric positive definite
system matrices of every distance. Its state holds the CSR sparsity pattern of
the matrix, read once from concrete sparse indices, the values, which `reinit`
replaces inside `jax.jit`, and a factorization or a preconditioner. It scales
right-hand sides, checks the sparse indices of each new matrix, and reports
failed solves. Subclass one of its two families.

`AbstractSparseDirectSolver` turns the phases of a sparse factorization library
into a solver, and checks each factorization and solve:

```python
import jax.numpy as jnp
from jaxscape.solvers import AbstractSparseDirectSolver


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

`AbstractSparseIterativeSolver` runs conjugate gradients with a preconditioner.
`preconditioner` builds what the state keeps across values, in `init`, and
`precondition` returns the preconditioner of the current values at each solve,
as a function of residuals with shape `(n, m)`. `matvec` and `vector_shape`
are optional: `GridMultigridSolver` multiplies with its stencil, and takes
vectors as rasters. Jacobi preconditioning keeps nothing:

```python
import jax.numpy as jnp
from jaxscape.solvers import AbstractSparseIterativeSolver


class JacobiCGSolver(AbstractSparseIterativeSolver):
    rtol: float = 1e-5
    max_steps: int = 1_000

    def preconditioner(self, pattern, values):
        return None

    def precondition(self, preconditioner, pattern, values):
        on_diagonal = jnp.where(pattern.rows == pattern.columns, values, 0)
        diagonal = jnp.zeros(pattern.n, values.dtype).at[pattern.rows].add(on_diagonal)
        return lambda residuals: residuals / diagonal[:, None]
```

## Solver reference

::: jaxscape.solvers.gridmultigridsolver.GridMultigridSolver
    options:
      members: false

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

::: jaxscape.solvers.amjaxcgsolver.AMJaxCGSolver
    options:
      members: false

::: jaxscape.solvers.cudsssolver.CuDSSSolver
    options:
      members: false

::: jaxscape.solvers.sparsesolver.AbstractSparseSolver
    options:
      members:
        - reinit

::: jaxscape.solvers.sparsedirectsolver.AbstractSparseDirectSolver
    options:
      members:
        - factorize
        - refactorize
        - solve
        - residual_tolerance

::: jaxscape.solvers.sparseiterativesolver.AbstractSparseIterativeSolver
    options:
      members:
        - preconditioner
        - precondition
        - matvec
        - vector_shape

::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      members: false

## Advanced: `BCOOLinearOperator`

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. It lets compatible Lineax solvers operate on sparse matrices without converting to a dense representation, and is used internally by the solvers above.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
