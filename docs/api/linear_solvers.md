# Linear solvers

`ResistanceDistance` and `FreeEnergyDistance` internally solve linear systems. The choice of solver directly affects runtime, memory usage and accuracy.

## The default solver

`ResistanceDistance()` solves with `GridMultigridSolver()`: conjugate gradients
preconditioned by geometric multigrid, which solves the grounded Laplacians of
grid graphs, such as those of a `GridGraph`. It runs in JAX on CPU and GPU,
inside `jax.jit`, in float32 and float64, and needs no optional dependency. It
solves grids of up to 1024 pixels with one Cholesky factorization, up to
rounding. On grid graphs in float32, it matched or beat the other solvers
measured, in time and accuracy:

| Workload | `GridMultigridSolver` | `CholmodSolver` | `AMJaxCGSolver` | `CuDSSSolver` |
|---|---|---|---|---|
| Inverse landscape genetics, CPU | 2.1 to 2.2 s (1.4 s at `rtol=1e-3`) | 6.05 s | 10.7 to 11.7 s | |
| Inverse landscape genetics, RTX 3090, float32 | 0.24 s (0.15 s at `rtol=1e-3`) | | 0.24 s | 0.89 s |
| Exact resistances on 1000 x 1000 pixels, RTX 3090, float32 | 33 ms, error `6.5e-5`, state in 0.7 s | | fails | 32 ms, error `9.4e-4`, analysis in 6.3 s |
| Exact resistances on 1000 x 1000 pixels, RTX 3090, float64 | 182 ms at `rtol=1e-8` | | 1298 ms | 55 ms |

The inverse landscape genetics benchmark of the repository fits the
resistances among 20 points of a 100 x 100 grid with L-BFGS, in at most 50
steps; every fit reached a relative RMSE of 0.0012, with `AMJaxCGSolver` at
`rtol=atol=1e-3`. The exact resistances are those among 20 nodes, with their
gradient, on a smooth permeability over two decades with a barrier at `1e-3`;
errors are relative to float64. Times are those of warm runs, medians of three
for the inverse benchmark, at the default `rtol=1e-5` unless stated.

Choose another solver in these cases:

- Graphs other than grid graphs, on which the default raises. On CPU, use
  `CholmodSolver` when a sparse factorization fits in memory, and
  `PyAMGSolver` when factorization storage is too costly. On an NVIDIA GPU, use
  `CuDSSSolver` when a sparse Cholesky factorization fits in GPU memory. For
  other JIT-compiled CPU or GPU execution, use Lineax CG, or `AMJaxCGSolver`
  when a reusable algebraic multigrid preconditioner helps convergence; AMJax
  setup runs on CPU.
- Float64 on an NVIDIA GPU: `CuDSSSolver` took 55 ms where the multigrid took
  182 ms in the table above.
- Small graphs of any kind: `solver=None` takes the pseudo-inverse of the dense
  Laplacian for exact all-pairs and nodes-to-nodes queries, and a dense
  Cholesky factorization of the grounded Laplacian for other queries. Its
  memory grows with the square of the number of vertices. In float32, the
  pseudo-inverse loses accuracy: on a grid of 24 x 40 pixels with conductances
  over two decades, its all-pairs resistances had a relative error of 40 %
  against float64, and those of the default `2.6e-6`. Use it in float64.
- Exact windowed analyses on a CPU. Their window queries solve for every
  vertex of a tile, and a direct solver reuses one factorization for all of
  them, where conjugate gradients iterate for each.

The exact windowed analyses below ran on a smooth raster of 200 x 200 pixels,
with `dependency_range=20`, `coarsening_factor=0.25` and `batch_size=64`: 25
tiles of 71 x 71 pixels. Times are those of a second run, for the connectivity
and the permeability sensitivity, on two AMD EPYC 7742 and on an RTX 3090:

| Exact windowed analysis | Default | `solver=None` | `CholmodSolver` | `CuDSSSolver` |
|---|---|---|---|---|
| CPU, float64 | 103 s, 223 s | 52 s, 109 s | 44 s, 109 s | |
| CPU, float32 | 66 s, 140 s | 21 s, 48 s | | |
| RTX 3090, float32 | 1.5 s, 3.0 s | 1.7 s, 3.1 s | | 2.4 s, 4.9 s |
| RTX 3090, float64 | 5.9 s, 11.9 s | 26.6 s, 49.7 s | | 3.4 s, 7.1 s |

With `dependency_range=10`, no coarsening and tiles of 28 x 28 pixels, which
the default solves with one Cholesky factorization, it took 14 s and 24 s in
float64 on the CPU, against 6.2 s and 20 s with `solver=None` and 30 s and
43 s with `CholmodSolver`.

With a solver, the graph must be connected: the grounded Laplacian of a
disconnected graph is singular, and queries raise. Edges of zero conductance
disconnect a grid graph, as `fun` gives between pixels of zero permeability.
Windowed analyses raise permeability to a floor, so that their graphs are
connected.

`FreeEnergyDistance()` solves with dense solves by default, `solver=None`,
which suits small graphs; it takes the same solvers.

Solver tolerances depend on the workload and precision. Check accuracy for your
queries; solver convergence alone does not establish accurate distances.

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
`ResistanceDistance`, and four optional high-performance solvers as extras:

| Solver | Execution | Typical use and limits |
|---|---|---|
| `GridMultigridSolver` | JAX on CPU or GPU | Conjugate gradients with a geometric multigrid preconditioner for the matrices of `GridGraph`s, whose hierarchy each solve builds inside `jax.jit` in milliseconds. No host step and no extra dependency. |
| `CholmodSolver` | CPU host callback | Direct factorization for fixed sparse systems; efficient reuse when factorization memory fits. GPU arrays cross the host-device boundary. |
| `PyAMGSolver` | CPU host callback | AMG-preconditioned CG when factorization storage is too costly. Multiple right-hand sides are solved serially. |
| `AMJaxCGSolver` | CPU setup; JAX solve on CPU or GPU | Reusable multigrid preconditioner for repeated solves, with fresh CG state when the operator changes. No per-solve host callback. |
| `CuDSSSolver` | NVIDIA GPU, through cuDSS and spineax | Direct factorization inside `jax.jit`, with cost and accuracy independent of conductance contrast. Linux x86-64 and Python 3.12 or newer. |

**Installation:**

```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra pyamg      # PyAMG solver
uv add jaxscape --extra amjax      # Lineax CG + AMJax preconditioner
uv add jaxscape --extra cudss      # cuDSS direct solver (NVIDIA GPU, CUDA 13)
```

!!! info "CI/CD coverage"
    The standard CI test suite exercises `GridMultigridSolver`, `CholmodSolver`,
    `PyAMGSolver` and `AMJaxCGSolver`. CI has no NVIDIA GPU: it runs the sparse direct solver logic
    on a dense solver, and the `CuDSSSolver` tests marked `gpu` need an NVIDIA GPU
    with spineax and cuDSS.

## Algebraic multigrid solver

`AMJaxCGSolver` separates hierarchy construction from the actual iterative solve. This matters when you want the solve itself to remain JIT-compatible and differentiable.

The `amjax` extra installs AMJax 0.0.3 or newer and PyAMG. JAXScape delegates
multigrid construction and cycles to AMJax's `MultilevelSolver`, and iteration
and convergence checks to Lineax CG. Hierarchy arrays retain the input matrix's
dtype. For ill-conditioned graph Laplacians, enable `JAX_ENABLE_X64=true` and
explicitly construct the matrix and right-hand sides in `float64`; enabling
x64 alone does not promote existing `float32` arrays. Single-level hierarchies
(typically tiny graphs) fall back to ordinary CG because AMJax 0.0.3 requires
at least two levels for a multigrid cycle.

In algebraic multigrid, the preconditioner is built as a hierarchy of progressively coarser linear systems derived from the original sparse operator. This hierarchy consists of coarse operators together with prolongation and restriction maps between levels; one multigrid cycle smooths the error on the fine level, transfers the residual to coarser levels, approximately solves there, and interpolates the correction back to the fine level.

For direct sparse solves against an unchanged matrix, initialize the full solver state once:

```python
import jax.numpy as jnp
import lineax as lx
from jax.experimental.sparse import BCOO
from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator

A = BCOO.from_scipy_sparse(...)
b = jnp.ones(A.shape[0], dtype=A.data.dtype)

solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
state = solver.init(BCOOLinearOperator(A), {})
x = lx.linear_solve(BCOOLinearOperator(A), b, solver, state=state).value
```

For a matrix with the same sparsity pattern and other values, `reinit` keeps the hierarchy and binds CG to the new matrix:

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
indices, which `distance.init(graph)` takes from the shape and contiguity of a
`GridGraph`, also inside `jax.jit`. `reinit` scatters the current conductances
into the stencil of the grid inside `jax.jit`. Each solve builds the multigrid hierarchy of these
conductances, in 2.3 ms for the 894 x 925 tile below on an RTX 3090, so that
every tile of a windowed analysis is solved with its own hierarchy.

```python
from jaxscape import ResistanceDistance, SpielmanApproximation
from jaxscape.solvers import GridMultigridSolver

exact = ResistanceDistance()  # GridMultigridSolver(rtol=1e-5, max_steps=200)
approximate = ResistanceDistance(
    solver=GridMultigridSolver(rtol=3e-4),
    method=SpielmanApproximation(epsilon=0.15),
)
```

`rtol` bounds the residual that conjugate gradients update, relative to the
right-hand side, and sets the cost: on the 894 x 925 tile of the Maryland land
cover of the connectivity analysis example, with conductances from `1e-4` to
1, 64 right-hand sides took 8 iterations at `rtol=1e-3`, 13 at `3e-4` and 27 at
`1e-5`: solves of 63, 106 and 201 ms with their hierarchy, on an RTX 3090 in
float32. In float32, the residual of the solution stops decreasing: on a test
grid, it stayed at `2e-5` to `6e-4` relative for every `rtol` from `3e-4` to
`1e-5` (see the solver reference below). Resistances to pixels behind barriers
converge last: at `rtol=3e-3`, the permeability sensitivity of
`SpielmanApproximation(epsilon=0.15)` had a Pearson correlation of 0.48 with
float64.

The example runs six analyses in float32: the connectivity, the permeability
sensitivity and the permeability sensitivity of a flat landscape, for
`LCPDistance` and for `ResistanceDistance` with
`SpielmanApproximation(epsilon=0.15)` on one tile. On an RTX 3090, with each
solver for the resistances:

| Six analyses of the example | `GridMultigridSolver(rtol=3e-4)` | `CuDSSSolver` |
|---|---|---|
| Time once compiled | 10.3 s | 9.5 s |
| Time in a fresh process | 33.7 s | 33.6 s |
| XLA compilation in a fresh process | 21.0 s | 16.2 s |
| Pearson correlation of the normalized map with float64 | 1.0000 | 0.9838 |
| 99.9th percentile of the per-pixel ratio to float64 | 1.18 | 8.96 |
| Largest ratio to float64 around the ground vertex | 6.3 | 1204 |

The normalized map, which the example draws, is the permeability sensitivity
divided by that of the flat landscape, and the float64 reference drew the same
random signs. With `CuDSSSolver` in float32, the permeability sensitivity is up
to 1204 times that of float64 around the ground vertex of the tile, a spot
visible on the log-scale maps (see the note on grounding of
[`ResistanceDistance`][jaxscape.resistance_distance.ResistanceDistance]).
The two solvers take as long in a fresh process: `CuDSSSolver` compiles for 5 s
less, and spends about as long on the sparse analysis of the tile. Once
compiled, `CuDSSSolver` takes 0.8 s less. Choose `GridMultigridSolver` for
float32 resistances and their sensitivity on grid graphs, as the example does:
its normalized map matches that of float64. Choose `CuDSSSolver` for graphs
other than grids, or in float64.

## Reusing solver state

Solver setup can cost far more than a solve, as a sparse analysis or a multigrid hierarchy does. `ResistanceDistance` and `FreeEnergyDistance` share one state lifecycle:

- `distance.init(graph)` builds the state of `graph`. Some solvers read concrete
  sparse indices or run host code, so call it outside `jax.jit`.
  `GridMultigridSolver` builds the state of a `GridGraph` inside `jax.jit` too,
  from its shape and contiguity.
- `distance.reinit(graph, state)` returns the state of another graph with the same
  sparsity pattern, reusing what the solver can. It runs inside `jax.jit`.
- Queries take that state, and all queries on a graph can share it. A query always
  solves the graph it is given: a state of another graph costs time and leaves the
  distances unchanged.

A loop over conductances carries the state from one step to the next. The state carries no gradient, and `has_aux=True` passes it through the gradient transformation:

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
| `GridMultigridSolver` | The grid; each solve builds the hierarchy of the new values |
| `AMJaxCGSolver` | The multigrid hierarchy |
| `CuDSSSolver` | The sparse analysis, and the factorization while the values stay the same |
| `CholmodSolver` | Its last factorization, while the values stay the same |
| `PyAMGSolver`, Lineax solvers | Nothing: each query runs `init`, which only records the matrix |

Windowed analyses carry states for you. Within a process, they keep the
latest state of each distance, graph shape and dtype: a windowed analysis
builds a state only for a shape without one, and each run starts from the
latest state and leaves its own. With `CuDSSSolver`, a run then costs one
refactorization per tile, and the cuDSS analysis of a sparsity pattern runs
once for all the windowed analyses and runs that share it. See
[`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis].

## Writing a sparse direct solver

`AbstractSparseDirectSolver` turns the phases of a sparse factorization library into a Lineax solver for symmetric positive definite matrices, usable by every distance. It provides the state lifecycle, the CSR sparsity pattern, and the checks of each factorization and solve.

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

The phases run inside `jax.jit`. Set `residual_tolerance` from the residuals the library leaves on valid matrices of your workload.

A solver of another kind whose state outlives the matrix values subclasses `AbstractReusableStateSolver` and implements `reinit`.

## Solver reference

::: jaxscape.solvers.gridmultigridsolver.GridMultigridSolver
    options:
      members: false

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      members: false

::: jaxscape.solvers.amjaxcgsolver.AMJaxCGSolver
    options:
      members: false

::: jaxscape.solvers.cudsssolver.CuDSSSolver
    options:
      members: false

::: jaxscape.solvers.sparsedirectsolver.AbstractSparseDirectSolver
    options:
      members:
        - factorize
        - refactorize
        - solve
        - residual_tolerance
        - reinit

::: jaxscape.solvers.operator.AbstractReusableStateSolver
    options:
      members:
        - reinit

## Advanced: `BCOOLinearOperator`

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. It lets compatible Lineax solvers operate on sparse matrices without converting to a dense representation, and is used internally by the solvers above.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
