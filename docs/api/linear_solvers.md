# Linear solvers

`ResistanceDistance` and `FreeEnergyDistance` internally solve linear systems. The choice of solver directly affects runtime and memory usage.

For substantial landscapes, choose an explicit sparse solver:

- On CPU, start with `CholmodSolver` when a sparse factorization fits in memory;
  try `PyAMGSolver` when factorization storage is too costly.
- On an NVIDIA GPU, use `CuDSSSolver` when a sparse Cholesky factorization fits
  in GPU memory.
- For other JIT-compiled CPU or GPU execution, use Lineax CG, or
  `AMJaxCGSolver` when a reusable multigrid preconditioner helps convergence.
  AMJax setup runs on CPU.
- The default, `solver=None`, uses a dense solve and is suited to small
  graphs only.

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

`JAXScape` provides four optional high-performance solvers as extras:

| Solver | Execution | Typical use and limits |
|---|---|---|
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
    The standard CI test suite exercises `CholmodSolver`, `PyAMGSolver` and
    `AMJaxCGSolver`. CI has no NVIDIA GPU: it runs the sparse direct solver logic
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

## Reusing solver state

Solver setup can cost far more than a solve, as a sparse analysis or a multigrid hierarchy does. `ResistanceDistance` and `FreeEnergyDistance` share one state lifecycle:

- `distance.init(graph)` builds the state of `graph`. Some solvers read concrete
  sparse indices or run host code, so call it outside `jax.jit`.
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
from jaxscape.solvers import AMJaxCGSolver


def landscape(log_conductance):
    return GridGraph(jnp.exp(log_conductance), fun=lambda x, y: (x + y) / 2)


sites = jnp.array([0, 45, 99])
true_log_conductance = jax.random.normal(jax.random.key(0), (10, 10))
target = ResistanceDistance()(landscape(true_log_conductance), nodes=sites)

distance = ResistanceDistance(
    solver=AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
)
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
| `AMJaxCGSolver` | The multigrid hierarchy |
| `CuDSSSolver` | The sparse analysis, and the factorization while the values stay the same |
| `CholmodSolver` | Its last factorization, while the values stay the same |
| `PyAMGSolver`, Lineax solvers | Nothing: each query runs `init`, which only records the matrix |

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

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. It lets compatible Lineax solvers operate on sparse matrices without converting to a dense representation, and is used internally by the four optional solvers above.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
