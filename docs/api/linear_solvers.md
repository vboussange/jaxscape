# Linear solvers

`ResistanceDistance` and `FreeEnergyDistance` internally solve linear systems. The choice of solver directly affects runtime and memory usage.

For substantial landscapes, choose an explicit sparse solver:

- On CPU, start with `CholmodSolver` when a sparse factorization fits in memory;
  try `PyAMGSolver` when factorization storage is too costly.
- For JIT-compiled CPU or GPU execution, use Lineax CG, or `AMJaxCGSolver` when
  a reusable multigrid preconditioner helps convergence. AMJax setup runs on CPU.
- `solver=None` remains the default for convenience on small graphs. Its dense
  storage makes it the least favorable option for substantial landscapes.

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

`JAXScape` provides three optional high-performance solvers as extras:

| Solver | Execution | Typical use and limits |
|---|---|---|
| `CholmodSolver` | CPU host callback | Direct factorization for fixed sparse systems; efficient reuse when factorization memory fits. GPU arrays cross the host-device boundary. |
| `PyAMGSolver` | CPU host callback | AMG-preconditioned CG when factorization storage is too costly. Multiple right-hand sides are solved serially. |
| `AMJaxCGSolver` | CPU setup; JAX solve on CPU or GPU | Reusable multigrid preconditioner for repeated solves, with fresh CG state when the operator changes. No per-solve host callback. |

**Installation:**

```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra pyamg      # PyAMG solver
uv add jaxscape --extra amjax      # Lineax CG + AMJax preconditioner
```

!!! info "CI/CD coverage"
    These optional solvers are exercised in the standard CI test suite.

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

For related matrices whose values change but whose geometry is stable, initialize only the preconditioner and materialize fresh CG state for each current operator:

```python
solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
preconditioner_state = solver.init_preconditioner(BCOOLinearOperator(A0), {})

operator = BCOOLinearOperator(A1)
state = solver.materialize_state(operator, {}, preconditioner_state)
x = lx.linear_solve(operator, b, solver, state=state).value
```

For resistance distance, initialize preconditioner state against a reference graph so the AMJax hierarchy is built on the grounded Laplacian used internally:

```python
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver

grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
distance = ResistanceDistance(
  solver=AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
)
state = distance.init_preconditioner(grid)

R = distance(grid, state=state)
```

Use `distance.init(grid)` instead when the graph is fixed and you want to reuse the full solver state.


::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      members: false

::: jaxscape.solvers.amjaxcgsolver.AMJaxCGSolver
    options:
      members: false

## Advanced: `BCOOLinearOperator`

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. It lets compatible Lineax solvers operate on sparse matrices without converting to a dense representation, and is used internally by the three optional solvers above.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
