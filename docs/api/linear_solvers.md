# Linear solvers

`ResistanceDistance` and `RSPDistance` internally solve one or more sparse linear systems. The choice of solver directly affects both runtime and memory usage.

By default, when no solver is provided, `JAXScape` falls back to dense matrix inversion, which is only suitable for small graphs. For any non-trivial landscape, passing an explicit solver is strongly recommended. `CholmodSolver` is the preferred choice when memory permits.

## `lineax` solvers

`JAXScape` accepts any [Lineax](https://docs.kidger.site/lineax/) solver. For example, to use the Conjugate Gradient method:

```python
import lineax as lx
from jaxscape import ResistanceDistance

solver = lx.CG(rtol=1e-6, atol=1e-6)
distance = ResistanceDistance(solver=solver)
```

See the [Lineax documentation](https://docs.kidger.site/lineax/) for the full list of available solvers and their options.

## `JAXScape` solvers

`JAXScape` provides three optional high-performance solvers as extras:

| Solver | GPU | Memory cost | Pros / Cons |
|---|---|---|---|
| `CholmodSolver` | CPU only. Uses an external host callback, so accelerator-backed arrays must cross the host-device boundary. | High | Fastest option on CPU for a fixed sparse SPD system when memory permits. Handles multiple right-hand sides efficiently once inside the direct solve. Main downside is factorization memory, plus the callback introduces a data-transfer bottleneck and prevents a fully on-device GPU solve path. |
| `PyAMGSolver` | CPU only. Built around SciPy CG + PyAMG through an external host callback. | Moderate | Lower memory than direct Cholesky and useful when a direct factorization is too expensive. Main downside is that each solve crosses into Python/NumPy space; batched solves are handled by a Python loop over right-hand sides, so throughput is usually weaker than the other two solvers. |
| `AMJaxCGSolver` | Yes after initialization. The one-time hierarchy build uses PyAMG on CPU, but the initialized CG + AMJax preconditioner solve runs in JAX and can execute on CPU or GPU. | Moderate | Best choice for repeated, JIT-compiled, differentiable solves. Reuses the initialized preconditioner state across vmapped batched solves without a per-solve host callback. Main downside is the upfront initialization cost and the need to initialize on the exact matrix/operator you plan to reuse. |

**Installation**:
```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra pyamg      # PyAMG solver
uv add jaxscape --extra amjax      # Lineax CG + AMJax preconditioner
```

!!! info "CI/CD coverage"
    These optional solvers are not included in the standard CI test suite.

## Algebraic multigrid solver

`AMJaxCGSolver` separates hierarchy construction from the actual iterative solve. This matters when you want the solve itself to remain JIT-compatible and differentiable.

In algebraic multigrid, the preconditioner is built as a hierarchy of progressively coarser linear systems derived from the original sparse operator. This hierarchy consists of coarse operators together with prolongation and restriction maps between levels; one multigrid cycle smooths the error on the fine level, transfers the residual to coarser levels, approximately solves there, and interpolates the correction back to the fine level.

For direct sparse solves, initialize the solver state once against the matrix shape you plan to reuse:

```python
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxscape.solvers import AMJaxCGSolver, BCOOLinearOperator, linear_solve

A = BCOO.from_scipy_sparse(...)
b = jnp.ones(A.shape[0], dtype=A.data.dtype)

solver = AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
state = solver.init(BCOOLinearOperator(A), {})
x = linear_solve(A, b, solver, state=state)
```

For resistance distance, initialize distance state against the graph once so the solver is initialized on the grounded Laplacian used internally:

```python
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import AMJaxCGSolver

grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
distance = ResistanceDistance(
  solver=AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
)
state = distance.init(grid)

R = distance(grid, state=state)
```

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

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. This allows any Lineax solver to operate directly on sparse matrices without converting to a dense representation, and is used internally by `CholmodSolver`, `PyAMGSolver`, and `AMJaxCGSolver`.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false