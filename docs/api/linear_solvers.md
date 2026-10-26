# Linear solvers

`ResistanceDistance` and `FreeEnergyDistance` use linear systems. The choice of solver directly affects runtime and memory usage.

`solver=None` is suitable for small graphs: exact resistance distance uses a dense pseudoinverse, and free-energy distance uses a dense solve. For larger graphs, pass an explicit sparse solver. `CholmodSolver` is a direct CPU option when factorization memory permits; iterative methods trade factorization storage for convergence work.

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

| Solver | GPU | Memory cost | Pros / Cons |
|---|---|---|---|
| `CholmodSolver` | CPU only. Uses an external host callback, so accelerator-backed arrays must cross the host-device boundary. | High | Fastest option on CPU for a fixed sparse SPD system when memory permits. Handles multiple right-hand sides efficiently once inside the direct solve. Main downside is factorization memory, plus the callback introduces a data-transfer bottleneck and prevents a fully on-device GPU solve path. |
| `PyAMGSolver` | CPU only. Built around SciPy CG + PyAMG through an external host callback. | Moderate | Lower memory than direct Cholesky and useful when a direct factorization is too expensive. Main downside is that each solve crosses into Python/NumPy space; batched solves are handled by a Python loop over right-hand sides, so throughput is usually weaker than the other two solvers. |
| `AMJaxCGSolver` | Yes after initialization. The one-time hierarchy build uses PyAMG on CPU, but the initialized CG + AMJax preconditioner solve runs in JAX and can execute on CPU or GPU. | Moderate | Best choice for repeated, JIT-compiled, differentiable solves. Reuses initialized preconditioner state across vmapped batched solves without a per-solve host callback. The preconditioner can be reused while CG state is refreshed for related operators with the same structure. |

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

## Free-energy solver state and accuracy

`FreeEnergyDistance` uses the same solver choices and state conventions as
resistance distance. Its operator is $H=D-B$, where
$D=\operatorname{diag}(A\mathbf1)$ and
$B_{ij}=a_{ij}e^{-\theta/a_{ij}}$ on present edges. This common symmetric
positive-definite system serves every selected endpoint. It is different from
the grounded Laplacian used for resistance distance, so state from one distance
must not be passed to the other.

Initialize full state for repeated queries with an unchanged graph and inverse
temperature:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxscape import FreeEnergyDistance, GridGraph
from jaxscape.solvers import AMJaxCGSolver

permeability = jnp.ones((3, 3), dtype=jnp.float64)
grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
solver = AMJaxCGSolver(rtol=1e-10, atol=1e-12, max_steps=1_000)
distance = FreeEnergyDistance(theta=0.2, solver=solver)
nodes = jnp.array([0, 4, 8])

state = distance.init(grid)
values = distance(grid, nodes=nodes, state=state)
```

Full solver state is bound to both the conductances $A$ and $\theta$. Rebuild
it after changing either; arbitrary stale state is not automatically detected.
With AMJax, preconditioner-only state can be reused across related operators
with the same structure. Fresh CG state is then prepared for the current
conductances and inverse temperature:

```python
preconditioner_state = distance.init_preconditioner(grid)
updated_grid = GridGraph(permeability * 1.1, fun=lambda x, y: (x + y) / 2)
updated_distance = FreeEnergyDistance(theta=0.25, solver=solver)
values = updated_distance(updated_grid, nodes=nodes, state=preconditioner_state)
```

For solvers without a separate preconditioner initializer,
`init_preconditioner` follows the existing full-state initialization convention;
that state remains bound to the unchanged operator.

The free-energy check can reject a solve that met the backend's convergence
criterion. A tiny absolute error in a small positive inverse entry can become
a large error after taking its logarithm. Tighter solver tolerances may help,
but extreme inverse temperatures can remain unresolved. Float64 is recommended
and must be requested explicitly; it is never enabled or substituted by the
library. See the [distance guide](distances.md#numerical-range) for numerical
policy and supported differentiation.

Distance calls return arrays and use normal Lineax/Equinox error behavior.
[Lineax documents](https://docs.kidger.site/lineax/api/linear_solve/) that errors
are reliably raised on CPUs, while GPU failures may only print an error.

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
