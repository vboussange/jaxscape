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

`JAXScape` provides four optional high-performance solvers as extras:

| Solver | GPU | Memory cost | Pros / Cons |
|---|---|---|---|
| `CholmodSolver` | CPU only. Uses an external host callback, so accelerator-backed arrays must cross the host-device boundary. | High | Fastest option on CPU for a fixed sparse SPD system when memory permits. Handles multiple right-hand sides efficiently once inside the direct solve. Main downside is factorization memory, plus the callback introduces a data-transfer bottleneck and prevents a fully on-device GPU solve path. |
| `PyAMGSolver` | CPU only. Built around SciPy CG + PyAMG through an external host callback. | Moderate | Lower memory than direct Cholesky and useful when a direct factorization is too expensive. Main downside is that each solve crosses into Python/NumPy space; batched solves are handled by a Python loop over right-hand sides, so throughput is usually weaker than the other two solvers. |
| `AMJaxCGSolver` | Yes after initialization. The one-time hierarchy build uses PyAMG on CPU, but the initialized CG + AMJax preconditioner solve runs in JAX and can execute on CPU or GPU. | Moderate | Best choice for repeated, JIT-compiled, differentiable solves. Reuses initialized preconditioner state across vmapped batched solves without a per-solve host callback. The preconditioner can be reused while CG state is refreshed for related operators with the same structure. |
| `JAXAMGCGSolver` | NVIDIA GPU only, with CUDA and native AmgX. | Moderate | Lineax CG with a native AMG V-cycle whose setup can refresh inside compiled execution. Supports implicit gradients and changing sparse matrix values. Refresh/reuse follows JAX-AMG; setup may occur on every preconditioner application and is not guaranteed to improve runtime. |

**Installation**:
```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra pyamg      # PyAMG solver
uv add jaxscape --extra amjax      # Lineax CG + AMJax preconditioner
uv add jaxscape --extra jaxamg     # Lineax CG + native JAX-AMG (prerequisites below)
```

!!! info "CI/CD coverage"
    CPU-compatible solvers and JAX-AMG availability checks are exercised in the
    standard CI test suite. JAX-AMG numerical tests explicitly skip when its
    NVIDIA GPU or native backend is unavailable; numerical validation requires
    a separate CUDA/AmgX environment.

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

## Native JAX-AMG with Lineax CG

`JAXAMGCGSolver` uses JAX-AMG's public sparse `make_preconditioner` API with
Lineax CG as the outer solver. The tested compatible upstream release is
[JAX-AMG 0.1.4](https://pypi.org/project/jaxamg/0.1.4/); the optional extra accepts
`>=0.1.4,<0.2`. JAX-AMG and [AMJax](https://pypi.org/project/amjax/) are separate
packages with different setup workflows.

Numerical validation covered float32 and float64 on an NVIDIA RTX 3090 with
JAX-AMG 0.1.4, AmgX 2.5.0, CUDA Toolkit 12.9, and JAX 0.10.0.

JAX-AMG builds from source and needs an NVIDIA GPU, CUDA-enabled JAX, a CUDA
Toolkit (12 or newer), and native NVIDIA AmgX (2.5 or newer). Install AmgX and
configure `CUDA_HOME`, `AMGX_ROOT`, and `AMGX_BUILD` using the
[upstream installation guide](https://jx-wang-s-group.github.io/JAX-AMG/install/)
before installing the extra. The runtime linker must find `libamgxsh.so` and
CUDA libraries, for example through `LD_LIBRARY_PATH`. Choose the JAX CUDA
extra matching your toolkit; for CUDA 12:

```console
pip install 'jax[cuda12]>=0.9.1' 'jaxscape[jaxamg]'
```

Selecting this solver reports missing dependencies or a CPU/non-CUDA backend
as an error. Native library loading is lazy in JAX-AMG, so a missing AmgX/CUDA
library can be reported on the first solve. There is no CPU fallback. Ordinary
JAXScape imports and existing solver defaults do not require this extra.

```python
import jax
import jax.numpy as jnp
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import JAXAMGCGSolver

distance = ResistanceDistance(
    solver=JAXAMGCGSolver(rtol=1e-6, atol=1e-7, max_steps=1_000)
)
nodes = jnp.array([0, 2, 5])

@jax.jit
def predict(permeability):
    graph = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)
    return distance(graph, nodes=nodes)

permeability = jnp.ones((2, 3), dtype=jnp.float32)
R = predict(permeability)
R_changed = predict(1.2 * permeability)
gradient = jax.grad(lambda p: jnp.sum(predict(p)))(permeability)
```

Use connected undirected graphs with positive conductances, or real symmetric
positive definite matrices at the sparse solve interface. Both `float32` and
`float64`, one RHS, and multiple RHS columns are supported. For float64, set
`JAX_ENABLE_X64=1` before starting Python and construct float64 inputs explicitly.
`rtol`, `atol`, `max_steps`, and `stabilise_every` control Lineax CG; sparse solve
options such as `y0` retain their Lineax meaning.

The adapter preserves sparse storage, sums duplicate entries, and replaces
unused BCOO capacity with distinct explicit zero entries before passing CSR to
AmgX. Matrix dimensions and storage shapes remain fixed under JIT while numerical
values change. Each preconditioner application is one classical AMG V-cycle
with matching L1-Jacobi pre/post sweeps and an exact coarse solve. This fixed
configuration preserves the symmetry and positive definiteness required by CG;
arbitrary AMG configurations or replacement preconditioners are not accepted.

**For changing matrices, omit `state`**, as in the example. Both CG and its
preconditioner then use the current values on each compiled call. Full state
from `solver.init(...)` or `distance.init(graph)` is valid only for that fixed
matrix. JAX-AMG does not provide AMJax's reusable preconditioner-only state:
`distance.init_preconditioner(graph)` falls back to full matrix-bound state
and must also be used only with the original matrix.

JAXScape delegates native setup, refresh, and reuse to upstream. The default can
refresh the hierarchy on every preconditioner application, including multiple
applications in one CG solve; there is no once-per-matrix refresh guarantee or
JAXScape cache lifecycle. By contrast, an AMJax inverse workflow can freeze a
CPU-built hierarchy before optimization and bind CG to each current matrix.
Implicit matrix/RHS gradients remain in Lineax and JAXScape's existing solve
machinery; they do not differentiate through native hierarchy construction.

The [inverse landscape genetics example](../examples/inverse_landscape_genetics/inverse_landscape_genetics.ipynb)
selects this workflow with `SOLVER_BACKEND = "jaxamg"`; its default remains
`"amjax"`. The notebook retains float32 model inputs and float64 resistance
calculations. It passes `state=None` for JAX-AMG inside the compiled objective
and reuses only preconditioner state for AMJax. This lets both solvers follow
changing permeability values while preserving their distinct hierarchy behavior.
For either solver, keep the graph dimensions and sparse storage shapes stable
across compiled calls and choose CG tolerances appropriate to your dtype.

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      members: false

::: jaxscape.solvers.amjaxcgsolver.AMJaxCGSolver
    options:
      members: false

::: jaxscape.solvers.jaxamgcgsolver.JAXAMGCGSolver
    options:
      members: false

## Advanced: `BCOOLinearOperator`

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. This allows any Lineax solver to operate directly on sparse matrices without converting to a dense representation, and is used internally by the four optional solvers above.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
