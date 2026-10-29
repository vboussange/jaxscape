# Linear solvers

`ResistanceDistance` and `FreeEnergyDistance` internally solve linear systems. The choice of solver directly affects runtime and memory usage.

For substantial landscapes, choose an explicit sparse solver:

- On CPU, start with `CholmodSolver` when a sparse factorization fits in memory;
  try `PyAMGSolver` when factorization storage is too costly.
- On an NVIDIA GPU, use `CuDSSSolver` for `ResistanceDistance` when a sparse
  Cholesky factorization fits in GPU memory. It runs inside `jax.jit`, its
  cost and accuracy stay the same however heterogeneous the conductances are,
  and an optimization loop passes its state from one step to the next. See
  [GPU direct solver: cuDSS](#gpu-direct-solver-cudss).
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
| `CuDSSSolver` | NVIDIA GPU only, through cuDSS and spineax | Sparse Cholesky for `ResistanceDistance`, with cost and accuracy independent of conductance contrast. Analysis once per graph geometry, refactorization when conductances change, one multi-RHS solve per query, in float32 or float64. Failed factorizations are reported through the Lineax result. Linux x86-64 and Python 3.12 or newer; the factorization must fit in GPU memory. |

**Installation:**

```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra pyamg      # PyAMG solver
uv add jaxscape --extra amjax      # Lineax CG + AMJax preconditioner
uv add jaxscape --extra cudss      # cuDSS direct solver (NVIDIA GPU, CUDA 13)
```

The `cudss` extra pairs with JAX's CUDA 13 plugin;
[Installing cuDSS support](#installing-cudss-support) also covers CUDA 12.

!!! info "CI/CD coverage"
    The standard CI test suite exercises `CholmodSolver`, `PyAMGSolver` and
    `AMJaxCGSolver`. CI has no NVIDIA GPU: it runs the `CuDSSSolver` tests
    against a CPU stand-in for spineax's phase API and skips their GPU
    variants, which need an NVIDIA GPU with spineax and cuDSS.

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

## GPU direct solver: cuDSS

`CuDSSSolver` computes resistance distances, exact or with
`SpielmanApproximation`, on an NVIDIA GPU with the sparse Cholesky
factorization of [NVIDIA cuDSS](https://developer.nvidia.com/cudss),
through the analysis, factorization, refactorization and solve phases of
[spineax](https://github.com/johnviljoen/spineax). Pass it to
`ResistanceDistance` like any other solver; it runs inside `jax.jit`:

- `distance.init(graph)` runs cuDSS analysis, which depends only on the
  sparsity pattern, and a first factorization;
- each query refactorizes for the current conductances;
- all right-hand sides of a query share that factorization in one multi-RHS
  solve.

Choose it when you compute exact resistance distances repeatedly on a GPU, for
example at every step of an inverse landscape genetics optimization, and the
factorization fits in GPU memory. A direct solve has no iteration budget and
no preconditioner, so its cost and accuracy stay the same as conductance
contrast grows. By comparison, CG preconditioned by an `AMJaxCGSolver`
hierarchy built for a uniform surface needs about 1100 iterations at 4 orders
of magnitude of contrast and about 10⁴ at 6.

Gradients never pass through analysis or factorization:

- gradients of exact resistance distances reuse the forward solutions, so a
  gradient step costs one refactorization and one multi-RHS solve;
- gradients of Spielman approximate distances solve the adjoint system with
  the forward factorization, because the grounded Laplacian is symmetric: the
  backward pass adds one multi-RHS solve and no factorization;
- gradients of sparse linear solves with `BCOOLinearOperator`, with respect to
  matrix values and right-hand sides, also solve the adjoint system with the
  forward factorization.

!!! note "Evidence"
    The [training-time solver report](https://github.com/vboussange/jaxscape/blob/0564df90691064ec3bb742a3bb8ff9bc03c51216/docs/benchmark_runs/training-solver-2026-10-29/index.md)
    benchmarked spineax's phases directly on an RTX 3090 in float64, with the
    state passed from step to step:

    - refactorizing and solving 20 right-hand sides took 1.9 ms, 11 ms and
      0.11 s on 100×100, 300×300 and 1000×1000 grids, on every surface up to 6
      orders of magnitude of contrast and a 10⁻⁶ conductance floor;
    - resistances agreed with CHOLMOD to within 2.3·10⁻⁹ (relative), and
      gradients with finite differences to within 4.5·10⁻⁶;
    - analysis alone took 4.7 s at 1000×1000, which is why the state is passed
      on;
    - an AMJax hierarchy rebuilt for each surface needed 0.65–1.22 s per solve
      at 1000×1000, plus about 3 s of host rebuild.

    The report measured neither JAXScape's value mapping and state handling
    nor memory use.

### Precision

`CuDSSSolver` factorizes and solves in the dtype of the conductances, float32
or float64, and a state keeps the dtype it was initialized with. Float64
needs `jax.config.update("jax_enable_x64", True)` and float64 conductances.

- **float64** gives resistance distances to direct-solver precision, within
  2.3·10⁻⁹ of CHOLMOD on every surface of the evidence report.
- **float32** halves the memory of factor values and solutions, and
  conductance contrast consumes its seven significant digits. In CPU
  measurements of float32 Cholesky solves, the relative error of resistance
  distances grew from about 10⁻⁵ on small grids to 10⁻² on 300×300 grids at
  6 orders of magnitude of contrast, and reached 10⁻¹ with a 10⁻⁶ conductance
  floor. Use float64 for high-contrast surfaces. A 10⁻⁶ conductance is also
  within float32 rounding of a zero conductance relative to unit ones, so
  float32 cannot reliably tell vertices joined through such edges from
  vertices cut off from the ground vertex (see
  [Failed factorizations](#failed-factorizations)).

### Failed factorizations

A Cholesky factorization fails when the grounded Laplacian is not positive
definite, for example with a negative conductance or a vertex whose
conductances are all zero, or is singular, for example when zero
conductances cut vertices off from the ground vertex, the graph's last
vertex. spineax reports no factorization status. On an RTX 3090, a cuDSS
factorization that meets a non-positive pivot left finite garbage or NaN,
and poisoned its cuDSS handle: every later refactorization on that handle
gave the same garbage, whatever the conductances.

`CuDSSSolver` therefore checks its results twice:

- **Each factorization**, in `init` and at each query, solves `A x = A 1` with
  the new factorization. It fails when the backward error
  `max|A 1 - A x| / (‖A‖∞ max|x|)` is not finite or exceeds the square root
  of the machine epsilon: 3.5·10⁻⁴ in float32 and 1.5·10⁻⁸ in float64. The
  state records the failure; the next query that refreshes it analyses the
  matrix afresh on a new cuDSS handle, so one bad step leaves the following
  steps exact.
- **Each solve** checks its solution `x` of a right-hand side `b`: it fails
  when `x` is not finite, or when `max|b - A x|` exceeds `max|b|` times 10⁻⁴
  in float64 or 10⁻¹ in float32. This catches singular matrices whose
  factorization passes its check, for the right-hand sides they make
  inconsistent.

A failed solve, including every solve with a failed factorization, returns
NaN values and reports `lineax.RESULTS.singular` through Lineax's result:

- resistance-distance queries solve with `throw=True` and raise, eagerly and
  inside `jax.jit`, for the state from `init` and after a query refactorizes
  for new conductances. On GPUs, Lineax may only print the error, and the
  failed distances are then NaN;
- a direct Lineax solve with `throw=False` returns the failed result.

```python
import lineax as lx
from jaxscape.solvers import BCOOLinearOperator, CuDSSSolver

operator = BCOOLinearOperator(A)  # A: a BCOO grounded Laplacian
solver = CuDSSSolver()
state = solver.init(operator, {})
solution = lx.linear_solve(operator, b, solver, state=state, throw=False)
failed = solution.result != lx.RESULTS.successful
```

The factorization check costs one solve of one right-hand side and two
sparse matrix-vector products per factorization; the solve check costs one
sparse matrix-vector product per right-hand side. Their thresholds come from
measurements on an RTX 3090 with grounded grid Laplacians of 100×100,
300×300 and 1000×1000 cells, smooth or with up to 6 orders of magnitude of
contrast and a straight barrier at a 10⁻⁶ conductance floor:

| Measurement | float32 | float64 |
|---|---|---|
| Factorization backward error, valid matrices | at most 1.1·10⁻⁶ | at most 1.0·10⁻¹⁵ |
| Factorization backward error, non-positive pivot | at least 3.9·10⁻², or not finite | at least 3.9·10⁻², or not finite |
| Solve residual of single-vertex currents, valid matrices | at most 6.6·10⁻³ | at most 1.2·10⁻¹¹ |
| Solve residual, vertices cut off from the ground vertex | 1.5·10⁻³ to 0.5 | 3·10⁻² to 0.16 |

In float64, both checks separate valid and failed matrices by orders of
magnitude. In float32, a factorization that meets a non-positive pivot is
always caught, but vertices cut off from the ground vertex can pass the
solve check on large grids: their residuals overlap those of valid
high-contrast matrices. Use float64 when conductances can be zero.

### Installing cuDSS support

spineax publishes a single wheel, for CUDA 13 on Linux x86-64 with Python 3.12
or newer. The `cudss` extra installs it with its cuDSS runtime:

```console
uv add jaxscape --extra cudss
```

- **JAX plugin.** The spineax wheel requires `jax[cuda13]`, so the extra also
  installs JAX's CUDA 13 plugin. JAXScape's `cuda` extra installs JAX's CUDA 12
  plugin; leave it out, so that the environment holds one CUDA plugin.
- **cuDSS pin.** spineax 0.0.5 links against cuDSS 0.x and fails to load with
  cuDSS 1.0. The extra pins `nvidia-cudss-cu13>=0.8,<1` until spineax supports
  cuDSS 1.0 ([spineax#29](https://github.com/johnviljoen/spineax/issues/29),
  [spineax#30](https://github.com/johnviljoen/spineax/pull/30)). Keep the same
  pin when you install spineax another way.
- **Other platforms.** On other platforms and Python versions, the extra
  installs nothing and `CuDSSSolver()` raises an `ImportError` naming it.

#### Building spineax against CUDA 12

To keep a CUDA 12 JAX stack, build spineax from source and leave out the
`cudss` extra, whose wheel brings in CUDA 13:

```bash
# A Python 3.12 (or newer) environment on Linux x86-64.
pip install 'jaxscape[cuda]'  # JAX 0.11 or newer with its CUDA 12 plugin
pip install 'cmake>=3.26' nanobind scikit-build-core 'nvidia-cudss-cu12>=0.8,<1'
git clone https://github.com/johnviljoen/spineax.git
git -C spineax checkout 97dae03f54c3d0698c9a714471cc64f471b5aedd  # v0.0.5
pip install --no-build-isolation --no-deps ./spineax
```

- `--no-deps` keeps spineax's requirements, `jax[cuda13]` and
  `nvidia-cudss-cu13`, from replacing the CUDA 12 JAX stack.
- `--no-build-isolation` compiles against the CUDA 12 and cuDSS libraries of
  this environment. An isolated build installs spineax's CUDA 13 build
  requirements and compiles against those.
- The build compiles C++ only and needs no `nvcc`. It took about 45 s in the
  evidence run, and spineax's 63 tests passed with it.
- With uv, run the same commands as `uv pip install ...`.

### Requirements and tested versions

- **GPU.** An NVIDIA GPU of Turing generation (compute capability 7.5) or
  newer, as spineax requires.
- **Driver.** A driver that supports the CUDA version of your JAX plugin.
  JAX's [installation guide](https://docs.jax.dev/en/latest/installation.html)
  requires driver 580 or newer for CUDA 13, and 525 or newer for CUDA 12, on
  Linux.
- **Platform.** Linux x86-64 and Python 3.12 or newer.
- **Backend.** `CuDSSSolver()` checks that JAX's default backend is an NVIDIA
  GPU and raises a `RuntimeError` on any other backend. It raises an
  `ImportError` when spineax is missing or cannot load the cuDSS runtime. It
  has no CPU fallback.

The evidence report ran the CUDA 12 build:

| Component | Tested version |
|---|---|
| spineax | 0.0.5, built from [`97dae03f`](https://github.com/johnviljoen/spineax/commit/97dae03f54c3d0698c9a714471cc64f471b5aedd) (the `v0.0.5` tag) |
| cuDSS | `nvidia-cudss-cu12` 0.8.0.10 |
| JAX | 0.10.0 with its CUDA 12 plugin |
| Lineax and Equinox | 0.1.1 and 0.13.8 |
| Python | 3.12 |
| GPU and driver | NVIDIA GeForce RTX 3090 (24 GiB), driver 580.178.04 |

JAXScape requires JAX 0.11 or newer, so a CUDA 12 build today pairs spineax
with JAX 0.11 or newer and its CUDA 12 plugin. The `cudss` extra installs the
CUDA 13 wheel of the same spineax release, with `nvidia-cudss-cu13` below 1.0.
That combination follows spineax's published requirements; the evidence run did
not include it.

### GPU memory

cuDSS allocates its factorizations outside JAX's memory pool, and JAX
preallocates 75 % of GPU memory by default. Leave room for the factors by
disabling preallocation, as the evidence runs did, or by lowering
`XLA_PYTHON_CLIENT_MEM_FRACTION`:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

spineax keeps the most recently used factorizations in device memory, 8 per
process by default; the `SPINEAX_FACTOR_CACHE` environment variable sets the
number.

### Passing the state from step to step

A `CuDSSSolver` state holds spineax's factorization for one sparsity pattern:

1. **Initialization.** `distance.init(graph)` derives the sparsity pattern of
   the grounded Laplacian, then runs analysis and factorization. It reads
   concrete sparse indices, so call it outside `jax.jit`. Analysis is the
   costly phase: 4.7 s at 1000×1000 in the evidence run, against 70 ms for a
   refactorization.
2. **Refactorization.** Each query refactorizes the state it receives for the
   current conductances, and solves with the new state. This uses up the
   factorization of the state it receives, because cuDSS handles cannot be
   copied ([spineax#27](https://github.com/johnviljoen/spineax/issues/27)).
3. **Passing it on.** The state-returning queries,
   `all_pairs_distance_and_state`, `nodes_to_nodes_distance_and_state` and
   `sources_to_targets_distance_and_state`, return the distances and the new
   state. Pass that state to the next call.

The new state has the structure of the state passed in, so a compiled step
called with new conductances and the returned state reuses one executable and
never repeats the analysis. The returned state carries no gradient, and
`has_aux=True` passes it through `jax.value_and_grad`:

```python
# Runs on an NVIDIA GPU with the `cudss` extra installed.
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jaxscape import GridGraph, ResistanceDistance
from jaxscape.solvers import CuDSSSolver


def landscape(log_conductance):
    return GridGraph(jnp.exp(log_conductance), fun=lambda x, y: (x + y) / 2)


shape = (300, 300)
key_surface, key_sites = jax.random.split(jax.random.key(0))
true_log_conductance = 2.0 * jax.random.normal(key_surface, shape)
sites = jax.random.choice(key_sites, shape[0] * shape[1], (20,), replace=False)

distance = ResistanceDistance(solver=CuDSSSolver())
log_conductance = jnp.zeros(shape)
# Analysis and factorization, once, outside `jax.jit`.
state = distance.init(landscape(log_conductance))
# Eager queries also return the refactorized state.
target, state = distance.nodes_to_nodes_distance_and_state(
    landscape(true_log_conductance), sites, state
)


def loss(log_conductance, state):
    predicted, state = distance.nodes_to_nodes_distance_and_state(
        landscape(log_conductance), sites, state
    )
    return jnp.mean((predicted - target) ** 2), state


@jax.jit
def step(log_conductance, state):
    (value, state), gradient = jax.value_and_grad(loss, has_aux=True)(
        log_conductance, state
    )
    return log_conductance - 0.1 * gradient, value, state


for _ in range(100):
    log_conductance, value, state = step(log_conductance, state)
```

### When analysis repeats

Analysis runs again, and a call costs about as much as `init`, when:

- **a query receives a used-up state**, such as the state from `init` after
  its first query, or any state already passed to a query. Query forms that
  return distances only, such as `distance(graph, state=state)`, discard the
  refactorized state, so each further call with the same `state` repeats the
  analysis;
- **spineax evicted the factorization**, because more recently used
  factorizations filled its cache. A loop that passes on more states than the
  cache holds, for example one per region, needs a larger
  `SPINEAX_FACTOR_CACHE`;
- **an eager query receives no state**, and initializes one, as
  `distance(graph)` and the `*_distance_and_state` forms do. A compiled
  query without a state raises a `ValueError` asking for
  `distance.init(graph)`;
- **a query receives a state whose factorization failed its check**, and
  analyses the matrix afresh, because cuDSS leaves the handle of such a
  factorization unusable.

A used-up or evicted state still gives exact distances: spineax repeats the
analysis and factorization for the current conductances. A mistake in passing
the state costs time and leaves the results unchanged.

A state belongs to the sparsity pattern it was initialized for: for a
`GridGraph`, the raster shape and contiguity, with any conductances. A query
whose grounded Laplacian has other sparse indices than the state raises an
error, so initialize a new state whenever the geometry changes.

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

::: jaxscape.solvers.operator.AbstractRefreshingSolver
    options:
      members:
        - refresh_state

## Advanced: `BCOOLinearOperator`

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. It lets compatible Lineax solvers operate on sparse matrices without converting to a dense representation, and is used internally by the four optional solvers above.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
