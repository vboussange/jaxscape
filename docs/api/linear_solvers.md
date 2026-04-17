# Linear solvers

`ResistanceDistance` and `RSPDistance` internally solve one or more sparse linear systems. The choice of solver directly affects both runtime and memory usage.

By default, when no solver is provided, `JAXScape` falls back to dense matrix inversion — this is only suitable for small graphs. For any non-trivial landscape, passing an explicit solver is strongly recommended. `CholmodSolver` is the preferred choice when memory permits.

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

`JAXScape` provides three optional high-performance sparse solvers:

| Solver | Method | Memory | Best for |
|---|---|---|---|
| `CholmodSolver` | Direct Cholesky factorization | Higher | Speed-critical workflows |
| `CuDSSSolver` | GPU direct Cholesky factorization | Higher | GPU-resident SPD solves |
| `PyAMGSolver` | Algebraic multigrid (iterative) | Moderate | Memory-constrained problems |

**Installation**:
```console
uv add jaxscape --extra cholespy   # Cholesky solver
uv add jaxscape --extra pyamg      # PyAMG solver
uv add jaxscape --extra cholespy --extra pyamg  # Both CPU-based Python extras
```

For `CuDSSSolver`, install [`spineax`](https://github.com/johnviljoen/spineax)
with cuDSS support by following its upstream installation instructions.
There is currently no combined JAXScape extra for `CuDSSSolver`, because the
required `spineax` installation depends on your CUDA setup.

!!! info "CI/CD coverage"
    `CuDSSSolver` is covered in CI through CPU-safe mocked tests. The native
    optional backends (`cholespy`, `pyamg`, and GPU-backed `spineax`) are not
    installed in the standard CI environment.

::: jaxscape.solvers.cholmodsolver.CholmodSolver
    options:
      members: false

::: jaxscape.solvers.cudsssolver.CuDSSSolver
    options:
      members: false

::: jaxscape.solvers.pyamgsolver.PyAMGSolver
    options:
      members: false

## Advanced: `BCOOLinearOperator`

`JAXScape` exposes a `lineax`-compatible linear operator that wraps JAX's native `BCOO` sparse matrix format. This allows any Lineax solver to operate directly on sparse matrices without converting to a dense representation, and is used internally by `CholmodSolver`, `CuDSSSolver`, and `PyAMGSolver`.

::: jaxscape.solvers.operator.BCOOLinearOperator
    options:
      members: false
