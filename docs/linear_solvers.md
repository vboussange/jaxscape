# Linear solvers

`{ResistanceDistance, RSPDistance}.__call__` require a linear solve operation, which significantly impacts both memory usage and computational speed. `JAXScape` provides two solver options:

- **PyAMG** (`PyAMGSolver`): Algebraic multigrid solver with moderate memory footprint
- **Cholesky** (`CholmodSolver`): Faster but requires more memory

**Installation**:
```console
uv add jaxscape --extra pyamg      # Install PyAMG solver
uv add jaxscape --extra cholespy   # Install Cholesky solver
uv add jaxscape --extra solvers    # Install both solvers
```

!!! info "Optional solvers support"
    These optional solvers are not tested in CI/CD. 

**Default behavior:** If no solver is specified, JAXScape falls back to matrix inversion, which is very inefficient for large graphs (>5k nodes). Always specify a solver for production use.