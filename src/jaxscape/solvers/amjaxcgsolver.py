"""Conjugate gradients with an AMJax algebraic multigrid preconditioner."""

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import numpy as np
import scipy.sparse
from jax import errors as jax_errors
from jaxtyping import Array

from .sparseiterativesolver import AbstractSparseIterativeSolver, csr_matvec
from .sparsesolver import CSRPattern


try:
    import pyamg as _pyamg
    from amjax import MultilevelSolver as _MultilevelSolver

    AMJAX_AVAILABLE = True
except ImportError:
    _pyamg = None
    _MultilevelSolver = None
    AMJAX_AVAILABLE = False


@final
class _Hierarchy(eqx.Module):
    """The preconditioner of an `AMJaxCGSolver` state: an AMJax multigrid
    hierarchy, as its leaves and tree structure.

    AMJax's hierarchies are pytrees whose children depend on which of their
    leaves are `None`, so that filters such as `eqx.partition`, which replace
    leaves with `None`, change their structure. A list of leaves keeps it.
    """

    leaves: list[Array]
    treedef: Any = eqx.field(static=True)

    @property
    def multilevel_solver(self) -> Any:
        """The AMJax `MultilevelSolver`."""
        return jax.tree.unflatten(self.treedef, self.leaves)


@final
class AMJaxCGSolver(AbstractSparseIterativeSolver[_Hierarchy]):
    """Conjugate gradients preconditioned by algebraic multigrid, with AMJax.

    Its preconditioner is a multigrid hierarchy, which `init` builds once on the
    host with PyAMG, from the values it is given, and converts to an AMJax
    hierarchy. Each solve applies one V-cycle of that hierarchy per iteration,
    in JAX, inside `jax.jit` and `jax.grad`. `reinit` keeps the hierarchy for
    the values of another matrix of the same sparsity pattern, whose solves
    then take more iterations the further their values are from those of the
    hierarchy. The hierarchy keeps the dtype of the matrix. With a hierarchy of
    a single level, as that of a tiny graph, conjugate gradients run without a
    preconditioner.

    !!! example

        ```python
        from jaxscape import ResistanceDistance
        from jaxscape.solvers import AMJaxCGSolver

        distance = ResistanceDistance(solver=AMJaxCGSolver(rtol=1e-6))
        state = distance.init(grid)  # builds the hierarchy
        dist = distance(grid, state=state)
        ```

    **Attributes:**

    - `rtol`: Largest Euclidean norm of the residual, relative to that of the
      right-hand side (see `AbstractSparseIterativeSolver`).
    - `max_steps`: Largest number of iterations. A hierarchy built on a graph
      of uniform conductance, as windowed analyses and `distance.init(graph)`
      inside `jax.jit` build it, also for queries without a state, takes more
      iterations on graphs of high conductance contrast than one built on
      their own values. When their solves exceed `max_steps`, build the state
      outside `jax.jit` with `distance.init(graph)` and pass it as `state`,
      so that the hierarchy is that of the values of the graph.
    - `pyamg_method`: The PyAMG function that builds the hierarchy from a
      SciPy CSR matrix. Defaults to `pyamg.smoothed_aggregation_solver`.
    - `pyamg_kwargs`: Keyword arguments of `pyamg_method`.
    - `presmoother`, `postsmoother`: AMJax smoothers, as `(name, kwargs)`.
      Default to one iteration of Jacobi.
    - `coarse_solver`: AMJax solver of the coarsest level.
    - `coarse_solver_kwargs`: Keyword arguments of `coarse_solver`.

    !!! warning

        `amjax` and `pyamg` must be installed to use this solver, with
        `uv add jaxscape --extra amjax`. Conjugate gradients need a symmetric
        V-cycle, as AMJax's `symmetric_smoothing` flag reports for the default
        smoothers.
    """

    rtol: float = 1e-5
    max_steps: int = 1_000
    pyamg_method: Callable | None = None
    pyamg_kwargs: dict[str, Any] | None = None
    presmoother: Any | None = None
    postsmoother: Any | None = None
    coarse_solver: str = "jacobi"
    coarse_solver_kwargs: dict[str, Any] | None = None

    def __check_init__(self):
        if not AMJAX_AVAILABLE:
            raise ImportError(
                "AMJaxCGSolver requires amjax and pyamg. "
                "Install the optional dependency with `uv add jaxscape --extra amjax`."
            )

    def preconditioner(self, pattern: CSRPattern, values: Array) -> _Hierarchy:
        try:
            data = np.asarray(values)
        except jax_errors.TracerArrayConversionError:
            raise ValueError(
                "AMJaxCGSolver builds its hierarchy on the host from concrete "
                "values, so it initializes outside `jax.jit`, except for the state "
                "of a `GridGraph`, which `distance.init(graph)` builds from a graph "
                "of its shape and contiguity. Create the state with "
                "`distance.init(graph)` (or `solver.init(operator, {})`) before "
                "compiling, and pass it as `state`."
            ) from None
        n = pattern.n
        matrix = scipy.sparse.csr_matrix(
            (data, np.asarray(pattern.columns), np.asarray(pattern.offsets)),
            shape=(n, n),
        )
        assert _pyamg is not None and _MultilevelSolver is not None
        method = self.pyamg_method or _pyamg.smoothed_aggregation_solver
        hierarchy = method(matrix, **(self.pyamg_kwargs or {}))
        keywords = {
            name: value
            for name, value in [
                ("presmoother", self.presmoother),
                ("postsmoother", self.postsmoother),
                ("coarse_solver_kwargs", self.coarse_solver_kwargs),
            ]
            if value is not None
        }
        multilevel_solver = _MultilevelSolver.from_pyamg(
            hierarchy, coarse_solver=self.coarse_solver, dtype=data.dtype, **keywords
        )
        return _Hierarchy(*jax.tree.flatten(multilevel_solver))

    def precondition(
        self, preconditioner: _Hierarchy, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        del pattern, values
        multilevel_solver = preconditioner.multilevel_solver
        if len(multilevel_solver.levels) == 1:
            # AMJax cycles need two levels; tiny systems need no multigrid.
            return lambda residuals: residuals
        v_cycle = multilevel_solver.aspreconditioner(cycle="V")
        return jax.vmap(v_cycle, in_axes=1, out_axes=1)

    def matvec(
        self, preconditioner: _Hierarchy, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        del preconditioner
        return csr_matvec(pattern, values)

    def vector_shape(
        self, preconditioner: _Hierarchy, pattern: CSRPattern
    ) -> tuple[int]:
        del preconditioner
        return (pattern.n,)
