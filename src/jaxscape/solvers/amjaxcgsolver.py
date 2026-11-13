"""Conjugate gradients with an AMJax algebraic multigrid preconditioner."""

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse
from jax import errors as jax_errors
from jax.experimental.sparse import BCOO
from jaxtyping import Array

from .sparseiterativesolver import AbstractSparseIterativeSolver, csr_matvec
from .sparsesolver import CSRPattern


try:
    import pyamg as _pyamg
    from amjax import (
        inverse_diagonal as _inverse_diagonal,
        MultilevelSolver as _MultilevelSolver,
    )

    AMJAX_AVAILABLE = True
except ImportError:
    _pyamg = None
    _inverse_diagonal = None
    _MultilevelSolver = None
    AMJAX_AVAILABLE = False


@final
class _Hierarchy(eqx.Module):
    """The preconditioner of an `AMJaxCGSolver` state: an AMJax multigrid
    hierarchy, as its leaves and tree structure, without the values and
    inverse diagonals of its levels, which each solve computes from the
    current values.

    AMJax's hierarchies are pytrees whose children depend on which of their
    leaves are `None`, so that filters such as `eqx.partition`, which replace
    leaves with `None`, change their structure. A list of leaves keeps it.

    **Attributes:**

    - `leaves`: The leaves of the hierarchy, with `None` in place of the values
      and the inverse diagonal of each level.
    - `galerkin`: For each level but the coarsest, the sparse matrix that maps
      its values to those of the next level, whose matrix is the Galerkin
      product `R A P` of the matrix `A` of the level with the restriction `R`
      and prolongation `P` of the hierarchy.
    - `treedef`: The tree structure of the hierarchy.
    - `levels`: For each level, the positions in `leaves` of its values, its
      sparse indices and its inverse diagonal, and its number of rows.
    """

    leaves: list[Array | None]
    galerkin: list[BCOO]
    treedef: Any = eqx.field(static=True)
    levels: tuple[tuple[int, int, int, int], ...] = eqx.field(static=True)

    def multilevel_solver(self, values: Array) -> Any:
        """The AMJax `MultilevelSolver` of the CSR values of the finest level."""
        assert _inverse_diagonal is not None
        leaves: list[Any] = list(self.leaves)
        for level, (data, indices, inverse_diagonal, n) in enumerate(self.levels):
            if level > 0:
                values = self.galerkin[level - 1] @ values
            leaves[data] = values
            matrix = BCOO((values, leaves[indices]), shape=(n, n))
            leaves[inverse_diagonal] = _inverse_diagonal(matrix)
        return jax.tree.unflatten(self.treedef, leaves)


@final
class AMJaxCGSolver(AbstractSparseIterativeSolver[_Hierarchy]):
    """Conjugate gradients preconditioned by algebraic multigrid, with AMJax.

    Its preconditioner is a multigrid hierarchy, which `init` builds once on the
    host with PyAMG, from the values it is given, and converts to an AMJax
    hierarchy. Each solve applies one V-cycle of that hierarchy per iteration,
    in JAX, inside `jax.jit` and `jax.grad`. `reinit` keeps the prolongation
    and restriction operators of the hierarchy, and each solve computes the
    matrices of its coarser levels from the current values, as Galerkin
    products with these operators. The hierarchy keeps the dtype of the
    matrix. With a hierarchy of a single level, as that of a tiny graph,
    conjugate gradients run without a preconditioner.

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
    - `max_steps`: Largest number of iterations.
    - `pyamg_method`: The PyAMG function that builds the hierarchy from a
      SciPy CSR matrix. Defaults to `pyamg.smoothed_aggregation_solver`.
    - `pyamg_kwargs`: Keyword arguments of `pyamg_method`.
    - `presmoother`, `postsmoother`: AMJax smoothers, as `(name, kwargs)`.
      Default to one iteration of Jacobi.
    - `coarse_solver`: AMJax solver of the coarsest level. The coarse solvers
      `"pinv"`, `"lu"` and `"qr"` keep the factorization of the coarsest
      matrix of `init`.
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
        leaves, treedef = jax.tree.flatten(multilevel_solver)
        # The positions of the leaves of each level, found by identity.
        position = {id(leaf): index for index, leaf in enumerate(leaves)}
        rows, columns = np.asarray(pattern.rows), np.asarray(pattern.columns)
        galerkin, levels = [], []
        for pyamg_level, level in zip(hierarchy.levels, multilevel_solver.levels):
            slots = (
                position[id(level.A.data)],
                position[id(level.A.indices)],
                position[id(level.Dinv)],
            )
            leaves[slots[0]] = leaves[slots[2]] = None
            leaves[slots[1]] = jnp.asarray(np.stack([rows, columns], 1), jnp.int32)
            levels.append(slots + (level.A.shape[0],))
            if level.P is not None:
                product, rows, columns = _galerkin(
                    rows, columns, pyamg_level.R, pyamg_level.P, data.dtype
                )
                galerkin.append(product)
        return _Hierarchy(leaves, galerkin, treedef, tuple(levels))

    def precondition(
        self, preconditioner: _Hierarchy, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        del pattern
        if len(preconditioner.levels) == 1:
            # AMJax cycles need two levels; tiny systems need no multigrid.
            return lambda residuals: residuals
        multilevel_solver = preconditioner.multilevel_solver(values)
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


def _galerkin(
    rows: np.ndarray,
    columns: np.ndarray,
    restriction: Any,
    prolongation: Any,
    dtype: np.dtype,
) -> tuple[BCOO, np.ndarray, np.ndarray]:
    """The Galerkin product `R A P` of a matrix `A` with entries at `rows` and
    `columns`, with SciPy sparse matrices `R` and `P`, as the sparse matrix
    that maps the values of `A` to those of the product.

    Entry `(i, j)` of `A` adds `R[I, i] * A[i, j] * P[j, J]` to entry `(I, J)`
    of the product, for every entry of column `i` of `R` and of row `j` of `P`.

    **Returns:**

    The sparse matrix, and the rows and columns of the entries of the product,
    in row-major order.
    """
    left, right = restriction.T.tocsr(), prolongation.tocsr()
    entry, r = _row_entries(left, rows)
    pair, p = _row_entries(right, columns[entry])
    entry, r = entry[pair], r[pair]
    size = right.shape[1]
    keys, targets = np.unique(
        left.indices[r].astype(np.int64) * size + right.indices[p],
        return_inverse=True,
    )
    matrix = BCOO(
        (
            jnp.asarray(left.data[r] * right.data[p], dtype),
            jnp.asarray(np.stack([targets, entry], 1), jnp.int32),
        ),
        shape=(keys.size, rows.size),
    )
    return matrix, keys // size, keys % size


def _row_entries(
    matrix: scipy.sparse.csr_array | scipy.sparse.csr_matrix, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """For every entry of the given rows of a CSR matrix, the index in `rows`
    of its row, and its position in the data of the matrix."""
    starts, counts = matrix.indptr[rows], np.diff(matrix.indptr)[rows]
    owners = np.repeat(np.arange(rows.size), counts)
    first = np.repeat(starts - np.cumsum(counts) + counts, counts)
    return owners, first + np.arange(owners.size)
