"""Conjugate gradients with a geometric multigrid preconditioner, for grid graphs."""

from collections.abc import Callable
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jaxtyping import Array

from .sparseiterativesolver import _ratio, AbstractSparseIterativeSolver
from .sparsesolver import CSRPattern


# Stencil offsets (row, column) of rook and queen contiguity, and of the coarse
# grids, whose Galerkin stencils couple the 3 x 3 neighbourhood of a point.
_ROOK = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
_QUEEN = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1))
# The largest number of points of the coarsest level, which a Cholesky
# factorization solves.
_MAX_COARSE = 1024
# Damped Jacobi relaxation: iterations before and after the coarse correction,
# and weight.
_ITERATIONS = 1
_OMEGA = 2 / 3


@final
class _Grid(eqx.Module):
    """The preconditioner of a `GeometricMultigridCGSolver` state: the grid, on
    which each solve builds the multigrid hierarchy of the current values.

    **Attributes:**

    - `slots`: For each CSR value, the index of its coefficient in the
      flattened stencil.
    - `shape`: The shape `(rows, columns)` of the grid.
    - `offsets`: The offsets (row, column) of the stencil.
    """

    slots: Array
    shape: tuple[int, int] = eqx.field(static=True)
    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)


class _Level(eqx.Module):
    """One level of a `_MultilevelSolver`, on a grid.

    **Attributes:**

    - `A`: The stencil coefficients of the matrix of the level: for each offset,
      an array of the grid's shape.
    - `P`: The weights of the interpolation from the next coarser level to this
      one, or `None` on the coarsest level.
    - `offsets`: The offsets (row, column) of the stencil of `A`.
    """

    A: tuple[Array, ...]
    P: tuple[Array, ...] | None
    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)


class _MultilevelSolver(eqx.Module):
    """The multigrid hierarchy of a matrix.

    Each solve builds it.

    **Attributes:**

    - `levels`: The levels, from the finest.
    - `factor`: The lower Cholesky factor of the matrix of the coarsest level.
    """

    levels: tuple[_Level, ...]
    factor: Array


@final
class GeometricMultigridCGSolver(AbstractSparseIterativeSolver[_Grid]):
    """Conjugate gradients preconditioned by geometric multigrid, for grid graphs.

    Solves the symmetric positive definite matrices whose sparsity pattern is a
    grid stencil, such as the grounded Laplacians of a `GridGraph` with rook or
    queen contiguity, in which `ResistanceDistance` solves linear systems. It
    is the default solver of `ResistanceDistance`. It runs in JAX, on any
    device, in float32 and float64, and needs no optional dependency. Its
    preconditioner is the grid, which `init` reads from concrete sparse
    indices. Each solve builds the multigrid hierarchy of the current values on
    it, at a small fraction of the cost of the solve. A compiled program then
    holds one hierarchy for each place where it solves, whatever the states it
    reinitializes.

    Each V-cycle has one iteration of damped Jacobi relaxation before and after
    the coarse correction. The coarse grids hold every other point along each
    axis. The values at the other points are interpolated from the matrix
    coefficients, as in black-box multigrid
    ([Dendy, 1982](https://doi.org/10.1016/0021-9991(82)90057-2)), so that
    convergence stays fast across jumps of conductance. The matrices of the
    coarse grids are the Galerkin products of the interpolation. The first grid
    of at most 1024 points is solved with a Cholesky factorization.

    With a larger `rtol`, the solves take fewer iterations, at the cost of the
    accuracy of the resistances to pixels behind barriers, which converge
    last. On large grids in float32, the solutions of this solver stay accurate
    around the ground vertex of `ResistanceDistance`, where those of sparse
    direct solvers lose accuracy. It factorizes only the matrix of the coarsest
    grid, which suits conductances that change at every step, as in
    optimization loops. In float64, sparse direct solvers can be faster.

    **Attributes:**

    - `rtol`: Largest Euclidean norm of the residual, relative to that of the
      right-hand side (see `AbstractSparseIterativeSolver`).
    - `max_steps`: Largest number of iterations.

    !!! warning

        A state belongs to one grid: a raster shape and contiguity, with any
        conductances. The matrix must be positive definite: a grounded
        Laplacian of a connected graph. The solves of a singular matrix fail
        with `lineax.RESULTS.singular` when their solution is not finite, and
        with `lineax.RESULTS.max_steps_reached` otherwise.
    """

    rtol: float = 1e-5
    max_steps: int = 200

    def preconditioner(self, pattern: CSRPattern, values: Array) -> _Grid:
        del values
        return _grid(pattern)

    def precondition(
        self, preconditioner: _Grid, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        del pattern
        stencil = _stencil(preconditioner, values)
        ml = _multilevel_solver(preconditioner.offsets, stencil)
        return lambda residuals: _v_cycle(ml, residuals)

    def matvec(
        self, preconditioner: _Grid, pattern: CSRPattern, values: Array
    ) -> Callable[[Array], Array]:
        del pattern
        stencil = _stencil(preconditioner, values)
        return lambda x: _matvec(preconditioner.offsets, stencil, x)

    def vector_shape(
        self, preconditioner: _Grid, pattern: CSRPattern
    ) -> tuple[int, int]:
        del pattern
        return preconditioner.shape


def _grid_width(gaps: Array | np.ndarray) -> Array | np.integer:
    """The number of columns of a grid graph, from the index differences `gaps`
    between the vertices of its stored entries.

    The vertex `i * width + j` is the point `(i, j)` of the grid. The width is
    the largest difference, or one less for queen contiguity, whose differences
    also include the width minus one and two. Differences of 0, as for padding
    entries, leave it unchanged, and it is 1 without entries. It takes NumPy
    arrays, for concrete sparse indices, as well as JAX arrays, also traced.
    """
    width = gaps.max(initial=1)
    queen = (width >= 3) & (gaps == width - 1).any() & (gaps == width - 2).any()
    return width - queen


def _grid(pattern: CSRPattern) -> _Grid:
    """The grid of a matrix of CSR `pattern`.

    The vertex `i * width + j` is the point `(i, j)` of the grid, of the width
    of `_grid_width`.
    """
    n = pattern.n
    rows = np.asarray(pattern.rows, np.int64)
    columns = np.asarray(pattern.columns, np.int64)
    width = int(_grid_width(np.abs(columns - rows)))
    steps = np.stack(
        [columns // width - rows // width, columns % width - rows % width], axis=1
    )
    if n % width or np.any(np.abs(steps) > 1):
        raise ValueError(
            "GeometricMultigridCGSolver solves the matrices of grid graphs, whose "
            "stored entries couple each vertex `i * width + j` to its neighbours on a "
            "grid of `width` columns, such as those of a `GridGraph`. This matrix has "
            "other entries. For other graphs, pass another solver, such as "
            "`CholmodSolver` or `CuDSSSolver`, or `solver=None` for the dense solve "
            "of a small graph, as in `ResistanceDistance(solver=None)`."
        )
    # The index of each CSR value's offset in a stencil of all nine offsets.
    position = (steps[:, 0] + 1) * 3 + steps[:, 1] + 1
    present = np.zeros(9, bool)
    present[position] = True
    offsets = _ROOK if not np.any(present[[0, 2, 6, 8]]) else _QUEEN
    index = np.zeros(9, np.int64)
    index[[(i + 1) * 3 + j + 1 for i, j in offsets]] = np.arange(len(offsets))
    slots = jnp.asarray(index[position] * n + rows, jnp.int32)
    return _Grid(slots, (n // width, width), offsets)


def _stencil(grid: _Grid, values: Array) -> tuple[Array, ...]:
    """The stencil coefficients of the matrix of CSR `values` on `grid`: for
    each offset, an array of the grid's shape."""
    height, width = grid.shape
    stencil = jnp.zeros(len(grid.offsets) * height * width, values.dtype)
    stencil = stencil.at[grid.slots].add(values)
    return tuple(stencil.reshape(len(grid.offsets), height, width))


def _multilevel_solver(
    offsets: tuple[tuple[int, int], ...], A: tuple[Array, ...]
) -> _MultilevelSolver:
    """The multigrid hierarchy of the matrix of stencil `A` with `offsets`."""
    height, width = A[0].shape
    shapes = [(height, width)]
    while shapes[-1][0] * shapes[-1][1] > _MAX_COARSE:
        height, width = shapes[-1]
        shapes.append(((height + 1) // 2, (width + 1) // 2))
    level_offsets = [offsets] + [_QUEEN] * (len(shapes) - 1)
    As, Ps = [A], []
    if len(shapes) > 1:
        P = _interpolation(level_offsets[0], A)
        As.append(_galerkin(level_offsets[0], A, P, shapes[1]))
        Ps.append(P)
    if len(shapes) > 2:
        coarser_As, coarser_Ps = _coarser_grids(As[-1], shapes[1:])
        As += coarser_As
        Ps += coarser_Ps
    height, width = shapes[-1]
    n = height * width
    eye = jnp.eye(n, dtype=A[0].dtype)
    matrix = _matvec(level_offsets[-1], As[-1], eye.reshape(height, width, n))
    matrix = matrix.reshape(n, n)
    factor = jnp.linalg.cholesky((matrix + matrix.T) / 2)
    levels = map(_Level, As, [*Ps, None], level_offsets)
    return _MultilevelSolver(tuple(levels), factor)


def _coarser_grids(
    A: tuple[Array, ...], shapes: list[tuple[int, int]]
) -> tuple[list[tuple[Array, ...]], list[tuple[Array, ...]]]:
    """The matrices of the grids of `shapes[1:]`, and the interpolations to
    each grid before it, from the queen matrix `A` of a grid of `shapes[0]`.

    One loop builds each grid from the one before it, in arrays of the shape
    of `shapes[0]`. Their points beyond a grid have zero coefficients: they
    couple to no point of the grid, and leave its matrix and interpolation
    unchanged. XLA compiles the loop once, faster than one copy of the code
    per grid.
    """
    height, width = shapes[0]
    coarse = ((height + 1) // 2, (width + 1) // 2)
    padding = ((0, height - coarse[0]), (0, width - coarse[1]))

    def coarser(A, _):
        P = _interpolation(_QUEEN, A)
        coarse_A = _galerkin(_QUEEN, A, P, coarse)
        return tuple(jnp.pad(a, padding) for a in coarse_A), (coarse_A, P)

    _, (padded_As, padded_Ps) = lax.scan(coarser, A, length=len(shapes) - 1)
    As, Ps = [], []
    for i, ((rows, columns), (coarse_rows, coarse_columns)) in enumerate(
        zip(shapes, shapes[1:])
    ):
        As.append(tuple(a[i, :coarse_rows, :coarse_columns] for a in padded_As))
        # The shapes of the weights, as `_interpolation` returns them.
        sizes = [(coarse_rows, columns // 2)] * 2 + [(rows // 2, coarse_columns)] * 2
        sizes += [(rows // 2, columns // 2)] * 4
        Ps.append(
            tuple(w[i, :a, :b] for w, (a, b) in zip(padded_Ps, sizes, strict=True))
        )
    return As, Ps


def _shift(x: Array, i: int, j: int) -> Array:
    """`x[r + i, c + j]` at each point `(r, c)`, zero outside the grid."""
    height, width = x.shape[:2]
    padding = [(1, 1), (1, 1)] + [(0, 0)] * (x.ndim - 2)
    return jnp.pad(x, padding)[1 + i : 1 + i + height, 1 + j : 1 + j + width]


def _matvec(offsets: tuple, A: tuple[Array, ...], x: Array) -> Array:
    """The product of the stencil matrix `A` and `x`, of shape `(height, width, m)`."""
    terms = [a[:, :, None] * _shift(x, *o) for a, o in zip(A, offsets)]
    return sum(terms[1:], terms[0])


def _interpolation(offsets: tuple, A: tuple[Array, ...]) -> tuple[Array, ...]:
    """The interpolation weights of the points of odd rows or columns.

    A point between two coarse points along a row takes their weights from the
    stencil of `A` summed along columns, so that it is in equilibrium with them
    along the row; likewise along columns. A point between four coarse points
    is in equilibrium with its eight neighbours.

    **Returns:**

    The weights of the points on even rows and odd columns from their left and
    right coarse points, with shape `(coarse rows, columns // 2)`; of the points
    on odd rows and even columns from the coarse points above and below,
    `(rows // 2, coarse columns)`; and of the other points from the coarse
    points up left, up right, down left and down right,
    `(rows // 2, columns // 2)`.
    """
    zero = jnp.zeros_like(A[0])
    a = {o: A[offsets.index(o)] if o in offsets else zero for o in _QUEEN}
    sides = (-1, 1)
    # The stencil summed over its rows (columns), whose own column (row) holds
    # the coefficient of the point.
    own_column = a[-1, 0] + a[0, 0] + a[1, 0]
    horizontal = {j: _ratio(-(a[-1, j] + a[0, j] + a[1, j]), own_column) for j in sides}
    own_row = a[0, -1] + a[0, 0] + a[0, 1]
    vertical = {i: _ratio(-(a[i, -1] + a[i, 0] + a[i, 1]), own_row) for i in sides}
    diagonal = [
        _ratio(
            -(
                a[i, j]
                + a[i, 0] * _shift(horizontal[j], i, 0)
                + a[0, j] * _shift(vertical[i], 0, j)
            ),
            a[0, 0],
        )
        for i in sides
        for j in sides
    ]
    even, odd = slice(0, None, 2), slice(1, None, 2)
    return (
        *(horizontal[j][even, odd] for j in sides),
        *(vertical[i][odd, even] for i in sides),
        *(weights[odd, odd] for weights in diagonal),
    )


def _spread(x: Array, start: tuple[int, int], shape: tuple[int, ...]) -> Array:
    """`x` at every other point of a grid of `shape` from `start`, zero elsewhere."""
    padding = [(s, n - s - 2 * m + 1, 1) for s, m, n in zip(start, x.shape, shape)]
    padding += [(0, 0, 0)] * (x.ndim - 2)
    return lax.pad(x, jnp.zeros((), x.dtype), padding)


def _prolong(P: tuple[Array, ...], x: Array, shape: tuple[int, ...]) -> Array:
    """Interpolate `x`, of shape `(coarse rows, coarse columns, m)`, to `shape`."""
    rows, columns = shape[0] // 2, shape[1] // 2
    coarse_rows, coarse_columns = x.shape[:2]
    left, right, up, down, *diagonal = (w[..., None] for w in P)
    x_ = jnp.pad(x, ((0, 1), (0, 1), (0, 0)))
    above, below = x_[:rows], x_[1 : rows + 1]
    fine = _spread(x, (0, 0), shape)
    if columns:
        between = (
            left * x_[:coarse_rows, :columns]
            + right * x_[:coarse_rows, 1 : columns + 1]
        )
        fine = fine + _spread(between, (0, 1), shape)
    if rows:
        between = up * above[:, :coarse_columns] + down * below[:, :coarse_columns]
        fine = fine + _spread(between, (1, 0), shape)
    if rows and columns:
        corners = (
            diagonal[0] * above[:, :columns]
            + diagonal[1] * above[:, 1 : columns + 1]
            + diagonal[2] * below[:, :columns]
            + diagonal[3] * below[:, 1 : columns + 1]
        )
        fine = fine + _spread(corners, (1, 1), shape)
    return fine


def _restrict(P: tuple[Array, ...], x: Array, coarse: tuple[int, ...]) -> Array:
    """The transpose of `_prolong`."""
    zeros = jnp.zeros(coarse + x.shape[2:], x.dtype)
    prolong = jax.linear_transpose(lambda y: _prolong(P, y, x.shape[:2]), zeros)
    return prolong(x)[0]


def _galerkin(
    offsets: tuple,
    A: tuple[Array, ...],
    P: tuple[Array, ...],
    coarse: tuple[int, int],
) -> tuple[Array, ...]:
    """The stencil of the coarse matrix `P.T A P`, from its products with 9
    vectors: those of the coarse points `(i, j)` of each `(i % 3, j % 3)`. The
    coarse stencil couples points less than 2 apart along each axis, so that
    each product gives one coefficient of each point."""
    i = np.arange(coarse[0])[:, None, None] % 3
    j = np.arange(coarse[1])[None, :, None] % 3
    colour = np.arange(9)
    probes = jnp.asarray(i == colour // 3) & jnp.asarray(j == colour % 3)
    fine = _prolong(P, probes.astype(A[0].dtype), A[0].shape)
    products = _restrict(P, _matvec(offsets, A, fine), coarse)
    # The product of the vector that holds the neighbour (i + di, j + dj) gives
    # the coefficient of the offset (di, dj) at (i, j).
    di, dj = (np.array([o[axis] for o in _QUEEN]) for axis in (0, 1))
    holder = jnp.asarray((i + di) % 3 * 3) + jnp.asarray((j + dj) % 3)
    coefficients = jnp.take_along_axis(products, holder, axis=2)
    # The coefficient of the offset (di, dj) at a point is that of (-di, -dj),
    # the reverse order of `_QUEEN`, at its neighbour: their mean is exactly
    # symmetric.
    return tuple(
        (coefficients[..., k] + _shift(coefficients[..., 8 - k], a, b)) / 2
        for k, (a, b) in enumerate(_QUEEN)
    )


def _jacobi(offsets: tuple, A: tuple[Array, ...], x: Array | None, b: Array) -> Array:
    """Damped Jacobi iterations on `A x = b` from `x`, or from zero for `None`."""
    diagonal = A[offsets.index((0, 0))][:, :, None]
    step = _ratio(jnp.asarray(_OMEGA, b.dtype), diagonal)
    iterations = _ITERATIONS
    if x is None:
        x, iterations = step * b, iterations - 1
    for _ in range(iterations):
        x = x + step * (b - _matvec(offsets, A, x))
    return x


def _v_cycle(ml: _MultilevelSolver, b: Array, lvl: int = 0) -> Array:
    if lvl == len(ml.levels) - 1:
        n, m = ml.factor.shape[0], b.shape[-1]
        # Right-hand sides as rows, solved on the right: XLA laid out all the
        # vectors of a batch with the batch axis major for triangular solves on
        # the left, which made V-cycles several times slower on a GPU, and
        # autotuned a product with the inverse at each compilation.
        rows = lax.linalg.triangular_solve(
            ml.factor,
            b.reshape(n, m).T,
            left_side=False,
            lower=True,
            transpose_a=True,
        )
        rows = lax.linalg.triangular_solve(ml.factor, rows, left_side=False, lower=True)
        return rows.T.reshape(b.shape)
    level, coarse = ml.levels[lvl], ml.levels[lvl + 1]
    offsets, A, P = level.offsets, level.A, level.P
    assert P is not None
    x = _jacobi(offsets, A, None, b)
    coarse_b = _restrict(P, b - _matvec(offsets, A, x), coarse.A[0].shape)
    coarse_x = _v_cycle(ml, coarse_b, lvl + 1)
    x = x + _prolong(P, coarse_x, b.shape[:2])
    return _jacobi(offsets, A, x, b)
