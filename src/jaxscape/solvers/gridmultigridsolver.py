"""Conjugate gradients with a geometric multigrid preconditioner, for grid graphs."""

from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import errors as jax_errors, lax
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from lineax import AbstractLinearOperator, RESULTS

from .operator import AbstractReusableStateSolver


# Stencil offsets (row, column) of rook and queen contiguity, and of the coarse
# grids, whose Galerkin stencils couple the 3 x 3 neighbourhood of a point.
_ROOK = ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0))
_QUEEN = tuple((i, j) for i in (-1, 0, 1) for j in (-1, 0, 1))
# Grids of at most this many points are solved with a Cholesky factorization.
_COARSEST = 1024
# Damped Jacobi smoothing: sweeps before and after the coarse correction, and
# damping.
_SWEEPS = 1
_DAMPING = 2 / 3


@final
class GridMultigridState(eqx.Module):
    """State of a `GridMultigridSolver`: the stencil of one matrix on its grid.

    **Attributes:**

    - `slots`: For each stored entry of the BCOO matrix, the index of its
      coefficient in the flattened stencil, or the size of the stencil for
      padding entries.
    - `stencil`: The stencil coefficients of the matrix: for each offset, an
      array of the grid's shape.
    - `offsets`: The offsets (row, column) of the stencil.
    """

    slots: Array
    stencil: tuple[Array, ...]
    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)


class _Hierarchy(eqx.Module):
    """The multigrid hierarchy of a matrix, which each solve builds.

    **Attributes:**

    - `stencils`: The stencil coefficients of each grid, from the finest: for
      each offset, an array of the grid's shape.
    - `interpolations`: The weights of the interpolation from each coarse grid
      to the grid before it.
    - `factor`: The lower Cholesky factor of the matrix of the coarsest grid.
    - `offsets`: The offsets (row, column) of the stencil of the finest grid.
    """

    stencils: tuple[tuple[Array, ...], ...]
    interpolations: tuple[tuple[Array, ...], ...]
    factor: Array
    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)


@final
class GridMultigridSolver(AbstractReusableStateSolver):
    """Conjugate gradients preconditioned by geometric multigrid, for grid graphs.

    Solves the symmetric positive definite matrices whose sparsity pattern is a
    grid stencil, such as the grounded Laplacians of a `GridGraph`, with rook or
    queen contiguity, that `ResistanceDistance` solves. It is the default solver
    of `ResistanceDistance`. It runs in JAX, on any device, in float32 and
    float64, and needs no optional dependency. The state holds the matrix
    coefficients on the grid: only `init` reads the grid from concrete sparse
    indices, and `reinit` takes new values inside `jax.jit`. Each solve builds
    the multigrid hierarchy of the state, at a small fraction of the cost of
    the solve. A compiled program then holds one hierarchy for each place
    where it solves, whatever the states it reinitializes.

    Each V-cycle smooths with a sweep of damped Jacobi before and after the
    coarse correction. Coarse grids keep every other point along each axis, and
    interpolate the others from the matrix coefficients, as black-box multigrid
    does ([Dendy, 1982](https://doi.org/10.1016/0021-9991(82)90057-2)), which
    keeps convergence fast across jumps of conductance. Their matrices are the
    Galerkin products of the interpolation. The first grid of at most 1024
    points is solved with a Cholesky factorization. A batch of right-hand
    sides, as `jax.vmap` makes, runs as one solve with the right-hand sides
    contiguous in memory, until all have converged.

    A larger `rtol` takes fewer iterations, at the cost of the resistances to
    pixels behind barriers, which converge last. On large grids in float32,
    its solutions stay accurate where those of sparse direct solvers degrade
    around the ground of `ResistanceDistance`, and it factorizes only the
    coarsest grid, which suits conductances that change at every step, as in
    optimization loops. In float64, sparse direct solvers can be faster.

    **Attributes:**

    - `rtol`: Largest Euclidean norm, relative to that of the right-hand side,
      of the residual that conjugate gradients update at each iteration. In
      exact arithmetic it equals the residual `b - A x` of the solution `x`.
      In float32, rounding stops the decrease of `b - A x`, so that solves
      can converge with a residual `b - A x` above `rtol`.
    - `max_steps`: Largest number of iterations. A solve that has not converged
      then fails with `lineax.RESULTS.max_steps_reached`.

    !!! warning

        `init` reads concrete sparse indices: `distance.init(graph)` builds the
        state of a `GridGraph` from its shape and contiguity, also inside
        `jax.jit`, and that of another graph outside `jax.jit`. A state belongs
        to one grid: a raster shape and contiguity, with any conductances. The
        matrix must be positive definite: a grounded Laplacian of a connected
        graph. The solves of a singular matrix fail with
        `lineax.RESULTS.singular` when their solution is not finite, and with
        `lineax.RESULTS.max_steps_reached` otherwise.
    """

    rtol: float = 1e-5
    max_steps: int = 200

    def __check_init__(self):
        if self.rtol < 0:
            raise ValueError("`rtol` must be non-negative.")
        if self.max_steps < 1:
            raise ValueError("`max_steps` must be positive.")

    def init(
        self, operator: AbstractLinearOperator, options: dict[str, Any]
    ) -> GridMultigridState:
        del options
        matrix = _matrix(operator)
        slots, shape, offsets = _grid(matrix)
        return _initial_state(slots, shape, offsets, matrix.data)

    def reinit(
        self,
        operator: AbstractLinearOperator,
        options: dict[str, Any],
        state: GridMultigridState,
    ) -> GridMultigridState:
        """Return the state of the values of `operator`.

        **Arguments:**

        - `operator`: BCOO-backed operator with the sparse indices `state` was
          initialized for.
        - `options`: Unused.
        - `state`: State from `init` or an earlier call.

        **Returns:**

        The state of the values of `operator`, with the grid of `state`.
        """
        del options
        data = _checked_data(_matrix(operator), state)
        shape = state.stencil[0].shape
        return _stencil(state.slots, (shape[0], shape[1]), state.offsets, data)

    def compute(
        self, state: GridMultigridState, vector: Array, options: dict[str, Any]
    ) -> tuple[Array, RESULTS, dict[str, Any]]:
        del options
        solution, solved = _solve(state, vector, self.rtol, self.max_steps)
        # Singular matrices, such as the grounded Laplacians of disconnected
        # graphs, give non-finite solutions, or solves that do not converge.
        failed = RESULTS.where(
            jnp.all(jnp.isfinite(solution)),
            RESULTS.max_steps_reached,
            RESULTS.singular,
        )
        return solution, RESULTS.where(solved, RESULTS.successful, failed), {}

    def transpose(self, state: GridMultigridState, options: dict[str, Any]):
        return state, options

    def conj(self, state: GridMultigridState, options: dict[str, Any]):
        return state, options

    def assume_full_rank(self):
        return True


def _matrix(operator: AbstractLinearOperator) -> BCOO:
    matrix = operator.as_matrix()
    if not isinstance(matrix, BCOO):
        raise TypeError("GridMultigridSolver requires a BCOO-backed operator.")
    return lax.stop_gradient(matrix)


def _grid(
    matrix: BCOO,
) -> tuple[Array, tuple[int, int], tuple[tuple[int, int], ...]]:
    """The stencil slot of each stored entry, the grid shape and the offsets.

    The vertex `i * width + j` is the point `(i, j)` of the grid. The width is
    the largest index difference of stored entries, or one less for queen
    contiguity, whose differences also include the width minus one and two.
    """
    try:
        indices = np.asarray(matrix.indices, dtype=np.int64)
    except jax_errors.TracerArrayConversionError:
        raise ValueError(
            "GridMultigridSolver reads its grid from concrete sparse indices, so it "
            "initializes outside `jax.jit`, except for the state of a `GridGraph`, "
            "which `distance.init(graph)` builds from its shape and contiguity. "
            "Create the state with `distance.init(graph)` (or "
            "`solver.init(operator, {})`) before compiling, and pass it as `state`."
        ) from None
    n = matrix.shape[0]
    rows, columns = indices[:, 0], indices[:, 1]
    stored = (rows >= 0) & (rows < n) & (columns >= 0) & (columns < n)
    rows, columns = rows[stored], columns[stored]
    gaps = np.abs(columns - rows)
    width = int(np.max(gaps, initial=1))
    if width >= 3 and np.any(gaps == width - 1) and np.any(gaps == width - 2):
        width -= 1
    steps = np.stack(
        [columns // width - rows // width, columns % width - rows % width], axis=1
    )
    if n % width or np.any(np.abs(steps) > 1):
        raise ValueError(
            "GridMultigridSolver solves the matrices of grid graphs, whose stored "
            "entries couple each vertex `i * width + j` to its neighbours on a grid "
            "of `width` columns, such as those of a `GridGraph`. This matrix has "
            "other entries. For other graphs, pass another solver, such as "
            "`CholmodSolver` or `CuDSSSolver`, or `solver=None` for the dense solve "
            "of a small graph, as in `ResistanceDistance(solver=None)`."
        )
    # The index of each stored entry's offset in a stencil of all nine offsets.
    position = (steps[:, 0] + 1) * 3 + steps[:, 1] + 1
    present = np.zeros(9, bool)
    present[position] = True
    offsets = _ROOK if not np.any(present[[0, 2, 6, 8]]) else _QUEEN
    index = np.zeros(9, np.int64)
    index[[(i + 1) * 3 + j + 1 for i, j in offsets]] = np.arange(len(offsets))
    slots = np.full(indices.shape[0], len(offsets) * n)
    slots[stored] = index[position] * n + rows
    return jnp.asarray(slots, jnp.int32), (n // width, width), offsets


def _checked_data(matrix: BCOO, state: GridMultigridState) -> Array:
    """Return the stored values of `matrix` after checking its sparse indices.

    It raises, at trace time for another number of stored entries and at run
    time for other indices.
    """
    mismatch = (
        "GridMultigridSolver: the operator's sparse indices differ from those the "
        "state was initialized for. A state belongs to one grid; initialize a new "
        "one, e.g. with `distance.init(graph)`, for this graph geometry."
    )
    if matrix.indices.shape != (state.slots.shape[0], 2):
        raise ValueError(mismatch)
    height, width = state.stencil[0].shape
    n = height * width
    stored = state.slots < len(state.offsets) * n
    slot = jnp.where(stored, state.slots, 0)
    step = jnp.array([i * width + j for i, j in state.offsets], slot.dtype)
    row = slot % n
    rows, columns = matrix.indices[:, 0], matrix.indices[:, 1]
    in_bounds = (rows >= 0) & (rows < n) & (columns >= 0) & (columns < n)
    same_entry = (rows == row) & (columns == row + step[slot // n])
    matches = jnp.all(jnp.where(stored, in_bounds & same_entry, ~in_bounds))
    return eqx.error_if(matrix.data, ~matches, mismatch)


@eqx.filter_jit
def _initial_state(
    slots: Array,
    shape: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
    data: Array,
) -> GridMultigridState:
    """`_stencil` in one compiled program."""
    return _stencil(slots, shape, offsets, data)


def _stencil(
    slots: Array,
    shape: tuple[int, int],
    offsets: tuple[tuple[int, int], ...],
    data: Array,
) -> GridMultigridState:
    """The state of the matrix that stores `data` at `slots`."""
    height, width = shape
    stencil = jnp.zeros(len(offsets) * height * width, data.dtype)
    stencil = stencil.at[slots].add(data, mode="drop")
    return GridMultigridState(
        slots, tuple(stencil.reshape(len(offsets), height, width)), offsets
    )


# Each solve traces it, and Lineax traces a solve up to three times: a jitted
# function is traced once for its shapes.
@eqx.filter_jit
def _hierarchy(state: GridMultigridState) -> _Hierarchy:
    """The multigrid hierarchy of the matrix of `state`."""
    offsets = state.offsets
    height, width = state.stencil[0].shape
    shapes = [(height, width)]
    while shapes[-1][0] * shapes[-1][1] > _COARSEST:
        height, width = shapes[-1]
        shapes.append(((height + 1) // 2, (width + 1) // 2))
    stencils, interpolations = [state.stencil], []
    if len(shapes) > 1:
        weights = _interpolation(offsets, state.stencil)
        stencils.append(_galerkin(offsets, state.stencil, weights, shapes[1]))
        interpolations.append(weights)
    if len(shapes) > 2:
        coarser, weights = _coarser_grids(stencils[-1], shapes[1:])
        stencils += coarser
        interpolations += weights
    height, width = shapes[-1]
    n = height * width
    level_offsets = offsets if len(shapes) == 1 else _QUEEN
    eye = jnp.eye(n, dtype=state.stencil[0].dtype)
    matrix = _apply(level_offsets, stencils[-1], eye.reshape(height, width, n))
    matrix = matrix.reshape(n, n)
    factor = jnp.linalg.cholesky((matrix + matrix.T) / 2)
    return _Hierarchy(tuple(stencils), tuple(interpolations), factor, offsets)


def _coarser_grids(
    stencil: tuple[Array, ...], shapes: list[tuple[int, int]]
) -> tuple[list[tuple[Array, ...]], list[tuple[Array, ...]]]:
    """The stencils of the grids of `shapes[1:]`, and the interpolation weights
    from each to the grid before it, from the queen `stencil` of a grid of
    `shapes[0]`.

    One loop builds each grid from the one before it, in arrays of the shape
    of `shapes[0]`. Their points beyond a grid have zero coefficients: they
    couple to no point of the grid, and leave its stencil and weights
    unchanged. XLA compiles the loop once, faster than one copy of the code
    per grid.
    """
    height, width = shapes[0]
    coarse = ((height + 1) // 2, (width + 1) // 2)
    padding = ((0, height - coarse[0]), (0, width - coarse[1]))

    def coarser(stencil, _):
        weights = _interpolation(_QUEEN, stencil)
        coarser = _galerkin(_QUEEN, stencil, weights, coarse)
        return tuple(jnp.pad(c, padding) for c in coarser), (coarser, weights)

    _, (stencils, weights) = lax.scan(coarser, stencil, length=len(shapes) - 1)
    grids, interpolations = [], []
    for i, ((rows, columns), (coarse_rows, coarse_columns)) in enumerate(
        zip(shapes, shapes[1:])
    ):
        grids.append(tuple(c[i, :coarse_rows, :coarse_columns] for c in stencils))
        # The shapes of the weights, as `_interpolation` returns them.
        sizes = [(coarse_rows, columns // 2)] * 2 + [(rows // 2, coarse_columns)] * 2
        sizes += [(rows // 2, columns // 2)] * 4
        interpolations.append(
            tuple(w[i, :a, :b] for w, (a, b) in zip(weights, sizes, strict=True))
        )
    return grids, interpolations


def _shift(x: Array, i: int, j: int) -> Array:
    """`x[r + i, c + j]` at each point `(r, c)`, zero outside the grid."""
    height, width = x.shape[:2]
    padding = [(1, 1), (1, 1)] + [(0, 0)] * (x.ndim - 2)
    return jnp.pad(x, padding)[1 + i : 1 + i + height, 1 + j : 1 + j + width]


def _apply(offsets: tuple, stencil: tuple[Array, ...], x: Array) -> Array:
    """The product of the stencil matrix and `x`, of shape `(height, width, m)`."""
    terms = [a[:, :, None] * _shift(x, *o) for a, o in zip(stencil, offsets)]
    return sum(terms[1:], terms[0])


def _ratio(numerator: Array, denominator: Array) -> Array:
    nonzero = denominator != 0
    return jnp.where(nonzero, numerator / jnp.where(nonzero, denominator, 1), 0)


def _interpolation(offsets: tuple, stencil: tuple[Array, ...]) -> tuple[Array, ...]:
    """The interpolation weights of the points of odd rows or columns.

    A point between two coarse points along a row takes their weights from the
    stencil summed along columns, so that it is in equilibrium with them along
    the row; likewise along columns. A point between four coarse points is in
    equilibrium with its eight neighbours.

    **Returns:**

    The weights of the points on even rows and odd columns from their left and
    right coarse points, with shape `(coarse rows, columns // 2)`; of the points
    on odd rows and even columns from the coarse points above and below,
    `(rows // 2, coarse columns)`; and of the other points from the coarse
    points up left, up right, down left and down right,
    `(rows // 2, columns // 2)`.
    """
    zero = jnp.zeros_like(stencil[0])
    a = {o: stencil[offsets.index(o)] if o in offsets else zero for o in _QUEEN}
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


def _prolong(weights: tuple[Array, ...], x: Array, shape: tuple[int, ...]) -> Array:
    """Interpolate `x`, of shape `(coarse rows, coarse columns, m)`, to `shape`."""
    rows, columns = shape[0] // 2, shape[1] // 2
    coarse_rows, coarse_columns = x.shape[:2]
    left, right, up, down, *diagonal = (w[..., None] for w in weights)
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


def _restrict(weights: tuple[Array, ...], x: Array, coarse: tuple[int, ...]) -> Array:
    """The transpose of `_prolong`."""
    zeros = jnp.zeros(coarse + x.shape[2:], x.dtype)
    prolong = jax.linear_transpose(lambda y: _prolong(weights, y, x.shape[:2]), zeros)
    return prolong(x)[0]


def _galerkin(
    offsets: tuple,
    stencil: tuple[Array, ...],
    weights: tuple[Array, ...],
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
    fine = _prolong(weights, probes.astype(stencil[0].dtype), stencil[0].shape)
    products = _restrict(weights, _apply(offsets, stencil, fine), coarse)
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


def _smooth(
    offsets: tuple, stencil: tuple[Array, ...], b: Array, x: Array | None = None
) -> Array:
    """Damped Jacobi sweeps from `x`, or from zero."""
    diagonal = stencil[offsets.index((0, 0))][:, :, None]
    step = _ratio(jnp.asarray(_DAMPING, b.dtype), diagonal)
    sweeps = _SWEEPS
    if x is None:
        x, sweeps = step * b, sweeps - 1
    for _ in range(sweeps):
        x = x + step * (b - _apply(offsets, stencil, x))
    return x


def _v_cycle(state: _Hierarchy, b: Array, level: int = 0) -> Array:
    if level == len(state.interpolations):
        n, m = state.factor.shape[0], b.shape[-1]
        # Right-hand sides as rows, solved on the right: XLA laid out all the
        # vectors of a batch with the batch axis major for triangular solves on
        # the left, which made V-cycles several times slower on a GPU, and
        # autotuned a product with the inverse at each compilation.
        rows = lax.linalg.triangular_solve(
            state.factor,
            b.reshape(n, m).T,
            left_side=False,
            lower=True,
            transpose_a=True,
        )
        rows = lax.linalg.triangular_solve(
            state.factor, rows, left_side=False, lower=True
        )
        return rows.T.reshape(b.shape)
    offsets = state.offsets if level == 0 else _QUEEN
    stencil = state.stencils[level]
    weights = state.interpolations[level]
    x = _smooth(offsets, stencil, b)
    residual = b - _apply(offsets, stencil, x)
    coarse = state.stencils[level + 1][0].shape
    correction = _v_cycle(state, _restrict(weights, residual, coarse), level + 1)
    x = x + _prolong(weights, correction, b.shape[:2])
    return _smooth(offsets, stencil, b, x)


def _dot(x: Array, y: Array) -> Array:
    return jnp.sum(x * y, axis=(0, 1))


# Traced once for its shapes, as `_hierarchy`.
@eqx.filter_jit
def _conjugate_gradients(
    state: _Hierarchy, b: Array, rtol: float, max_steps: int
) -> tuple[Array, Array]:
    """Solve for the columns of `b`, of shape `(height, width, m)`.

    Each column is scaled exactly, by a power of two, to a largest magnitude in
    [0.5, 1), so that float32 right-hand sides of about 1e-37 keep their
    precision; those below the smallest normal number solve to 0. All columns
    iterate until the last one converges, which costs no time.

    **Returns:**

    The solutions, and whether each column converged.
    """
    largest = jnp.max(jnp.abs(b), axis=(0, 1))
    exponent = jnp.frexp(largest)[1]
    b = jnp.where(largest >= jnp.finfo(b.dtype).tiny, jnp.ldexp(b, -exponent), 0)
    offsets, stencil = state.offsets, state.stencils[0]
    tolerance = (rtol**2) * _dot(b, b)

    def step(carry):
        x, r, p, rz, squared_residual, steps = carry
        z = _v_cycle(state, r)
        rz_new = _dot(r, z)
        # The first step starts from p = 0.
        p = z + _ratio(rz_new, rz) * p
        q = _apply(offsets, stencil, p)
        alpha = _ratio(rz_new, _dot(p, q))
        r = r - alpha * q
        return x + alpha * p, r, p, rz_new, _dot(r, r), steps + 1

    def unconverged(carry):
        return jnp.any(carry[4] > tolerance) & (carry[5] < max_steps)

    # One V-cycle in the program, inside the loop, halves its compilation.
    zeros = jnp.zeros_like(b)
    carry = (zeros, b, zeros, jnp.ones_like(tolerance), _dot(b, b), 0)
    x, _, _, _, squared_residual, _ = lax.while_loop(unconverged, step, carry)
    # A NaN residual compares false.
    solved = squared_residual <= tolerance
    return jnp.ldexp(x, exponent), solved


def _solve(
    state: GridMultigridState, vector: Array, rtol: float, max_steps: int
) -> tuple[Array, Array]:
    """The solution of one right-hand side, and whether it converged. Under
    `jax.vmap`, the right-hand sides of a batch solve together."""
    height, width = state.stencil[0].shape

    def solve(state, vectors):
        # Right-hand sides as columns.
        b = vectors.reshape(height, width, -1)
        x, solved = _conjugate_gradients(_hierarchy(state), b, rtol, max_steps)
        return x.reshape(height * width, -1), solved

    @jax.custom_batching.custom_vmap
    def solve_one(state, vector):
        x, solved = solve(state, vector[:, None])
        return x[:, 0], solved[0]

    @solve_one.def_vmap
    def solve_batch(axis_size, in_batched, state, vectors):
        state_batched, vectors_batched = in_batched
        if any(jax.tree.leaves(state_batched)):
            # Other matrices for the right-hand sides: one solve each.
            axes = jax.tree.map(lambda batched: 0 if batched else None, state_batched)
            one = jax.vmap(
                lambda state, vector: solve_one.fun(state, vector),
                in_axes=(axes, 0 if vectors_batched else None),
            )
            return one(state, vectors), (True, True)
        if not vectors_batched:
            vectors = jnp.broadcast_to(vectors, (axis_size,) + vectors.shape)
        x, solved = solve(state, vectors.T)
        return (x.T, solved), (True, True)

    return solve_one(state, vector)
