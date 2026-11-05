import math
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from tqdm import tqdm

from jaxscape.distance import AbstractDistance
from jaxscape.graph import GridGraph
from jaxscape.window_operation import WindowOperation


# Permeability below this fraction of the maximum permeability is raised to it.
# Every window graph is then connected and its grounded Laplacian symmetric
# positive definite with one sparsity pattern. An edge between two floored
# pixels conducts 1e4 times less than one between pixels of maximum
# permeability, and float32 resistances keep a relative error of about 1e-4;
# lower floors cost float32 accuracy (3e-3 at 1e-6).
PERMEABILITY_FLOOR = 1e-4


def _average(x: Array, y: Array) -> Array:
    return (x + y) / 2


class _PixelExponential(eqx.Module):
    """Negative exponential `exp(-r / alpha)` of the distance in equivalent pixels `r`.

    `profile[k]`, non-decreasing in `k`, is the distance `k` pixels from the
    centre of a window of permeability 1 along an axis. A distance converts to
    the pixels that give it: linearly between profile points, and in proportion
    past the window edge.
    """

    profile: Array
    alpha: float = eqx.field(static=True)

    def __call__(self, distance: Array) -> Array:
        last = self.profile.size - 1
        # Comparing with every point of a short profile is about ten times
        # faster than `jnp.interp`, whose search is sequential. Past about 30
        # points, a binary search is faster.
        method = "compare_all" if self.profile.size <= 32 else "scan"
        k = jnp.searchsorted(self.profile, distance, "right", method=method)
        k = jnp.clip(k, 1, last)
        lower, step = self.profile[k - 1], self.profile[k] - self.profile[k - 1]
        between = k - 1 + (distance - lower) / jnp.where(step > 0, step, 1)
        edge = self.profile[-1]
        r = jnp.where(distance > edge, distance * (last / edge), between)
        return jnp.exp(-r / self.alpha)


def _centres(n: int, size: int) -> np.ndarray:
    """Block centres along an axis of `n` pixels, for blocks of `size` pixels.

    Every `size` pixels from `(size - 1) // 2`, and the first and last pixels.
    """
    lattice = np.arange((size - 1) // 2, n, size)
    return np.unique(np.concatenate([[0], lattice, [n - 1]]))


def _tiles(n: int, size: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Block centres along an axis of `n` pixels, in tiles of `count` centres.

    Returns the knots of each centre, with shape `(tiles, count, 3)`: the
    previous centre, the centre and the next centre, where the interpolation
    weight of the centre falls to 0. The last tile repeats its last centre, and
    the mask of the other centres, with shape `(tiles, count)`, is returned too.
    """
    centres = _centres(n, size)
    knots = np.stack(
        [np.r_[centres[0] - 1, centres[:-1]], centres, np.r_[centres[1:], n]], axis=1
    )
    tiles = -(-len(centres) // count)
    index = np.arange(tiles * count)
    valid = index < len(centres)
    knots = knots[np.minimum(index, len(centres) - 1)]
    return knots.reshape(tiles, count, 3), valid.reshape(tiles, count)


def _tile_shape(
    distance: AbstractDistance, batch_size: int, blocks: tuple[int, ...]
) -> tuple[int, int]:
    """Block centres along each axis of a tile.

    `batch_size` centres, as square as `blocks` allow, if the queries of
    `distance` share work between sources, or one centre.
    """
    if not distance.sources_share_work:
        return 1, 1
    rows = min(math.isqrt(batch_size), blocks[0])
    return rows, min(batch_size // rows, blocks[1])


def _tile_side(knots: np.ndarray, dependency_range: int) -> int:
    """Pixels along an axis of the tiles of `knots`, from `_tiles`."""
    span = np.max(knots[:, -1, 1] - knots[:, 0, 1])
    return int(span) + 2 * dependency_range + 1


class WindowedAnalysis(eqx.Module):
    """Base class for windowed connectivity analyses on large rasters.

    Every pixel is the centre of a window of `2 * dependency_range + 1` pixels
    a side, and counts the pixels of its window. Distances are computed on the
    graph of the window, or of a tile that holds it (see below). The rasters
    are padded on all sides so that windows of edge pixels fit: padding has
    quality 0 and the permeability floor. Permeability below the floor, `1e-4`
    times the maximum permeability, is raised to it: every window and tile
    graph is connected, and regions of zero permeability stay nearly
    impassable.

    With coarsening, the raster is divided into blocks of `b` pixels a side,
    where `b` is `dependency_range * coarsening_factor` rounded up to an odd
    integer, and only the windows centred on block centres are computed: about
    `b**2` times fewer windows, each the same as without coarsening. Along each
    axis, block centres lie every `b` pixels from `(b - 1) / 2`, and on the
    first and last pixels, so that every pixel lies between two centres.
    Values between block centres are interpolated bilinearly: the weight of a
    centre falls linearly from 1 at the centre to 0 at the neighbouring
    centres, along each axis, and the weights of all centres sum to 1 at every
    pixel.

    Block centres run in batches of `batch_size`, one compiled program for all
    batches, in one of two ways:

    - **Tiles**, for distances whose queries from several sources share work,
      such as `ResistanceDistance` (see `AbstractDistance.sources_share_work`).
      A batch is a grid of block centres, consecutive along each axis:
      `isqrt(batch_size)` rows and `batch_size // isqrt(batch_size)` columns
      of them, or fewer on smaller rasters. Its tile is the union of their
      windows. One query from all of them, on the graph of the tile, gives each
      centre its distances to every pixel of the tile, and each centre counts
      the pixels of its window. The tile graph extends beyond each window, so
      distances on it are lower: with `ResistanceDistance` and
      `dependency_range` 20, connectivity is about 3 % higher than in windows,
      and the sensitivity maps have a Pearson correlation above 0.999 with
      theirs. With `batch_size=1`, each tile is a window. The query holds
      `batch_size` distances for each pixel of the tile, and as many
      gradients: with `dependency_range` 100 and `batch_size=400`, tiles of
      `20 + 200` pixels a side hold 19 million distances. Each query costs
      about the same whatever its number of sources, so the cost per centre
      falls as `batch_size` grows, until the tile side reaches about
      `2 * dependency_range`.
    - **Windows**, for other distances, such as `LCPDistance` and
      `EuclideanDistance`, whose queries cost about as much per source: each
      block centre has its own window, and a batch is `batch_size` windows.

    Randomized distances, such as `SpielmanApproximation`, apply the proximity
    returned by `AbstractDistance.debiased_proximity` to their distances.

    Distances without solver state, such as `LCPDistance`, `EuclideanDistance`
    and `ResistanceDistance()`, process the tiles of a batch in parallel.
    Distances with a solver process them one after another and share one
    solver state, built when the analysis is created on a tile of uniform
    permeability, and reinitialized for each tile (see
    [Reusing solver state](linear_solvers.md#reusing-solver-state)). Every run
    of the analysis starts from that state and reuses the compiled programs.

    **Arguments:**

    - `quality_raster`: Habitat quality values.
    - `permeability_raster`: Movement permeability, or function
      `quality -> permeability`.
    - `distance`: Distance metric (e.g., `LCPDistance()`,
      `ResistanceDistance()`).
    - `dependency_range`: Window half-width `D` in pixels. Pairs farther apart
      than `D` along either axis do not contribute, and the default
      `proximity` decays over `D / 4` equivalent pixels.
    - `batch_size`: Number of block centres per batch: the sources of one tile
      query, or windows. It sets the memory, which holds all the windows of a
      batch, or the distances of a tile query.
    - `proximity`: Function mapping distances, in the units of `distance`, to
      proximities. Defaults to `exp(-r / (D / 4))`, where `r` is the distance
      in equivalent pixels: the number of pixels through habitat of
      permeability 1 that gives the same distance. See
      [`ConnectivityAnalysis`][jaxscape.connectivity_analysis.ConnectivityAnalysis]
      for the units of each distance and the ecological rationale.
    - `coarsening_factor`: Spatial coarsening in [0, 1]. 0 = finest resolution,
      higher = faster.

    !!! example

        The subclasses share these arguments:

        ```python
        from jaxscape import (
            ConnectivityAnalysis,
            ResistanceDistance,
            SensitivityAnalysis,
        )
        from jaxscape.solvers import CholmodSolver

        settings = dict(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=ResistanceDistance(solver=CholmodSolver()),
            dependency_range=20,
            batch_size=100,
            coarsening_factor=0.25,  # blocks of 5 x 5 pixels
        )
        connectivity = ConnectivityAnalysis(**settings).run()
        d_permeability = SensitivityAnalysis(**settings).run("permeability")
        ```
    """

    quality_raster: Array
    permeability_raster: Array
    floor: Array
    distance: AbstractDistance
    window_op: WindowOperation
    proximity: Callable[[Array], Array]
    state: Any
    shape: tuple[int, int] = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)

    def __init__(
        self,
        quality_raster: Array,
        permeability_raster: Array | Callable[[Array], Array],
        distance: AbstractDistance,
        dependency_range: int,
        batch_size: int,
        proximity: Callable[[Array], Array] | None = None,
        coarsening_factor: float = 0.0,
    ):
        if not 0 <= coarsening_factor <= 1:
            raise ValueError("`coarsening_factor` must be in [0, 1].")
        if not isinstance(dependency_range, int) or not isinstance(batch_size, int):
            raise ValueError("`dependency_range` and `batch_size` must be integers.")
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")

        coarsening = math.ceil(dependency_range * coarsening_factor)
        if coarsening % 2 == 0:
            coarsening += 1
        buffer_size = dependency_range - (coarsening - 1) // 2
        if buffer_size < 1:
            raise ValueError("Buffer size too small.")

        if callable(permeability_raster):
            permeability_raster = permeability_raster(quality_raster)
        if quality_raster.shape != permeability_raster.shape:
            raise ValueError("Quality and permeability rasters must have one shape.")

        # The floor is a constant of the analysis: sensitivities do not
        # differentiate it.
        floor = PERMEABILITY_FLOOR * lax.stop_gradient(jnp.max(permeability_raster))
        permeability_raster = jnp.where(
            permeability_raster > floor, permeability_raster, floor
        )
        # With `dependency_range` pixels before the raster, the tile whose first
        # block centre is at raster index `i` starts at index `i` of the padded
        # rasters. The padding after the raster fits the last tiles, in steps
        # that keep the shape a valid one for `WindowOperation`.
        height, width = quality_raster.shape
        blocks = tuple(len(_centres(n, coarsening)) for n in (height, width))
        tile_shape = _tile_shape(distance, batch_size, blocks)
        pad = []
        for n, count in zip((height, width), tile_shape):
            knots, _ = _tiles(n, coarsening, count)
            after = dependency_range + (-n - coarsening + 1) % coarsening
            end = knots[-1, 0, 1] + _tile_side(knots, dependency_range)
            missing = max(0, end - dependency_range - n - after)
            steps = math.ceil(missing / coarsening)
            pad.append((dependency_range, after + steps * coarsening))

        self.quality_raster = jnp.pad(quality_raster, pad)
        self.permeability_raster = jnp.pad(
            permeability_raster, pad, constant_values=floor
        )
        self.floor = floor
        self.distance = distance
        self.shape = (height, width)
        self.batch_size = batch_size
        self.window_op = WindowOperation(
            shape=(self.quality_raster.shape[0], self.quality_raster.shape[1]),
            window_size=coarsening,
            buffer_size=buffer_size,
        )
        # Solver setup, outside any trace: windows reinitialize this state.
        graph = self._uniform_graph(1)
        self.state = distance.init(graph)
        if proximity is None:
            size = self.window_op.total_window_size
            centre = size // 2
            distances = distance(
                graph, sources=jnp.array([centre * size + centre]), state=self.state
            )
            # Randomized distances, such as `SpielmanApproximation`, can
            # decrease along the axis: the conversion stays monotone.
            profile = lax.cummax(distances.reshape(size, size)[centre, centre:])
            proximity = _PixelExponential(profile, dependency_range / 4)
        self.proximity = proximity
        if self._tile_sides != graph.grid.shape:
            # Tiles of several block centres have a sparsity pattern of their own.
            self.state = distance.init(self._uniform_graph(1, self._tile_sides))

    @abstractmethod
    def run(self) -> Array:
        """Run the connectivity analysis. Must be implemented by subclasses."""

    def _crop(self, raster: Array) -> Array:
        """Crop a padded raster to the input shape."""
        start = self.window_op.total_window_size // 2
        return raster[start : start + self.shape[0], start : start + self.shape[1]]

    @property
    def _centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Block centres along each axis."""
        rows, columns = (_centres(n, self.window_op.window_size) for n in self.shape)
        return rows, columns

    @property
    def _blocks(self) -> tuple[int, int]:
        """Number of block centres along each axis."""
        rows, columns = self._centres
        return len(rows), len(columns)

    def _interpolate(self, values: Array) -> Array:
        """Interpolate `values` at the block centres to every pixel of the raster."""

        def along_rows(values, centres):
            # The last centre is the last pixel.
            pixels = jnp.arange(centres[-1] + 1, dtype=values.dtype)

            def interpolate(column):
                return jnp.interp(pixels, centres.astype(values.dtype), column)

            return jax.vmap(interpolate, in_axes=1, out_axes=1)(values)

        rows, columns = self._centres
        values = along_rows(values, rows)
        return along_rows(values.T, columns).T

    @property
    def _tiles(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Knots of the block centres of the tiles along each axis, and their mask.

        See `_tiles` (the function).
        """
        shape = _tile_shape(self.distance, self.batch_size, self._blocks)
        size = self.window_op.window_size
        return tuple(_tiles(n, size, count) for n, count in zip(self.shape, shape))

    @property
    def _tile_sides(self) -> tuple[int, int]:
        """Pixels along each axis of a tile."""
        dependency_range = self.window_op.total_window_size // 2
        rows, columns = (
            _tile_side(knots, dependency_range) for knots, _ in self._tiles
        )
        return rows, columns

    @property
    def _tiles_per_batch(self) -> int:
        """Number of tiles in a batch of `batch_size` block centres or fewer."""
        (rows, _), (columns, _) = self._tiles
        sources = rows.shape[1] * columns.shape[1]
        return max(1, min(self.batch_size // sources, len(rows) * len(columns)))

    def _uniform_graph(
        self, permeability: Array | float, shape: tuple[int, ...] | None = None
    ) -> GridGraph:
        """Graph of uniform `permeability`, of `shape` or of one window."""
        if shape is None:
            shape = (self.window_op.total_window_size,) * 2
        grid = jnp.full(shape, permeability, self.permeability_raster.dtype)
        return GridGraph(grid, fun=_average)

    def _accumulate(
        self, tile_fn: Callable, initial: Array, *args: Any, state: Any = None
    ) -> Array:
        """Accumulate `tile_fn` over the tiles of all block centres.

        `tile_fn(windowed, tile, state, *args)` maps a `_Tile` and the solver
        state to one value per block centre of the tile, in row-major order, or
        to a map of the tile. Values add to `initial`, of one entry per block
        centre; maps add to the padded raster `initial`. The first tile
        reinitializes the array part `state` of a solver state, by default that
        of the analysis.
        """
        if state is None:
            state = eqx.partition(self.state, eqx.is_array)[0]
        accumulated = initial
        (rows, _), (columns, _) = self._tiles
        n_tiles = len(rows) * len(columns)
        per_batch = self._tiles_per_batch
        for start in tqdm(
            range(0, n_tiles, per_batch),
            desc="Batch progress",
            miniters=max(1, n_tiles // per_batch // 100),
        ):
            batch = _Batch(self, jnp.asarray(start), state, tile_fn, args)
            accumulated, state = _batch(batch, accumulated)
        return accumulated


class _Tile(eqx.Module):
    """A tile of the padded rasters, and the block centres whose windows it holds.

    A tile spans the windows of a grid of block centres, consecutive along each
    axis: the pixels from `dependency_range` before the first centre to
    `dependency_range` after the last, and more after the raster's last
    centres, so that all tiles have one shape. A window is a tile of one block
    centre.

    **Attributes:**

    - `quality`, `permeability`: The tile of each padded raster.
    - `inside`: Mask of the tile's pixels inside the input raster.
    - `weights`: The interpolation weights of the block centres over the tile,
      along rows and along columns: shapes `(rows, tile rows)` and
      `(columns, tile columns)`.
    - `sources`: The tile rows and the tile columns of the block centres.
    - `valid`: Mask of the block centres of the raster, with shape
      `(rows, columns)`: the other centres repeat the last one.
    """

    quality: Array
    permeability: Array
    inside: Array
    weights: tuple[Array, ...]
    sources: tuple[Array, ...]
    valid: Array


def _tile(
    windowed: WindowedAnalysis, knots: tuple[Array, Array], valid: Array
) -> _Tile:
    """The tile of the block centres of `knots` along each axis, from `_tiles`."""
    dtype = windowed.quality_raster.dtype
    dependency_range = windowed.window_op.total_window_size // 2
    sides = windowed._tile_sides
    # A tile starts at the raster index of its first block centre in the padded
    # rasters, and at `dependency_range` pixels before it in the raster.
    origin = tuple(k[0, 1] for k in knots)
    pixels = [o - dependency_range + jnp.arange(side) for o, side in zip(origin, sides)]
    inside = [(p >= 0) & (p < n) for p, n in zip(pixels, windowed.shape)]

    def weights(pixels, knots):
        tent = jnp.array([0, 1, 0], dtype)
        return jnp.interp(pixels.astype(dtype), knots.astype(dtype), tent)

    return _Tile(
        quality=lax.dynamic_slice(windowed.quality_raster, origin, sides),
        permeability=lax.dynamic_slice(windowed.permeability_raster, origin, sides),
        inside=inside[0][:, None] & inside[1][None, :],
        weights=tuple(
            jax.vmap(weights, in_axes=(None, 0))(p, k) for p, k in zip(pixels, knots)
        ),
        sources=tuple(k[:, 1] - o + dependency_range for k, o in zip(knots, origin)),
        valid=valid,
    )


class _Batch(eqx.Module):
    """A batch of tiles: the inputs of `_batch` that it does not donate.

    **Attributes:**

    - `windowed`: The windowed analysis.
    - `start`: Index of the first tile of the batch.
    - `state`: Array part of the solver state that the first tile
      reinitializes.
    - `tile_fn`, `args`: As in `WindowedAnalysis._accumulate`.
    """

    windowed: WindowedAnalysis
    start: Array
    state: Any
    tile_fn: Callable
    args: tuple


@eqx.filter_jit(donate="all-except-first")
def _batch(batch: _Batch, accumulated: Array) -> tuple[Array, Any]:
    """Add the tiles of `batch` to `accumulated`, whose buffer it reuses.

    Returns the accumulated value and the array part of the solver state after
    the batch.
    """
    windowed, tile_fn, args = batch.windowed, batch.tile_fn, batch.args
    dynamic, static = batch.state, eqx.partition(windowed.state, eqx.is_array)[1]
    (row_knots, row_valid), (column_knots, column_valid) = windowed._tiles
    n_tiles = len(row_knots) * len(column_knots)

    # Dummy tiles complete the last batch, so all batches share one program.
    index = batch.start + jnp.arange(windowed._tiles_per_batch)
    valid = index < n_tiles
    index = jnp.minimum(index, n_tiles - 1)
    row, column = index // len(column_knots), index % len(column_knots)
    knots = (jnp.asarray(row_knots)[row], jnp.asarray(column_knots)[column])
    valid = (
        valid[:, None, None]
        & jnp.asarray(row_valid)[row][:, :, None]
        & jnp.asarray(column_valid)[column][:, None, :]
    )

    if dynamic is None:

        def tile(knots, valid):
            return tile_fn(windowed, _tile(windowed, knots, valid), None, *args)

        outputs = jax.vmap(tile)(knots, valid)
    else:
        # Solvers that run on the host or hold one factorization cannot vmap:
        # tiles run one after another, each reinitializing the carried state.
        def step(dynamic, inputs):
            tile = _tile(windowed, *inputs)
            graph = GridGraph(lax.stop_gradient(tile.permeability), fun=_average)
            state = windowed.distance.reinit(graph, eqx.combine(dynamic, static))
            output = tile_fn(windowed, tile, state, *args)
            return eqx.partition(state, eqx.is_array)[0], output

        dynamic, outputs = lax.scan(step, dynamic, (knots, valid))

    if outputs.ndim == 2:
        # One value per block centre, zero for the repeated ones.
        def centres(tiles, count, blocks):
            return jnp.minimum(tiles[:, None] * count + jnp.arange(count), blocks - 1)

        _, rows, columns = valid.shape
        blocks = windowed._blocks
        rows = centres(row, rows, blocks[0])[:, :, None]
        columns = centres(column, columns, blocks[1])[:, None, :]
        values = jnp.where(valid, outputs.reshape(valid.shape), 0)
        return accumulated.at[rows, columns].add(values), dynamic

    def add(raster, inputs):
        knots, output = inputs
        origin = (knots[0][0, 1], knots[1][0, 1])
        window = lax.dynamic_slice(raster, origin, output.shape)
        return lax.dynamic_update_slice(raster, window + output, origin), None

    outputs = jnp.where(jnp.any(valid, axis=(1, 2))[:, None, None], outputs, 0)
    return lax.scan(add, accumulated, (knots, outputs))[0], dynamic
