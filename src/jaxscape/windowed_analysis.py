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


def _block_centres(n: int, block_size: int) -> np.ndarray:
    """Block centres along an axis of `n` pixels, for blocks of `block_size` pixels.

    Every `block_size` pixels from `(block_size - 1) // 2`, and the first and
    last pixels.
    """
    lattice = np.arange((block_size - 1) // 2, n, block_size)
    return np.unique(np.concatenate([[0], lattice, [n - 1]]))


def _tile_shape(
    distance: AbstractDistance, batch_size: int, blocks: tuple[int, ...]
) -> tuple[int, int]:
    """Block centres along each axis of a tile.

    If the queries of `distance` share work between sources, `batch_size`
    centres or fewer, as square as the numbers of centres `blocks` along each
    axis allow, and shared evenly between the tiles along each axis.
    Otherwise one centre.
    """
    if not distance.sources_share_work:
        return 1, 1
    rows = min(math.isqrt(batch_size), blocks[0])
    columns = min(batch_size // rows, blocks[1])
    rows = min(batch_size // columns, blocks[0])
    return _even(rows, blocks[0]), _even(columns, blocks[1])


def _even(most: int, total: int) -> int:
    """The fewest items per group that split `total` items into as few groups
    of at most `most` items as possible."""
    return -(-total // -(-total // most))


class _Axis(eqx.Module):
    """The block centres along an axis of a raster, and the tiles that hold them.

    **Attributes:**

    - `knots`: For each block centre, the previous centre, the centre and the
      next centre, where its interpolation weight falls to 0, with shape
      `(centres, 3)`.
    - `members`: The block centres of each tile, consecutive, with shape
      `(tiles, count)`.
    - `valid`: Mask of the members, with shape `(tiles, count)`: the last
      tile repeats its last centre.
    """

    knots: Array
    members: Array
    valid: Array


def _axis(n: int, block_size: int, count: int) -> _Axis:
    """The block centres along an axis of `n` pixels, in tiles of `count`."""
    centres = _block_centres(n, block_size)
    knots = np.stack(
        [np.r_[centres[0] - 1, centres[:-1]], centres, np.r_[centres[1:], n]], axis=1
    )
    tiles = -(-len(centres) // count)
    members = np.arange(tiles * count).reshape(tiles, count)
    return _Axis(
        jnp.asarray(knots),
        jnp.asarray(np.minimum(members, len(centres) - 1)),
        jnp.asarray(members < len(centres)),
    )


def _first_centres(axis: _Axis) -> Array:
    """The first block centre of each tile along `axis`."""
    return axis.knots[axis.members[:, 0], 1]


def _tile_side(axis: _Axis, dependency_range: int) -> int:
    """Pixels along `axis` of its tiles, which hold the windows of their centres."""
    span = axis.knots[axis.members[:, -1], 1] - _first_centres(axis)
    return int(jnp.max(span)) + 2 * dependency_range + 1


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

    Block centres run in batches of at most `batch_size`, as few as possible
    and of equal sizes, which one compiled program computes, in one of two
    ways:

    - **Tiles**, for distances whose queries from several sources share work,
      such as `ResistanceDistance` (see `AbstractDistance.sources_share_work`).
      A tile is a grid of block centres, consecutive along each axis and at
      most `dependency_range // 10` pixels apart from the first to the last,
      as square as `batch_size` and the raster allow, and a batch holds
      `batch_size` block centres in as many tiles as fit. The graph of a tile
      is the union of the windows of its centres. One query from all of them,
      on the graph of the tile, gives each centre its distances to every pixel
      of the tile, and each centre counts the pixels of its window. The tile
      graph extends beyond each window, so its distances are lower than in
      windows, and differ with the place of a centre in the tile. Larger
      tiles leave seams along their edges in sensitivity maps: on the
      Maryland land cover of the connectivity analysis example, with exact
      `ResistanceDistance` and `dependency_range` 30, tiles of 20 x 20 pixels
      raised connectivity by 2.5 % over windows and their permeability
      sensitivity had a Pearson correlation of 0.991 with that of windows,
      against 0.7 % and 0.9991 for tiles of 4 x 4 pixels, the largest
      allowed. With `dependency_range` below 10, or with blocks of more than
      `dependency_range // 10` pixels, or with `batch_size=1`, each tile is a
      window. A query holds a distance from each centre to each pixel of its
      tile, and as many gradients: with `dependency_range` 100 and
      `batch_size=400`, three tiles of 11 x 11 centres, `10 + 201` pixels a
      side, hold 5.4 million distances each. Each query costs about the same
      whatever its number of sources, so the cost per centre falls as tiles
      grow: on an RTX 3090 in float64, with `dependency_range` 30, the
      permeability sensitivity took 4.6 ms per pixel on tiles of 4 x 4
      pixels, against 0.76 on tiles of 20 x 20 and 116 on windows, and 2.6
      against 0.29 in float32.
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
    - `batch_size`: Largest number of block centres per batch: the sources of
      its tile queries, or windows. It sets the memory, which holds all the windows of
      a batch, or the distances of its tile queries.
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
    proximity: Callable[[Array], Array]
    state: Any
    window_state: Any
    axes: tuple[_Axis, _Axis]
    shape: tuple[int, int] = eqx.field(static=True)
    dependency_range: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    tile_sides: tuple[int, int] = eqx.field(static=True)

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
        if dependency_range < 1 or batch_size < 1:
            raise ValueError("`dependency_range` and `batch_size` must be positive.")

        # An odd number of pixels, so that a block centre is the centre of its
        # block. It is at most `dependency_range + 1`.
        block_size = math.ceil(dependency_range * coarsening_factor)
        block_size += 1 - block_size % 2

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
        height, width = quality_raster.shape
        blocks = tuple(len(_block_centres(n, block_size)) for n in (height, width))
        counts = _tile_shape(distance, batch_size, blocks)
        axes = tuple(
            _axis(n, block_size, count) for n, count in zip((height, width), counts)
        )
        sides = tuple(_tile_side(axis, dependency_range) for axis in axes)
        # With `dependency_range` pixels before the raster, the tile whose
        # first block centre is at raster index `i` starts at index `i` of the
        # padded rasters. The padding after the raster fits the last tiles.
        last = [int(jnp.max(_first_centres(axis))) for axis in axes]
        pad = [
            (dependency_range, start + side - dependency_range - n)
            for start, side, n in zip(last, sides, (height, width))
        ]

        self.quality_raster = jnp.pad(quality_raster, pad)
        self.permeability_raster = jnp.pad(
            permeability_raster, pad, constant_values=floor
        )
        self.floor = floor
        self.distance = distance
        self.axes = (axes[0], axes[1])
        self.shape = (height, width)
        self.dependency_range = dependency_range
        self.block_size = block_size
        self.batch_size = batch_size
        self.tile_sides = (sides[0], sides[1])
        size = 2 * dependency_range + 1
        windows = self.tile_sides == (size, size)
        # The solver states of tiles and of windows, built outside any trace.
        # Tiles reinitialize the first; with coarsening, the block correction
        # of the permeability sensitivity is calibrated on a window with the
        # second.
        self.state = distance.init(self._uniform_graph(1, self.tile_sides))
        self.window_state = None
        if block_size > 1:
            self.window_state = distance.init(self._uniform_graph(1, (size, size)))
        if proximity is None:
            # The distances from the centre of a window to the pixels on its
            # right, up to its edge, with a state of windows if the analysis
            # has one. Otherwise the query builds its own.
            state = self.state if windows else self.window_state
            row = dependency_range * size + jnp.arange(dependency_range, size)
            # Randomized distances, such as `SpielmanApproximation`, read it
            # off the exact distances that they estimate.
            distances = distance.exact()(
                self._uniform_graph(1, (size, size)),
                sources=row[:1],
                targets=row,
                state=state,
            )
            profile = lax.cummax(distances[0])
            proximity = _PixelExponential(profile, dependency_range / 4)
        self.proximity = proximity

    @abstractmethod
    def run(self) -> Array:
        """Run the connectivity analysis. Must be implemented by subclasses."""

    @property
    def _blocks(self) -> tuple[int, int]:
        """Number of block centres along each axis."""
        rows, columns = (axis.knots.shape[0] for axis in self.axes)
        return rows, columns

    @property
    def _tiles_per_batch(self) -> int:
        """Number of tiles in a batch of `batch_size` block centres or fewer,
        the same in all batches, with as few batches as possible."""
        rows, columns = self.axes
        tiles = rows.members.shape[0] * columns.members.shape[0]
        sources = rows.members.shape[1] * columns.members.shape[1]
        return _even(max(1, self.batch_size // sources), tiles)

    def _crop(self, raster: Array) -> Array:
        """Crop a padded raster to the input shape."""
        start = self.dependency_range
        return raster[start : start + self.shape[0], start : start + self.shape[1]]

    def _interpolate(self, values: Array) -> Array:
        """Interpolate `values` at the block centres to every pixel of the raster."""

        def along_rows(values, centres, n):
            pixels = jnp.arange(n, dtype=values.dtype)

            def interpolate(column):
                return jnp.interp(pixels, centres.astype(values.dtype), column)

            return jax.vmap(interpolate, in_axes=1, out_axes=1)(values)

        rows, columns = (axis.knots[:, 1] for axis in self.axes)
        values = along_rows(values, rows, self.shape[0])
        return along_rows(values.T, columns, self.shape[1]).T

    def _uniform_graph(
        self, permeability: Array | float, shape: tuple[int, int]
    ) -> GridGraph:
        """Graph of uniform `permeability` and of `shape`."""
        grid = jnp.full(shape, permeability, self.permeability_raster.dtype)
        return GridGraph(grid, fun=_average)

    def _reinit(self, permeability: Array, dynamic: Any) -> tuple[Any, Any]:
        """The solver state of the graph of `permeability`, and its array part.

        `dynamic` is the array part of a state of the analysis, which the
        state reinitializes.
        """
        static = eqx.partition(self.state, eqx.is_array)[1]
        graph = GridGraph(lax.stop_gradient(permeability), fun=_average)
        state = self.distance.reinit(graph, eqx.combine(dynamic, static))
        return state, eqx.partition(state, eqx.is_array)[0]

    def _accumulate(self, tile_fn: Callable, initial: Array, *args: Any) -> Array:
        """Accumulate `tile_fn` over the tiles of all block centres.

        `tile_fn(windowed, tile, state, *args)` maps a `_TileData` and the
        solver state to one value per block centre of the tile, in row-major
        order, or to a map of the tile. Values add to `initial`, of one entry
        per block centre, except those of repeated centres; maps add to the
        padded raster `initial`. The first tile reinitializes the solver state
        of the analysis.
        """
        state = eqx.partition(self.state, eqx.is_array)[0]
        accumulated = initial
        rows, columns = self.axes
        n_tiles = rows.members.shape[0] * columns.members.shape[0]
        per_batch = self._tiles_per_batch
        for start in tqdm(
            range(0, n_tiles, per_batch),
            desc="Batch progress",
            miniters=max(1, n_tiles // per_batch // 100),
        ):
            batch = _Batch(self, jnp.asarray(start), state, tile_fn, args)
            accumulated, state = _batch(batch, accumulated)
        return accumulated


class _TileData(eqx.Module):
    """A tile of the padded rasters, and the block centres whose windows it holds.

    A tile spans the windows of a grid of block centres, consecutive along each
    axis: the pixels from `dependency_range` before the first centre to
    `dependency_range` after
    the last, and more after the raster's last centres, so that all tiles have
    one shape. A window is a tile of one block centre.

    **Attributes:**

    - `quality`, `permeability`: The tile of each padded raster.
    - `inside`: Mask of the tile's pixels inside the input raster.
    - `weights`: The interpolation weights of the block centres over the tile,
      along rows and along columns: shapes `(rows, tile rows)` and
      `(columns, tile columns)`.
    - `sources`: The tile rows and the tile columns of the block centres.
    - `valid`: Mask of the block centres of the raster, with shape
      `(rows, columns)`: the others repeat the last one.
    - `index`: The raster index of the first block centre, in row-major order.
      Randomized distances draw the estimates of the tile with it (see
      `AbstractDistance.reseeded`).
    """

    quality: Array
    permeability: Array
    inside: Array
    weights: tuple[Array, ...]
    sources: tuple[Array, ...]
    valid: Array
    index: Array


def _extract_tile(
    windowed: WindowedAnalysis, members: tuple[Array, Array], valid: Array
) -> _TileData:
    """The tile of the block centres `members` along each axis."""
    dtype = windowed.quality_raster.dtype
    half, sides = windowed.dependency_range, windowed.tile_sides
    knots = tuple(axis.knots[m] for axis, m in zip(windowed.axes, members))
    # A tile starts at the raster index of its first block centre in the padded
    # rasters, and at `dependency_range` pixels before it in the raster.
    origin = tuple(k[0, 1] for k in knots)
    pixels = [o - half + jnp.arange(side) for o, side in zip(origin, sides)]
    inside = [(p >= 0) & (p < n) for p, n in zip(pixels, windowed.shape)]

    def weights(pixels, knots):
        tent = jnp.array([0, 1, 0], dtype)
        return jnp.interp(pixels.astype(dtype), knots.astype(dtype), tent)

    return _TileData(
        quality=lax.dynamic_slice(windowed.quality_raster, origin, sides),
        permeability=lax.dynamic_slice(windowed.permeability_raster, origin, sides),
        inside=inside[0][:, None] & inside[1][None, :],
        weights=tuple(
            jax.vmap(weights, in_axes=(None, 0))(p, k) for p, k in zip(pixels, knots)
        ),
        sources=tuple(k[:, 1] - o + half for k, o in zip(knots, origin)),
        valid=valid,
        index=origin[0] * windowed.shape[1] + origin[1],
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
    rows, columns = windowed.axes
    n_columns = columns.members.shape[0]
    n_tiles = rows.members.shape[0] * n_columns

    # Dummy tiles complete the last batch, so all batches share one program.
    index = batch.start + jnp.arange(windowed._tiles_per_batch)
    real = index < n_tiles
    index = jnp.minimum(index, n_tiles - 1)
    row, column = index // n_columns, index % n_columns
    members = (rows.members[row], columns.members[column])
    valid = (
        real[:, None, None]
        & rows.valid[row][:, :, None]
        & columns.valid[column][:, None, :]
    )

    if batch.state is None:

        def tile(members, valid):
            return tile_fn(
                windowed, _extract_tile(windowed, members, valid), None, *args
            )

        dynamic, outputs = None, jax.vmap(tile)(members, valid)
    else:
        # Solvers that run on the host or hold one factorization cannot vmap:
        # tiles run one after another, each reinitializing the carried state.
        def step(dynamic, inputs):
            tile = _extract_tile(windowed, *inputs)
            state, dynamic = windowed._reinit(tile.permeability, dynamic)
            return dynamic, tile_fn(windowed, tile, state, *args)

        dynamic, outputs = lax.scan(step, batch.state, (members, valid))

    if outputs.ndim == 2:
        # One value per block centre, zero for the repeated ones.
        values = jnp.where(valid, outputs.reshape(valid.shape), 0)
        indices = (members[0][:, :, None], members[1][:, None, :])
        return accumulated.at[indices].add(values), dynamic

    def add(raster, inputs):
        origin, output = inputs
        window = lax.dynamic_slice(raster, origin, output.shape)
        return lax.dynamic_update_slice(raster, window + output, origin), None

    origins = (rows.knots[members[0][:, 0], 1], columns.knots[members[1][:, 0], 1])
    outputs = jnp.where(real[:, None, None], outputs, 0)
    return lax.scan(add, accumulated, (origins, outputs))[0], dynamic
