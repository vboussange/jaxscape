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


def _axis(n: int, block_size: int, count: int) -> tuple[_Axis, int, int]:
    """The block centres along an axis of `n` pixels, in tiles of `count`.

    Also returns the largest span from the first to the last block centre of a
    tile, and the first block centre of the last tile.
    """
    centres = _block_centres(n, block_size)
    knots = np.stack(
        [np.r_[centres[0] - 1, centres[:-1]], centres, np.r_[centres[1:], n]], axis=1
    )
    tiles = -(-len(centres) // count)
    members = np.minimum(np.arange(tiles * count), len(centres) - 1)
    members = members.reshape(tiles, count)
    span = np.max(centres[members[:, -1]] - centres[members[:, 0]])
    axis = _Axis(
        jnp.asarray(knots),
        jnp.asarray(members),
        jnp.asarray(np.arange(tiles * count).reshape(tiles, count) < len(centres)),
    )
    return axis, int(span), int(centres[members[-1, 0]])


@eqx.filter_jit
def _padded(raster: Array, pad: tuple, value: Array | float = 0) -> Array:
    """`raster` padded by `pad` pixels with `value`, in one compiled program."""
    return jnp.pad(raster, pad, constant_values=value)


class WindowedAnalysis(eqx.Module):
    """Base class for windowed connectivity analyses on large rasters.

    Every pixel is the centre of a window of `2 * dependency_range + 1` pixels
    a side, and counts the pixels of its window. The rasters are padded so
    that windows of edge pixels fit: padding has quality 0 and the
    permeability floor. Permeability below the floor, `1e-4` times the
    maximum permeability, is raised to it: every window graph is connected,
    and regions of zero permeability stay nearly impassable.

    With coarsening, only the windows of block centres are computed, every
    `b` pixels along each axis and on the first and last pixels, where `b` is
    `dependency_range * coarsening_factor` rounded up to an odd integer.
    Values between block centres are interpolated bilinearly: the weight of a
    centre falls linearly from 1 at the centre to 0 at the neighbouring
    centres, along each axis, and the weights of all centres sum to 1 at every
    pixel.

    Block centres run in batches of at most `batch_size`. Distances whose
    queries from several sources share work, such as `ResistanceDistance`
    (see `AbstractDistance.sources_share_work`), compute a batch on tiles: a
    tile is the union of the windows of a grid of block centres, and one
    query on it gives each centre its distances to the pixels of its window.
    The tile graph extends beyond each window, so its distances are lower
    than in windows, and sensitivity maps have seams along the edges between
    tiles. Larger tiles cost less per block centre and leave fewer seams.
    Other distances, such as `LCPDistance`, compute each block centre on its
    own window. Randomized distances, such as `SpielmanApproximation`, apply
    the proximity returned by `AbstractDistance.debiased_proximity`.

    An analysis builds its solver states once, and every run starts from them
    and reuses the compiled programs (see
    [Reusing solver state](linear_solvers.md#reusing-solver-state)).

    **Arguments:**

    - `quality_raster`: Habitat quality values. Both rasters take their common
      dtype, which is floating, and so do the results: an integer or boolean
      quality takes the floating dtype of the permeability, and a float32
      raster with a float64 one gives float64.
    - `permeability_raster`: Movement permeability, or function
      `quality -> permeability`. An integer or boolean permeability takes
      JAX's default floating dtype.
    - `distance`: Distance metric (e.g., `LCPDistance()`,
      `ResistanceDistance()`).
    - `dependency_range`: Window half-width `D` in pixels. Pairs farther apart
      than `D` along either axis do not contribute, and the default
      `proximity` decays over `D / 4` equivalent pixels.
    - `batch_size`: Largest number of block centres per batch, which sets the
      memory: a batch holds its windows, or the distances of its tile
      queries. Exact `ResistanceDistance` and `FreeEnergyDistance` hold
      `m * n` distances for `m` sources and `n` tile vertices, besides the
      memory of the solver. With `batch_size` at least the number of block
      centres, one tile covers the raster.
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

        settings = dict(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=ResistanceDistance(),  # GridMultigridSolver
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
        # Both rasters take their common dtype, in which the analyses sum and
        # differentiate. It is floating, as the floored permeability is.
        dtype = jnp.result_type(quality_raster, permeability_raster)
        quality_raster = quality_raster.astype(dtype)
        permeability_raster = permeability_raster.astype(dtype)
        height, width = quality_raster.shape
        blocks = tuple(len(_block_centres(n, block_size)) for n in (height, width))
        counts = _tile_shape(distance, batch_size, blocks)
        axes, spans, last = zip(
            *(_axis(n, block_size, count) for n, count in zip((height, width), counts))
        )
        # A tile holds the windows of its block centres.
        sides = tuple(span + 2 * dependency_range + 1 for span in spans)
        # With `dependency_range` pixels before the raster, the tile whose
        # first block centre is at raster index `i` starts at index `i` of the
        # padded rasters. The padding after the raster fits the last tiles.
        pad = tuple(
            (dependency_range, start + side - dependency_range - n)
            for start, side, n in zip(last, sides, (height, width))
        )

        self.quality_raster = _padded(quality_raster, pad)
        self.permeability_raster = _padded(permeability_raster, pad, floor)
        self.floor = floor
        self.distance = distance
        self.axes = (axes[0], axes[1])
        self.shape = (height, width)
        self.dependency_range = dependency_range
        self.block_size = block_size
        self.batch_size = batch_size
        self.tile_sides = (sides[0], sides[1])
        size = 2 * dependency_range + 1
        window = (size, size)
        # The solver states of tiles and of windows. Tiles reinitialize the
        # first; the profile of the default proximity and, with coarsening,
        # the calibration of the block correction run on a window with the
        # second.
        self.state = distance.init(self._uniform_graph(1, self.tile_sides))
        self.window_state = None
        if block_size > 1 or proximity is None:
            self.window_state = distance.init(self._uniform_graph(1, window))
        if proximity is None:
            graph = self._uniform_graph(1, window)
            profile, self.window_state = _profile(distance, graph, self.window_state)
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

    def _reinit(self, permeability: Array, state: Any) -> Any:
        """The solver state of the graph of `permeability`, reinitialized from
        `state`."""
        graph = GridGraph(lax.stop_gradient(permeability), fun=_average)
        return self.distance.reinit(graph, state)

    def _accumulate(self, tile_fn: Callable, initial: Array, *args: Any) -> Array:
        """Accumulate `tile_fn` over the tiles of all block centres.

        `tile_fn(windowed, tile, state, *args)` maps a `_TileData` and the
        solver state to one value per block centre of the tile, in row-major
        order, or to a map of the tile. Values add to `initial`, of one entry
        per block centre, except those of repeated centres; maps add to the
        padded raster `initial`. The first tile reinitializes `self.state`.
        """
        state, accumulated = self.state, initial
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
    - `state`: The solver state that the first tile reinitializes.
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

    Returns the accumulated value and the solver state after the batch.
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

        state, outputs = None, jax.vmap(tile)(members, valid)
    else:
        # Solvers that run on the host or hold one factorization cannot vmap:
        # tiles run one after another, each reinitializing the carried state.
        dynamic, static = eqx.partition(batch.state, eqx.is_array)

        def step(dynamic, inputs):
            tile = _extract_tile(windowed, *inputs)
            state = windowed._reinit(tile.permeability, eqx.combine(dynamic, static))
            outputs = tile_fn(windowed, tile, state, *args)
            return eqx.partition(state, eqx.is_array)[0], outputs

        dynamic, outputs = lax.scan(step, dynamic, (members, valid))
        state = eqx.combine(dynamic, static)

    if outputs.ndim == 2:
        # One value per block centre, zero for the repeated ones.
        values = jnp.where(valid, outputs.reshape(valid.shape), 0)
        indices = (members[0][:, :, None], members[1][:, None, :])
        return accumulated.at[indices].add(values), state

    def add(raster, inputs):
        origin, output = inputs
        window = lax.dynamic_slice(raster, origin, output.shape)
        return lax.dynamic_update_slice(raster, window + output, origin), None

    origins = (rows.knots[members[0][:, 0], 1], columns.knots[members[1][:, 0], 1])
    outputs = jnp.where(real[:, None, None], outputs, 0)
    return lax.scan(add, accumulated, (origins, outputs))[0], state


@eqx.filter_jit
def _profile(
    distance: AbstractDistance, graph: GridGraph, state: Any
) -> tuple[Array, Any]:
    """The distances from the centre of the square window `graph` to the pixels
    on its right, up to its edge, made non-decreasing, and the solver state of
    `graph`, reinitialized from `state`.

    Randomized distances, such as `SpielmanApproximation`, read them off the
    exact distances that they estimate.
    """
    exact = distance.exact()
    state = exact.reinit(graph, state)
    half = graph.width // 2
    row = half * graph.width + jnp.arange(half, graph.width)
    distances = exact(graph, sources=row[:1], targets=row, state=state)
    return lax.cummax(distances[0]), state
