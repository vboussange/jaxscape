import atexit
import math
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.core import Tracer
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


# The latest solver state of each distance, graph shape and dtype, which
# windowed analyses share (see `WindowedAnalysis`), the most recently used last.
# spineax keeps as many cuDSS factorizations by default.
_STATES: OrderedDict[Hashable, Any] = OrderedDict()
_MOST_STATES = 8
# Native objects of states, such as CHOLMOD factorizations, are released
# before the interpreter shuts their modules down.
atexit.register(_STATES.clear)


def _average(x: Array, y: Array) -> Array:
    return (x + y) / 2


def _tracing() -> bool:
    """Whether JAX is tracing the running code, as `jax.jit` does: a transfer
    to the device then gives a tracer. Outside a trace, the transfer compiles
    no program; `jnp.array(1) + 1` would compile two, in 36 ms on a CPU."""
    return isinstance(jax.device_put(0), Tracer)


def _traced(tree: Any) -> bool:
    """Whether a leaf of `tree` is a tracer."""
    return any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(tree))


# XLA:GPU options of the programs that windowed analyses compile. Their matrix
# products and convolutions are small next to their distance queries, and ran
# as fast with the heuristics of cuBLAS and cuDNN as autotuned: autotuning,
# with candidate Triton kernels for each matrix product, took 7 s of the
# compilation of the connectivity analysis example on an RTX 3090. Other
# backends ignore these options.
_COMPILER_OPTIONS = {"xla_gpu_autotune_level": 0, "xla_gpu_enable_triton_gemm": False}


class _Program(eqx.Module):
    """A function jitted with `_COMPILER_OPTIONS`, and without them for the
    calls that JAX traces, as inside `jax.jit`: JAX takes compiler options
    for the programs that it compiles alone only.

    **Attributes:**

    - `compiled`: The function jitted with the options.
    - `traced`: The function jitted without them.
    """

    compiled: Callable
    traced: Callable

    def __call__(self, *args: Any) -> Any:
        return (self.traced if _tracing() else self.compiled)(*args)


def _program(**kwargs: Any) -> Callable[[Callable], _Program]:
    """Decorator that makes a `_Program` with the `eqx.filter_jit` arguments
    `kwargs`."""

    def program(fun: Callable) -> _Program:
        # `eqx.filter_jit` passes `compiler_options` to `jax.jit`, which its
        # type overloads leave out.
        options = {"compiler_options": _COMPILER_OPTIONS}
        compiled = eqx.filter_jit(fun, **kwargs, **options)  # pyright: ignore
        return _Program(compiled, eqx.filter_jit(fun, **kwargs))

    return program


def _state_key(
    distance: AbstractDistance, shape: tuple[int, int], dtype: Any
) -> Hashable | None:
    """The key of the solver states of `distance` on grid graphs of `shape` and
    `dtype`, or `None` if it is unhashable.

    Distances share their states with the exact distances that they estimate
    (see `AbstractDistance.exact`).
    """
    key = (distance.exact(), shape, jnp.dtype(dtype))
    try:
        hash(key)
    except TypeError:
        return None
    return key


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
      A tile is a grid of up to `batch_size` block centres, consecutive along
      each axis and as square as `batch_size` and the raster allow, and the
      tiles along an axis hold as many block centres as they can share
      evenly. With `batch_size` at least the number of block centres, one
      tile covers the padded raster, and with `batch_size=1` each tile is a
      window. The graph of a tile is the union of the windows of its
      centres. One window query from all of them, on the graph of the tile,
      gives each centre its distances to the pixels of its window (see
      `AbstractDistance.sources_to_windows_distance`). The tile graph extends
      beyond each window, so its distances are lower than in windows, and
      differ with the place of a centre in the tile. On a smooth raster of
      30 x 30 pixels with `dependency_range` 10 and exact
      `ResistanceDistance`, one tile raised connectivity by 2.7 % over
      windows, and its permeability sensitivity had a Pearson correlation of
      0.9996 with that of windows. On the Maryland land cover of the
      connectivity analysis example, with `dependency_range` 100 and
      `SpielmanApproximation`, one tile raised connectivity by 2.1 to 3.1 %
      over windows, with two seeds at `epsilon` 0.1 and one at 0.15. Each
      query costs about the same whatever its number of sources, so the cost
      per centre falls as tiles grow: with `SpielmanApproximation(epsilon=0.1)`
      and `CuDSSSolver` on an RTX 3090 in float32, the connectivity took
      137 s with a window per block centre, and 2.9 s on one tile, about 47
      times less. A raster that needs several tiles has seams along their
      edges in sensitivity maps: on the same land cover, with `epsilon` 0.1,
      `batch_size=400` gives 2 x 2 tiles, across whose edges along columns
      the mean absolute step of the logarithm of the permeability
      sensitivity was 1.79 times that between other neighbouring pixels,
      against 0.94 at the same places on one tile. The maps of four tiles
      and of one tile had a Pearson correlation of 0.9946, and the total
      sensitivity of four tiles was 0.74 % higher.
    - **Windows**, for other distances, such as `LCPDistance` and
      `EuclideanDistance`, whose queries cost about as much per source: each
      block centre has its own window, and a batch is `batch_size` windows.

    Randomized distances, such as `SpielmanApproximation`, apply the proximity
    returned by `AbstractDistance.debiased_proximity` to their distances.

    Distances without solver state, such as `LCPDistance`, `EuclideanDistance`
    and `ResistanceDistance()`, process the tiles of a batch in parallel.
    Distances with a solver process them one after another, each tile
    reinitializing the solver state that the previous one left (see
    [Reusing solver state](linear_solvers.md#reusing-solver-state)).
    Windowed analyses share solver states within a process: for each
    distance, graph shape and dtype, they keep the latest state, which the
    next run, calibration of the block correction on a window, or profile of
    the default proximity starts from and replaces. Randomized distances
    share the states of the exact distances that they estimate. A windowed
    analysis builds a state, on a graph of uniform permeability, only for a
    shape without one, so that a sparse direct solver analyses each sparsity
    pattern once, however many windowed analyses and runs use it. Queries
    reinitialize the state they receive: up to rounding, the results do not
    depend on the state that a run starts from. The states of the 8 most
    recently used combinations of distance, shape and dtype are kept, until
    the interpreter exits: on the 894 x 925 tile of the Maryland land cover
    in float32, a state of `GridMultigridSolver` holds 52 MiB of JAX arrays,
    and one of `CuDSSSolver` 132 MiB besides its cuDSS factorization.
    Distances that cannot be hashed, such as those with an `AMJaxCGSolver`,
    keep the state that each analysis builds, which its runs start from.
    Analyses created or run while JAX traces them, as inside `jax.jit`,
    neither read nor replace the shared states: their runs start from the
    state that the analysis holds. Every run reuses the compiled programs.

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
      its tile queries, or windows. It sets the memory, which holds all the
      windows of a batch, or the distances of its tile queries to the windows
      of their sources. Distances that keep the default window query, such
      as exact `ResistanceDistance` and `FreeEnergyDistance`, first compute
      the distances from each source to every vertex of its tile, and hold
      `m * n` distances for `m` sources and `n` tile vertices, besides the
      memory of the solver: on the Maryland land cover of the connectivity
      analysis example, with `dependency_range` 100 and `coarsening_factor`
      0.2, `batch_size=1024` gives 2 x 2 tiles of 547 x 558 pixels and 99
      million distances per tile, and `batch_size=4096` one tile of
      894 x 925 pixels and 1.04 billion distances, 3.9 GiB in float32.
      `batch_size` trades memory for seams: fewer, larger tiles leave fewer
      seams.
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
        # The solver states of tiles and of windows: the latest of their
        # shapes, or built outside any trace. Tiles reinitialize the first;
        # with coarsening, the block correction of the permeability
        # sensitivity is calibrated on a window with the second.
        self.state = self._initial_state(self.tile_sides)
        self.window_state = None
        if block_size > 1:
            self.window_state = self._initial_state(window)
        if proximity is None:
            state = self.state if self.tile_sides == window else self.window_state
            if state is None:
                state = self._initial_state(window)
            profile, state = _profile(distance, self._uniform_graph(1, window), state)
            self._leave_state(window, state)
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

    def _initial_state(self, shape: tuple[int, int]) -> Any:
        """The latest solver state of graphs of `shape`, or else a new state of
        a graph of `shape` and uniform permeability, which becomes the latest."""
        state = self._latest_state(shape, None)
        if state is None:
            state = self.distance.init(self._uniform_graph(1, shape))
            self._leave_state(shape, state)
        return state

    def _latest_state(self, shape: tuple[int, int], state: Any) -> Any:
        """The latest solver state that windowed analyses left for graphs of
        `shape`, or `state` if there is none or if JAX is tracing."""
        key = _state_key(self.distance, shape, self.permeability_raster.dtype)
        if key not in _STATES or _tracing():
            return state
        _STATES.move_to_end(key)
        return _STATES[key]

    def _leave_state(self, shape: tuple[int, int], state: Any) -> None:
        """Make `state` the latest solver state of graphs of `shape`, unless it
        holds tracers: those of a traced computation are invalid outside it."""
        key = _state_key(self.distance, shape, self.permeability_raster.dtype)
        if key is None or state is None or _traced(state):
            return
        _STATES[key] = state
        _STATES.move_to_end(key)
        if len(_STATES) > _MOST_STATES:
            _STATES.popitem(last=False)

    def _reinit(self, permeability: Array, state: Any) -> Any:
        """The solver state of the graph of `permeability`, reinitialized from
        `state`."""
        graph = GridGraph(lax.stop_gradient(permeability), fun=_average)
        return self.distance.reinit(graph, state)

    def _while_compiling(
        self, work: Callable[[], Any], tile_fn: Callable, initial: Array, *args: Any
    ) -> Any:
        """Return `work()`, run while another thread compiles the batch program
        of `_accumulate(tile_fn, initial, *args)`, which `_accumulate` then
        runs. Arrays of the shapes and dtypes of those of `args` can stand in
        for them. In a trace, as inside `jax.jit`, nothing compiles alone.
        """
        if _tracing():
            return work()
        state = self._latest_state(self.tile_sides, self.state)
        batch = _Batch(self, jnp.asarray(0), state, tile_fn, args)
        with ThreadPoolExecutor(max_workers=1) as pool:
            compiled = pool.submit(
                lambda: _batch.compiled.lower(batch, initial).compile()
            )
            result = work()
            compiled.result()
        return result

    def _accumulate(self, tile_fn: Callable, initial: Array, *args: Any) -> Array:
        """Accumulate `tile_fn` over the tiles of all block centres.

        `tile_fn(windowed, tile, state, *args)` maps a `_TileData` and the
        solver state to one value per block centre of the tile, in row-major
        order, or to a map of the tile. Values add to `initial`, of one entry
        per block centre, except those of repeated centres; maps add to the
        padded raster `initial`. The first tile reinitializes the latest solver
        state of the tile shape, and the state of the last tile becomes the
        latest.
        """
        state = self._latest_state(self.tile_sides, self.state)
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
        self._leave_state(self.tile_sides, state)
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


@_program(donate="all-except-first")
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


@_program()
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
