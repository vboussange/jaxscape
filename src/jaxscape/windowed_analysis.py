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
        # Comparing with every point of the short profile is about ten times
        # faster than `jnp.interp`, whose search is sequential.
        k = jnp.searchsorted(self.profile, distance, "right", method="compare_all")
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


class WindowedAnalysis(eqx.Module):
    """Base class for windowed connectivity analyses on large rasters.

    Every pixel is the centre of a window of `2 * dependency_range + 1` pixels
    a side, in which distances are computed. The rasters are padded on all sides
    so that windows of edge pixels fit: padding has quality 0 and the
    permeability floor. Permeability below the floor, `1e-4` times the maximum
    permeability, is raised to it: every window graph is connected, and
    regions of zero permeability stay nearly impassable.

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

    Windows run in batches of `batch_size`, one compiled program for all
    batches. Distances without solver state, such as `LCPDistance`,
    `EuclideanDistance` and `ResistanceDistance()`, process the windows of a
    batch in parallel. Distances with a solver process them one after another
    and share one solver state, built when the analysis is created on a window
    of uniform permeability, and reinitialized for each window (see
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
    - `batch_size`: Number of windows per batch. Distances without solver
      state hold all windows of a batch in memory at once.
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
        # With `dependency_range` pixels before the raster, the window of the
        # block centre at raster index `i` starts at index `i` of the padded
        # rasters. The padding after the raster also rounds the shape up for
        # `WindowOperation`.
        height, width = quality_raster.shape
        pad = [
            (dependency_range, dependency_range + (-n - coarsening + 1) % coarsening)
            for n in (height, width)
        ]

        self.quality_raster = jnp.pad(quality_raster, pad)
        self.permeability_raster = jnp.pad(
            permeability_raster, pad, constant_values=floor
        )
        self.floor = floor
        self.distance = distance
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
        self.shape = (height, width)
        self.batch_size = batch_size

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

    def _uniform_graph(self, permeability: Array | float) -> GridGraph:
        """Window graph of uniform `permeability`."""
        size = self.window_op.total_window_size
        grid = jnp.full((size, size), permeability, self.permeability_raster.dtype)
        return GridGraph(grid, fun=_average)

    def _accumulate(
        self, window_fn: Callable, initial: Array, *args: Any, state: Any = None
    ) -> Array:
        """Accumulate `window_fn` over the windows of all block centres.

        `window_fn(windowed, quality, permeability, inside, tent, state, *args)`
        maps the windows of the padded rasters, the mask of their pixels inside
        the input raster, the interpolation weights of the window's block centre
        and the solver state, to a scalar or to a window. Scalars fill
        `initial`, of one entry per block centre in row-major order; windows
        add to the padded raster `initial`. The first window reinitializes the
        array part `state` of a solver state, by default that of the analysis.
        """
        if state is None:
            state = eqx.partition(self.state, eqx.is_array)[0]
        accumulated = initial
        n_windows = math.prod(self._blocks)
        for start in tqdm(
            range(0, n_windows, self.batch_size),
            desc="Batch progress",
            miniters=max(1, n_windows // self.batch_size // 100),
        ):
            batch = _Batch(self, jnp.asarray(start), state, window_fn, args)
            accumulated, state = _batch(batch, accumulated)
        return accumulated


class _Batch(eqx.Module):
    """A batch of windows: the inputs of `_batch` that it does not donate.

    **Attributes:**

    - `windowed`: The windowed windowed.
    - `start`: Index of the first window of the batch.
    - `state`: Array part of the solver state that the first window
      reinitializes.
    - `window_fn`, `args`: As in `WindowedAnalysis._accumulate`.
    """

    windowed: WindowedAnalysis
    start: Array
    state: Any
    window_fn: Callable
    args: tuple


@eqx.filter_jit(donate="all-except-first")
def _batch(batch: _Batch, accumulated: Array) -> tuple[Array, Any]:
    """Add the windows of `batch` to `accumulated`, whose buffer it reuses.

    Returns the accumulated value and the array part of the solver state after
    the batch.
    """
    windowed, window_fn, args = batch.windowed, batch.window_fn, batch.args
    dynamic, static = batch.state, eqx.partition(windowed.state, eqx.is_array)[1]
    op = windowed.window_op
    size = op.total_window_size
    rows, columns = windowed._blocks

    # Dummy windows complete the last batch, so all batches share one program.
    index = batch.start + jnp.arange(windowed.batch_size)
    valid = index < rows * columns
    index = jnp.minimum(index, rows * columns - 1)
    # The block centres of the windows along each axis, and the neighbouring
    # centres where their interpolation weights reach 0: shape (batch, 2, 3).
    knots = [
        jnp.asarray(np.stack([np.r_[c[0] - 1, c[:-1]], c, np.r_[c[1:], c[-1] + 1]]))
        for c in windowed._centres
    ]
    knots = jnp.stack(
        [knots[0][:, index // columns], knots[1][:, index % columns]], axis=1
    ).transpose(2, 1, 0)
    # A window starts at the index of its centre in the padded rasters.
    xy = knots[:, :, 1]

    def windows(knots):
        xy = knots[:, 1]
        pixels = xy[:, None] - size // 2 + jnp.arange(size)
        inside = ((pixels[0] >= 0) & (pixels[0] < windowed.shape[0]))[:, None] & (
            (pixels[1] >= 0) & (pixels[1] < windowed.shape[1])
        )[None, :]
        tent = [
            jnp.interp(pixels[axis], knots[axis], jnp.array([0, 1, 0]))
            for axis in range(2)
        ]
        tent = jnp.outer(tent[0], tent[1]).astype(windowed.quality_raster.dtype)
        quality = op.extract_total_window(xy, windowed.quality_raster)
        permeability = op.extract_total_window(xy, windowed.permeability_raster)
        return quality, permeability, inside, tent

    if dynamic is None:

        def window(knots):
            return window_fn(windowed, *windows(knots), None, *args)

        outputs = jax.vmap(window)(knots)
    else:
        # Solvers that run on the host or hold one factorization cannot vmap:
        # windows run one after another, each reinitializing the carried state.
        def step(dynamic, knots):
            quality, permeability, inside, tent = windows(knots)
            graph = GridGraph(lax.stop_gradient(permeability), fun=_average)
            state = windowed.distance.reinit(graph, eqx.combine(dynamic, static))
            output = window_fn(
                windowed, quality, permeability, inside, tent, state, *args
            )
            return eqx.partition(state, eqx.is_array)[0], output

        dynamic, outputs = lax.scan(step, dynamic, knots)

    outputs = jnp.where(valid.reshape((-1,) + (1,) * (outputs.ndim - 1)), outputs, 0)
    if outputs.ndim == 1:
        return accumulated.at[index].add(outputs), dynamic

    def add(raster, window):
        xy, output = window
        return op.update_raster_with_window(xy, raster, output, fun=jnp.add), None

    return lax.scan(add, accumulated, (xy, outputs))[0], dynamic
