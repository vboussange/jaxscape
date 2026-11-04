import math
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
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


def _uniform_graph(size: int, permeability, dtype) -> GridGraph:
    return GridGraph(jnp.full((size, size), permeability, dtype), fun=_average)


class _PixelExponential(eqx.Module):
    """Negative exponential `exp(-r / alpha)` of the distance `r` in equivalent pixels.

    `profile[k]` is the distance `k` pixels from the centre of a window of
    permeability 1, along an axis. A distance converts to the pixels that give
    it, interpolated linearly between profile points and extrapolated linearly
    past the window edge.
    """

    profile: Array
    alpha: float = eqx.field(static=True)

    def __call__(self, distance: Array) -> Array:
        pixels = jnp.arange(self.profile.size, dtype=self.profile.dtype)
        edge = self.profile[-1]
        slope = 1 / (edge - self.profile[-2])
        beyond = jnp.where(distance > edge, slope * (distance - edge), 0)
        r = jnp.interp(distance, self.profile, pixels) + beyond
        return jnp.exp(-r / self.alpha)


class WindowedAnalysis(eqx.Module):
    """Base class for windowed connectivity analyses on large rasters.

    Every pixel is the centre of a window of `2 * dependency_range + 1` pixels
    a side, in which distances are computed. With coarsening, one window serves
    each block of `coarsening` pixels a side. The rasters are padded on all sides
    so that windows of edge pixels fit: padding has quality 0 and the
    permeability floor. Permeability below the floor, `1e-4` times the maximum
    permeability, is raised to it: every window graph is connected, and
    regions of zero permeability stay nearly impassable.

    Windows run in batches of `batch_size`, one compiled program for all
    batches. Distances without solver state, such as `LCPDistance`,
    `EuclideanDistance` and `ResistanceDistance()`, process the windows of a
    batch in parallel. Distances with a solver process them one after another
    and share one solver state, built once on a window of uniform permeability
    and reinitialized for each window (see
    [Reusing solver state](linear_solvers.md#reusing-solver-state)).

    **Arguments:**

    - `quality_raster`: Habitat quality values.
    - `permeability_raster`: Movement permeability, or function
      `quality -> permeability`.
    - `distance`: Distance metric (e.g., `LCPDistance()`,
      `ResistanceDistance()`).
    - `proximity`: Distance-to-proximity transform (e.g.,
      `lambda d: jnp.exp(-d/D)`).
    - `dependency_range`: Spatial dependency range in pixels (window half-width).
    - `batch_size`: Number of windows per batch. Distances without solver
      state hold all windows of a batch in memory at once.
    - `coarsening_factor`: Spatial coarsening in [0, 1]. 0 = finest resolution,
      higher = faster.

    !!! example

        ```python
        from jaxscape.windowed_analysis import WindowedAnalysis

        # process_window is an application-supplied callback returning a scalar.
        class CustomConnectivity(WindowedAnalysis):
            def run(self):
                result = 0.0
                for xy, quality in self.window_op.lazy_iterator(self.quality_raster):
                    result += process_window(quality)
                return result
        ```
    """

    quality_raster: Array
    permeability_raster: Array
    floor: Array
    distance: AbstractDistance
    window_op: WindowOperation
    proximity: Callable[[Array], Array]
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
        height, width = quality_raster.shape
        pad = (
            (buffer_size, buffer_size + (-height) % coarsening),
            (buffer_size, buffer_size + (-width) % coarsening),
        )

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
        if proximity is None:
            size = self.window_op.total_window_size
            graph = _uniform_graph(size, 1, self.permeability_raster.dtype)
            centre = size // 2
            distances = distance(graph, sources=jnp.array([centre * size + centre]))
            profile = distances.reshape(size, size)[centre, centre:]
            proximity = _PixelExponential(profile, dependency_range / 4)
        self.proximity = proximity
        self.shape = (height, width)
        self.batch_size = batch_size

    @abstractmethod
    def run(self) -> Array:
        """Run the connectivity analysis. Must be implemented by subclasses."""

    def _crop(self, raster: Array) -> Array:
        """Crop a padded raster to the input shape."""
        start = self.window_op.buffer_size
        return raster[start : start + self.shape[0], start : start + self.shape[1]]

    def _accumulate(self, window_fn: Callable, initial: Array, *args: Any) -> Array:
        """Accumulate `window_fn` over all windows, one batch at a time.

        `window_fn(analysis, quality, permeability, inside, state, *args)` maps
        the windows of the padded rasters, the mask of their pixels inside the
        input raster, and the solver state, to a scalar added to `initial` or a
        window added to the padded raster `initial`.
        """
        size = self.window_op.total_window_size
        state = self.distance.init(
            _uniform_graph(size, 1, self.permeability_raster.dtype)
        )
        dynamic, static = eqx.partition(state, eqx.is_array)

        accumulated = initial
        n_windows = self.window_op.nb_steps
        for start in tqdm(
            range(0, n_windows, self.batch_size),
            desc="Batch progress",
            miniters=max(1, n_windows // self.batch_size // 100),
        ):
            inputs = (self, jnp.asarray(start), dynamic, static, window_fn, args)
            accumulated, dynamic = _batch(inputs, accumulated)
        return accumulated


@eqx.filter_jit(donate="all-except-first")
def _batch(inputs: tuple, accumulated: Array) -> tuple[Array, Any]:
    """Add the windows of one batch to `accumulated`, whose buffer it reuses.

    The windows of the batch start at window `start`. Returns the accumulated
    value and the array part of the solver state after the batch.
    """
    analysis, start, dynamic, static, window_fn, args = inputs
    op = analysis.window_op
    size = op.total_window_size

    # Dummy windows complete the last batch, so all batches share one program.
    index = start + jnp.arange(analysis.batch_size)
    valid = index < op.nb_steps
    index = jnp.minimum(index, op.nb_steps - 1)
    xy = jnp.stack([index // op.y_steps, index % op.y_steps], axis=1) * op.window_size

    def windows(xy):
        rows = xy[0] - op.buffer_size + jnp.arange(size)
        columns = xy[1] - op.buffer_size + jnp.arange(size)
        inside = ((rows >= 0) & (rows < analysis.shape[0]))[:, None] & (
            (columns >= 0) & (columns < analysis.shape[1])
        )[None, :]
        quality = op.extract_total_window(xy, analysis.quality_raster)
        permeability = op.extract_total_window(xy, analysis.permeability_raster)
        return quality, permeability, inside

    if dynamic is None:

        def window(xy):
            return window_fn(analysis, *windows(xy), None, *args)

        outputs = jax.vmap(window)(xy)
    else:
        # Solvers that run on the host or hold one factorization cannot vmap:
        # windows run one after another, each reinitializing the carried state.
        def step(dynamic, xy):
            quality, permeability, inside = windows(xy)
            graph = GridGraph(lax.stop_gradient(permeability), fun=_average)
            state = analysis.distance.reinit(graph, eqx.combine(dynamic, static))
            output = window_fn(analysis, quality, permeability, inside, state, *args)
            return eqx.partition(state, eqx.is_array)[0], output

        dynamic, outputs = lax.scan(step, dynamic, xy)

    outputs = jnp.where(valid.reshape((-1,) + (1,) * (outputs.ndim - 1)), outputs, 0)
    if accumulated.ndim == 0:
        return accumulated + jnp.sum(outputs), dynamic

    def add(raster, window):
        xy, output = window
        return op.update_raster_with_window(xy, raster, output, fun=jnp.add), None

    return lax.scan(add, accumulated, (xy, outputs))[0], dynamic
