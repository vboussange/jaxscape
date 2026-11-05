from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax

from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.free_energy_distance import FreeEnergyDistance
from jaxscape.graph import GridGraph
from jaxscape.lcp_distance import LCPDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.window_operation import WindowOperation
from jaxscape.windowed_analysis import _average, _Tile, WindowedAnalysis


# Distances that are symmetric on the window graphs, whose edges average the
# permeability of their two pixels.
_SYMMETRIC_DISTANCES = (
    EuclideanDistance,
    FreeEnergyDistance,
    LCPDistance,
    ResistanceDistance,
)


def _excluded(op: WindowOperation, block: bool) -> Array:
    """Mask of the window centre, or of the block around it if `block`."""
    size = op.total_window_size
    half = op.window_size // 2 if block else 0
    core = slice(size // 2 - half, size // 2 + half + 1)
    return jnp.zeros((size, size), bool).at[core, core].set(True)


# Neighbourhood sums hold about this many proximities at once, and recompute
# them for the gradient.
_PROXIMITIES_PER_STEP = 2**22


def _reachable(
    windowed: WindowedAnalysis,
    weight: Array,
    permeability: Array,
    sources: tuple[Array, ...],
    state: Any,
    excluded: Array,
) -> Array:
    """Proximity-weighted `weight` reachable from each source, outside `excluded`.

    One query on the graph of the tile `permeability` gives the distances from
    a grid of sources, at the tile rows and columns `sources`, to every pixel.
    Each source counts the pixels of its window, outside `excluded`. Returns
    one value per source, in row-major order.
    """
    size = excluded.shape[0]
    height, width = permeability.shape
    rows, columns = (x.ravel() for x in jnp.meshgrid(*sources, indexing="ij"))
    graph = GridGraph(permeability, fun=_average)
    distances = windowed.distance(graph, sources=rows * width + columns, state=state)
    proximity = windowed.distance.debiased_proximity(windowed.proximity, graph)

    def reachable(inputs):
        distances, row, column = inputs
        start = (row - size // 2, column - size // 2)
        distances = lax.dynamic_slice(
            distances.reshape(height, width), start, (size, size)
        )
        weights = lax.dynamic_slice(weight, start, (size, size))
        # Excluded pixels, the centre among them, get a unit distance: it keeps
        # the gradient finite for proximities that are singular at zero.
        distances = jnp.where(excluded, 1, distances)
        return jnp.sum(jnp.where(excluded, 0, proximity(distances) * weights))

    per_step = max(1, _PROXIMITIES_PER_STEP // size**2)
    if rows.size <= per_step:
        return jax.vmap(reachable)((distances, rows, columns))
    return lax.map(
        jax.checkpoint(reachable), (distances, rows, columns), batch_size=per_step
    )


def _tile_connectivity(
    windowed: WindowedAnalysis,
    tile: _Tile,
    state: Any,
    q_weighted: bool,
    block: bool = False,
) -> Array:
    """Connectivity of the pixels that each block centre of `tile` stands for.

    Their weight, summed with the interpolation weights of the centre, times the
    proximity-weighted weight seen from the centre: outside the centre, or
    outside its block if `block`.
    """
    weight = tile.quality if q_weighted else tile.inside.astype(tile.quality.dtype)
    excluded = _excluded(windowed.window_op, block)
    reachable = _reachable(
        windowed, weight, tile.permeability, tile.sources, state, excluded
    )
    rows, columns = tile.weights
    mass = jnp.einsum(
        "iy,yx,jx->ij", rows, weight, columns, precision=lax.Precision.HIGHEST
    )
    return jnp.where(tile.valid, mass, 0).ravel() * reachable


def _tile_reachable_quality(
    windowed: WindowedAnalysis, tile: _Tile, state: Any
) -> Array:
    """Proximity-weighted quality reachable from each block centre of `tile`."""
    excluded = _excluded(windowed.window_op, block=False)
    reachable = _reachable(
        windowed, tile.quality, tile.permeability, tile.sources, state, excluded
    )
    return jnp.where(tile.valid.ravel(), reachable, 0)


def _tile_sensitivity(
    windowed: WindowedAnalysis,
    tile: _Tile,
    state: Any,
    var: str,
    q_weighted: bool,
    calibration: tuple[Array, Array] | None,
) -> Array:
    """Gradient of the connectivity of `tile` with respect to its `var`.

    With a `calibration` of the block correction, the permeability gradient
    leaves out the pairs within blocks and is corrected for them.
    """
    block = calibration is not None

    def connectivity(quality, permeability):
        varied = eqx.tree_at(
            lambda t: (t.quality, t.permeability), tile, (quality, permeability)
        )
        return jnp.sum(_tile_connectivity(windowed, varied, state, q_weighted, block))

    argnums = 1 if var == "permeability" else 0
    gradient = jax.grad(connectivity, argnums)(tile.quality, tile.permeability)
    if calibration is None:
        return gradient
    return _block_correction(windowed, tile, calibration) * gradient


def _block_correction(
    windowed: WindowedAnalysis, tile: _Tile, calibration: tuple[Array, Array]
) -> Array:
    """Correction of the permeability gradient of `tile` for its blocks.

    The calibration's gradient with all pairs, summed over the pixel windows
    that the tile's block centres stand for, with their interpolation weights,
    divided by its gradient without the pairs within the block, summed over the
    block centres with the sums of their weights.
    """
    pixels, far = calibration
    half = windowed.window_op.total_window_size // 2
    reach = windowed.window_op.window_size - 1
    rows, columns = tile.weights
    height, width = tile.inside.shape
    inside = tile.inside.astype(rows.dtype)
    # The calibration's source is at index `shape // 2` of its gradients. The
    # interpolation weights of the raster pixels reach at most `reach` pixels
    # beyond the tile's first and last block centres, which are `half` pixels
    # from its edges.
    weights = inside * jnp.einsum("ij,iy,jx->yx", tile.valid, rows, columns)
    weights = weights[
        half - reach : height - half + reach, half - reach : width - half + reach
    ]
    start = tuple(n // 2 - half + reach for n in pixels.shape)
    spread = _convolve(weights, pixels, start, (height, width))
    # The sums of the weights, at the block centres.
    sums = jnp.einsum(
        "iy,yx,jx->ij", rows, inside, columns, precision=lax.Precision.HIGHEST
    )
    centres = (
        jnp.zeros_like(inside)
        .at[tile.sources[0][:, None], tile.sources[1][None, :]]
        .add(jnp.where(tile.valid, sums, 0))
    )
    centres = centres[half : height - half, half : width - half]
    start = tuple(n // 2 - half for n in far.shape)
    block = _convolve(centres, far, start, (height, width))
    above = block != 0
    return jnp.where(above, spread / jnp.where(above, block, 1), 1)


def _convolve(
    x: Array, y: Array, start: tuple[int, ...], shape: tuple[int, int]
) -> Array:
    """The entries of shape `shape` from index `start` of the convolution of `x`
    and `y`, which has `x.shape + y.shape - 1` entries.

    The smaller array is the kernel: the cost is the entries times its size.
    """
    if x.size < y.size:
        x, y = y, x
    # The convolution's entry `k - 1 - low` is the first one that a
    # cross-correlation with the flipped kernel of size `k` and padding `low`
    # computes.
    padding = [
        (k - 1 - s, n + s - m) for m, k, s, n in zip(x.shape, y.shape, start, shape)
    ]
    return lax.conv_general_dilated(
        x[None, None],
        y[None, None, ::-1, ::-1],
        (1, 1),
        padding,
        precision=lax.Precision.HIGHEST,
    )[0, 0]


@eqx.filter_jit
def _permeability_calibration(
    windowed: WindowedAnalysis, state: Any
) -> tuple[tuple[Array, Array], Any]:
    """Block correction gradients on a tile of the maximum permeability.

    With quality 1 and one source at index `shape // 2` of the tile: the
    permeability gradients of the proximity-weighted quality reachable from
    it, and of the same without the pairs within its block, set to 0 at or
    below its noise floor, `1e-6` times its maximum. Returns them and the array
    part of the solver state, reinitialized for the tile from the array part
    `state`.
    """
    level = lax.stop_gradient(jnp.max(windowed.permeability_raster))
    graph = windowed._uniform_graph(level, windowed._tile_sides)
    static = eqx.partition(windowed.state, eqx.is_array)[1]
    state = windowed.distance.reinit(graph, eqx.combine(state, static))
    ones = jnp.ones_like(graph.grid)
    sources = tuple(jnp.array([n // 2]) for n in graph.grid.shape)

    def reachable(permeability, block):
        excluded = _excluded(windowed.window_op, block)
        return _reachable(windowed, ones, permeability, sources, state, excluded)[0]

    pixels = jax.grad(reachable)(graph.grid, False)
    far = jax.grad(reachable)(graph.grid, True)
    far = jnp.where(jnp.abs(far) > 1e-6 * jnp.max(jnp.abs(far)), far, 0)
    return (pixels, far), eqx.partition(state, eqx.is_array)[0]


class ConnectivityAnalysis(WindowedAnalysis):
    r"""Compute landscape connectivity metrics for large rasters.

    Quantifies connectivity by summing pairwise proximity in the graph. When
    quality-weighted, retrieves the quantity:

    $$C = \sum_i \sum_{j \neq i} q_i \, K_{ij} \, q_j$$

    where $q_i$ is habitat quality at pixel $i$ and $K_{ij}$ is the proximity
    between pixels $i$ and $j$, a function `proximity` of their distance
    $d_{ij}$ computed in the window centred on pixel $i$, or in a tile that
    holds it (see
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis]). Pairs
    farther apart than `dependency_range` $D$ along either axis do not
    contribute.

    By default, $K_{ij} = e^{-r_{ij}/\alpha}$ with $\alpha = D/4$, where
    $r_{ij}$ is the distance in equivalent pixels: the number of pixels through
    habitat of permeability 1 that gives the distance $d_{ij}$. This is the
    negative-exponential dispersal kernel of metapopulation theory
    ([Hanski & Ovaskainen, 2000](https://doi.org/10.1038/35008063)). In two
    dimensions, its mean dispersal distance is $2\alpha = D/2$, and 91 % of its
    mass lies within distance $4\alpha = D$ of the source.

    With coarsening, each pixel takes the term $\sum_{j \neq c} K_{cj} q_j$ of
    block centres $c$, interpolated with the weights $T_c(i)$ of
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis]:

    $$C \approx \sum_c \Big(\sum_i T_c(i) \, q_i\Big) \sum_{j \neq c} K_{cj} \, q_j$$

    Pairs within a block count, and no calibration is needed.

    Uses windowed processing for memory efficiency. See
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis] for
    windowing parameters, coarsening, edge padding and the permeability floor.

    !!! note "Distance units"
        A custom `proximity` receives distances in the units of `distance`:

        - `EuclideanDistance`: pixels.
        - `LCPDistance`: edge costs `1 / permeability` summed along the path,
          hence pixels at permeability 1.
        - `ResistanceDistance`: effective resistance, with edge resistance
          `1 / permeability`. It grows like the logarithm of the pixel
          distance: at permeability 1 in a window of 41 pixels a side, it is
          0.50, 1.28 and 1.89 at 1, 10 and 20 pixels from the centre. A kernel
          `exp(-d / c)` of the resistance therefore decays as a power law in
          pixels, and the window edge sets its range.
        - `FreeEnergyDistance`: from the least-cost distance (large `theta`)
          to a multiple of the resistance (small `theta`).

        When the analysis is created, the default `proximity` reads equivalent
        pixels off one window of permeability 1: the distances from its centre
        along an axis, interpolated linearly between pixels and in proportion
        past the window edge. For `EuclideanDistance` and `LCPDistance`,
        $r_{ij} = d_{ij}$. Scale permeability so that 1 is optimal habitat.

    !!! example

        ```python
        from jaxscape import ConnectivityAnalysis, LCPDistance

        conn = ConnectivityAnalysis(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=LCPDistance(),
            dependency_range=20,  # Window half-width in pixels
            batch_size=1000,
        )
        connectivity_index = conn.run(q_weighted=True)
        ```

    !!! note "Contiguity"
        The underlying graph is constructed using rook contiguity (4-neighbor
        connectivity), and edge weights are computed as the average permeability
        between adjacent pixels.
    """

    def run(self, q_weighted: bool = True) -> Array:
        """Compute the connectivity metric across the landscape.

        **Arguments:**

        - `q_weighted`: If `True`, compute quality-weighted connectivity (expected
          habitat amount). If `False`, compute unweighted proximity sum, with
          $q_i = 1$ inside the raster.

        **Returns:**

        Scalar landscape connectivity value.
        """
        initial = jnp.zeros(self._blocks, self.quality_raster.dtype)
        return jnp.sum(self._accumulate(_tile_connectivity, initial, q_weighted))


class SensitivityAnalysis(WindowedAnalysis):
    r"""Compute connectivity sensitivity via automatic differentiation.

    Calculates gradients $\partial C / \partial q_i$ or $\partial C / \partial p_i$
    showing how connectivity responds to changes in habitat quality or permeability
    at each pixel. Without coarsening, the result is the gradient of
    [`ConnectivityAnalysis.run`][jaxscape.connectivity_analysis.ConnectivityAnalysis.run]
    with the same arguments.

    With coarsening, the windows of block centres $c$ estimate it:

    - **Quality.** For a symmetric proximity $K$, the derivative of
      $\sum_i \sum_{j \neq i} q_i K_{ij} q_j$ is twice the term
      $\sum_{j \neq i} K_{ij} q_j$, which is interpolated from the block
      centres: one distance query per window, without differentiation.
      `EuclideanDistance`, `LCPDistance`, `ResistanceDistance` and
      `FreeEnergyDistance` are symmetric on the window graphs, whose edges
      average permeability. For other distances, the result is the gradient of
      the coarsened connectivity. Within `dependency_range` of the raster
      edges, the term falls steeply towards the edge and the interpolation
      underestimates it: by up to 20 % in a uniform landscape with
      `dependency_range` 8, the default proximity and `b = 3` or 5.
    - **Permeability.** The gradient of each tile's connectivity without the
      pairs within blocks,
      $\sum_c \big(\sum_i T_c(i) \, q_i\big) \sum_{j \notin \text{block}(c)} K_{cj} q_j$
      over its block centres $c$, is multiplied pixel by pixel by a correction
      for the block artifacts, after Omniscape's block correction:
      $\sum_c \sum_s w_c(s) \, g(\cdot - c - s)$ divided by
      $\sum_c g_\text{far}(\cdot - c) \sum_s w_c(s)$. Here $g$ and
      $g_\text{far}$ are the permeability gradients of
      $\sum_{j \neq c} K_{cj}$ and $\sum_{j \notin \text{block}} K_{cj}$ for one
      block centre $c$ at the centre of a tile of quality 1 and of the
      raster's maximum permeability, computed once per run, and the weights
      $w_c(s) = T_c(c + s)$ of the raster pixels $c + s$ select the pixel
      windows that each block centre stands for. The correction is 1 at or
      below its noise floor, where the denominator is at most `1e-6` times the
      maximum of $|g_\text{far}|$, for example where proximities underflow.
      In a uniform landscape, away from its edges, the corrected gradient of a
      block window is the weighted sum of the gradients of these pixel
      windows, except at or below the noise floor and in the band of up to
      $b - 1$ pixels that the pixel windows reach beyond the block window.
      The gradients in that band are lost, so the coarsened permeability
      sensitivity of windows is biased low. Measured in a uniform landscape
      with `dependency_range` 8 and the default proximity, away from the
      edges, it is 0.96 to 1.00 times the uncoarsened one for
      `ResistanceDistance` with `b = 5` and one block centre per batch (0.97
      to 1.00 with `b = 3`), and 0.99 to 1.00 for `LCPDistance`. Within
      `dependency_range` of the edges, where the calibration window differs
      most from the block windows, resistance sensitivity is as low as 0.7
      times the uncoarsened one in the outer rows. Tiles keep the band, as
      their calibration covers the whole tile; with `b = 5` and tiles of
      8 x 8 block centres, the coarsened resistance sensitivity is 0.88 to
      1.09 times the uncoarsened one on tiles of 8 x 8 pixels, 1.02 on
      average.

    Parameters and windowing strategy are identical to
    [`ConnectivityAnalysis`][jaxscape.connectivity_analysis.ConnectivityAnalysis],
    including the units of `proximity` and its default, the negative
    exponential of the distance in equivalent pixels with decay length
    `dependency_range / 4`.

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import ResistanceDistance, SensitivityAnalysis
        from jaxscape.solvers import CholmodSolver

        sens = SensitivityAnalysis(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=ResistanceDistance(solver=CholmodSolver()),
            dependency_range=20,
            batch_size=100,
        )

        d_quality = sens.run("quality", q_weighted=True)
        d_permeability = sens.run("permeability", q_weighted=True)

        # Identify conservation priorities
        high_impact = d_quality > jnp.percentile(d_quality, 90)
        ```
    """

    def run(self, var: str = "quality", q_weighted: bool = True) -> Array:
        """Compute connectivity gradients with respect to a landscape parameter.

        **Arguments:**

        - `var`: Parameter to differentiate: `"quality"` or `"permeability"`.
        - `q_weighted`: If `True`, compute quality-weighted connectivity gradients.
          If `False`, compute unweighted gradients.

        **Returns:**

        Gradient raster with the same shape as the input, showing sensitivity at
        each pixel. Pixels whose permeability is at or below the permeability
        floor have zero permeability sensitivity.
        """
        coarse = self.window_op.window_size > 1
        calibration = state = None
        if var == "quality":
            symmetric = isinstance(self.distance, _SYMMETRIC_DISTANCES)
            if coarse and q_weighted and symmetric:
                initial = jnp.zeros(self._blocks, self.quality_raster.dtype)
                reachable = self._accumulate(_tile_reachable_quality, initial)
                return 2 * self._interpolate(reachable)
            raster = self.quality_raster
        elif var == "permeability":
            if coarse:
                # The windows carry on from the calibration's solver state.
                state = eqx.partition(self.state, eqx.is_array)[0]
                calibration, state = _permeability_calibration(self, state)
            raster = self.permeability_raster
        else:
            raise ValueError("`var` must be either 'quality' or 'permeability'")

        gradient = self._accumulate(
            _tile_sensitivity,
            jnp.zeros_like(raster),
            var,
            q_weighted,
            calibration,
            state=state,
        )
        gradient = self._crop(gradient)
        if var == "permeability":
            gradient = jnp.where(
                self._crop(self.permeability_raster) > self.floor, gradient, 0
            )
        return gradient
