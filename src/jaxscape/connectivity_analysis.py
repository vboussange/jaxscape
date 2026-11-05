import math
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.scipy.signal import convolve2d

from jaxscape.distance import AbstractDistance
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.free_energy_distance import FreeEnergyDistance
from jaxscape.graph import GridGraph
from jaxscape.lcp_distance import LCPDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.window_operation import WindowOperation
from jaxscape.windowed_analysis import _average, WindowedAnalysis


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


def _reachable(
    distance: AbstractDistance,
    proximity: Callable[[Array], Array],
    weight: Array,
    permeability: Array,
    state: Any,
    excluded: Array,
) -> Array:
    """Proximity-weighted `weight` reachable from the centre, outside `excluded`."""
    size = weight.shape[0]
    centre = jnp.array([(size // 2) * size + size // 2])
    graph = GridGraph(permeability, fun=_average)
    distances = distance(graph, sources=centre, state=state).reshape(size, size)
    # Excluded pixels, the centre among them, get a unit distance: it keeps the
    # gradient finite for proximities that are singular at zero.
    distances = jnp.where(excluded, 1, distances)
    return jnp.sum(jnp.where(excluded, 0, proximity(distances) * weight))


def _window_connectivity(
    windowed: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    tent: Array,
    state: Any,
    q_weighted: bool,
    block: bool = False,
) -> Array:
    """Connectivity of the pixels that `tent` interpolates from the window centre.

    Their weight, summed with the interpolation weights `tent`, times the
    proximity-weighted weight seen from the centre: outside the centre, or
    outside its block if `block`.
    """
    weight = quality if q_weighted else inside.astype(quality.dtype)
    excluded = _excluded(windowed.window_op, block)
    reachable = _reachable(
        windowed.distance, windowed.proximity, weight, permeability, state, excluded
    )
    return jnp.sum(tent * weight) * reachable


def _window_reachable_quality(
    windowed: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    tent: Array,
    state: Any,
) -> Array:
    """Proximity-weighted quality reachable from the window centre."""
    del inside, tent
    excluded = _excluded(windowed.window_op, block=False)
    return _reachable(
        windowed.distance, windowed.proximity, quality, permeability, state, excluded
    )


def _window_sensitivity(
    windowed: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    tent: Array,
    state: Any,
    var: str,
    q_weighted: bool,
    calibration: tuple[Array, Array] | None,
) -> Array:
    """Gradient of the window connectivity with respect to the window of `var`.

    With a `calibration` of the block correction, the permeability gradient
    leaves out the pairs within the block and is corrected for them.
    """
    block = calibration is not None

    def connectivity(quality, permeability):
        return _window_connectivity(
            windowed, quality, permeability, inside, tent, state, q_weighted, block
        )

    argnums = 1 if var == "permeability" else 0
    gradient = jax.grad(connectivity, argnums)(quality, permeability)
    if calibration is None:
        return gradient
    pixels, far = calibration
    # The pixel windows that the window stands for: the raster pixels around
    # its centre, weighted by their interpolation weights.
    b = windowed.window_op.window_size
    centre = tent.shape[0] // 2
    near = slice(centre - b + 1, centre + b)
    weights = (tent * inside)[near, near]
    spread = convolve2d(pixels, weights, mode="same", precision=lax.Precision.HIGHEST)
    above = far != 0
    factor = spread / (jnp.sum(weights) * jnp.where(above, far, 1))
    return jnp.where(above, factor, 1) * gradient


@eqx.filter_jit
def _permeability_calibration(
    windowed: WindowedAnalysis, state: Any
) -> tuple[tuple[Array, Array], Any]:
    """Block correction gradients on a window of the maximum permeability.

    With quality 1: the permeability gradient of the proximity-weighted
    quality reachable from the centre, and that gradient without the pairs
    within the block, set to 0 at or below its noise floor, `1e-6` times its
    maximum. Returns them and the array part of the solver state, reinitialized
    for the window from the array part `state`.
    """
    op = windowed.window_op
    level = lax.stop_gradient(jnp.max(windowed.permeability_raster))
    graph = windowed._uniform_graph(level)
    static = eqx.partition(windowed.state, eqx.is_array)[1]
    state = windowed.distance.reinit(graph, eqx.combine(state, static))
    ones = jnp.ones_like(graph.grid)

    def reachable(permeability, block):
        excluded = _excluded(op, block)
        return _reachable(
            windowed.distance, windowed.proximity, ones, permeability, state, excluded
        )

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
    $d_{ij}$ computed in the window centred on pixel $i$. Pairs farther apart
    than `dependency_range` $D$ along either axis do not contribute.

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
        initial = jnp.zeros(math.prod(self._blocks), self.quality_raster.dtype)
        return jnp.sum(self._accumulate(_window_connectivity, initial, q_weighted))


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
    - **Permeability.** The gradient of each block window's connectivity
      without the pairs within the block,
      $\big(\sum_i T_c(i) \, q_i\big) \sum_{j \notin \text{block}} K_{cj} q_j$,
      is multiplied pixel by pixel by a correction for the block artifacts,
      after Omniscape's block correction:
      $\sum_s w_c(s) \, g(\cdot - s) \big/ \big(g_\text{far} \sum_s w_c(s)\big)$.
      Here $g$ and $g_\text{far}$ are the permeability gradients of
      $\sum_{j \neq c} K_{cj}$ and $\sum_{j \notin \text{block}} K_{cj}$ on a
      window of quality 1 and of the raster's maximum permeability, computed
      once per run, and the weights $w_c(s) = T_c(c + s)$ of the raster pixels
      $c + s$ select the pixel windows that the block window stands for. The
      correction is 1 at or below its noise floor, where $|g_\text{far}|$ is at
      most `1e-6` times its maximum, for example where proximities underflow.
      In a uniform landscape, away from its edges, the corrected gradient of a
      block window is the weighted sum of the gradients of these pixel
      windows, except at or below the noise floor and in the band of up to
      $b - 1$ pixels that the pixel windows reach beyond the block window.
      The gradients in that band are lost, so the coarsened permeability
      sensitivity is biased low. Measured in a uniform landscape with
      `dependency_range` 8 and the default proximity, away from the edges, it
      is 0.96 to 1.00 times the uncoarsened one for `ResistanceDistance` with
      `b = 5` (0.97 to 1.00 with `b = 3`), and 0.99 to 1.00 for
      `LCPDistance`. Within `dependency_range` of the edges, where the
      calibration window differs most from the block windows, resistance
      sensitivity is as low as 0.7 times the uncoarsened one in the outer rows.

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
                initial = jnp.zeros(math.prod(self._blocks), self.quality_raster.dtype)
                reachable = self._accumulate(_window_reachable_quality, initial)
                return 2 * self._interpolate(reachable.reshape(self._blocks))
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
            _window_sensitivity,
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
