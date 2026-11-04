import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

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


def _own(
    distance: AbstractDistance,
    proximity: Callable[[Array], Array],
    weight: Array,
    permeability: Array,
    state: Any,
    excluded: Array,
) -> Array:
    """Proximity-weighted `weight` seen from the window centre, outside `excluded`."""
    size = weight.shape[0]
    centre = jnp.array([(size // 2) * size + size // 2])
    graph = GridGraph(permeability, fun=_average)
    distances = distance(graph, sources=centre, state=state).reshape(size, size)
    # Excluded pixels, the centre among them, get a unit distance: it keeps the
    # gradient finite for proximities that are singular at zero.
    distances = jnp.where(excluded, 1, distances)
    return jnp.sum(jnp.where(excluded, 0, proximity(distances) * weight))


def _window_connectivity(
    analysis: WindowedAnalysis,
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
    excluded = _excluded(analysis.window_op, block)
    own = _own(
        analysis.distance, analysis.proximity, weight, permeability, state, excluded
    )
    return jnp.sum(tent * weight) * own


def _window_own(
    analysis: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    tent: Array,
    state: Any,
) -> Array:
    """Proximity-weighted quality seen from the window centre."""
    del inside, tent
    excluded = _excluded(analysis.window_op, block=False)
    return _own(
        analysis.distance, analysis.proximity, quality, permeability, state, excluded
    )


def _window_sensitivity(
    analysis: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    tent: Array,
    state: Any,
    var: str,
    q_weighted: bool,
) -> Array:
    """Gradient of the window connectivity with respect to the window of `var`.

    The permeability gradient leaves out the pairs within the block.
    """
    block = var == "permeability"

    def connectivity(quality, permeability):
        return _window_connectivity(
            analysis, quality, permeability, inside, tent, state, q_weighted, block
        )

    argnums = 1 if block else 0
    return jax.grad(connectivity, argnums)(quality, permeability)


class ConnectivityAnalysis(WindowedAnalysis):
    r"""Compute landscape connectivity metrics for large rasters.

    Quantifies connectivity by summing pairwise proximity in the graph. When
    quality-weighted, retrieves the quantity:

    $$C = \sum_i \sum_{j \neq i} q_i \, K_{ij} \, q_j$$

    where $q_i$ is habitat quality at pixel $i$ and $K_{ij} = f(d_{ij})$ is the
    proximity between pixels based on distance $d_{ij}$, computed in the window
    centred on pixel $i$. Pairs farther apart than `dependency_range` along
    either axis do not contribute.

    With coarsening, each pixel takes the term $\sum_{j \neq c} K_{cj} q_j$ of
    block centres $c$, interpolated with the weights $T_c(i)$ of
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis]:

    $$C \approx \sum_c \Big(\sum_i T_c(i) \, q_i\Big) \sum_{j \neq c} K_{cj} \, q_j$$

    Pairs within a block count, and no calibration is needed.

    Uses windowed processing for memory efficiency. See
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis] for
    windowing parameters, coarsening, edge padding and the permeability floor.

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import ConnectivityAnalysis, LCPDistance

        D = 20  # Dispersal range in pixels
        conn = ConnectivityAnalysis(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=LCPDistance(),
            proximity=lambda dist: jnp.exp(-dist / D),
            dependency_range=D,
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
      the coarsened connectivity.
    - **Permeability.** The gradient of each block window's connectivity
      without the pairs within the block,
      $\big(\sum_i T_c(i) \, q_i\big) \sum_{j \notin \text{block}} K_{cj} q_j$.

    Parameters and windowing strategy are identical to
    [`ConnectivityAnalysis`][jaxscape.connectivity_analysis.ConnectivityAnalysis].

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import ResistanceDistance, SensitivityAnalysis
        from jaxscape.solvers import CholmodSolver

        D = 20
        sens = SensitivityAnalysis(
            quality_raster=quality,
            permeability_raster=permeability,
            distance=ResistanceDistance(solver=CholmodSolver()),
            proximity=lambda dist: jnp.exp(-dist / D),
            dependency_range=D,
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
        each pixel. Pixels whose permeability is below the floor have zero
        permeability sensitivity.
        """
        coarse = self.window_op.window_size > 1
        if var == "quality":
            symmetric = isinstance(self.distance, _SYMMETRIC_DISTANCES)
            if coarse and q_weighted and symmetric:
                initial = jnp.zeros(math.prod(self._blocks), self.quality_raster.dtype)
                own = self._accumulate(_window_own, initial)
                return 2 * self._interpolate(own.reshape(self._blocks))
            raster = self.quality_raster
        elif var == "permeability":
            raster = self.permeability_raster
        else:
            raise ValueError("`var` must be either 'quality' or 'permeability'")

        gradient = self._accumulate(
            _window_sensitivity, jnp.zeros_like(raster), var, q_weighted
        )
        gradient = self._crop(gradient)
        if var == "permeability":
            gradient = jnp.where(
                self._crop(self.permeability_raster) > self.floor, gradient, 0
            )
        return gradient
