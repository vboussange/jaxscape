from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from jaxscape.graph import GridGraph
from jaxscape.windowed_analysis import _average, WindowedAnalysis


def _window_connectivity(
    analysis: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    state: Any,
    q_weighted: bool,
) -> Array:
    """Connectivity between the core of a window and the rest of the window."""
    op = analysis.window_op
    size = op.total_window_size
    core = slice(op.buffer_size, op.buffer_size + op.window_size)
    is_core = jnp.zeros((size, size), bool).at[core, core].set(True)

    centre = jnp.array([(size // 2) * size + size // 2])
    graph = GridGraph(permeability, fun=_average)
    distances = analysis.distance(graph, sources=centre, state=state)
    # Pairs within the core do not count. A unit distance there keeps the
    # gradient finite for proximities that are singular at zero.
    distances = jnp.where(is_core, 1, distances.reshape(size, size))
    weight = quality if q_weighted else inside.astype(quality.dtype)
    outside = jnp.where(is_core, 0, analysis.proximity(distances) * weight)
    return jnp.sum(jnp.where(is_core, weight, 0)) * jnp.sum(outside)


def _window_sensitivity(
    analysis: WindowedAnalysis,
    quality: Array,
    permeability: Array,
    inside: Array,
    state: Any,
    var: str,
    q_weighted: bool,
) -> Array:
    """Gradient of the window connectivity with respect to the window of `var`."""

    def connectivity(quality, permeability):
        return _window_connectivity(
            analysis, quality, permeability, inside, state, q_weighted
        )

    argnums = 0 if var == "quality" else 1
    return jax.grad(connectivity, argnums=argnums)(quality, permeability)


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

    Uses windowed processing for memory efficiency. See
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis] for
    windowing parameters, edge padding and the permeability floor.

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
          habitat amount). If `False`, compute unweighted proximity sum.

        **Returns:**

        Scalar landscape connectivity value.
        """
        initial = jnp.zeros((), self.quality_raster.dtype)
        return self._accumulate(_window_connectivity, initial, q_weighted)


class SensitivityAnalysis(WindowedAnalysis):
    r"""Compute connectivity sensitivity via automatic differentiation.

    Calculates gradients $\partial C / \partial q_i$ or $\partial C / \partial p_i$
    showing how connectivity responds to changes in habitat quality or permeability
    at each pixel. Without coarsening, the result is the gradient of
    [`ConnectivityAnalysis.run`][jaxscape.connectivity_analysis.ConnectivityAnalysis.run]
    with the same arguments.

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
        each pixel. Pixels whose permeability is below the floor have zero
        permeability sensitivity.
        """
        if var == "quality":
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
