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

    where $q_i$ is habitat quality at pixel $i$ and $K_{ij} = f(d_{ij})$ is the
    proximity between pixels based on distance $d_{ij}$, computed in the window
    centred on pixel $i$. Pairs farther apart than `dependency_range` along
    either axis do not contribute.

    Uses windowed processing for memory efficiency. See
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis] for
    windowing parameters, edge padding and the permeability floor.

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
