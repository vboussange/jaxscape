from collections.abc import Callable
from typing import Union

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from tqdm import tqdm

from jaxscape.distance import AbstractDistance
from jaxscape.graph import GridGraph
from jaxscape.window_operation import WindowOperation
from jaxscape.windowed_analysis import WindowedAnalysis


def _connectivity(
    quality_raster: Array,
    permeability_raster: Array,
    window_op: WindowOperation,
    distance: AbstractDistance,
    proximity: Callable[[Array], Array],
    q_weighted: bool,
) -> Union[Array, float]:
    """Internal function to compute connectivity within a window."""
    grid = GridGraph(grid=permeability_raster, fun=lambda x, y: (x + y) / 2)
    window_center = jnp.array(
        [[permeability_raster.shape[0] // 2, permeability_raster.shape[1] // 2]]
    )
    window_center_index = grid.coord_to_index(window_center[:, 0], window_center[:, 1])

    x_core_window, y_core_window = jnp.meshgrid(
        jnp.arange(
            window_op.buffer_size, window_op.window_size + window_op.buffer_size
        ),
        jnp.arange(
            window_op.buffer_size, window_op.window_size + window_op.buffer_size
        ),
    )
    window_core_indices = grid.coord_to_index(x_core_window, y_core_window)

    dist = distance(grid, sources=window_center_index).flatten()
    K = proximity(dist)

    K = K.at[window_core_indices].set(0)

    if q_weighted:
        core_window_qual = lax.dynamic_slice(
            quality_raster,
            start_indices=(window_op.buffer_size, window_op.buffer_size),
            slice_sizes=(window_op.window_size, window_op.window_size),
        )

        q = grid.array_to_node_values(quality_raster)

        qKqT = jnp.sum(core_window_qual) * (K @ q.T)

        return qKqT
    else:
        return jnp.sum(K)


_connectivity_vmap = eqx.filter_vmap(_connectivity, in_axes=(0, 0, 0, None, None, None))


class ConnectivityAnalysis(WindowedAnalysis):
    r"""Compute landscape connectivity metrics for large rasters.

    Quantifies connectivity by suming pairwise proximity in the graph. When
    quality-weighted, retrieves the quantity:

    $$C = \sum_i \sum_{j \neq i} q_i \, K_{ij} \, q_j$$

    where $q_i$ is habitat quality at pixel $i$ and $K_{ij} = f(d_{ij})$ is the
    proximity between pixels based on distance $d_{ij}$.

    Uses windowed processing for memory efficiency. See
    [`WindowedAnalysis`][jaxscape.windowed_analysis.WindowedAnalysis] for
    windowing parameters and strategy.

    !!! example

        ```python
        from jaxscape import ConnectivityAnalysis, LCPDistance import
        jax.numpy as jnp

        # Setup landscape D = 20  # Dispersal range distance = LCPDistance()
        proximity = lambda dist: jnp.exp(-dist / D)

        # Pad rasters to avoid edge effects quality_padded = jnp.pad(quality, D,
        constant_values=0) permeability_padded = jnp.pad(permeability, D,
        constant_values=0)

        # Initialize analysis conn = ConnectivityAnalysis(
            quality_raster=quality_padded,
            permeability_raster=permeability_padded, distance=distance,
            proximity=proximity, dependency_range=D, coarsening_factor=0.2,
            batch_size=50
        )

        # Compute connectivity connectivity_index = conn.run(q_weighted=True)
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
        output = jnp.array(0.0)
        for xy_batch, quality_batch in tqdm(
            self.batch_op.lazy_iterator(self.quality_raster),
            desc="Batch progress",
            total=self.batch_op.nb_steps,
            miniters=max(1, self.batch_op.nb_steps // 100),
        ):
            permeability_batch = self.batch_op.extract_total_window(
                xy_batch, self.permeability_raster
            )
            _, quality_windows = self.window_op.eager_iterator(quality_batch)
            _, permeability_windows = self.window_op.eager_iterator(permeability_batch)
            output += jnp.sum(
                _connectivity_vmap(
                    quality_windows,
                    permeability_windows,
                    self.window_op,
                    self.distance,
                    self.proximity,
                    q_weighted,
                )
            )
        return output


d_quality = eqx.filter_jit(eqx.filter_grad(_connectivity))
d_quality_vmap = eqx.filter_vmap(d_quality, in_axes=(0, 0, 0, None, None, None))


@eqx.filter_jit
@eqx.filter_grad
def d_permeability(
    permeability_raster: Array, quality_raster: Array, *args, **kwargs
) -> float:
    return _connectivity(quality_raster, permeability_raster, *args, **kwargs)


d_permeability_vmap = eqx.filter_vmap(
    d_permeability, in_axes=(0, 0, 0, None, None, None)
)


class SensitivityAnalysis(WindowedAnalysis):
    r"""Compute connectivity sensitivity via automatic differentiation.

    Calculates gradients $\partial C / \partial q_i$ or $\partial C / \partial p_i$
    showing how connectivity responds to changes in habitat quality or permeability
    at each pixel. Uses JAX's automatic differentiation for efficient gradient computation.

    Parameters and windowing strategy are identical to [`ConnectivityAnalysis`][jaxscape.connectivity_analysis.ConnectivityAnalysis].

    !!! example

        ```python
        from jaxscape import SensitivityAnalysis, ResistanceDistance
        import jax.numpy as jnp

        # Setup (same as ConnectivityAnalysis)
        D = 20
        distance = ResistanceDistance()
        proximity = lambda dist: jnp.exp(-dist / D)

        sens = SensitivityAnalysis(
            quality_raster=quality_padded,
            permeability_raster=permeability_padded,
            distance=distance,
            proximity=proximity,
            dependency_range=D,
            coarsening_factor=0.2,
            batch_size=50
        )

        # Compute sensitivity gradients
        d_quality = sens.run("quality", q_weighted=True)
        d_permeability = sens.run("permeability", q_weighted=True)

        # Identify conservation priorities
        high_impact = d_quality > jnp.percentile(d_quality, 90)
        ```
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(args) > 0:
            self.original_shape = args[0].shape
        else:
            self.original_shape = np.array(
                kwargs.get("quality_raster", None).shape
            )  # todo: do we need np.array here?

    def _scan_fn(self, raster_buffer: Array, x: tuple) -> tuple:
        _xy, _rast = x
        raster_buffer = self.window_op.update_raster_with_window(
            _xy, raster_buffer, _rast, fun=jnp.add
        )
        return raster_buffer, None

    def run(self, var: str = "quality", q_weighted: bool = True) -> Array:
        """Compute connectivity gradients with respect to a landscape parameter.

        **Arguments:**

        - `var`: Parameter to differentiate: `"quality"` or `"permeability"`.
        - `q_weighted`: If `True`, compute quality-weighted connectivity gradients.
          If `False`, compute unweighted gradients.

        **Returns:**

        Gradient raster with the same shape as the input, showing sensitivity at each pixel.
        """
        if var == "quality":
            sensitivity_fun = d_quality_vmap
        elif var == "permeability":
            sensitivity_fun = d_permeability_vmap
        else:
            raise ValueError("`var` must be either 'quality' or 'permeability'")

        output = jnp.zeros_like(self.quality_raster)

        for xy_batch, quality_batch in tqdm(
            self.batch_op.lazy_iterator(self.quality_raster),
            desc="Batch progress",
            total=self.batch_op.nb_steps,
            miniters=max(1, self.batch_op.nb_steps // 100),
        ):
            permeability_batch = self.batch_op.extract_total_window(
                xy_batch, self.permeability_raster
            )
            xy, quality_windows = self.window_op.eager_iterator(quality_batch)
            _, permeability_windows = self.window_op.eager_iterator(permeability_batch)

            raster_buffer = jnp.zeros(
                (self.batch_op.total_window_size, self.batch_op.total_window_size)
            )
            res = sensitivity_fun(
                quality_windows,
                permeability_windows,
                self.window_op,
                self.distance,
                self.proximity,
                q_weighted,
            )

            # handling padding
            padding = jnp.all(
                xy_batch + xy + self.window_op.total_window_size <= self.original_shape,
                axis=1,
            )[:, None, None]
            res = res * padding

            raster_buffer, _ = lax.scan(self._scan_fn, raster_buffer, (xy, res))
            output = self.batch_op.update_raster_with_window(
                xy_batch, output, raster_buffer, fun=jnp.add
            )

        output = output[: self.original_shape[0], : self.original_shape[1]]
        return output
