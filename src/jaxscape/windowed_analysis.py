from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Union

import jax.numpy as jnp
from jax import Array

from jaxscape.distance import AbstractDistance
from jaxscape.utils import padding
from jaxscape.window_operation import WindowOperation


class WindowedAnalysis(ABC):
    """Base class for windowed connectivity analyses on large rasters.

    Processes landscapes through hierarchical decomposition: batches (processed sequentially)
    contain windows (processed in parallel) with buffer zones for spatial dependencies.

    **Parameters:**

    - `quality_raster`: Habitat quality values.
    - `permeability_raster`: Movement permeability, or function `quality -> permeability`.
    - `distance`: Distance metric (e.g., `LCPDistance()`, `ResistanceDistance()`).
    - `proximity`: Distance-to-proximity transform (e.g., `lambda d: jnp.exp(-d/D)`).
    - `coarsening_factor`: Spatial coarsening in [0, 1]. 0 = finest resolution, higher = faster.
    - `dependency_range`: Spatial dependency range in pixels (buffer size).
    - `batch_size`: Number of coarsened windows per batch (higher = more memory).

    !!! example

        ```python
        from jaxscape.connectivity_analysis import WindowedAnalysis

        class CustomConnectivity(WindowedAnalysis):
            def run(self, **kwargs):
                result = 0.0
                for xy_batch, quality_batch in self.batch_op.lazy_iterator(self.quality_raster):
                    result += process_batch(quality_batch)
                return result
        ```
    """

    def __init__(
        self,
        quality_raster: Array,
        permeability_raster: Union[Array, Callable[[Array], Array]],
        distance: AbstractDistance,
        proximity: Callable[[Array], Array],
        dependency_range: int,
        batch_size: int,
        coarsening_factor: float = 0.0,
    ):
        assert 0 <= coarsening_factor <= 1
        assert isinstance(dependency_range, int)
        assert isinstance(batch_size, int)

        coarsening = int(jnp.ceil(dependency_range * coarsening_factor))
        if coarsening % 2 == 0:
            coarsening += 1

        buffer_size = int(dependency_range - (coarsening - 1) / 2)
        if buffer_size < 1:
            raise ValueError("Buffer size too small.")

        batch_window_size = batch_size * coarsening

        if callable(permeability_raster):
            permeability_raster = permeability_raster(quality_raster)
        else:
            permeability_raster = permeability_raster
        assert quality_raster.shape == permeability_raster.shape

        quality_padded = padding(quality_raster, buffer_size, batch_window_size)
        permeability_padded = padding(
            permeability_raster, buffer_size, batch_window_size
        )

        batch_op = WindowOperation(
            shape=quality_padded.shape,
            window_size=batch_window_size,
            buffer_size=buffer_size,
        )
        window_op = WindowOperation(
            shape=(batch_op.total_window_size, batch_op.total_window_size),
            window_size=coarsening,
            buffer_size=buffer_size,
        )

        self.quality_raster = quality_padded
        self.permeability_raster = permeability_padded
        self.distance = distance
        self.proximity = proximity
        self.batch_op = batch_op
        self.window_op = window_op
        # self.shape = quality_raster.shape

    @abstractmethod
    def run(self, **kwargs) -> Array:
        """Run the connectivity analysis. Must be implemented by subclasses."""
        pass
