"""The landscapes of the benchmark, and the sites and graph of E1."""

import numpy as np
from PIL import Image


def synthetic(side, correlation_length, habitat_threshold, seed):
    """Return the quality and permeability rasters of a synthetic landscape.

    The field `f` is a Gaussian random field with the covariance
    `exp(-r**2 / (2 * correlation_length**2))`, on a periodic grid, with ranks
    normalized to [0, 1]. Quality equals `f` where `f >= habitat_threshold`,
    and 0 elsewhere. Permeability is `10 ** (-2 * (1 - f))`, from 0.01 to 1;
    resistance is its inverse.

    **Arguments:**

    - `side`: Number of pixels along each side. A larger raster covers a larger
      area at the same grain.
    - `correlation_length`: Correlation length of the field, in pixels.
    - `habitat_threshold`: Rank above which a pixel is habitat; 0.7 makes 30 %
      of the pixels habitat.
    - `seed`: Seed of the field.

    **Returns:**

    Quality and permeability, both float64 arrays with shape `(side, side)`.
    """
    noise = np.random.default_rng(seed).standard_normal((side, side))
    frequency = np.fft.fftfreq(side)
    squared = frequency[:, None] ** 2 + frequency[None, :] ** 2
    kernel = np.exp(-((np.pi * correlation_length) ** 2) * squared)
    field = np.fft.ifft2(np.fft.fft2(noise) * kernel).real
    ranks = np.empty(side * side)
    ranks[np.argsort(field, axis=None)] = np.arange(side * side)
    field = ranks.reshape(side, side) / (side * side - 1)
    quality = np.where(field >= habitat_threshold, field, 0.0)
    permeability = 10.0 ** (-2.0 * (1.0 - field))
    return quality, permeability


def maryland(path, resistance, side=None):
    """Return the quality and permeability rasters of the Maryland case.

    **Arguments:**

    - `path`: The land cover raster, a GeoTIFF of NLCD classes.
    - `resistance`: The resistance of each class, keyed by the class as a
      string, as Omniscape's reclassification table gives it. Classes left out
      are barriers, of permeability 0.
    - `side`: Number of pixels along each side of the top-left square to keep,
      or `None` for the whole raster.

    **Returns:**

    Quality, 1 on the pixels of resistance 1 and 0 elsewhere, and permeability,
    the inverse of resistance: float64 arrays of the raster's shape.
    """
    land_cover = np.asarray(Image.open(path))[:side, :side]
    table = np.full(256, np.inf)
    for land_class, value in resistance.items():
        table[int(land_class)] = value
    resistance = table[land_cover]
    return (resistance == 1).astype(float), 1 / resistance


def resistance(permeability):
    """Return the resistance raster: 1 / permeability, and NaN on barriers.

    Barriers are the pixels of permeability 0.
    """
    barrier = permeability == 0
    return np.where(barrier, np.nan, 1 / np.where(barrier, 1, permeability))


def sites(quality, number, seed):
    """Return `number` habitat pixels drawn at random, without replacement.

    **Returns:**

    Row and column of each site, an integer array with shape `(number, 2)`, in
    the order of their vertex indices.
    """
    habitat = np.flatnonzero(quality > 0)
    drawn = np.sort(np.random.default_rng(seed).choice(habitat, number, replace=False))
    return np.column_stack(np.unravel_index(drawn, quality.shape))


def conductance(x, y):
    """Return the conductance of the edge between pixels of permeability `x` and `y`.

    The resistance of an edge is the mean resistance of its two pixels, so that
    its conductance is the harmonic mean of their permeabilities. Every tool of
    E1 builds its graph with this conductance and rook contiguity, which
    Circuitscape.jl, ConScape.jl, `gdistance` and `skimage.graph.MCP_Geometric`
    all offer. JAXScape's windowed analyses take the arithmetic mean of the
    permeabilities, which ConScape.jl and `MCP_Geometric` do not offer.
    """
    return 2 / (1 / x + 1 / y)
