"""The synthetic landscape of the benchmark, and the sites of E1."""

import numpy as np


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


def sites(quality, number, seed):
    """Return `number` habitat pixels drawn at random, without replacement.

    **Returns:**

    Row and column of each site, an integer array with shape `(number, 2)`, in
    the order of their vertex indices.
    """
    habitat = np.flatnonzero(quality > 0)
    drawn = np.sort(np.random.default_rng(seed).choice(habitat, number, replace=False))
    return np.column_stack(np.unravel_index(drawn, quality.shape))
