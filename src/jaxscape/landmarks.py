import equinox as eqx
from jax import Array, numpy as jnp
from jax.experimental.sparse import BCOO

from jaxscape.graph import GridGraph


@eqx.filter_jit
def sum_neighborhood(grid: GridGraph, xy: Array, npix: int) -> Array:
    """
    Computes the sum of pixels within an npix x npix neighborhood around each target in xy.
    Pixels outside the grid boundaries are treated as NaN and ignored.
    """

    # Generate the relative offsets for the neighborhood
    # TODO: here we can surely make something more efficient
    offsets = jnp.array(
        [(i, j) for i in range(-npix, npix + 1) for j in range(-npix, npix + 1)]
    )  # Shape: (npix*npix, 2)

    # Compute absolute positions for each target in xy
    positions = xy[:, None, :] + offsets[None, :, :]  # Shape: (n, npix*npix, 2)

    # Create a mask for positions within grid boundaries
    mask = (
        (positions[:, :, 0] >= 0)
        & (positions[:, :, 0] < grid.height)
        & (positions[:, :, 1] >= 0)
        & (positions[:, :, 1] < grid.width)
    )  # Shape: (n, npix*npix)

    # Clip positions to valid indices and convert to integers
    positions_clipped = jnp.stack(
        [
            jnp.clip(positions[:, :, 0], 0, grid.height - 1),
            jnp.clip(positions[:, :, 1], 0, grid.width - 1),
        ],
        axis=-1,
    ).astype(int)

    values = grid.grid[
        positions_clipped[:, :, 0], positions_clipped[:, :, 1]
    ]  # Shape: (n, npix*npix)
    values = jnp.where(mask, values, jnp.nan)
    sums = jnp.nansum(values, axis=1)
    return sums


@eqx.filter_jit
def coarse_graining(grid: GridGraph, buffer_size: int) -> BCOO:
    """
    Creates a coarse-grained matrix of target qualities by aggregating npix pixels
    into a single central pixel.
    See https://docs.circuitscape.org/Omniscape.jl/latest/usage/#General-Options for documentation.
    """
    assert isinstance(buffer_size, int)
    assert (
        grid.height % (2 * buffer_size + 1) == 0
    ), f"`grid height`  must be divisible by `(2*buffer_size + 1)`, consider padding the raster data."
    assert (
        grid.width % (2 * buffer_size + 1) == 0
    ), f"`grid width`  must be divisible by `(2*buffer_size + 1)`, consider padding the raster data."

    step = 2 * buffer_size + 1

    row_indices = jnp.arange(buffer_size, grid.height - buffer_size + 1, step)
    col_indices = jnp.arange(buffer_size, grid.width - buffer_size + 1, step)
    x, y = jnp.meshgrid(row_indices, col_indices)

    # potential landmarks
    xy = jnp.column_stack((x.ravel(), y.ravel()))[:, :]

    values = sum_neighborhood(grid, xy, buffer_size) / ((2 * buffer_size + 1) ** 2)
    coarse_target_matrix = BCOO(
        (values, xy), shape=(grid.height, grid.width)
    )  # dimension to be checked

    return coarse_target_matrix
