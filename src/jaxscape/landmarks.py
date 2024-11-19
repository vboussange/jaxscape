from jax import numpy as jnp
from jax import jit
from jax.experimental.sparse import BCOO


# TODO: this is not jittable, because npix should be a constant
def sum_neighborhood(grid, xy, npix):
    """
    Computes the sum of pixels within an npix x npix neighborhood around each target in xy.
    Pixels outside the grid boundaries are treated as NaN and ignored.
    """
    half_width = npix // 2

    # Generate the relative offsets for the neighborhood
    # TODO: here we can surely make something more efficient
    offsets = jnp.array([
        (i, j)
        for i in range(-half_width, half_width + 1)
        for j in range(-half_width, half_width + 1)
    ])  # Shape: (npix*npix, 2)

    # Compute absolute positions for each target in xy
    positions = xy[:, None, :] + offsets[None, :, :]  # Shape: (n, npix*npix, 2)

    # Create a mask for positions within grid boundaries
    mask = (
        (positions[:, :, 0] >= 0) & (positions[:, :, 0] < grid.height) &
        (positions[:, :, 1] >= 0) & (positions[:, :, 1] < grid.width)
    )  # Shape: (n, npix*npix)

    # Clip positions to valid indices and convert to integers
    positions_clipped = jnp.stack([
        jnp.clip(positions[:, :, 0], 0, grid.height - 1),
        jnp.clip(positions[:, :, 1], 0, grid.width - 1)
    ], axis=-1).astype(int)

    # Use advanced indexing to gather values
    values = grid.vertex_weights[positions_clipped[:, :, 0], positions_clipped[:, :, 1]]  # Shape: (n, npix*npix)

    # Set values outside the grid to NaN
    values = jnp.where(mask, values, jnp.nan)

    # Compute the sum, ignoring NaNs
    sums = jnp.nansum(values, axis=1)

    return sums




def coarse_graining(grid, npix):
    """
    Creates a coarse-grained matrix of target qualities by aggregating npix pixels
    into a single central pixel.
    """
    half_width = npix // 2
    row_indices = jnp.arange(half_width, grid.height - half_width, npix)
    col_indices = jnp.arange(half_width, grid.width - half_width, npix)
    x, y = jnp.meshgrid(row_indices, col_indices)
    
    # potential landmarks
    xy = jnp.column_stack((x.ravel(), y.ravel()))[:, :, None]
    
    # stacking it up to active vertices
    xy_active_vertices = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active)).T[None, :, :]
    
    # finding closest active vertex
    dist = jnp.sum((xy_active_vertices-xy)**2, axis=1)
    idx_closest_vertex = jnp.argmin(dist, axis=1)
    
    # final landmarks (magnetised to closest active vertices)
    xy = xy_active_vertices[0, :, idx_closest_vertex]
    values = sum_neighborhood(grid, xy, npix)
    coarse_target_matrix = BCOO((values, xy), shape=(grid.height, grid.width)) # dimension to be checked

    return coarse_target_matrix
