import jax.numpy as jnp
from jax import jit

def euclidean_distance_matrix(grid, res):
    """
    Calculate the Euclidean distance on CPU.
    Args:
    - coordinate_list: Array of shape (N, 2), representing coordinates of vertices.
    - res: Scaling factor.

    Returns:
    - Euclidean distance matrix scaled by `res`.
    """
    coordinate_list = grid.active_vertex_coordinate(jnp.arange(grid.nb_active()))
    X = coordinate_list[:, 0]
    Y = coordinate_list[:, 1]
    Xmat = X[:, None] - X[None, :]
    Ymat = Y[:, None] - Y[None, :]
    euclidean_distance = jnp.hypot(Xmat, Ymat)
    return euclidean_distance * res