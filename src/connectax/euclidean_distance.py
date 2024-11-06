import jax.numpy as jnp
from jax import jit
from connectax.gridgraph import GridGraph

class EuclideanGridGraph(GridGraph):
    def __init__(self, **kwargs):
        """
        A grid graph where distance returns `euclidean_distance`.
        """
        super().__init__(**kwargs)
        
    def get_distance_matrix(self, res):
        return euclidean_distance(self, res)

def euclidean_distance(grid, res):
    """
    Calculate the Euclidean distance on CPU.
    Args:
    - coordinate_list: Array of shape (N, 2), representing coordinates of vertices.
    - res: Scaling factor.

    Returns:
    - Euclidean distance matrix scaled by `res`.
    """
    coordinate_list = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active()))
    X = coordinate_list[:, 0]
    Y = coordinate_list[:, 1]
    Xmat = X[:, None] - X[None, :]
    Ymat = Y[:, None] - Y[None, :]
    euclidean_distance = jnp.hypot(Xmat, Ymat)
    return euclidean_distance * res