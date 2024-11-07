import jax.numpy as jnp
from jax import jit
from connectax.gridgraph import GridGraph
from connectax.distance import Distance

class EuclideanDistance(Distance):
    def __init__(self, res):
        """
        Calculate the Euclidean distance.
        """
        super().__init__()
        self.res = res
        
    def __call__(self, grid):
        return euclidean_distance(grid, self.res)

@jit
def euclidean_distance(grid, res):
    coordinate_list = grid.active_vertex_index_to_coord(jnp.arange(grid.nb_active))
    X = coordinate_list[:, 0]
    Y = coordinate_list[:, 1]
    Xmat = X[:, None] - X[None, :]
    Ymat = Y[:, None] - Y[None, :]
    euclidean_distance = jnp.hypot(Xmat, Ymat)
    return euclidean_distance * res