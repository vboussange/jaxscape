import jax.numpy as jnp
from jax import jit
from jaxscape.distance import AbstractDistance
import equinox as eqx

class EuclideanDistance(AbstractDistance):        
    def __call__(self, grid, sources=None, targets=None):
        if sources is None:
            sources = jnp.arange(grid.nv)
            
        if targets is None:
            targets = jnp.arange(grid.nv)
            
        coordinate_list = []
        for nodes in [sources, targets]:
            if nodes.ndim == 1:
                # already vertex indices
                coordinate_list.append(grid.index_to_coord(nodes))
            elif nodes.ndim == 2:
                coordinate_list.append(nodes)
                
        return euclidean_distance(coordinate_list[0], coordinate_list[1])
                
            
@eqx.filter_jit
def euclidean_distance(coordinate_list_sources, coordinate_list_target):
    """Computes the euclidean distance between two sets of coordinates. 
    `coordinate_list_sources` must be of size (N, 2) and 
    `coordinate_list_target` must be of size (M, 2)."""
    Xmat = coordinate_list_sources[:, 0, None] - coordinate_list_target[None, :, 0]
    Ymat = coordinate_list_sources[:, 1, None] - coordinate_list_target[None, :, 1]
    euclidean_distance = jnp.hypot(Xmat, Ymat)
    return euclidean_distance