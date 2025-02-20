import jax.numpy as jnp
from jax.numpy.linalg import pinv
from jaxscape.distance import AbstractDistance
import equinox as eqx
from jaxscape.utils import graph_laplacian
from jax.experimental.sparse import BCOO

class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances from all to `targets`, or to all if `targets` is None.
    """         
    @eqx.filter_jit
    def __call__(self, grid, sources=None, targets=None):
        A = grid.get_adjacency_matrix()

        sources = jnp.arange(grid.nv) if sources is None else sources
        targets = jnp.arange(grid.nv) if targets is None else targets

        sources = grid.coord_to_index(sources[:, 0], sources[:, 1]) if sources.ndim == 2 else sources
        targets = grid.coord_to_index(targets[:, 0], targets[:, 1]) if targets.ndim == 2 else targets

        R = p_inv_resistance_distance(A)
        return R[sources, :][:, targets]
            
@eqx.filter_jit
def p_inv_resistance_distance(A: BCOO):
    # see implementation here: https://networkx.org/documentation/stable/_modules/networkx/algorithms/distance_measures.html#resistance_distance
    """
    Computes the resistance distance matrix.
    Args:
        A: Adjacency matrix (sparse BCOO).
    Returns:
        Resistance distance matrix.
    """
    L = graph_laplacian(A)
    # V = pinv(L.todense(), hermitian=True)  # TODO: in the future, we want to only permit undirected graphs for resistance distance calculation, hence enforce `hermitian`
    V = pinv(L.todense())  # Moore-Penrose pseudoinverse of Laplacian

    # Compute resistance distances
    Vuu = jnp.diag(V)
    R = Vuu[:, None] + Vuu[None, :] - V - V.T

    return R