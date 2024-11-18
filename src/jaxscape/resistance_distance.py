import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv
from jax.experimental import sparse
from jax import jit
from jaxscape.distance import AbstractDistance
from typing import Union
import equinox as eqx
from typing import Callable, Union
from jaxscape.utils import bcoo_diag

class ResistanceDistance(AbstractDistance):        
    @eqx.filter_jit
    def __call__(self, grid):
        A = grid.get_adjacency_matrix()
        return resistance_distance(A)

def graph_laplacian(A):
    """
    Computes the graph Laplacian given an adjacency matrix A.
    """
    D = bcoo_diag(A.sum(axis=1).todense())  # Degree matrix
    L = D - A  # Laplacian matrix
    return L

@jit
def resistance_distance(A):
    # see implementation here: https://networkx.org/documentation/stable/_modules/networkx/algorithms/distance_measures.html#resistance_distance
    """
    Computes the resistance distance matrix.
    Args:
        A: Adjacency matrix (sparse BCOO).
    Returns:
        Resistance distance matrix.
    """
    L = graph_laplacian(A)
    L_pseudo = pinv(L.todense())  # Moore-Penrose pseudoinverse of Laplacian

    # Compute resistance distances
    diag_Lp = jnp.diag(L_pseudo)
    R = diag_Lp[:, None] + diag_Lp[None, :] - L_pseudo - L_pseudo.T

    return R
