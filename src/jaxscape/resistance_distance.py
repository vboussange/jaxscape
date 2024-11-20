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
import lineax as lx
from jax import vmap

class ResistanceDistance(AbstractDistance):        
    @eqx.filter_jit
    def __call__(self, grid, landmarks=None):
        A = grid.get_adjacency_matrix()
        if landmarks is None:
            return _full_resistance_distance(A)
        else:
            landmark_indices = grid.coord_to_active_vertex_index(landmarks[:, 0], landmarks[:, 1])
            return _landmark_resistance_distance(A, landmark_indices)

def graph_laplacian(A):
    """
    Computes the graph Laplacian given an adjacency matrix A.
    """
    D = bcoo_diag(A.sum(axis=1).todense())  # Degree matrix
    L = D - A  # Laplacian matrix
    return L

@eqx.filter_jit
def _full_resistance_distance(A):
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

@eqx.filter_jit
def _landmark_resistance_distance(A, landmark_indices):
    """
    Computes the resistance distances between all nodes and a set of landmark nodes.

    Args:
        A: Adjacency matrix (sparse BCOO).
        landmarks: Indices of landmarks (array-like).

    Returns:
        Resistance distance matrix of shape (n_nodes, n_landmarks).
    TODO: to be implemented. To avoid calculating full pseudo inverse of laplacian, 
    you need to find a way to approximate its diaonal terms. This can be done with 
    """
    L = graph_laplacian(A)
    n_nodes = L.shape[0]
    n_landmarks = len(landmark_indices)

    # Prepare the RHS matrix for all landmarks
    indices = jnp.column_stack([landmark_indices,jnp.arange(len(landmark_indices))])
    B = sparse.BCOO((jnp.ones(indices.shape[0]), indices), shape=(n_nodes, n_landmarks))
    # Solve L_g V = B
    V = pinv(L.todense())  # Moore-Penrose pseudoinverse of Laplacian

    ## This returns Vjl with l being landmarks, but we also need Vjj
    ## This is tricky to obtain
    # operator = lx.MatrixLinearOperator(A.todense())
    # solver = lx.SVD()
    # state = solver.init(operator, options={})
    # def solve_single(b):
    #     x = lx.linear_solve(operator, b, solver=solver, state=state).value
    #     return x
    # V = vmap(solve_single, in_axes=1, out_axes=1)(B.todense())
    # complete to calculate R, which must be a n_nodes x n_landmarks
    
    # return R
