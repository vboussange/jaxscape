import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv
from jax.experimental.sparse import BCOO
from jaxscape.distance import AbstractDistance
import equinox as eqx
import lineax as lx
from jaxscape.utils import graph_laplacian

class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances from all to `targets`, or to all if `targets` is None.
    """     
    @eqx.filter_jit
    def __call__(self, grid, targets=None):
        A = grid.get_adjacency_matrix()
        # A = mapnz(A, lambda x: 1/x)
        if targets is None:
            return resistance_distance(A)
        else:
            if targets.ndim == 1:
                # This is a hack to get the resistance distance to targets, but it is not efficient
                return resistance_distance(A)[:, targets]
            elif targets.ndim == 2:
                landmark_indices = grid.coord_to_active_vertex_index(
                    targets[:, 0], targets[:, 1]
                )
                return resistance_distance(A)[:, landmark_indices]
            else:
                raise ValueError("Invalid landmarks dimensions")
            
@eqx.filter_jit
def resistance_distance(A: BCOO):
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
def calculate_voltage(A, currents, solver=lx.SVD()):
    """
    Computes the voltage given adjacency matrix and currents.

    TODO: to be implemented.
    """
    raise NotImplementedError
    # L = graph_laplacian(A)
    # Lreg = ...
    ## This returns Vjl with l being landmarks, but we also need Vjj
    ## This is tricky to obtain
    # operator = lx.MatrixLinearOperator(Lreg)
    # state = solver.init(operator, options={})
    # you may want to store state for future use
    # x = lx.linear_solve(operator, currents, solver=solver, state=state).value
    # x
