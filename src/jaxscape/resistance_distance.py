import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv
import jax.experimental.sparse as jsp
from jax.experimental.sparse import BCOO, sparsify
from jaxscape.distance import AbstractDistance
import equinox as eqx
import lineax as lx
from jaxscape.utils import graph_laplacian
from jaxscape.linear_solve import SparseMatrixLinearOperator
from jaxscape.utils import bcoo_diag, bcoo_at_set

class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances from all to `targets`, or to all if `targets` is None.
    """ 
    solver: lx.AbstractLinearSolver
    def __init__(self, solver=None):
        self.solver = solver
        
    @eqx.filter_jit
    def __call__(self, grid, sources=None, targets=None):
        A = grid.get_adjacency_matrix()

        if self.solver is None:
            sources = jnp.arange(grid.nb_active) if sources is None else sources
            targets = jnp.arange(grid.nb_active) if targets is None else targets

            sources = grid.coord_to_active_vertex_index(sources[:, 0], sources[:, 1]) if sources.ndim == 2 else sources
            targets = grid.coord_to_active_vertex_index(targets[:, 0], targets[:, 1]) if targets.ndim == 2 else targets

            R = p_inv_resistance_distance(A)
            return R[sources, :][:, targets]
        elif (not self.solver is None )and (sources is None):
            # all to few with indirect solves
            R = vmap_lineax_solver_resistance_distance(A, targets, self.solver)
            return R
        else:
            raise ValueError("Method not implemented")

            
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


@eqx.filter_jit
# TODO: to test
def lineax_solver_resistance_distance(A, target, solver):
    assert len(target) == 1
    target = target.astype(A.indices.dtype)
    
    # maybe check that A is square
    n = A.shape[0]
    Pc = A.sum(axis=1).todense()
    Pc = Pc.at[target].set(0)
    I = bcoo_diag(jnp.ones(n))
    L=I-A

    L = bcoo_at_set(L, jnp.repeat(target, n), jnp.arange(n, dtype=A.indices.dtype), jnp.zeros(n, dtype=A.data.dtype))
    L = bcoo_at_set(L, target, target, jnp.array(jnp.ones(1, dtype=A.data.dtype)))
    operator = SparseMatrixLinearOperator(L)
    x = lx.linear_solve(operator, 
                        Pc, 
                        solver=solver, 
                        # options={"preconditioner":preconditioner}
                        ).value

    return x

vmap_lineax_solver_resistance_distance = eqx.filter_vmap(lineax_solver_resistance_distance, in_axes=(None, 0, None))

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
