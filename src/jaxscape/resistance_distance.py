import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv
from jaxscape.distance import AbstractDistance
import equinox as eqx
import lineax as lx
from jaxscape.utils import graph_laplacian, bcoo_diag, connected_component_labels
from jaxscape.solvers import batched_linear_solve

from jax.experimental.sparse import BCOO

class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances. 
    
    Attributes:
        solver: Optional lineax.AbstractLinearSolver. Must be compatible with
        BCOO matrices. We currently support jaxscape.solvers.CholmodSolver and
        jaxscape.solvers.PyAMGSolver. If None, uses pseudo-inverse method, which
        is very memory intensive for large graphs (densifies the Laplacian
        matrix).
    
    !!! Warning
        The graph must be undirected for resistance distance to be well-defined.
    """
    solver: tuple[None, lx.AbstractLinearSolver] = None
    
    @eqx.filter_jit
    def all_pairs_distance(self, grid):
        A = grid.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)
        else:
            nodes = jnp.arange(grid.nv)
            return lineax_solver_nodes_to_nodes_resistance_distance(A, nodes, self.solver)
    
    @eqx.filter_jit
    def nodes_to_nodes_distance(self, grid, nodes):
        A = grid.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)[nodes[:, None], nodes[None, :]]
        else:
            return lineax_solver_nodes_to_nodes_resistance_distance(A, nodes, self.solver)
        
    @eqx.filter_jit
    def sources_to_targets_distance(self, grid, sources, targets):
        R = self.all_pairs_distance(grid)
        return R[sources[:, None], targets[None, :]]

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
    # V = pinv(L.todense(), hermitian=True)  # TODO: in the future, we want a GridGraph to have a flag `symmetric`, and in this case enforce `hermitian`
    V = pinv(L.todense())  # Moore-Penrose pseudoinverse of Laplacian

    # Compute resistance distances
    Vuu = jnp.diag(V)
    R = Vuu[:, None] + Vuu[None, :] - V - V.T

    return R


@eqx.filter_jit
def lineax_solver_nodes_to_nodes_resistance_distance(A: BCOO, nodes, solver):
    """
    Computes pairwise resistance distance from `nodes` to `nodes`, returning a |`nodes`| x |`nodes`| matrix,
    using a lineax `solver`.
    Requires |`nodes`| linear solves.
    !!! Warning
        The graph must be undirected.
    """
    L = graph_laplacian(A)

    # Ground the last node to obtain a full-rank system (see: http://epubs.siam.org/doi/10.1137/050645452)
    L_reduced = L[:-1, :-1]

    nodes = nodes.astype(A.indices.dtype)

    node_basis = jax.nn.one_hot(nodes, L_reduced.shape[0], dtype=L_reduced.dtype).T # if node `L_reduced.shape[0]+1` is included in `nodes`, its one-hot will consists of the zero vector

    potentials = batched_linear_solve(L_reduced, node_basis, solver)

    potentials_ii = jnp.sum(node_basis * potentials, axis=0)
    potentials_ij = jnp.sum(node_basis[:, :, None] * potentials[:, None, :], axis=0)

    R = potentials_ii[:, None] + potentials_ii[None, :] - (potentials_ij + potentials_ij.T)
    return R