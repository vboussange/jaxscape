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
    """
    solver: tuple[None, lx.AbstractLinearSolver] = None
    
    @eqx.filter_jit
    def nodes_to_nodes_distance(self, grid, nodes):
        A = grid.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)[nodes[:, None], nodes[None, :]]
        else:
            return lineax_solver_nodes_to_nodes_resistance_distance(A, nodes, self.solver)
        
    @eqx.filter_jit
    def sources_to_targets_distance(self, grid, sources, targets):
        A = grid.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)[sources[:, None], targets[None, :]]
        else:
            return lineax_solver_sources_to_targets_resistance_distance(A, sources, targets, self.solver)
             
    @eqx.filter_jit
    def all_pairs_distance(self, grid):
        A = grid.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)
        else:
            nodes = jnp.arange(grid.nv)
            return lineax_solver_nodes_to_nodes_resistance_distance(A, nodes, self.solver)
    
    
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
def lineax_solver_sources_to_targets_resistance_distance(A: BCOO, sources, targets, solver):
    """
    Computes resistance distance from `sources` to `targets` using a lineax `solver`.
    Requires |`sources`| x |`targets`| linear solves. This is more
    memory-efficient for large graphs than p-inv, which densifies the Laplacian.
    !!! Warning
        The graph must be undirected.
    """
    L = graph_laplacian(A)
    
    # grounding last node; see sec 2.4 in http://epubs.siam.org/doi/10.1137/050645452
    L_reduced = L[:-1, :-1]
    n = L_reduced.shape[0]
    
    # resistance computation via linear solves
    sources = sources.astype(A.indices.dtype)
    targets = targets.astype(A.indices.dtype)

    source_basis = jax.nn.one_hot(sources, n, dtype=L_reduced.dtype).T # if node n+1 is a source, its one-hot vector is all zeros

    def solve_for_target(target):
        e_target = jax.nn.one_hot(target, n, dtype=L_reduced.dtype) # if node n+1 is a target, its one-hot vector is all zeros
        rhs = e_target[:, None] - source_basis
        potentials = batched_linear_solve(L_reduced, rhs, solver) # for now, does not work
        # compute dot product per RHS column (energy = (e_target - e_source)^T * potentials)
        return jnp.sum(rhs * potentials, axis=0)

    resistances = jax.vmap(solve_for_target)(targets)  # shape (|targets|, |sources|)
    R = resistances.T

    # Masking out impossible distances
    # Ideally, this should be done before the linear solves to avoid unnecessary computations.
    # connected component labels (label propagation)
    # component_labels = connected_component_labels(A)
    # source_components = component_labels[sources]
    # target_components = component_labels[targets]
    # component_mask = source_components[:, None] == target_components[None, :]
    # R = jnp.where(component_mask, R, jnp.inf)
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

    rhs = jax.nn.one_hot(nodes, L_reduced.shape[0], dtype=L_reduced.dtype).T

    potentials = batched_linear_solve(L_reduced, rhs, solver)

    diagonal = potentials[nodes, jnp.arange(nodes.shape[0])]
    row_potentials = potentials[nodes, :]

    R = diagonal[:, None] + diagonal[None, :] - (row_potentials + row_potentials.T)
    return R