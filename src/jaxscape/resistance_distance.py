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
    Compute the resistance distances from all to `targets`, or to all if `targets` is None.
    """
    solver: tuple[None, lx.AbstractLinearSolver] = None
             
    @eqx.filter_jit
    def __call__(self, grid, sources=None, targets=None):
        A = grid.get_adjacency_matrix()

        sources = jnp.arange(grid.nv) if sources is None else sources
        targets = jnp.arange(grid.nv) if targets is None else targets

        sources = grid.coord_to_index(sources[:, 0], sources[:, 1]) if sources.ndim == 2 else sources
        targets = grid.coord_to_index(targets[:, 0], targets[:, 1]) if targets.ndim == 2 else targets

        if self.solver is None:
            R = p_inv_resistance_distance(A)
            return R[sources, :][:, targets]
        else:
            return lineax_solver_resistance_distance(A, sources, targets, self.solver)

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
def lineax_solver_resistance_distance(A: BCOO, sources, targets, solver):
    """
    Computes resistance distance using a lineax solver.
    This is more memory-efficient for large graphs than p-inv, which densifies the Laplacian.
    """
    L = graph_laplacian(A)
    n = L.shape[0]

    # positive-definite regularisation
    fro_norm = jnp.linalg.norm(L.data)
    eps = jnp.finfo(L.dtype).eps * fro_norm
    # eps = 1e0 * fro_norm
    nzval_reg = L.data + eps
    L_reg = BCOO((nzval_reg, L.indices), shape=L.shape)

    # resistance computation via linear solves
    sources = sources.astype(A.indices.dtype)
    targets = targets.astype(A.indices.dtype)
    num_sources = sources.shape[0]

    source_basis = jax.nn.one_hot(sources, n, dtype=L_reg.dtype).T
    source_range = jnp.arange(num_sources, dtype=A.indices.dtype)

    def solve_for_target(target):
        e_target = jax.nn.one_hot(target, n, dtype=L_reg.dtype)
        rhs = e_target[:, None] - source_basis
        # potentials = batched_linear_solve(L_reg, rhs, solver) # Throws error NaNs when constructing the multigrid hierarchy
        potentials = jnp.linalg.solve(L_reg.todense(), rhs) # Temporary: use dense solver until batched_linear_solve is fixed for sparse L_reg
        source_potentials = potentials[sources, source_range]
        potentials = potentials - source_potentials[None, :]
        return potentials[target, :]

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