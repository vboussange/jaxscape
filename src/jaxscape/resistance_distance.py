import jax
import jax.numpy as jnp
from jax.numpy.linalg import pinv
from jaxscape.distance import AbstractDistance
import equinox as eqx
import lineax as lx
from jaxscape.utils import graph_laplacian, bcoo_at_set
from jaxscape.linear_solve import batched_linear_solve

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

    # Ground the last node by zeroing its row/column and enforcing V_ground = 0
    # This produces a block matrix with the reduced Laplacian and a 1 on the ground node
    ground = n - 1
    rows = L.indices[:, 0]
    cols = L.indices[:, 1]
    mask = jnp.logical_and(rows != ground, cols != ground)
    grounded_data = jnp.where(mask, L.data, jnp.zeros_like(L.data))
    L_grounded = BCOO((grounded_data, L.indices), shape=L.shape)
    L_grounded = bcoo_at_set(
        L_grounded,
        jnp.array([ground]),
        jnp.array([ground]),
        jnp.array([1], dtype=L.data.dtype),
    )

    # We need to solve L'x = b for b being the standard basis vectors
    # for the target nodes.
    B = jnp.eye(n, dtype=L.dtype)[:, targets]

    # Solve the batched linear system
    V_cols = batched_linear_solve(L_grounded, B, solver)

    # The resistance distance R_ij = V_ii + V_jj - 2V_ij
    # where V is the pseudo-inverse. The columns we solved for are columns
    # of a modified inverse, but they can be used to compute the resistance.
    # V_ii can be computed from the diagonal of the inverse.
    # We only need the diagonal elements corresponding to sources and targets.
    
    # To comply with JIT, we solve for diagonal elements for sources and targets separately.
    # This avoids data-dependent shapes from jnp.union1d.
    
    # Solve for diagonal elements corresponding to sources
    B_diag_sources = jnp.eye(n, dtype=L.dtype)[:, sources]
    V_diag_cols_sources = batched_linear_solve(L_grounded, B_diag_sources, solver)
    Vuu = jax.vmap(lambda x, i: x[i], in_axes=(1, 0))(V_diag_cols_sources, sources)
    Vuu = jnp.where(sources == ground, 0, Vuu)

    # Solve for diagonal elements corresponding to targets
    B_diag_targets = jnp.eye(n, dtype=L.dtype)[:, targets]
    V_diag_cols_targets = batched_linear_solve(L_grounded, B_diag_targets, solver)
    Vvv = jax.vmap(lambda x, i: x[i], in_axes=(1, 0))(V_diag_cols_targets, targets)
    Vvv = jnp.where(targets == ground, 0, Vvv)

    # Vuv can be extracted from the columns we solved for earlier.
    column_mask = 1 - jnp.equal(targets, ground).astype(L.dtype)
    V_cols = V_cols.at[ground, :].set(0)
    V_cols = V_cols * column_mask
    row_mask = 1 - jnp.equal(sources, ground).astype(L.dtype)
    Vuv = (V_cols[sources, :].T * row_mask).T

    # Final resistance distance matrix calculation
    R = Vuu[:, None] + Vvv[None, :] - 2 * Vuv
    return R