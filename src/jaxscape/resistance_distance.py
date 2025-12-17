from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO
from jax.numpy.linalg import pinv

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph
from jaxscape.solvers import batched_linear_solve
from jaxscape.utils import graph_laplacian


class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances.

    **Attributes**:

    - `solver`: Optional `lineax.AbstractLinearSolver`. Must be compatible with
    BCOO matrices. We currently support `jaxscape.solvers.CholmodSolver` and
    `jaxscape.solvers.PyAMGSolver`. If None, uses pseudo-inverse method, which
    is very memory intensive for large graphs (densifies the Laplacian
    matrix).

    !!! example

        ```python
        from jaxscape import ResistanceDistance
        from jaxscape.solvers import PyAMGSolver

        # Default: pseudo-inverse (small graphs)
        distance = ResistanceDistance()

        # With solver (large graphs)
        distance = ResistanceDistance(solver=PyAMGSolver())

        dist = distance(grid)
        ```

    !!! Warning

        The graph must be undirected for resistance distance to be well-defined.
    """

    solver: Optional[lx.AbstractLinearSolver] = None

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        A = graph.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)
        else:
            nodes = jnp.arange(graph.nv)
            return lineax_solver_nodes_to_nodes_resistance_distance(
                A, nodes, self.solver
            )

    @eqx.filter_jit
    def nodes_to_nodes_distance(self, graph: AbstractGraph, nodes: Array) -> Array:
        A = graph.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)[nodes[:, None], nodes[None, :]]
        else:
            return lineax_solver_nodes_to_nodes_resistance_distance(
                A, nodes, self.solver
            )

    @eqx.filter_jit
    def sources_to_targets_distance(
        self, graph: AbstractGraph, sources: Array, targets: Array
    ) -> Array:
        R = self.all_pairs_distance(graph)
        return R[sources[:, None], targets[None, :]]


@eqx.filter_jit
def p_inv_resistance_distance(A: BCOO) -> Array:
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
def lineax_solver_nodes_to_nodes_resistance_distance(
    A: BCOO, nodes: Array, solver: lx.AbstractLinearSolver
) -> Array:
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

    node_basis = jax.nn.one_hot(
        nodes, L_reduced.shape[0], dtype=L_reduced.dtype
    ).T  # if node `L_reduced.shape[0]+1` is included in `nodes`, its one-hot will consists of the zero vector

    potentials = batched_linear_solve(L_reduced, node_basis, solver)

    potentials_ii = jnp.sum(node_basis * potentials, axis=0)
    potentials_ij = jnp.sum(node_basis[:, :, None] * potentials[:, None, :], axis=0)

    R = (
        potentials_ii[:, None]
        + potentials_ii[None, :]
        - (potentials_ij + potentials_ij.T)
    )
    return R
