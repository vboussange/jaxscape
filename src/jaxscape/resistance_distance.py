from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax.experimental.sparse import BCOO
from jax.numpy.linalg import pinv

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph
from jaxscape.solvers import batched_linear_solve, BCOOLinearOperator
from jaxscape.utils import graph_laplacian


class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances.

    **Attributes**:

    - `solver`: Optional `lineax.AbstractLinearSolver`. Must be compatible with
    BCOO matrices. We currently support `jaxscape.solvers.CholmodSolver`,
    `jaxscape.solvers.PyAMGSolver`, and `jaxscape.solvers.AMJaxCGSolver`.
    If None, uses pseudo-inverse method, which is very memory intensive for
    large graphs (densifies the Laplacian matrix).

    !!! example

        ```python
        from jaxscape import ResistanceDistance
        from jaxscape.solvers import AMJaxCGSolver, PyAMGSolver

        # Default: pseudo-inverse (small graphs)
        distance = ResistanceDistance()

        # With solver (large graphs)
        distance = ResistanceDistance(solver=PyAMGSolver())

        # With an initialized AMG-preconditioned CG solver
        distance = ResistanceDistance(
            solver=AMJaxCGSolver(rtol=1e-6, atol=1e-6, max_steps=1_000)
        )
        state = distance.init(grid)

        dist = distance(grid, state=state)
        ```

    !!! Warning

        The graph must be undirected for resistance distance to be well-defined.
    """

    solver: lx.AbstractLinearSolver | None = None

    def init(self, graph: AbstractGraph) -> Any:
        """Initialize matrix-dependent solver state for `graph`.

        This is primarily useful for solvers such as `AMJaxCGSolver`, which
        build reusable state once outside the traced solve path so repeated
        resistance solves remain JIT-compatible and differentiable.
        """
        if self.solver is None:
            return None

        A = graph.get_adjacency_matrix()
        L_reduced = graph_laplacian(A)[:-1, :-1]
        return self.solver.init(BCOOLinearOperator(L_reduced), {})

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        A = graph.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)
        else:
            nodes = jnp.arange(graph.nv)
            return lineax_solver_nodes_to_nodes_resistance_distance(
                A, nodes, self.solver, state
            )

    @eqx.filter_jit
    def nodes_to_nodes_distance(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> Array:
        A = graph.get_adjacency_matrix()
        if self.solver is None:
            return p_inv_resistance_distance(A)[nodes[:, None], nodes[None, :]]
        else:
            return lineax_solver_nodes_to_nodes_resistance_distance(
                A, nodes, self.solver, state
            )

    @eqx.filter_jit
    def sources_to_targets_distance(
        self,
        graph: AbstractGraph,
        sources: Array,
        targets: Array,
        state: Any = None,
    ) -> Array:
        R = self.all_pairs_distance(graph, state)
        return R[sources[:, None], targets[None, :]]


@eqx.filter_jit
def p_inv_resistance_distance(A: BCOO) -> Array:
    # See NetworkX's [resistance_distance implementation for the same formula](https://networkx.org/documentation/stable/_modules/networkx/algorithms/distance_measures.html#resistance_distance)
    """
    Computes the resistance distance matrix.
    Args:
        A: Adjacency matrix (sparse BCOO).
    Returns:
        Resistance distance matrix.
    """
    L = graph_laplacian(A)
    # TODO: once graphs expose symmetry, use pinv(..., hermitian=True) here.
    V = pinv(L.todense())  # Moore-Penrose pseudoinverse of Laplacian

    # Compute resistance distances
    Vuu = jnp.diag(V)
    R = Vuu[:, None] + Vuu[None, :] - V - V.T

    return R


@eqx.filter_jit
def lineax_solver_nodes_to_nodes_resistance_distance(
    A: BCOO,
    nodes: Array,
    solver: lx.AbstractLinearSolver,
    state: Any = None,
) -> Array:
    """
    Computes pairwise resistance distance from `nodes` to `nodes`, returning a
    |`nodes`| x |`nodes`| matrix, using a lineax `solver`.
    Requires |`nodes`| linear solves.
    !!! Warning
        The graph must be undirected.
    """
    L = graph_laplacian(A)

    # Ground the last node to obtain a full-rank system.
    L_reduced = L[:-1, :-1]

    nodes = nodes.astype(A.indices.dtype)

    node_basis = jax.nn.one_hot(
        nodes, L_reduced.shape[0], dtype=L_reduced.dtype
    ).T

    potentials = batched_linear_solve(
        L_reduced,
        node_basis,
        solver,
        state=state,
    )

    potentials_ii = jnp.sum(node_basis * potentials, axis=0)
    potentials_ij = jnp.sum(node_basis[:, :, None] * potentials[:, None, :], axis=0)

    R = (
        potentials_ii[:, None]
        + potentials_ii[None, :]
        - (potentials_ij + potentials_ij.T)
    )
    return R
