import math
from functools import partial
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
    is very memory intensive for exact all-pairs distances (densifies the
    Laplacian matrix), or dense solves for approximate distances.
    - `method`: A resistance distance method. Defaults to `ExactResistance()`.
    Use `SpielmanApproximation(epsilon=..., seed=...)` for the randomized
    Spielman-Srivastava approximation.

    !!! example

        ```python
        from jaxscape import ResistanceDistance, SpielmanApproximation
        from jaxscape.solvers import PyAMGSolver

        # Default: pseudo-inverse (small graphs)
        distance = ResistanceDistance()

        # With solver (large graphs)
        distance = ResistanceDistance(solver=PyAMGSolver())

        # Approximate resistance distance
        distance = ResistanceDistance(
            method=SpielmanApproximation(epsilon=0.05),
            solver=PyAMGSolver(),
        )

        dist = distance(grid)
        ```

    !!! Warning

        The graph must be undirected for resistance distance to be well-defined.
    """

    solver: Optional[lx.AbstractLinearSolver] = None
    method: "AbstractResistanceMethod" = eqx.field(
        default_factory=lambda: ExactResistance()
    )

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        A = graph.get_adjacency_matrix()
        if isinstance(self.method, SpielmanApproximation):
            return spielman_resistance_distance(
                A, self.method.epsilon, self.method.seed, self.solver
            )
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
        if isinstance(self.method, SpielmanApproximation):
            return spielman_resistance_distance(
                A, self.method.epsilon, self.method.seed, self.solver
            )[nodes[:, None], nodes[None, :]]
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


class AbstractResistanceMethod(eqx.Module):
    """Abstract base class for resistance distance algorithms."""


class ExactResistance(AbstractResistanceMethod):
    """Exact resistance distance via pseudoinverse or grounded linear solves."""


class SpielmanApproximation(AbstractResistanceMethod):
    """
    Spielman-Srivastava randomized resistance distance approximation.

    **Attributes**:

    - `epsilon`: Accuracy parameter. Smaller values use more random
      projections: `ceil(log(n_vertices) / epsilon**2)`, which increases memory
      use.
    - `seed`: Random seed for the Rademacher edge projections.
    """

    epsilon: float = 0.1
    seed: int = 0

    def __check_init__(self):
        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive (controls approximation accuracy)."
            )


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


@eqx.filter_jit
def spielman_resistance_distance(
    A: BCOO,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: Optional[lx.AbstractLinearSolver] = None,
) -> Array:
    """
    Approximate resistance distances using Spielman-Srivastava projections.

    This follows Spielman and Srivastava, "Graph sparsification by effective
    resistances" (SIAM J. Comput., 2011; arXiv version 2009). Let `B` be an
    oriented incidence matrix, `W` the diagonal edge-weight matrix, and
    `L = B.T @ W @ B` the graph Laplacian. The exact resistance is
    `R_ij = ||W**0.5 @ B @ L^+ @ (e_i - e_j)||_2**2`. The algorithm replaces
    `W**0.5 @ B` by a Johnson-Lindenstrauss/Rademacher sketch `Q W**0.5 B`
    and computes node embeddings `Z = Q W**0.5 B L^+`; squared Euclidean
    distances between columns of `Z` approximate resistance distances.
    """
    return _spielman_resistance_distance_from_data(
        A.data, A.indices, A.shape, epsilon, seed, solver
    )


def _projection_size(n: int, epsilon: float) -> int:
    """Number of Spielman-Srivastava random projections for accuracy epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive (controls approximation accuracy).")
    return max(1, math.ceil(math.log(max(n, 2)) / epsilon**2))


def _solve_reduced_laplacian(
    L_reduced: BCOO, rhs: Array, solver: Optional[lx.AbstractLinearSolver]
) -> Array:
    if solver is None:
        return jnp.linalg.solve(L_reduced.todense(), rhs)
    return batched_linear_solve(L_reduced, rhs, solver)


def _spielman_projection(
    data: Array, indices: Array, shape: tuple[int, int], epsilon: float, seed: int
) -> Array:
    k = _projection_size(shape[0], epsilon)
    key = jax.random.PRNGKey(seed)
    signs = jax.random.rademacher(key, (k, data.shape[0])).astype(data.dtype)
    rows = indices[:, 0]
    cols = indices[:, 1]
    normalized_weights = signs * _sqrt_half_weights(data)[None, :] * _projection_scale(
        k, data.dtype
    )
    projection = jnp.zeros((k, shape[0]), dtype=data.dtype)
    projection = projection.at[:, rows].add(normalized_weights)
    projection = projection.at[:, cols].add(-normalized_weights)
    return projection


def _sqrt_half_weights(data: Array) -> Array:
    """Square root of half the non-negative edge weights."""
    return jnp.sqrt(jnp.maximum(data, 0) / 2)


def _projection_scale(k: int, dtype_: jnp.dtype) -> Array:
    return jnp.asarray(k, dtype=dtype_) ** -0.5


def _spielman_features_reduced(
    data: Array,
    indices: Array,
    shape: tuple[int, int],
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
) -> Array:
    A = BCOO((data, indices), shape=shape)
    L_reduced = graph_laplacian(A)[:-1, :-1]
    projection_reduced = _spielman_projection(data, indices, shape, epsilon, seed)[
        :, :-1
    ]
    return _solve_reduced_laplacian(L_reduced, projection_reduced.T, solver).T


def _distances_from_features_reduced(features_reduced: Array) -> Array:
    """Pairwise approximate resistance distances from reduced embeddings."""
    features = jnp.pad(features_reduced, ((0, 0), (0, 1)))
    feature_norms = jnp.sum(features**2, axis=0)
    # Pairwise squared distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u*v.
    return feature_norms[:, None] + feature_norms[None, :] - 2 * features.T @ features


@partial(jax.custom_vjp, nondiff_argnums=(2, 3, 4, 5))
def _spielman_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, int],
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
) -> Array:
    features_reduced = _spielman_features_reduced(
        data, indices, shape, epsilon, seed, solver
    )
    return _distances_from_features_reduced(features_reduced)


def _spielman_resistance_distance_fwd(
    data: Array,
    indices: Array,
    shape: tuple[int, int],
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
):
    features_reduced = _spielman_features_reduced(
        data, indices, shape, epsilon, seed, solver
    )
    distances = _distances_from_features_reduced(features_reduced)
    return distances, (data, indices, features_reduced)


def _spielman_resistance_distance_bwd(
    shape: tuple[int, int],
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
    residual,
    cotangent: Array,
):
    # Custom VJP derivation. Ground the final node and write the sketched
    # embeddings as `F = P_r L_r^{-1}`, where `P_r` is the sketched incidence
    # matrix with the grounded column removed and `L_r` is the reduced
    # Laplacian. The primal output is `D_ij = ||f_i - f_j||^2`.
    #
    # Given an output cotangent `G = d loss / dD`, first differentiate the
    # squared-distance map. With `S = G + G.T`, the embedding cotangent is
    # `bar_F = 2 * (F * row_sum(S) - F @ S.T)`; the last grounded column is then
    # discarded. The linear solve is the implicit equation `F_r L_r = P_r`.
    # Differentiating gives `dF_r L_r + F_r dL_r = dP_r`. Multiplying by the
    # adjoint variable `Y = L_r^{-1} bar_F.T` yields the closed-form pullbacks
    # `bar_P_r = Y.T` and `bar_L_r = -Y @ F_r`. Finally, each edge weight `w_e`
    # contributes to `L_r` as `(e_u - e_v)(e_u - e_v)^T` after grounding, and
    # to `P_r` as `sign_e * sqrt(w_e / 2) / sqrt(k) * (e_u - e_v)`. The code
    # below applies these two edge-local contributions to `A.data`.
    data, indices, features_reduced = residual
    A = BCOO((data, indices), shape=shape)
    L_reduced = graph_laplacian(A)[:-1, :-1]
    features = jnp.pad(features_reduced, ((0, 0), (0, 1)))

    symmetric_cotangent = cotangent + cotangent.T
    row_sum = jnp.sum(symmetric_cotangent, axis=1)
    feature_cotangent = 2 * (
        features * row_sum[None, :] - features @ symmetric_cotangent.T
    )
    reduced_feature_cotangent = feature_cotangent[:, :-1].T

    adjoint = _solve_reduced_laplacian(L_reduced, reduced_feature_cotangent, solver)
    projection_cotangent = adjoint.T
    laplacian_cotangent = -adjoint @ features_reduced

    rows = indices[:, 0]
    cols = indices[:, 1]
    n_reduced = shape[0] - 1
    rows_reduced = rows < n_reduced
    cols_reduced = cols < n_reduced
    clamped_rows = jnp.minimum(rows, n_reduced - 1)
    clamped_cols = jnp.minimum(cols, n_reduced - 1)

    diagonal_grad = jnp.where(
        rows_reduced, laplacian_cotangent[clamped_rows, clamped_rows], 0
    )
    off_diagonal_grad = jnp.where(
        rows_reduced & cols_reduced,
        laplacian_cotangent[clamped_rows, clamped_cols],
        0,
    )
    laplacian_data_cotangent = diagonal_grad - off_diagonal_grad

    k = _projection_size(shape[0], epsilon)
    signs = (
        jax.random.rademacher(jax.random.PRNGKey(seed), (k, data.shape[0]))
        .astype(data.dtype)
        * _projection_scale(k, data.dtype)
    )
    projection_rows = jnp.where(
        rows_reduced, projection_cotangent[:, clamped_rows], 0
    )
    projection_cols = jnp.where(
        cols_reduced, projection_cotangent[:, clamped_cols], 0
    )
    sqrt_weights = _sqrt_half_weights(data)
    sqrt_weights_or_one = jnp.where(data > 0, sqrt_weights, 1)
    projection_data_cotangent = jnp.where(
        data > 0,
        jnp.sum(signs * (projection_rows - projection_cols), axis=0)
        / (4 * sqrt_weights_or_one),
        0,
    )

    return laplacian_data_cotangent + projection_data_cotangent, None


_spielman_resistance_distance_from_data.defvjp(
    _spielman_resistance_distance_fwd, _spielman_resistance_distance_bwd
)
