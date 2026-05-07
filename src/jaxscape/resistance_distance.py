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
    is very memory intensive for large graphs (densifies the Laplacian
    matrix), or dense solves for the approximate method.
    - `approximate`: If True, uses the Spielman-Srivastava random projection
    algorithm.
    - `epsilon`: Accuracy parameter for the approximate method. Smaller values
    use more random projections: `ceil(log(n_vertices) / epsilon**2)`, which
    increases memory use.
    - `seed`: Random seed for the approximate method projections.

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
    approximate: bool = False
    epsilon: float = 0.1
    seed: int = 0

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        A = graph.get_adjacency_matrix()
        if self.approximate:
            return spielman_resistance_distance(A, self.epsilon, self.seed, self.solver)
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
        if self.approximate:
            return spielman_resistance_distance(A, self.epsilon, self.seed, self.solver)[
                nodes[:, None], nodes[None, :]
            ]
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


@eqx.filter_jit
def spielman_resistance_distance(
    A: BCOO,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: Optional[lx.AbstractLinearSolver] = None,
) -> Array:
    """Approximate resistance distances using Spielman-Srivastava projections."""
    return _spielman_resistance_distance_from_data(
        A.data, A.indices, A.shape, epsilon, seed, solver
    )


def _projection_size(n: int, epsilon: float) -> int:
    if epsilon <= 0:
        raise ValueError("`epsilon` must be positive.")
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
    return jnp.sqrt(jnp.maximum(data, 0) / 2)


def _projection_scale(k: int, target_dtype) -> Array:
    return jnp.asarray(k, dtype=target_dtype) ** -0.5


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
    safe_sqrt_weights = jnp.where(data > 0, sqrt_weights, 1)
    projection_data_cotangent = jnp.where(
        data > 0,
        jnp.sum(signs * (projection_rows - projection_cols), axis=0)
        / (4 * safe_sqrt_weights),
        0,
    )

    return laplacian_data_cotangent + projection_data_cotangent, None


_spielman_resistance_distance_from_data.defvjp(
    _spielman_resistance_distance_fwd, _spielman_resistance_distance_bwd
)
