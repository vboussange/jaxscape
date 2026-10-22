import math
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array
from jax import lax
from jax.experimental.sparse import BCOO
from jax.numpy.linalg import pinv

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph
from jaxscape.solvers import BCOOLinearOperator
from jaxscape.solvers.operator import _batched_linear_solve_impl, _solver_state
from jaxscape.utils import graph_laplacian


# Building the full Rademacher projection matrix is faster for moderate sizes,
# but beyond this many sign entries the temporary allocation becomes large
# enough that the streamed scan path is the safer default.
SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES = 200_000_000


class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances.

    **Attributes**:

    - `solver`: Optional `lineax.AbstractLinearSolver`. Must be compatible with
    BCOO matrices. We currently support `jaxscape.solvers.CholmodSolver`,
    `jaxscape.solvers.PyAMGSolver`, and `jaxscape.solvers.AMJaxCGSolver`.
    If None, uses pseudo-inverse method, which is very memory intensive for
    exact all-pairs distances (densifies the Laplacian matrix), or dense solves
    for approximate distances.
    - `method`: A resistance distance method. Defaults to `ExactResistance()`.
    Use `SpielmanApproximation(epsilon=..., seed=...)` for the randomized
    Spielman-Srivastava approximation.

    !!! example

        ```python
        from jaxscape import ResistanceDistance, SpielmanApproximation
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

        # Or reuse only the AMJax preconditioner across related graphs.
        preconditioner_state = distance.init_preconditioner(grid)
        updated_permeability = permeability * 1.1
        updated_grid = GridGraph(updated_permeability, fun=lambda x, y: (x + y) / 2)
        dist = distance(updated_grid, state=preconditioner_state)

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

    def init_preconditioner(self, graph: AbstractGraph) -> Any:
        """Initialize reusable preconditioner state for `graph` when supported.

        For `AMJaxCGSolver`, this builds the AMJax hierarchy without binding the
        state to the current Laplacian values. Passing the returned state to
        distance calls lets JAXScape materialize fresh CG state for each graph
        while reusing the preconditioner.
        """
        if self.solver is None:
            return None

        init_preconditioner = getattr(self.solver, "init_preconditioner", None)
        if init_preconditioner is None:
            return self.init(graph)

        A = graph.get_adjacency_matrix()
        L_reduced = graph_laplacian(A)[:-1, :-1]
        return init_preconditioner(BCOOLinearOperator(L_reduced), {})

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        A = graph.get_adjacency_matrix()
        if isinstance(self.method, SpielmanApproximation):
            return spielman_resistance_distance(
                A, self.method.epsilon, self.method.seed, self.solver, state
            )
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
        if isinstance(self.method, SpielmanApproximation):
            return spielman_nodes_to_nodes_resistance_distance(
                A, nodes, self.method.epsilon, self.method.seed, self.solver, state
            )
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
        if isinstance(self.method, SpielmanApproximation):
            A = graph.get_adjacency_matrix()
            return spielman_sources_to_targets_resistance_distance(
                A,
                sources,
                targets,
                self.method.epsilon,
                self.method.seed,
                self.solver,
                state,
            )
        R = self.all_pairs_distance(graph, state)
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
    selected_inverse = _exact_selected_inverse(
        A.data, A.indices, A.shape, nodes.astype(A.indices.dtype), solver, state
    )
    diagonal = jnp.diag(selected_inverse)
    return (
        diagonal[:, None]
        + diagonal[None, :]
        - selected_inverse
        - selected_inverse.T
    )


@eqx.filter_jit
def spielman_resistance_distance(
    A: BCOO,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: Optional[lx.AbstractLinearSolver] = None,
    state: Any = None,
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
        A.data, A.indices, A.shape, epsilon, seed, solver, state
    )


@eqx.filter_jit
def spielman_nodes_to_nodes_resistance_distance(
    A: BCOO,
    nodes: Array,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: Optional[lx.AbstractLinearSolver] = None,
    state: Any = None,
) -> Array:
    """Approximate pairwise resistance distances among selected nodes."""
    return _spielman_nodes_to_nodes_resistance_distance_from_data(
        A.data,
        A.indices,
        A.shape,
        nodes.astype(A.indices.dtype),
        epsilon,
        seed,
        solver,
        state,
    )


@eqx.filter_jit
def spielman_sources_to_targets_resistance_distance(
    A: BCOO,
    sources: Array,
    targets: Array,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: Optional[lx.AbstractLinearSolver] = None,
    state: Any = None,
) -> Array:
    """Approximate resistance distances from sources to targets."""
    return _spielman_sources_to_targets_resistance_distance_from_data(
        A.data,
        A.indices,
        A.shape,
        sources.astype(A.indices.dtype),
        targets.astype(A.indices.dtype),
        epsilon,
        seed,
        solver,
        state,
    )


def _projection_size(n: int, epsilon: float) -> int:
    """Number of Spielman-Srivastava random projections for accuracy epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive (controls approximation accuracy).")
    return max(1, math.ceil(math.log(max(n, 2)) / epsilon**2))


def _prepare_reduced_laplacian(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> tuple[BCOO, Any]:
    reduced = graph_laplacian(BCOO((data, indices), shape=shape))[:-1, :-1]
    if solver is not None:
        state = _solver_state(reduced, solver, None, state)
    return reduced, state


def _solve_reduced_laplacian(
    L_reduced: BCOO,
    rhs: Array,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
    options: dict[str, Any] | None = None,
) -> Array:
    """Execute with already prepared state; all RHS keep independent solves."""
    if solver is None:
        return jnp.linalg.solve(L_reduced.todense(), rhs)
    return _batched_linear_solve_impl(
        L_reduced, rhs, solver, state=state, options=options
    )


def _solve_adjoint(
    L_reduced: BCOO,
    rhs: Array,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> Array:
    options = None
    if solver is not None:
        state, options = solver.transpose(state, {})
    return _solve_reduced_laplacian(L_reduced.T, rhs, solver, state, options)


def _node_potentials(
    L_reduced: BCOO,
    nodes: Array,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> Array:
    # The ground-node column of E is zero; duplicate nodes retain duplicate RHS.
    basis = jax.nn.one_hot(nodes, L_reduced.shape[0], dtype=L_reduced.dtype).T
    return _solve_reduced_laplacian(L_reduced, basis, solver, state)


def _grounded_values(potentials: Array, nodes: Array) -> Array:
    """Select rows of (n - 1, rhs) potentials, with zero at the ground node."""
    return jnp.where(
        (nodes < potentials.shape[0])[:, None],
        potentials[jnp.minimum(nodes, potentials.shape[0] - 1)],
        0,
    )


def _laplacian_data_pullback(
    adjoint: Array, potentials: Array, indices: Array, shape: tuple[int, ...]
) -> Array:
    # L = diag(row sums) - adjacency: each original stored coefficient (u, v)
    # receives -<Lambda_u, V_u - V_v>. Do not merge duplicates or double edges.
    # Keep gather/multiply/reduce fused; explicit edge vmap/chunking is slower.
    values = -jnp.sum(
        _grounded_values(adjoint, indices[:, 0])
        * _edge_potential_deltas(potentials, indices, shape),
        axis=1,
    )
    return jnp.where(jnp.all(indices < jnp.asarray(shape), axis=1), values, 0)


@eqx.filter_custom_vjp
def _exact_selected_inverse(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    nodes: Array,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> Array:
    return _exact_selected_inverse_fwd(
        None, data, indices, shape, nodes, solver, state
    )[0]


def _exact_selected_inverse_fwd(perturbed, data, indices, shape, nodes, solver, state):
    del perturbed
    reduced, solve_state = _prepare_reduced_laplacian(data, indices, shape, solver, state)
    potentials = _node_potentials(reduced, nodes, solver, solve_state)
    return _grounded_values(potentials, nodes), potentials


_exact_selected_inverse.def_fwd(_exact_selected_inverse_fwd)


@_exact_selected_inverse.def_bwd
def _exact_selected_inverse_bwd(
    potentials, cotangent, perturbed, data, indices, shape, nodes, solver, state
):
    del data, nodes, solver, state
    if cotangent is None or not perturbed:
        return None
    # For undirected graphs A is symmetric: S = E.T A^-1 E, X = A^-1 E.
    # An arbitrary (possibly nonsymmetric) cotangent G gives Lambda = X G.
    # Selecting S avoids a vertex-by-site-by-site broadcast; the backward pass
    # reuses X and performs no additional linear solves.
    adjoint = potentials @ cotangent
    return _laplacian_data_pullback(adjoint, potentials, indices, shape)


def _rademacher_projection_signs(
    k: int, n_edges: int, seed: int, dtype_: jnp.dtype
) -> Array:
    return jax.random.rademacher(jax.random.PRNGKey(seed), (k, n_edges)).astype(dtype_)


def _spielman_projection(
    data: Array, indices: Array, shape: tuple[int, ...], epsilon: float, seed: int
) -> Array:
    k = _projection_size(shape[0], epsilon)
    signs = _rademacher_projection_signs(k, data.shape[0], seed, data.dtype)
    rows = indices[:, 0]
    cols = indices[:, 1]
    valid = jnp.all(indices < jnp.asarray(shape), axis=1)
    weights = _sqrt_half_weights(jnp.where(valid, data, 0))
    normalized_weights = signs * weights[None, :] * _projection_scale(k, data.dtype)
    projection = jnp.zeros((k, shape[0]), dtype=data.dtype)
    projection = projection.at[:, rows].add(normalized_weights)
    projection = projection.at[:, cols].add(-normalized_weights)
    return projection


def _sqrt_half_weights(data: Array) -> Array:
    """Square root of half the non-negative edge weights."""
    # Guard the square-root argument itself: masking sqrt(0) afterward still
    # produces 0 * inf in autodiff. Zero and negative weights have zero gradient.
    positive = data > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, data, 2) / 2), 0)


def _projection_scale(k: int, dtype_: jnp.dtype) -> Array:
    return jnp.asarray(k, dtype=dtype_) ** -0.5


@eqx.filter_custom_vjp
def _spielman_features_reduced(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> Array:
    return _spielman_features_reduced_fwd(
        None, data, indices, shape, epsilon, seed, solver, state
    )[0]


def _spielman_features_reduced_fwd(
    perturbed, data, indices, shape, epsilon, seed, solver, state
):
    del perturbed
    reduced, solve_state = _prepare_reduced_laplacian(data, indices, shape, solver, state)
    projection = _spielman_projection(data, indices, shape, epsilon, seed)[:, :-1]
    potentials = _solve_reduced_laplacian(reduced, projection.T, solver, solve_state)
    return potentials.T, (potentials, reduced, solve_state)


_spielman_features_reduced.def_fwd(_spielman_features_reduced_fwd)


@_spielman_features_reduced.def_bwd
def _spielman_features_reduced_bwd(
    residuals, cotangent, perturbed, data, indices, shape, epsilon, seed, solver, state
):
    del state
    if cotangent is None or not perturbed:
        return None
    potentials, reduced, solve_state = residuals
    # A Y = P.T, F = Y.T: A.T Lambda = H.T. Contract -Lambda Y.T
    # directly in adjacency storage rather than constructing an n-by-n matrix.
    adjoint = _solve_adjoint(reduced, cotangent.T, solver, solve_state)
    _, projection_pullback = jax.vjp(
        lambda weights: _spielman_projection(weights, indices, shape, epsilon, seed),
        data,
    )
    (projection_gradient,) = projection_pullback(
        jnp.pad(adjoint.T, ((0, 0), (0, 1)))
    )
    return (
        _laplacian_data_pullback(adjoint, potentials, indices, shape)
        + projection_gradient
    )


def _edge_potential_deltas(
    potentials: Array, indices: Array, shape: tuple[int, ...]
) -> Array:
    deltas = (
        _grounded_values(potentials, indices[:, 0])
        - _grounded_values(potentials, indices[:, 1])
    )
    valid = jnp.all(indices < jnp.asarray(shape), axis=1)
    return jnp.where(valid[:, None], deltas, 0)


def _spielman_project_potentials(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    potentials: Array,
    epsilon: float,
    seed: int,
) -> Array:
    k = _projection_size(shape[0], epsilon)
    edge_deltas = _edge_potential_deltas(potentials, indices, shape)
    scaled_weights = _sqrt_half_weights(data) * _projection_scale(k, data.dtype)

    if k * data.shape[0] <= SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES:
        signs = _rademacher_projection_signs(k, data.shape[0], seed, data.dtype)
        return (signs * scaled_weights[None, :]) @ edge_deltas

    key = jax.random.PRNGKey(seed)

    # Budget 8 GiB for row-dependent forward/backward working arrays, estimating
    # three edge-sized floating arrays plus two int32 sign buffers. This excludes
    # fixed potentials/edge deltas, returned embeddings, and allocator reserves.
    # This selects 64 rows for four million float64 entries. Larger problems
    # reduce to one row through the same map; one row is the unavoidable floor.
    bytes_per_row = data.shape[0] * (3 * data.dtype.itemsize + 2 * 4)
    batch_size = max(1, min(64, (8 * 1024**3) // max(1, bytes_per_row)))

    @jax.checkpoint
    def project_one(projection_index):
        projection_key = jax.random.fold_in(key, projection_index)
        signs = jax.random.rademacher(projection_key, data.shape).astype(data.dtype)
        return (signs * scaled_weights) @ edge_deltas

    # lax.map handles a final partial batch with the same global fold_in keys.
    # Checkpointing each row recomputes signs in reverse mode instead of saving
    # all k rows. Dense projections retain their original whole-matrix PRNG key.
    return lax.map(project_one, jnp.arange(k), batch_size=batch_size)


def _distances_from_selected_features(
    source_features: Array, target_features: Array
) -> Array:
    source_norms = jnp.sum(source_features**2, axis=0)
    target_norms = jnp.sum(target_features**2, axis=0)
    return (
        source_norms[:, None]
        + target_norms[None, :]
        - 2 * source_features.T @ target_features
    )


@eqx.filter_custom_vjp
def _spielman_selected_features(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    nodes: Array,
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> Array:
    return _spielman_selected_features_fwd(
        None, data, indices, shape, nodes, epsilon, seed, solver, state
    )[0]


def _spielman_selected_features_fwd(
    perturbed, data, indices, shape, nodes, epsilon, seed, solver, state
):
    del perturbed
    reduced, solve_state = _prepare_reduced_laplacian(data, indices, shape, solver, state)
    potentials = _node_potentials(reduced, nodes, solver, solve_state)
    features = _spielman_project_potentials(data, indices, shape, potentials, epsilon, seed)
    return features, (potentials, reduced, solve_state)


_spielman_selected_features.def_fwd(_spielman_selected_features_fwd)


@_spielman_selected_features.def_bwd
def _spielman_selected_features_bwd(
    residuals, cotangent, perturbed, data, indices, shape, nodes, epsilon, seed, solver, state
):
    del nodes, state
    if cotangent is None or not perturbed:
        return None
    potentials, reduced, solve_state = residuals
    # Differentiate only P X, holding the solve X fixed. This returns both the
    # weight contribution and P.T H without ever forming H X.T. Streamed rows
    # regenerate their signs during this pullback, so no k-by-edge tape survives.
    _, projection_pullback = jax.vjp(
        lambda weights, values: _spielman_project_potentials(
            weights, indices, shape, values, epsilon, seed
        ),
        data,
        potentials,
    )
    projection_gradient, adjoint_rhs = projection_pullback(cotangent)
    adjoint = _solve_adjoint(reduced, adjoint_rhs, solver, solve_state)
    return (
        _laplacian_data_pullback(adjoint, potentials, indices, shape)
        + projection_gradient
    )


def _spielman_sources_to_targets_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    sources: Array,
    targets: Array,
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any = None,
) -> Array:
    nodes = jnp.concatenate([sources, targets])
    features = _spielman_selected_features(
        data, indices, shape, nodes, epsilon, seed, solver, state
    )
    n_sources = sources.shape[0]
    return _distances_from_selected_features(
        features[:, :n_sources], features[:, n_sources:]
    )


def _spielman_nodes_to_nodes_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    nodes: Array,
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any = None,
) -> Array:
    features = _spielman_selected_features(
        data, indices, shape, nodes, epsilon, seed, solver, state
    )
    distances = _distances_from_selected_features(features, features)
    return distances.at[jnp.diag_indices(nodes.shape[0])].set(0)


def _distances_from_features_reduced(features_reduced: Array) -> Array:
    """Pairwise approximate resistance distances from reduced embeddings."""
    features = jnp.pad(features_reduced, ((0, 0), (0, 1)))
    feature_norms = jnp.sum(features**2, axis=0)
    # Pairwise squared distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u*v.
    return feature_norms[:, None] + feature_norms[None, :] - 2 * features.T @ features


def _spielman_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    epsilon: float,
    seed: int,
    solver: Optional[lx.AbstractLinearSolver],
    state: Any,
) -> Array:
    features_reduced = _spielman_features_reduced(
        data, indices, shape, epsilon, seed, solver, state
    )
    return _distances_from_features_reduced(features_reduced)
