import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array, lax
from jax.experimental.sparse import BCOO
from jax.numpy.linalg import pinv

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph
from jaxscape.solvers import BCOOLinearOperator
from jaxscape.solvers.operator import _batched_linear_solve_impl, _solver_states
from jaxscape.utils import graph_laplacian


# Building the full Rademacher projection matrix is faster for moderate sizes,
# but beyond this many sign entries the temporary allocation becomes large
# enough that the streamed scan path is the safer default.
SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES = 200_000_000


class ResistanceDistance(AbstractDistance):
    """
    Compute the resistance distances.

    **Attributes:**

    - `solver`: Optional `lineax.AbstractLinearSolver`. Must be compatible with
      BCOO matrices. We currently support `jaxscape.solvers.CholmodSolver`,
      `jaxscape.solvers.PyAMGSolver`, `jaxscape.solvers.AMJaxCGSolver`, and
      `jaxscape.solvers.CuDSSSolver` (NVIDIA GPUs).
      If None, uses the pseudo-inverse method, which is very memory intensive
      for exact all-pairs distances (densifies the Laplacian matrix), or dense
      solves for approximate distances.
    - `method`: A resistance distance method. Defaults to `ExactResistance()`.
      Use `SpielmanApproximation(epsilon=..., seed=...)` for the randomized
      Spielman-Srivastava approximation.

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import GridGraph, ResistanceDistance, SpielmanApproximation
        from jaxscape.solvers import AMJaxCGSolver, PyAMGSolver

        permeability = jnp.ones((3, 3))
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)

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

        # Query forms ending in `_and_state` also return the solver state
        # to pass to the next call, e.g. the next step of an optimization.
        nodes = jnp.array([0, 4, 8])
        dist, preconditioner_state = distance.nodes_to_nodes_distance_and_state(
            updated_grid, nodes, preconditioner_state
        )

        # Approximate resistance distance
        distance = ResistanceDistance(
            method=SpielmanApproximation(epsilon=0.05),
            solver=PyAMGSolver(),
        )

        dist = distance(grid)
        ```

    !!! warning

        The graph must be undirected for resistance distance to be well-defined.
    """

    solver: lx.AbstractLinearSolver | None = None
    method: "AbstractResistanceMethod" = eqx.field(
        default_factory=lambda: ExactResistance()
    )

    def init(self, graph: AbstractGraph) -> Any:
        """Initialize matrix-dependent solver state for `graph`.

        This is primarily useful for solvers such as `AMJaxCGSolver`, which
        build reusable state once outside the traced solve path so repeated
        resistance solves remain JIT-compatible and differentiable. For
        `CuDSSSolver`, it runs cuDSS analysis and factorization, reads concrete
        sparse indices, and must run outside `jax.jit`. Queries then
        refactorize the state for the edge weights of any graph with the same
        sparsity pattern.

        **Arguments:**

        - `graph`: Undirected graph whose grounded Laplacian the state is built
          for.

        **Returns:**

        Solver state to pass as `state` to distance queries, or `None` when
        `solver` is `None`.
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

    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        return self.all_pairs_distance_and_state(graph, state)[0]

    def nodes_to_nodes_distance(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> Array:
        return self.nodes_to_nodes_distance_and_state(graph, nodes, state)[0]

    def sources_to_targets_distance(
        self,
        graph: AbstractGraph,
        sources: Array,
        targets: Array,
        state: Any = None,
    ) -> Array:
        return self.sources_to_targets_distance_and_state(
            graph, sources, targets, state
        )[0]

    def all_pairs_distance_and_state(
        self, graph: AbstractGraph, state: Any = None
    ) -> tuple[Array, Any]:
        """Compute resistance distances between all vertices, and the solver state.

        **Arguments:**

        - `graph`: Undirected graph with `graph.nv` vertices.
        - `state`: Optional solver state from `init`, `init_preconditioner` or
          an earlier state-returning query. Without it, the call first
          initializes one with `init`, as `distance(graph)` does. A solver
          that reads concrete sparse indices, such as `CuDSSSolver`, then
          needs the call to run outside `jax.jit`.

        **Returns:**

        A tuple `(distances, state)`. `distances` has shape
        `(graph.nv, graph.nv)` and equals `all_pairs_distance(graph, state)`.
        `state` is the solver state to pass to the next call: the state an
        `AbstractRefreshingSolver` refreshed for `graph`, and otherwise the
        supplied or initialized state. It carries no gradient.
        """
        return self._distance_and_state(graph, state)

    def nodes_to_nodes_distance_and_state(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> tuple[Array, Any]:
        """Compute resistance distances among `nodes`, and the solver state.

        **Arguments:**

        - `graph`: Undirected graph.
        - `nodes`: Vertex indices with shape `(k,)`, possibly repeated.
        - `state`: Optional solver state, as for `all_pairs_distance_and_state`.

        **Returns:**

        A tuple `(distances, state)`. `distances` has shape `(k, k)` and equals
        `nodes_to_nodes_distance(graph, nodes, state)`. `state` is the solver
        state to pass to the next call, as for `all_pairs_distance_and_state`.
        """
        return self._distance_and_state(graph, state, nodes)

    def sources_to_targets_distance_and_state(
        self,
        graph: AbstractGraph,
        sources: Array,
        targets: Array,
        state: Any = None,
    ) -> tuple[Array, Any]:
        """Compute distances from `sources` to `targets`, and the solver state.

        **Arguments:**

        - `graph`: Undirected graph.
        - `sources`: Source vertex indices with shape `(s,)`.
        - `targets`: Target vertex indices with shape `(t,)`.
        - `state`: Optional solver state, as for `all_pairs_distance_and_state`.

        **Returns:**

        A tuple `(distances, state)`. `distances` has shape `(s, t)` and equals
        `sources_to_targets_distance(graph, sources, targets, state)`. `state`
        is the solver state to pass to the next call, as for
        `all_pairs_distance_and_state`.
        """
        return self._distance_and_state(graph, state, sources, targets)

    def _distance_and_state(
        self,
        graph: AbstractGraph,
        state: Any,
        nodes: Array | None = None,
        targets: Array | None = None,
    ) -> tuple[Array, Any]:
        if state is None:
            # As in `AbstractDistance.__call__`: an eager call initializes the
            # state outside `jax.jit`, from graph values without gradient.
            state = self.init(jax.tree.map(jax.lax.stop_gradient, graph))
        return _graph_distance_and_state(self, graph, state, nodes, targets)


class AbstractResistanceMethod(eqx.Module):
    """Abstract base class for resistance distance algorithms."""


class ExactResistance(AbstractResistanceMethod):
    """Exact resistance distance via pseudoinverse or grounded linear solves."""


class SpielmanApproximation(AbstractResistanceMethod):
    """
    Spielman-Srivastava randomized resistance distance approximation.

    **Attributes:**

    - `epsilon`: Accuracy parameter. Smaller values use more random
      projections: `ceil(log(n_vertices) / epsilon**2)`, which increases memory
      use.
    - `seed`: Random seed for the Rademacher edge projections.
    - `projection_batch_size`: Number of projection rows evaluated together.
      Larger batches can improve throughput at the cost of more working memory;
      choose a size that fits your device.
    """

    epsilon: float = 0.1
    seed: int = 0
    projection_batch_size: int = 64

    def __check_init__(self):
        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be positive (controls approximation accuracy)."
            )
        if (
            isinstance(self.projection_batch_size, bool)
            or not isinstance(self.projection_batch_size, int)
            or self.projection_batch_size <= 0
        ):
            raise ValueError("projection_batch_size must be a positive Python integer.")


@eqx.filter_jit
def p_inv_resistance_distance(A: BCOO) -> Array:
    # See NetworkX's [resistance_distance implementation for the same formula](https://networkx.org/documentation/stable/_modules/networkx/algorithms/distance_measures.html#resistance_distance)
    """
    Compute the resistance distance matrix.

    **Arguments:**

    - `A`: Sparse BCOO adjacency matrix with shape `(n, n)`.

    **Returns:**

    Resistance distance matrix with shape `(n, n)`.
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

    !!! warning
        The graph must be undirected.
    """
    distance = ResistanceDistance(solver=solver)
    return _distance_and_state(distance, A, state, nodes)[0]


@eqx.filter_jit
def spielman_resistance_distance(
    A: BCOO,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: lx.AbstractLinearSolver | None = None,
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
    method = SpielmanApproximation(epsilon=epsilon, seed=seed)
    distance = ResistanceDistance(solver=solver, method=method)
    return _distance_and_state(distance, A, state)[0]


@eqx.filter_jit
def spielman_nodes_to_nodes_resistance_distance(
    A: BCOO,
    nodes: Array,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: lx.AbstractLinearSolver | None = None,
    state: Any = None,
    *,
    projection_batch_size: int = 64,
) -> Array:
    """Approximate pairwise resistance distances among selected nodes."""
    method = SpielmanApproximation(epsilon, seed, projection_batch_size)
    distance = ResistanceDistance(solver=solver, method=method)
    return _distance_and_state(distance, A, state, nodes)[0]


@eqx.filter_jit
def spielman_sources_to_targets_resistance_distance(
    A: BCOO,
    sources: Array,
    targets: Array,
    epsilon: float = 0.1,
    seed: int = 0,
    solver: lx.AbstractLinearSolver | None = None,
    state: Any = None,
    *,
    projection_batch_size: int = 64,
) -> Array:
    """Approximate resistance distances from sources to targets."""
    method = SpielmanApproximation(epsilon, seed, projection_batch_size)
    distance = ResistanceDistance(solver=solver, method=method)
    return _distance_and_state(distance, A, state, sources, targets)[0]


@eqx.filter_jit
def _graph_distance_and_state(
    distance: ResistanceDistance,
    graph: AbstractGraph,
    state: Any,
    nodes: Array | None,
    targets: Array | None,
) -> tuple[Array, Any]:
    A = graph.get_adjacency_matrix()
    return _distance_and_state(distance, A, state, nodes, targets)


def _distance_and_state(
    distance: ResistanceDistance,
    A: BCOO,
    state: Any,
    nodes: Array | None = None,
    targets: Array | None = None,
) -> tuple[Array, Any]:
    """Resistance distances of the graph with adjacency `A`, and the solver state.

    The distances are between all vertices without `nodes`, among `nodes`
    without `targets`, and from `nodes` to `targets` otherwise. The solver
    state is refreshed once per query against the grounded Laplacian of `A`;
    `_solver_states` gives the state the solves use and the state returned.
    The refresh sees only stop-gradient matrix values and runs outside the
    custom VJPs, so it is never differentiated and the states it builds carry
    no gradient. The backward passes reuse the state the solves used.
    """
    solve_state = None
    if distance.solver is not None:
        solve_state, state = _solver_states(
            _reduced_laplacian(A.data, A.indices, A.shape),
            distance.solver,
            None,
            state,
        )
    return _distances(distance, A, solve_state, nodes, targets), state


def _distances(
    distance: ResistanceDistance,
    A: BCOO,
    solve_state: Any,
    nodes: Array | None,
    targets: Array | None,
) -> Array:
    """The distances of `_distance_and_state`, solved with `solve_state`."""
    method, solver = distance.method, distance.solver
    if isinstance(method, SpielmanApproximation):
        return _spielman_distances(A, method, solver, solve_state, nodes, targets)
    if targets is not None:
        assert nodes is not None
        all_pairs = _distances(distance, A, solve_state, None, None)
        return all_pairs[nodes[:, None], targets[None, :]]
    if solver is None:
        all_pairs = p_inv_resistance_distance(A)
        if nodes is None:
            return all_pairs
        return all_pairs[nodes[:, None], nodes[None, :]]
    if nodes is None:
        nodes = jnp.arange(A.shape[0])
    selected_inverse = _exact_selected_inverse(
        A.data, A.indices, A.shape, nodes.astype(A.indices.dtype), solver, solve_state
    )
    diagonal = jnp.diag(selected_inverse)
    return diagonal[:, None] + diagonal[None, :] - selected_inverse - selected_inverse.T


def _spielman_distances(
    A: BCOO,
    method: "SpielmanApproximation",
    solver: lx.AbstractLinearSolver | None,
    solve_state: Any,
    nodes: Array | None,
    targets: Array | None,
) -> Array:
    """Spielman-Srivastava distances, selected as in `_distance_and_state`."""
    if nodes is None:
        features_reduced = _spielman_features_reduced(
            A.data, A.indices, A.shape, method.epsilon, method.seed, solver, solve_state
        )
        return _distances_from_features_reduced(features_reduced)
    sources = nodes.astype(A.indices.dtype)
    selected = sources
    if targets is not None:
        selected = jnp.concatenate([sources, targets.astype(A.indices.dtype)])
    features = _spielman_selected_features(
        A.data,
        A.indices,
        A.shape,
        selected,
        method.epsilon,
        method.seed,
        solver,
        solve_state,
        method.projection_batch_size,
    )
    if targets is None:
        distances = _distances_from_selected_features(features, features)
        return distances.at[jnp.diag_indices(sources.shape[0])].set(0)
    n_sources = sources.shape[0]
    return _distances_from_selected_features(
        features[:, :n_sources], features[:, n_sources:]
    )


def _projection_size(n: int, epsilon: float) -> int:
    """Number of Spielman-Srivastava random projections for accuracy epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive (controls approximation accuracy).")
    return max(1, math.ceil(math.log(max(n, 2)) / epsilon**2))


def _reduced_laplacian(data: Array, indices: Array, shape: tuple[int, ...]) -> BCOO:
    """Grounded Laplacian: the Laplacian without the last vertex's row and column."""
    return graph_laplacian(BCOO((data, indices), shape=shape))[:-1, :-1]


def _solve_reduced_laplacian(
    L_reduced: BCOO,
    rhs: Array,
    solver: lx.AbstractLinearSolver | None,
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
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    options = None
    if solver is not None:
        state, options = solver.transpose(state, {})
    return _solve_reduced_laplacian(L_reduced.T, rhs, solver, state, options)


def _node_potentials(
    L_reduced: BCOO,
    nodes: Array,
    solver: lx.AbstractLinearSolver | None,
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
    # The kernel VJPs below give the grounded-Laplacian cotangent
    # K = -Lambda V.T, where Lambda = adjoint and V = potentials. To pull this
    # back to one stored adjacency coefficient a_uv, use L = diag(row sums) - a:
    #   dL = e_u (e_u - e_v).T da_uv
    #   dloss/da_uv = K_uu - K_uv
    #              = -sum_j Lambda[u, j] * (V[u, j] - V[v, j]).
    # Grounding is equivalent to setting the ground row of Lambda and V to zero.
    # Thus we can evaluate the last line without ever constructing dense K.
    # Each duplicate storage position gets its own contribution; upstream AD
    # sums contributions if one parameter controls both directed entries of an
    # undirected edge. No factor of two is needed here. Self-loops give zero
    # because V_u - V_u = 0; padding is masked out below.
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
    solver: lx.AbstractLinearSolver | None,
    solve_state: Any,
) -> Array:
    return _exact_selected_inverse_fwd(
        None, data, indices, shape, nodes, solver, solve_state
    )[0]


def _exact_selected_inverse_fwd(
    perturbed, data, indices, shape, nodes, solver, solve_state
):
    del perturbed
    reduced = _reduced_laplacian(data, indices, shape)
    potentials = _node_potentials(reduced, nodes, solver, solve_state)
    return _grounded_values(potentials, nodes), potentials


_exact_selected_inverse.def_fwd(_exact_selected_inverse_fwd)


@_exact_selected_inverse.def_bwd
def _exact_selected_inverse_bwd(
    potentials, cotangent, perturbed, data, indices, shape, nodes, solver, solve_state
):
    del data, nodes, solver, solve_state
    if cotangent is None or not perturbed:
        return None
    # Let q = n - 1 be the number of ungrounded vertices and m the site count.
    # A is the (q, q) grounded Laplacian; E is the fixed (q, m) site basis.
    # Forward: A X = E, S = E.T X. Here potentials = X and cotangent = G,
    # where G = dloss/dS has shape (m, m). With <U, V> = sum(U * V), a VJP
    # finds the coefficient of dA in dloss = <G, dS>:
    #   A dX = -(dA) X                    (differentiate A X = E)
    #   dloss = <G, E.T dX> = <E G, dX>  (differentiate S = E.T X)
    # Define the adjoint by A.T Lambda = E G. Then
    #   dloss = <Lambda, A dX>
    #         = -<Lambda, (dA) X> = <-Lambda X.T, dA>.
    # The graph is undirected, so A.T = A and Lambda = A^-1 E G = X G.
    # This uses the saved forward solutions: no backward solve is needed.
    # G need not be symmetric; the formula uses X G, not X G.T. The shared
    # helper applies the remaining chain rule from A to the stored edge data.
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
    solver: lx.AbstractLinearSolver | None,
    solve_state: Any,
) -> Array:
    return _spielman_features_reduced_fwd(
        None, data, indices, shape, epsilon, seed, solver, solve_state
    )[0]


def _spielman_features_reduced_fwd(
    perturbed, data, indices, shape, epsilon, seed, solver, solve_state
):
    del perturbed
    reduced = _reduced_laplacian(data, indices, shape)
    projection = _spielman_projection(data, indices, shape, epsilon, seed)[:, :-1]
    potentials = _solve_reduced_laplacian(reduced, projection.T, solver, solve_state)
    return potentials.T, (potentials, reduced)


_spielman_features_reduced.def_fwd(_spielman_features_reduced_fwd)


@_spielman_features_reduced.def_bwd
def _spielman_features_reduced_bwd(
    residuals,
    cotangent,
    perturbed,
    data,
    indices,
    shape,
    epsilon,
    seed,
    solver,
    solve_state,
):
    if cotangent is None or not perturbed:
        return None
    potentials, reduced = residuals
    # Let A be the (q, q) grounded Laplacian, q = n - 1, and P the (k, q)
    # weighted projection with its ground column removed. Its signs are fixed,
    # but its weights depend on data. Forward: A Y = P.T, F = Y.T.
    # Here potentials = Y and cotangent = H = dloss/dF, with shape (k, q).
    # Using the Frobenius inner product <U, V> = sum(U * V):
    #   A dY = dP.T - (dA) Y               (differentiate the solve)
    #   dloss = <H, dF> = <H.T, dY>        (differentiate F = Y.T)
    # Solve A.T Lambda = H.T to eliminate dY:
    #   dloss = <Lambda, A dY>
    #         = <Lambda, dP.T - (dA) Y>
    #         = <Lambda.T, dP> + <-Lambda Y.T, dA>.
    # Both P and A depend on the same edge data, so their pullbacks must be
    # added. The helper contracts -Lambda Y.T directly into sparse storage;
    # ordinary AD pulls Lambda.T through the actual weighted projection.
    adjoint = _solve_adjoint(reduced, cotangent.T, solver, solve_state)
    _, projection_pullback = jax.vjp(
        lambda weights: _spielman_projection(weights, indices, shape, epsilon, seed),
        data,
    )
    # The projection helper returns all n columns; the removed ground column
    # did not affect F, so its cotangent must be zero.
    (projection_gradient,) = projection_pullback(jnp.pad(adjoint.T, ((0, 0), (0, 1))))
    return (
        _laplacian_data_pullback(adjoint, potentials, indices, shape)
        + projection_gradient
    )


def _edge_potential_deltas(
    potentials: Array, indices: Array, shape: tuple[int, ...]
) -> Array:
    deltas = _grounded_values(potentials, indices[:, 0]) - _grounded_values(
        potentials, indices[:, 1]
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
    projection_batch_size: int,
) -> Array:
    k = _projection_size(shape[0], epsilon)
    edge_deltas = _edge_potential_deltas(potentials, indices, shape)
    scaled_weights = _sqrt_half_weights(data) * _projection_scale(k, data.dtype)

    if k * data.shape[0] <= SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES:
        signs = _rademacher_projection_signs(k, data.shape[0], seed, data.dtype)
        # Scale the (edges, sites) differences once instead of weighting every
        # entry of the (projections, edges) sign matrix before the contraction.
        return signs @ (scaled_weights[:, None] * edge_deltas)

    key = jax.random.PRNGKey(seed)

    @jax.checkpoint
    def project_one(projection_index):
        projection_key = jax.random.fold_in(key, projection_index)
        signs = jax.random.rademacher(projection_key, data.shape).astype(data.dtype)
        return (signs * scaled_weights) @ edge_deltas

    # lax.map handles a final partial batch with the same global fold_in keys.
    # Checkpointing each row recomputes signs in reverse mode instead of saving
    # all k rows. Working storage scales with the user-selected batch size;
    # there is no automatic memory-based reduction. Dense projections retain
    # their original whole-matrix PRNG key.
    return lax.map(project_one, jnp.arange(k), batch_size=projection_batch_size)


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
    solver: lx.AbstractLinearSolver | None,
    solve_state: Any,
    projection_batch_size: int,
) -> Array:
    return _spielman_selected_features_fwd(
        None,
        data,
        indices,
        shape,
        nodes,
        epsilon,
        seed,
        solver,
        solve_state,
        projection_batch_size,
    )[0]


def _spielman_selected_features_fwd(
    perturbed,
    data,
    indices,
    shape,
    nodes,
    epsilon,
    seed,
    solver,
    solve_state,
    projection_batch_size,
):
    del perturbed
    reduced = _reduced_laplacian(data, indices, shape)
    potentials = _node_potentials(reduced, nodes, solver, solve_state)
    features = _spielman_project_potentials(
        data, indices, shape, potentials, epsilon, seed, projection_batch_size
    )
    return features, (potentials, reduced)


_spielman_selected_features.def_fwd(_spielman_selected_features_fwd)


@_spielman_selected_features.def_bwd
def _spielman_selected_features_bwd(
    residuals,
    cotangent,
    perturbed,
    data,
    indices,
    shape,
    nodes,
    epsilon,
    seed,
    solver,
    solve_state,
    projection_batch_size,
):
    del nodes
    if cotangent is None or not perturbed:
        return None
    potentials, reduced = residuals
    # Let A be the (q, q) grounded Laplacian, q = n - 1, E the fixed (q, m)
    # site basis, and P the (k, q) weighted projection with fixed random signs.
    # Forward: A X = E, Z = P X. Here potentials = X and the incoming cotangent
    # is H = dloss/dZ of shape (k, m). With <U, V> = sum(U * V):
    #   A dX = -(dA) X                       (E is fixed)
    #   dZ = (dP) X + P dX                   (both P and X depend on data)
    #   dloss = <H, dZ> = <H X.T, dP> + <P.T H, dX>.
    # Define Lambda by A.T Lambda = P.T H. Substituting for A dX gives
    #   dloss = <H X.T, dP> + <Lambda, A dX>
    #         = <H X.T, dP> + <-Lambda X.T, dA>.
    # To evaluate the first term without forming the dense H X.T, take the VJP
    # of (data, X) -> P(data) X, treating these as independent arguments. It
    # returns the projection's data gradient with X held fixed, and P.T H as
    # the X cotangent. The latter is the RHS for the adjoint solve; the sparse
    # helper then evaluates the second term. Add both data gradients below.
    # Unlike the exact kernel, a realized finite sketch does not give P.T P = A,
    # so it cannot eliminate this adjoint solve. Streamed projection rows
    # regenerate the same signs in this VJP, avoiding a k-by-edge saved tape.
    _, projection_pullback = jax.vjp(
        lambda weights, values: _spielman_project_potentials(
            weights, indices, shape, values, epsilon, seed, projection_batch_size
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


def _distances_from_features_reduced(features_reduced: Array) -> Array:
    """Pairwise approximate resistance distances from reduced embeddings."""
    features = jnp.pad(features_reduced, ((0, 0), (0, 1)))
    feature_norms = jnp.sum(features**2, axis=0)
    # Pairwise squared distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u*v.
    return feature_norms[:, None] + feature_norms[None, :] - 2 * features.T @ features
