import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array, lax
from jax.experimental.sparse import BCOO
from jax.numpy.linalg import pinv

from jaxscape.distance import AbstractSolverDistance
from jaxscape.graph import AbstractGraph
from jaxscape.solvers.operator import _batched_linear_solve_impl, _solver_state
from jaxscape.utils import _matmul, graph_laplacian


# Building the full Rademacher projection matrix is faster for moderate sizes,
# but beyond this many sign entries the temporary allocation becomes large
# enough that the streamed scan path is the safer default.
SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES = 200_000_000

# Exact sources-to-targets queries solve all target columns at once, and keep
# them for the gradient, when their potentials hold at most this many values.
_UNCHUNKED_POTENTIALS = 2**22
# Otherwise they solve the target columns this many at a time, and again for the
# gradient. Sparse direct solvers reach their full throughput from a few hundred
# columns on.
_TARGET_CHUNK_SIZE = 1024


class ResistanceDistance(AbstractSolverDistance):
    """
    Compute the resistance distances.

    **Attributes:**

    - `solver`: Optional `lineax.AbstractLinearSolver`. Must be compatible with
      BCOO matrices. We currently support `jaxscape.solvers.CholmodSolver`,
      `jaxscape.solvers.PyAMGSolver`, `jaxscape.solvers.AMJaxCGSolver`, and
      `jaxscape.solvers.CuDSSSolver`.
      If None, uses the pseudo-inverse method, which is very memory intensive
      for exact all-pairs distances (densifies the Laplacian matrix), or dense
      solves for approximate distances. Exact sources-to-targets queries
      without a solver use a dense Cholesky factorization of the grounded
      Laplacian, which needs a connected graph: the distances of a
      disconnected graph are NaN.
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

        # Carry a state from one graph to the next, as an optimization loop does.
        # AMJax keeps its hierarchy.
        updated_permeability = permeability * 1.1
        updated_grid = GridGraph(updated_permeability, fun=lambda x, y: (x + y) / 2)
        state = distance.reinit(updated_grid, state)
        dist = distance(updated_grid, state=state)

        # Approximate resistance distance
        distance = ResistanceDistance(
            method=SpielmanApproximation(epsilon=0.05),
            solver=PyAMGSolver(),
        )

        dist = distance(grid)
        ```

    !!! note "Grounding, float32 and memory"

        Apart from the pseudo-inverse of dense all-pairs and nodes-to-nodes
        queries, queries solve the Laplacian grounded at the middle vertex
        `n // 2`: the central pixel of a grid with an odd number of rows, such
        as the window of a windowed analysis. Distances from that vertex involve
        no cancellation. On windows of up to 101 x 101 pixels with a
        permeability floor of `1e-4`, float32 permeability gradients stayed
        within 0.4 % of float64 with `CuDSSSolver`, and within 3.5 % with
        the dense solve and `CholmodSolver`, which takes under 10 % longer in
        float64. Grounded at the last vertex, in a corner of the window, they were
        off by up to 17 times their norm with `CuDSSSolver`.

        Exact sources-to-targets queries on `n` vertices, from `m` sources to
        `k` targets, hold the potentials of all targets when `n * k` is at most
        `2**22`, as in windows of up to 45 x 45 pixels. Beyond, they solve the
        targets in chunks of 1024 and hold at most `n * (m + 1024)` potentials,
        and their gradient solves each target once more.

    !!! warning

        The graph must be undirected for resistance distance to be well-defined.
    """

    solver: lx.AbstractLinearSolver | None = None
    method: "AbstractResistanceMethod" = eqx.field(
        default_factory=lambda: ExactResistance()
    )

    def system_matrix(self, graph: AbstractGraph) -> BCOO:
        """Return the grounded Laplacian of `graph`."""
        return _grounded_laplacian(graph.get_adjacency_matrix())

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
                A,
                nodes,
                self.method.epsilon,
                self.method.seed,
                self.solver,
                state,
                projection_batch_size=self.method.projection_batch_size,
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
                projection_batch_size=self.method.projection_batch_size,
            )
        A = graph.get_adjacency_matrix()
        return _exact_sources_to_targets(
            A.data,
            A.indices,
            A.shape,
            sources.astype(A.indices.dtype),
            targets.astype(A.indices.dtype),
            self.solver,
            state,
        )


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
    selected_inverse = _exact_selected_inverse(
        A.data, A.indices, A.shape, nodes.astype(A.indices.dtype), solver, state
    )
    diagonal = jnp.diag(selected_inverse)
    return diagonal[:, None] + diagonal[None, :] - selected_inverse - selected_inverse.T


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
    return _spielman_resistance_distance_from_data(
        A.data, A.indices, A.shape, epsilon, seed, solver, state
    )


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
    return _spielman_nodes_to_nodes_resistance_distance_from_data(
        A.data,
        A.indices,
        A.shape,
        nodes.astype(A.indices.dtype),
        epsilon,
        seed,
        solver,
        state,
        projection_batch_size,
    )


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
        projection_batch_size,
    )


def _projection_size(n: int, epsilon: float) -> int:
    """Number of Spielman-Srivastava random projections for accuracy epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive (controls approximation accuracy).")
    return max(1, math.ceil(math.log(max(n, 2)) / epsilon**2))


def _grounded_rows(nodes: Array, n: int) -> Array:
    """Rows of the grounded system for `nodes` of a graph of `n` vertices.

    The ground is the middle vertex `n // 2`, the central pixel of a grid with
    an odd number of rows. Grounding a window at its centre, where distances
    start, keeps float32 gradients accurate: potentials grounded in a corner of
    floored padding lose all their digits. The ground vertex, and padding
    indices from `n` on, map to rows from `n - 1` on, past the last row.
    """
    ground = n // 2
    return jnp.where(nodes == ground, n - 1, nodes - (nodes > ground))


def _grounded_laplacian(A: BCOO) -> BCOO:
    """The Laplacian of `A` without the row and the column of the ground vertex."""
    L = graph_laplacian(A)
    n = A.shape[0]
    indices = _grounded_rows(L.indices, n)
    # As in a sliced BCOO matrix, removed entries become zero padding past the
    # last row and column.
    removed = jnp.any(indices >= n - 1, axis=1)
    return BCOO(
        (
            jnp.where(removed, 0, L.data),
            jnp.where(removed[:, None], n - 1, indices),
        ),
        shape=(n - 1, n - 1),
    )


def _without_ground(vertex_values: Array) -> Array:
    """Drop the column of the ground vertex from values of all vertices."""
    return jnp.delete(vertex_values, vertex_values.shape[1] // 2, axis=1)


def _with_ground(values: Array) -> Array:
    """Insert a zero column for the ground vertex into grounded-system values."""
    return jnp.insert(values, (values.shape[1] + 1) // 2, 0, axis=1)


def _prepare_reduced_laplacian(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> tuple[BCOO, Any]:
    reduced = _grounded_laplacian(BCOO((data, indices), shape=shape))
    if solver is not None:
        state = _solver_state(reduced, solver, None, state)
    return reduced, state


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
    return _solve_reduced_laplacian(L_reduced.transpose(), rhs, solver, state, options)


def _node_potentials(
    L_reduced: BCOO,
    nodes: Array,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    return _solve_reduced_laplacian(
        L_reduced, _unit_vectors(L_reduced, nodes), solver, state
    )


def _unit_vectors(L_reduced: BCOO, nodes: Array) -> Array:
    """Columns `e_i` of the grounded system for `nodes`, zero for the ground node."""
    # Duplicate nodes retain duplicate columns.
    q = L_reduced.shape[0]
    return jax.nn.one_hot(_grounded_rows(nodes, q + 1), q, dtype=L_reduced.dtype).T


def _grounded_values(potentials: Array, nodes: Array) -> Array:
    """Select rows of (n - 1, rhs) potentials, with zero at the ground node."""
    q = potentials.shape[0]
    rows = _grounded_rows(nodes, q + 1)
    return jnp.where((rows < q)[:, None], potentials[jnp.minimum(rows, q - 1)], 0)


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
    state: Any,
) -> Array:
    return _exact_selected_inverse_fwd(
        None, data, indices, shape, nodes, solver, state
    )[0]


def _exact_selected_inverse_fwd(perturbed, data, indices, shape, nodes, solver, state):
    del perturbed
    reduced, solve_state = _prepare_reduced_laplacian(
        data, indices, shape, solver, state
    )
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
    adjoint = _matmul(potentials, cotangent)
    return _laplacian_data_pullback(adjoint, potentials, indices, shape)


def _target_chunks(targets: Array, n: int) -> Array:
    """Split `targets` of a graph of `n` vertices into chunks solved together.

    One chunk when the potentials of all targets hold at most
    `_UNCHUNKED_POTENTIALS` values, otherwise equal chunks of at most
    `_TARGET_CHUNK_SIZE`. The last chunk is padded with the ground vertex, whose
    potentials are zero.
    """
    k = targets.shape[0]
    n_chunks = 1 if n * k <= _UNCHUNKED_POTENTIALS else -(-k // _TARGET_CHUNK_SIZE)
    padding = n_chunks * -(-k // n_chunks) - k
    return jnp.pad(targets, (0, padding), constant_values=n // 2).reshape(n_chunks, -1)


def _exact_solver_state(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> tuple[BCOO, Any]:
    """The grounded Laplacian, and the solver state or dense factor that solves it."""
    reduced, solve_state = _prepare_reduced_laplacian(
        data, indices, shape, solver, state
    )
    if solver is None:
        # The grounded Laplacian of a connected graph is positive definite: a
        # dense Cholesky factorization solves it 1.3 times faster than the LU
        # factorization of `_solve_reduced_laplacian`.
        solve_state = jnp.linalg.cholesky(reduced.todense())
    return reduced, solve_state


def _exact_potentials(
    reduced: BCOO,
    nodes: Array,
    solver: lx.AbstractLinearSolver | None,
    solve_state: Any,
) -> Array:
    """Columns of the grounded inverse for `nodes`, zero for the ground vertex."""
    if solver is None:
        basis = _unit_vectors(reduced, nodes)
        return jax.scipy.linalg.cho_solve((solve_state, True), basis)
    return _node_potentials(reduced, nodes, solver, solve_state)


def _own_values(potentials: Array, nodes: Array) -> Array:
    """The value of each potential column at its own node."""

    def own_value(column, node):
        return _grounded_values(column[:, None], node[None])[0, 0]

    return jax.vmap(own_value, in_axes=(1, 0))(potentials, nodes)


@eqx.filter_custom_vjp
def _exact_sources_to_targets(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    sources: Array,
    targets: Array,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    return _exact_sources_to_targets_fwd(
        None, data, indices, shape, sources, targets, solver, state
    )[0]


def _exact_sources_to_targets_fwd(
    perturbed, data, indices, shape, sources, targets, solver, state
):
    del perturbed
    reduced, solve_state = _exact_solver_state(data, indices, shape, solver, state)
    # With S the grounded inverse, R_ij = S_ii + S_jj - 2 S_ij for source i and
    # target j. The source columns of S give S_ii and S_ij, the target columns
    # their diagonal S_jj.
    m = sources.shape[0]
    chunks = _target_chunks(targets, shape[0])
    if chunks.shape[0] == 1:
        # One solve for all columns, whose potentials stay for the gradient.
        nodes = jnp.concatenate([sources, chunks[0]])
        potentials = _exact_potentials(reduced, nodes, solver, solve_state)
        diagonal = _own_values(potentials[:, m:], chunks[0])
        residuals = (potentials, None)
    else:
        # The target columns, chunk by chunk, keeping their diagonal only: the
        # gradient solves them again.
        potentials = _exact_potentials(reduced, sources, solver, solve_state)

        def target_diagonal(chunk):
            chunk_potentials = _exact_potentials(reduced, chunk, solver, solve_state)
            return _own_values(chunk_potentials, chunk)

        diagonal = lax.map(target_diagonal, chunks)
        residuals = (potentials, (reduced, solve_state))
    source_potentials = potentials[:, :m]
    distances = (
        _own_values(source_potentials, sources)[:, None]
        + diagonal.reshape(-1)[None, : targets.shape[0]]
        - 2 * _grounded_values(source_potentials, targets).T
    )
    return distances, residuals


_exact_sources_to_targets.def_fwd(_exact_sources_to_targets_fwd)


@_exact_sources_to_targets.def_bwd
def _exact_sources_to_targets_bwd(
    residuals,
    cotangent,
    perturbed,
    data,
    indices,
    shape,
    sources,
    targets,
    solver,
    state,
):
    del state
    if cotangent is None or not perturbed:
        return None
    # As in `_exact_selected_inverse_bwd`, the adjoint is X G, where X are the
    # potentials of the source columns X_s and target columns X_t, and G is the
    # cotangent of S. For the cotangent H of the (m, k) distances,
    #   G = [[diag(H 1), -H], [-H.T, diag(H.T 1)]],
    # so X G = [X_s diag(H 1) - X_t H.T, X_t diag(H.T 1) - X_s H]. This costs
    # O(n m k) and never forms G.
    potentials, solver_state = residuals
    m = sources.shape[0]
    source_potentials = potentials[:, :m]
    if solver_state is None:
        # The forward pass kept the target columns after the source columns.
        target_potentials = potentials[:, m:]
        adjoint = jnp.concatenate(
            [
                source_potentials * jnp.sum(cotangent, axis=1)
                - _matmul(target_potentials, cotangent.T),
                target_potentials * jnp.sum(cotangent, axis=0)
                - _matmul(source_potentials, cotangent),
            ],
            axis=1,
        )
        return _laplacian_data_pullback(adjoint, potentials, indices, shape)

    # The data pullback sums over columns: each chunk of target columns, solved
    # again, adds its own term and its part of X_t H.T.
    reduced, solve_state = solver_state
    chunks = _target_chunks(targets, shape[0])
    padding = chunks.size - targets.shape[0]
    chunk_cotangents = jnp.pad(cotangent, ((0, 0), (0, padding)))
    chunk_cotangents = chunk_cotangents.reshape(-1, *chunks.shape).swapaxes(0, 1)

    def add_chunk(carry, chunk_and_cotangent):
        gradient, weighted_targets = carry
        chunk, chunk_cotangent = chunk_and_cotangent
        target_potentials = _exact_potentials(reduced, chunk, solver, solve_state)
        adjoint = target_potentials * jnp.sum(chunk_cotangent, axis=0) - _matmul(
            source_potentials, chunk_cotangent
        )
        gradient = gradient + _laplacian_data_pullback(
            adjoint, target_potentials, indices, shape
        )
        weighted_targets = weighted_targets + _matmul(
            target_potentials, chunk_cotangent.T
        )
        return (gradient, weighted_targets), None

    carry = (jnp.zeros_like(data), jnp.zeros_like(source_potentials))
    (gradient, weighted_targets), _ = lax.scan(
        add_chunk, carry, (chunks, chunk_cotangents)
    )
    adjoint = source_potentials * jnp.sum(cotangent, axis=1) - weighted_targets
    return gradient + _laplacian_data_pullback(
        adjoint, source_potentials, indices, shape
    )


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
    state: Any,
) -> Array:
    return _spielman_features_reduced_fwd(
        None, data, indices, shape, epsilon, seed, solver, state
    )[0]


def _spielman_features_reduced_fwd(
    perturbed, data, indices, shape, epsilon, seed, solver, state
):
    del perturbed
    reduced, solve_state = _prepare_reduced_laplacian(
        data, indices, shape, solver, state
    )
    projection = _without_ground(
        _spielman_projection(data, indices, shape, epsilon, seed)
    )
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
    (projection_gradient,) = projection_pullback(_with_ground(adjoint.T))
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
        return _matmul(signs, scaled_weights[:, None] * edge_deltas)

    key = jax.random.PRNGKey(seed)

    @jax.checkpoint
    def project_one(projection_index):
        projection_key = jax.random.fold_in(key, projection_index)
        signs = jax.random.rademacher(projection_key, data.shape).astype(data.dtype)
        return _matmul(signs * scaled_weights, edge_deltas)

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
        - 2 * _matmul(source_features.T, target_features)
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
    state: Any,
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
        state,
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
    state,
    projection_batch_size,
):
    del perturbed
    reduced, solve_state = _prepare_reduced_laplacian(
        data, indices, shape, solver, state
    )
    potentials = _node_potentials(reduced, nodes, solver, solve_state)
    features = _spielman_project_potentials(
        data, indices, shape, potentials, epsilon, seed, projection_batch_size
    )
    return features, (potentials, reduced, solve_state)


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
    state,
    projection_batch_size,
):
    del nodes, state
    if cotangent is None or not perturbed:
        return None
    potentials, reduced, solve_state = residuals
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


def _spielman_sources_to_targets_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    sources: Array,
    targets: Array,
    epsilon: float,
    seed: int,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    projection_batch_size: int,
) -> Array:
    nodes = jnp.concatenate([sources, targets])
    features = _spielman_selected_features(
        data, indices, shape, nodes, epsilon, seed, solver, state, projection_batch_size
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
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    projection_batch_size: int,
) -> Array:
    features = _spielman_selected_features(
        data, indices, shape, nodes, epsilon, seed, solver, state, projection_batch_size
    )
    distances = _distances_from_selected_features(features, features)
    return distances.at[jnp.diag_indices(nodes.shape[0])].set(0)


def _distances_from_features_reduced(features_reduced: Array) -> Array:
    """Pairwise approximate resistance distances from reduced embeddings."""
    features = _with_ground(features_reduced)
    feature_norms = jnp.sum(features**2, axis=0)
    # Pairwise squared distances: ||u - v||^2 = ||u||^2 + ||v||^2 - 2 u*v.
    return (
        feature_norms[:, None]
        + feature_norms[None, :]
        - 2 * _matmul(features.T, features)
    )


def _spielman_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    epsilon: float,
    seed: int,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    features_reduced = _spielman_features_reduced(
        data, indices, shape, epsilon, seed, solver, state
    )
    return _distances_from_features_reduced(features_reduced)
