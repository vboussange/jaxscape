import math
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax import Array, lax
from jax.experimental.sparse import BCOO
from jax.numpy.linalg import pinv

from jaxscape.distance import AbstractSolverDistance
from jaxscape.graph import AbstractGraph, GridGraph
from jaxscape.solvers.operator import _batched_linear_solve_impl, _solver_state
from jaxscape.utils import _matmul, bcoo_diag, graph_laplacian


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
# Spielman window queries take the differences of about this many potentials
# at once, and again for the gradient.
_WINDOW_DIFFERENCES_PER_STEP = 2**24


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
        queries, queries solve the Laplacian grounded at one vertex: among the
        vertices whose weighted degree is at least half the largest, the
        nearest to the middle vertex `n // 2` by index. On a grid, it is a
        pixel of near-maximum permeability near the centre, such as the centre
        of the window of a windowed analysis, from which distances involve no
        cancellation. The ground's degree is added to its diagonal entry, so
        that every ground gives one sparsity pattern. On windows of up to
        101 x 101 pixels with a permeability floor of `1e-4`, float32
        permeability gradients stayed within 0.4 % of float64 with
        `CuDSSSolver`, and within 3.5 % with the dense solve and
        `CholmodSolver`, which takes under 10 % longer in float64. Grounded at
        a floored pixel, they were off by up to 17 times their norm with
        `CuDSSSolver`.

        Exact sources-to-targets queries on `n` vertices, from `m` sources to
        `k` targets, hold the potentials of all targets when they are a few
        million values, as in small windows. Beyond, they solve the targets in
        chunks of `c`, about a thousand, and hold at most `n * (m + c)`
        potentials, and their gradient solves each target once more.

    !!! warning

        The graph must be undirected for resistance distance to be well-defined.
    """

    solver: lx.AbstractLinearSolver | None = None
    method: "AbstractResistanceMethod" = eqx.field(
        default_factory=lambda: ExactResistance()
    )
    sources_share_work: ClassVar[bool] = True
    symmetric: ClassVar[bool] = True

    def system_matrix(self, graph: AbstractGraph) -> BCOO:
        """Return the grounded Laplacian of `graph`."""
        return _grounded_laplacian(graph.get_adjacency_matrix())[0]

    def exact(self) -> "ResistanceDistance":
        """Return the exact resistance distance, with the same solver.

        **Returns:**

        `ResistanceDistance` with `ExactResistance` and the solver of this one,
        whose solver states serve each other.
        """
        return ResistanceDistance(solver=self.solver)

    def reseeded(self, index: Array) -> "ResistanceDistance":
        """Return the distance whose random projections depend on `index` too.

        With `SpielmanApproximation`, the seed of the projections becomes one
        drawn from `seed` and `index`. Exact resistances return themselves.

        **Arguments:**

        - `index`: Integer, which may be traced.

        **Returns:**

        `ResistanceDistance` with the same solver and method, but the seed.
        """
        if not isinstance(self.method, SpielmanApproximation):
            return self
        key = jax.random.fold_in(jax.random.PRNGKey(self.method.seed), index)
        seed = jax.random.bits(key, dtype=jnp.uint32)
        return eqx.tree_at(lambda d: d.method.seed, self, seed)

    def debiased_proximity(
        self, proximity: Callable[[Array], Array], graph: AbstractGraph
    ) -> Callable[[Array], Array]:
        r"""Return the proximity to apply to the distances of queries on `graph`.

        With `SpielmanApproximation`, an estimate $\hat R$ of the resistance
        $R$ is unbiased, with variance $2 R^2 / k$ for $k$ projections. A
        proximity $K$ convex in $R$ is then biased upward: by about 5 %, 15 %
        and 30 % for `epsilon` 0.1, 0.2 and 0.3 with the default proximity of
        the windowed analyses. The returned proximity is corrected to second
        order:

        $$K(\hat R) - \frac{\hat R^2}{k} K''(\hat R)$$

        For a negative exponential, it is
        $K(\hat R) \big(1 - (\hat R K'(\hat R) / K(\hat R))^2 / k\big)$. The
        correction does not reduce the variance of the proximities. Exact
        resistances keep `proximity`.

        **Arguments:**

        - `proximity`: Function mapping resistances to proximities, twice
          differentiable.
        - `graph`: Graph of the queries.

        **Returns:**

        Function mapping the resistances of queries on `graph` to proximities.
        """
        if not isinstance(self.method, SpielmanApproximation):
            return proximity
        k = _projection_size(graph.nv, self.method.epsilon)

        def slope(distance):
            return jax.jvp(proximity, (distance,), (jnp.ones_like(distance),))

        def debiased(distance):
            (value, _), (_, curvature) = jax.jvp(
                slope, (distance,), (jnp.ones_like(distance),)
            )
            return value - distance**2 * curvature / k

        return debiased

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        A = graph.get_adjacency_matrix()
        if isinstance(self.method, SpielmanApproximation):
            return spielman_resistance_distance(
                A,
                self.method.epsilon,
                self.method.seed,
                self.solver,
                state,
                projection_batch_size=self.method.projection_batch_size,
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

    def sources_to_windows_distance(
        self, graph: GridGraph, sources: Array, size: int, state: Any = None
    ) -> Array:
        """Return the distances from each source to the pixels of its window.

        As `AbstractDistance.sources_to_windows_distance`. With
        `SpielmanApproximation`, the query solves the `k` projections
        `projection_batch_size` at a time, and adds the squared differences of
        their potentials between each source and the pixels of its window. It
        holds the potentials of one batch of projections and the windows.

        Its gradient estimates the gradient of the exact resistances,
        contracted with the cotangent of the estimates, from `k` random
        probes of the pairs of a source and a pixel of its window, solved
        `projection_batch_size` at a time with two right-hand sides each. Each
        probe draws a sign per source and a sign per pixel, whose product
        weighs a pair. The estimate of each stored edge is unbiased. The
        gradient of the estimates themselves varies much more between seeds:
        on a crop of 120 x 120 pixels of the Maryland land cover of the
        connectivity analysis example, with `dependency_range` 30, one tile
        and `epsilon` 0.2, the permeability sensitivities of two seeds had a
        Pearson correlation of 0.16 with it, and 0.96 with the probes, which
        reached 0.99 at `epsilon` 0.1 and took about half the time.

        **Arguments:**

        - `graph`: Grid graph on which to compute distances.
        - `sources`: Source vertex indices with shape `(m,)`.
        - `size`: Odd number of pixels along a side of the windows.
        - `state`: Optional solver state. Defaults to automatic initialization.

        **Returns:**

        Distances with shape `(m, size, size)`.
        """
        if not isinstance(self.method, SpielmanApproximation):
            return super().sources_to_windows_distance(graph, sources, size, state)
        if size % 2 == 0:
            raise ValueError("`size` must be odd.")
        if state is None:
            state = self.init(jax.tree.map(lax.stop_gradient, graph))
        return _spielman_window_query(
            graph, sources, size, self.method, self.solver, state
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
      projections: `k = ceil(log(n_vertices) / epsilon**2)`. All-pairs and
      sources-to-targets queries solve `k` linear systems and hold `k`
      potentials of every vertex; nodes-to-nodes queries solve one per node.
      Window queries, those of windowed analyses, solve `k` linear systems
      and hold the potentials of `projection_batch_size` of them.
    - `seed`: Random seed for the Rademacher edge projections. Each projection
      draws its signs from `seed` and its index, so that all queries on a graph
      share their projections, in float32 and float64, with and without 64-bit
      types (`jax_enable_x64`).
    - `projection_batch_size`: Number of projections that window queries solve
      together. Other queries draw the random signs of this many projections
      together when the signs of all projections, one per stored edge and
      projection, exceed `SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES`
      entries, and all at once below. Larger batches can improve throughput
      at the cost of more working memory.
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
    *,
    projection_batch_size: int = 64,
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
        A.data, A.indices, A.shape, epsilon, seed, solver, state, projection_batch_size
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
    """Approximate resistance distances from sources to targets.

    Embeds every vertex with `ceil(log(n_vertices) / epsilon**2)` solves, and as
    many adjoint solves for the gradient, whatever the numbers of sources and
    targets.
    """
    features = _spielman_features_reduced(
        A.data,
        A.indices,
        A.shape,
        epsilon,
        seed,
        solver,
        state,
        projection_batch_size,
    )
    return _distances_from_selected_features(features[:, sources], features[:, targets])


def _projection_size(n: int, epsilon: float) -> int:
    """Number of Spielman-Srivastava random projections for accuracy epsilon."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive (controls approximation accuracy).")
    return max(1, math.ceil(math.log(max(n, 2)) / epsilon**2))


def _grounded_laplacian(A: BCOO) -> tuple[BCOO, Array]:
    """The Laplacian of `A` grounded at a vertex, and that vertex.

    The ground is, among the vertices whose weighted degree is at least half the
    largest, the nearest to the middle vertex `n // 2` in a grid of `w` columns,
    where `w` is the largest index difference between neighbours, with the
    lowest index among the nearest. On a grid graph, it is the pixel of
    near-maximum permeability nearest to the centre, in Manhattan distance. In
    float32, a ground on a barrier or in floored padding makes the system too
    ill-conditioned to solve, and a ground near the sources of windows and
    tiles, at their centre, avoids cancellation in their distances.

    The returned matrix `L + d e_g e_g^T` adds the degree `d` of the ground `g`
    to its diagonal entry: it keeps the sparsity pattern of the Laplacian `L`
    whatever the ground, and it is positive definite for a connected graph. Its
    solutions minus their value at `g` are those of `L` without the row and the
    column of `g`, and give the same resistances.
    """
    n = A.shape[0]
    rows, columns = A.indices[:, 0], A.indices[:, 1]
    # Padding indices, from `n` on, are dropped.
    degree = jax.ops.segment_sum(A.data, rows, num_segments=n)
    width = jnp.maximum(jnp.max(jnp.where(rows < n, jnp.abs(rows - columns), 0)), 1)
    vertices, middle = jnp.arange(n), n // 2
    offset = jnp.abs(vertices // width - middle // width) + jnp.abs(
        vertices % width - middle % width
    )
    ground = jnp.argmin(jnp.where(degree >= jnp.max(degree) / 2, offset, 2 * n))
    diagonal = bcoo_diag(degree.at[ground].multiply(2), A.indices.dtype)
    return diagonal - A, ground


class _GroundedSystem(eqx.Module):
    """A grounded Laplacian, its ground vertex, and what solves it.

    **Attributes:**

    - `matrix`: The grounded Laplacian, from `_grounded_laplacian`.
    - `ground`: The ground vertex.
    - `state`: The solver state for `matrix`, or its dense Cholesky factor
      without a solver.
    """

    matrix: BCOO
    ground: Array
    state: Any


def _grounded_system(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> _GroundedSystem:
    """The grounded Laplacian of the adjacency matrix of `data` and `indices`.

    A solver reinitializes `state` for it. Without a solver, the grounded
    Laplacian of a connected graph is positive definite: a dense Cholesky
    factorization solves it 1.3 times faster than an LU factorization.
    """
    matrix, ground = _grounded_laplacian(BCOO((data, indices), shape=shape))
    if solver is None:
        state = jnp.linalg.cholesky(matrix.todense())
    else:
        state = _solver_state(matrix, solver, None, state)
    return _GroundedSystem(matrix, ground, state)


def _solve(
    system: _GroundedSystem,
    rhs: Array,
    solver: lx.AbstractLinearSolver | None,
    transpose: bool = False,
) -> Array:
    """The potentials of the right-hand side columns `rhs`, zero at the ground.

    With `transpose`, of the transposed system, as adjoint solves need.
    """
    if solver is None:
        potentials = jax.scipy.linalg.cho_solve((system.state, True), rhs)
    else:
        matrix, state, options = system.matrix, system.state, None
        if transpose:
            matrix = matrix.transpose()
            state, options = solver.transpose(state, {})
        # Zero columns complete a multiple of 8 right-hand sides: cuDSS solved
        # some other numbers of them twice as slowly, 1079 against 1080.
        m = rhs.shape[1]
        rhs = jnp.pad(rhs, ((0, 0), (0, -m % 8)))
        potentials = _batched_linear_solve_impl(
            matrix, rhs, solver, state=state, options=options
        )[:, :m]
    return potentials - potentials[system.ground]


def _node_potentials(
    system: _GroundedSystem, nodes: Array, solver: lx.AbstractLinearSolver | None
) -> Array:
    """Columns of the grounded inverse for `nodes`, zero for the ground vertex."""
    # Duplicate nodes retain duplicate columns.
    n = system.matrix.shape[0]
    basis = jax.nn.one_hot(nodes, n, dtype=system.matrix.dtype).T
    return _solve(system, basis, solver)


def _values(potentials: Array, nodes: Array) -> Array:
    """The rows of `potentials` for `nodes`, zero for padding indices past them."""
    return potentials.at[nodes].get(mode="fill", fill_value=0)


def _laplacian_data_pullback(
    adjoint: Array, potentials: Array, indices: Array, shape: tuple[int, ...]
) -> Array:
    # The kernel VJPs below give the grounded-Laplacian cotangent
    # K = -Lambda V.T, where Lambda = adjoint and V = potentials. To pull this
    # back to one stored adjacency coefficient a_uv, use L = diag(row sums) - a:
    #   dL = e_u (e_u - e_v).T da_uv
    #   dloss/da_uv = K_uu - K_uv
    #              = -sum_j Lambda[u, j] * (V[u, j] - V[v, j]).
    # Lambda and V are zero at the ground vertex, as for the Laplacian without
    # the row and the column of the ground. Thus we can evaluate the last line
    # without ever constructing dense K.
    # Each duplicate storage position gets its own contribution; upstream AD
    # sums contributions if one parameter controls both directed entries of an
    # undirected edge. No factor of two is needed here. Self-loops give zero
    # because V_u - V_u = 0; padding is masked out below.
    # Keep gather/multiply/reduce fused; explicit edge vmap/chunking is slower.
    values = -jnp.sum(
        _values(adjoint, indices[:, 0])
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
    system = _grounded_system(data, indices, shape, solver, state)
    potentials = _node_potentials(system, nodes, solver)
    return _values(potentials, nodes), potentials


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
    `_TARGET_CHUNK_SIZE`. The last chunk is padded with vertex 0, whose
    values the query drops.
    """
    k = targets.shape[0]
    n_chunks = 1 if n * k <= _UNCHUNKED_POTENTIALS else -(-k // _TARGET_CHUNK_SIZE)
    padding = n_chunks * -(-k // n_chunks) - k
    return jnp.pad(targets, (0, padding)).reshape(n_chunks, -1)


def _own_values(potentials: Array, nodes: Array) -> Array:
    """The value of each potential column at its own node."""

    return potentials[nodes, jnp.arange(nodes.shape[0])]


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
    system = _grounded_system(data, indices, shape, solver, state)
    # With S the grounded inverse, R_ij = S_ii + S_jj - 2 S_ij for source i and
    # target j. The source columns of S give S_ii and S_ij, the target columns
    # their diagonal S_jj.
    m = sources.shape[0]
    chunks = _target_chunks(targets, shape[0])
    if chunks.shape[0] == 1:
        # One solve for all columns, whose potentials stay for the gradient.
        nodes = jnp.concatenate([sources, chunks[0]])
        potentials = _node_potentials(system, nodes, solver)
        diagonal = _own_values(potentials[:, m:], chunks[0])
        residuals = (potentials, None)
    else:
        # The target columns, chunk by chunk, keeping their diagonal only: the
        # gradient solves them again.
        potentials = _node_potentials(system, sources, solver)

        def target_diagonal(chunk):
            chunk_potentials = _node_potentials(system, chunk, solver)
            return _own_values(chunk_potentials, chunk)

        diagonal = lax.map(target_diagonal, chunks)
        residuals = (potentials, system)
    source_potentials = potentials[:, :m]
    distances = (
        _own_values(source_potentials, sources)[:, None]
        + diagonal.reshape(-1)[None, : targets.shape[0]]
        - 2 * _values(source_potentials, targets).T
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
    potentials, system = residuals
    m = sources.shape[0]
    source_potentials = potentials[:, :m]
    if system is None:
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
    chunks = _target_chunks(targets, shape[0])
    padding = chunks.size - targets.shape[0]
    chunk_cotangents = jnp.pad(cotangent, ((0, 0), (0, padding)))
    chunk_cotangents = chunk_cotangents.reshape(-1, *chunks.shape).swapaxes(0, 1)

    def add_chunk(carry, chunk_and_cotangent):
        gradient, weighted_targets = carry
        chunk, chunk_cotangent = chunk_and_cotangent
        target_potentials = _node_potentials(system, chunk, solver)
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


def _projections(
    k: int,
    n_edges: int,
    seed: int,
    dtype: jnp.dtype,
    projection_batch_size: int,
    project: Callable[[Array], Array],
) -> Array:
    """Stack `project(signs)` over the random signs of `k` projections.

    `project` maps the signs of some projections, with shape `(m, n_edges)`, to
    `m` rows. Each projection draws its signs from `seed` and its index (see
    `_signs`). When the signs of all projections hold at most
    `SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES` entries, they are drawn at
    once. Otherwise they are drawn `projection_batch_size` projections at a
    time, and the gradient draws them again instead of keeping them.
    """
    key = jax.random.PRNGKey(seed)
    if k * n_edges <= SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES:
        return project(_signs(key, jnp.arange(k), k, n_edges).astype(dtype))

    @jax.checkpoint
    def project_one(index):
        return project(_signs(key, index[None], k, n_edges).astype(dtype))[0]

    # A last partial batch keeps the keys of its projection indices.
    return lax.map(project_one, jnp.arange(k), batch_size=projection_batch_size)


def _spielman_projection(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    epsilon: float,
    seed: int,
    projection_batch_size: int,
) -> Array:
    """The weighted projections of the incidence matrix, with shape `(k, n)`."""
    k = _projection_size(shape[0], epsilon)

    def project(signs):
        return _project_incidence(signs, data, indices, shape, k)

    return _projections(
        k, data.shape[0], seed, data.dtype, projection_batch_size, project
    )


def _project_incidence(
    signs: Array, data: Array, indices: Array, shape: tuple[int, ...], k: int
) -> Array:
    """The projections of the weighted incidence matrix with the random `signs`
    of some of `k` projections, with shape `(m, n_edges)`, as `(m, n)` rows."""
    valid = jnp.all(indices < jnp.asarray(shape), axis=1)
    weights = _sqrt_half_weights(jnp.where(valid, data, 0))
    weighted = signs * (weights * _projection_scale(k, data.dtype))
    projection = jnp.zeros((signs.shape[0], shape[0]), dtype=data.dtype)
    return (
        projection.at[:, indices[:, 0]]
        .add(weighted)
        .at[:, indices[:, 1]]
        .add(-weighted)
    )


def _signs(key: Array, projections: Array, k: int, n_edges: int) -> Array:
    """The random signs of the stored edges for the projection indices
    `projections`, each drawn from `key` and its index, and 0 from index `k` on.
    """

    def draw(index):
        return _rademacher(jax.random.fold_in(key, index), (n_edges,))

    return jnp.where(projections[:, None] < k, jax.vmap(draw)(projections), 0)


def _rademacher(key: Array, shape: tuple[int, ...]) -> Array:
    """Random signs, 1 or -1 with equal probability, as `int32` of `shape`.

    They compare float32 uniforms with one half, whether or not 64-bit types
    are enabled, so that a key draws the same signs in float32 and float64.
    `jax.random.rademacher` draws uniforms of the default float type, and
    other signs with 64-bit types.
    """
    return 2 * jax.random.bernoulli(key, jnp.float32(0.5), shape).astype(jnp.int32) - 1


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
    projection_batch_size: int,
) -> Array:
    return _spielman_features_reduced_fwd(
        None,
        data,
        indices,
        shape,
        epsilon,
        seed,
        solver,
        state,
        projection_batch_size,
    )[0]


def _spielman_features_reduced_fwd(
    perturbed, data, indices, shape, epsilon, seed, solver, state, batch_size
):
    del perturbed
    system = _grounded_system(data, indices, shape, solver, state)
    projection = _spielman_projection(data, indices, shape, epsilon, seed, batch_size)
    potentials = _solve(system, projection.T, solver)
    return potentials.T, (potentials, system)


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
    state,
    batch_size,
):
    del state
    if cotangent is None or not perturbed:
        return None
    potentials, system = residuals
    # Let A be the grounded Laplacian, without the row and the column of the
    # ground (the potentials are zero there), and P the (k, n) weighted
    # projection without its ground column. Its signs are fixed, but its
    # weights depend on data. Forward: A Y = P.T, F = Y.T.
    # Here potentials = Y and cotangent = H = dloss/dF, with shape (k, n).
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
    adjoint = _solve(system, cotangent.T, solver, transpose=True)
    _, projection_pullback = jax.vjp(
        lambda weights: _spielman_projection(
            weights, indices, shape, epsilon, seed, batch_size
        ),
        data,
    )
    # The adjoint is zero at the ground, whose column of P does not affect F.
    (projection_gradient,) = projection_pullback(adjoint.T)
    return (
        _laplacian_data_pullback(adjoint, potentials, indices, shape)
        + projection_gradient
    )


@eqx.filter_jit
def _spielman_window_query(
    graph: GridGraph,
    sources: Array,
    size: int,
    method: SpielmanApproximation,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    """`ResistanceDistance.sources_to_windows_distance` with `method`."""
    A = graph.get_adjacency_matrix()
    grid = (graph.height, graph.width)
    return _spielman_windows(
        A.data, A.indices, A.shape, grid, sources, size, method, solver, state
    )


def _batches(k: int, size: int) -> Array:
    """The indices of `k` projections or probes, in batches of `size`, the last
    completed with indices from `k` on."""
    return jnp.arange(-(-k // size) * size).reshape(-1, size)


@eqx.filter_custom_vjp
def _spielman_windows(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    grid: tuple[int, int],
    sources: Array,
    size: int,
    method: SpielmanApproximation,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    """Spielman estimates of the distances from `sources` to their windows, on
    the grid of shape `grid` whose adjacency matrix holds `data` at `indices`."""
    return _spielman_windows_fwd(
        None, data, indices, shape, grid, sources, size, method, solver, state
    )[0]


def _spielman_windows_fwd(
    perturbed, data, indices, shape, grid, sources, size, method, solver, state
):
    del perturbed
    # The squared differences of the potentials of each batch of projections,
    # between a source and the pixels of its window, add up to the estimates.
    system = _grounded_system(data, indices, shape, solver, state)
    k = _projection_size(shape[0], method.epsilon)
    key = jax.random.PRNGKey(method.seed)
    half = size // 2
    rows, columns = sources // grid[1], sources % grid[1]

    def window(potentials, row, column):
        own = potentials[row, column]
        start = (row - half, column - half, 0)
        pixels = lax.dynamic_slice(potentials, start, (size, size, own.shape[0]))
        return jnp.sum((pixels - own) ** 2, axis=-1)

    def add(total, projections):
        signs = _signs(key, projections, k, data.shape[0]).astype(data.dtype)
        projection = _project_incidence(signs, data, indices, shape, k)
        potentials = _solve(system, projection.T, solver).reshape(*grid, -1)
        per_step = max(1, _WINDOW_DIFFERENCES_PER_STEP // (size**2 * signs.shape[0]))
        squares = lax.map(
            lambda source: window(potentials, *source),
            (rows, columns),
            batch_size=per_step,
        )
        return total + squares, None

    initial = jnp.zeros((sources.shape[0], size, size), data.dtype)
    batches = _batches(k, method.projection_batch_size)
    return lax.scan(add, initial, batches)[0], system


_spielman_windows.def_fwd(_spielman_windows_fwd)


@_spielman_windows.def_bwd
def _spielman_windows_bwd(
    system,
    cotangent,
    perturbed,
    data,
    indices,
    shape,
    grid,
    sources,
    size,
    method,
    solver,
    state,
):
    del state
    if cotangent is None or not perturbed:
        return None
    # The exact resistance R_p = b_p.T L^-1 b_p of a pair p of a source s and
    # a pixel w of its window, b_p = e_s - e_w, has the derivative
    # -x_p[u] (x_p[u] - x_p[v]) with respect to the stored entry a_uv of the
    # adjacency matrix, where x_p = L^-1 b_p (see `_laplacian_data_pullback`).
    # Contracted with the cotangent G of the distances, it is the pullback of
    # L^-1 K L^-1 with K = sum_p G_p b_p b_p.T. With random signs q_p,
    # r = sum_p q_p |G_p|^(1/2) b_p and r' = sum_p q_p sign(G_p) |G_p|^(1/2) b_p
    # give E[r' r.T] = K, so that the pullback of L^-1 r' and L^-1 r estimates
    # it without bias. The sign of a pair is the product of a sign per source
    # and a sign per pixel, whose products are uncorrelated between pairs.
    k = _projection_size(shape[0], method.epsilon)
    # The probes draw from another key than the projections.
    key = jax.random.fold_in(jax.random.PRNGKey(method.seed), 2**31 - 1)
    height, width = grid
    half = size // 2
    rows, columns = sources // width, sources % width
    magnitude = jnp.sqrt(jnp.abs(cotangent))
    nonzero = cotangent != 0
    signed = jnp.where(nonzero, cotangent / jnp.where(nonzero, magnitude, 1), 0)
    weights = jnp.stack([magnitude, signed], axis=-1)

    def add(gradient, probes):
        keys = jax.vmap(lambda probe: jax.random.fold_in(key, probe))(probes)
        valid = (probes < k).astype(data.dtype)
        per_source = jax.vmap(
            lambda key: _rademacher(jax.random.fold_in(key, 0), rows.shape)
        )(keys)
        per_pixel = jax.vmap(lambda key: _rademacher(jax.random.fold_in(key, 1), grid))(
            keys
        )
        per_source = (per_source * valid[:, None]).astype(data.dtype).T
        per_pixel = jnp.moveaxis(per_pixel.astype(data.dtype), 0, -1)

        def place(spread, inputs):
            # The window terms of a source, and its own term.
            row, column, weight, sign = inputs
            start = (row - half, column - half, 0, 0)
            pixels = lax.dynamic_slice(spread, start, (size, size) + spread.shape[2:])
            pixels = pixels + weight[:, :, None, :] * sign[None, None, :, None]
            signs = lax.dynamic_slice(per_pixel, start[:3], (size, size, sign.size))
            own = sign[:, None] * jnp.einsum(
                "xyb,xyc->bc", signs, weight, precision=lax.Precision.HIGHEST
            )
            return lax.dynamic_update_slice(spread, pixels, start), own

        spread = jnp.zeros((height, width, probes.size, 2), data.dtype)
        spread, own = lax.scan(place, spread, (rows, columns, weights, per_source))
        rhs = (-per_pixel[..., None] * spread).at[rows, columns].add(own)
        potentials = _solve(system, rhs.reshape(height * width, -1), solver)
        potentials = potentials.reshape(-1, probes.size, 2)
        pullback = _laplacian_data_pullback(
            potentials[..., 1], potentials[..., 0], indices, shape
        )
        return gradient + pullback, None

    batches = _batches(k, method.projection_batch_size)
    gradient = lax.scan(add, jnp.zeros_like(data), batches)[0]
    return gradient / k


def _edge_potential_deltas(
    potentials: Array, indices: Array, shape: tuple[int, ...]
) -> Array:
    deltas = _values(potentials, indices[:, 0]) - _values(potentials, indices[:, 1])
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
    # Scale the (edges, sites) differences once instead of weighting every
    # entry of the (projections, edges) signs before the contraction.
    edge_deltas = _edge_potential_deltas(potentials, indices, shape)
    scaled_weights = _sqrt_half_weights(data) * _projection_scale(k, data.dtype)
    weighted_deltas = scaled_weights[:, None] * edge_deltas

    def project(signs):
        return _matmul(signs, weighted_deltas)

    return _projections(
        k, data.shape[0], seed, data.dtype, projection_batch_size, project
    )


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
    system = _grounded_system(data, indices, shape, solver, state)
    potentials = _node_potentials(system, nodes, solver)
    features = _spielman_project_potentials(
        data, indices, shape, potentials, epsilon, seed, projection_batch_size
    )
    return features, (potentials, system)


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
    potentials, system = residuals
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
    adjoint = _solve(system, adjoint_rhs, solver, transpose=True)
    return (
        _laplacian_data_pullback(adjoint, potentials, indices, shape)
        + projection_gradient
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


def _spielman_resistance_distance_from_data(
    data: Array,
    indices: Array,
    shape: tuple[int, ...],
    epsilon: float,
    seed: int,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    projection_batch_size: int,
) -> Array:
    features = _spielman_features_reduced(
        data, indices, shape, epsilon, seed, solver, state, projection_batch_size
    )
    return _distances_from_selected_features(features, features)
