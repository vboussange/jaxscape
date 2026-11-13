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

from jaxscape.distance import (
    _geometric_multigrid_state,
    _inside_windows,
    _windows,
    AbstractSolverDistance,
)
from jaxscape.graph import AbstractGraph, GridGraph
from jaxscape.solvers.geometricmultigridcgsolver import (
    _grid_width,
    GeometricMultigridCGSolver,
)
from jaxscape.solvers.operator import _batched_linear_solve_impl, _solver_state
from jaxscape.utils import _matmul, _VALUES_PER_STEP, bcoo_diag, graph_laplacian


# Building the full Rademacher projection matrix is faster for moderate sizes,
# but beyond this many sign entries the temporary allocation becomes large
# enough that the streamed scan path is the safer default.
SPIELMAN_DENSE_PROJECTION_MAX_SIGN_ENTRIES = 200_000_000

# Target columns per step of exact queries above `_VALUES_PER_STEP`. Sparse
# direct solvers reach their full throughput from a few hundred columns on.
_TARGET_CHUNK_SIZE = 1024


class ResistanceDistance(AbstractSolverDistance):
    """Resistance distance: the effective resistance between two vertices of a
    graph whose edge weights are conductances.

    Queries compute it by solving linear systems in the Laplacian of the graph
    grounded at one vertex, or from the pseudo-inverse of the Laplacian.

    **Attributes:**

    - `solver`: Lineax solver of linear systems in the grounded Laplacian, for
      `BCOO` sparse matrices, or `None` for dense linear algebra. Defaults to
      `jaxscape.solvers.GeometricMultigridCGSolver()`, conjugate gradients with a
      geometric multigrid preconditioner, which solves the grounded
      Laplacians of grid graphs, such as those of a `GridGraph`, on any
      device without an optional dependency, also in float32 and inside
      `jax.jit`. With the default, queries on other graphs raise an error:
      pass `jaxscape.solvers.CholmodSolver`, `AMJaxCGSolver` or
      `CuDSSSolver`, or `solver=None`.
      [Choosing a solver](linear_solvers.md#choosing-a-solver) gives the
      solver for each device and query. Without a solver, exact all-pairs
      and nodes-to-nodes queries compute the pseudo-inverse of the dense
      Laplacian, and other queries a dense Cholesky factorization of the
      grounded Laplacian: their memory grows with the square of the number
      of vertices, which suits small graphs. Windowed analyses with exact
      resistances, which solve a linear system for every vertex of a tile,
      can run faster on a CPU without a solver. In float32, the
      pseudo-inverse loses accuracy when conductances span orders of
      magnitude.
    - `method`: A resistance distance method. Defaults to `ExactResistance()`.
      Use `SpielmanApproximation(epsilon=..., seed=...)` for the randomized
      Spielman-Srivastava approximation.

    !!! example

        ```python
        import jax.numpy as jnp
        from jaxscape import GridGraph, ResistanceDistance, SpielmanApproximation
        from jaxscape.solvers import AMJaxCGSolver, CholmodSolver

        permeability = jnp.ones((3, 3))
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)

        # Default: GeometricMultigridCGSolver, for grid graphs
        distance = ResistanceDistance()
        dist = distance(grid)

        # Dense solves, for small graphs of any kind
        distance = ResistanceDistance(solver=None)

        # A sparse direct solver, for other graphs
        distance = ResistanceDistance(solver=CholmodSolver())

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

        # Approximate resistance distance, with the default solver
        distance = ResistanceDistance(method=SpielmanApproximation(epsilon=0.05))

        dist = distance(grid)
        ```

    !!! note "Grounding, float32 and memory"

        Except dense all-pairs and nodes-to-nodes queries, which compute the
        pseudo-inverse, queries solve linear systems in the Laplacian grounded
        at one vertex: among the
        vertices whose weighted degree is at least half the largest, the
        nearest to the middle vertex. On a grid graph of `h` rows and `w`
        columns, with rook or queen contiguity, the middle vertex is its centre
        pixel, of row `h // 2` and column `w // 2`. The ground of a window or a
        tile of a windowed analysis is thus a pixel of high permeability at or
        near its centre, where its sources lie, which keeps float32 distances
        free of cancellation. A ground on a barrier would make the float32
        system too ill-conditioned to solve.

        On large tiles, the float32 solutions of sparse direct solvers are
        inaccurate around the ground vertex. For the sensitivity of
        connectivity to permeability on large tiles, use `GeometricMultigridCGSolver`,
        the default, in float32, or a direct solver in float64.

        Exact sources-to-targets queries on `n` vertices, from `m` sources to
        `k` targets, hold the potentials of all targets when they are a few
        million values, as in small windows. Beyond, they solve the targets in
        chunks of `c`, about a thousand, and hold at most `n * (m + c)`
        potentials. Computing their gradient solves the system of each target
        once more.

    !!! warning

        The graph must be undirected for resistance distance to be well-defined.
        With a solver, it must also be connected: the grounded Laplacian of a
        disconnected graph is singular, and queries raise an error.
        `GeometricMultigridCGSolver`
        then reports a non-finite solution, or a solve that did not converge
        within `max_steps`. A grid graph is disconnected when edges of zero
        conductance cut it, as `fun` gives between pixels of zero permeability.
        Windowed analyses raise permeability to a floor, so that their graphs
        are connected. Without a solver, the pseudo-inverse gives the
        resistances within each connected component, and finite values between
        components, and sources-to-targets queries give NaN.
    """

    solver: lx.AbstractLinearSolver | None = eqx.field(
        default_factory=GeometricMultigridCGSolver
    )
    method: "AbstractResistanceMethod" = eqx.field(
        default_factory=lambda: ExactResistance()
    )
    sources_share_work: ClassVar[bool] = True
    symmetric: ClassVar[bool] = True
    averages_sources: ClassVar[bool] = True

    @eqx.filter_jit
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
        proximity $K$ convex in $R$ is then biased upward, by about
        $R^2 K''(R) / k$, which grows with `epsilon`. The returned proximity
        is corrected to second order:

        $$K(\hat R) - \frac{\hat R^2}{k} K''(\hat R)$$

        For a negative exponential, it is
        $K(\hat R) \big(1 - (\hat R K'(\hat R) / K(\hat R))^2 / k\big)$. The
        correction does not reduce the variance of the proximities. It assumes
        the variance of one pair: the mean resistances from source weights
        (see `sources_to_windows_distance`) vary less, and the correction of a
        convex proximity of them is slightly too large. Exact resistances keep
        `proximity`.

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

    def init(self, graph: AbstractGraph) -> Any:
        """Build the solver state for `graph`.

        With `GeometricMultigridCGSolver`, the default, the state of a `GridGraph`
        builds inside `jax.jit` too: the solver reads its grid from concrete
        sparse indices, which the shape and contiguity of the graph set. Other
        solvers and graphs build their state as `AbstractSolverDistance.init`
        does.

        **Arguments:**

        - `graph`: Graph whose grounded Laplacian the state is built for.

        **Returns:**

        State to pass to queries and to `reinit`, or `None` without a solver.
        """
        if isinstance(self.solver, GeometricMultigridCGSolver) and isinstance(
            graph, GridGraph
        ):
            return _geometric_multigrid_state(self, graph)
        return super().init(graph)

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        if state is None:
            state = self.init(graph)
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
        if state is None:
            state = self.init(graph)
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
        if state is None:
            state = self.init(graph)
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
            _unit_columns(sources, A.shape[0], A.data.dtype),
            targets.astype(A.indices.dtype),
            self.solver,
            state,
        )

    def sources_to_windows_distance(
        self,
        graph: GridGraph,
        sources: Array,
        size: int,
        state: Any = None,
        source_weights: Array | None = None,
    ) -> Array:
        r"""Return the distances from each source to the pixels of its window.

        As `AbstractDistance.sources_to_windows_distance`. With
        `source_weights` $u$ of a source, its distance to pixel $j$ is the
        mean of the resistances $R_{ij}$ from the pixels $i$ around it, with
        the weights $u_i$:

        $$\sum_i u_i R_{ij} = R_{uj} + \sum_i u_i R_{ui}, \qquad
        R_{uj} = (u - e_j)^T L^+ (u - e_j),$$

        where $L^+$ is the pseudo-inverse of the Laplacian, and $R_{uj}$ the
        resistance between pixel $j$ and the pixels of $u$, which a unit
        current enters with the weights $u$. The query computes $R_{uj}$ for
        the pixels of the window, and adds the within-block mean resistance
        $\sum_i u_i R_{ui}$, which is 0 for a source alone. Exact queries
        solve for the vector $u$ in place of the source, and for every pixel
        of the graph.

        With `SpielmanApproximation`, the query solves the `k` projections
        `projection_batch_size` at a time, and adds the squared differences of
        their potentials between each source, the mean of its pixels with
        the weights $u$, and the pixels of its window. It holds the potentials
        of one batch of projections and the windows.

        Its gradient estimates the gradient of the exact resistances,
        contracted with the cotangent of the estimates, from `k` random
        probes of the pairs of a source and a pixel of its window, solved
        `projection_batch_size` at a time with two right-hand sides each. Each
        probe draws a sign per source and a sign per pixel, whose product
        weighs a pair. The estimate of each stored edge is unbiased, and
        varies much less between seeds than the gradient of the estimates
        themselves. Its noise falls as `epsilon` decreases.

        **Arguments:**

        - `graph`: Grid graph on which to compute distances.
        - `sources`: Source vertex indices with shape `(m,)`.
        - `size`: Odd number of pixels along a side of the windows.
        - `state`: Optional solver state. Defaults to automatic initialization.
        - `source_weights`: Optional non-negative weights of the pixels around
          each source, centred on it, which sum to 1, with shape
          `(m, side, side)` for an odd `side` of at most `size`. Defaults to
          the source alone, `side = 1`.

        **Returns:**

        Distances with shape `(m, size, size)`.
        """
        if size % 2 == 0:
            raise ValueError("`size` must be odd.")
        if state is None:
            state = self.init(jax.tree.map(lax.stop_gradient, graph))
        if source_weights is None:
            source_weights = jnp.ones((sources.shape[0], 1, 1), graph.grid.dtype)
        return _window_query(
            graph, sources, size, self.method, self.solver, state, source_weights
        )


class AbstractResistanceMethod(eqx.Module):
    """Abstract base class for resistance distance algorithms."""


class ExactResistance(AbstractResistanceMethod):
    """Exact resistance distances, from the pseudo-inverse of the Laplacian or
    from linear systems in the grounded Laplacian."""


class SpielmanApproximation(AbstractResistanceMethod):
    """
    Randomized approximation of resistance distances by the random projections
    of Spielman and Srivastava.

    **Attributes:**

    - `epsilon`: Accuracy parameter. Smaller values use more random
      projections: `k = ceil(log(n_vertices) / epsilon**2)`. All-pairs and
      sources-to-targets queries solve `k` linear systems and hold `k`
      potentials of every vertex; nodes-to-nodes queries solve one per node.
      Window queries, those of windowed analyses, solve `k` linear systems
      and hold the potentials of `projection_batch_size` of them. Their
      gradient solves `2 * k` more, two right-hand sides for each of `k`
      random probes, `projection_batch_size` probes at a time.
    - `seed`: Random seed for the Rademacher edge projections. Each projection
      draws its signs from `seed` and its index, so that all queries on a graph
      share their projections, in float32 and float64, with and without 64-bit
      types (`jax_enable_x64`). The probes of the gradient of window queries
      draw their signs from the same seed.
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
    largest, the nearest to the middle vertex in Manhattan distance on a grid of
    `n // w` rows and `w` columns, with the lowest index among the nearest. The
    middle vertex is the pixel of row `n // w // 2` and column `w // 2`. The
    width `w` is that which `GeometricMultigridCGSolver` reads (`_grid_width`): the
    largest index difference between neighbours, or one less for queen
    contiguity. On a grid graph of `h` rows and `w` columns, the middle vertex
    is thus its centre pixel, of row `h // 2` and column `w // 2`, and the
    ground is the pixel of near-maximum permeability nearest to it. In float32,
    a ground on a barrier or in floored padding makes the system too
    ill-conditioned to solve, and a ground near the sources of windows and
    tiles, at their centre, avoids cancellation in their distances.

    The returned matrix `L + d e_g e_g^T` adds the degree `d` of the ground `g`
    to its diagonal entry, or `d = 1` for a ground without neighbours, as in a
    graph of one vertex: it keeps the sparsity pattern of the Laplacian `L`
    whatever the ground, and it is positive definite for a connected graph. Its
    solutions minus their value at `g` are those of `L` without the row and the
    column of `g`, and give the same resistances.
    """
    n = A.shape[0]
    rows, columns = A.indices[:, 0], A.indices[:, 1]
    # Padding indices, from `n` on, are dropped.
    degree = jax.ops.segment_sum(A.data, rows, num_segments=n)
    width = _grid_width(jnp.where(rows < n, jnp.abs(rows - columns), 0))
    vertices = jnp.arange(n)
    middle = n // width // 2 * width + width // 2
    offset = jnp.abs(vertices // width - middle // width) + jnp.abs(
        vertices % width - middle % width
    )
    ground = jnp.argmin(jnp.where(degree >= jnp.max(degree) / 2, offset, 2 * n))
    grounding = jnp.where(degree[ground] > 0, degree[ground], 1)
    diagonal = bcoo_diag(degree.at[ground].add(grounding), A.indices.dtype)
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
    factorization solves it faster than an LU factorization.
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


def _unit_columns(nodes: Array, n: int, dtype: Any) -> Array:
    """The unit vectors of `nodes` among `n` vertices, as columns."""
    # Duplicate nodes retain duplicate columns.
    return jax.nn.one_hot(nodes, n, dtype=dtype).T


def _node_potentials(
    system: _GroundedSystem, nodes: Array, solver: lx.AbstractLinearSolver | None
) -> Array:
    """Columns of the grounded inverse for `nodes`, zero for the ground vertex."""
    basis = _unit_columns(nodes, system.matrix.shape[0], system.matrix.dtype)
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
    `_VALUES_PER_STEP` values, otherwise equal chunks of at most
    `_TARGET_CHUNK_SIZE`. The last chunk is padded with vertex 0, whose
    values the query drops.
    """
    k = targets.shape[0]
    n_chunks = 1 if n * k <= _VALUES_PER_STEP else -(-k // _TARGET_CHUNK_SIZE)
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
    # Each source is a column u of weights over the vertices, which sum to 1:
    # the unit vector of a vertex, or a block source. With S the grounded
    # inverse, R_uj = u.S u + S_jj - 2 (S u)_j for target j. The source columns
    # of S give u.S u and (S u)_j, the target columns their diagonal S_jj.
    m = sources.shape[1]
    chunks = _target_chunks(targets, shape[0])
    if chunks.shape[0] == 1:
        # One solve for all columns, whose potentials stay for the gradient.
        basis = _unit_columns(chunks[0], shape[0], data.dtype)
        potentials = _solve(system, jnp.concatenate([sources, basis], 1), solver)
        diagonal = _own_values(potentials[:, m:], chunks[0])
        residuals = (potentials, None)
    else:
        # The target columns, chunk by chunk, keeping their diagonal only: the
        # gradient solves them again.
        potentials = _solve(system, sources, solver)

        def target_diagonal(chunk):
            chunk_potentials = _node_potentials(system, chunk, solver)
            return _own_values(chunk_potentials, chunk)

        diagonal = lax.map(target_diagonal, chunks)
        residuals = (potentials, system)
    source_potentials = potentials[:, :m]
    distances = (
        jnp.sum(sources * source_potentials, axis=0)[:, None]
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
    # As in `_exact_selected_inverse_bwd`, with the source columns in place of
    # unit vectors, the adjoint is X G, where X are the potentials of the source
    # columns X_s and target columns X_t, and G is the cotangent of their
    # products with S. For the cotangent H of the (m, k) distances,
    #   G = [[diag(H 1), -H], [-H.T, diag(H.T 1)]],
    # so X G = [X_s diag(H 1) - X_t H.T, X_t diag(H.T 1) - X_s H]. This costs
    # O(n m k) and never forms G.
    potentials, system = residuals
    m = sources.shape[1]
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


@eqx.filter_jit
def _window_query(
    graph: GridGraph,
    sources: Array,
    size: int,
    method: AbstractResistanceMethod,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
    source_weights: Array,
) -> Array:
    """`ResistanceDistance.sources_to_windows_distance`."""
    sources = _inside_windows(graph, sources, size)
    A = graph.get_adjacency_matrix()
    grid = (graph.height, graph.width)
    reach = source_weights.shape[1] // 2
    if isinstance(method, SpielmanApproximation):
        windows = _spielman_windows(
            A.data,
            A.indices,
            A.shape,
            grid,
            sources,
            source_weights,
            size,
            method,
            solver,
            state,
        )
    else:
        # The source weights over the vertices, from every vertex.
        def column(weights, source):
            start = (source // graph.width - reach, source % graph.width - reach)
            canvas = jnp.zeros(grid, weights.dtype)
            return lax.dynamic_update_slice(canvas, weights, start).ravel()

        columns = jax.vmap(column, out_axes=1)(source_weights, sources)
        targets = jnp.arange(graph.nv, dtype=A.indices.dtype)
        distances = _exact_sources_to_targets(
            A.data, A.indices, A.shape, columns, targets, solver, state
        )
        windows = _windows(distances.reshape(-1, *grid), sources, size)
    # The within-block mean resistance: the mean of the resistances R_ui to the
    # pixels i of the source weights u.
    half = size // 2
    pixels = windows[
        :, half - reach : half + reach + 1, half - reach : half + reach + 1
    ]
    within = jnp.sum(source_weights * pixels, axis=(1, 2))
    return windows + within[:, None, None]


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
    source_weights: Array,
    size: int,
    method: SpielmanApproximation,
    solver: lx.AbstractLinearSolver | None,
    state: Any,
) -> Array:
    """Spielman estimates of the resistances between the pixels of the windows
    of `sources` and their `source_weights`, on the grid of shape `grid` whose
    adjacency matrix holds `data` at `indices`."""
    return _spielman_windows_fwd(
        None,
        data,
        indices,
        shape,
        grid,
        sources,
        source_weights,
        size,
        method,
        solver,
        state,
    )[0]


def _spielman_windows_fwd(
    perturbed,
    data,
    indices,
    shape,
    grid,
    sources,
    source_weights,
    size,
    method,
    solver,
    state,
):
    del perturbed
    # The squared differences of the potentials of each batch of projections,
    # between a source and the pixels of its window, add up to the estimates.
    # The potentials of a source are their mean over its source weights.
    system = _grounded_system(data, indices, shape, solver, state)
    k = _projection_size(shape[0], method.epsilon)
    key = jax.random.PRNGKey(method.seed)
    half, reach = size // 2, source_weights.shape[1] // 2
    rows, columns = sources // grid[1], sources % grid[1]

    def window(potentials, row, column, weights):
        start = (row - half, column - half, 0)
        pixels = lax.dynamic_slice(
            potentials, start, (size, size, potentials.shape[-1])
        )
        around = pixels[
            half - reach : half + reach + 1, half - reach : half + reach + 1
        ]
        own = jnp.einsum("xy,xyk->k", weights, around, precision=lax.Precision.HIGHEST)
        return jnp.sum((pixels - own) ** 2, axis=-1)

    def add(total, projections):
        signs = _signs(key, projections, k, data.shape[0]).astype(data.dtype)
        projection = _project_incidence(signs, data, indices, shape, k)
        potentials = _solve(system, projection.T, solver).reshape(*grid, -1)
        per_step = max(1, _VALUES_PER_STEP // (size**2 * signs.shape[0]))
        squares = lax.map(
            lambda source: window(potentials, *source),
            (rows, columns, source_weights),
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
    source_weights,
    size,
    method,
    solver,
    state,
):
    del state
    if cotangent is None or not perturbed:
        return None
    # The exact resistance R_p = b_p.T L^-1 b_p of a pair p of a source with
    # source weights u and a pixel w of its window, b_p = u - e_w, has the
    # derivative -x_p[a] (x_p[a] - x_p[b]) with respect to the stored entry
    # a_ab of the adjacency matrix, where x_p = L^-1 b_p (see
    # `_laplacian_data_pullback`). Contracted with the cotangent G of the
    # distances, it is the pullback of L^-1 K L^-1 with K = sum_p G_p b_p b_p.T.
    # With random signs q_p, r = sum_p q_p |G_p|^(1/2) b_p and
    # r' = sum_p q_p sign(G_p) |G_p|^(1/2) b_p give E[r' r.T] = K, so that the
    # pullback of L^-1 r' and L^-1 r estimates it without bias. The sign of a
    # pair is the product of a sign per source and a sign per pixel, whose
    # products are uncorrelated between pairs.
    k = _projection_size(shape[0], method.epsilon)
    # The probes draw from another key than the projections.
    key = jax.random.fold_in(jax.random.PRNGKey(method.seed), 2**31 - 1)
    height, width = grid
    rows, columns = sources // width, sources % width
    magnitude = jnp.sqrt(jnp.abs(cotangent))
    nonzero = cotangent != 0
    signed = jnp.where(nonzero, cotangent / jnp.where(nonzero, magnitude, 1), 0)
    roots = jnp.stack([magnitude, signed], axis=-1)

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
        rhs = _probe_right_hand_sides(
            roots, source_weights, rows, columns, per_source, per_pixel
        )
        potentials = _solve(system, rhs.reshape(height * width, -1), solver)
        potentials = potentials.reshape(-1, probes.size, 2)
        pullback = _laplacian_data_pullback(
            potentials[..., 1], potentials[..., 0], indices, shape
        )
        return gradient + pullback, None

    batches = _batches(k, method.projection_batch_size)
    gradient = lax.scan(add, jnp.zeros_like(data), batches)[0]
    return gradient / k


def _probe_right_hand_sides(
    roots: Array,
    source_weights: Array,
    rows: Array,
    columns: Array,
    per_source: Array,
    per_pixel: Array,
) -> Array:
    """The right-hand sides of the probes of the gradient of a window query.

    Source `s`, at row `rows[s]` and column `columns[s]`, weighs the pixels
    of its window with `roots[s]`, of shape `(size, size, 2)`, and the pixels
    around it with `source_weights[s]`, of shape `(side, side)`. Each probe
    draws a sign per source, `per_source`, of shape `(m, probes)`, and a sign
    per pixel, `per_pixel`, of shape `(height, width, probes)`. A pixel takes
    minus its sign times the roots that it gets from the windows that hold
    it, each times the sign of its source, and a source adds to its pixels,
    with its source weights, its sign times the roots of its window times the
    signs of their pixels.

    The sources run in chunks of up to as many consecutive sources as probes,
    whose placed windows then hold at most as many values as the right-hand
    sides. A pass adds, for each chunk, the windows of its sources within
    `size // 2` rows of its first remaining source to a strip of rows, with
    one matrix product, and passes repeat until every window is added: one
    pass for a grid of sources, row after row, which runs faster than one
    window after another.

    **Returns:**

    The right-hand sides, with shape `(height, width, probes, 2)`.
    """
    m, size = roots.shape[:2]
    half = size // 2
    height, width, probes = per_pixel.shape
    slack = max(0, min(half, height - size))
    strip = size + slack
    count = min(probes, m)
    chunks = -(-m // count)
    padding = chunks * count - m

    def chunked(x, value=0):
        widths = [(0, padding)] + [(0, 0)] * (x.ndim - 1)
        x = jnp.pad(x, widths, constant_values=value)
        return x.reshape(chunks, count, *x.shape[1:])

    # The offsets of the pixels of the source weights from their source.
    around = jnp.arange(source_weights.shape[1]) - source_weights.shape[1] // 2

    def chunk(rhs, inputs):
        rows, columns, roots, source_weights, signs, remaining = inputs
        start = jnp.min(jnp.where(remaining, rows, height)) - half
        start = jnp.minimum(start, height - strip)
        offset = rows - half - start
        added = remaining & (offset >= 0) & (offset <= slack)

        def shifted(root, offset, column, added):
            canvas = jnp.zeros((strip, width, 2), root.dtype)
            root = jnp.where(added, root, 0)
            return lax.dynamic_update_slice(canvas, root, (offset, column - half, 0))

        offset = jnp.clip(offset, 0, slack)
        windows = jax.vmap(shifted)(roots, offset, columns, added)
        corner = (start, 0, 0, 0)
        pixels = lax.dynamic_slice(per_pixel, corner[:3], (strip, width, probes))
        precision = lax.Precision.HIGHEST
        received = jnp.einsum("jxyc,jb->xybc", windows, signs, precision=precision)
        old = lax.dynamic_slice(rhs, corner, received.shape)
        rhs = lax.dynamic_update_slice(rhs, old - pixels[..., None] * received, corner)
        own = jnp.einsum("xyb,jxyc->jbc", pixels, windows, precision=precision)
        own = source_weights[..., None, None] * (signs[:, :, None] * own)[:, None, None]
        placed = (
            rows[:, None, None] + around[:, None],
            columns[:, None, None] + around,
        )
        return rhs.at[placed].add(own), remaining & ~added

    inputs = (
        chunked(rows, half),
        chunked(columns, half),
        chunked(roots),
        chunked(source_weights),
        chunked(per_source),
    )

    def unfinished(carry):
        return jnp.any(carry[1])

    def add(carry):
        rhs, remaining = carry
        return lax.scan(chunk, rhs, (*inputs, remaining))

    rhs = jnp.zeros((height, width, probes, 2), per_pixel.dtype)
    remaining = (jnp.arange(chunks * count) < m).reshape(chunks, count)
    return lax.while_loop(unfinished, add, (rhs, remaining))[0]


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
