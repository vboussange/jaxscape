from functools import partial
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax, ops
from jax.experimental.sparse import BCOO

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph


@final
class LCPDistance(AbstractDistance):
    """Compute least-cost path distances using shortest path algorithms.

    Supports two algorithms:

    - **Bellman-Ford** (default): Sparse relaxation with exact convergence
      stopping and a custom first-order reverse-mode derivative. At ties,
      shares derivatives equally among optimal incoming edges at each vertex.
    - **Floyd-Warshall**: Efficient for all-pairs on small dense graphs.
      Complexity O(V³), converts to dense matrix.

    **Attributes:**

    - `algorithm`: Algorithm choice: `"bellman-ford"` (default) or
      `"floyd-warshall"`.

    !!! note

        Bellman-Ford supports float32/float64 conductances, JIT and first-order
        reverse mode. Present conductances must be finite, positive and normal,
        with finite positive reciprocal costs. Zero stored entries are absent;
        positive duplicates remain distinct parallel edges. Tight edges must
        strictly increase representable distances. Forward-mode and higher-order
        differentiation are outside its support contract.

    !!! example

        ```python
        from jaxscape import LCPDistance, GridGraph
        import jax.numpy as jnp

        permeability = jnp.ones((5, 5))
        grid = GridGraph(permeability, fun=lambda x, y: (x + y) / 2)

        # Default: Bellman-Ford (efficient for sparse graphs)
        distance = LCPDistance()
        D = distance(grid, sources=jnp.array([0, 1]), targets=jnp.array([10, 20]))

        # Floyd-Warshall (efficient for small all-pairs)
        distance_fw = LCPDistance(algorithm="floyd-warshall")
        D_all = distance_fw(grid)  # All-pairs distance
        ```
    """

    algorithm: str = "bellman-ford"

    def __check_init__(self):
        if self.algorithm not in ("bellman-ford", "floyd-warshall"):
            raise ValueError("`algorithm` must be 'bellman-ford' or 'floyd-warshall'")

    def init(self, graph: AbstractGraph) -> None:
        del graph
        return None

    @eqx.filter_jit
    def nodes_to_nodes_distance(
        self, graph: AbstractGraph, nodes: Array, state: Any = None
    ) -> Array:
        del state
        A = graph.get_adjacency_matrix()
        if self.algorithm == "floyd-warshall":
            distances = floyd_warshall(A)
            return distances[nodes[:, None], nodes[None, :]]
        else:
            distances = bellman_ford_multi_sources(A, nodes)
            return distances[:, nodes]

    @eqx.filter_jit
    def sources_to_targets_distance(
        self,
        graph: AbstractGraph,
        sources: Array,
        targets: Array,
        state: Any = None,
    ) -> Array:
        del state
        A = graph.get_adjacency_matrix()
        if self.algorithm == "floyd-warshall":
            distances = floyd_warshall(A)
            return distances[sources[:, None], targets[None, :]]
        else:
            distances = bellman_ford_multi_sources(A, sources)
            return distances[:, targets]

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph, state: Any = None) -> Array:
        del state
        A = graph.get_adjacency_matrix()
        if self.algorithm == "floyd-warshall":
            return floyd_warshall(A)
        else:
            return bellman_ford_multi_sources(A, jnp.arange(graph.nv))


@eqx.filter_jit
def floyd_warshall(A: BCOO) -> Array:
    """
    Computes shortest paths between all pairs of nodes with Floyd-Warshall.
    Complexity O(V^3).
    Converts A to a dense matrix, which may lead to out of memory problems.
    """

    D = 1 / A.todense()  # convert proximity to cost
    n = D.shape[0]
    ks = jnp.arange(n)

    @eqx.filter_checkpoint
    def per_k_update(D: Array, k: Array) -> tuple[Array, None]:
        D_ik = D[:, k][:, None]  # Shape: (n, 1)
        D_kj = D[k, :][None, :]  # Shape: (1, n)
        D_ik_kj = D_ik + D_kj  # Shape: (n, n)
        D_new = jnp.minimum(D, D_ik_kj)  # Element-wise minimum
        return D_new, None

    # Sequentially apply per_k_update over k using lax.scan
    D_final, _ = jax.lax.scan(per_k_update, D, ks)
    return D_final


@eqx.filter_jit
def bellman_ford(A: BCOO, source: int | Array) -> Array:
    """Compute least-cost distances from one source with a shortest-path DAG VJP.

    **Arguments:**

    - `A`: Unbatched scalar BCOO adjacency with shape `(n, n)` and float32/float64
      data. Present conductances must be finite, positive and normal, with finite
      positive reciprocal costs. Zero entries are absent; positive duplicates are
      parallel edges. Tight edges must strictly increase representable distances.
    - `source`: Integer vertex index in `[0, n)`.

    **Returns:**

    Distances with shape `(n,)`, zero at the source and infinity if unreachable.
    """
    return bellman_ford_multi_sources(A, jnp.asarray(source)[None])[0]


@eqx.filter_jit
def bellman_ford_multi_sources(A: BCOO, sources: Array) -> Array:
    """Compute least-cost distances from multiple sources with a shortest-path DAG VJP.

    **Arguments:**

    - `A`: Unbatched scalar BCOO adjacency with shape `(n, n)` and float32/float64
      data. Present conductances must be finite, positive and normal, with finite
      positive reciprocal costs. Zero entries are absent; positive duplicates are
      parallel edges. Tight edges must strictly increase representable distances.
    - `sources`: Integer vertex indices with shape `(s,)`, each in `[0, n)`.
      Repeated sources retain their order and accumulate gradient contributions.

    **Returns:**

    Distances with shape `(s, n)`, zero at each source and infinity if unreachable.
    """
    if (
        len(A.shape) != 2
        or A.shape[0] != A.shape[1]
        or A.data.ndim != 1
        or A.indices.ndim != 2
        or A.indices.shape[1] != 2
    ):
        raise ValueError(
            "bellman-ford requires a square, unbatched scalar BCOO adjacency."
        )
    if A.data.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("bellman-ford requires float32 or float64 conductances.")
    if sources.ndim != 1 or not jnp.issubdtype(sources.dtype, jnp.integer):
        raise ValueError("bellman-ford requires a 1D array of integer sources.")

    n = A.shape[0]
    # Some devices flush subnormal inputs to zero in floating-point comparisons.
    # Inspect the magnitude bits so a nonzero stored edge cannot become padding.
    bits_dtype = jnp.uint32 if A.data.dtype == jnp.float32 else jnp.uint64
    magnitude_bits = lax.bitcast_convert_type(A.data, bits_dtype) << 1
    normal_threshold = jnp.array(1, dtype=bits_dtype) << (
        jnp.finfo(A.data.dtype).nmant + 1
    )
    subnormal = (magnitude_bits != 0) & (magnitude_bits < normal_threshold)
    data = eqx.error_if(
        A.data,
        jnp.any(subnormal),
        "bellman-ford cannot represent reciprocal conductance costs when "
        "subnormal conductances may flush to zero. Rescale them or use float64.",
    )
    data = eqx.error_if(
        data,
        jnp.any(~jnp.isfinite(data) | (data < 0)),
        "bellman-ford requires finite positive present conductances; "
        "zero conductance denotes an absent edge.",
    )
    present = data > 0
    in_bounds = jnp.all((A.indices >= 0) & (A.indices < n), axis=1)
    data = eqx.error_if(
        data,
        jnp.any(present & ~in_bounds),
        "bellman-ford requires present-edge indices within adjacency bounds.",
    )
    # BCOO capacity padding may use out-of-range indices. Sanitize before any
    # gather or segment reduction; a zero entry never becomes a parallel edge.
    indices = jnp.where((present & in_bounds)[:, None], A.indices, 0)
    sources = eqx.error_if(
        sources,
        jnp.any((sources < 0) | (sources >= n)),
        "bellman-ford source indices must be within adjacency bounds.",
    )
    return _bellman_ford_distance(data, indices, sources, n)


def _bellman_ford_solve(
    data: Array, indices: Array, sources: Array, n: int
) -> tuple[Array, Array]:
    """Solve to an exact fixed point and check the final tight-edge DAG."""
    present = data > 0
    safe_data = jnp.where(present, data, 1)
    reciprocal = 1 / safe_data
    reciprocal = eqx.error_if(
        reciprocal,
        jnp.any(present & (~jnp.isfinite(reciprocal) | (reciprocal <= 0))),
        "bellman-ford requires finite positive representable reciprocal "
        "conductance costs. Rescale the conductances or use float64.",
    )
    costs = jnp.where(present, reciprocal, jnp.inf)
    u, v = indices.T
    distances = jnp.full((sources.size, n), jnp.inf, dtype=data.dtype)
    distances = distances.at[jnp.arange(sources.size), sources].set(0)

    def relax(state):
        iteration, previous, _ = state
        candidates = previous[:, u] + costs[None, :]
        best = ops.segment_min(candidates.T, v, num_segments=n).T
        updated = jnp.minimum(previous, best)
        return iteration + 1, updated, jnp.any(updated != previous)

    _, distances, _ = lax.while_loop(
        lambda state: (state[0] < n - 1) & state[2],
        relax,
        (jnp.int32(0), distances, jnp.bool_(True)),
    )

    before, after = distances[:, u], distances[:, v]
    tight = present[None, :] & jnp.isfinite(after) & (before + costs == after)
    distances, tight = eqx.error_if(
        (distances, tight),
        jnp.any(tight & ~(before < after)),
        "bellman-ford requires strict representable distance ordering on "
        "every reachable tight edge. A positive cost rounded away; rescale "
        "the conductances or use float64.",
    )
    tight = tight & (v[None, :] != sources[:, None])
    counts = ops.segment_sum(tight.T.astype(data.dtype), v, num_segments=n).T
    shares = tight.astype(data.dtype) / jnp.maximum(counts[:, v], 1)
    missing = (
        jnp.isfinite(distances)
        & (jnp.arange(n)[None, :] != sources[:, None])
        & (counts == 0)
    )
    # Attach checks to both outputs: a value-only call may discard shares, while
    # a gradient-only call may discard distances after the custom-VJP forward.
    return eqx.error_if(
        (distances, shares),
        jnp.any(missing),
        "bellman-ford found a finite non-source distance without a tight "
        "predecessor. Rescale the conductances or use float64.",
    )


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _bellman_ford_distance(
    data: Array, indices: Array, sources: Array, n: int
) -> Array:
    return _bellman_ford_solve(data, indices, sources, n)[0]


def _bellman_ford_forward(data, indices, sources, n):
    distances, shares = _bellman_ford_solve(data, indices, sources, n)
    return distances, (data, indices, shares)


def _bellman_ford_backward(n, residual, cotangent):
    # For each source s, let T(v) be all tight incoming edges, and let
    # alpha_uv = 1 / |T(v)| on T(v), zero elsewhere. The selected linearization
    # of the hard shortest-path objective is
    #
    #   delta d_s = 0,
    #   delta d_v = sum_(u,v) alpha_uv (delta d_u + delta c_uv).
    #
    # This is the ordinary derivative on unique paths. At ties it corresponds
    # to choosing a tight incoming edge uniformly at each vertex, so complete
    # paths need not have equal probabilities. It uses only the final distances,
    # never the number of relaxation rounds used to discover a route.
    #
    # Substitution into delta L = sum_v g_v delta d_v gives the adjoint equations
    #
    #   lambda_u = g_u + sum_(u,v) alpha_uv lambda_v,
    #   dL/dc_uv = alpha_uv lambda_v.
    #
    # The tight graph is acyclic: B_uv = alpha_uv satisfies B^n = 0. Thus
    # lambda = sum_(j=0)^(n-1) B^j g, evaluated by sparse fixed-point updates
    # lambda <- g + B lambda. This holds for arbitrary finite signed g.
    # Finally c_uv = 1/a_uv gives dL/da_uv = -alpha_uv lambda_v / a_uv^2;
    # shared conductances receive the sum of these contributions over sources.
    data, indices, shares = residual
    u, v = indices.T

    def propagate(state):
        iteration, previous, _ = state
        contributions = previous[:, v] * shares
        updated = cotangent + ops.segment_sum(contributions.T, u, num_segments=n).T
        return iteration + 1, updated, jnp.any(updated != previous)

    # The strictly ordered tight graph is acyclic, so lambda = g + B lambda
    # reaches its fixed point within n - 1 updates, including signed g.
    _, adjoint, _ = lax.while_loop(
        lambda state: (state[0] < n - 1) & state[2],
        propagate,
        (jnp.int32(0), cotangent, jnp.bool_(True)),
    )
    flow = jnp.sum(adjoint[:, v] * shares, axis=0)
    # Zero-flow edges, including absent entries, have zero derivative. Mask
    # their denominator before arithmetic so unused tiny conductances stay safe.
    safe_data = jnp.where((data > 0) & (flow != 0), data, 1)
    # Preserve the division order: XLA may otherwise form a**2, which can
    # under/overflow even when flow/a/a is representable. Both operands of the
    # final division cross the barrier to prevent reassociation across it.
    scaled_flow, safe_data = lax.optimization_barrier((-flow / safe_data, safe_data))
    return scaled_flow / safe_data, None, None


_bellman_ford_distance.defvjp(_bellman_ford_forward, _bellman_ford_backward)
