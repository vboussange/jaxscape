import math
from collections.abc import Callable
from functools import partial, reduce
from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax, ops
from jax.experimental.sparse import BCOO

from jaxscape.distance import (
    _WINDOW_OUT_OF_GRID,
    _windows,
    _windows_out_of_grid,
    AbstractDistance,
)
from jaxscape.graph import AbstractGraph, GridGraph


@final
class LCPDistance(AbstractDistance):
    """Compute least-cost path distances with edge cost `1/a` for conductance `a`,
    using `algorithm`.

    For `V` vertices, `E` stored edges and `S` requested sources:

    - **Bellman–Ford** (default): Relaxes sparse edges until distances stop
      changing. Each round costs O(S(V + E)), and it takes one round more than
      the most edges on a shortest path, at most V − 1. Favor it for sparse
      graphs or a small number of sources. Window queries on a `GridGraph`
      relax the grid by shifting rasters (see `sources_to_windows_distance`).
    - **Floyd–Warshall**: Computes a dense all-pairs distance matrix by considering
      each vertex as an intermediate stop. Costs O(V³) time, even when requesting
      only selected pairs. Favor it for small dense graphs when most or all pairs
      are needed.

    For a unique shortest path, the derivative of its distance with respect to
    `a` is `-1/a**2` on that path and zero elsewhere: increasing conductance
    decreases the distance. At exact ties, Bellman–Ford chooses a generalized
    gradient by splitting sensitivity equally among optimal incoming edges at
    each vertex; complete paths need not receive equal weight. Raster gradients
    follow by the chain rule through `GridGraph.fun`.

    **Attributes:**

    - `algorithm`: `"bellman-ford"` (default) or `"floyd-warshall"`.

    !!! note

        Bellman–Ford requires positive, finite conductances; zero means no edge.
        Gradients assume the graph connections stay fixed. `jax.jvp`, `jax.jacfwd`,
        and higher-order derivatives are currently unsupported.

    !!! example

        ```python
        from jaxscape import LCPDistance, GridGraph
        import jax.numpy as jnp

        grid = GridGraph(jnp.ones((5, 5)), fun=lambda x, y: (x + y) / 2)
        distance = LCPDistance()
        D = distance(grid, sources=jnp.array([0, 1]), targets=jnp.array([10, 20]))
        ```
    """

    algorithm: str = "bellman-ford"
    symmetric: ClassVar[bool] = True

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

    def sources_to_windows_distance(
        self,
        graph: GridGraph,
        sources: Array,
        size: int,
        state: Any = None,
        source_weights: Array | None = None,
    ) -> Array:
        """Return the distances from each source to the pixels of its window.

        As `AbstractDistance.sources_to_windows_distance`. With Bellman–Ford,
        a relaxation round shifts the distance rasters of all sources by each
        neighbour offset of `graph`, adds the edge costs and takes elementwise
        minima: a few fused operations, without the gathers and scatters of
        sparse edges, which are slow on GPUs. With four neighbours, one
        expression runs two rounds. The adjoint propagation of the gradient
        shifts rasters the same way. The distances are those of the sparse
        edges, exactly, and the gradients the same to rounding. Windowed
        analyses run faster this way than with sparse edges, on GPUs and on
        CPUs.

        Rounds cannot be fewer: each distance is the sum of the edge costs of
        a shortest path, rounded after each edge, which the backward pass
        relies on to find the edges of shortest paths. Sweeps along rows and
        columns as prefix scans would add costs in another order.

        The query needs concrete `graph.neighbors`; when they are traced, as
        for a `graph` passed into `jax.jit`, it relaxes the sparse edges.

        **Arguments:**

        - `graph`: Grid graph on which to compute distances.
        - `sources`: Source vertex indices with shape `(m,)`.
        - `size`: Odd number of pixels along a side of the windows.
        - `state`: Ignored.
        - `source_weights`: Raises if given: least-cost distances do not
          average sources.

        **Returns:**

        Distances with shape `(m, size, size)`.
        """
        offsets = None
        if self.algorithm == "bellman-ford" and isinstance(graph, GridGraph):
            offsets = _grid_offsets(graph)
        if offsets is None or source_weights is not None:
            # The default query, which raises for `source_weights`.
            return super().sources_to_windows_distance(
                graph, sources, size, state, source_weights
            )
        if size % 2 == 0:
            raise ValueError("`size` must be odd.")
        return _grid_windows(graph, sources, size, offsets)

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
    """Compute least-cost distances from one source.

    **Arguments:**

    - `A`: Square, unbatched scalar BCOO conductance adjacency with shape `(n, n)`.
      See `LCPDistance` for edge constraints and gradient semantics.
    - `source`: Integer vertex index in `[0, n)`.

    **Returns:**

    Distances with shape `(n,)`, zero at the source and infinity if unreachable.
    """
    return bellman_ford_multi_sources(A, jnp.asarray(source)[None])[0]


@eqx.filter_jit
def bellman_ford_multi_sources(A: BCOO, sources: Array) -> Array:
    """Compute least-cost distances from multiple sources.

    **Arguments:**

    - `A`: Square, unbatched scalar BCOO conductance adjacency with shape `(n, n)`.
      See `LCPDistance` for edge constraints and gradient semantics.
    - `sources`: Integer vertex indices with shape `(s,)`, each in `[0, n)`.

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
    n = A.shape[0]
    present = A.data > 0
    in_bounds = jnp.all((A.indices >= 0) & (A.indices < n), axis=1)
    out_of_bounds = (
        present & ~in_bounds,
        "bellman-ford requires present-edge indices within adjacency bounds.",
    )
    data, is_source = _checked_inputs(A.data, sources, n, out_of_bounds)
    # BCOO capacity padding may use out-of-range indices. Sanitize before any
    # gather or segment reduction; a zero entry never becomes a parallel edge.
    indices = jnp.where((present & in_bounds)[:, None], A.indices, 0)
    return _bellman_ford_distance(data, _EdgeList(indices, n), is_source)


def _grid_offsets(graph: GridGraph) -> tuple[tuple[int, int], ...] | None:
    """The offsets `graph.neighbors` as static integers, or `None` when traced."""
    try:
        neighbors = np.asarray(graph.neighbors)
    except jax.errors.TracerArrayConversionError:
        return None
    return tuple((int(dy), int(dx)) for dy, dx in neighbors)


def _grid_bellman_ford(
    graph: GridGraph,
    sources: Array,
    offsets: tuple[tuple[int, int], ...],
    *checks: tuple[Array, str],
) -> Array:
    """Compute least-cost distances from multiple sources on a grid graph.

    The same distances and gradients as `bellman_ford_multi_sources` on the
    adjacency of `graph`, whose edges join the pixels `offsets` apart. The
    relaxation and the adjoint propagation shift whole rasters by each offset.

    **Arguments:**

    - `graph`: Grid graph.
    - `sources`: Integer vertex indices with shape `(s,)`.
    - `offsets`: The rows of `graph.neighbors` (see `_grid_offsets`).
    - `checks`: Further masks and messages of `_checked`.

    **Returns:**

    Distances with shape `(s, height, width)`, zero at each source and infinity
    if unreachable.
    """
    permeability = graph.grid
    shape = permeability.shape
    data = []
    for offset in offsets:
        # Edges into each pixel from the pixel `offset` before it. Pixels
        # without that neighbour pair with themselves, so that `fun` sees real
        # pixels, and their edge is absent.
        inside = _shifted(jnp.ones(shape, bool), offset, False)
        tail = jnp.where(inside, _shifted(permeability, offset, 0), permeability)
        data.append(jnp.where(inside, graph.fun(tail, permeability), 0))
    data, is_source = _checked_inputs(jnp.stack(data), sources, graph.nv, *checks)
    is_source = is_source.reshape(-1, *shape)
    return _bellman_ford_distance(data, _GridEdges(offsets), is_source)


@eqx.filter_jit
def _grid_windows(
    graph: GridGraph, sources: Array, size: int, offsets: tuple[tuple[int, int], ...]
) -> Array:
    """The windows of `size` pixels of `_grid_bellman_ford` around each source.

    The check that every window lies inside `graph` joins those of the
    conductances, in one branch."""
    outside = (_windows_out_of_grid(graph, sources, size), _WINDOW_OUT_OF_GRID)
    distances = _grid_bellman_ford(graph, sources, offsets, outside)
    return _windows(distances, sources, size)


def _shifted(x: Array, offset: tuple[int, int], fill: Any) -> Array:
    """`y[..., i, j] = x[..., i - dy, j - dx]` for `offset = (dy, dx)`, and `fill`
    where `x` has no such entry."""
    dy, dx = offset
    padding = [(0, 0, 0)] * (x.ndim - 2) + [(dy, -dy, 0), (dx, -dx, 0)]
    return lax.pad(x, jnp.asarray(fill, x.dtype), padding)


def _shifter(
    x: Array, fill: Any, reach: int, sliced: bool
) -> Callable[[tuple[int, int]], Array]:
    """The function that shifts `x` as `_shifted` by offsets of at most `reach`.

    If `sliced`, it slices `x` padded once, which an optimization barrier keeps
    from being fused into each shift.
    """
    if not sliced:
        return lambda offset: _shifted(x, offset, fill)
    height, width = x.shape[-2:]
    padding = [(0, 0)] * (x.ndim - 2) + [(reach, reach)] * 2
    padded = lax.optimization_barrier(jnp.pad(x, padding, constant_values=fill))

    def shifted(offset):
        top, left = reach - offset[0], reach - offset[1]
        return padded[..., top : top + height, left : left + width]

    return shifted


def _checked(x: Any, *checks: tuple[Array, str]) -> Any:
    """`x`, which raises the message of the first check whose mask holds `True`.

    One branched check holds one host callback for all the messages.
    """
    failed = jnp.stack([jnp.any(mask) for mask, _ in checks])
    messages = [message for _, message in checks]
    return eqx.branched_error_if(x, jnp.any(failed), jnp.argmax(failed), messages)


def _checked_inputs(
    data: Array, sources: Array, n: int, *checks: tuple[Array, str]
) -> tuple[Array, Array]:
    """Check the conductances `data` of the edges, the `sources`, and `checks`.

    Returns `data` and the mask of each source among `n` vertices, with shape
    `(s, n)`.
    """
    if data.dtype not in (jnp.float32, jnp.float64):
        raise ValueError("bellman-ford requires float32 or float64 conductances.")
    if sources.ndim != 1 or not jnp.issubdtype(sources.dtype, jnp.integer):
        raise ValueError("bellman-ford requires a 1D array of integer sources.")
    # Some devices flush subnormal inputs to zero in floating-point comparisons.
    # Inspect the magnitude bits so a nonzero stored edge cannot become padding.
    bits_dtype = jnp.uint32 if data.dtype == jnp.float32 else jnp.uint64
    magnitude_bits = lax.bitcast_convert_type(data, bits_dtype) << 1
    normal_threshold = jnp.array(1, dtype=bits_dtype) << (
        jnp.finfo(data.dtype).nmant + 1
    )
    subnormal = (magnitude_bits != 0) & (magnitude_bits < normal_threshold)
    present = data > 0
    reciprocal = 1 / jnp.where(present, data, 1)
    # XLA:CPU flushes subnormal reciprocals to zero; GPUs keep them, so test
    # against the smallest normal number.
    tiny = jnp.finfo(data.dtype).tiny
    data, sources = _checked(
        (data, sources),
        (
            subnormal,
            "bellman-ford cannot represent reciprocal conductance costs when "
            "subnormal conductances may flush to zero. Rescale them or use "
            "float64.",
        ),
        (
            ~jnp.isfinite(data) | (data < 0),
            "bellman-ford requires finite positive present conductances; "
            "zero conductance denotes an absent edge.",
        ),
        (
            present & (~jnp.isfinite(reciprocal) | (reciprocal < tiny)),
            "bellman-ford requires finite positive representable reciprocal "
            "conductance costs. Rescale the conductances or use float64.",
        ),
        (
            (sources < 0) | (sources >= n),
            "bellman-ford source indices must be within adjacency bounds.",
        ),
        *checks,
    )
    return data, jnp.arange(n)[None, :] == sources[:, None]


class _EdgeList(eqx.Module):
    """The stored edges of a graph of `n` vertices.

    Vertex arrays have shape `(s, n)`, one row per source, and edge arrays
    `(s, E)`: edge `e` goes from vertex `indices[e, 0]` to `indices[e, 1]`.
    """

    indices: Array
    n: int = eqx.field(static=True)

    # Each round gathers and scatters: check for convergence after each.
    relax_rounds: ClassVar[int] = 1
    rounds_per_check: ClassVar[int] = 1

    def tails(self, x: Array) -> Array:
        return x[:, self.indices[:, 0]]

    def heads(self, x: Array) -> Array:
        return x[:, self.indices[:, 1]]

    def sum_into(self, x: Array) -> Array:
        return ops.segment_sum(x.T, self.indices[:, 1], num_segments=self.n).T

    def relax(self, distances: Array, costs: Array) -> Array:
        """One relaxation round."""
        candidates = self.tails(distances) + costs
        best = ops.segment_min(candidates.T, self.indices[:, 1], num_segments=self.n)
        return jnp.minimum(distances, best.T)

    def propagate(
        self, adjoint: Array, cotangent: Array, tight: Array, share: Array
    ) -> Array:
        """One update `lambda <- g + B lambda` of the adjoint."""
        along_tight = jnp.where(tight, self.heads(adjoint * share), 0)
        flow = ops.segment_sum(along_tight.T, self.indices[:, 0], num_segments=self.n)
        return cotangent + flow.T


class _GridEdges(eqx.Module):
    """The edges of a grid graph, between the pixels `offsets` apart.

    Vertex arrays have shape `(s, height, width)`, one raster per source, and
    edge arrays `(s, k, height, width)`: entry `[:, k, i, j]` is the edge into
    pixel `(i, j)` from pixel `(i - dy, j - dx)`, for `(dy, dx) = offsets[k]`.

    With four neighbours, a relaxation step runs two rounds as one elementwise
    expression of shifted rasters over the walks of up to two edges, which
    reads and writes every raster once, faster than one round per step. With
    eight neighbours, two rounds hold 64 walks, which save little time for a
    much longer compilation, so a step runs one round. Two adjoint updates
    per step gained nothing.

    GPUs shift rasters with padding operations, which bound-check every read.
    On CPUs, which vectorize those reads poorly, a step pads its rasters once
    and slices them, which is much faster there and slower on GPUs.
    """

    offsets: tuple[tuple[int, int], ...] = eqx.field(static=True)

    # Check for convergence every few rounds; rounds past it change nothing.
    rounds_per_check: ClassVar[int] = 8

    @property
    def relax_rounds(self) -> int:
        return 2 if len(self.offsets) <= 4 else 1

    def tails(self, x: Array) -> Array:
        return jnp.stack([_shifted(x, o, jnp.inf) for o in self.offsets], axis=1)

    def heads(self, x: Array) -> Array:
        return x[:, None]

    def sum_into(self, x: Array) -> Array:
        return reduce(jnp.add, [x[:, k] for k in range(len(self.offsets))])

    def walks(self, length: int) -> list[list[tuple[int, tuple[int, int]]]]:
        """The walks of `length` edges that visit no pixel twice.

        A walk lists its edges in order, each as its offset index `k` and the
        offset of its head from the start of the walk. A walk through a pixel
        twice holds a cycle. Without it, the walk is shorter, and its rounded
        sum of positive costs is no larger, so the rounds may leave such walks
        out.
        """
        walks = [[]]
        for _ in range(length):
            longer = []
            for walk in walks:
                visited = {(0, 0)} | {head for _, head in walk}
                row, column = walk[-1][1] if walk else (0, 0)
                for k, (dy, dx) in enumerate(self.offsets):
                    if (row + dy, column + dx) not in visited:
                        longer.append(walk + [(k, (row + dy, column + dx))])
            walks = longer
        return walks

    def relax(self, distances: Array, costs: Array) -> Array:
        """`relax_rounds` relaxation rounds.

        A round takes the minimum of each distance and of the rounded sums
        `d_u + c_uv` over its incoming edges. Rounding is monotone, so `r`
        rounds give the minimum over the walks of up to `r` edges ending at
        each pixel of their costs, added in walk order to the distance of their
        start: exactly the same distances.
        """

        def relax(distances, costs, sliced):
            reach = self.relax_rounds * self.reach
            shifted = _shifter(distances, jnp.inf, reach, sliced)
            shifted_costs = _shifter(costs, jnp.inf, reach, sliced)
            best = distances
            for length in range(1, self.relax_rounds + 1):
                for walk in self.walks(length):
                    (row, column) = end = walk[-1][1]
                    total = shifted(end)
                    for k, (dy, dx) in walk:
                        # The cost of edge k at its head, `end - head` before
                        # the end of the walk.
                        total = total + shifted_costs((row - dy, column - dx))[k]
                    best = jnp.minimum(best, total)
            return best

        return lax.platform_dependent(
            distances,
            costs,
            cpu=partial(relax, sliced=True),
            default=partial(relax, sliced=False),
        )

    def propagate(
        self, adjoint: Array, cotangent: Array, tight: Array, share: Array
    ) -> Array:
        """One update `lambda <- g + B lambda` of the adjoint."""

        def propagate(scaled, tight, sliced):
            # alpha_uv lambda_v on the edge from each pixel to its neighbour v.
            shifted = _shifter(scaled, 0, self.reach, sliced)
            shifted_tight = _shifter(tight, False, self.reach, sliced)
            flows = [
                jnp.where(shifted_tight((-dy, -dx))[:, k], shifted((-dy, -dx)), 0)
                for k, (dy, dx) in enumerate(self.offsets)
            ]
            return cotangent + reduce(jnp.add, flows)

        return lax.platform_dependent(
            adjoint * share,
            tight,
            cpu=partial(propagate, sliced=True),
            default=partial(propagate, sliced=False),
        )

    @property
    def reach(self) -> int:
        """The largest offset of a neighbour along an axis."""
        return max(max(abs(dy), abs(dx)) for dy, dx in self.offsets)


_Edges = _EdgeList | _GridEdges


def _fixed_point(
    step: Callable[[Array], Array], x: Array, rounds: int, edges: _Edges
) -> Array:
    """Apply `step`, which runs `rounds` rounds, to `x` until it stops changing,
    for at most one round per vertex."""
    n = math.prod(x.shape[1:])
    steps = max(1, edges.rounds_per_check // rounds)

    def apply(state):
        iteration, previous, _ = state
        # Traced once and unrolled: without a loop between steps.
        updated = lax.fori_loop(0, steps, lambda _, x: step(x), previous, unroll=True)
        changed = jnp.any(updated != previous)
        return iteration + rounds * steps, updated, changed

    return lax.while_loop(
        lambda state: (state[0] < n - 1) & state[2],
        apply,
        (jnp.int32(0), x, jnp.bool_(True)),
    )[1]


def _bellman_ford_solve(
    data: Array, edges: _Edges, is_source: Array
) -> tuple[Array, Array, Array]:
    """Solve to an exact fixed point and check the final tight-edge DAG.

    Returns the distances, the mask of the tight edges and the share of each
    vertex that each of its tight incoming edges takes.
    """
    present = data > 0
    costs = jnp.where(present, 1 / jnp.where(present, data, 1), jnp.inf)
    distances = jnp.where(is_source, 0, jnp.inf).astype(data.dtype)

    # Rounds past convergence change nothing.
    distances = _fixed_point(
        lambda d: edges.relax(d, costs), distances, edges.relax_rounds, edges
    )

    before, after = edges.tails(distances), edges.heads(distances)
    tight = present & jnp.isfinite(after) & (before + costs == after)
    disordered = tight & ~(before < after)
    tight = tight & ~edges.heads(is_source)
    counts = edges.sum_into(tight.astype(data.dtype))
    # Each tight edge into a vertex takes the share `1 / counts` of it.
    share = 1 / jnp.maximum(counts, 1)
    missing = jnp.isfinite(distances) & ~is_source & (counts == 0)
    # Attach checks to all outputs: a value-only call may discard the tight
    # edges, while a gradient-only call may discard distances after the
    # custom-VJP forward.
    return _checked(
        (distances, tight, share),
        (
            disordered,
            "bellman-ford requires strict representable distance ordering on "
            "every reachable tight edge. A positive cost rounded away; rescale "
            "the conductances or use float64.",
        ),
        (
            missing,
            "bellman-ford found a finite non-source distance without a tight "
            "predecessor. Rescale the conductances or use float64.",
        ),
    )


@jax.custom_vjp
def _bellman_ford_distance(data: Array, edges: _Edges, is_source: Array) -> Array:
    return _bellman_ford_solve(data, edges, is_source)[0]


def _bellman_ford_forward(data, edges, is_source):
    distances, tight, share = _bellman_ford_solve(data, edges, is_source)
    return distances, (data, edges, tight, share)


def _bellman_ford_backward(residual, cotangent):
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
    data, edges, tight, share = residual

    def propagate(adjoint):
        return edges.propagate(adjoint, cotangent, tight, share)

    # The strictly ordered tight graph is acyclic, so lambda = g + B lambda
    # reaches its fixed point within n - 1 updates, including signed g.
    adjoint = _fixed_point(propagate, cotangent, 1, edges)
    # alpha_uv lambda_v on each edge, summed over sources.
    flow = jnp.sum(jnp.where(tight, edges.heads(adjoint * share), 0), axis=0)
    # Zero-flow edges, including absent entries, have zero derivative. Mask
    # their denominator before arithmetic so unused tiny conductances stay safe.
    safe_data = jnp.where((data > 0) & (flow != 0), data, 1)
    # Preserve the division order: XLA may otherwise form a**2, which can
    # under/overflow even when flow/a/a is representable. Both operands of the
    # final division cross the barrier to prevent reassociation across it.
    scaled_flow, safe_data = lax.optimization_barrier((-flow / safe_data, safe_data))
    return scaled_flow / safe_data, None, None


_bellman_ford_distance.defvjp(_bellman_ford_forward, _bellman_ford_backward)
