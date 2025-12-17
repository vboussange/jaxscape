import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array, lax, ops
from jax.experimental.sparse import BCOO

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph


class LCPDistance(AbstractDistance):
    """Compute least-cost path distances using shortest path algorithms.

    Currently supports two algorithms:

    - **Bellman-Ford** (default): Efficient for sparse graphs and few sources.
      Complexity O(V × E × S) where S is the number of sources.
    - **Floyd-Warshall**: Efficient for all-pairs on small dense graphs.
      Complexity O(V³), converts to dense matrix.

    **Parameters:**

    - `algorithm`: Algorithm choice: `"bellman-ford"` (default) or `"floyd-warshall"`.

    !!! example

        ```python
        from jaxscape import LCPDistance, GridGraph
        import jax.numpy as jnp

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
        if self.algorithm not in ["bellman-ford", "floyd-warshall"]:
            raise ValueError("`algorithm` must be 'bellman-ford' or 'floyd-warshall'")

    @eqx.filter_jit
    def nodes_to_nodes_distance(self, graph: AbstractGraph, nodes: Array) -> Array:
        A = graph.get_adjacency_matrix()
        if self.algorithm == "floyd-warshall":
            distances = floyd_warshall(A)
            return distances[nodes[:, None], nodes[None, :]]
        else:
            distances = bellman_ford_multi_sources(A, nodes)
            return distances[:, nodes]

    @eqx.filter_jit
    def sources_to_targets_distance(
        self, graph: AbstractGraph, sources: Array, targets: Array
    ) -> Array:
        A = graph.get_adjacency_matrix()
        if self.algorithm == "floyd-warshall":
            distances = floyd_warshall(A)
            return distances[sources[:, None], targets[None, :]]
        else:
            distances = bellman_ford_multi_sources(A, sources)
            return distances[:, targets]

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        A = graph.get_adjacency_matrix()
        if self.algorithm == "floyd-warshall":
            return floyd_warshall(A)
        else:
            return bellman_ford_multi_sources(A, jnp.arange(graph.nv))


@eqx.filter_jit
def floyd_warshall(A: BCOO) -> Array:
    """
    Computes the shortest paths between all pairs of nodes in a graph using the Floyd-Warshall algorithm. Complexity O(V^3).
    Converts A to a dense matrix, which may lead to out of memory problems.
    """

    D = 1 / A.todense()  # convert proximity to cost
    n = D.shape[0]
    ks = jnp.arange(n)

    @eqx.filter_checkpoint
    def per_k_update(D: Array, k: int) -> tuple[Array, None]:
        D_ik = D[:, k][:, None]  # Shape: (n, 1)
        D_kj = D[k, :][None, :]  # Shape: (1, n)
        D_ik_kj = D_ik + D_kj  # Shape: (n, n)
        D_new = jnp.minimum(D, D_ik_kj)  # Element-wise minimum
        return D_new, None

    # Sequentially apply per_k_update over k using lax.scan
    D_final, _ = jax.lax.scan(per_k_update, D, ks)
    return D_final


@eqx.filter_jit
def bellman_ford(A: BCOO, source: int) -> Array:
    """
    Computes the shortest paths from a source node to all other nodes in a graph using the Bellman-Ford algorithm.
    Should you need to compute the shortest paths from multiple source nodes, consider using `jax.vmap` to vectorize this function.
    """

    N = A.shape[0]
    D = jnp.full(N, jnp.inf, dtype=A.data.dtype)  # distance matrix
    D = D.at[source].set(0.0)

    W_indices = A.indices  # Shape: (nnz, 2)
    W_data = 1 / A.data  # Shape: (nnz,), we convert proximity to cost

    @eqx.filter_checkpoint
    def body_fun(D: Array, _) -> tuple[Array, None]:
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=N)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=N - 1)
    return D


bellman_ford_multi_sources = jax.vmap(bellman_ford, in_axes=(None, 0))
