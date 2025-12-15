import jax
import jax.numpy as jnp
from jaxscape.distance import AbstractDistance
from jax.experimental.sparse import BCOO
from jax import lax, ops
import equinox as eqx


class LCPDistance(AbstractDistance):
    """
    Compute the shortest path distances from `sources` to all vertices, or
    from all to all if sources not provided. Currently relies on Bellman-Ford
    algorithm, with complexity O(V * E * S) where L is the number of sources.
    
    !!! example
        ```python
        from jaxscape import LCPDistance

        distance = LCPDistance()
        dist = distance(grid, sources=source_indices, targets=target_indices)
        ```
    """
    @eqx.filter_jit
    def nodes_to_nodes_distance(self, grid, nodes):
        A = grid.get_adjacency_matrix()
        distances = bellman_ford_multi_sources(A, nodes)
        return distances[:, nodes]

    @eqx.filter_jit
    def sources_to_targets_distance(self, grid, sources, targets):
        A = grid.get_adjacency_matrix()
        distances = bellman_ford_multi_sources(A, sources)
        return distances[:, targets]

    @eqx.filter_jit
    def all_pairs_distance(self, grid):
        A = grid.get_adjacency_matrix()
        return bellman_ford_multi_sources(A, jnp.arange(grid.nv))

@eqx.filter_jit
def floyd_warshall(A: BCOO):
    """
    Computes the shortest paths between all pairs of nodes in a graph using the Floyd-Warshall algorithm. Complexity O(V^3).
    Converts A to a dense matrix, which may lead to out of memory problems.
    """
    
    D = 1 / A.todense()  # convert proximity to cost
    n = D.shape[0]
    ks = jnp.arange(n)

    @eqx.filter_checkpoint
    def per_k_update(D, k):
        D_ik = D[:, k][:, None]  # Shape: (n, 1)
        D_kj = D[k, :][None, :]  # Shape: (1, n)
        D_ik_kj = D_ik + D_kj  # Shape: (n, n)
        D_new = jnp.minimum(D, D_ik_kj)  # Element-wise minimum
        return D_new, None

    # Sequentially apply per_k_update over k using lax.scan
    D_final, _ = jax.lax.scan(per_k_update, D, ks)
    return D_final

@eqx.filter_jit
def bellman_ford(A: BCOO, source: int):
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
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=N)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=N - 1)
    return D

bellman_ford_multi_sources = jax.vmap(bellman_ford, in_axes=(None, 0))
