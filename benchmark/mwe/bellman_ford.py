import jax
import jax.numpy as jnp
from jax import lax, ops
import networkx as nx
from jax.experimental.sparse import BCOO

@jax.jit
def bellman_ford(W_data, W_indices, source: int):
    N = adj.shape[0]
    D = jnp.full(N, jnp.inf)
    D = D.at[source].set(0.0)
    
    @jax.checkpoint
    def body_fun(i, D):
        D_u = D[W_indices[:, 0]]
        D_u_plus_w = D_u + W_data

        D_v_min = ops.segment_min(
            D_u_plus_w,
            W_indices[:, 1],
            num_segments=N
        )
        D = jnp.minimum(D, D_v_min)
        return D
    D = lax.fori_loop(0, N - 1, body_fun, D)
    return D

N = 500  # Grid size
G = nx.grid_2d_graph(N, N, create_using=nx.DiGraph)
adj_matrix = nx.adjacency_matrix(G)
adj = BCOO.from_scipy_sparse(adj_matrix)

W_data = adj.data.astype(jnp.float32)
W_indices = adj.indices
dist_to_node = bellman_ford(W_data, W_indices, 0) # This works beautifully

grad_bellman = jax.grad(lambda *args: jnp.sum(bellman_ford(*args))) 
W_data_grad = grad_bellman(W_data, W_indices, 0) # This fails with: XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while ...