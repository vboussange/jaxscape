import jax
import jax.numpy as jnp
from jax import lax, ops
import networkx as nx
from jax.experimental.sparse import BCOO
import equinox

@equinox.filter_jit
def bellman_ford(W_data, W_indices, source, nb_nodes):
    D = jnp.full(nb_nodes, jnp.inf).at[source].set(0.0)
    
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=nb_nodes)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=nb_nodes - 1)
    return D

N = 100
G = nx.grid_2d_graph(N, N, create_using=nx.DiGraph)
adj = BCOO.from_scipy_sparse(nx.adjacency_matrix(G))
nb_nodes = adj.shape[0]
W_data, W_indices = adj.data.astype(jnp.float32), adj.indices
dist_to_node = bellman_ford(W_data, W_indices, 0, nb_nodes)  # Forward pass works

grad_bellman = equinox.filter_grad(lambda *args: jnp.sum(bellman_ford(*args))) 
W_data_grad = grad_bellman(W_data, W_indices, 0, nb_nodes) # This fails with: XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while ...