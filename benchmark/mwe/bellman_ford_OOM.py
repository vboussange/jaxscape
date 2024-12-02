"""
Reproducing out of memory issue with Bellman Ford algorithm
"""

import jax
import jax.numpy as jnp
from jax import lax, ops
import networkx as nx
from jax.experimental.sparse import BCOO
from equinox import filter_jit, filter_grad, filter_checkpoint, filter_vmap
from jax import jacfwd

def bellman_ford(W_data, W_indices, source, nb_nodes):
    D = jnp.full(nb_nodes, jnp.inf).at[source].set(0.0)
    
    @filter_checkpoint
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=nb_nodes)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=nb_nodes - 1)
    return D

N = 200
G = nx.grid_2d_graph(N, N, create_using=nx.DiGraph)
adj = BCOO.from_scipy_sparse(nx.adjacency_matrix(G))
nb_nodes = adj.shape[0]

device = jax.devices("gpu")[0]
W_data, W_indices = jax.device_put(adj.data.astype(jnp.float32), device), jax.device_put(adj.indices, device)

jit_bellman_ford = filter_jit(bellman_ford)
dist_to_node = jit_bellman_ford(W_data, W_indices, 0, nb_nodes)  # Forward pass works

def loss_fn(W_data, W_indices, source, nb_nodes):
    return jnp.sum(bellman_ford(W_data, W_indices, source, nb_nodes))

# backward mode
grad_loss_fn = filter_grad(filter_jit(loss_fn))
W_data_grad = grad_loss_fn(W_data, W_indices, 0, nb_nodes) # This fails for N > 200
# with: XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while ...

# vmapping grad_loss_fn
vmap_grad_loss_fn = filter_vmap(grad_loss_fn, in_axes=(None, None, 0, None))
W_data_grad = vmap_grad_loss_fn(W_data, W_indices, jnp.arange(3), nb_nodes) # This fails even with N <= 200
W_data_grad = jnp.column_stack([grad_loss_fn(W_data, W_indices, i, nb_nodes) for i in jnp.arange(10)]) # This works with N <= 200

# Forward mode
fwd_grad_bellman = filter_jit(jacfwd(loss_fn, argnums=0))
W_data_fwd_grad = fwd_grad_bellman(W_data, W_indices, 0, nb_nodes)
print(W_data_fwd_grad)  # This should print the gradient w.r.t W_data