"""
MWE of Bellman Ford parallelization in JAX
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 0
import jax
import jax.numpy as jnp
from jax import lax, ops
import networkx as nx
from jax.experimental.sparse import BCOO
from equinox import filter_jit, filter_grad, filter_checkpoint, filter_vmap
from jax import jacfwd
import matplotlib.pyplot as plt
from jaxscape import GridGraph
import timeit

def benchmark(method, func):
    times = timeit.repeat(func, number=1, repeat=10)
    print(f"{method} min time: {min(times)}\n")
    
def bellman_ford(W_data, W_indices, source, nb_nodes):
    D = jnp.full(nb_nodes, jnp.inf, dtype=W_data.dtype).at[source].set(0.0)
    
    @filter_checkpoint
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=nb_nodes)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=nb_nodes - 1)
    return D
jit_bellman_ford = filter_jit(bellman_ford)

def loss_fn(permeability, source):
    grid = GridGraph(activities=jnp.ones(permeability.shape, dtype=bool),
                    grid=permeability,
                    nb_active=permeability.size)
    A = grid.get_adjacency_matrix()
    W_indices, W_data = A.indices, A.data
    return jnp.sum(bellman_ford(W_data, W_indices, source, permeability.size))

jit_loss_fn = filter_jit(loss_fn)
grad_loss_fn = filter_grad(filter_jit(loss_fn))
vmap_grad_loss_fn = filter_vmap(grad_loss_fn, in_axes=(None, 0))

def loss_fn_scan(permeability, sources):
    grid = GridGraph(activities=jnp.ones(permeability.shape, dtype=bool),
                    grid=permeability,
                    nb_active=permeability.size)
    A = grid.get_adjacency_matrix()
    W_indices, W_data = A.indices, A.data
    ech = jnp.array(0.0, dtype=W_data.dtype)
    @filter_checkpoint
    def body_fun(ech, source):
        dist = bellman_ford(W_data, W_indices, source, permeability.size)
        return ech + dist.sum(), None

    ech, _ = lax.scan(body_fun, ech, sources)
    return ech

jit_loss_fn_scan = filter_jit(loss_fn_scan)
grad_loss_fn_scan = filter_grad(filter_jit(loss_fn_scan))


N = 100
permeability = jnp.ones((N, N), dtype=jnp.bfloat16)
sources = jnp.arange(100)
jit_loss_fn(permeability, 0)
jit_loss_fn_scan(permeability, sources)

r2 = vmap_grad_loss_fn(permeability, sources)
r3 = grad_loss_fn_scan(permeability, sources)
assert jnp.allclose(r3, r3)


benchmark("grad_loss_fn", lambda: jax.block_until_ready(jnp.column_stack([grad_loss_fn(permeability, i) for i in sources])))
benchmark("vmap_grad_loss_fn", lambda: jax.block_until_ready(vmap_grad_loss_fn(permeability, sources))) # does not work for N > ?? due to OOM
benchmark("grad_loss_fn_scan", lambda: jax.block_until_ready(grad_loss_fn_scan(permeability, sources)))
"""
For N = 50
grad_loss_fn min time: 2.5798115129582584
vmap_grad_loss_fn min time: 0.7824910848867148
grad_loss_fn_scan min time: 2.8481826891656965

For N = 100
grad_loss_fn min time: 9.154960246058181
vmap_grad_loss_fn min time: 6.63644264196045
grad_loss_fn_scan min time: 10.362271466990933

Read on here: https://github.com/jax-ml/jax/discussions/16106
to acccelerate calculations, the best would be to calculate batches of vmap_grad_loss_fn
"""