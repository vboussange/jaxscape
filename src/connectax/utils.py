import jax
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.sparse import coo_array
import numpy as np


def mapnz(mat, f):
    indices, data = mat.indices, mat.data
    mapped_values = f(data)
    return sparse.BCOO((mapped_values, indices), shape=mat.shape)

def dense(sp_mat):
    if isinstance(sp_mat, sparse.BCOO):
        return sp_mat.todense()
    return sp_mat

# cost function
def well_adapted_movement(A):
    return mapnz(A, lambda x: -jnp.log(x))


def BCOO_to_sparse(A):
    assert isinstance(A, sparse.BCOO)
    ij = np.array(A.indices)
    v = np.array(A.data)
    sparse_matrix = coo_array((v, (ij[:,0], ij[:,1])), shape=A.shape)
    return sparse_matrix


def get_largest_component_label(labels):
    largest_component_label = np.bincount(labels).argmax()
    # largest_component_nodes = np.where(labels == largest_component_label)[0]
    return largest_component_label

# def prune_matrix(bcoo, vertices):
#     indices = bcoo.indices
#     valid_indices = jnp.where(indices[:,0])
#     return BCOO()
    
# # not working
# def strongly_connected_components_tarjan(adj_matrix: BCOO):
#     n = adj_matrix.shape[0]
#     zero = jnp.int32(0)
#     one = jnp.int32(1)
    
#     # Initialization of arrays
#     index = -jnp.ones(n, dtype=jnp.int32)  # -1 indicates unvisited
#     lowlink = jnp.zeros(n, dtype=jnp.int32)
#     onstack = jnp.zeros(n, dtype=bool)
#     stack = jnp.array([], dtype=jnp.int32)
#     components = []

#     def dfs_fn(state, s):
#         index, lowlink, onstack, stack, components, count = state
#         index = index.at[s].set(count)
#         lowlink = lowlink.at[s].set(count)
#         onstack = onstack.at[s].set(True)
#         stack = jnp.append(stack, s)
#         count += one

#         def for_each_neighbor_fn(carry, u):
#             v, index, lowlink, onstack, stack, count = carry
#             if index[u] == -1:
#                 index, lowlink, onstack, stack, components, count = dfs_fn((index, lowlink, onstack, stack, components, count), u)
#                 lowlink = lowlink.at[v].set(jnp.minimum(lowlink[v], lowlink[u]))
#             elif onstack[u]:
#                 lowlink = lowlink.at[v].set(jnp.minimum(lowlink[v], index[u]))
#             return (v, index, lowlink, onstack, stack, count), None

#         v_neihb_idx = jnp.where(adj_matrix.indices[:, 0] == s)
#         v_neighbors = adj_matrix.indices[v_neihb_idx, 1]
#         _, _ = jax.lax.scan(for_each_neighbor_fn, (s, index, lowlink, onstack, stack, count), v_neighbors)

#         def pop_scc(carry, _):
#             stack, onstack, component = carry
#             w = stack[-1]
#             stack = stack[:-1]
#             onstack = onstack.at[w].set(False)
#             component = jnp.append(component, w)
#             return (stack, onstack, component), None

#         if lowlink[s] == index[s]:
#             component = jnp.array([], dtype=jnp.int32)
#             (stack, onstack, component), _ = jax.lax.scan(pop_scc, (stack, onstack, component), jnp.arange(len(stack)))
#             components.append(component)
        
#         return (index, lowlink, onstack, stack, components, count)

#     # Initial state for `fori_loop`
#     initial_state = (index, lowlink, onstack, stack, components, zero)

#     def initialize_state(i, carry):
#         index, lowlink, onstack, stack, components, count = carry
#         carry = jax.lax.cond(
#             index[i] == -1, 
#             lambda _: dfs_fn(carry, i), 
#             lambda _: carry, 
#             None
#         )
#         return carry

#     # Run loop for all nodes in the graph
#     final_state = jax.lax.fori_loop(0, n, initialize_state, initial_state)

#     # Return the components
#     return final_state[4]  # components list