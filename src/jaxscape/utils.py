import jax
import jax.numpy as jnp
from jax.experimental import sparse
from scipy.sparse import coo_array, csr_array
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from equinox import filter_jit
from scipy import sparse as ssp

def graph_laplacian(A):
    """
    Computes the graph Laplacian given an adjacency matrix A.
    """
    D = bcoo_diag(A.sum(axis=1).todense(), indices_dtype=A.indices.dtype)  # Degree matrix
    L = D - A  # Laplacian matrix
    return L

def mapnz(mat, f):
    """
    Apply a function to the non-zero elements of a sparse matrix.
    Parameters:
    mat (sparse.BCOO): A sparse matrix in BCOO format.
    f (callable): A function to apply to the non-zero elements of the matrix.
    Returns:
    sparse.BCOO: A new sparse matrix with the function applied to its non-zero elements.
    Example:
    >>> import jax.numpy as jnp
    >>> from jax.experimental import sparse
    >>> mat = sparse.BCOO.fromdense(jnp.array([[1, 0], [0, 2]]))
    >>> def square(x):
    ...     return x ** 2
    >>> new_mat = mapnz(mat, square)
    >>> new_mat.todense()
    DeviceArray([[1, 0],
                 [0, 4]], dtype=int32)
    """
    
    indices, data = mat.indices, mat.data
    mapped_values = jnp.where(data > 0, f(data), 0.)
    return sparse.BCOO((mapped_values, indices), shape=mat.shape)

# def dense(sp_mat):
#     if isinstance(sp_mat, sparse.BCOO):
#         return sp_mat.todense()
#     return sp_mat

def zero_copy_jax_csr_to_scipy_csr(A_jax):
    data, indices, indptr = A_jax.data, A_jax.indices, A_jax.indptr
    A_scipy = ssp.csr_matrix((np.from_dlpack(data), 
                              np.from_dlpack(indices), 
                              np.from_dlpack(indptr)), shape=A_jax.shape)
    return A_scipy

def BCOO_to_coo(A):
    assert isinstance(A, sparse.BCOO)
    ij = np.array(A.indices)
    v = np.array(A.data)
    sparse_matrix = coo_array((v, (ij[:,0], ij[:,1])), shape=A.shape)
    return sparse_matrix

def BCOO_to_csr(A):
    assert isinstance(A, sparse.BCOO)
    A_BCSR = BCSR.from_bcoo(A)
    indices = np.array(A_BCSR.indices)
    indptr = np.array(A_BCSR.indptr)
    data = np.array(A_BCSR.data)
    sparse_matrix = csr_array((data, indices, indptr), shape=A.shape)
    return sparse_matrix


def get_largest_component_label(labels):
    largest_component_label = np.bincount(labels).argmax()
    # largest_component_nodes = np.where(labels == largest_component_label)[0]
    return largest_component_label

def bcoo_diag(diagonal, indices_dtype=jnp.int32):
    """
    Create a sparse diagonal matrix in BCOO format from the given diagonal elements.

    Args:
        diagonal (array-like): A 1D array of elements to be placed on the diagonal of the matrix.

    Returns:
        BCOO: A sparse matrix in BCOO format with the given diagonal elements.

    Example:
        >>> diagonal = jnp.array([1, 2, 3])
        >>> sparse_matrix = bcoo_diag(diagonal)
        >>> print(sparse_matrix)
        BCOO(float32[3,3], nse=3)
    """
    n = len(diagonal)
    indices = jnp.column_stack([jnp.arange(n, dtype=indices_dtype), jnp.arange(n, dtype=indices_dtype)])
    sparse_matrix = BCOO((diagonal, indices), shape=(n, n))
    return sparse_matrix


@filter_jit
def bcoo_tril(mat: BCOO, k: int = 0) -> BCOO:
    """
    Return the upper-triangular part of the given 2D BCOO matrix.
    The result has zeros below the k-th diagonal.

    """
    rows = mat.indices[:, 0]
    cols = mat.indices[:, 1]
    mask = jnp.where(rows >= cols - k, 1.0, 0.0)
    new_data = mat.data * mask
    out = BCOO((new_data, mat.indices), shape=mat.shape)
    return out

@filter_jit
def bcoo_triu(mat: BCOO, k: int = 0) -> BCOO:
    """
    Return the upper-triangular part of the given 2D BCOO matrix.
    The result has zeros below the k-th diagonal.

    """
    rows = mat.indices[:, 0]
    cols = mat.indices[:, 1]
    mask = jnp.where(rows <= cols - k, 1.0, 0.0)
    new_data = mat.data * mask
    out = BCOO((new_data, mat.indices), shape=mat.shape)
    return out

def bcoo_at_set(mat, row_idx, col_idx, vals):
    update_indices = jnp.stack([row_idx, col_idx], axis=-1)
    updates_coo = BCOO((vals, update_indices), shape=mat.shape)
    mask_vals = jnp.ones_like(vals, dtype=mat.data.dtype)
    mask_coo = BCOO((mask_vals, update_indices), shape=mat.shape)
    new_mat = mat - mat * mask_coo + updates_coo
    # new_mat = new_mat.sum_duplicates()
    # no need to sum_duplicates, this is done in later operations
    return new_mat


def padding(raster, buffer_size, window_size):
    """
    Pads the given raster array to ensure its dimensions are compatible with the
    specified window size, i.e. assert (raster.shape[i] - 2 * buffer_size) %
    window_size == 0
    """
    inner_height = raster.shape[0] - 2 * buffer_size
    inner_width = raster.shape[1] - 2 * buffer_size

    pad_height = (window_size - (inner_height % window_size)) % window_size
    pad_width = (window_size - (inner_width % window_size)) % window_size


    padded_raster = jnp.pad(
        raster,
        ((0,pad_height),(0,pad_width)),
        mode='constant'
    )
    return padded_raster


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