from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import filter_jit
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO, BCSR
from scipy import sparse as ssp
from scipy.sparse import coo_array, csr_array


def graph_laplacian(A: BCOO) -> BCOO:
    """Compute graph Laplacian L = D - A where D is the degree matrix.

    !!! example

        ```python
        from jax.experimental.sparse import BCOO
        from jaxscape.utils import graph_laplacian

        A = BCOO.fromdense(jnp.array([[0, 1], [1, 0]]))
        L = graph_laplacian(A)
        ```
    """
    D = bcoo_diag(
        A.sum(axis=1).todense(), indices_dtype=A.indices.dtype
    )  # Degree matrix
    L = D - A  # Laplacian matrix
    return L


def mapnz(mat: BCOO, f: Callable[[Array], Array]) -> BCOO:
    """Apply function to non-zero elements of sparse matrix.

    **Arguments:**

    - `mat`: Sparse matrix in BCOO format.
    - `f`: Function to apply to non-zero elements.

    **Returns:**

    Sparse matrix with function applied.

    !!! example

        ```python
        from jax.experimental.sparse import BCOO
        from jaxscape.utils import mapnz

        mat = BCOO.fromdense(jnp.array([[1, 0], [0, 2]]))
        squared = mapnz(mat, lambda x: x ** 2)
        # squared.todense() = [[1, 0], [0, 4]]
        ```
    """

    indices, data = mat.indices, mat.data
    mapped_values = jnp.where(data > 0, f(data), 0.0)
    return sparse.BCOO((mapped_values, indices), shape=mat.shape)


# def dense(sp_mat):
#     if isinstance(sp_mat, sparse.BCOO):
#         return sp_mat.todense()
#     return sp_mat


def zero_copy_jax_csr_to_scipy_csr(A_jax: BCSR) -> ssp.csr_matrix:
    data, indices, indptr = A_jax.data, A_jax.indices, A_jax.indptr
    A_scipy = ssp.csr_matrix(
        (np.from_dlpack(data), np.from_dlpack(indices), np.from_dlpack(indptr)),
        shape=A_jax.shape,
    )
    return A_scipy


def BCOO_to_coo(A: BCOO) -> coo_array:
    assert isinstance(A, sparse.BCOO)
    ij = np.array(A.indices)
    v = np.array(A.data)
    sparse_matrix = coo_array((v, (ij[:, 0], ij[:, 1])), shape=A.shape)
    return sparse_matrix


def BCOO_to_csr(A: BCOO) -> csr_array:
    assert isinstance(A, sparse.BCOO)
    A_BCSR = BCSR.from_bcoo(A)
    indices = np.array(A_BCSR.indices)
    indptr = np.array(A_BCSR.indptr)
    data = np.array(A_BCSR.data)
    sparse_matrix = csr_array((data, indices, indptr), shape=A.shape)
    return sparse_matrix


def get_largest_component_label(labels: np.ndarray) -> int:
    largest_component_label = np.bincount(labels).argmax()
    # largest_component_nodes = np.where(labels == largest_component_label)[0]
    return largest_component_label


def bcoo_diag(diagonal: Array, indices_dtype=jnp.int32) -> BCOO:
    """Create sparse diagonal matrix from 1D array.

    !!! example

        ```python
        from jaxscape.utils import bcoo_diag

        D = bcoo_diag(jnp.array([1.0, 2.0, 3.0]))
        # D.todense() = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        ```
    """
    n = len(diagonal)
    indices = jnp.column_stack(
        [jnp.arange(n, dtype=indices_dtype), jnp.arange(n, dtype=indices_dtype)]
    )
    sparse_matrix = BCOO((diagonal, indices), shape=(n, n))
    return sparse_matrix


@filter_jit
def bcoo_tril(mat: BCOO, k: int = 0) -> BCOO:
    """Extract lower-triangular part of sparse matrix (zeros above k-th diagonal)."""
    rows = mat.indices[:, 0]
    cols = mat.indices[:, 1]
    mask = jnp.where(rows >= cols - k, 1.0, 0.0)
    new_data = mat.data * mask
    out = BCOO((new_data, mat.indices), shape=mat.shape)
    return out


@filter_jit
def bcoo_triu(mat: BCOO, k: int = 0) -> BCOO:
    """Extract upper-triangular part of sparse matrix (zeros below k-th diagonal)."""
    rows = mat.indices[:, 0]
    cols = mat.indices[:, 1]
    mask = jnp.where(rows <= cols - k, 1.0, 0.0)
    new_data = mat.data * mask
    out = BCOO((new_data, mat.indices), shape=mat.shape)
    return out


def bcoo_at_set(mat: BCOO, row_idx: Array, col_idx: Array, vals: Array) -> BCOO:
    update_indices = jnp.stack([row_idx, col_idx], axis=-1)
    updates_coo = BCOO((vals, update_indices), shape=mat.shape)
    mask_vals = jnp.ones_like(vals, dtype=mat.data.dtype)
    mask_coo = BCOO((mask_vals, update_indices), shape=mat.shape)
    new_mat = mat - mat * mask_coo + updates_coo
    # new_mat = new_mat.sum_duplicates()
    # no need to sum_duplicates, this is done in later operations
    return new_mat


def padding(raster: Array, buffer_size: int, window_size: int) -> Array:
    """Pad raster to ensure dimensions are compatible with [`WindowOperation`][jaxscape.window_operation.WindowOperation].

    Ensures `(raster.shape[i] - 2 * buffer_size) % window_size == 0`.

    !!! example

        ```python
        from jaxscape.utils import padding

        raster = jnp.ones((100, 100))
        padded = padding(raster, buffer_size=10, window_size=25)
        ```
    """
    inner_height = raster.shape[0] - 2 * buffer_size
    inner_width = raster.shape[1] - 2 * buffer_size

    pad_height = (window_size - (inner_height % window_size)) % window_size
    pad_width = (window_size - (inner_width % window_size)) % window_size

    padded_raster = jnp.pad(raster, ((0, pad_height), (0, pad_width)), mode="constant")
    return padded_raster


@eqx.filter_jit
def connected_component_labels(A: BCOO):
    rows = A.indices[:, 0]
    cols = A.indices[:, 1]
    weights = A.data
    n = A.shape[0]
    edge_count = rows.shape[0]
    initial_labels = jnp.arange(n, dtype=rows.dtype)

    def relax_edge(e_idx, labels):
        u = rows[e_idx]
        v = cols[e_idx]
        w = weights[e_idx]

        def _update(current):
            lu = current[u]
            lv = current[v]
            new_label = jnp.minimum(lu, lv)
            updated = current.at[u].set(new_label)
            updated = updated.at[v].set(jnp.minimum(updated[v], new_label))
            return updated

        valid = jnp.logical_and(w != 0, u != v)
        return jax.lax.cond(valid, _update, lambda x: x, labels)

    def sweep(_, labels):
        return jax.lax.fori_loop(0, edge_count, relax_edge, labels)

    num_iters = max(n - 1, 1)
    return jax.lax.fori_loop(0, num_iters, sweep, initial_labels)


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
