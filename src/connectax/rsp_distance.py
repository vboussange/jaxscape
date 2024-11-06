import jax
import jax.numpy as jnp
from jax.scipy.linalg import inv
from jax.experimental import sparse

def fundamental_matrix(W):
    L = sparse.eye(W.shape[0], dtype=W.dtype, index_dtype=W.indices.dtype) - W
    return inv(L.todense())

def mapnz(mat, f):
    indices, data = mat.indices, mat.data
    mapped_values = f(data)
    return sparse.BCOO((mapped_values, indices), shape=mat.shape)

def dense(sp_mat):
    if isinstance(sp_mat, sparse.BCOO):
        return sp_mat.todense()
    return sp_mat

def well_adapted_movement(A):
    return mapnz(A, lambda x: -jnp.log(x))

def rsp_distance(A, theta):
    C = well_adapted_movement(A) # cost matrix with well-adapted movements
    row_sum = A.sum(1).todense()
    Prw = A / row_sum  # random walk probability
    W = Prw * jnp.exp(-theta * C.todense()) # TODO: to optimze
    Z = fundamental_matrix(W)

    C_weighted = C * W
    C̄ = Z @ (C_weighted @ Z)
    C̄ = jnp.where(Z != 0, C̄ / Z, jnp.inf)  # Set to infinity where Z is zero

    diag_C̄ = jnp.diag(C̄)
    C̄ = C̄ - diag_C̄[:, None]

    return C̄
