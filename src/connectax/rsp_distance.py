import jax
import jax.numpy as jnp
from jax.scipy.linalg import inv
from jax.experimental import sparse
import lineax as lx # TODO: use it!

def fundamental_matrix(W):
    L = sparse.eye(W.shape[0]) - W
    return inv(L.todense())

def mapnz(mat, f):
    indices, data = mat.indices, mat.data
    mapped_values = f(data)
    return sparse.BCOO((mapped_values, indices), shape=mat.shape)

def dense(sp_mat):
    if isinstance(sp_mat, sparse.BCOO):
        return sp_mat.todense()
    return sp_mat

def rsp_distance(A, theta):
    C = mapnz(A, lambda x: -jnp.log(x))  # cost matrix with well-adapted movements
    row_sum = A.sum(1).todense()
    Prw = A / row_sum  # random walk probability
    W = Prw * jnp.exp(-theta * dense(C))
    Z = fundamental_matrix(W)

    C_weighted = dense(C) * W
    C̄ = Z @ (C_weighted @ Z)
    C̄ = jnp.where(Z != 0, C̄ / Z, jnp.inf)  # Set to infinity where Z is zero

    diag_C̄ = jnp.diag(C̄)
    C̄ = C̄ - diag_C̄[:, None]

    return C̄
