"""
TODO: testing out `jax.scipy.sparse.linalg.gmres`

https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.sparse.linalg.gmres.html

It seems like we need to play around with the hyper parameters, that may need to ba adapated w.r.t the problem size.
You could also try the gmres implemented in lineax, although it may not work when b is a matrix.
You may want to provide a conditioner for improving convergence, this is not done here.
"""

from jax.scipy.sparse.linalg import gmres
from jax.experimental import sparse
from jax.experimental.sparse import BCOO, random_bcoo, bcoo_multiply_sparse

import jax
import jax.random as jr
import jax.numpy as jnp
from jax import jit

from jaxscape.gridgraph import GridGraph
from jaxscape.rsp_distance import rsp_distance as rsp_distance_inv 
from jaxscape.utils import well_adapted_movement, mapnz

def fundamental_matrix_gmres(W):
    L = sparse.eye(W.shape[0], dtype=W.dtype, index_dtype=W.indices.dtype) - W
    Z, flag = gmres(L, 
                    jnp.identity(W.shape[0], dtype=W.dtype), 
                    solve_method = "incremental",
                    maxiter=1000, 
                    restart = 50)
    # jax.debug.print(flag) # currently not working
    return Z

# @jit
# def rsp_distance(theta, A, C):
#     row_sum = A.sum(1).todense()[:, None]
#     Prw = A / row_sum  # random walk probability # BCOO
#     W = Prw * mapnz(theta * C, lambda x: jnp.exp(-x)) # BCOO matrix
#     Z = fundamental_matrix_gmres(W) # dense matrix
#     C_weighted = C * W # multiplication of two BCOO matrices, for some reason, this operation is very costly
#     C̄ = Z @ (C_weighted @ Z) # BCOO matrix
#     C̄ = jnp.where(Z != 0, C̄ / Z, jnp.inf) # dense matrix
#     # jax.debug.print(C̄)

#     diag_C̄ = jnp.diag(C̄)
#     C̄ = C̄ - diag_C̄[None, :]

#     return C̄

@jit
def rsp_distance_gmres(theta, A, C):
    row_sum = A.sum(1).todense()[:, None]
    Prw = A / row_sum  # random walk probability # BCOO

    # Compute W without converting C to dense
    W_data = Prw.data * jnp.exp(-theta * C.data)
    W = BCOO((W_data, Prw.indices), shape=Prw.shape)

    Z = fundamental_matrix_gmres(W)  # dense matrix

    # Use bcoo_multiply_sparse for element-wise multiplication
    C_weighted_data = C.data * W.data
    C_weighted =  BCOO((C_weighted_data, C.indices), shape=C.shape)


    C̄ = Z @ (C_weighted @ Z)  # BCOO matrix
    C̄ = jnp.where(Z != 0, C̄ / Z, jnp.inf)  # dense matrix

    diag_C̄ = jnp.diag(C̄)
    C̄ = C̄ - diag_C̄[None, :]

    return C̄

key = jr.key(0)
permeability_raster = jr.uniform(key, (20, 20))
activities = jnp.ones(permeability_raster.shape, dtype=bool)

grid = GridGraph(activities=activities,
                vertex_weights = permeability_raster)

theta = jnp.array(0.1)
A = grid.get_adjacency_matrix() # BCOO matrix
C = well_adapted_movement(A) # BCOO matrix

# C̄ = rsp_distance(theta, A, C)
C̄_gmres = rsp_distance_gmres(theta, A, C)
C̄_inv = rsp_distance_inv(theta, A, C)
assert jnp.allclose(C̄_inv, C̄_gmres, rtol=1e-1) # not converging for large matrices

%timeit rsp_distance_gmres(theta, A, C) # 11.6 ms ± 70.7 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
%timeit rsp_distance_inv(theta, A, C) # 23.7 ms ± 436 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
