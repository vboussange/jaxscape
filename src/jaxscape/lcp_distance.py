import jax
import jax.numpy as jnp
from jaxscape.distance import AbstractDistance
from jax.experimental.sparse import BCOO
from jax import lax, ops
import equinox
class LCPDistance(AbstractDistance):
        
    def __call__(self, grid, landmarks=None):
            A = grid.get_adjacency_matrix()
            if landmarks is None:
                # D = 1 / A.todense()
                # return floyd_warshall(D) # floyd warshall is not differentiable
                return bellman_ford(A, jnp.arange(grid.nb_active))
            else:
                if landmarks.ndim == 1:
                    # already vertex indices
                    return bellman_ford(A, landmarks)
                elif landmarks.ndim == 2:
                    landmark_indices = grid.coord_to_active_vertex_index(landmarks[:, 0], landmarks[:, 1])
                    return bellman_ford(A, landmark_indices)
                else:
                    raise ValueError("Invalid landmarks dimensions")
                    

def floyd_warshall(D):
    # D must be a cost matrix
    n = D.shape[0]
    ks = jnp.arange(n)
    
    @equinox.filter_checkpoint
    def per_k_update(D, k):
        D_ik = D[:, k][:, None]           # Shape: (n, 1)
        D_kj = D[k, :][None, :]           # Shape: (1, n)
        D_ik_kj = D_ik + D_kj             # Shape: (n, n)
        D_new = jnp.minimum(D, D_ik_kj)   # Element-wise minimum
        return D_new, None

    # Sequentially apply per_k_update over k using lax.scan
    D_final, _ = jax.lax.scan(per_k_update, D, ks)
    return D_final

def _bellman_ford(adj: BCOO, source: int):
    N = adj.shape[0]
    D = jnp.full(N, jnp.inf, dtype=adj.data.dtype) # distance matrix
    D = D.at[source].set(0.0)

    W_indices = adj.indices  # Shape: (nnz, 2)
    W_data = 1 / adj.data # Shape: (nnz,), we convert proximity to cost
    
    @equinox.filter_checkpoint
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_min = ops.segment_min(D_u_plus_w, W_indices[:, 1], num_segments=N)
        return jnp.minimum(D, D_v_min), None

    D, _ = lax.scan(body_fun, D, None, length=N - 1)
    return D

bellman_ford = jax.vmap(_bellman_ford, in_axes=(None, 0))
