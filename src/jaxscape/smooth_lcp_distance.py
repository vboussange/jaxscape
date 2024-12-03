import jax
import jax.numpy as jnp
from jaxscape.distance import AbstractDistance
from jax.experimental.sparse import BCOO
from jax import lax, ops
import equinox

class SmoothLCPDistance(AbstractDistance):
    tau: float = equinox.field(static=True)
    def __init__(self, tau: float = 1e-10):
        """
        Calculates least cost distance using Bellman Ford with a smooth minimum based on LogSumExp.
        """
        self.tau = tau
        
    def __call__(self, grid, landmarks):
            A = grid.get_adjacency_matrix()
            assert landmarks.ndim == 1, "Landmarks must be a 1D array where each element is a vertex index"
            return bellman_ford_smoothmin(A.data, A.indices, grid.nb_active, landmarks, self.tau)

def segment_logsumexp(data, segment_ids, num_segments):
    # This follows implementation of jax.numpy.logsumexp
    # check here https://github.com/jax-ml/jax/blob/f2f02eeac9462a7f33dd3f678b893c95cf7b1694/jax/_src/numpy/reductions.py#L726
    segment_maxes = ops.segment_max(data, segment_ids, num_segments)    
    segment_maxes = lax.stop_gradient(lax.select(lax.is_finite(segment_maxes), segment_maxes, lax.full_like(segment_maxes, 0)))
    data_max = segment_maxes[segment_ids]
    exp_data = jnp.exp(data - data_max)
    sumexp_data = ops.segment_sum(exp_data, segment_ids, num_segments)
    result = jnp.log(sumexp_data) + segment_maxes
    return result

def segment_smoothmin(data, segment_ids, num_segments, tau):
    data = -data / tau
    logsumexp_values = segment_logsumexp(data, segment_ids, num_segments)
    return -tau * logsumexp_values

def _bellman_ford_smoothmin(W_data, W_indices, N, source, tau):
    max_distance = lax.stop_gradient(W_data.sum()) # otherwise, the gradient throws nans
    D = jnp.full(N, max_distance)
    D = D.at[source].set(0.0)
    
    @equinox.filter_checkpoint
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        D_v_softmin = segment_smoothmin(D_u_plus_w, W_indices[:, 1], N, tau)
        D_new = jnp.minimum(D, D_v_softmin)
        return D_new, None
    
    D, _ = lax.scan(body_fun, D, None, length=N - 1)
    return D

bellman_ford_smoothmin = jax.vmap(_bellman_ford_smoothmin, in_axes=(None, None, None, 0, None))
