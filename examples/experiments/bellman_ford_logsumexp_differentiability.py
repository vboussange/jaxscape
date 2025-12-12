import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax import lax, ops
import equinox
from jax import jit, grad
from jaxscape import GridGraph

def segment_logsumexp(data, segment_ids, num_segments, epsilon=1e-10):
    # Compute per-segment maximums for numerical stability
    segment_maxes = ops.segment_max(data, segment_ids, num_segments)
    # Gather the maximum for each data point
    data_max = segment_maxes[segment_ids]
    # Subtract the maximum from the data
    data_shifted = data - data_max
    # Exponentiate the shifted data
    exp_data = jnp.exp(data_shifted)
    # Compute per-segment sums of exponentials
    segment_sums = ops.segment_sum(exp_data, segment_ids, num_segments)
    # Add epsilon to prevent log(0)
    segment_sums = jnp.maximum(segment_sums, epsilon)
    # Compute the logarithm of the sums
    log_segment_sums = jnp.log(segment_sums)
    # Add back the maximums
    result = log_segment_sums + segment_maxes
    return result

def _bellman_ford_softmin(W_data, W_indices, N, source, tau):
    max_distance = 1e6  # Choose an appropriate large value
    D = jnp.full(N, max_distance)
    D = D.at[source].set(0.0)
    
    @equinox.filter_checkpoint
    def body_fun(D, _):
        D_u_plus_w = D[W_indices[:, 0]] + W_data
        scaled_D_u_plus_w = D_u_plus_w / tau
        neg_scaled_D_u_plus_w = -scaled_D_u_plus_w
        # Compute segment_logsumexp
        logsumexp_values = segment_logsumexp(neg_scaled_D_u_plus_w, W_indices[:, 1], num_segments=N)
        D_v_softmin = -tau * logsumexp_values
        # Update distances
        D_new = jnp.minimum(D, D_v_softmin)
        return D_new, None
    
    D, _ = lax.scan(body_fun, D, None, length=N - 1)
    return D

bellman_ford_softmin = jax.vmap(_bellman_ford_softmin, in_axes=(None, None, None, 0, None))

def test_bellman_ford_differentiability():
    permeability_raster = jnp.ones((2, 2))
    activities = jnp.ones(permeability_raster.shape, dtype=bool)
    grid = GridGraph(activities=activities,
                     grid=permeability_raster,
                     nb_active=permeability_raster.size)
    A = grid.get_adjacency_matrix()
    W_indices, W_data = A.indices, A.data
    
    def sum_bellman_ford(W_data):
        distances_jax = _bellman_ford_softmin(W_data, W_indices, A.shape[0], 0, tau=0.1)
        return jnp.sum(distances_jax)
    
    grad_sum_bellman_ford = jit(grad(sum_bellman_ford))
    sensitivity_bellman_ford = grad_sum_bellman_ford(W_data)
    print(sensitivity_bellman_ford)
    return sensitivity_bellman_ford

# Run the test function
sensitivity = test_bellman_ford_differentiability()

# Now we plot sensitivity
import matplotlib.pyplot as plt
from pathlib import Path

# Define a habitat permeability raster
N = 20
corridor_length = 6
corridor_width = 4
habitat_permability = jnp.zeros((N, N), dtype="float32")
habitat_permability = habitat_permability.at[:, :int((N-corridor_length)/2)].set(1)
habitat_permability = habitat_permability.at[int((N-corridor_width)/2):int((N-corridor_width)/2) + corridor_width, int((N-corridor_length)/2):int((N-corridor_length)/2) + corridor_length].set(1)
habitat_permability = habitat_permability.at[:, int((N-corridor_length)/2) + corridor_length:].set(1)

activities = habitat_permability > 0

# Distance calculation from focal pixel
focal_pixel_coord = (N-2, N-2)
habitat_quality = jnp.ones((N, N), dtype="float32") * 0.0
habitat_quality = habitat_quality.at[N-2, 1].set(1.)
habitat_quality = habitat_quality.at[N-2, N-2].set(1.)


def calculate_ech(habitat_permability, habitat_quality, activities, D):
    grid = GridGraph(activities=activities, grid=habitat_permability)
    A = grid.get_adjacency_matrix()
    W_indices, W_data = A.indices, A.data
    nb_active = A.shape[0]
    dist = bellman_ford_softmin(W_data, W_indices, nb_active, jnp.arange(nb_active), 1e0)
    # scaling
    # dist = dist / dist.max()
    K = jnp.exp(-dist / D)
    active_ij = grid.active_vertex_index_to_coord(jnp.arange(nb_active))
    q = habitat_quality[active_ij[:,0], active_ij[:,1]]

    ech = jnp.sqrt(q @ (K @ q))
    # ech = jnp.sum(dist)
    return ech

calculate_d_ech_dp = jax.grad(calculate_ech) # sensitivity to permeability

d_ech_dq = calculate_d_ech_dp(habitat_permability, habitat_quality, activities, 10e0)
plt.imshow(d_ech_dq)
plt.colorbar()
plt.show()
