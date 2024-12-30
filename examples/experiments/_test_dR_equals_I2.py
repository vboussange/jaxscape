# Here we test the derivation that \frac{\partial R(s, t)}{\partial c_e} = -\left( \frac{I_e}{c_e} \right)^2, where c_e is the connected of edge e

import jax
import jax.numpy as jnp
from jaxscape.gridgraph import GridGraph
from jaxscape.resistance_distance import resistance_distance
import jax.random as jr
import matplotlib.pyplot as plt
from jax.experimental.sparse import BCOO

key = jr.PRNGKey(0)
N = 10
permeability = jr.uniform(key, (N, N))
activities = jnp.ones((N, N), dtype="bool")
grid = GridGraph(activities=activities, vertex_weights=permeability)
A = grid.get_adjacency_matrix(fun=lambda x, y: (x + y)/2).sum_duplicates()

conductances, indices = A.data, A.indices

source = 0
ground = N**2 - 2
# calculating dR/dce
def loss_fun(conductances, indices):
    A = BCOO((conductances, indices), shape=(N**2, N**2)) 
    dist_mat = resistance_distance(A)
    return dist_mat[source, ground]

grad_loss_fun = jax.grad(loss_fun)
dR = grad_loss_fun(conductances, indices)



# calculating Ie
import numpy as np
activities = jnp.ones((N, N), dtype="bool")
G = A.todense()

# Laplacian matrix
L = jnp.diag(G.sum(axis=1)) - G

# Inject current at node `source` and ground node `ground`
I = jnp.zeros(N**2)

I = I.at[source].set(1.)  # Inject current at the source node
I = I.at[ground].set(-1.)  # Inject current at the source node

# Compute the pseudoinverse of the Laplacian
L_pinv = jnp.linalg.pinv(L)

# Calculate the voltage vector
v = L_pinv @ I

# Calculate the currents between nodes using Ohm's Law
currents = G * (v[:, None] - v[None, :])  # Element-wise Ohm's Law
current_vec = currents[indices[:, 0], indices[:, 1]]
I_ce = (current_vec / conductances)**2

fig, ax = plt.subplots()
ax.scatter(jnp.abs(dR), I_ce)
# ax.set_xscale("symlog")
# ax.set_yscale("symlog")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
correlation = jnp.corrcoef(jnp.abs(dR), I_ce)[0, 1]
ax.set_title(f"Correlation: {correlation:.2f}")
plt.show()
