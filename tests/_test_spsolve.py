import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import jax.random as jr
import jax
from jaxscape.gridgraph import GridGraph
from jaxscape.utils import bcoo_diag
from jax.experimental import sparse
from jax.scipy.sparse.linalg import gmres, cg
from jax.experimental.sparse.linalg import spsolve
from jax.experimental.sparse import BCSR
import timeit
from jax import jit
import equinox as eqx

N = 20
key = jr.PRNGKey(0)  # Random seed is explicit in JAX
permeability_raster = jr.uniform(key, (N, N))  # Start with a uniform permeability
activities = jnp.ones(permeability_raster.shape, dtype=bool)
grid = GridGraph(activities, permeability_raster)
A = grid.get_adjacency_matrix()
D = bcoo_diag(A.sum(axis=1).todense())  # Degree matrix
L = D - A  # Laplacian matrix
I = bcoo_diag(jnp.ones(L.shape[0]))
Idense = I.todense()
eps = 5e-7
Lreg = L + eps * I

Lreg_bcsr = BCSR.from_bcoo(Lreg)
tol = 1e-8
b=Idense[0,:]
spsolve(Lreg_bcsr.data, Lreg_bcsr.indices, Lreg_bcsr.indptr, b, tol, reorder=0)
# does not work