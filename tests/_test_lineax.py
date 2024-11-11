import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import jax.random as jr
from jaxscape.gridgraph import GridGraph
from jax.experimental import sparse
from jax.scipy.sparse.linalg import gmres, cg

# TODO: rewrite `MWE_lineax.py` benchamrks with the below problem.
key = jr.PRNGKey(0)  # Random seed is explicit in JAX
permeability_raster = jr.uniform(key, (10, 10))  # Start with a uniform permeability
activities = jnp.ones(permeability_raster.shape, dtype=bool)
grid = GridGraph(activities, permeability_raster)
A = grid.get_adjacency_matrix()
W = sparse.eye(A.shape[0]) - A

matrix_key, vector_key = jr.split(jr.PRNGKey(0))
operator = lx.MatrixLinearOperator(A)
vector = sparse.eye(A.shape[0])
# This does not work, but check `MWE_lineax.py` for a fix
solution = lx.linear_solve(operator, vector, solver=lx.CG(rtol=1e-3, atol=1e-3))

A = sparse.random_bcoo(key, (10, 10), nse=0.1)
W =  A @ A.T
