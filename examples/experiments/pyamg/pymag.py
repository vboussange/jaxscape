import pyamg
import numpy as np

# multigrid solver for 2D Poisson problem
A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
print(ml)                                           # print hierarchy information
b = np.random.rand(A.shape[0])                      # pick a random right hand side
x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-10
print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector


from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.gallery import poisson
from scipy.sparse.linalg import cg
import numpy as np
# conjugate gradient method with AMG preconditioner
A = poisson((500, 500), format='csr')          # matrix
b = np.random.rand(A.shape[0])                 # random RHS
ml = smoothed_aggregation_solver(A)            # AMG solver
M = ml.aspreconditioner(cycle='V')             # preconditioner
x = M(b)
print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector
x, info = cg(A, b, rtol=1e-5, maxiter=30, M=M)  # solve with CG
print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector


from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg as jcg
A_jax = BCOO.from_scipy_sparse(A)
M_jax = BCOO.from_scipy_sparse(ml.levels[0].A)
b_jax = jnp.ones(A.shape[0])
x_jax, info = cg(A_jax, b, rtol=1e-5, maxiter=30, M=lambda b: M_jax @ b) # solve with CG



x = M(b_jax)