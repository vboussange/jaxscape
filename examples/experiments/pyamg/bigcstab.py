import pyamg
import numpy as np

# multigrid solver for 2D Poisson problem
A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
print(ml)                                           # print hierarchy information
b = np.random.rand(A.shape[0])                      # pick a random right hand side
x = ml.solve(b, tol=1e-10)                          # solve Ax=b to a tolerance of 1e-10
print("residual: ", np.linalg.norm(b-A*x))          # compute norm of residual vector


from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import jax

A_jax = BCOO.from_scipy_sparse(A)
b_jax = jnp.array(b)
jacobi = jnp.array(A.diagonal())
pc = lambda x: x * (1. / jacobi)
x_jax, info = jax.scipy.sparse.linalg.bicgstab(A_jax,
                                            b_jax,
                                            M=pc,
                                            tol=1e-10,
                                            atol=1e-10,
                                            maxiter=10000)
print("residual: ", jnp.linalg.norm(b-A*x))          # compute norm of residual vector