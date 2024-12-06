import jax
import jax.numpy as jnp
from jax.experimental import sparse as jsparse
from pyamg.multilevel import MultilevelSolver
import lineax as lx


class JAXMultilevelSolver():
    def __init__(self, levels):
        # Convert levels to use JAX sparse matrices
        for level in levels:
            level.A = jsparse.BCOO.from_scipy_sparse(level.A)
            if hasattr(level, 'P'):
                level.P = jsparse.BCOO.from_scipy_sparse(level.P)
            if hasattr(level, 'R'):
                level.R = jsparse.BCOO.from_scipy_sparse(level.R)
        
        self.levels = levels
        self.coarse_operator = lx.MatrixLinearOperator(levels[-1].A.todense())

    def aspreconditioner(self):
        @jax.jit
        def preconditioner(b):
            b_jax = jnp.asarray(b)
            x = self.solve(b_jax, maxiter=1, tol=1e-12)
            return x
        return preconditioner

    def solve(self, b, x0=None, tol=1e-5, maxiter=100, residuals=None):
        # Adjust this method to use JAX functions and arrays
        if x0 is None:
            x = jnp.zeros_like(b)
        else:
            x = jnp.array(x0)
        A = self.levels[0].A

        normb = jnp.linalg.norm(b)
        normb = jnp.where(normb == 0.0, 1.0, normb)

        normr = jnp.linalg.norm(b - A @ x)
        if residuals is not None:
            residuals[:] = [normr]

        it = 0
        while True:
            if len(self.levels) == 1:
                # x = lx.linear_solve(A, b)
                ValueError("levels should be greater than 1")
            else:
                x = self.__solve(x, b)

            it += 1
            normr = jnp.linalg.norm(b - A @ x)
            if residuals is not None:
                residuals.append(normr)

            if normr < tol * normb:
                return x
            if it == maxiter:
                return x

    def __solve(self, x, b):
        num_levels = len(self.levels)
        x_list = []
        b_list = [b]

        # Downward sweep
        for lvl in range(num_levels - 1):
            level = self.levels[lvl]
            A = level.A
            P = level.P
            R = level.R

            # Apply presmoother
            x = level.presmoother(A, x, b)
            # takes forever

            # Compute residual
            residual = b - A @ x

            # Restrict residual
            b = R @ residual
            b_list.append(b)

            # Initialize x on next level
            x = jnp.zeros_like(b)
            x_list.append(x)

        # Coarse grid solve
        solution = lx.linear_solve(self.coarse_operator, b, solver=lx.SVD())
        x = solution.value
        x_list.append(x)
        print(len(x_list))
        # Upward sweep
        for lvl in reversed(range(num_levels - 1)):
            level = self.levels[lvl]
            P = level.P
            A = level.A
            b = b_list[lvl]

            # Prolongate correction
            # TODO: problem here
            x = x_list[lvl] + P @ x # add got incompatible shapes for broadcasting: (10,), (30,).

            # Apply postsmoother
            x = level.postsmoother(A, x, b)
            x_list[lvl] = x

        return x_list[0]

import pyamg
import numpy as np
A = pyamg.gallery.poisson((500,500), format='csr')  # 2D Poisson problem on 500x500 grid
ml = pyamg.ruge_stuben_solver(A)                    # construct the multigrid hierarchy
b = np.random.rand(A.shape[0])                      # pick a random right hand side

P = ml.aspreconditioner()
P(b)

b_jax = jnp.array(b)
ml_jax = JAXMultilevelSolver(ml.levels)
def gauss_seidel(A, x, b, iterations=1):
    """Perform Gauss-Seidel smoothing."""
    # TODO: here we should sparsify jnp.tril, jnp.triu and jax.scipy.linalg.solve_triangular
    A_dense = A.todense()
    L = jnp.tril(A_dense)
    U = jnp.triu(A_dense, 1)
    
    for _ in range(iterations):
        x = jax.scipy.linalg.solve_triangular(L, b - U @ x, lower=True)
    return x

# Add the custom Gauss-Seidel presmoother to each level
for level in ml_jax.levels:
    level.presmoother = gauss_seidel
    level.postsmoother = gauss_seidel

P_jax = ml_jax.aspreconditioner()
# Test the preconditioner with the custom Gauss-Seidel smoother
P_jax(b_jax)