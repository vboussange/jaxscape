import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.numpy.linalg import pinv
# This runs, but very large error
# inspired from https://pyamg.readthedocs.io/en/latest/_modules/pyamg/classical/classical.html#ruge_stuben_solver
@jit
def jacobi_prolongation_smoother(S, T, omega=4.0/3.0, degree=1):
    """
    Jacobi prolongation smoother.

    Parameters
    ----------
    S : array
        Sparse NxN matrix used for smoothing, typically the system matrix A.
    T : array
        Tentative prolongator.
    omega : float
        Damping parameter.
    degree : int
        Number of smoothing passes.

    Returns
    -------
    P : array
        Smoothed prolongator.
    """
    D_inv = 1.0 / jnp.diag(S)

    def smooth_step(P, _):
        P_new = P - omega * (D_inv[:, None] * (S @ P))
        return P_new, None

    P, _ = lax.scan(smooth_step, T, None, length=degree)
    return P

class Level:
    def __init__(self, A, P=None, R=None):
        self.A = A
        self.P = P
        self.R = R

class MultigridPreconditioner:
    def __init__(self, A, max_levels=10, min_coarse_size=10):
        self.levels = self.generate_levels(A, max_levels, min_coarse_size)

    def generate_levels(self, A_fine, max_levels, min_coarse_size):
        levels = []
        A = A_fine
        for level_num in range(max_levels):
            n = A.shape[0]
            if n <= min_coarse_size:
                # Coarsest level
                levels.append(Level(A=A))
                break

            # Aggregation: group every two nodes into one aggregate
            num_aggregates = n // 2
            T = jnp.zeros((n, num_aggregates))
            for i in range(num_aggregates):
                T = T.at[2*i, i].set(1.0)
                T = T.at[2*i+1, i].set(1.0)

            # Smooth the tentative prolongator
            P = jacobi_prolongation_smoother(A, T)

            # Restriction operator
            R = P.T

            # Coarse level matrix
            A_coarse = R @ A @ P

            # Append the level
            levels.append(Level(A=A, P=P, R=R))

            # Update A for the next level
            A = A_coarse

        return levels

    def apply(self, b, x0=None):
        if x0 is None:
            x = jnp.zeros_like(b)
        else:
            x = x0
        x = self.v_cycle(0, x, b)
        return x

    def v_cycle(self, lvl, x, b):
        level = self.levels[lvl]
        A = level.A

        # Pre-smoothing
        x = self.smooth(A, x, b)

        # Compute residual
        r = b - A @ x

        if lvl == len(self.levels) - 1:
            # Coarsest level, solve directly using pseudoinverse
            coarse_x = self.coarse_solver(level.A, r)
        else:
            # Restrict residual to coarse grid
            rc = level.R @ r
            # Initial guess is zero
            xc = jnp.zeros_like(rc)
            # Recursive call to v_cycle
            xc = self.v_cycle(lvl + 1, xc, rc)
            # Prolongate and correct
            x = x + level.P @ xc

        # Post-smoothing
        x = self.smooth(A, x, b)

        return x

    @staticmethod
    @jit
    def smooth(A, x, b, num_iterations=3):
        omega = 2/3  # Relaxation parameter
        D_inv = 1.0 / jnp.diag(A)

        def body_fun(i, x):
            x = x + omega * D_inv * (b - A @ x)
            return x

        x = lax.fori_loop(0, num_iterations, body_fun, x)
        return x

    @staticmethod
    @jit
    def coarse_solver(A, b):
        return pinv(A) @ b

# Example usage
if __name__ == "__main__":
    
    # Define problem size
    n = 64

    # Generate a sample problem (1D Poisson matrix)
    diag = 2.0 * jnp.ones(n)
    off_diag = -1.0 * jnp.ones(n - 1)
    A = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

    # Initialize multigrid preconditioner
    mg_preconditioner = MultigridPreconditioner(A)

    # Right-hand side and initial guess
    b = jnp.ones(n)
    x0 = jnp.zeros_like(b)

    # Apply the preconditioner
    x = mg_preconditioner.apply(b, x0)

    # Print the solution
    print("residual: ", jnp.linalg.norm(b-A*x)) # compute norm of residual vector
