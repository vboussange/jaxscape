import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.numpy.linalg import pinv
# This runs, but construction of levels is poor
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
    def __init__(self, A, T=None):
        self.A = A
        self.T = T  # Tentative prolongator
        self.P = None  # Smoothed prolongator
        self.R = None  # Restriction operator

class MultigridPreconditioner:
    def __init__(self, levels):
        self.levels = levels
        # Build prolongation and restriction operators with smoothing
        self.setup_prolongation()

    def setup_prolongation(self):
        # Apply Jacobi prolongation smoother to the tentative prolongator
        for i in range(len(self.levels) - 1):
            fine_level = self.levels[i]
            coarse_level = self.levels[i + 1]
            # Smooth the tentative prolongator
            fine_level.P = jacobi_prolongation_smoother(
                fine_level.A, fine_level.T
            )
            # Restriction operator is the transpose of prolongation
            fine_level.R = fine_level.P.T

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

    # Tentative prolongator (simple interpolation)
    T = jnp.zeros((n, n // 2))
    for i in range(n // 2):
        T = T.at[2*i, i].set(1.0)
        T = T.at[2*i+1, i].set(1.0)

    # Create levels with tentative prolongator
    levels = []
    fine_level = Level(A=A, T=T)
    levels.append(fine_level)

    # Coarse level
    # Coarse A is computed using the tentative prolongator
    coarse_A = fine_level.T.T @ fine_level.A @ fine_level.T
    coarse_level = Level(A=coarse_A)
    levels.append(coarse_level)

    # Initialize multigrid preconditioner
    mg_preconditioner = MultigridPreconditioner(levels)

    # Right-hand side and initial guess
    b = jnp.ones(n)
    x0 = jnp.zeros_like(b)

    # Apply the preconditioner
    x = mg_preconditioner.apply(b, x0)

    # Print the solution
    print("residual: ", jnp.linalg.norm(b-A*x))          # compute norm of residual vector
