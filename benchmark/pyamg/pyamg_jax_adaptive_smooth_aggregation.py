import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.numpy.linalg import pinv
# This is failing
# inspired from https://pyamg.readthedocs.io/en/latest/_modules/pyamg/aggregation/adaptive.html#adaptive_sa_solver
# and https://pyamg.readthedocs.io/en/latest/_modules/pyamg/aggregation/smooth.html#jacobi_prolongation_smoother
class Level:
    def __init__(self, A):
        self.A = A
        self.P = None  # Prolongation operator
        self.R = None  # Restriction operator
        self.splitting = None  # C/F splitting

class MultigridPreconditioner:
    def __init__(self, A, max_levels=10, max_coarse=10):
        # Generate multigrid hierarchy
        self.levels = []
        self.max_levels = max_levels
        self.max_coarse = max_coarse
        self.generate_hierarchy(A)
    
    def generate_hierarchy(self, A):
        # Initialize the first level
        level = Level(A=A)
        self.levels.append(level)
        
        while (len(self.levels) < self.max_levels and 
               self.levels[-1].A.shape[0] > self.max_coarse):
            # Extend the hierarchy
            bottom = self.extend_hierarchy()
            if bottom:
                break
    
    def extend_hierarchy(self):
        # Get the current finest level
        fine_level = self.levels[-1]
        A = fine_level.A
        
        # Compute strength-of-connection matrix C
        C = self.strength_of_connection(A)
        
        # Generate the C/F splitting
        splitting = self.rs_splitting(C)
        fine_level.splitting = splitting
        
        # Check if splitting failed
        num_coarse = jnp.sum(splitting == 1)
        num_fine = jnp.sum(splitting == 0)
        if num_coarse == 0 or num_fine == 0:
            return True  # Cannot coarsen further
        
        # Generate the interpolation operator
        P = self.classical_interpolation(A, C, splitting)
        fine_level.P = P
        fine_level.R = P.T  # Transpose for restriction
        
        # Form the coarse-level matrix
        A_coarse = fine_level.R @ A @ fine_level.P
        
        # Add the new coarse level
        coarse_level = Level(A=A_coarse)
        self.levels.append(coarse_level)
        
        return False
    
    @staticmethod
    def strength_of_connection(A, theta=0.25):
        """
        Classical strength-of-connection.

        Parameters
        ----------
        A : array
            System matrix.
        theta : float
            Threshold parameter.

        Returns
        -------
        C : array
            Strength-of-connection matrix.
        """
        n = A.shape[0]
        # Zero out the diagonal
        A_off_diag = A - jnp.diag(jnp.diag(A))
        # Maximum off-diagonal value per row
        row_max = jnp.max(jnp.abs(A_off_diag), axis=1)
        # Compute the strength matrix
        def strength_condition(i, j):
            return jnp.abs(A[i, j]) >= theta * row_max[i]
        C = jnp.array([[strength_condition(i, j) if i != j else False for j in range(n)] for i in range(n)], dtype=jnp.float32)
        return C
    
    @staticmethod
    def rs_splitting(C):
        """
        Simplified Ruge-Stüben C/F splitting.

        Parameters
        ----------
        C : array
            Strength-of-connection matrix.

        Returns
        -------
        splitting : array
            Array indicating coarse (1) and fine (0) points.
        """
        n = C.shape[0]
        splitting = -jnp.ones(n, dtype=jnp.int32)  # -1 indicates undecided
        
        # Initialize lists
        unassigned = jnp.arange(n)
        
        # Assign all points as F-points initially
        splitting = splitting.at[unassigned].set(0)
        
        # Compute the number of strong connections for each node
        num_strong = jnp.sum(C, axis=1)
        
        # Sort nodes based on the number of strong connections (descending)
        sorted_indices = jnp.argsort(-num_strong)
        
        # Assign coarse points
        def body_fun(i, splitting):
            idx = sorted_indices[i]
            if splitting[idx] == 0:
                splitting = splitting.at[idx].set(1)  # Set as coarse point
                # Set all strongly connected neighbors as influenced
                neighbors = jnp.where(C[idx])[0]
                for neighbor in neighbors:
                    if splitting[neighbor] == 0:
                        splitting = splitting.at[neighbor].set(0)  # Keep as fine point
            return splitting

        splitting = lax.fori_loop(0, n, body_fun, splitting)
        
        # Ensure all undecided points are set as fine points
        splitting = jnp.where(splitting == -1, 0, splitting)
        
        return splitting
    
    @staticmethod
    def classical_interpolation(A, C, splitting):
        """
        Simplified classical interpolation operator.

        Parameters
        ----------
        A : array
            System matrix.
        C : array
            Strength-of-connection matrix.
        splitting : array
            C/F splitting array.

        Returns
        -------
        P : array
            Interpolation operator.
        """
        n = A.shape[0]
        num_coarse = jnp.sum(splitting)
        # Mapping from fine index to coarse index
        coarse_indices = jnp.where(splitting == 1)[0]
        coarse_map = -jnp.ones(n, dtype=jnp.int32)
        coarse_map = coarse_map.at[coarse_indices].set(jnp.arange(num_coarse))
        
        # Build interpolation matrix P
        P = jnp.zeros((n, num_coarse))
        # For each fine point, interpolate from neighboring coarse points
        for i in range(n):
            if splitting[i] == 1:
                # Coarse point interpolates itself
                ci = coarse_map[i]
                P = P.at[i, ci].set(1.0)
            else:
                # Fine point, interpolate from neighboring coarse points
                neighbors = jnp.where(C[i])[0]
                coarse_neighbors = neighbors[splitting[neighbors] == 1]
                if len(coarse_neighbors) == 0:
                    # No coarse neighbors, set to zero
                    continue
                weights = -A[i, coarse_neighbors] / A[i, i]
                ci = coarse_map[coarse_neighbors]
                P = P.at[i, ci].set(weights)
        return P
    
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
    def smooth(A, x, b, num_iterations=3):
        omega = 2/3  # Relaxation parameter
        D_inv = 1.0 / jnp.diag(A)

        def body_fun(i, x):
            x = x + omega * D_inv * (b - A @ x)
            return x

        x = lax.fori_loop(0, num_iterations, body_fun, x)
        return x

    @staticmethod
    def coarse_solver(A, b):
        return pinv(A) @ b

# Example usage
if __name__ == "__main__":
    # Define problem size
    n = 16  # Reduced size for demonstration purposes

    # Generate a sample problem (1D Poisson matrix)
    diag = 2.0 * jnp.ones(n)
    off_diag = -1.0 * jnp.ones(n - 1)
    A = jnp.diag(diag) + jnp.diag(off_diag, k=1) + jnp.diag(off_diag, k=-1)

    # Initialize multigrid preconditioner with default parameters
    mg_preconditioner = MultigridPreconditioner(A, max_levels=4, max_coarse=2)

    # Right-hand side and initial guess
    b = jnp.ones(n)
    x0 = jnp.zeros_like(b)

    # Apply the preconditioner
    x = mg_preconditioner.apply(b, x0)

    # Print the solution
    print("Solution x:", x)
