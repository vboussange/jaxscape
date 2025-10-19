"""Simple test to verify CholmodSolver implementation."""

import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxscape.solvers import CholmodSolver, BCOOLinearOperator
import lineax as lx

def test_simple_cholmod():
    """Test CholmodSolver on a simple positive definite system."""
    # Create a simple SPD matrix: A = [[2, -1], [-1, 2]]
    A_dense = jnp.array([[2.0, -1.0], [-1.0, 2.0]])
    A = BCOO.fromdense(A_dense)
    
    # Right-hand side
    b = jnp.array([1.0, 1.0])
    
    # Expected solution (can be verified manually or with numpy)
    # 2x - y = 1
    # -x + 2y = 1
    # Solution: x = 1, y = 1
    expected = jnp.array([1.0, 1.0])
    
    # Solve using CholmodSolver
    solver = CholmodSolver()
    operator = BCOOLinearOperator(A)
    solution = lx.linear_solve(operator, b, solver=solver).value
    
    print(f"Solution: {solution}")
    print(f"Expected: {expected}")
    print(f"Close: {jnp.allclose(solution, expected, atol=1e-5)}")
    
    assert jnp.allclose(solution, expected, atol=1e-5), f"Solution {solution} != {expected}"
    print("✓ Simple CHOLMOD test passed!")


def test_batched_cholmod():
    """Test CholmodSolver with multiple right-hand sides."""
    # Create a simple SPD matrix
    A_dense = jnp.array([[4.0, -1.0, 0.0], 
                         [-1.0, 4.0, -1.0], 
                         [0.0, -1.0, 4.0]])
    A = BCOO.fromdense(A_dense)
    
    # Multiple right-hand sides
    B = jnp.array([[1.0, 2.0], 
                   [2.0, 3.0], 
                   [1.0, 2.0]])
    
    # Solve using CholmodSolver
    from jaxscape.solvers import batched_linear_solve
    solver = CholmodSolver()
    
    solutions = batched_linear_solve(A, B, solver)
    
    # Verify by checking A @ x = b for each column
    for i in range(B.shape[1]):
        residual = A @ solutions[:, i] - B[:, i]
        residual_norm = jnp.linalg.norm(residual)
        print(f"RHS {i}: residual norm = {residual_norm}")
        assert residual_norm < 1e-5, f"Large residual for RHS {i}: {residual_norm}"
    
    print("✓ Batched CHOLMOD test passed!")


if __name__ == "__main__":
    try:
        test_simple_cholmod()
        test_batched_cholmod()
        print("\n✅ All CHOLMOD tests passed!")
    except ImportError as e:
        print(f"⚠️  Skipping CHOLMOD tests: {e}")
        print("Install cholespy with: pip install cholespy")
