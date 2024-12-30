"""
Wrapping pyamg aspreconditioner with pure_callback and calling with lineax cg
This cannot be jitted!
"""

import jax
import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.scipy.sparse.linalg import cg as jcg
import numpy as np
from pyamg.aggregation import smoothed_aggregation_solver
from pyamg.gallery import poisson
import lineax as lx
# Enforce CPU usage
jax.config.update("jax_platform_name", "cpu")
from jax.experimental.sparse import BCOO

# Define operator and solve with GMRES
class AMGPreconditioner(lx.FunctionLinearOperator):
    def __init__(self, A):
        # A = A.to_scipy_sparse()
        ml = smoothed_aggregation_solver(A)
        out_shape = A.shape[0]
        out_dtype = A.dtype
        out_spec = ShapeDtypeStruct(out_shape, out_dtype) # TODO: this fails
        # 
        M = ml.aspreconditioner(cycle='V')
        super().__init__(lambda x: jax.pure_callback(M, out_spec, x), 
                         out_spec, 
                         tags=[lx.positive_semidefinite_tag])
   
if __name__ == "__main__":
    
    # Original setup
    A = poisson((500, 500), format='csr')  # matrix
    b = np.random.rand(A.shape[0])         # random RHS
    ml = smoothed_aggregation_solver(A)    # AMG solver
    M = ml.aspreconditioner(cycle='V')     # preconditioner (M is a linear operator)

    # Convert A and b to JAX arrays
    A_jax = BCOO.from_scipy_sparse(A)
    b_jax = jnp.array(b)
    preconditioner = AMGPreconditioner(A)