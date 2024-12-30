# TODO: tests are not yet implemented

# import pyamg
# import numpy as np
# from jax.experimental.sparse import BCOO, BCSR
# import jax.numpy as jnp
# import jax
# import lineax as lx
# from jaxscape.linear_solve import AMGSolver, SparseMatrixLinearOperator
# import timeit
# from jax import ShapeDtypeStruct
# def benchmark(method, func):
#     times = timeit.repeat(func, number=1, repeat=10)
#     _, error = func()
#     print(f"{method} solve error: {error:2e}")
#     print(f"{method} min time: {min(times)}\n")
    
# def pyamg_solve(A, b):
#     ml = pyamg.ruge_stuben_solver(A)
#     x = ml.solve(b, tol=1e-10)
#     error = jnp.linalg.norm(b-A @ x)
#     return x, error

# @jax.jit
# def lineax_amg_solve(A, b):
#     operator = SparseMatrixLinearOperator(A)
#     solver = AMGSolver()
#     x = lx.linear_solve(operator, 
#                         b, 
#                         solver=solver, 
#                         throw=False).value
#     error = jnp.linalg.norm(b-A @ x)
#     return x, error

# if __name__ == "__main__":

#     A = pyamg.gallery.poisson((100, 100), format='csr')  # 2D Poisson problem on 500x500 grid
#     b = np.random.rand(A.shape[0],)                   # pick a random right hand side

#     A_jax = BCOO.from_scipy_sparse(A)
#     b_jax = jnp.array(b)
#     assert jnp.allclose(A.diagonal(), get_diagonal(A_jax))

#     maxiter = 100

#     benchmark("pyamg_solve", lambda: jax.block_until_ready(pyamg_solve(A, B)))
#     benchmark("cg_solve", lambda: jax.block_until_ready(cg_solve(A_jax, B_jax)))
#     benchmark("lineax_gmres_solve", lambda: jax.block_until_ready(lineax_gmres_solve(A_jax, B_jax))) #TODO: to fix, fails
#     benchmark("lineax_cg_solve", lambda: jax.block_until_ready(lineax_cg_solve(A_jax, B_jax))) 
#     benchmark("lineax_auto_solve", lambda: jax.block_until_ready(lineax_auto_solve(A_jax, B_jax)))