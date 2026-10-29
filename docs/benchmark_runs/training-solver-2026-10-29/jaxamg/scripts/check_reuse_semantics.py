"""Does reuse_setup=True really keep a stale (uniform) hierarchy?

Fresh process: set up on the uniform matrix, then apply with reuse_setup=True on
the heterogeneous values, and compare with a V-cycle freshly set up on the
heterogeneous values. Also checks the true residual of the frozen-uniform CG.
"""

import sys

sys.argv = sys.argv[:1] + ["300"]
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import lineax as lx  # noqa: E402

import jaxamg  # noqa: E402
from jaxamg.preconditioners import _prepare_preconditioner_config  # noqa: E402
from bench_jaxamg_problem import A_het, A_uni, B, bcsr, cg_solve, residual, structure  # noqa: E402

V_CYCLE = {
    "solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS", "interpolator": "D2",
    "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 1.0},
    "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER", "exact_coarse_solve": 1,
}
CFG = _prepare_preconditioner_config(V_CYCLE)
apply = {
    flag: jax.jit(lambda d, r, flag=flag: jaxamg.solve(bcsr(d), r, config=CFG, reuse_setup=flag)[0])
    for flag in (False, True)
}
r = jax.random.normal(jax.random.PRNGKey(0), (A_het.shape[0],))

y_uni_fresh = apply[False](A_uni.data, r)          # cache miss: setup on uniform
y_het_on_uni = apply[True](A_het.data, r)          # het fine matrix, kept uniform hierarchy
y_het_fresh = apply[False](A_het.data, r)          # resetup on het
y_het_reuse = apply[True](A_het.data, r)           # het hierarchy kept


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


print("rel(het_on_uni_hierarchy, het_fresh) =", rel(y_het_on_uni, y_het_fresh))
print("rel(het_on_uni_hierarchy, uni_fresh) =", rel(y_het_on_uni, y_uni_fresh))
print("rel(het_reuse, het_fresh)            =", rel(y_het_reuse, y_het_fresh))

P = lambda d: lx.FunctionLinearOperator(
    lambda v: jaxamg.solve(bcsr(d), v, config=CFG, reuse_setup=True)[0], structure,
    tags=(lx.symmetric_tag, lx.positive_semidefinite_tag))
f = jax.jit(lambda d, b: cg_solve(d, b, P(d)))
jax.block_until_ready(apply[False](A_uni.data, r))  # back to uniform hierarchy
x, steps, res = f(A_het.data, B[0])
print("frozen-uniform AmgX hierarchy: CG iters", int(steps), "true relres", residual(A_het.data, x, B[0]))
jax.block_until_ready(apply[False](A_het.data, r))  # het hierarchy
x, steps, res = f(A_het.data, B[0])
print("het AmgX hierarchy:            CG iters", int(steps), "true relres", residual(A_het.data, x, B[0]))
