"""Reduced scaling check (default 1000x1000): setup and per-call costs, reuse-mode
training-step cost (value_and_grad over 20 RHS), and AMJax refreshed/frozen."""

import json
import sys
import time

import jax
import jax.numpy as jnp
import lineax as lx

import jaxamg
from jaxamg.preconditioners import _prepare_preconditioner_config

from bench_jaxamg_problem import (  # noqa: E402  (grid from sys.argv[1])
    A_het, A_uni, B, GRID, N_RHS, RESULTS, bcsr, cg_solve, log, residual, sites, structure, timeit,
)
from jaxscape.solvers.amjaxcgsolver import amjax_preconditioner_operator, build_amjax_solver

b0 = B[0]
V_CYCLE = {
    "solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS", "interpolator": "D2",
    "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 1.0},
    "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER", "exact_coarse_solve": 1,
}
PRE = _prepare_preconditioner_config(V_CYCLE)
PCG = {"solver": "PCG", "preconditioner": V_CYCLE, "convergence": "RELATIVE_INI",
       "tolerance": 1e-6, "max_iters": 5000, "norm": "L2"}

apply_d = jax.jit(lambda d, r: jaxamg.solve(bcsr(d), r, config=PRE)[0])
apply_r = jax.jit(lambda d, r: jaxamg.solve(bcsr(d), r, config=PRE, reuse_setup=True)[0])
t0 = time.perf_counter(); jax.block_until_ready(apply_d(A_het.data, b0))
log("jaxamg_cold_setup_incl_compile_s", time.perf_counter() - t0)
log("a_apply_default_refresh_s", timeit(apply_d, A_het.data, b0, repeats=10))
log("b_apply_reuse_s", timeit(apply_r, A_het.data, b0, repeats=10))

pcg = {f: jax.jit(lambda d, b, f=f: jaxamg.solve(bcsr(d), b, config=PCG, reuse_setup=f)) for f in (False, True)}
x, info = pcg[False](A_het.data, b0)
log("c_pcg_iters", int(info["iterations"]))
log("c_pcg_relres", residual(A_het.data, x, b0))
log("c_pcg_1rhs_default_s", timeit(lambda d, b: pcg[False](d, b)[0], A_het.data, b0, repeats=5))
log("c_pcg_1rhs_reuse_s", timeit(lambda d, b: pcg[True](d, b)[0], A_het.data, b0, repeats=5))


def loss(d, B, reuse):
    A = jaxamg.with_cache(bcsr(d), is_symmetric=True)
    xs = jax.vmap(lambda b: jaxamg.solve(A, b, config=PCG, reuse_setup=reuse)[0])(B)
    return jnp.sum(xs[jnp.arange(N_RHS), sites])


for reuse in (False, True):
    g = jax.jit(jax.value_and_grad(lambda d, B, reuse=reuse: loss(d, B, reuse)))
    if reuse:  # one resetup on the current matrix, then reuse for 2x20 solves
        step = lambda d, B: (apply_d(d, B[0]), g(d, B))
    else:
        step = g
    log(f"c_pcg_value_and_grad_{N_RHS}rhs_{'setup_once' if reuse else 'resetup_per_solve'}_s",
        timeit(step, A_het.data, B, repeats=3))

t0 = time.perf_counter()
amg = build_amjax_solver(A_het.to_bcoo(), coarse_solver="pinv")
log("amjax_host_setup_heterogeneous_s", time.perf_counter() - t0)
P = amjax_preconditioner_operator(amg, structure)
f = jax.jit(lambda d, b: cg_solve(d, b, P))
x, steps, res = f(A_het.data, b0)
log("amjax_refreshed_cg_iters", int(steps))
fb = jax.jit(jax.vmap(lambda d, b: cg_solve(d, b, P)[0], in_axes=(None, 0)))
log(f"amjax_refreshed_time_{N_RHS}rhs_s", timeit(fb, A_het.data, B, repeats=3))
P_uni = amjax_preconditioner_operator(build_amjax_solver(A_uni.to_bcoo(), coarse_solver="pinv"), structure)
f = jax.jit(lambda d, b: cg_solve(d, b, P_uni))
x, steps, res = f(A_het.data, b0)
log("amjax_frozen_cg_iters", int(steps))
log("amjax_frozen_converged", bool(res == lx.RESULTS.successful))

with open(f"results_scaling_{GRID}.json", "w") as fh:
    json.dump(RESULTS, fh, indent=2)
