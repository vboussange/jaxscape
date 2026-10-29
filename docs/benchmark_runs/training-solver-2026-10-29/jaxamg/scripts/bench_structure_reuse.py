"""Follow-up: cost of AmgX resetup with `structure_reuse_levels` (keep coarsening,
recompute Galerkin values) vs full resetup, per V-cycle application, and the
CG-iteration effect. Reuses the problem construction from bench_jaxamg.py."""

import json
import statistics
import sys
import time

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import lineax as lx
import numpy as np

import jaxamg
from jaxamg.preconditioners import _prepare_preconditioner_config

sys.argv = sys.argv[:1] + ["300"]
from bench_jaxamg_problem import A_het, A_uni, B, N, bcsr, cg_solve, structure  # noqa: E402

b0 = B[0]
BASE = {
    "solver": "AMG", "algorithm": "CLASSICAL", "selector": "PMIS", "interpolator": "D2",
    "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 1.0},
    "presweeps": 1, "postsweeps": 1, "max_iters": 1, "cycle": "V",
    "coarse_solver": "DENSE_LU_SOLVER", "exact_coarse_solve": 1,
}
out = {}


def timeit(fn, *args, repeats=20):
    jax.block_until_ready(fn(*args))
    s = []
    for _ in range(repeats):
        t0 = time.perf_counter(); jax.block_until_ready(fn(*args)); s.append(time.perf_counter() - t0)
    return statistics.median(s)


for levels in (0, -1):
    cfg = _prepare_preconditioner_config({**BASE, "structure_reuse_levels": levels})
    apply = jax.jit(lambda data, r, cfg=cfg: jaxamg.solve(bcsr(data), r, config=cfg)[0])
    jax.block_until_ready(apply(A_uni.data, b0))  # cache miss: structure from uniform matrix
    t = timeit(apply, A_het.data, b0)
    P = lambda data, cfg=cfg: lx.FunctionLinearOperator(
        lambda r: jaxamg.solve(bcsr(data), r, config=cfg)[0], structure,
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag))
    f = jax.jit(lambda data, b: cg_solve(data, b, P(data)))
    x, steps, res = f(A_het.data, b0)
    out[f"structure_reuse_levels={levels}"] = {"apply_s": t, "cg_iters": int(steps)}
    print(levels, out[f"structure_reuse_levels={levels}"], flush=True)
    jaxamg.clear_solver_cache()

with open("results_structure_reuse.json", "w") as fh:
    json.dump(out, fh, indent=2)
