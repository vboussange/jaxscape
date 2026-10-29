"""spineax: jit re-use (no re-analysis / recompile), implicit grads, Lineax path.

Effective resistances among T terminals: X = L_g^{-1} E_T (T RHS),
R_ij = X_ii + X_jj - 2 X_ij; loss = sum_{i<j} R_ij.
"""

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np

from spineax import cudss
from spineax.cudss import CSROperator, CuDSS

from bench_spineax import (
    grounded_laplacian_structure,
    lognormal_conductance,
    make_values_fn,
    terminals,
)

jax.config.update("jax_enable_x64", True)


def resistance_loss_raw(token, vals, T, M):
    E = jnp.zeros((len(T), M)).at[jnp.arange(len(T)), T].set(1.0)
    tok = cudss.refactorize(token, vals)
    X = cudss.solve(tok, E)  # (T, M) multi-RHS
    S = X[:, T]
    d = jnp.diag(S)
    R = d[:, None] + d[None, :] - S - S.T
    return jnp.sum(jnp.triu(R, 1)), tok


def main(n, k, reps=3):
    out = {"n": n, "k": k}
    S = grounded_laplacian_structure(n)
    M = S["M"]
    values_fn = make_values_fn(S)
    offs = jnp.asarray(S["offsets"])
    cols = jnp.asarray(S["columns"])
    T = jnp.asarray(terminals(n, k, seed=7))
    conds = [jnp.asarray(lognormal_conductance(n, s)) for s in range(reps + 2)]

    tok0 = jax.jit(lambda v: cudss.factorize(
        cudss.analyze(v, offs, cols, mtype_id="spd", mview_id="full"), v))(
        values_fn(conds[0]))

    def loss_fn(cond, token):
        return resistance_loss_raw(token, values_fn(cond), T, M)

    step = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    # (a) token threaded through steps: expect 0 rebuilds, 1 compilation
    (l, tok), g = step(conds[0], tok0)
    jax.block_until_ready(g)
    rb0 = cudss.rebuild_count()
    ts = []
    for c in conds[1:]:
        t = time.perf_counter()
        (l, tok), g = step(c, tok)
        jax.block_until_ready(g)
        ts.append(time.perf_counter() - t)
    out["threaded_step_fwd_bwd_s"] = statistics.median(ts)
    out["threaded_rebuilds"] = cudss.rebuild_count() - rb0
    out["threaded_cache_size"] = step._cache_size()

    # (b) frozen token (as a jaxscape `state` would be): every step re-analyzes
    rb0 = cudss.rebuild_count()
    frozen = tok
    ts = []
    for c in conds[1:]:
        t = time.perf_counter()
        (l, _), g = step(c, frozen)
        jax.block_until_ready(g)
        ts.append(time.perf_counter() - t)
    out["frozen_step_fwd_bwd_s"] = statistics.median(ts)
    out["frozen_rebuilds_per_step"] = (cudss.rebuild_count() - rb0) / len(ts)

    # (c) gradient check: central FD along a random multiplicative direction
    c = conds[1]
    rng = np.random.default_rng(0)
    dirn = jnp.asarray(rng.standard_normal(c.shape)) * c
    (l0, tok), g = step(c, tok)
    h = 1e-6
    (lp, tok), _ = step(c + h * dirn, tok)
    (lm, tok), _ = step(c - h * dirn, tok)
    fd = (lp - lm) / (2 * h)
    ad = jnp.vdot(g, dirn)
    out["grad_fd"] = float(fd)
    out["grad_ad"] = float(ad)
    out["grad_relerr"] = float(abs(fd - ad) / abs(fd))

    print(json.dumps(out), flush=True)

    # (e) raw API but vmapped over RHS (the shape jaxscape uses)
    def vraw_loss(cond, token):
        vals = values_fn(cond)
        tok = cudss.refactorize(token, vals)
        E = jnp.zeros((len(T), M)).at[jnp.arange(len(T)), T].set(1.0)
        X = jax.vmap(lambda b: cudss.solve(tok, b))(E)
        Sx = X[:, T]
        d = jnp.diag(Sx)
        R = d[:, None] + d[None, :] - Sx - Sx.T
        return jnp.sum(jnp.triu(R, 1)), tok
    try:
        vstep = jax.jit(jax.value_and_grad(vraw_loss, has_aux=True))
        (lv, tok), gv = vstep(c, tok)
        out["vmapped_raw_grad_relerr"] = float(jnp.linalg.norm(gv - g) / jnp.linalg.norm(g))
    except Exception as ex:
        out["vmapped_raw_error"] = str(ex).splitlines()[0][:200]

    # (d) Lineax contract (what a jaxscape solver would expose):
    #     vmap(lx.linear_solve) over RHS with a materialized state, under grad
    solver = CuDSS()

    def lineax_loss(cond, token):
        vals = values_fn(cond)
        op = CSROperator(vals, offs, cols, (lx.symmetric_tag,
                                            lx.positive_semidefinite_tag))
        state = solver.refactorize(token, jax.lax.stop_gradient(op))
        E = jnp.zeros((len(T), M)).at[jnp.arange(len(T)), T].set(1.0)
        X = jax.vmap(lambda b: lx.linear_solve(op, b, solver, state=state).value)(E)
        Sx = X[:, T]
        d = jnp.diag(Sx)
        R = d[:, None] + d[None, :] - Sx - Sx.T
        return jnp.sum(jnp.triu(R, 1)), state

    lstep = jax.jit(jax.value_and_grad(lineax_loss, has_aux=True))
    try:
        (ll, tok), lg = lstep(c, tok)
    except Exception as ex:
        out["lineax_vmap_grad_error"] = str(ex).splitlines()[0][:200]
        print(json.dumps(out), flush=True)
        return
    rb0 = cudss.rebuild_count()
    ts = []
    for cc in conds[1:]:
        t = time.perf_counter()
        (ll, tok), lg = lstep(cc, tok)
        jax.block_until_ready(lg)
        ts.append(time.perf_counter() - t)
    (ll, tok), lg = lstep(c, tok)
    out["lineax_step_fwd_bwd_s"] = statistics.median(ts)
    out["lineax_rebuilds"] = cudss.rebuild_count() - rb0
    out["lineax_vs_raw_loss_relerr"] = float(abs(ll - l0) / abs(l0))
    out["lineax_vs_raw_grad_relerr"] = float(jnp.linalg.norm(lg - g) / jnp.linalg.norm(g))
    out["lineax_cache_size"] = lstep._cache_size()
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 300])
    p.add_argument("--k", type=int, nargs="+", default=[20, 100])
    a = p.parse_args()
    for n in a.sizes:
        for k in a.k:
            main(n, k)
