"""Validate + time the JAX-native refresh of variant E (plain aggregation).

For each surface: refresh the E hierarchy on device from the current
Laplacian, compare it with PyAMG's E built on the host, count CG iterations
and check that the compiled AMG solve is reused (no retrace).
"""

import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

import common as cm
import ejax

n = int(sys.argv[1])
max_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
surfaces = sys.argv[3:] or ["uniform", "grf2", "grf4", "grf6", "barrier", "floor"]
nodes_np = cm.terminal_nodes(n)
nodes = jnp.asarray(nodes_np)

L_uni = cm.reduced_laplacian(jnp.ones((n, n), jnp.float64))
hB = cm.build_hierarchy(L_uni, {"keep": True})
agg = cm.aggops(hB.pyamg_h)
hE = cm.build_hierarchy(L_uni, cm.predefined_kwargs(agg, smooth=None, improve=None))
t = time.perf_counter()
seg0, segs, weights, diag_pos, nse = ejax.build_plan(L_uni, hE.ml)
t_plan = time.perf_counter() - t
refresh = ejax.make_refresh(nse)
solve = cm.make_cg_solve("amg", max_steps)
out = open(os.path.join(os.path.dirname(__file__), "results", f"ejax_{n}.jsonl"), "a")
print(f"[{n}] host plan {t_plan:.3f}s, levels {cm.level_signature(hE.ml)}", flush=True)

for surf in surfaces:
    c = jnp.asarray(cm.make_surface(surf, n))
    L = jax.block_until_ready(cm.reduced_laplacian(c))
    R_ref, _ = cm.direct_resistances(L, nodes_np)
    assert np.array_equal(np.asarray(L.indices), np.asarray(L_uni.indices))
    args = (L.data, seg0, segs, weights, diag_pos, hE.ml)
    t = time.perf_counter()
    ml = jax.block_until_ready(refresh(*args))
    t_first = time.perf_counter() - t
    ts = []
    for _ in range(5):
        t = time.perf_counter()
        ml = refresh(*args)
        jax.block_until_ready(jax.tree_util.tree_leaves(ml))
        ts.append(time.perf_counter() - t)
    # Compare with PyAMG E on the current matrix.
    host = cm.build_hierarchy(L, cm.predefined_kwargs(agg, smooth=None, improve=None)).ml
    same = cm.same_compiled_signature(ml, hE.ml)
    diffs = []
    for a, b in zip(ml.levels, host.levels):
        x = np.asarray(a.A.data); y = np.asarray(b.A.data)
        if x.shape == y.shape and np.array_equal(np.asarray(a.A.indices), np.asarray(b.A.indices)):
            diffs.append(float(np.max(np.abs(x - y)) / np.max(np.abs(y))))
        else:
            diffs.append(None)
    before = dict(cm.TRACE_COUNT)
    (R, steps), t_solve_first, times = cm.timed_runs(solve, (L, nodes, ml), 3)
    if surf == surfaces[0]:
        retraced = None  # first call compiles by construction
    else:
        retraced = cm.TRACE_COUNT != before
    steps = np.asarray(steps)
    rec = dict(size=n, surface=surf, variant="E_jax", t_refresh_first=t_first,
               t_refresh_warm_median=float(np.median(ts)), t_host_plan=t_plan,
               same_sig_as_template=same, level_relmaxdiff_vs_pyamg=diffs,
               steps_max=int(steps.max()), n_hit_max=int((steps >= max_steps).sum()),
               t_warm_median=float(np.median(times)), retraced=retraced,
               rel_err=cm.rel_error(R, R_ref))
    print(rec, flush=True)
    out.write(json.dumps(rec) + "\n"); out.flush()
