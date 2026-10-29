"""Iteration counts / timings of CG preconditioners on heterogeneous grids.

Usage: python run.py --size 100 [--max-steps 20000] [--variants ...] [--surfaces ...]
Writes one JSON line per (size, surface, variant) to results/results_<size>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

import common as cm

ALL_SURFACES = ["uniform", "grf2", "grf4", "grf6", "barrier", "floor"]
ALL_VARIANTS = ["cg", "jacobi", "B", "B_fine", "C", "C_t25", "D", "E"]


def aggops_equal(h1, h2) -> bool:
    a1, a2 = cm.aggops(h1), cm.aggops(h2)
    if len(a1) != len(a2):
        return False
    for x, y in zip(a1, a2):
        if x.shape != y.shape or (x != y).nnz != 0:
            return False
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--max-steps-cg", type=int, default=None,
                   help="override for unpreconditioned/Jacobi CG")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--variants", nargs="*", default=ALL_VARIANTS)
    p.add_argument("--surfaces", nargs="*", default=ALL_SURFACES)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    n = args.size
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "results", f"results_{n}.jsonl"
    )
    fout = open(out_path, "a")
    nodes_np = cm.terminal_nodes(n)
    nodes = jnp.asarray(nodes_np)
    ms_cg = args.max_steps_cg or args.max_steps

    solvers = {
        "none": cm.make_cg_solve("none", ms_cg),
        "jacobi": cm.make_cg_solve("jacobi", ms_cg),
        "amg": cm.make_cg_solve("amg", args.max_steps),
    }

    # --- frozen uniform hierarchy (ilg-toolkit's current setup) ---
    L_uni = cm.reduced_laplacian(jnp.ones((n, n), jnp.float64))
    hB_cold = cm.build_hierarchy(L_uni, {"keep": True})
    hB = cm.build_hierarchy(L_uni, {"keep": True})  # warm: excludes one-off jit of conversion ops
    print(f"[{n}] B cold setup {hB_cold.t_total:.3f}s (pyamg {hB_cold.t_pyamg:.3f}s)", flush=True)
    fout.write(json.dumps({"size": n, "variant": "B_setup", "t_cold_total": hB_cold.t_total,
                           "t_cold_pyamg": hB_cold.t_pyamg, "t_warm_total": hB.t_total,
                           "t_warm_pyamg": hB.t_pyamg}) + "\n")
    sigB = cm.level_signature(hB.ml)
    print(f"[{n}] B setup total {hB.t_total:.3f}s pyamg {hB.t_pyamg:.3f}s; levels {sigB}",
          flush=True)
    aggB = cm.aggops(hB.pyamg_h)

    for surf in args.surfaces:
        c = jnp.asarray(cm.make_surface(surf, n))
        L = jax.block_until_ready(cm.reduced_laplacian(c))
        R_ref, t_direct = cm.direct_resistances(L, nodes_np)
        print(f"[{n}] {surf}: log10 c in [{np.log10(float(c.min())):.2f}, "
              f"{np.log10(float(c.max())):.2f}], direct {t_direct:.2f}s", flush=True)
        base = {"size": n, "surface": surf, "t_direct": t_direct,
                "log10c_min": float(np.log10(float(c.min()))),
                "log10c_max": float(np.log10(float(c.max())))}

        for var in args.variants:
            rec = dict(base, variant=var)
            ml = None
            hier = None
            try:
                if var == "cg":
                    fn = solvers["none"]
                    rec["max_steps"] = ms_cg
                elif var == "jacobi":
                    fn = solvers["jacobi"]
                    rec["max_steps"] = ms_cg
                else:
                    fn = solvers["amg"]
                    rec["max_steps"] = args.max_steps
                    if var == "B":
                        ml = hB.ml
                        rec.update(t_setup_total=0.0, t_pyamg=0.0, t_other=0.0)
                    elif var == "B_fine":
                        ml, t = cm.fine_refreshed(hB.ml, L)
                        rec.update(t_setup_total=t, t_pyamg=0.0, t_other=t)
                    else:
                        if var == "C":
                            kw = {"keep": True}
                        elif var == "C_t25":
                            kw = {"keep": True,
                                  "strength": ("symmetric", {"theta": 0.25})}
                        elif var == "D":
                            kw = cm.predefined_kwargs(aggB)
                        elif var == "E":
                            kw = cm.predefined_kwargs(aggB, smooth=None, improve=None)
                        else:
                            raise ValueError(var)
                        hier = cm.build_hierarchy(L, kw)
                        ml = hier.ml
                        rec.update(t_setup_total=hier.t_total, t_pyamg=hier.t_pyamg,
                                   t_other=hier.t_other)
                    rec["levels"] = cm.level_signature(ml)
                    rec["n_levels"] = len(ml.levels)
                    rec["same_sig_as_B"] = cm.same_compiled_signature(ml, hB.ml)
                    if var == "C" and hier is not None:
                        rec["C_aggops_equal_B"] = aggops_equal(hier.pyamg_h, hB.pyamg_h)
                    rec["op_complexity"] = float(ml.operator_complexity())

                cache_before = fn._cache_size()
                traces_before = dict(cm.TRACE_COUNT)
                (R, steps), t_first, times = cm.timed_runs(
                    fn, (L, nodes, ml), args.repeats
                )
                steps = np.asarray(steps)
                rec.update(
                    steps_max=int(steps.max()), steps_mean=float(steps.mean()),
                    steps_min=int(steps.min()),
                    n_hit_max=int((steps >= rec["max_steps"]).sum()),
                    exceeds_1000=bool(steps.max() > 1000),
                    t_first=t_first, t_warm=times,
                    t_warm_median=float(np.median(times)),
                    cache_before=cache_before, cache_after=fn._cache_size(),
                    retraced=cm.TRACE_COUNT != traces_before,
                    rel_err=cm.rel_error(R, R_ref),
                )
                rec["per_iter_ms"] = 1e3 * rec["t_warm_median"] / max(1, rec["steps_max"])
            except Exception as e:  # keep the sweep going, record the failure
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
            print(f"[{n}] {surf:8s} {var:7s} "
                  + " ".join(f"{k}={rec.get(k)}" for k in (
                      "steps_max", "t_warm_median", "t_setup_total", "t_pyamg",
                      "rel_err", "same_sig_as_B", "retraced", "n_levels", "error")),
                  flush=True)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            del ml, hier


if __name__ == "__main__":
    main()
