"""Compact tables from results/results_*.jsonl."""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SURF = ["uniform", "grf2", "grf4", "grf6", "barrier", "floor"]
VARS = ["cg", "jacobi", "B", "B_fine", "C", "C_t25", "D", "E"]

rows = []
setup = {}
for path in sorted(glob.glob(os.path.join(HERE, "results", "results_*.jsonl"))):
    for line in open(path):
        r = json.loads(line)
        if r.get("variant") == "B_setup":
            setup[r["size"]] = r
        else:
            rows.append(r)

idx = {(r["size"], r["surface"], r["variant"]): r for r in rows}
sizes = sorted({r["size"] for r in rows})


def fmt_steps(r):
    if r is None:
        return "-"
    if "error" in r:
        return "ERR"
    s = str(r["steps_max"])
    if r["n_hit_max"]:
        s = f">{r['max_steps']}"
    return s


print("## CG iterations (max over 20 RHS); '>N' = hit max_steps\n")
for n in sizes:
    print(f"### {n}x{n}")
    print("| surface | " + " | ".join(VARS) + " |")
    print("|---" * (len(VARS) + 1) + "|")
    for s in SURF:
        print(f"| {s} | " + " | ".join(fmt_steps(idx.get((n, s, v))) for v in VARS) + " |")
    print()

print("## Warm GPU solve time, 20 RHS (s, median of warm runs)\n")
for n in sizes:
    print(f"### {n}x{n}")
    print("| surface | " + " | ".join(VARS) + " |")
    print("|---" * (len(VARS) + 1) + "|")
    for s in SURF:
        cells = []
        for v in VARS:
            r = idx.get((n, s, v))
            cells.append("-" if r is None or "error" in r else f"{r['t_warm_median']:.3g}")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print()

print("## Rebuild cost (s): total [pyamg host] ; same compiled signature as B ; retraced on swap\n")
for n in sizes:
    print(f"### {n}x{n}  (B cold build {setup.get(n, {}).get('t_cold_total', float('nan')):.2f}s, "
          f"warm {setup.get(n, {}).get('t_warm_total', float('nan')):.3f}s)")
    print("| surface | " + " | ".join(["B_fine", "C", "C_t25", "D", "E"]) + " |")
    print("|---" * 6 + "|")
    for s in SURF:
        cells = []
        for v in ["B_fine", "C", "C_t25", "D", "E"]:
            r = idx.get((n, s, v))
            if r is None or "error" in r:
                cells.append("-")
                continue
            cells.append(
                f"{r['t_setup_total']:.3g} [{r['t_pyamg']:.3g}] "
                f"{'=' if r['same_sig_as_B'] else '≠'}B "
                f"{'RECOMP ' + format(r['t_first'] - r['t_warm_median'], '.2g') + 's' if r['retraced'] else 'reuse'}"
            )
        print(f"| {s} | " + " | ".join(cells) + " |")
    print()

print("## Max relative resistance error vs direct (cholespy)\n")
for n in sizes:
    print(f"### {n}x{n}")
    print("| surface | " + " | ".join(VARS) + " |")
    print("|---" * (len(VARS) + 1) + "|")
    for s in SURF:
        cells = []
        for v in VARS:
            r = idx.get((n, s, v))
            cells.append("-" if r is None or "error" in r else f"{r['rel_err']:.1e}")
        print(f"| {s} | " + " | ".join(cells) + " |")
    print()

print("## Levels / nse (B vs D vs C_t25 vs E) and errors\n")
for n in sizes:
    for v in ["B", "D", "C_t25", "E"]:
        r = idx.get((n, "grf4", v))
        if r and "levels" in r:
            print(n, v, [(l["n"], l["A_nse"], l.get("P_nse")) for l in r["levels"]])
    for s in SURF:
        for v in ["C", "D"]:
            r = idx.get((n, s, v))
            if r and "same_sig_as_B" in r and not r["same_sig_as_B"]:
                print("  signature differs:", n, s, v, [(l["n"], l["A_nse"], l.get("P_nse")) for l in r["levels"]])
        r = idx.get((n, s, "C"))
        if r and "C_aggops_equal_B" in r:
            print("  C aggregates == B aggregates:", n, s, r["C_aggops_equal_B"])
for r in rows:
    if "error" in r:
        print("ERROR", r["size"], r["surface"], r["variant"], r["error"][:200])

print("\n## E_jax (plain aggregation refreshed on device by segment_sum)\n")
print("| size | surface | refresh warm (ms) | iters | solve (s) | =template sig | retraced | max level rel diff vs PyAMG | rel_err |")
print("|---|---|---|---|---|---|---|---|---|")
for path in sorted(glob.glob(os.path.join(HERE, "results", "ejax_*.jsonl"))):
    for line in open(path):
        r = json.loads(line)
        d = max(x for x in r["level_relmaxdiff_vs_pyamg"] if x is not None)
        print(f"| {r['size']} | {r['surface']} | {1e3*r['t_refresh_warm_median']:.2f} | {r['steps_max']} | "
              f"{r['t_warm_median']:.3g} | {r['same_sig_as_template']} | {r['retraced']} | {d:.1e} | {r['rel_err']:.1e} |")
