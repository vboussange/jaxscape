"""Run the standard exact/inverse comparison into an isolated artifact directory.

Only run after correctness validation. ``--plan`` lists coverage without workloads.
Existing approximate profiles and published artifacts are unaffected.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from benchmark.benchmark_distances import (
    CASES,
    collect_inverse_results,
    collect_resistance_results,
    default_jaxscape_config,
    write_results,
)
from benchmark.inverse_benchmark_settings import inverse_settings_payload
from benchmark.jaxscape.utils import configure_logging


PROFILE_KEYS = {
    "resistance": [
        "pyamg", "cholmod_f32", "cholmod_f64",
        "amjaxcg_f32", "amjaxcg_f64", "jaxamgcg_f32", "jaxamgcg_f64",
    ],
    "inverse": [
        "cholmod_f32", "amjaxcg_f32", "amjaxcg_f64", "jaxamgcg_f32", "jaxamgcg_f64",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    plan = {group: {"profiles": keys, "cases": [case.name for case in CASES[group]]}
            for group, keys in PROFILE_KEYS.items()}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    output = args.output_dir.resolve()
    if output == (ROOT / "benchmark" / "results").resolve():
        parser.error("Use an isolated run directory, not the canonical published results.")
    output.mkdir(parents=True, exist_ok=False)
    configure_logging()
    config = replace(
        default_jaxscape_config(),
        results_json=output / "benchmark_results.json",
        results_csv=output / "benchmark_results.csv",
    )
    manifest = {
        "command": [sys.executable, *sys.argv],
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True).strip(),
        "profiles": plan,
        "inverse_settings": inverse_settings_payload(),
        "environment_overrides": {key: value for key, value in os.environ.items()
            if key.startswith(("JAXSCAPE_BENCHMARK_", "BENCHMARK_", "CUDA_", "AMGX_"))},
        "timing": "one untimed warm-up, then three synchronized repeats (default)",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    records = []
    cases = CASES["resistance"] + CASES["inverse"]
    for group, collector in [("resistance", collect_resistance_results),
                             ("inverse", collect_inverse_results)]:
        for case in CASES[group]:
            records.extend(collector(case, config, profile_keys=PROFILE_KEYS[group]))
            write_results(records, config, cases=cases)


if __name__ == "__main__":
    main()
