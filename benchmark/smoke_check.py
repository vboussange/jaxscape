"""Run representative benchmark smoke checks with isolated artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

import jax

from benchmark.benchmark_distances import (
    as_array,
    benchmark_graph_volume,
    build_config,
    CASES,
    collect_least_cost_results,
    collect_sensitivity_results,
    configure_benchmark_point_count,
    gdistance_record_from_payload,
    REPEATS,
    RESULTS_DIR,
    run_circuitscape_resistance,
    run_gdistance_payload,
    run_resistancega_inverse,
    write_results,
)


DEFAULT_SMOKE_RESULTS_DIR = RESULTS_DIR / "smoke"
DEFAULT_SMOKE_RESULTS_JSON = DEFAULT_SMOKE_RESULTS_DIR / "benchmark_smoke_results.json"
DEFAULT_SMOKE_RESULTS_CSV = DEFAULT_SMOKE_RESULTS_DIR / "benchmark_smoke_results.csv"
DEFAULT_SMOKE_POINT_COUNT = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--point-count", type=int, default=DEFAULT_SMOKE_POINT_COUNT)
    parser.add_argument("--device", default=os.environ.get("JAXSCAPE_BENCHMARK_DEVICE", "default"))
    parser.add_argument("--results-json", type=Path, default=DEFAULT_SMOKE_RESULTS_JSON)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_SMOKE_RESULTS_CSV)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail when a smoke-check record is skipped as well as failed.",
    )
    return parser.parse_args()


def representative_cases() -> list:
    return [
        CASES["resistance"][0],
        CASES["lcp"][0],
        CASES["sensitivity"][0],
        CASES["inverse"][0],
    ]


def collect_resistance_smoke_records(case, config):
    from benchmark.jaxscape import resistance_distance

    records = resistance_distance.collect_jaxscape_resistance_results(
        case,
        config,
        profile_keys=["pyamg"],
    )
    gdistance_payload = run_gdistance_payload(case, config, "resistance_distance")
    graph_volume = float(jax.device_get(benchmark_graph_volume(as_array(case.raster))))
    if (
        gdistance_payload["status"] == "ok"
        and gdistance_payload.get("metrics", {}).get("distance_matrix") is not None
        and graph_volume > 0
    ):
        commute_matrix = np.asarray(
            gdistance_payload["metrics"]["distance_matrix"], dtype=float
        )
        gdistance_payload["metrics"] = {
            "graph_volume": graph_volume,
            "distance_matrix": (commute_matrix / graph_volume).tolist(),
        }
    records.append(
        gdistance_record_from_payload(
            case,
            "gdistance / commuteDistance",
            gdistance_payload,
            note_suffix=(
                "Reported values are rescaled from commute time to effective "
                "resistance by dividing by the graph volume."
            ),
        )
    )
    records.append(
        run_circuitscape_resistance(
            case,
            config,
            "cg+amg",
            "Circuitscape.jl / cg+amg / f64",
        )
    )
    return records


def collect_inverse_smoke_records(case, config):
    from benchmark.jaxscape import inverse_landscape_genetics

    records = inverse_landscape_genetics.collect_jaxscape_inverse_results(
        case,
        config,
        profile_keys=["amjaxcg_f32"],
    )
    records.append(run_resistancega_inverse(case, config))
    return records


def collect_smoke_records(config) -> tuple[list, list]:
    resistance_case, lcp_case, sensitivity_case, inverse_case = representative_cases()
    records = []
    records.extend(collect_resistance_smoke_records(resistance_case, config))
    records.extend(collect_least_cost_results(lcp_case, config))
    records.extend(collect_sensitivity_results(sensitivity_case, config))
    records.extend(collect_inverse_smoke_records(inverse_case, config))
    return records, [resistance_case, lcp_case, sensitivity_case, inverse_case]


def failing_records(records, *, require_complete: bool) -> list:
    failing = []
    for record in records:
        if record.status == "failed":
            failing.append(record)
            continue
        if require_complete and record.status != "ok":
            failing.append(record)
    return failing


def main() -> int:
    args = parse_args()
    configure_benchmark_point_count(args.point_count)
    config = build_config(
        repeats=args.repeats,
        requested_device_platform=args.device,
        include_conefor=False,
        require_complete=args.require_complete,
        results_json=args.results_json,
        results_csv=args.results_csv,
    )
    records, cases = collect_smoke_records(config)
    write_results(records, config, cases=cases)

    status_counts = Counter(record.status for record in records)
    failures = failing_records(records, require_complete=args.require_complete)
    payload = {
        "results_json": str(config.results_json),
        "results_csv": str(config.results_csv),
        "status_counts": dict(status_counts),
        "records": [asdict(record) for record in records],
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())