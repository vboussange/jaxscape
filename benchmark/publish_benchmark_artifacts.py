"""Promote a validated benchmark run into the canonical published artifact paths."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

from benchmark.benchmark_distances import RESULTS_CSV, RESULTS_JSON
from benchmark.benchmark_registry import benchmark_scorecard_outputs


CANONICAL_ASSET_PATHS = benchmark_scorecard_outputs(ROOT)
CANONICAL_ASSETS_DIR = ROOT / "docs" / "assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--scorecard-dir", type=Path, required=True)
    return parser.parse_args()


def copy_if_exists(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    copy_if_exists(args.results_json, RESULTS_JSON)
    copy_if_exists(args.results_csv, RESULTS_CSV)
    for output_path in CANONICAL_ASSET_PATHS.values():
        source = args.scorecard_dir / output_path.name
        copy_if_exists(source, output_path)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmark" / "generate_benchmark_docs.py"),
            "--results-json",
            str(args.results_json),
            "--output",
            str(ROOT / "docs" / "benchmark.md"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()