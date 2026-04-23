#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python benchmark/benchmark_distances.py
python benchmark/render_scorecard.py
