#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BENCHMARK_DIR="$ROOT_DIR/benchmark"
source "$BENCHMARK_DIR/toolchain_env.sh"

benchmark_ensure_local_julia

JULIA_PROJECT_DIR="$BENCHMARK_DIR/julia"
JULIA_DEPOT_DIR="$BENCHMARK_DIR/.julia"
R_LIBS_DIR="$BENCHMARK_DIR/.r-lib"

mkdir -p "$JULIA_PROJECT_DIR" "$JULIA_DEPOT_DIR" "$R_LIBS_DIR"

export JULIA_DEPOT_PATH="$JULIA_DEPOT_DIR"
export R_LIBS_USER="$R_LIBS_DIR"

if ! command -v julia >/dev/null 2>&1; then
	echo "Julia is not available on PATH. Install Julia system-wide or with juliaup." >&2
	exit 1
fi

echo "Installing local Julia benchmark toolchain into $JULIA_PROJECT_DIR and $JULIA_DEPOT_DIR"
julia --project="$JULIA_PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "Installing local R benchmark toolchain into $R_LIBS_DIR"
Rscript --vanilla "$BENCHMARK_DIR/external/install_r_packages.R" "$R_LIBS_DIR"

echo "Installed Circuitscape.jl in $JULIA_PROJECT_DIR with depot $JULIA_DEPOT_DIR."
echo "Installed R benchmark packages in $R_LIBS_DIR."
echo "Set CONEFOR_BIN to include the optional Conefor adapter in a local run."
