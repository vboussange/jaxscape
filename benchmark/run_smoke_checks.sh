#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source benchmark/toolchain_env.sh
benchmark_ensure_local_julia

BENCHMARK_THREADS="${BENCHMARK_THREADS:-4}"
export BENCHMARK_THREADS
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$BENCHMARK_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$BENCHMARK_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$BENCHMARK_THREADS}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-$BENCHMARK_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$BENCHMARK_THREADS}"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-$BENCHMARK_THREADS}"
export RCPP_PARALLEL_NUM_THREADS="${RCPP_PARALLEL_NUM_THREADS:-$BENCHMARK_THREADS}"
export RCPPTHREAD_NUM_THREADS="${RCPPTHREAD_NUM_THREADS:-$BENCHMARK_THREADS}"

if [[ "${BENCHMARK_THREADS}" == "1" ]]; then
	if [[ " ${XLA_FLAGS:-} " != *" --xla_cpu_multi_thread_eigen=false "* ]]; then
		if [[ -n "${XLA_FLAGS:-}" ]]; then
			export XLA_FLAGS="${XLA_FLAGS} --xla_cpu_multi_thread_eigen=false"
		else
			export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
		fi
	fi
else
	if [[ " ${XLA_FLAGS:-} " != *" --xla_cpu_multi_thread_eigen=true "* ]]; then
		if [[ -n "${XLA_FLAGS:-}" ]]; then
			export XLA_FLAGS="${XLA_FLAGS} --xla_cpu_multi_thread_eigen=true"
		else
			export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
		fi
	fi
fi

if [[ " ${XLA_FLAGS:-} " != *" intra_op_parallelism_threads=${BENCHMARK_THREADS} "* ]]; then
	export XLA_FLAGS="${XLA_FLAGS:-} intra_op_parallelism_threads=${BENCHMARK_THREADS}"
	XLA_FLAGS="${XLA_FLAGS# }"
	export XLA_FLAGS
fi

PYTHON_RUNNER=(python)
if command -v uv >/dev/null 2>&1; then
	PYTHON_RUNNER=(uv run --extra benchmark python)
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_RUNNER=(python3)
elif ! command -v python >/dev/null 2>&1; then
	echo "Python is not available on PATH." >&2
	exit 1
fi

SMOKE_RESULTS_JSON="${SMOKE_RESULTS_JSON:-benchmark/results/smoke/benchmark_smoke_results.json}"
SMOKE_RESULTS_CSV="${SMOKE_RESULTS_CSV:-benchmark/results/smoke/benchmark_smoke_results.csv}"
SMOKE_OUTPUT_DIR="${SMOKE_OUTPUT_DIR:-benchmark/results/smoke/assets}"
SMOKE_DOC_PATH="${SMOKE_DOC_PATH:-benchmark/results/smoke/benchmark.md}"

"${PYTHON_RUNNER[@]}" benchmark/smoke_check.py \
	--results-json "${SMOKE_RESULTS_JSON}" \
	--results-csv "${SMOKE_RESULTS_CSV}" \
	"$@"

"${PYTHON_RUNNER[@]}" benchmark/render_scorecard.py \
	--results-json "${SMOKE_RESULTS_JSON}" \
	--output-dir "${SMOKE_OUTPUT_DIR}"

"${PYTHON_RUNNER[@]}" benchmark/generate_benchmark_docs.py \
	--results-json "${SMOKE_RESULTS_JSON}" \
	--output "${SMOKE_DOC_PATH}"

if [[ ! -f "${SMOKE_RESULTS_JSON}" || ! -f "${SMOKE_RESULTS_CSV}" || ! -f "${SMOKE_DOC_PATH}" ]]; then
	echo "Smoke pipeline did not produce the expected result artifacts." >&2
	exit 1
fi

if [[ $(find "${SMOKE_OUTPUT_DIR}" -maxdepth 1 -name 'benchmark_*.png' | wc -l) -lt 4 ]]; then
	echo "Smoke pipeline did not produce the expected scorecard assets." >&2
	exit 1
fi

if [[ $(find "${SMOKE_OUTPUT_DIR}" -maxdepth 1 -name 'benchmark_*.pdf' | wc -l) -lt 4 ]]; then
	echo "Smoke pipeline did not produce the expected PDF scorecard assets." >&2
	exit 1
fi