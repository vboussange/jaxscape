#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

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
fi

"${PYTHON_RUNNER[@]}" benchmark/benchmark_distances.py "$@"
"${PYTHON_RUNNER[@]}" benchmark/render_scorecard.py
