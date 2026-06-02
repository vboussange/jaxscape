#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source benchmark/toolchain_env.sh
benchmark_ensure_local_julia

CUDA_DEVICE="${CUDA_DEVICE:-3}"
RUN_NAME="${RUN_NAME:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="benchmark/results/runs/${RUN_NAME}"
LOG_DIR="${RUN_DIR}/logs"
ASSET_DIR="${RUN_DIR}/assets"
RESULTS_JSON="${RUN_DIR}/benchmark_results.json"
RESULTS_CSV="${RUN_DIR}/benchmark_results.csv"
STDOUT_LOG="${LOG_DIR}/stdout.log"
STDERR_LOG="${LOG_DIR}/stderr.log"
STATUS_LOG="${RUN_DIR}/run_status.txt"

mkdir -p "${LOG_DIR}" "${ASSET_DIR}"

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
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

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

echo "Launching benchmark run ${RUN_NAME}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  results=${RESULTS_JSON}"
echo "  stdout=${STDOUT_LOG}"
echo "  stderr=${STDERR_LOG}"
echo "  status=${STATUS_LOG}"

nohup bash -lc '
set -uo pipefail
cd "'$PWD'"
export BENCHMARK_THREADS="'"${BENCHMARK_THREADS}"'"
export OMP_NUM_THREADS="'"${OMP_NUM_THREADS}"'"
export OPENBLAS_NUM_THREADS="'"${OPENBLAS_NUM_THREADS}"'"
export MKL_NUM_THREADS="'"${MKL_NUM_THREADS}"'"
export VECLIB_MAXIMUM_THREADS="'"${VECLIB_MAXIMUM_THREADS}"'"
export NUMEXPR_NUM_THREADS="'"${NUMEXPR_NUM_THREADS}"'"
export JULIA_NUM_THREADS="'"${JULIA_NUM_THREADS}"'"
export RCPP_PARALLEL_NUM_THREADS="'"${RCPP_PARALLEL_NUM_THREADS}"'"
export RCPPTHREAD_NUM_THREADS="'"${RCPPTHREAD_NUM_THREADS}"'"
export XLA_FLAGS="'"${XLA_FLAGS}"'"
export CUDA_VISIBLE_DEVICES="'"${CUDA_VISIBLE_DEVICES}"'"
benchmark_exit=0
render_exit=0
docs_exit=0
'"${PYTHON_RUNNER[*]}"' benchmark/benchmark_distances.py --device gpu --require-complete --results-json "'"${RESULTS_JSON}"'" --results-csv "'"${RESULTS_CSV}"'" || benchmark_exit=$?
if [[ -f "'"${RESULTS_JSON}"'" ]]; then
	'"${PYTHON_RUNNER[*]}"' benchmark/render_scorecard.py --results-json "'"${RESULTS_JSON}"'" --output-dir "'"${ASSET_DIR}"'" || render_exit=$?
	'"${PYTHON_RUNNER[*]}"' benchmark/generate_benchmark_docs.py --results-json "'"${RESULTS_JSON}"'" --output "'"${RUN_DIR}"'"/benchmark.md || docs_exit=$?
fi
{
	echo "benchmark_exit=${benchmark_exit}"
	echo "render_exit=${render_exit}"
	echo "docs_exit=${docs_exit}"
} > "'"${STATUS_LOG}"'"
if [[ ${benchmark_exit} -ne 0 ]]; then
	exit ${benchmark_exit}
fi
if [[ ${render_exit} -ne 0 ]]; then
	exit ${render_exit}
fi
exit ${docs_exit}
' >"${STDOUT_LOG}" 2>"${STDERR_LOG}" &

echo $! > "${RUN_DIR}/pid"
echo "PID $(cat "${RUN_DIR}/pid")"