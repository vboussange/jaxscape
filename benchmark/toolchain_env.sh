#!/usr/bin/env bash

benchmark_prepend_dir_once() {
	local candidate_dir="$1"
	if [[ ! -d "${candidate_dir}" ]]; then
		return 0
	fi
	case ":${PATH}:" in
		*":${candidate_dir}:"*) ;;
		*) export PATH="${candidate_dir}:${PATH}" ;;
	esac
}

benchmark_ensure_local_julia() {
	benchmark_prepend_dir_once "$HOME/.cargo/bin"
	benchmark_prepend_dir_once "$HOME/.local/bin"

	if command -v julia >/dev/null 2>&1; then
		return 0
	fi

	benchmark_prepend_dir_once "$HOME/.juliaup/bin"
	if command -v julia >/dev/null 2>&1; then
		return 0
	fi
}