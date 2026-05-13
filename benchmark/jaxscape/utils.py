"""Shared utilities for standalone JAXScape benchmark task modules."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import jax
    from benchmark.benchmark_distances import BenchmarkCase, BenchmarkConfig, BenchmarkRecord


LOGGER = logging.getLogger(__name__)
DEFAULT_WALLTIME_SECONDS = 60.0
WORKER_PAYLOAD_ENV = "JAXSCAPE_BENCHMARK_WORKER_PAYLOAD"
OOM_ERROR_MARKERS = (
    "out of memory",
    "oom",
    "resource exhausted",
    "std::bad_alloc",
    "memoryerror",
    "cuda_error_out_of_memory",
)


def configure_standalone_environment() -> None:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def benchmark_walltime_seconds() -> float:
    raw_value = os.environ.get("JAXSCAPE_BENCHMARK_WALLTIME_SECONDS")
    if raw_value is None:
        return DEFAULT_WALLTIME_SECONDS
    try:
        return max(1.0, float(raw_value))
    except ValueError as error:
        message = (
            "JAXSCAPE_BENCHMARK_WALLTIME_SECONDS must be numeric, received "
            f"{raw_value!r}."
        )
        raise RuntimeError(message) from error


def looks_like_oom(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in OOM_ERROR_MARKERS)


def format_walltime_seconds(seconds: float) -> str:
    if seconds.is_integer():
        return f"{seconds:.0f}s"
    return f"{seconds:.3g}s"


def case_by_name(cases: Sequence[BenchmarkCase], case_name: str) -> BenchmarkCase:
    case_lookup = {case.name: case for case in cases}
    try:
        return case_lookup[case_name]
    except KeyError as error:
        available = ", ".join(sorted(case_lookup))
        raise KeyError(
            f"Unknown benchmark case {case_name!r}. Available cases: {available}"
        ) from error


def device_for_backend(backend: str) -> jax.Device:
    import jax

    try:
        return jax.devices(backend)[0]
    except IndexError as error:
        raise RuntimeError(f"No JAX device available for backend {backend!r}.") from error


def worker_payload() -> dict[str, Any] | None:
    raw_payload = os.environ.get(WORKER_PAYLOAD_ENV)
    if raw_payload is None:
        return None
    return json.loads(raw_payload)


def emit_worker_record(record: BenchmarkRecord) -> None:
    print(json.dumps(asdict(record)))


def worker_failure_note(
    tool_label: str,
    *,
    timeout_seconds: float | None = None,
    returncode: int | None = None,
    stderr: str = "",
    stdout: str = "",
) -> str:
    if timeout_seconds is not None:
        return (
            f"Exceeded walltime of {format_walltime_seconds(timeout_seconds)} "
            f"while running {tool_label}."
        )

    combined_output = "\n".join(part for part in (stderr.strip(), stdout.strip()) if part)
    if looks_like_oom(combined_output):
        return f"Out-of-memory while running {tool_label}."
    if returncode is not None and returncode < 0:
        if returncode == -9:
            return (
                f"Worker was killed while running {tool_label}; this often indicates "
                "an out-of-memory condition."
            )
        return f"Worker exited from signal {-returncode} while running {tool_label}."
    if combined_output:
        return combined_output.splitlines()[-1]
    return f"Worker exited with status {returncode} while running {tool_label}."


def run_worker_subprocess(
    *,
    script_path: Path,
    payload: dict[str, Any],
    case_task: str,
    case_name: str,
    tool_label: str,
    software: str,
    walltime_seconds: float,
    logger: logging.Logger,
) -> BenchmarkRecord:
    from benchmark.benchmark_distances import BenchmarkRecord, failed_record

    env = os.environ.copy()
    env[WORKER_PAYLOAD_ENV] = json.dumps(payload)
    logger.info(
        "Dispatching benchmark worker for case=%s tool=%s walltime=%s",
        case_name,
        tool_label,
        format_walltime_seconds(walltime_seconds),
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=walltime_seconds,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        logger.error(
            "Benchmark worker timed out for case=%s tool=%s after %s",
            case_name,
            tool_label,
            format_walltime_seconds(walltime_seconds),
        )
        return failed_record(
            case_task,
            tool_label,
            software,
            case_name,
            worker_failure_note(tool_label, timeout_seconds=walltime_seconds),
        )

    if completed.returncode != 0:
        note = worker_failure_note(
            tool_label,
            returncode=completed.returncode,
            stderr=completed.stderr,
            stdout=completed.stdout,
        )
        log_message = (
            "Benchmark worker likely hit OOM for case=%s tool=%s: %s"
            if looks_like_oom(note)
            else "Benchmark worker failed for case=%s tool=%s: %s"
        )
        logger.error(log_message, case_name, tool_label, note)
        return failed_record(case_task, tool_label, software, case_name, note)

    try:
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        payload = json.loads(output_lines[-1])
        record = BenchmarkRecord(**payload)
    except Exception:
        logger.exception(
            "Benchmark worker returned invalid payload for case=%s tool=%s",
            case_name,
            tool_label,
        )
        note = worker_failure_note(
            tool_label,
            returncode=completed.returncode,
            stderr=completed.stderr,
            stdout=completed.stdout,
        )
        return failed_record(
            case_task,
            tool_label,
            software,
            case_name,
            f"Failed to decode worker result: {note}",
        )

    if completed.stderr.strip():
        logger.info(
            "Worker logs for case=%s tool=%s:\n%s",
            case_name,
            tool_label,
            completed.stderr.strip(),
        )
    return record


def run_standalone_task(
    *,
    collect_task_results,
    config_factory,
    cases,
    worker_handler=None,
) -> None:
    from benchmark.benchmark_distances import BenchmarkConfig, write_results

    configure_logging()
    if worker_handler is not None:
        payload = worker_payload()
        if payload is not None:
            emit_worker_record(worker_handler(payload))
            return

    config: BenchmarkConfig = config_factory()
    records = collect_task_results(config)
    write_results(records, config, cases=cases)
    print(json.dumps({"records": [asdict(record) for record in records]}, indent=2))