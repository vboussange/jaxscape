"""Shared configuration for inverse landscape genetics benchmarks."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class JaxscapeInverseSettings:
    optimizer: str = "lbfgs"
    max_steps: int = 50
    rtol: float = 1e-5
    atol: float = 1e-5


@dataclass(frozen=True)
class ResistanceGAInverseSettings:
    method: str = "LL"
    transformation: str = "M"
    pop_size: int = 12
    maxiter: int = 3
    run: int = 1
    max_cont: int = 25
    seed: int = 7


@dataclass(frozen=True)
class InverseBenchmarkSettings:
    jaxscape: JaxscapeInverseSettings
    resistancega: ResistanceGAInverseSettings


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return float(raw_value)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def load_inverse_benchmark_settings() -> InverseBenchmarkSettings:
    return InverseBenchmarkSettings(
        jaxscape=JaxscapeInverseSettings(
            optimizer=_env_str("JAXSCAPE_BENCHMARK_INVERSE_OPTIMIZER", "lbfgs"),
            max_steps=_env_int("JAXSCAPE_BENCHMARK_INVERSE_MAX_STEPS", 50),
            rtol=_env_float("JAXSCAPE_BENCHMARK_INVERSE_RTOL", 1e-5),
            atol=_env_float("JAXSCAPE_BENCHMARK_INVERSE_ATOL", 1e-5),
        ),
        resistancega=ResistanceGAInverseSettings(
            method=_env_str("JAXSCAPE_BENCHMARK_RGA_METHOD", "LL"),
            transformation=_env_str("JAXSCAPE_BENCHMARK_RGA_TRANSFORMATION", "M"),
            pop_size=_env_int("JAXSCAPE_BENCHMARK_RGA_POP_SIZE", 12),
            maxiter=_env_int("JAXSCAPE_BENCHMARK_RGA_MAXITER", 3),
            run=_env_int("JAXSCAPE_BENCHMARK_RGA_RUN", 1),
            max_cont=_env_int("JAXSCAPE_BENCHMARK_RGA_MAX_CONT", 25),
            seed=_env_int("JAXSCAPE_BENCHMARK_RGA_SEED", 7),
        ),
    )


def inverse_settings_payload() -> dict[str, dict[str, object]]:
    settings = load_inverse_benchmark_settings()
    return {
        "jaxscape": asdict(settings.jaxscape),
        "resistancega": asdict(settings.resistancega),
    }