"""Shared solver and method factories for resistance-distance benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxscape import SpielmanApproximation


@dataclass(frozen=True)
class ResistanceProfile:
    key: str
    tool: str
    solver_factory: Callable[[], Any] | None = None
    method_factory: Callable[[], Any] | None = None
    gpu_capable: bool = False
    requires_preparation: bool = False
    dtype: Any = jnp.float32


RESISTANCE_SOLVER_RTOL = 1e-3
RESISTANCE_SOLVER_ATOL = 1e-3
RESISTANCE_SOLVER_MAX_ITERATIONS: int | None = None
RESISTANCE_APPROXIMATION_EPSILON = 0.05


def make_pyamg_solver() -> Any:
    try:
        from jaxscape.solvers.pyamgsolver import PyAMGSolver
    except ImportError as error:
        raise ImportError(
            "Install the benchmark Python extra to enable the PyAMG resistance profile."
        ) from error
    return PyAMGSolver(
        rtol=RESISTANCE_SOLVER_RTOL,
        maxiter=RESISTANCE_SOLVER_MAX_ITERATIONS,
    )


def make_cholmod_solver() -> Any:
    try:
        from jaxscape.solvers.cholmodsolver import CholmodSolver
    except ImportError as error:
        raise ImportError(
            "Install the cholespy Python extra to enable the Cholmod profile."
        ) from error
    return CholmodSolver()


def make_amjaxcg_solver() -> Any:
    try:
        from jaxscape.solvers.amjaxcgsolver import AMJaxCGSolver
    except ImportError as error:
        raise ImportError(
            "Install the amjax Python extra to enable the AMJaxCG resistance profile."
        ) from error
    return AMJaxCGSolver(
        rtol=RESISTANCE_SOLVER_RTOL,
        atol=RESISTANCE_SOLVER_ATOL,
        max_steps=RESISTANCE_SOLVER_MAX_ITERATIONS,
    )


def make_spielman_method() -> Any:
    return SpielmanApproximation(epsilon=RESISTANCE_APPROXIMATION_EPSILON, seed=0)
