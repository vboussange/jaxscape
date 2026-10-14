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
    gpu_only: bool = False
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


def make_jaxamgcg_solver() -> Any:
    """Use upstream-managed native refresh inside compiled applications."""
    from jaxscape.solvers import JAXAMGCGSolver

    try:
        return JAXAMGCGSolver(
            rtol=RESISTANCE_SOLVER_RTOL,
            atol=RESISTANCE_SOLVER_ATOL,
            max_steps=RESISTANCE_SOLVER_MAX_ITERATIONS,
        )
    except (ImportError, OSError, RuntimeError) as error:
        raise ImportError(
            "JAX-AMG benchmark unavailable: install jaxscape[jaxamg] and its "
            f"native CUDA/AmgX backend on an NVIDIA GPU. {error}"
        ) from error


JAXAMG_RESISTANCE_PROFILES = (
    ResistanceProfile(
        "jaxamgcg_f32",
        "JAXScape / JAXAMGCGSolver / f32",
        solver_factory=make_jaxamgcg_solver,
        gpu_capable=True,
        gpu_only=True,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "jaxamgcg_f64",
        "JAXScape / JAXAMGCGSolver / f64",
        solver_factory=make_jaxamgcg_solver,
        gpu_capable=True,
        gpu_only=True,
        dtype=jnp.float64,
    ),
)


def profile_unavailable_reason(profile: ResistanceProfile, backend: str) -> str | None:
    """Check GPU-only hardware without launching any numerical workload."""
    if not profile.gpu_only:
        return None
    message = "JAX-AMG requires an NVIDIA GPU with CUDA-enabled JAX; no CPU fallback."
    if backend != "gpu":
        return message
    import jax

    try:
        devices = jax.devices("gpu")
    except RuntimeError:
        return message
    if not devices or "cuda" not in devices[0].client.platform_version.lower():
        return message
    return None


def profile_metadata(profile: ResistanceProfile, backend: str) -> dict[str, Any]:
    """Interpretation metadata, independent of timing and native cache internals."""
    metadata = {"dtype": str(jnp.dtype(profile.dtype)), "backend": backend}
    if profile.gpu_only:
        from importlib.metadata import version

        metadata.update(
            jaxamg_version=version("jaxamg"),
            hierarchy_behavior="upstream-managed refresh; may refresh each application",
        )
    elif profile.requires_preparation:
        metadata["hierarchy_behavior"] = (
            "AMJax hierarchy prepared outside timing and frozen"
        )
    return metadata
