"""Shared solver and method factories for resistance-distance benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
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
    passes_state: bool = False
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


def make_cudss_solver() -> Any:
    from jaxscape.solvers import CuDSSSolver

    return CuDSSSolver()


CUDSS_RESISTANCE_PROFILES = (
    ResistanceProfile(
        "cudss_f32",
        "JAXScape / CuDSSSolver / f32",
        solver_factory=make_cudss_solver,
        gpu_capable=True,
        gpu_only=True,
        passes_state=True,
        dtype=jnp.float32,
    ),
    ResistanceProfile(
        "cudss_f64",
        "JAXScape / CuDSSSolver / f64",
        solver_factory=make_cudss_solver,
        gpu_capable=True,
        gpu_only=True,
        passes_state=True,
        dtype=jnp.float64,
    ),
)


def make_spielman_method() -> Any:
    return SpielmanApproximation(epsilon=RESISTANCE_APPROXIMATION_EPSILON, seed=0)


def profile_unavailable_reason(profile: ResistanceProfile, backend: str) -> str | None:
    """Why `profile` cannot run on `backend`, found without running a workload.

    GPU-only profiles never fall back to another solver or device, so an
    environment that lacks what they need records them as skipped.
    """
    if not profile.gpu_only:
        return None
    if backend != "gpu":
        return f"{profile.tool} runs only on an NVIDIA GPU; it has no CPU fallback."
    assert profile.solver_factory is not None
    try:
        profile.solver_factory()
    except (ImportError, RuntimeError) as error:
        return f"{profile.tool} needs an NVIDIA GPU with spineax and cuDSS: {error}"
    return None


def profile_metadata(profile: ResistanceProfile, backend: str) -> dict[str, Any]:
    """Interpretation metadata, independent of timing and native cache internals."""
    metadata = {"dtype": str(jnp.dtype(profile.dtype)), "backend": backend}
    if profile.requires_preparation:
        metadata[
            "hierarchy_behavior"
        ] = "AMJax hierarchy prepared outside timing and frozen"
    if profile.passes_state:
        metadata["solver_state_behavior"] = (
            "analysis and first factorization prepared outside timing; each "
            "call refactorizes and passes its solver state to the next call"
        )
    if profile.solver_factory is make_cudss_solver:
        metadata.update(cudss_versions())
    return metadata


def passing_state(
    fun: Callable[..., tuple[Any, Any]], state: Any
) -> Callable[..., Any]:
    """Make `fun(*args, state) -> (result, state)` a call `(*args) -> result`.

    Each call hands `fun` the state that the previous call returned, starting
    from `state`, so timing it repeatedly follows the solver state from one
    call to the next.
    """

    def call(*args: Any) -> Any:
        nonlocal state
        result, state = fun(*args, state)
        return result

    return call


# spineax 0.0.5 wheels pull the CUDA 13 runtime; CUDA 12 builds of spineax use
# the CUDA 12 one.
CUDSS_RUNTIME_PACKAGES = ("nvidia-cudss-cu13", "nvidia-cudss-cu12")


def cudss_versions() -> dict[str, str | None]:
    """Installed spineax version and cuDSS runtime package, as `name==version`."""
    cudss_version = None
    for package in CUDSS_RUNTIME_PACKAGES:
        package_version = _installed_version(package)
        if package_version is not None:
            cudss_version = f"{package}=={package_version}"
            break
    return {
        "spineax_version": _installed_version("spineax"),
        "cudss_version": cudss_version,
    }


def _installed_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None
