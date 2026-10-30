"""Shared solver and method factories for resistance-distance benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
from jaxscape import SpielmanApproximation

from benchmark.benchmark_distances import (
    BenchmarkCase,
    BenchmarkConfig,
    BenchmarkRecord,
    dependency_versions,
    gpu_placeholder_record,
    skip_record,
)
from benchmark.benchmark_registry import (
    backend_tool_label,
    CPU_ONLY,
    CUDSS_TOOL_LABELS,
    GPU_ONLY,
)


@dataclass(frozen=True)
class ResistanceProfile:
    key: str
    tool: str
    solver_factory: Callable[[], Any] | None = None
    method_factory: Callable[[], Any] | None = None
    backends: tuple[str, ...] = CPU_ONLY
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


CUDSS_RESISTANCE_PROFILES = tuple(
    ResistanceProfile(
        f"cudss_{precision}",
        label,
        solver_factory=make_cudss_solver,
        backends=GPU_ONLY,
        passes_state=True,
        dtype={"f32": jnp.float32, "f64": jnp.float64}[precision],
    )
    for precision, label in CUDSS_TOOL_LABELS.items()
)


def make_spielman_method() -> Any:
    return SpielmanApproximation(epsilon=RESISTANCE_APPROXIMATION_EPSILON, seed=0)


def profile_tool_label(profile: ResistanceProfile, backend: str) -> str:
    """The record label of `profile` on `backend`.

    **Arguments:**

    - `profile`: The benchmark profile.
    - `backend`: `"cpu"` or `"gpu"`.

    **Returns:**

    The plain tool label of a CPU-only profile on the CPU, and otherwise the
    tool label with a `(CPU)` or `(GPU)` suffix.
    """
    if backend == "cpu" and profile.backends == CPU_ONLY:
        return profile.tool
    return backend_tool_label(profile.tool, backend)


def profile_unavailable_reason(profile: ResistanceProfile, backend: str) -> str | None:
    """Return why `profile` cannot run on `backend`, without running a workload.

    GPU-only profiles never fall back to another solver or device, so an
    environment that lacks what they need records them as skipped.

    **Arguments:**

    - `profile`: The benchmark profile.
    - `backend`: `"cpu"` or `"gpu"`.

    **Returns:**

    The reason, or `None` when the profile can run.
    """
    if profile.backends != GPU_ONLY:
        return None
    if backend != "gpu":
        return f"{profile.tool} runs only on an NVIDIA GPU; it has no CPU fallback."
    assert profile.solver_factory is not None
    try:
        profile.solver_factory()
    except (ImportError, RuntimeError) as error:
        return f"{profile.tool} needs an NVIDIA GPU with spineax and cuDSS: {error}"
    return None


def unavailable_profile_record(
    case: BenchmarkCase, profile: ResistanceProfile, backend: str, software: str
) -> BenchmarkRecord | None:
    """Return the skipped record of a profile that cannot run on `backend`.

    **Arguments:**

    - `case`: The benchmark case.
    - `profile`: The benchmark profile.
    - `backend`: `"cpu"` or `"gpu"`.
    - `software`: The software column of the record.

    **Returns:**

    A skipped record naming what is missing, or `None` when the profile can
    run.
    """
    reason = profile_unavailable_reason(profile, backend)
    if reason is None:
        return None
    label = profile_tool_label(profile, backend)
    return skip_record(case.task, label, software, case.name, reason)


def collect_profile_records(
    case: BenchmarkCase,
    config: BenchmarkConfig,
    profile: ResistanceProfile,
    software: str,
    run: Callable[[str, Any], BenchmarkRecord],
) -> list[BenchmarkRecord]:
    """Run `profile` on each of its backends.

    **Arguments:**

    - `case`: The benchmark case.
    - `config`: The benchmark configuration, with its devices.
    - `profile`: The benchmark profile.
    - `software`: The software column of placeholder records.
    - `run`: Runs the profile, given its record label and device.

    **Returns:**

    One record per backend of the profile. Without a GPU, the GPU record is a
    skipped placeholder.
    """
    records = []
    for backend in profile.backends:
        if backend == "gpu" and config.gpu_device is None:
            records.append(gpu_placeholder_record(case, profile.tool, software))
            continue
        device = config.gpu_device if backend == "gpu" else config.cpu_device
        records.append(run(profile_tool_label(profile, backend), device))
    return records


def profile_metadata(profile: ResistanceProfile, backend: str) -> dict[str, Any]:
    """Return interpretation metadata, independent of timing and native caches.

    **Arguments:**

    - `profile`: The benchmark profile.
    - `backend`: `"cpu"` or `"gpu"`.

    **Returns:**

    The dtype and backend, how solver state is prepared, and the spineax and
    cuDSS versions of cuDSS profiles.
    """
    metadata: dict[str, Any] = {
        "dtype": str(jnp.dtype(profile.dtype)),
        "backend": backend,
    }
    if profile.requires_preparation:
        metadata[
            "hierarchy_behavior"
        ] = "AMJax hierarchy prepared outside timing and frozen"
    if profile.passes_state:
        metadata["solver_state_behavior"] = (
            "analysis and first factorization run outside timing; each timed "
            "call starts from the solver state the previous call returned; "
            "spineax_rebuilds counts the analyses spineax repeated"
        )
    if profile.solver_factory is make_cudss_solver:
        metadata.update(cudss_versions())
    return metadata


def stateful_closure(
    fun: Callable[..., tuple[Any, Any]], state: Any
) -> Callable[..., Any]:
    """Return a closure `(*args) -> result` that carries the state of `fun`.

    **Arguments:**

    - `fun`: A function `fun(*args, state) -> (result, state)`.
    - `state`: The state of the first call.

    **Returns:**

    A closure that calls `fun` with the state the previous call returned, so
    timing it repeatedly follows the solver state from one call to the next.
    """

    def call(*args: Any) -> Any:
        nonlocal state
        result, state = fun(*args, state)
        return result

    return call


def spineax_rebuild_count() -> int:
    """Return spineax's count of factorizations it rebuilt by a fresh analysis.

    spineax rebuilds a factorization when a phase receives a token that is no
    longer resident, e.g. a used-up one. The count is process-wide.
    """
    from jaxscape.solvers.cudsssolver import _load_cudss

    return int(_load_cudss().rebuild_count())


# spineax 0.0.5 wheels pull the CUDA 13 runtime; CUDA 12 builds of spineax use
# the CUDA 12 one.
CUDSS_RUNTIME_PACKAGES = ("nvidia-cudss-cu13", "nvidia-cudss-cu12")


def cudss_versions() -> dict[str, str | None]:
    """Return the installed spineax version and cuDSS runtime package.

    **Returns:**

    `spineax_version`, and `cudss_version` as `name==version` of the first
    installed cuDSS runtime package, each `None` when not installed.
    """
    versions = dependency_versions()
    cudss_version = next(
        (
            f"{package}=={versions[package]}"
            for package in CUDSS_RUNTIME_PACKAGES
            if versions[package] is not None
        ),
        None,
    )
    return {"spineax_version": versions["spineax"], "cudss_version": cudss_version}
