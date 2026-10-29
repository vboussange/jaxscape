"""Shared benchmark metadata and tool registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def backend_tool_label(tool: str, backend: str) -> str:
    return f"{tool} ({backend.upper()})"


def tool_family(tool: str) -> str:
    family = tool
    if family.endswith(" (CPU)") or family.endswith(" (GPU)"):
        family = family.rsplit(" (", 1)[0]
    if family.endswith(" / f32") or family.endswith(" / f64"):
        family = family.rsplit(" / ", 1)[0]
    return family


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    group_name: str
    task: str
    title: str
    size_by_label: dict[str, int]
    seed_base: int
    scorecard_asset_name: str


# The backends a tool runs on. A CPU-only tool keeps its plain label; the
# others carry a "(CPU)" or "(GPU)" suffix for each backend.
CPU_ONLY = ("cpu",)
CPU_AND_GPU = ("cpu", "gpu")
GPU_ONLY = ("gpu",)


def backend_labels(label: str, backends: tuple[str, ...]) -> dict[str, str]:
    """Return the record label of a tool on each of its backends.

    **Arguments:**

    - `label`: The tool label.
    - `backends`: The backends the tool runs on.

    **Returns:**

    A map from each backend to its label: the plain label for a CPU-only
    tool, and otherwise the label with a `(CPU)` or `(GPU)` suffix.
    """
    if backends == CPU_ONLY:
        return {"cpu": label}
    return {backend: backend_tool_label(label, backend) for backend in backends}


@dataclass(frozen=True)
class ToolSpec:
    label: str
    backends: tuple[str, ...] = CPU_ONLY
    required: bool = True
    automated: bool = True

    def display_labels(self) -> tuple[str, ...]:
        return tuple(backend_labels(self.label, self.backends).values())


SIZE_LABELS = ("small", "medium", "large")
SIZE_ORDER = {label: index for index, label in enumerate(SIZE_LABELS)}

BENCHMARK_TASK_SPECS = (
    BenchmarkTaskSpec(
        group_name="free_energy",
        task="free_energy_distance",
        title="Free-energy distance",
        size_by_label={"small": 10, "medium": 100, "large": 1000},
        seed_base=0,
        scorecard_asset_name="benchmark_free_energy_distance.png",
    ),
    BenchmarkTaskSpec(
        group_name="resistance",
        task="resistance_distance",
        title="Resistance distance",
        size_by_label={"small": 10, "medium": 100, "large": 1000},
        seed_base=0,
        scorecard_asset_name="benchmark_resistance_distance.png",
    ),
    BenchmarkTaskSpec(
        group_name="lcp",
        task="least_cost_path",
        title="Least-cost path",
        size_by_label={"small": 10, "medium": 100, "large": 1000},
        seed_base=10,
        scorecard_asset_name="benchmark_least_cost_path.png",
    ),
    BenchmarkTaskSpec(
        group_name="sensitivity",
        task="sensitivity_analysis",
        title="Centrality and gradient sensitivity",
        size_by_label={"small": 10, "medium": 100, "large": 175},
        seed_base=20,
        scorecard_asset_name="benchmark_sensitivity_analysis.png",
    ),
    BenchmarkTaskSpec(
        group_name="inverse",
        task="inverse_landscape_genetics",
        title="Inverse landscape genetics",
        size_by_label={"medium": 100},
        seed_base=30,
        scorecard_asset_name="benchmark_inverse_landscape_genetics.png",
    ),
)

TASK_SPECS_BY_GROUP = {spec.group_name: spec for spec in BENCHMARK_TASK_SPECS}
TASK_SPECS_BY_TASK = {spec.task: spec for spec in BENCHMARK_TASK_SPECS}
TASK_LABELS = {spec.task: spec.title for spec in BENCHMARK_TASK_SPECS}

# cuDSS runs only on NVIDIA GPUs, so these profiles have no CPU series.
CUDSS_TOOL_LABELS = {
    "f32": "JAXScape / CuDSSSolver / f32",
    "f64": "JAXScape / CuDSSSolver / f64",
}
CUDSS_TOOL_SPECS = tuple(
    ToolSpec(label, backends=GPU_ONLY, required=False, automated=False)
    for label in CUDSS_TOOL_LABELS.values()
)

TOOL_SPECS_BY_TASK = {
    "free_energy_distance": (ToolSpec("JAXScape / CG / f64", backends=CPU_AND_GPU),),
    "resistance_distance": (
        # Resistance pinv-family profiles are kept out of the full suite because
        # the large resistance case is not a scientifically useful dense-pinv
        # benchmark target.
        ToolSpec("JAXScape / PyAMG", required=False),
        ToolSpec("JAXScape / CholmodSolver / f32", required=False, automated=False),
        ToolSpec("JAXScape / CholmodSolver / f64", required=False, automated=False),
        *CUDSS_TOOL_SPECS,
        ToolSpec(
            "JAXScape / AMJaxCGSolver / f32",
            backends=CPU_AND_GPU,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / AMJaxCGSolver / f64",
            backends=CPU_AND_GPU,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / approx AMJaxCGSolver / f32",
            backends=CPU_AND_GPU,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / approx AMJaxCGSolver / f64",
            backends=CPU_AND_GPU,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / approx CholmodSolver / f32",
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / approx CholmodSolver / f64",
            required=False,
            automated=False,
        ),
        ToolSpec("gdistance / commuteDistance", required=False),
        ToolSpec("Circuitscape.jl / cg+amg / f64"),
        ToolSpec("Circuitscape.jl / cholmod / f64"),
        ToolSpec("Conefor", required=False, automated=False),
    ),
    "least_cost_path": (
        ToolSpec("JAXScape", backends=CPU_AND_GPU),
        ToolSpec("gdistance / costDistance"),
        ToolSpec("Conefor", required=False, automated=False),
    ),
    "sensitivity_analysis": (
        ToolSpec("JAXScape / shortest-path gradient", backends=CPU_AND_GPU),
        ToolSpec("gdistance / shortestPath"),
        ToolSpec("JAXScape / resistance gradient", backends=CPU_AND_GPU),
        ToolSpec("gdistance / passage"),
    ),
    "inverse_landscape_genetics": (
        ToolSpec("JAXScape / CholmodSolver / f32", required=False),
        *CUDSS_TOOL_SPECS,
        ToolSpec("JAXScape / AMJaxCGSolver / f32", backends=CPU_AND_GPU),
        ToolSpec("JAXScape / AMJaxCGSolver / f64", backends=CPU_AND_GPU),
        ToolSpec("JAXScape / approx pinv / f32", backends=CPU_AND_GPU, required=False),
        ToolSpec("JAXScape / approx pinv / f64", backends=CPU_AND_GPU, required=False),
        ToolSpec("ResistanceGA"),
    ),
}

REQUIRED_BASE_TOOL_LABELS_BY_TASK = {
    task: {tool.label for tool in tool_specs if tool.required}
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

GPU_CAPABLE_TOOL_LABELS_BY_TASK = {
    task: {tool.label for tool in tool_specs if "gpu" in tool.backends}
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

AUTOMATED_BASE_TOOL_LABELS_BY_TASK = {
    task: {tool.label for tool in tool_specs if tool.automated}
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

OPTIONAL_BASE_TOOL_LABELS_BY_TASK = {
    task: {tool.label for tool in tool_specs if not tool.automated}
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

DISPLAY_ORDER_BY_TASK = {
    task: [
        display_label for tool in tool_specs for display_label in tool.display_labels()
    ]
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

TOOL_FAMILY_COLORS = {
    "JAXScape": "#1F77B4",
    "JAXScape / pinv": "#1F77B4",
    "JAXScape / approx pinv": "#5B8FF9",
    "JAXScape / PyAMG": "#17BECF",
    "JAXScape / CholmodSolver": "#9467BD",
    "JAXScape / CuDSSSolver": "#E45756",
    "JAXScape / AMJaxCGSolver": "#4C78A8",
    "JAXScape / approx AMJaxCGSolver": "#72B7B2",
    "JAXScape / approx CholmodSolver": "#B279A2",
    "JAXScape / shortest-path gradient": "#1F77B4",
    "JAXScape / resistance gradient": "#D62728",
    "gdistance / commuteDistance": "#F28E2B",
    "gdistance / costDistance": "#F28E2B",
    "gdistance / shortestPath": "#F28E2B",
    "gdistance / passage": "#59A14F",
    "Circuitscape.jl / cg+amg": "#59A14F",
    "Circuitscape.jl / cholmod": "#8C564B",
    "ResistanceGA": "#E377C2",
    "Conefor": "#6C757D",
}


def benchmark_scorecard_outputs(root: Path) -> dict[str, Path]:
    docs_assets = root / "docs" / "assets"
    return {
        spec.task: docs_assets / spec.scorecard_asset_name
        for spec in BENCHMARK_TASK_SPECS
    }
