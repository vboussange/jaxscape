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


@dataclass(frozen=True)
class ToolSpec:
    label: str
    gpu_capable: bool = False
    required: bool = True
    automated: bool = True

    def display_labels(self) -> tuple[str, ...]:
        if self.gpu_capable:
            return (
                backend_tool_label(self.label, "cpu"),
                backend_tool_label(self.label, "gpu"),
            )
        return (self.label,)


SIZE_LABELS = ("small", "medium", "large")
SIZE_ORDER = {label: index for index, label in enumerate(SIZE_LABELS)}

BENCHMARK_TASK_SPECS = (
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

TOOL_SPECS_BY_TASK = {
    "resistance_distance": (
        # Resistance pinv-family profiles are kept out of the full suite because
        # the large resistance case is not a scientifically useful dense-pinv
        # benchmark target.
        ToolSpec("JAXScape / PyAMG", required=False),
        ToolSpec("JAXScape / CholmodSolver / f32", required=False, automated=False),
        ToolSpec("JAXScape / CholmodSolver / f64", required=False, automated=False),
        ToolSpec(
            "JAXScape / AMJaxCGSolver / f32",
            gpu_capable=True,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / AMJaxCGSolver / f64",
            gpu_capable=True,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / approx AMJaxCGSolver / f32",
            gpu_capable=True,
            required=False,
            automated=False,
        ),
        ToolSpec(
            "JAXScape / approx AMJaxCGSolver / f64",
            gpu_capable=True,
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
        ToolSpec("JAXScape", gpu_capable=True),
        ToolSpec("gdistance / costDistance"),
        ToolSpec("Conefor", required=False, automated=False),
    ),
    "sensitivity_analysis": (
        ToolSpec("JAXScape / shortest-path gradient", gpu_capable=True),
        ToolSpec("gdistance / shortestPath"),
        ToolSpec("JAXScape / resistance gradient", gpu_capable=True),
        ToolSpec("gdistance / passage"),
    ),
    "inverse_landscape_genetics": (
        ToolSpec("JAXScape / CholmodSolver / f32", required=False),
        ToolSpec("JAXScape / AMJaxCGSolver / f32", gpu_capable=True),
        ToolSpec("JAXScape / AMJaxCGSolver / f64", gpu_capable=True),
        ToolSpec("JAXScape / approx pinv / f32", gpu_capable=True, required=False),
        ToolSpec("JAXScape / approx pinv / f64", gpu_capable=True, required=False),
        ToolSpec("ResistanceGA"),
    ),
}

REQUIRED_BASE_TOOL_LABELS_BY_TASK = {
    task: {tool.label for tool in tool_specs if tool.required}
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

GPU_CAPABLE_TOOL_LABELS_BY_TASK = {
    task: {tool.label for tool in tool_specs if tool.gpu_capable}
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
    task: [display_label for tool in tool_specs for display_label in tool.display_labels()]
    for task, tool_specs in TOOL_SPECS_BY_TASK.items()
}

TOOL_FAMILY_COLORS = {
    "JAXScape": "#1F77B4",
    "JAXScape / pinv": "#1F77B4",
    "JAXScape / approx pinv": "#5B8FF9",
    "JAXScape / PyAMG": "#17BECF",
    "JAXScape / CholmodSolver": "#9467BD",
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