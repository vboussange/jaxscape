"""Render one benchmark scorecard per feature from benchmark/results/benchmark_results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "benchmark" / "results" / "benchmark_results.json"
SIZE_ORDER = {"small": 0, "medium": 1, "large": 2}
OUTPUTS = {
    "resistance_distance": ROOT / "docs" / "assets" / "benchmark_resistance_distance.png",
    "least_cost_path": ROOT / "docs" / "assets" / "benchmark_least_cost_path.png",
    "sensitivity_analysis": ROOT / "docs" / "assets" / "benchmark_sensitivity_analysis.png",
    "inverse_landscape_genetics": ROOT / "docs" / "assets" / "benchmark_inverse_landscape_genetics.png",
}
TASK_LABELS = {
    "resistance_distance": "Resistance distance",
    "least_cost_path": "Least-cost path",
    "sensitivity_analysis": "Centrality and gradient sensitivity",
    "inverse_landscape_genetics": "Inverse landscape genetics",
}
DISPLAY_ORDER = {
    "resistance_distance": [
        "JAXScape / pinv (CPU)",
        "JAXScape / pinv (GPU)",
        "JAXScape / PyAMG",
        "JAXScape / CholmodSolver",
        "gdistance / commuteDistance",
        "Circuitscape.jl / cg+amg",
        "Circuitscape.jl / cholmod",
        "Circuitscape.jl / accelerate",
        "Conefor",
    ],
    "least_cost_path": ["JAXScape (CPU)", "JAXScape (GPU)", "gdistance / costDistance", "Conefor"],
    "sensitivity_analysis": [
        "JAXScape / shortest-path gradient (CPU)",
        "JAXScape / shortest-path gradient (GPU)",
        "gdistance / shortestPath",
        "JAXScape / resistance gradient (CPU)",
        "JAXScape / resistance gradient (GPU)",
        "gdistance / passage",
    ],
    "inverse_landscape_genetics": ["JAXScape + Optimistix (CPU)", "JAXScape + Optimistix (GPU)", "ResistanceGA"],
}
TOOL_COLORS = {
    "JAXScape / pinv": "#1F77B4",
    "JAXScape / PyAMG": "#17BECF",
    "JAXScape / CholmodSolver": "#9467BD",
    "gdistance / commuteDistance": "#FF7F0E",
    "Circuitscape.jl / cg+amg": "#2CA02C",
    "Circuitscape.jl / cholmod": "#8C564B",
    "Circuitscape.jl / accelerate": "#BCBD22",
    "JAXScape": "#1F77B4",
    "gdistance / costDistance": "#FF7F0E",
    "JAXScape / shortest-path gradient": "#1F77B4",
    "gdistance / shortestPath": "#FF7F0E",
    "JAXScape / resistance gradient": "#D62728",
    "gdistance / passage": "#2CA02C",
    "JAXScape + Optimistix": "#C44E52",
    "ResistanceGA": "#E377C2",
    "Conefor": "#6C757D",
}
BAR_HATCHES = {
    "cpu": "",
    "gpu": "//",
}
DEFAULT_COLOR = "#6B7280"


def load_payload() -> dict:
    return json.loads(RESULTS_JSON.read_text())


def case_lookup(payload: dict) -> dict[str, dict]:
    if payload.get("cases"):
        return {case["name"]: case for case in payload["cases"]}

    inferred: dict[str, dict] = {}
    for record in payload["records"]:
        inferred.setdefault(
            record["scenario"],
            {
                "name": record["scenario"],
                "task": record["task"],
                "size_label": record["scenario"],
            },
        )
    return inferred


def format_runtime(seconds: float) -> str:
    if seconds < 1e-2:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def format_error(value: float) -> str:
    return f"{value:.3f}"


def ordered_cases(task: str, cases: dict[str, dict]) -> list[dict]:
    task_cases = [case for case in cases.values() if case["task"] == task]
    return sorted(
        task_cases,
        key=lambda case: (SIZE_ORDER.get(case.get("size_label", ""), len(SIZE_ORDER)), case.get("grid_size", 0)),
    )


def case_tick_label(case: dict) -> str:
    size = case.get("grid_size")
    label = str(case.get("size_label", case["name"])).replace("_", " ").title()
    if size is None:
        return label
    return f"{label}\n{size}x{size}"


def record_index(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {(record["tool"], record["scenario"]): record for record in records}


def display_tools(task: str, records: list[dict]) -> list[str]:
    available = {record["tool"] for record in records}
    ordered = [tool for tool in DISPLAY_ORDER.get(task, []) if tool in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def metric_value(record: dict, key: str) -> float | None:
    metrics = record.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def convergence_value(record: dict) -> bool | None:
    metrics = record.get("metrics") or {}
    value = metrics.get("converged")
    if value is None:
        return None
    return bool(value)


def tool_family(tool: str) -> str:
    if tool.endswith(" (CPU)") or tool.endswith(" (GPU)"):
        return tool.rsplit(" (", 1)[0]
    return tool


def tool_backend(tool: str) -> str | None:
    if tool.endswith(" (CPU)"):
        return "cpu"
    if tool.endswith(" (GPU)"):
        return "gpu"
    return None


def skipped_placeholder_records(task: str, records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record["task"] == task and record.get("status") == "skipped" and "placeholder" in (record.get("note") or "").lower()
    ]


def legend_columns(tool_count: int) -> int:
    if tool_count > 6:
        return 3
    if tool_count > 3:
        return 2
    return 1


def plot_grouped_bars(
    axis: plt.Axes,
    *,
    task: str,
    cases: list[dict],
    records: list[dict],
    value_getter,
    formatter,
    ylabel: str,
    use_log_scale: bool,
    show_convergence: bool,
    tool_subset: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
) -> list[str]:
    indexed = record_index(records)
    candidate_tools = tool_subset or display_tools(task, records)
    visible_tools = [
        tool
        for tool in candidate_tools
        if any(value_getter(indexed[(tool, case["name"])]) is not None for case in cases if (tool, case["name"]) in indexed)
    ]
    x_positions = np.arange(len(cases), dtype=float)
    tick_labels = [case_tick_label(case) for case in cases]

    if not visible_tools:
        axis.text(0.5, 0.5, "No compatible benchmark data", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return []

    width = min(0.84 / max(len(visible_tools), 1), 0.22)
    for tool_index, tool in enumerate(visible_tools):
        offset = (tool_index - (len(visible_tools) - 1) / 2) * width
        base_color = TOOL_COLORS.get(tool_family(tool), DEFAULT_COLOR)
        backend = tool_backend(tool)
        label_added = False

        for case_index, case in enumerate(cases):
            record = indexed.get((tool, case["name"]))
            if record is None:
                continue
            y_value = value_getter(record)
            if y_value is None:
                continue

            converged = convergence_value(record) if show_convergence else None
            hatch = BAR_HATCHES.get(backend, "")
            facecolor = base_color
            edgecolor = base_color
            if converged is False:
                facecolor = "white"
                hatch = f"{hatch}xx" if hatch else "xx"

            x_value = x_positions[case_index] + offset
            axis.bar(
                x_value,
                float(y_value),
                width=width * 0.92,
                color=facecolor,
                edgecolor=edgecolor,
                linewidth=1.2,
                hatch=hatch,
                label=tool if not label_added else None,
                zorder=3,
            )
            label_added = True

            axis.annotate(
                formatter(float(y_value)),
                (x_value, float(y_value)),
                textcoords="offset points",
                xytext=(0, 4 if not use_log_scale else 6),
                ha="center",
                fontsize=8,
            )

    if use_log_scale:
        axis.set_yscale("log")
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_positions, tick_labels)
    axis.grid(axis="y", alpha=0.2, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    return visible_tools


def add_placeholder_note(fig: plt.Figure, task: str, records: list[dict]) -> None:
    placeholders = skipped_placeholder_records(task, records)
    if not placeholders:
        return
    placeholder_tools = ", ".join(sorted({record["tool"] for record in placeholders}))
    fig.text(
        0.01,
        0.01,
        f"Placeholder only in this artifact: {placeholder_tools}.",
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )


def render_runtime_chart(task: str, records: list[dict], output_path: Path, cases: dict[str, dict]) -> None:
    task_cases = ordered_cases(task, cases)
    fig, axis = plt.subplots(figsize=(9.2, 5.2))
    fig.suptitle(f"{TASK_LABELS[task]} benchmark")
    visible_tools = plot_grouped_bars(
        axis,
        task=task,
        cases=task_cases,
        records=records,
        value_getter=lambda record: record.get("median_seconds"),
        formatter=format_runtime,
        ylabel="Median runtime (seconds)",
        use_log_scale=True,
        show_convergence=False,
    )
    axis.set_xlabel("Problem size")
    if visible_tools:
        axis.legend(frameon=False, loc="upper left", ncol=legend_columns(len(visible_tools)))
    add_placeholder_note(fig, task, records)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_centrality_chart(task: str, records: list[dict], output_path: Path, cases: dict[str, dict]) -> None:
    task_cases = ordered_cases(task, cases)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.0]})
    fig.suptitle(f"{TASK_LABELS[task]} benchmark")

    visible_tools = plot_grouped_bars(
        axes[0],
        task=task,
        cases=task_cases,
        records=records,
        value_getter=lambda record: record.get("median_seconds"),
        formatter=format_runtime,
        ylabel="Median runtime (seconds)",
        use_log_scale=True,
        show_convergence=False,
    )
    if visible_tools:
        axes[0].legend(frameon=False, loc="upper left", ncol=legend_columns(len(visible_tools)))

    alignment_tools = [
        tool
        for tool in display_tools(task, records)
        if any(metric_value(record, "cosine_similarity") is not None for record in records if record["tool"] == tool)
    ]
    plot_grouped_bars(
        axes[1],
        task=task,
        cases=task_cases,
        records=records,
        value_getter=lambda record: metric_value(record, "cosine_similarity"),
        formatter=format_error,
        ylabel="Cosine similarity\nvs. gdistance",
        use_log_scale=False,
        show_convergence=False,
        tool_subset=alignment_tools,
        ylim=(0.0, 1.05),
    )
    axes[1].set_xlabel("Problem size")

    add_placeholder_note(fig, task, records)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_inverse_chart(task: str, records: list[dict], output_path: Path, cases: dict[str, dict]) -> None:
    task_cases = ordered_cases(task, cases)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 7.4), sharex=True, gridspec_kw={"height_ratios": [1.35, 1.0]})
    fig.suptitle(f"{TASK_LABELS[task]} benchmark")

    visible_tools = plot_grouped_bars(
        axes[0],
        task=task,
        cases=task_cases,
        records=records,
        value_getter=lambda record: record.get("median_seconds"),
        formatter=format_runtime,
        ylabel="Median runtime (seconds)",
        use_log_scale=True,
        show_convergence=True,
    )
    if visible_tools:
        axes[0].legend(frameon=False, loc="upper left", ncol=legend_columns(len(visible_tools)))

    plot_grouped_bars(
        axes[1],
        task=task,
        cases=task_cases,
        records=records,
        value_getter=lambda record: metric_value(record, "relative_rmse"),
        formatter=format_error,
        ylabel="Final relative RMSE",
        use_log_scale=False,
        show_convergence=True,
    )
    axes[1].set_xlabel("Problem size")
    axes[1].legend(
        handles=[
            Patch(facecolor="#111827", edgecolor="#111827", label="converged"),
            Patch(facecolor="white", edgecolor="#111827", hatch="xx", label="budget hit"),
        ],
        frameon=False,
        loc="upper right",
        title="Inverse status",
    )

    add_placeholder_note(fig, task, records)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def render_task(task: str, records: list[dict], output_path: Path, cases: dict[str, dict]) -> None:
    task_cases = ordered_cases(task, cases)
    if not records and not task_cases:
        fig, axis = plt.subplots(figsize=(8, 2.4))
        axis.text(0.5, 0.5, "No compatible benchmark data", ha="center", va="center", fontsize=11)
        axis.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    if task == "inverse_landscape_genetics":
        render_inverse_chart(task, records, output_path, cases)
        return
    if task == "sensitivity_analysis":
        render_centrality_chart(task, records, output_path, cases)
        return

    render_runtime_chart(task, records, output_path, cases)


def render() -> None:
    payload = load_payload()
    cases = case_lookup(payload)
    grouped: dict[str, list[dict]] = {task: [] for task in OUTPUTS}
    for record in payload["records"]:
        if record["task"] in grouped:
            grouped[record["task"]].append(record)

    for task, output_path in OUTPUTS.items():
        renderable_records = [
            record
            for record in grouped.get(task, [])
            if record["status"] == "ok" and record.get("median_seconds") is not None
        ] + skipped_placeholder_records(task, grouped.get(task, []))
        render_task(task, renderable_records, output_path, cases)


if __name__ == "__main__":
    render()