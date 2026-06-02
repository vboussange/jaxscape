"""Render benchmark scorecards from benchmark/results/benchmark_results.json."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

from benchmark.benchmark_registry import (
    benchmark_scorecard_outputs,
    DISPLAY_ORDER_BY_TASK,
    SIZE_ORDER,
    TASK_LABELS,
    tool_family,
)


DEFAULT_RESULTS_JSON = ROOT / "benchmark" / "results" / "benchmark_results.json"
DEFAULT_OUTPUTS = benchmark_scorecard_outputs(ROOT)
BAR_HATCHES = {
    "cpu": "..",
    "gpu": "////",
}
DEFAULT_COLOR = "#6B7280"
SOFTWARE_COLORS = {
    "JAXScape": "#0072B2",
    "gdistance": "#E69F00",
    "Circuitscape.jl": "#009E73",
    "ResistanceGA": "#CC79A7",
    "Conefor": "#111827",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "assets",
        help="Directory where benchmark_*.png scorecards are written.",
    )
    return parser.parse_args()


def scorecard_outputs(output_dir: Path) -> dict[str, Path]:
    return {
        task: output_dir / default_output.name
        for task, default_output in DEFAULT_OUTPUTS.items()
    }


def load_payload(results_json: Path) -> dict:
    return json.loads(results_json.read_text())


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


def format_loss(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e3:
        return f"{value:.2e}"
    return f"{value:.4f}" if abs(value) < 0.1 else f"{value:.3f}"


def ordered_cases(task: str, cases: dict[str, dict]) -> list[dict]:
    task_cases = [case for case in cases.values() if case["task"] == task]
    return sorted(
        task_cases,
        key=lambda case: (
            SIZE_ORDER.get(case.get("size_label", ""), len(SIZE_ORDER)),
            case.get("grid_size", 0),
        ),
    )


def case_tick_label(case: dict) -> str:
    size = case.get("grid_size")
    if size is None:
        return str(case.get("name", ""))
    return f"{size}x{size}"


def record_index(records: list[dict]) -> dict[tuple[str, str], dict]:
    return {(record["tool"], record["scenario"]): record for record in records}


def display_tools(task: str, records: list[dict]) -> list[str]:
    available = {record["tool"] for record in records}
    ordered = [tool for tool in DISPLAY_ORDER_BY_TASK.get(task, []) if tool in available]
    extras = sorted(available - set(ordered))
    return ordered + extras


def metric_value(record: dict, key: str) -> float | None:
    metrics = record.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def inverse_loss_value(record: dict) -> float | None:
    final_mse = metric_value(record, "final_mse")
    if final_mse is not None:
        return final_mse
    rmse = metric_value(record, "rmse")
    if rmse is None:
        return None
    return rmse**2


def should_use_log_scale(values: list[float]) -> bool:
    positive_values = [value for value in values if value > 0]
    if len(positive_values) < 2:
        return False
    return max(positive_values) / min(positive_values) >= 20


def metric_values_for_case(
    case: dict,
    records: list[dict],
    value_getter,
    *,
    tool_subset: list[str] | None = None,
) -> list[float]:
    indexed = record_index(records)
    candidate_tools = tool_subset or display_tools(case["task"], records)
    values: list[float] = []
    for tool in candidate_tools:
        record = indexed.get((tool, case["name"]))
        if record is None:
            continue
        value = value_getter(record)
        if value is None:
            continue
        values.append(float(value))
    return values


def convergence_value(record: dict) -> bool | None:
    metrics = record.get("metrics") or {}
    value = metrics.get("converged")
    if value is None:
        return None
    return bool(value)


def tool_backend(tool: str) -> str | None:
    if tool.endswith(" (CPU)"):
        return "cpu"
    if tool.endswith(" (GPU)"):
        return "gpu"
    return None


def tool_software(tool: str) -> str:
    if tool.startswith("JAXScape"):
        return "JAXScape"
    if tool.startswith("gdistance"):
        return "gdistance"
    if tool.startswith("Circuitscape.jl"):
        return "Circuitscape.jl"
    if tool.startswith("ResistanceGA"):
        return "ResistanceGA"
    if tool.startswith("Conefor"):
        return "Conefor"
    return tool_family(tool)


def tool_color(tool: str, visible_tools: list[str]) -> tuple[float, float, float] | str:
    del visible_tools
    software = tool_software(tool)
    return SOFTWARE_COLORS.get(software, DEFAULT_COLOR)


def config_tick_label(tool: str) -> str:
    backend = tool_backend(tool)
    label = tool
    if backend is not None:
        label = label.rsplit(" (", 1)[0]
    if label.startswith("JAXScape / "):
        label = label.removeprefix("JAXScape / ")
    elif label == "JAXScape":
        label = "default"
    if backend is not None:
        label = f"{label}\n{backend.upper()}"
    return label.replace(" / ", "\n")


def skipped_placeholder_records(task: str, records: list[dict]) -> list[dict]:
    return [
        record
        for record in records
        if record["task"] == task
        and record.get("status") == "skipped"
        and "placeholder" in (record.get("note") or "").lower()
    ]


def panel_title(case: dict) -> str:
    size_label = str(case.get("size_label", "")).replace("_", " ").title()
    grid_label = case_tick_label(case)
    if size_label:
        return f"{size_label} grid ({grid_label})"
    return grid_label


def legend_columns(tool_count: int) -> int:
    if tool_count > 9:
        return 4
    if tool_count > 6:
        return 3
    if tool_count > 3:
        return 2
    return 1


def legend_rows(tool_count: int) -> int:
    return max(1, math.ceil(tool_count / legend_columns(tool_count)))


def chart_size(
    tool_count: int,
    case_count: int,
    *,
    columns: int,
    row_height: float,
) -> tuple[float, float]:
    panel_width = min(max(8.8, 0.36 * max(tool_count, 1) + 5.6), 13.2)
    width = panel_width * columns
    height = row_height * max(case_count, 1) + 0.5 * max(legend_rows(tool_count) - 1, 0)
    return width, height


def layout_rect_top(tool_count: int) -> float:
    return max(0.82, 0.92 - 0.04 * max(legend_rows(tool_count) - 1, 0))


def add_style_legend(
    fig: plt.Figure,
    records: list[dict],
    *,
    include_inverse_status: bool = False,
) -> float:
    softwares = [
        software
        for software in SOFTWARE_COLORS
        if any(tool_software(record["tool"]) == software for record in records)
    ]
    handles = [
        Patch(facecolor=SOFTWARE_COLORS[software], edgecolor=SOFTWARE_COLORS[software], label=software)
        for software in softwares
    ]
    handles.extend(
        [
            Patch(facecolor="#E5E7EB", edgecolor="#111827", hatch=BAR_HATCHES["cpu"], label="CPU"),
            Patch(facecolor="#E5E7EB", edgecolor="#111827", hatch=BAR_HATCHES["gpu"], label="GPU"),
        ]
    )
    if include_inverse_status:
        handles.extend(
            [
                Patch(
                    facecolor="#F9FAFB",
                    edgecolor="#166534",
                    linewidth=2.0,
                    label="Converged",
                ),
                Patch(
                    facecolor="#F9FAFB",
                    edgecolor="#B91C1C",
                    linewidth=2.2,
                    label="Budget hit",
                ),
            ]
        )
    if not handles:
        return 0.95
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(handles), 7),
    )
    return layout_rect_top(len(handles))


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
    show_group_labels: bool = True,
) -> list[str]:
    indexed = record_index(records)
    candidate_tools = tool_subset or display_tools(task, records)
    visible_tools = [
        tool
        for tool in candidate_tools
        if any(value_getter(record) is not None for record in records if record["tool"] == tool)
    ]

    if not visible_tools:
        axis.text(
            0.5,
            0.5,
            "No compatible benchmark data",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return []

    width = 0.72
    annotation_fontsize = 8 if len(visible_tools) <= 8 else 7 if len(visible_tools) <= 16 else 6
    x_ticks: list[float] = []
    x_tick_labels: list[str] = []
    group_centers: list[tuple[float, str]] = []
    x_cursor = 0.0
    for case in cases:
        group_positions: list[float] = []
        for tool in visible_tools:
            record = indexed.get((tool, case["name"]))
            x_value = x_cursor
            group_positions.append(x_value)
            x_ticks.append(x_value)
            x_tick_labels.append(config_tick_label(tool))

            if record is not None:
                y_value = value_getter(record)
                if y_value is not None:
                    converged = convergence_value(record) if show_convergence else None
                    facecolor = tool_color(tool, visible_tools)
                    edgecolor = "#1F2937"
                    linewidth = 1.2
                    if show_convergence and converged is True:
                        edgecolor = "#166534"
                        linewidth = 2.0
                    elif show_convergence and converged is False:
                        edgecolor = "#B91C1C"
                        linewidth = 2.2

                    axis.bar(
                        x_value,
                        float(y_value),
                        width=width,
                        color=facecolor,
                        edgecolor=edgecolor,
                        linewidth=linewidth,
                        hatch=BAR_HATCHES.get(tool_backend(tool), ""),
                        zorder=3,
                    )

                    axis.annotate(
                        formatter(float(y_value)),
                        (x_value, float(y_value)),
                        textcoords="offset points",
                        xytext=(0, 4 if not use_log_scale else 6),
                        ha="center",
                        fontsize=annotation_fontsize,
                        clip_on=False,
                        zorder=6,
                    )
            x_cursor += 1.0

        if group_positions and show_group_labels:
            group_centers.append((float(np.mean(group_positions)), case_tick_label(case)))
        if group_positions:
            x_cursor += 1.35

    if use_log_scale:
        axis.set_yscale("log")
    if ylim is not None:
        axis.set_ylim(*ylim)
    else:
        axis.margins(y=0.28)
    axis.set_ylabel(ylabel)
    axis.set_xticks(x_ticks, x_tick_labels)
    axis.tick_params(axis="x", rotation=46, labelsize=8)
    for label in axis.get_xticklabels():
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    if show_group_labels:
        for center, label in group_centers:
            axis.text(
                center,
                -0.34,
                label,
                ha="center",
                va="top",
                transform=axis.get_xaxis_transform(),
                fontsize=9,
                fontweight="bold",
                clip_on=False,
            )
    if x_ticks:
        axis.set_xlim(min(x_ticks) - 0.8, max(x_ticks) + 0.8)
    axis.grid(axis="y", alpha=0.2, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_axisbelow(True)
    return visible_tools


def save_scorecard_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


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


def render_runtime_chart(
    task: str, records: list[dict], output_path: Path, cases: dict[str, dict]
) -> None:
    task_cases = ordered_cases(task, cases)
    tool_count = len(display_tools(task, records))
    fig, axes = plt.subplots(
        len(task_cases),
        1,
        squeeze=False,
        figsize=chart_size(tool_count, len(task_cases), columns=1, row_height=4.3),
    )
    fig.suptitle(f"{TASK_LABELS[task]} benchmark")
    for axis, case in zip(axes[:, 0], task_cases, strict=False):
        visible_tools = plot_grouped_bars(
            axis,
            task=task,
            cases=[case],
            records=records,
            value_getter=lambda record: record.get("median_seconds"),
            formatter=format_runtime,
            ylabel="Median runtime (seconds)",
            use_log_scale=should_use_log_scale(
                metric_values_for_case(case, records, lambda record: record.get("median_seconds"))
            ),
            show_convergence=False,
            show_group_labels=False,
        )
        axis.set_title(panel_title(case), loc="left", fontsize=11, fontweight="bold", pad=10)
        if axis is axes[-1, 0]:
            axis.set_xlabel("Configuration (CPU/GPU shown in labels)", labelpad=12)
    top = add_style_legend(fig, records if visible_tools else []) if task_cases else 0.95
    add_placeholder_note(fig, task, records)
    fig.tight_layout(rect=(0, 0.05, 1, top))
    save_scorecard_figure(fig, output_path)
    plt.close(fig)


def render_centrality_chart(
    task: str, records: list[dict], output_path: Path, cases: dict[str, dict]
) -> None:
    task_cases = ordered_cases(task, cases)
    tool_count = len(display_tools(task, records))
    fig, axes = plt.subplots(
        len(task_cases),
        2,
        figsize=chart_size(tool_count, len(task_cases), columns=2, row_height=4.8),
        sharex=False,
        squeeze=False,
    )
    fig.suptitle(f"{TASK_LABELS[task]} benchmark")

    alignment_tools = [
        tool
        for tool in display_tools(task, records)
        if any(
            metric_value(record, "cosine_similarity") is not None
            for record in records
            if record["tool"] == tool
        )
    ]
    visible_tools: list[str] = []
    for row_index, case in enumerate(task_cases):
        runtime_axis = axes[row_index, 0]
        similarity_axis = axes[row_index, 1]
        visible_tools = plot_grouped_bars(
            runtime_axis,
            task=task,
            cases=[case],
            records=records,
            value_getter=lambda record: record.get("median_seconds"),
            formatter=format_runtime,
            ylabel="Median runtime (seconds)",
            use_log_scale=should_use_log_scale(
                metric_values_for_case(case, records, lambda record: record.get("median_seconds"))
            ),
            show_convergence=False,
            show_group_labels=False,
        )
        runtime_axis.set_title(panel_title(case), loc="left", fontsize=11, fontweight="bold", pad=10)
        plot_grouped_bars(
            similarity_axis,
            task=task,
            cases=[case],
            records=records,
            value_getter=lambda record: metric_value(record, "cosine_similarity"),
            formatter=format_error,
            ylabel="Cosine similarity\nvs. gdistance",
            use_log_scale=False,
            show_convergence=False,
            tool_subset=alignment_tools,
            ylim=(0.0, 1.05),
            show_group_labels=False,
        )
        if row_index == len(task_cases) - 1:
            runtime_axis.set_xlabel("Configuration (CPU/GPU shown in labels)", labelpad=12)
            similarity_axis.set_xlabel("Configuration (CPU/GPU shown in labels)", labelpad=12)

    top = add_style_legend(fig, records if visible_tools else []) if task_cases else 0.95
    add_placeholder_note(fig, task, records)
    fig.tight_layout(rect=(0, 0.05, 1, top))
    save_scorecard_figure(fig, output_path)
    plt.close(fig)


def render_inverse_chart(
    task: str, records: list[dict], output_path: Path, cases: dict[str, dict]
) -> None:
    task_cases = ordered_cases(task, cases)
    tool_count = len(display_tools(task, records))
    fig, axes = plt.subplots(
        len(task_cases),
        2,
        figsize=chart_size(tool_count, len(task_cases), columns=2, row_height=4.8),
        sharex=False,
        squeeze=False,
    )
    fig.suptitle(f"{TASK_LABELS[task]} benchmark")

    visible_tools: list[str] = []
    for row_index, case in enumerate(task_cases):
        runtime_axis = axes[row_index, 0]
        loss_axis = axes[row_index, 1]
        visible_tools = plot_grouped_bars(
            runtime_axis,
            task=task,
            cases=[case],
            records=records,
            value_getter=lambda record: record.get("median_seconds"),
            formatter=format_runtime,
            ylabel="Median runtime (seconds)",
            use_log_scale=should_use_log_scale(
                metric_values_for_case(case, records, lambda record: record.get("median_seconds"))
            ),
            show_convergence=True,
            show_group_labels=False,
        )
        runtime_axis.set_title(panel_title(case), loc="left", fontsize=11, fontweight="bold", pad=10)
        plot_grouped_bars(
            loss_axis,
            task=task,
            cases=[case],
            records=records,
            value_getter=lambda record: metric_value(record, "relative_rmse"),
            formatter=format_loss,
            ylabel="Final fit relative RMSE",
            use_log_scale=should_use_log_scale(
                metric_values_for_case(
                    case,
                    records,
                    lambda record: metric_value(record, "relative_rmse"),
                )
            ),
            show_convergence=True,
            show_group_labels=False,
        )
        if row_index == len(task_cases) - 1:
            runtime_axis.set_xlabel("Configuration (CPU/GPU shown in labels)", labelpad=12)
            loss_axis.set_xlabel("Configuration (CPU/GPU shown in labels)", labelpad=12)

    top = (
        add_style_legend(fig, records if visible_tools else [], include_inverse_status=True)
        if task_cases
        else 0.95
    )

    add_placeholder_note(fig, task, records)
    fig.tight_layout(rect=(0, 0.05, 1, top))
    save_scorecard_figure(fig, output_path)
    plt.close(fig)


def render_task(
    task: str, records: list[dict], output_path: Path, cases: dict[str, dict]
) -> None:
    task_cases = ordered_cases(task, cases)
    if not records and not task_cases:
        fig, axis = plt.subplots(figsize=(8, 2.4))
        axis.text(
            0.5,
            0.5,
            "No compatible benchmark data",
            ha="center",
            va="center",
            fontsize=11,
        )
        axis.set_axis_off()
        fig.tight_layout()
        save_scorecard_figure(fig, output_path)
        plt.close(fig)
        return

    if task == "inverse_landscape_genetics":
        render_inverse_chart(task, records, output_path, cases)
        return
    if task == "sensitivity_analysis":
        render_centrality_chart(task, records, output_path, cases)
        return

    render_runtime_chart(task, records, output_path, cases)


def render(*, results_json: Path = DEFAULT_RESULTS_JSON, output_dir: Path | None = None) -> None:
    payload = load_payload(results_json)
    cases = case_lookup(payload)
    outputs = DEFAULT_OUTPUTS if output_dir is None else scorecard_outputs(output_dir)
    grouped: dict[str, list[dict]] = {task: [] for task in outputs}
    for record in payload["records"]:
        if record["task"] in grouped:
            grouped[record["task"]].append(record)

    for task, output_path in outputs.items():
        renderable_records = [
            record
            for record in grouped.get(task, [])
            if record["status"] == "ok" and record.get("median_seconds") is not None
        ] + skipped_placeholder_records(task, grouped.get(task, []))
        render_task(task, renderable_records, output_path, cases)


if __name__ == "__main__":
    args = parse_args()
    render(results_json=args.results_json, output_dir=args.output_dir)
