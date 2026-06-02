"""Generate docs/benchmark.md from the benchmark template and latest results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
for path in (ROOT, SRC_DIR):
    path_string = str(path)
    while path_string in sys.path:
        sys.path.remove(path_string)
for path in (ROOT, SRC_DIR):
    sys.path.insert(0, str(path))

from benchmark.benchmark_distances import CASES, iter_cases
from benchmark.benchmark_registry import BENCHMARK_TASK_SPECS, TOOL_SPECS_BY_TASK


DEFAULT_RESULTS_JSON = ROOT / "benchmark" / "results" / "benchmark_results.json"
DEFAULT_TEMPLATE = ROOT / "docs" / "benchmark.template.md"
DEFAULT_OUTPUT = ROOT / "docs" / "benchmark.md"

COMPATIBILITY_NOTES = {
    "resistance_distance": (
        "The published `gdistance` series is rescaled from commute time to "
        "effective resistance by dividing by graph volume."
    ),
    "least_cost_path": (
        "The published comparison uses the same grid-graph least-cost task in "
        "JAXScape and `gdistance`."
    ),
    "sensitivity_analysis": (
        "The chart compares single-origin JAXScape gradients against the "
        "matching single-origin `gdistance::shortestPath` incidence and "
        "`gdistance::passage(..., totalNet = \"total\")` references on the "
        "same origin/destination set."
    ),
    "inverse_landscape_genetics": (
        "Runtime, convergence status, and final fit quality are reported "
        "together; the published fit-quality chart uses relative RMSE because "
        "raw inverse MSE is not directly comparable across the current tool-"
        "specific objective scales."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_payload(results_json: Path) -> dict[str, Any]:
    if results_json.exists():
        return json.loads(results_json.read_text())
    return {"environment": {}, "cases": [], "records": []}


def tool_ecosystem(label: str) -> str:
    if label.startswith("JAXScape"):
        return "JAXScape"
    if label.startswith("gdistance"):
        return "gdistance"
    if label.startswith("Circuitscape.jl"):
        return "Circuitscape.jl"
    if label.startswith("ResistanceGA"):
        return "ResistanceGA"
    if label.startswith("Conefor"):
        return "Conefor"
    return label


def published_tool_ecosystems(task: str) -> list[str]:
    ecosystems: list[str] = []
    for tool in TOOL_SPECS_BY_TASK[task]:
        if not tool.automated:
            continue
        ecosystem = tool_ecosystem(tool.label)
        if ecosystem not in ecosystems:
            ecosystems.append(ecosystem)
    return ecosystems


def published_coverage(task: str) -> str:
    ecosystems = published_tool_ecosystems(task)
    if not ecosystems:
        return "none"
    return ", ".join(f"`{ecosystem}`" for ecosystem in ecosystems)


def compatibility_table() -> str:
    lines = [
        "| Feature | Published benchmark coverage | Notes |",
        "| --- | --- | --- |",
    ]
    for task_spec in BENCHMARK_TASK_SPECS:
        task = task_spec.task
        coverage = published_coverage(task)
        note = COMPATIBILITY_NOTES[task]
        lines.append(f"| {task_spec.title} | {coverage} | {note} |")
    return "\n".join(lines)


def fallback_cases() -> list[dict[str, Any]]:
    return [
        {
            "name": case.name,
            "task": case.task,
            "size_label": case.size_label,
            "grid_size": case.grid_size,
            "point_count": len(case.points),
        }
        for case in iter_cases(CASES)
    ]


def merged_case_index(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for case in fallback_cases():
        indexed[(case["task"], case["size_label"])] = case
    for case in payload.get("cases", []):
        indexed[(case["task"], case["size_label"])] = case
    return indexed


def environment_table(payload: dict[str, Any]) -> str:
    environment = payload.get("environment") or {}
    thread_environment = environment.get("thread_environment") or {}
    rows = [
        ("Requested device", environment.get("requested_device_platform", "n/a")),
        (
            "Available JAX platforms",
            ", ".join(environment.get("available_jax_platforms", [])) or "n/a",
        ),
        ("Repeats", environment.get("repeats", "n/a")),
        ("Benchmark threads", environment.get("benchmark_threads", "n/a")),
        ("CPU device", environment.get("cpu_device", "n/a")),
        ("GPU device", environment.get("gpu_device", "n/a") or "n/a"),
        (
            "Thread environment",
            ", ".join(f"{key}={value}" for key, value in thread_environment.items())
            or "n/a",
        ),
    ]
    lines = ["| Setting | Value |", "| --- | --- |"]
    for key, value in rows:
        lines.append(f"| {key} | `{value}` |")
    return "\n".join(lines)


def strip_backend(tool: str) -> str:
    if tool.endswith(" (CPU)") or tool.endswith(" (GPU)"):
        return tool.rsplit(" (", 1)[0]
    return tool


def largest_case_names_by_task(payload: dict[str, Any]) -> dict[str, str]:
    case_index = merged_case_index(payload)
    selected: dict[str, tuple[int, str]] = {}
    for (task, _size_label), case in case_index.items():
        grid_size = int(case.get("grid_size") or 0)
        case_name = str(case["name"])
        current = selected.get(task)
        if current is None or grid_size > current[0]:
            selected[task] = (grid_size, case_name)
    return {task: case_name for task, (_grid_size, case_name) in selected.items()}


def format_runtime(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1e-2:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3g}"


def best_jaxscape_record_for_task(
    payload: dict[str, Any], task: str, scenario: str
) -> dict[str, Any] | None:
    task_records = [
        record
        for record in payload.get("records", [])
        if record.get("task") == task
        and record.get("scenario") == scenario
        and record.get("status") == "ok"
        and record.get("median_seconds") is not None
        and str(record.get("tool", "")).startswith("JAXScape")
    ]
    if not task_records:
        return None

    if task == "inverse_landscape_genetics":
        converged = [
            record
            for record in task_records
            if (record.get("metrics") or {}).get("converged") is True
        ]
        if converged:
            task_records = converged

    return min(
        task_records,
        key=lambda record: (
            float(record["median_seconds"]),
            0 if str(record["tool"]).endswith("(GPU)") else 1,
            str(record["tool"]),
        ),
    )


def matching_cpu_record(payload: dict[str, Any], record: dict[str, Any]) -> dict[str, Any] | None:
    tool = str(record["tool"])
    if not tool.endswith("(GPU)"):
        return None
    cpu_tool = f"{strip_backend(tool)} (CPU)"
    for candidate in payload.get("records", []):
        if (
            candidate.get("task") == record.get("task")
            and candidate.get("scenario") == record.get("scenario")
            and candidate.get("tool") == cpu_tool
            and candidate.get("status") == "ok"
            and candidate.get("median_seconds") is not None
        ):
            return candidate
    return None


def recommendation_box(payload: dict[str, Any]) -> str:
    largest_cases = largest_case_names_by_task(payload)
    lines = ['!!! tip "Observed Best JAXScape Configurations"']
    for task_spec in BENCHMARK_TASK_SPECS:
        scenario = largest_cases.get(task_spec.task)
        if scenario is None:
            lines.append(
                f"    - {task_spec.title}: no benchmark scenario is available in the current artifact."
            )
            continue

        record = best_jaxscape_record_for_task(payload, task_spec.task, scenario)
        if record is None:
            lines.append(
                f"    - {task_spec.title}: no successful JAXScape run is available for `{scenario}` in the current artifact."
            )
            continue

        line = (
            f"    - {task_spec.title}: `{record['tool']}` is the fastest successful "
            f"JAXScape configuration on `{scenario}` at {format_runtime(record['median_seconds'])}"
        )
        metrics = record.get("metrics") or {}
        if task_spec.task == "inverse_landscape_genetics":
            line += (
                f", final MSE {format_metric(metrics.get('final_mse'))}, "
                f"converged={metrics.get('converged', 'n/a')}"
            )

        cpu_record = matching_cpu_record(payload, record)
        if cpu_record is not None:
            speedup = float(cpu_record["median_seconds"]) / float(record["median_seconds"])
            line += f", about {speedup:.2f}x faster than the matching CPU run"

        line += "."
        lines.append(line)
    return "\n".join(lines)


def task_profiles(task: str) -> str:
    return published_coverage(task)


def render_page(template_text: str, payload: dict[str, Any]) -> str:
    replacements = {
        "{{RECOMMENDATION_BOX}}": recommendation_box(payload),
        "{{ENVIRONMENT_TABLE}}": environment_table(payload),
        "{{COMPATIBILITY_TABLE}}": compatibility_table(),
        "{{RESISTANCE_PROFILES}}": task_profiles("resistance_distance"),
        "{{LCP_PROFILES}}": task_profiles("least_cost_path"),
        "{{SENSITIVITY_PROFILES}}": task_profiles("sensitivity_analysis"),
        "{{INVERSE_PROFILES}}": task_profiles("inverse_landscape_genetics"),
    }

    rendered = template_text
    for placeholder, replacement in replacements.items():
        rendered = rendered.replace(placeholder, replacement)
    return (
        "<!-- Generated from docs/benchmark.template.md by "
        "benchmark/generate_benchmark_docs.py. -->\n\n"
        + rendered
    )


def main() -> None:
    args = parse_args()
    payload = load_payload(args.results_json)
    template_text = args.template.read_text()
    rendered = render_page(template_text, payload)
    args.output.write_text(rendered)


if __name__ == "__main__":
    main()