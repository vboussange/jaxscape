"""Render the committed benchmark scorecard from benchmark/results/benchmark_results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_JSON = ROOT / "benchmark" / "results" / "benchmark_results.json"
OUTPUT = ROOT / "docs" / "assets" / "benchmark_scorecard.png"
TASKS = [
    "resistance_distance",
    "least_cost_path",
    "sensitivity_analysis",
    "inverse_landscape_genetics",
]
TOOLS = ["JAXScape", "Circuitscape.jl", "Conefor", "samc", "JAXScape + Optimistix", "ResistanceGA"]


def load_records() -> list[dict]:
    payload = json.loads(RESULTS_JSON.read_text())
    return payload["records"]


def format_runtime(seconds: float) -> str:
    if seconds < 1e-2:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.3f} s"


def render() -> None:
    records = load_records()
    matrix = np.full((len(TOOLS), len(TASKS)), np.nan)
    notes = [["" for _ in TASKS] for _ in TOOLS]
    for record in records:
        row = TOOLS.index(record["tool"])
        col = TASKS.index(record["task"])
        if record["status"] == "ok" and record["median_seconds"] is not None:
            matrix[row, col] = record["median_seconds"]
        else:
            notes[row][col] = record["note"] or record["status"]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#d9d9d9")
    image = ax.imshow(masked, aspect="auto", cmap=cmap)

    for row, tool in enumerate(TOOLS):
        for col, task in enumerate(TASKS):
            value = matrix[row, col]
            if np.isfinite(value):
                label = format_runtime(float(value))
                color = "white" if value > np.nanmedian(matrix) else "black"
            else:
                label = "n/a"
                color = "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(TASKS)), [task.replace("_", "\n") for task in TASKS])
    ax.set_yticks(range(len(TOOLS)), TOOLS)
    ax.set_title("JAXScape benchmark scorecard (lower is faster)")
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("Median runtime (seconds)")
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    render()
