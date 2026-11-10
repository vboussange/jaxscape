"""Turn the records of a run into its CSV, figures and Benchmark docs page.

    uv run python report.py RUN_DIR

writes `results.csv`, one figure per question, such as `e1.png`, and the page
`index.md` into RUN_DIR. The report reads the records, the environment snapshot
and the header comment of each adapter that the records name.
"""

import argparse
import csv
import json
from itertools import groupby
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent

FIELDS = (
    "question tool distance device dtype solver case cells status reason warmup_s "
    "median_s times_s measure agreement tier version files adapter"
).split()
CONFIGURATION = ("tool", "distance", "device", "dtype", "solver")

QUESTIONS = {
    "E1": (
        "Isolation between populations",
        "How isolated are two populations once the landscape between them is "
        "taken into account? *Quantity:* the effective distance $d_{ij}$ between "
        "focal habitat sites, which replaces Euclidean distance in dispersal "
        "kernels and in models of isolation by distance.",
    ),
    "E2": (
        "Habitat connectivity",
        "How much habitat can each location functionally reach, and how much "
        "connected habitat does the landscape hold in total? *Quantity:* "
        "Hanski's connectivity $S_i = \\sum_j K(d_{ij}) q_j$ at each location, "
        "and its quality-weighted total $\\sum_i q_i S_i$, the quantity behind "
        "the equivalent connected area (ECA).",
    ),
    "E3": (
        "Corridors and pinch points",
        "Which locations carry dispersal between habitat areas, so that losing "
        "their permeability would reduce connectivity the most? *Quantity:* a "
        "map of how much each location contributes to connectivity: the "
        "marginal sensitivity $\\partial C / \\partial p_k$ in JAXScape, and a "
        "flow, such as current, betweenness or visits, in the other tools.",
    ),
}

SETUPS = {
    "E1": (
        "Each tool computes the distances among 32 sites drawn at random from "
        "the habitat cells with a fixed seed, across the whole raster: 496 "
        "pairs. The agreement of a record is the largest relative error over "
        "the pairs against a float64 reference, computed once for each case and "
        "distance: a sparse direct solve with `CholmodSolver` for resistance "
        "and free energy, and Dijkstra's algorithm in SciPy for least-cost "
        "distances."
    ),
}

EXCLUDED = [
    ("Linkage Mapper", "Needs ArcGIS."),
    ("Circuitscape 4", "Superseded by Circuitscape.jl."),
    ("GFlow", "Abandoned."),
    ("UNICOR", "Has no release and no licence."),
    ("Conefor", "Starts from a patch graph."),
    ("Makurhini", "Starts from vector patches."),
    ("GRASS `r.cost`", "Duplicates least-cost paths."),
    ("WhiteboxTools", "Duplicates least-cost paths."),
    ("`terra`", "Duplicates least-cost paths."),
    ("DiffiScape", "Wraps JAXScape."),
    ("circuitscaper", "Wraps Circuitscape.jl."),
    ("enerscape", "Computes no connectivity from resistance."),
    ("Zonation", "Computes no connectivity from resistance."),
    ("MSPA", "Computes no connectivity from resistance."),
    ("ResistanceGA", "Moves to `ilg-toolkit` with inverse landscape genetics."),
    ("radish", "Moves to `ilg-toolkit` with inverse landscape genetics."),
]

STATUS_MARKERS = {"failed": "X", "timeout": "v", "oom": "s", "skipped": "o"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    run_dir = parser.parse_args().run_dir
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    environment = json.loads((run_dir / "environment.json").read_text())

    write_csv(records, run_dir / "results.csv")
    questions = sorted({record["question"] for record in records})
    for question in questions:
        plot(question, of(question, records), run_dir / f"{question.lower()}.png")
    (run_dir / "index.md").write_text(page(run_dir.name, records, environment))


def of(question, records):
    return [record for record in records if record["question"] == question]


def write_csv(records, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            times = record.get("times_s") or []
            writer.writerow({**record, "times_s": " ".join(map(str, times))})


def configurations(records):
    """Group records by configuration, in the order of their first record."""
    order = {}
    for record in records:
        order.setdefault(tuple(record[field] for field in CONFIGURATION), len(order))

    def key(record):
        return order[tuple(record[field] for field in CONFIGURATION)]

    for _, group in groupby(sorted(records, key=key), key=key):
        group = sorted(group, key=lambda record: record["cells"])
        yield dict(zip(CONFIGURATION, (group[0][f] for f in CONFIGURATION))), group


def plot(question, records, path):
    """Plot the warm runtime of each configuration against the raster cells.

    GPU lines are solid and CPU lines dashed. A cell without a runtime shows
    its status as a marker along the top edge, one row per configuration.
    """
    figure, axes = plt.subplots(figsize=(8, 4.5))
    colors = {}
    for index, (configuration, group) in enumerate(configurations(records)):
        color = colors.setdefault(
            (configuration["tool"], configuration["distance"]), f"C{len(colors) % 10}"
        )
        ok = [record for record in group if record["status"] == "ok"]
        axes.plot(
            [record["cells"] for record in ok],
            [record["median_s"] for record in ok],
            "-" if configuration["device"] == "gpu" else "--",
            marker="o",
            color=color,
            label=f"{configuration['tool']} · {configuration['distance']} · "
            f"{configuration['device'].upper()} · Tier {group[0]['tier']}",
        )
        for record in group:
            if record["status"] != "ok":
                axes.plot(
                    record["cells"],
                    0.96 - 0.04 * index,
                    marker=STATUS_MARKERS[record["status"]],
                    color=color,
                    markerfacecolor="none" if record["status"] == "skipped" else color,
                    transform=axes.get_xaxis_transform(),
                    clip_on=False,
                )
    cells = [record["cells"] for record in records]
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlim(min(cells) / 2, max(cells) * 2)
    if not any(record["status"] == "ok" for record in records):
        axes.set_ylim(1e-3, 1e3)
    if any(record["status"] != "ok" for record in records):
        # Headroom above the runtimes for the band of status markers.
        low, high = np.log10(axes.get_ylim())
        band = 0.06 + 0.04 * (index + 1)
        axes.set_ylim(10**low, 10 ** (high + (high - low) * band / (1 - band)))
    axes.set_xlabel("Raster cells")
    axes.set_ylabel("Warm runtime (s)")
    axes.set_title(f"{question}. {QUESTIONS[question][0]}")
    statuses = [
        Line2D(
            [],
            [],
            linestyle="",
            marker=marker,
            color="gray",
            markerfacecolor="none" if status == "skipped" else "gray",
            label=status,
        )
        for status, marker in STATUS_MARKERS.items()
    ]
    handles = axes.get_legend_handles_labels()[0] + statuses
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def row(cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def table(header, rows):
    lines = [row(header), row(["---"] * len(header))]
    return "\n".join(lines + [row(cells) for cells in rows])


def case_label(case):
    side = case.removeprefix("synthetic-")
    return f"{side}²" if side != case else case


def value(record):
    if record["status"] != "ok":
        return record["status"]
    if record["agreement"] is None:
        return "missing"
    return f"{record['agreement']:.1e}"


def agreement_table(records):
    cases = sorted({(r["cells"], r["case"]) for r in records})
    rows = []
    for configuration, group in configurations(records):
        by_case = {record["case"]: value(record) for record in group}
        rows.append(
            [configuration[field] for field in CONFIGURATION]
            + [group[0]["tier"]]
            + [by_case.get(case, "") for _, case in cases]
        )
    rows = [[cells[0], cells[1], cells[2].upper(), *cells[3:]] for cells in rows]
    header = ["Tool", "Distance", "Device", "dtype", "Solver", "Tier"]
    return table(header + [case_label(case) for _, case in cases], rows)


def mapping(adapter):
    """Return the header comment of an adapter, which states its mapping."""
    lines = []
    for line in (HERE / adapter).read_text().splitlines():
        if line.startswith("#!"):
            continue
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("#").strip())
    return " ".join(line for line in lines if line).replace("|", "\\|")


def page(run_name, records, environment):
    questions = sorted({record["question"] for record in records})
    tools = {}
    for record in records:
        tool = tools.setdefault(
            record["tool"],
            {"adapter": record["adapter"], "questions": set(), "versions": set()},
        )
        tool["questions"].add(record["question"])
        if record["version"]:
            tool["versions"].add(record["version"])
    versions = ", ".join(
        f"{name} {', '.join(sorted(tool['versions']))}" for name, tool in tools.items()
    )
    measures = {
        question: sorted({record["measure"] for record in of(question, records)})
        for question in questions
    }

    sections = [
        "# Benchmark",
        "This page compares JAXScape "
        f"{environment['packages']['jaxscape']} with the software that "
        "ecologists use for connectivity analysis, on three questions from "
        f"ecology. It is generated from the records of the run `{run_name}`, "
        f"started on {environment['date'][:10]} on {environment['host']}. "
        "[`results.csv`](results.csv) lists every cell of the run.",
        "## Questions",
        "\n".join(
            f"- **{name}. {title}.** {text}"
            for name, (title, text) in QUESTIONS.items()
        ),
        "Each comparison carries a tier of comparability. Tier 1 compares the "
        "same quantity, so that the error of each tool against a float64 "
        "reference measures its accuracy. Tier 2 compares the same quantity "
        "computed on another graph or with another kernel, and Tier 3 a related "
        "quantity that answers the same question; both report Spearman's rank "
        "correlation between maps. A runtime gap measures a speed-up only at "
        "Tier 1.",
        "## Landscapes",
        "Each synthetic case draws a Gaussian random field with a Gaussian "
        "covariance of correlation length 25 pixels, on a periodic grid with a "
        "fixed seed, and normalizes its ranks to [0, 1]. Quality equals the "
        "field where the field is at least 0.7, and 0 elsewhere, so that 30 % of "
        "the cells are habitat. Permeability is $10^{-2(1 - f)}$ for the field "
        "$f$, from 0.01 to 1, and resistance is its inverse. A larger raster "
        "covers a larger area at the same grain.",
        "## Comparison",
        "Each cell runs in its own process, pinned to the cores of the "
        "environment table, with thread settings matched to them; XLA sizes "
        "its thread pools to these cores. A warm-up call comes first and "
        "compiles the code of tools that compile. The warm runtime is the "
        "median of the five timed calls that follow it, or the time of a single "
        "call when the warm-up takes more than five minutes. Every call redoes "
        "all work that depends on the "
        "landscape's values: graph assembly, factorization, multigrid hierarchy "
        "and solver state. A call stops after one hour. Within a configuration, "
        "rasters run from the smallest up, and after a failure, a timeout or "
        "running out of memory (`oom`), the configuration skips its larger "
        "rasters. In the figures, JAXScape on the GPU is solid and on the CPU "
        "dashed, and cells without a runtime show their status along the top "
        "edge.",
    ]
    for question in questions:
        records_of_question = of(question, records)
        sections += [
            f"### {question}. {QUESTIONS[question][0]}",
            SETUPS.get(question, ""),
            f"![{question}: warm runtime against raster cells]({question.lower()}.png)",
            f"Agreement ({', '.join(measures[question])}) of each configuration "
            "and raster, or the status of the cell; `missing` marks a cell "
            "whose reference is unavailable.",
            agreement_table(records_of_question),
        ]
    threads = ", ".join(f"`{k}={v}`" for k, v in environment["threads"].items())
    packages = ", ".join(f"{k} {v}" for k, v in environment["packages"].items())
    sections += [
        "## Environment",
        table(
            ["Setting", "Value"],
            [
                ["Host", environment["host"]],
                ["GPU", environment["gpu"]],
                ["CPU", environment["cpu"]],
                ["Cores", environment["cores"]],
                ["Threads", threads],
                ["Python", environment["python"]],
                ["Packages", packages],
                ["Tools", versions],
            ],
        ),
        "## Parameter mappings",
        "The header comment of each adapter states how it maps the shared "
        "parameters onto its tool.",
        table(
            ["Tool", "Questions", "Adapter", "Mapping"],
            [
                [
                    name,
                    ", ".join(sorted(tool["questions"])),
                    f"`{tool['adapter']}`",
                    mapping(tool["adapter"]),
                ]
                for name, tool in tools.items()
            ],
        ),
        "## Excluded tools",
        table(["Tool", "Reason"], EXCLUDED),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


if __name__ == "__main__":
    main()
