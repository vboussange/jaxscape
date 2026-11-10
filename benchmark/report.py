"""Turn the records of a run into its CSV, figures and Benchmark docs page.

    uv run python report.py RUN_DIR              # report the run
    uv run python report.py RUN_DIR --publish    # and make it the docs page

writes `results.csv`, one figure per question, such as `e1.png`, the Maryland
figure `maryland.png`, the figure of the solver view `solvers.png`, and the page
`index.md` into RUN_DIR. The report reads the records, the configuration and the
environment snapshot of the run, the maps of the Maryland figure and the header
comment of each adapter that the records name. `--publish` points
`docs/benchmark`, the Benchmark page of the docs, at RUN_DIR.
"""

import argparse
import csv
import json
import os
from itertools import groupby
from pathlib import Path

import matplotlib
import numpy as np
from run import FIELDS, map_path, RHO
from scipy.stats import rankdata


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs" / "benchmark"  # the Benchmark page of the docs

# The fields of a configuration in the tables of the comparison and of the
# solver view, and their headers.
COMPARISON = ("tool", "distance", "device", "dtype", "solver", "tier")
SOLVER_VIEW = ("question", "distance", "dtype", "solver")
HEADERS = {
    "question": "Question",
    "tool": "Tool",
    "distance": "Distance",
    "device": "Device",
    "dtype": "dtype",
    "solver": "Solver",
    "tier": "Tier",
}
LINE = ("tool", "distance", "device")  # a line of a figure of the comparison
SOLID = {"gpu", "float32"}  # the lines of the figures drawn solid, others dashed
STATUS_MARKERS = {"failed": "X", "timeout": "v", "oom": "s", "skipped": "o"}
# The Maryland figure: JAXScape's map of a question next to a tool's map.
MARYLAND = {"question": "E3", "tool": "Omniscape.jl", "map": "normalized current"}
BLUES = "#cde2fb #9ec5f4 #6da7ec #3987e5 #256abf #184f95 #0d366b".split()

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--publish",
        nargs="?",
        const=DOCS,
        type=Path,
        metavar="LINK",
        help="point LINK, by default docs/benchmark, at RUN_DIR",
    )
    arguments = parser.parse_args()
    report(arguments.run_dir)
    if arguments.publish:
        publish(arguments.run_dir, arguments.publish)


def report(run_dir):
    """Write the CSV, the figures and the page of a run into its directory.

    **Arguments:**

    - `run_dir`: The run directory, with `records.jsonl`, `config.json`,
      `environment.json` and, when they are kept, the maps of the Maryland
      figure. Without the maps, the Maryland figure drawn before stays.
    """
    lines = (run_dir / "records.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    config = json.loads((run_dir / "config.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())

    write_csv(records, run_dir / "results.csv")
    comparison = [record for record in records if record["view"] == "comparison"]
    for question in sorted({record["question"] for record in comparison}):
        path = run_dir / f"{question.lower()}.png"
        plot(of_question(question, comparison), ("question",), LINE, path)
    solvers = [record for record in records if record["view"] == "solver"]
    if solvers:
        plot(
            solvers,
            ("question", "distance"),
            ("solver", "dtype"),
            run_dir / "solvers.png",
        )
    pair = maryland_pair(comparison)
    if pair and all(map_path(run_dir, record).exists() for record in pair):
        plot_maryland(run_dir, *pair, run_dir / "maryland.png")
    figure = pair if (run_dir / "maryland.png").exists() else None
    text = page(run_dir.name, records, config, environment, figure)
    (run_dir / "index.md").write_text(text)


def publish(run_dir, link):
    """Point a link at a run directory, so that the page there is the run's.

    **Arguments:**

    - `run_dir`: The run directory.
    - `link`: The link to point at it, by a relative path, such as
      `docs/benchmark`.
    """
    link.unlink(missing_ok=True)
    link.symlink_to(os.path.relpath(run_dir.resolve(), link.parent.resolve()))


def of_question(question, records):
    """Return the records of a question."""
    return [record for record in records if record["question"] == question]


def side(case):
    """Return the side of a synthetic case, such as 256 for `synthetic-256`, or
    `None` for Maryland."""
    prefix = "synthetic-"
    return int(case.removeprefix(prefix)) if case.startswith(prefix) else None


def case_label(case):
    """Return the label of a case in the tables, such as `256²`."""
    return f"{side(case)}²" if side(case) else case.capitalize()


def rank(record):
    """Order the records of a raster: the fastest that succeeds first, then the
    others by their status, in the order of `STATUS_MARKERS`."""
    if record["status"] == "ok":
        return (0, record["median_s"])
    return (1, list(STATUS_MARKERS).index(record["status"]))


def write_csv(records, path):
    """Write the records as a CSV, one row per record, with the timed calls
    separated by spaces."""
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS)
        writer.writeheader()
        for record in records:
            times = record["times_s"] or []
            writer.writerow({**record, "times_s": " ".join(map(str, times))})


def group_by(records, fields):
    """Yield the values of `fields` and the records that share them, in the
    order of their first record, each group sorted by raster cells."""
    order = {}
    for record in records:
        order.setdefault(tuple(record[field] for field in fields), len(order))

    def key(record):
        return order[tuple(record[field] for field in fields)]

    for _, group in groupby(sorted(records, key=key), key=key):
        group = sorted(group, key=lambda record: record["cells"])
        yield {field: group[0][field] for field in fields}, group


def plot(records, panel, line, path):
    """Plot the warm runtime of each line against the raster cells.

    At each raster, a line takes the fastest of its records that succeeds, and
    joins the synthetic rasters; Maryland is a star. A raster where none of its
    records succeeds shows a status as a marker along the top edge, one row per
    line.

    **Arguments:**

    - `records`: The records to plot.
    - `panel`: The fields whose values give a panel, such as `("question",)`.
    - `line`: The fields whose values give a line. The last, the device or the
      dtype, sets its style: GPU and float32 lines are solid, and CPU and
      float64 lines dashed. The others set its colour, and the legend gives
      their values, with the tier of a tool.
    - `path`: Where to save the figure.
    """
    panels = list(group_by(records, panel))
    colors = list(dict.fromkeys(tuple(r[f] for f in line[:-1]) for r in records))
    tiers = {tuple(r[f] for f in line[:-1]): r["tier"] for r in records}
    columns = min(2, len(panels))
    rows = -(-len(panels) // columns)
    figure, grid = plt.subplots(
        rows, columns, figsize=(6 * columns, 4 * rows), squeeze=False
    )
    for axes in grid.flat[len(panels) :]:
        axes.set_visible(False)
    for axes, (values, group) in zip(grid.flat, panels):
        band = 0
        for of_line, records_of_line in group_by(group, line):
            color = f"C{colors.index(tuple(of_line.values())[:-1]) % 10}"
            best = [
                min(of_raster, key=rank)
                for _, of_raster in groupby(records_of_line, key=lambda r: r["cells"])
            ]
            ok = [record for record in best if record["status"] == "ok"]
            synthetic = [record for record in ok if side(record["case"])]
            axes.plot(
                [record["cells"] for record in synthetic],
                [record["median_s"] for record in synthetic],
                "-" if of_line[line[-1]] in SOLID else "--",
                marker="o",
                color=color,
            )
            for record in ok:
                if not side(record["case"]):
                    axes.plot(record["cells"], record["median_s"], "*", color=color)
            failed = [record for record in best if record["status"] != "ok"]
            for record in failed:
                axes.plot(
                    record["cells"],
                    0.96 - 0.04 * band,
                    marker=STATUS_MARKERS[record["status"]],
                    color=color,
                    markerfacecolor="none" if record["status"] == "skipped" else color,
                    transform=axes.get_xaxis_transform(),
                    clip_on=False,
                )
            band += bool(failed)
        cells = [record["cells"] for record in group]
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlim(min(cells) / 2, max(cells) * 2)
        if not any(record["status"] == "ok" for record in group):
            axes.set_ylim(1e-3, 1e3)
        if band:
            # Headroom above the runtimes for the band of status markers.
            low, high = np.log10(axes.get_ylim())
            share = 0.06 + 0.04 * band
            axes.set_ylim(10**low, 10 ** (high + (high - low) * share / (1 - share)))
        axes.set_xlabel("Raster cells")
        axes.set_ylabel("Warm runtime (s)")
        axes.set_title(
            " · ".join(
                f"{value}. {QUESTIONS[value][0]}" if value in QUESTIONS else value
                for value in values.values()
            )
        )
    handles = [
        Line2D(
            [],
            [],
            color=f"C{index % 10}",
            label=" · ".join(key) + (f" · Tier {tiers[key]}" if tiers[key] else ""),
        )
        for index, key in enumerate(colors)
    ]
    styles = sorted(
        {record[line[-1]] for record in records}, key=lambda v: v not in SOLID
    )
    handles += [
        Line2D(
            [],
            [],
            color="gray",
            linestyle="-" if style in SOLID else "--",
            label=style.upper() if line[-1] == "device" else style,
        )
        for style in styles
    ]
    if not all(side(record["case"]) for record in records):
        maryland = dict(linestyle="", marker="*", color="gray", label="Maryland")
        handles.append(Line2D([], [], **maryland))
    handles += [
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
    figure.tight_layout()
    figure.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def maryland_pair(records):
    """Return the records of the Maryland figure, or `None` when either is
    missing: the JAXScape GPU record under the tool's distance, whose map the
    tool's maps are compared with, and the fastest setting of the tool that
    succeeds."""
    on_maryland = [
        record
        for record in of_question(MARYLAND["question"], records)
        if record["case"] == "maryland" and record["status"] == "ok"
    ]
    others = [record for record in on_maryland if record["tool"] == MARYLAND["tool"]]
    if not others:
        return None
    other = min(others, key=rank)
    for record in on_maryland:
        if (record["tool"], record["device"], record["distance"]) == (
            "JAXScape",
            "gpu",
            other["distance"],
        ):
            return record, other
    return None


def quantiles(values):
    """Return the rank of each finite pixel as a quantile in [0, 1], and NaN
    elsewhere."""
    finite = np.isfinite(values)
    quantile = np.full(values.shape, np.nan)
    quantile[finite] = (rankdata(values[finite]) - 1) / max(finite.sum() - 1, 1)
    return quantile


def plot_maryland(run_dir, jaxscape, other, path):
    """Plot JAXScape's map on Maryland next to the other tool's.

    Each map is coloured by the rank of its pixels, which Spearman's ρ
    compares, and pixels without a value, such as barriers, are gray.
    """
    colormap = LinearSegmentedColormap.from_list("blues", BLUES)
    colormap = colormap.with_extremes(bad="#9e9e9e")
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")
    labels = (f"sensitivity under {other['distance']}", MARYLAND["map"])
    for axis, record, label in zip(axes, (jaxscape, other), labels):
        axis.set_axis_off()
        axis.set_title(
            f"{record['tool']}: {label}\n{record['device'].upper()}, {runtime(record)}"
        )
        quantile = quantiles(np.load(map_path(run_dir, record)))
        image = axis.imshow(quantile, cmap=colormap, vmin=0, vmax=1)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Rank of the pixel (quantile)")
    figure.suptitle(f"Maryland · {RHO} = {agreement(other)}")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def runtime(record):
    """Return the warm runtime of a record that succeeded."""
    return f"{record['median_s']:.3g} s"


def agreement(record):
    """Return the agreement of a record, or `missing` without one."""
    if record["agreement"] is None:
        return "missing"
    if record["measure"] == RHO:
        return f"{record['agreement']:.2f}"
    return f"{record['agreement']:.1e}"


def outcome(record):
    """Return the text of a record in a table: its warm runtime, its agreement
    and its landscape total, or its status."""
    if record["status"] != "ok":
        return record["status"]
    parts = [runtime(record)]
    if record["measure"]:
        parts.append(agreement(record))
    if record["total"] is not None:
        parts.append(f"total {record['total']:.3g}")
    return " · ".join(parts)


def row(cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def table(header, rows):
    lines = [row(header), row(["---"] * len(header))]
    return "\n".join(lines + [row(cells) for cells in rows])


def outcome_table(records, fields):
    """Return the table of the outcome of each configuration, the values of
    `fields`, on each raster."""
    cases = sorted({(record["cells"], record["case"]) for record in records})
    rows = []
    for values, group in group_by(records, fields):
        by_case = {record["case"]: outcome(record) for record in group}
        configuration = [
            str(value).upper() if field == "device" else value or ""
            for field, value in values.items()
        ]
        rows.append(configuration + [by_case.get(case, "") for _, case in cases])
    header = [HEADERS[field] for field in fields]
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


def duration(seconds):
    """Return a duration in hours, minutes or seconds, such as `15 min`."""
    for unit, length in (("h", 3600), ("min", 60)):
        if seconds % length == 0:
            return f"{seconds // length} {unit}"
    return f"{seconds} s"


def jaxscape_setting(config, question, distance, name):
    """Return a setting of JAXScape's entry of a question for a distance."""
    for entry in config["tools"]:
        if (entry["tool"], entry["question"]) == ("JAXScape", question):
            return entry["distances"][distance][name]


def combination(entry):
    """Return the text of an entry of `fails_by_design`, such as `` `free_energy`
    in `float32` ``."""
    words = []
    for field, joiner in (
        ("distance", ""),
        ("solver", "with "),
        ("dtype", "in "),
        ("question", "on "),
    ):
        if field in entry:
            values = entry[field] if isinstance(entry[field], list) else [entry[field]]
            either = " or ".join(f"`{value}`" for value in values)
            words.append((joiner if words else "") + either)
    return " ".join(words)


def setups(config):
    """Return the text of the setup of each question, from the configuration of
    the run."""
    windows = config["windows"]
    reach = windows["dependency_range"]
    scale = f"{reach / 4:g}"
    sites = config["E1"]["sites"]
    return {
        "E1": (
            f"Each tool computes the distances among {sites} sites drawn at "
            "random from the habitat cells with a fixed seed, across the whole "
            f"raster: {sites * (sites - 1) // 2} pairs. Every tool builds the "
            "same graph: rook contiguity, and the resistance of an edge is the "
            "mean resistance of its two pixels. The agreement of a record is the "
            "largest relative error over the pairs against a float64 reference, "
            "computed once for each case and distance: a sparse direct solve "
            "with `CholmodSolver` for resistance and free energy, and Dijkstra's "
            "algorithm in SciPy for least-cost distances."
        ),
        "E2": (
            "JAXScape computes, in each call, the map of $\\partial C / \\partial "
            "q_i$ with `SensitivityAnalysis`, which equals $2 S_i$ for symmetric "
            "distances, and the landscape total $C = \\sum_i q_i S_i$ with "
            "`ConnectivityAnalysis`. The table gives this total next to "
            "Graphab's, the square of its equivalent connectivity, "
            "$\\sum_I \\sum_J a_I a_J e^{-d_{IJ}/25}$ over its patches. Its "
            f"windows have `dependency_range` {reach} and `coarsening_factor` "
            f"{windows['coarsening_factor']}, which gives blocks of "
            f"{windows['block_size']} pixels; Omniscape's Maryland example takes "
            "`radius` 100 and `block_size` 21. Its proximity is the default, "
            f"$e^{{-r/{scale}}}$ with $r$ in equivalent pixels. The windowed "
            "analyses build their graphs with JAXScape's own edge rule: rook "
            "contiguity, and the conductance of an edge is the mean "
            "permeability of its two pixels. Resistance takes the setting of the "
            "[connectivity example]"
            "(../examples/connectivity_analysis/connectivity_analysis.ipynb): "
            f"`{jaxscape_setting(config, 'E2', 'resistance', 'solver')}` and "
            f"`{jaxscape_setting(config, 'E2', 'resistance', 'method')}`, which "
            "estimates resistances from random projections, on tiles of up to "
            f"{jaxscape_setting(config, 'E2', 'resistance', 'batch_size'):,} "
            "block centres. Exact resistance took 24 minutes a call of E3 at "
            "1024², and would take 32 hours at 8192². Free energy is left out of "
            "E2 and E3: its window query holds the dense inverse of the pixels "
            "of a window, 40,401 at `dependency_range` 100, and in float64, the "
            "only precision that resolves its windows, one such window ran out "
            "of the 24 GB of the GPU, and out of memory on the CPU after taking "
            "300 GB. The agreement of another tool is "
            "Spearman's ρ between its map and the map of JAXScape's GPU cell "
            "under the distance of its row, over the pixels where both maps are "
            "defined; it is missing when that cell fails. ConScape.jl computes "
            "its maps under free energy on the whole raster, with targets at "
            f"landmarks every {windows['block_size']} pixels and the kernel "
            f"$e^{{-d/{scale}}}$ on its free-energy distance $d$; JAXScape has no "
            "free-energy map to compare them with, and they are compared with "
            "its maps under least-cost distance, which ranks pairs of sites most "
            "like free energy at the $\\theta$ of the run, "
            f"{config['distances']['free_energy']['theta']}."
        ),
        "E3": (
            "JAXScape computes the map of $\\partial C / \\partial p_k$ with "
            "`SensitivityAnalysis`, with the windows and proximity of E2. The "
            "other tools map a flow, and the agreement of each is Spearman's ρ "
            "over pixels between its map and the map of JAXScape's GPU cell "
            "under the distance of its row, or under least-cost distance for "
            "ConScape.jl (see E2). Omniscape.jl computes its normalized current, "
            "the cumulative current divided by the flow potential, with the "
            "configuration of its Maryland example on JAXScape's windows, on a "
            "graph of eight neighbours."
        ),
    }


def solver_view_text(config):
    """Return the text of the setup of the solver view, from the configuration
    of the run."""
    view = config["solver_view"]
    runs = []
    for question, distances in view["questions"].items():
        for distance, settings in distances.items():
            sides = settings.get("sides")
            rasters = " and ".join(f"{side}²" for side in sides) if sides else None
            runs.append(
                f"{question} with {distance} on "
                + (rasters or "every raster of the distance")
            )
    left_out = "; ".join(map(combination, view.get("fails_by_design", [])))
    return (
        "The solver view runs JAXScape with each of its linear solvers, in "
        f"{' and '.join(view['dtypes'])} on the {view['device'].upper()}, so "
        "that a user can trade speed for accuracy. `dense` is the dense solve "
        f"(`solver=None`). It runs {', '.join(runs)}: free energy does not run "
        "on windows, and exact windowed resistance takes about 23 minutes a "
        "call of E2 or E3 at 1024² with `GridMultigridSolver` in float32, "
        "which keeps E2 and E3 on fewer rasters than E1. Each cell runs like a "
        f"cell of the comparison, except that a call stops after "
        f"{duration(view['timeout'])}. The error of a cell, for E1, is the "
        "largest relative error over the pairs against the reference of E1, and "
        "for E2 and E3 the relative L2 error of the map $m$, $\\lVert m - "
        "m_{\\mathrm{ref}} \\rVert_2 / \\lVert m_{\\mathrm{ref}} \\rVert_2$, "
        "against the map of exact resistance in float64 with "
        f"`{view['reference_solver']}` on the {view['device'].upper()}, on the "
        "same tiles. In E2 and E3, `SpielmanApproximation(epsilon=0.15), "
        "GridMultigridSolver(rtol=3e-4)` is the setting of the comparison for "
        "resistance: Spielman's estimate of resistance, in float32. The view "
        f"leaves out the combinations that fail by design: {left_out}."
    )


def page(run_name, records, config, environment, maryland=None):
    """Return the Benchmark page of a run.

    **Arguments:**

    - `run_name`: The name of the run directory.
    - `records`: The records of the run.
    - `config`: The configuration of the run, which gives the settings that the
      page states.
    - `environment`: The environment snapshot of the run.
    - `maryland`: The records of the Maryland figure, when the run directory
      has it.

    **Returns:**

    The page, in Markdown.
    """
    solvers = [record for record in records if record["view"] == "solver"]
    records = [record for record in records if record["view"] == "comparison"]
    questions = sorted({record["question"] for record in records})
    tools = {}
    for record in records:
        tool = tools.setdefault(record["tool"], (record["adapter"], set()))
        tool[1].add(record["question"])
    landscape, run = config["landscape"], config["run"]
    habitat = (1 - landscape["habitat_threshold"]) * 100
    texts = setups(config)

    sections = [
        "# Benchmark",
        f"This page compares JAXScape {environment['tools']['JAXScape']} with the "
        "software that ecologists use for connectivity analysis, on three "
        "questions from ecology. It is generated from the records of the run "
        f"`{run_name}`, started on {environment['date'][:10]} on "
        f"{environment['host']}. [`results.csv`](results.csv) lists every cell "
        "of the run.",
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
        f"covariance of correlation length {landscape['correlation_length']} "
        "pixels, on a periodic grid with a fixed seed, and normalizes its ranks "
        "to [0, 1]. Quality equals the field where the field is at least "
        f"{landscape['habitat_threshold']}, and 0 elsewhere, so that "
        f"{habitat:.0f} % of the cells are habitat. Permeability is "
        "$10^{-2(1 - f)}$ for the field $f$, from 0.01 to 1, and resistance is "
        "its inverse. A larger raster covers a larger area at the same grain.",
        f"{' and '.join(config['maryland']['questions'])} also run on the "
        "Maryland case of Omniscape's example: the 2016 land cover of "
        "Frederick, Maryland, from the National Land Cover Database, 694 × 725 "
        "pixels of 30 m"
        + (
            f", of which this run takes the top-left {config['maryland']['side']} "
            f"× {config['maryland']['side']} pixels"
            if config["maryland"].get("side")
            else ""
        )
        + ". Each class takes the resistance of Omniscape's reclassification "
        "table, medium- and high-intensity development are barriers, and "
        "forest, the classes of resistance 1, is habitat of quality 1.",
        "## Comparison",
        "Each cell runs in its own process, pinned to the cores of the "
        "environment table, with thread settings matched to them; XLA sizes "
        "its thread pools to these cores. A warm-up call comes first and "
        "compiles the code of tools that compile. The warm runtime is the "
        f"median of the timed calls that follow it, {run['repeats']} per cell, "
        "or the time of a single call when the warm-up takes more than "
        f"{duration(run['single_call_after'])}. Every call redoes all work that "
        "depends on the landscape's values: graph assembly, factorization, "
        "multigrid hierarchy and solver state. A call stops after "
        f"{duration(run['timeout'])}. Within a configuration, synthetic rasters "
        "run from the smallest up, and after a failure, a timeout or running "
        "out of memory (`oom`), the configuration skips its larger synthetic "
        "rasters; Maryland always runs, and its failure skips nothing. A tool "
        "runs each setting of a small grid fixed in advance from its "
        "documentation: the tables list every setting, and the figures show, "
        "at each raster, the fastest setting that succeeds. In the figures, "
        "tools on the GPU are solid and on the CPU dashed, Maryland is a star, "
        "and rasters where no setting succeeds show a status along the top "
        "edge.",
    ]
    for question in questions:
        measures = sorted(
            {r["measure"] for r in of_question(question, records) if r["measure"]}
        )
        parts = ["Warm runtime"]
        if measures:
            parts.append(f"agreement ({', '.join(measures)})")
        if question == "E2":
            parts.append("landscape total, where the tool computes one,")
        caption = ", ".join(parts[:-1]) + " and " + parts[-1] if parts[1:] else parts[0]
        sections += [
            f"### {question}. {QUESTIONS[question][0]}",
            texts[question],
            f"![{question}: warm runtime against raster cells]({question.lower()}.png)",
            f"{caption} of each configuration and raster, or the status of the "
            "cell; `missing` marks an agreement whose reference is unavailable "
            "or undefined.",
            outcome_table(of_question(question, records), COMPARISON),
        ]
        if question == MARYLAND["question"] and maryland:
            jaxscape, other = maryland
            sections += [
                f"![Maryland: JAXScape's sensitivity next to {other['tool']}'s "
                f"{MARYLAND['map']}](maryland.png)",
                "On Maryland, JAXScape's map of $\\partial C / \\partial p_k$ "
                f"under {other['distance']} ({jaxscape['device'].upper()}, "
                f"{runtime(jaxscape)}) next to {other['tool']}'s {MARYLAND['map']} "
                f"({other['device'].upper()}, `{other['solver']}`, "
                f"{runtime(other)}), each coloured by the rank of its pixels, with "
                f"barriers in gray: {RHO} = {agreement(other)}.",
            ]
    if solvers:
        sections += [
            "## Solvers",
            solver_view_text(config),
            "![Solvers: warm runtime against raster cells](solvers.png)",
            "Warm runtime, error and, in E2, landscape total of each solver, "
            "dtype and raster, or the status of the cell; `missing` marks an "
            "error whose reference is unavailable or undefined.",
            outcome_table(solvers, SOLVER_VIEW),
        ]
    gpu = environment["gpu"]
    if gpu:
        gpu = (
            f"{gpu['name']} (GPU {gpu['index']}, {gpu['uuid']}), driver "
            f"{gpu['driver']}, {gpu['memory']}"
        )
    sections += [
        "## Environment",
        table(
            ["Setting", "Value"],
            [
                ["Host", environment["host"]],
                ["GPU", gpu or "none"],
                ["CPU", environment["cpu"]],
                ["Cores", environment["cores"]],
                [
                    "Threads",
                    ", ".join(f"`{k}={v}`" for k, v in environment["threads"].items()),
                ],
                [
                    "Runtimes",
                    ", ".join(f"{k} {v}" for k, v in environment["runtimes"].items()),
                ],
                [
                    "Tools",
                    ", ".join(f"{k} {v}" for k, v in environment["tools"].items()),
                ],
            ],
        ),
        "## Parameter mappings",
        "The header comment of each adapter states how it maps the shared "
        "parameters onto its tool.",
        table(
            ["Tool", "Questions", "Adapter", "Mapping"],
            [
                [name, ", ".join(sorted(asked)), f"`{adapter}`", mapping(adapter)]
                for name, (adapter, asked) in tools.items()
            ],
        ),
        "## Excluded tools",
        table(["Tool", "Reason"], EXCLUDED),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


if __name__ == "__main__":
    main()
