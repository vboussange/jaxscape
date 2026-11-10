"""Turn the records of a run into its CSV, figures and Benchmark docs page.

    uv run python report.py RUN_DIR

writes `results.csv`, one figure per question, such as `e1.png`, the Maryland
figure `maryland.png`, the figure of the solver view `solvers.png`, and the page
`index.md` into RUN_DIR. The report reads the records, the environment
snapshot, the maps of the Maryland figure and the header comment of each
adapter that the records name.
"""

import argparse
import csv
import json
from itertools import groupby
from pathlib import Path

import matplotlib
import numpy as np
from run import cell_id, map_path
from scipy.stats import rankdata


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent

FIELDS = (
    "view question tool distance device dtype solver case cells status reason warmup_s "
    "median_s times_s measure agreement tier version files adapter"
).split()
CONFIGURATION = ("tool", "distance", "device", "dtype", "solver")
LINE = ("tool", "distance", "device")  # a line of a figure
SOLVER = ("question", "distance", "dtype", "solver")  # a row of the solver view
RHO = "Spearman's ρ"
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

SETUPS = {
    "E1": (
        "Each tool computes the distances among 32 sites drawn at random from "
        "the habitat cells with a fixed seed, across the whole raster: 496 "
        "pairs. Every tool builds the same graph: rook contiguity, and the "
        "resistance of an edge is the mean resistance of its two pixels. The "
        "agreement of a record is the largest relative error over "
        "the pairs against a float64 reference, computed once for each case and "
        "distance: a sparse direct solve with `CholmodSolver` for resistance "
        "and free energy, and Dijkstra's algorithm in SciPy for least-cost "
        "distances."
    ),
    "E2": (
        "JAXScape computes, in each call, the map of $\\partial C / \\partial q_i$ "
        "with `SensitivityAnalysis`, which equals $2 S_i$ for symmetric "
        "distances, and the total $C = \\sum_i q_i S_i$ with "
        "`ConnectivityAnalysis`. Its windows have `dependency_range` 100 and "
        "`coarsening_factor` 0.2, which gives blocks of 21 pixels: Omniscape's "
        "`radius` and `block_size` in its Maryland example. Its proximity is "
        "the default, $e^{-r/25}$ with $r$ in equivalent pixels. The windowed "
        "analyses build their graphs with JAXScape's own edge rule: rook "
        "contiguity, and the conductance of an edge is the mean permeability "
        "of its two pixels. Resistance takes the setting of the [connectivity "
        "example](../examples/connectivity_analysis/connectivity_analysis.ipynb): "
        "`GridMultigridSolver(rtol=3e-4)` and "
        "`SpielmanApproximation(epsilon=0.15)`, which estimates resistances "
        "from random projections, on tiles of up to 2,601 block centres: at "
        "8192², 64 tiles, the fewest at which E3 runs fastest on the GPU. Exact "
        "resistance took 24 minutes a call of E3 at 1024², and would take 32 "
        "hours at 8192². Free energy is left out of E2 and E3: its window "
        "query holds the dense inverse of the 40,401 pixels of a window, and in "
        "float64, the only precision that resolves its windows, one window ran "
        "out of the 24 GB of the GPU, and out of memory on the CPU after taking "
        "300 GB. The agreement of another tool is Spearman's ρ between its map "
        "and JAXScape's map under the distance of its row, over the pixels "
        "where both maps are defined. ConScape.jl computes its maps under free "
        "energy on the whole raster, with targets at landmarks every 21 pixels "
        "and the kernel $e^{-d/25}$ on its free-energy distance $d$; JAXScape "
        "has no free-energy map to compare them with, and they are compared "
        "with its maps under least-cost distance, which ranks pairs of sites "
        "most like free energy at the $\\theta$ of the run."
    ),
    "E3": (
        "JAXScape computes the map of $\\partial C / \\partial p_k$ with "
        "`SensitivityAnalysis`, with the windows and proximity of E2. The "
        "other tools map a flow, and the agreement of each is Spearman's ρ "
        "over pixels between its map and the map of JAXScape under the "
        "distance of its row, or under least-cost distance for ConScape.jl "
        "(see E2). Omniscape.jl computes its normalized current, the "
        "cumulative current divided by the flow potential, with the "
        "configuration of its Maryland example on JAXScape's windows, on a "
        "graph of eight neighbours."
    ),
}

SOLVERS = (
    "The solver view runs JAXScape with each of its linear solvers, in float32 "
    "and float64 on the GPU, on the cases of E1, E2 and E3, so that a user can "
    "trade speed for accuracy. `dense` is the dense solve (`solver=None`). "
    "Resistance runs on every question, and free energy on E1 alone, since it "
    "does not run on windows. Each cell runs like a cell of the comparison, "
    "except that a call stops after 15 minutes, and on fewer rasters: E1 on "
    "256² and 1024², and E2 and E3 on 256², since exact windowed resistance "
    "takes about 23 minutes a call at 1024² with `GridMultigridSolver` in "
    "float32. The error of a cell, for E1, is the largest relative error over "
    "the pairs against the reference of E1, and for E2 and E3 the relative L2 "
    "error of the map $m$, $\\lVert m - m_{\\mathrm{ref}} \\rVert_2 / "
    "\\lVert m_{\\mathrm{ref}} \\rVert_2$, against the map of exact "
    "resistance in float64 with `CuDSSSolver` on the GPU, on the same tiles. "
    "In E2 and E3, `SpielmanApproximation(epsilon=0.15), "
    "GridMultigridSolver(rtol=3e-4)` is the setting of the comparison for "
    "resistance: Spielman's estimate of resistance, in float32."
)

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


def view(record):
    """Return the view of a record; a record without one belongs to the
    comparison."""
    return record.get("view", "comparison")


def is_synthetic(record):
    return record["case"].startswith("synthetic-")


def rank(record):
    """Order the records of a raster: the fastest that succeeds first, then the
    others by their status, in the order of `STATUS_MARKERS`."""
    if record["status"] == "ok":
        return (0, record["median_s"])
    return (1, list(STATUS_MARKERS).index(record["status"]))


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
    comparison = [record for record in records if view(record) == "comparison"]
    questions = sorted({record["question"] for record in comparison})
    for question in questions:
        plot(question, of(question, comparison), run_dir / f"{question.lower()}.png")
    solvers = [record for record in records if view(record) == "solver"]
    if solvers:
        plot_solvers(solvers, run_dir / "solvers.png")
    # The maps are not committed: without them, the figure drawn before stays.
    pair = maryland_pair(comparison)
    if pair and all(path.exists() for path in maryland_maps(run_dir, *pair)):
        plot_maryland(run_dir, *pair, run_dir / "maryland.png")
    figure = pair if (run_dir / "maryland.png").exists() else None
    (run_dir / "index.md").write_text(page(run_dir.name, records, environment, figure))


def of(question, records):
    return [record for record in records if record["question"] == question]


def write_csv(records, path):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            times = record.get("times_s") or []
            writer.writerow({**record, "times_s": " ".join(map(str, times))})


def configurations(records, fields=CONFIGURATION):
    """Group records by the values of `fields`, in the order of their first record."""
    order = {}
    for record in records:
        order.setdefault(tuple(record[field] for field in fields), len(order))

    def key(record):
        return order[tuple(record[field] for field in fields)]

    for _, group in groupby(sorted(records, key=key), key=key):
        group = sorted(group, key=lambda record: record["cells"])
        yield dict(zip(fields, (group[0][f] for f in fields))), group


def plot(question, records, path):
    """Plot the warm runtime of each tool, distance and device against the
    raster cells.

    At each raster, a line takes the fastest of the tool's settings that
    succeeds. GPU lines are solid and CPU lines dashed, and join the synthetic
    rasters; Maryland is a star. A raster where no setting succeeds shows a
    status as a marker along the top edge, one row per line.
    """
    figure, axes = plt.subplots(figsize=(8, 4.5))
    colors = {}
    for index, (configuration, group) in enumerate(configurations(records, LINE)):
        color = colors.setdefault(
            (configuration["tool"], configuration["distance"]), f"C{len(colors) % 10}"
        )
        group = [
            min(of_raster, key=rank)
            for _, of_raster in groupby(group, key=lambda record: record["cells"])
        ]
        ok = [record for record in group if record["status"] == "ok"]
        synthetic = [record for record in ok if is_synthetic(record)]
        tier = f" · Tier {group[0]['tier']}" if group[0]["tier"] else ""
        axes.plot(
            [record["cells"] for record in synthetic],
            [record["median_s"] for record in synthetic],
            "-" if configuration["device"] == "gpu" else "--",
            marker="o",
            color=color,
            label=f"{configuration['tool']} · {configuration['distance']} · "
            f"{configuration['device'].upper()}{tier}",
        )
        for record in ok:
            if not is_synthetic(record):
                axes.plot(record["cells"], record["median_s"], "*", color=color)
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
    if not all(map(is_synthetic, records)):
        maryland = dict(linestyle="", marker="*", color="gray", label="Maryland")
        statuses.insert(0, Line2D([], [], **maryland))
    handles = axes.get_legend_handles_labels()[0] + statuses
    axes.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_solvers(records, path):
    """Plot the warm runtime of each solver and dtype against the raster cells,
    one panel per question and distance.

    float32 lines are solid and float64 dashed, and join the synthetic rasters;
    Maryland is a star. A raster where a solver does not succeed shows its
    status as a marker along the top edge, one row per line.
    """
    panels = list(configurations(records, ("question", "distance")))
    solvers = list(dict.fromkeys(record["solver"] for record in records))
    columns = min(2, len(panels))
    rows = -(-len(panels) // columns)
    figure, grid = plt.subplots(
        rows, columns, figsize=(5 * columns, 3.6 * rows), squeeze=False
    )
    for axes in grid.flat[len(panels) :]:
        axes.set_visible(False)
    for axes, (panel, group) in zip(grid.flat, panels):
        band = 0
        for line, of_line in configurations(group, ("solver", "dtype")):
            color = f"C{solvers.index(line['solver']) % 10}"
            ok = [record for record in of_line if record["status"] == "ok"]
            synthetic = [record for record in ok if is_synthetic(record)]
            axes.plot(
                [record["cells"] for record in synthetic],
                [record["median_s"] for record in synthetic],
                "-" if line["dtype"] == "float32" else "--",
                marker="o",
                color=color,
            )
            for record in ok:
                if not is_synthetic(record):
                    axes.plot(record["cells"], record["median_s"], "*", color=color)
            failed = [record for record in of_line if record["status"] != "ok"]
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
        axes.set_title(f"{panel['question']} · {panel['distance']}")
    handles = [
        Line2D([], [], color=f"C{index % 10}", label=solver)
        for index, solver in enumerate(solvers)
    ]
    handles += [
        Line2D([], [], color="gray", linestyle=style, label=dtype)
        for dtype, style in (("float32", "-"), ("float64", "--"))
    ]
    if not all(map(is_synthetic, records)):
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
    """Return the records of the Maryland figure: the first JAXScape record to
    succeed under the tool's distance, whose map the runner keeps and compares
    the tool's maps with, and the fastest setting of the tool that succeeds.
    Return `None` when either is missing."""
    question, tool = MARYLAND["question"], MARYLAND["tool"]
    on_maryland = [
        r
        for r in of(question, records)
        if r["case"] == "maryland" and r["status"] == "ok"
    ]
    others = [r for r in on_maryland if r["tool"] == tool]
    if not others:
        return None
    other = min(others, key=rank)
    for record in on_maryland:
        if record["tool"] == "JAXScape" and record["distance"] == other["distance"]:
            return record, other
    return None


def maryland_maps(run_dir, jaxscape, other):
    """Return the paths of the maps of the Maryland figure, which the runner
    keeps."""
    return (
        map_path(run_dir, jaxscape, jaxscape["distance"]),
        run_dir / "maps" / f"{cell_id(other)}.npy",
    )


def runtime(record):
    """Return the warm runtime of a record, or its status when it has none."""
    if record["status"] != "ok":
        return record["status"]
    return f"{record['median_s']:.3g} s"


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
    maps = maryland_maps(run_dir, jaxscape, other)
    for axis, record, label, map_ in zip(axes, (jaxscape, other), labels, maps):
        axis.set_axis_off()
        axis.set_title(
            f"{record['tool']}: {label}\n{record['device'].upper()}, {runtime(record)}"
        )
        image = axis.imshow(quantiles(np.load(map_)), cmap=colormap, vmin=0, vmax=1)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Rank of the pixel (quantile)")
    figure.suptitle(f"Maryland · {RHO} = {value(other)}")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def row(cells):
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def table(header, rows):
    lines = [row(header), row(["---"] * len(header))]
    return "\n".join(lines + [row(cells) for cells in rows])


def case_label(case):
    side = case.removeprefix("synthetic-")
    return f"{side}²" if side != case else case.capitalize()


def value(record):
    """Return the agreement of a record, or its status when it has none."""
    if record["status"] != "ok" or not record["tier"]:
        return record["status"]
    if record["agreement"] is None:
        return "missing"
    if record["tier"] > 1:
        return f"{record['agreement']:.2f}"
    return f"{record['agreement']:.1e}"


def agreement_table(records):
    cases = sorted({(r["cells"], r["case"]) for r in records})
    rows = []
    for configuration, group in configurations(records):
        by_case = {record["case"]: value(record) for record in group}
        rows.append(
            [configuration[field] for field in CONFIGURATION]
            + [group[0]["tier"] or ""]
            + [by_case.get(case, "") for _, case in cases]
        )
    rows = [[cells[0], cells[1], cells[2].upper(), *cells[3:]] for cells in rows]
    header = ["Tool", "Distance", "Device", "dtype", "Solver", "Tier"]
    return table(header + [case_label(case) for _, case in cases], rows)


def runtime_and_error(record):
    """Return the warm runtime and the error of a record, or its status."""
    if record["status"] != "ok":
        return record["status"]
    error = "missing" if record["agreement"] is None else f"{record['agreement']:.1e}"
    return f"{runtime(record)} · {error}"


def solver_table(records):
    cases = sorted({(r["cells"], r["case"]) for r in records})
    rows = [
        list(configuration.values())
        + [
            next((runtime_and_error(r) for r in group if r["case"] == case), "")
            for _, case in cases
        ]
        for configuration, group in configurations(records, SOLVER)
    ]
    header = ["Question", "Distance", "dtype", "Solver"]
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


def page(run_name, records, environment, maryland=None):
    """Return the Benchmark page; `maryland` holds the records of the Maryland
    figure, when the run directory has it."""
    solvers = [record for record in records if view(record) == "solver"]
    records = [record for record in records if view(record) == "comparison"]
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
        question: sorted(
            {record["measure"] for record in of(question, records) if record["tier"]}
        )
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
        "E2 and E3 also run on the Maryland case of Omniscape's example: the "
        "2016 land cover of Frederick, Maryland, from the National Land Cover "
        "Database, 694 × 725 pixels of 30 m. Each class takes the resistance "
        "of Omniscape's reclassification table, medium- and high-intensity "
        "development are barriers, and forest, the classes of resistance 1, is "
        "habitat of quality 1.",
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
        "rasters. A tool runs each setting of a small grid fixed in advance "
        "from its documentation: the tables list every setting, and the "
        "figures show, at each raster, the fastest setting that succeeds. In "
        "the figures, tools on the GPU are solid and on the CPU dashed, "
        "Maryland is a star, and rasters where no setting succeeds show a "
        "status along the top edge.",
    ]
    for question in questions:
        records_of_question = of(question, records)
        sections += [
            f"### {question}. {QUESTIONS[question][0]}",
            SETUPS.get(question, ""),
            f"![{question}: warm runtime against raster cells]({question.lower()}.png)",
            f"Agreement ({', '.join(measures[question])}) of each configuration "
            "and raster, or the status of the cell; `missing` marks a cell "
            "whose reference is unavailable."
            if measures[question]
            else "Status of each configuration and raster.",
            agreement_table(records_of_question),
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
                f"barriers in gray: {RHO} = {value(other)}.",
            ]
    if solvers:
        sections += [
            "## Solvers",
            SOLVERS,
            "![Solvers: warm runtime against raster cells](solvers.png)",
            "Warm runtime and error of each solver, dtype and raster, or the "
            "status of the cell; `missing` marks a cell whose reference is "
            "unavailable.",
            solver_table(solvers),
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
