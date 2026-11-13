"""Turn the records of a run into its figures and Benchmark docs page.

    uv run python report.py RUN_DIR              # report the run
    uv run python report.py RUN_DIR --publish    # and make it the docs page

writes one figure per question, such as `distance.png`, the Maryland figure
`maryland.png`, the two figures of the solver view, `solvers.png` and
`solver-errors.png`, and the page `index.md` into RUN_DIR. The report reads the
records, the configuration and the environment snapshot of the run, the maps of
the Maryland figure, and the header comment of each adapter that the records
name. `--publish` points `docs/benchmark`, the Benchmark page of the docs, at
RUN_DIR.
"""

import argparse
import json
import os
import textwrap
from itertools import groupby
from pathlib import Path

import matplotlib
import numpy as np
from run import map_path, read_records, RHO
from scipy.stats import rankdata


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter


HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs" / "benchmark"  # the Benchmark page of the docs
EXAMPLE = "../examples/connectivity_analysis/connectivity_analysis.ipynb"

# The questions of the benchmark, and the title of the section of each.
TASKS = {
    "distance": "Distance calculation",
    "connectivity": "Connectivity analysis",
    "sensitivity": "Sensitivity analysis",
}
# The names of distances and labels on the page, where they differ.
NAMES = {
    "lcp": "least-cost path",
    "free_energy": "free energy",
    "resistance, K = -R, normalized": "normalized sensitivity (K = −R)",
}

# The fields of a setting in the tables of the comparison and of the solver
# view, and their headers.
COMPARISON = ("tool", "distance", "device", "dtype", "solver", "tier")
SOLVER_VIEW = ("question", "distance", "dtype", "solver")
HEADERS = {
    "question": "Task",
    "tool": "Tool",
    "distance": "Distance",
    "device": "Device",
    "dtype": "dtype",
    "solver": "Solver",
    "tier": "Tier",
}
LINE = ("tool", "distance", "device")  # a line of a figure of the comparison
SOLID = {"gpu", "float32"}  # the lines of the figures drawn solid, others dashed
# The marker of each raster side in the figure of runtime against error.
SHAPES = {256: "o", 1024: "s", 4096: "^", 8192: "D", None: "*"}
SMALLEST_ERROR = 1e-16  # errors below it are drawn at it, on a log scale
# The Maryland figure: JAXScape's map of a question next to a tool's map.
MARYLAND = {
    "question": "sensitivity",
    "tool": "Omniscape.jl",
    "map": "normalized current",
}
BLUES = "#cde2fb #9ec5f4 #6da7ec #3987e5 #256abf #184f95 #0d366b".split()
# Figures 8 inches wide, at 200 dpi, keep their text legible at the width of
# the docs page.
plt.rcParams.update({"font.size": 10, "legend.fontsize": 9})


def main():
    """Report a run from the command line, as the module docstring shows."""
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
    """Write the figures and the page of a run into its directory.

    **Arguments:**

    - `run_dir`: The run directory, with `records.csv`, `config.json`,
      `environment.json` and, when they are kept, the maps of the Maryland
      figure. Without the maps, the Maryland figure drawn before stays.
    """
    records = read_records(run_dir / "records.csv")
    config = json.loads((run_dir / "config.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())

    comparison = [record for record in records if record["view"] == "comparison"]
    for question in TASKS:
        if of_question(question, comparison):
            path = run_dir / f"{question}.png"
            plot_runtimes(of_question(question, comparison), ("question",), LINE, path)
    solvers = solver_view(records)
    if solvers:
        panel, line = ("question", "distance"), ("solver", "dtype")
        plot_runtimes(solvers, panel, line, run_dir / "solvers.png")
        plot_errors(solvers, run_dir / "solver-errors.png")
    pair = maryland_pair(comparison)
    if pair and all(map_path(run_dir, record).exists() for record in pair):
        plot_maryland(run_dir, *pair, run_dir / "maryland.png")
    figure = pair if (run_dir / "maryland.png").exists() else None
    (run_dir / "index.md").write_text(page(records, config, environment, figure))


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


def solver_view(records):
    """Return the records of the solver view, by question in the order of `TASKS`."""
    solvers = [record for record in records if record["view"] == "solver"]
    return sorted(solvers, key=lambda record: list(TASKS).index(record["question"]))


def name(value):
    """Return the name on the page of a distance, label or solver of the records."""
    return NAMES.get(value, value or "")


def side(case):
    """Return the side of a case: 256 for `synthetic-256`, `None` for Maryland."""
    prefix = "synthetic-"
    return int(case.removeprefix(prefix)) if case.startswith(prefix) else None


def case_label(case):
    """Return the label of a case on the page, such as `256²`."""
    return f"{side(case)}²" if side(case) else case.capitalize()


def group_by(records, fields):
    """Group records by the values of `fields`.

    **Arguments:**

    - `records`: The records.
    - `fields`: The fields whose values make a group.

    **Yields:**

    The values of `fields` and the records that share them, sorted by raster
    cells, in the order of the first record of each group.
    """
    order = {}
    for record in records:
        order.setdefault(tuple(record[field] for field in fields), len(order))

    def key(record):
        return order[tuple(record[field] for field in fields)]

    for _, group in groupby(sorted(records, key=key), key=key):
        group = sorted(group, key=lambda record: (record["cells"], record["case"]))
        yield {field: group[0][field] for field in fields}, group


def panel_title(values):
    """Return the title of a panel, such as `Distance calculation · resistance`."""
    return " · ".join(
        TASKS[value] if field == "question" else name(value)
        for field, value in values.items()
    )


def panels_figure(count):
    """Return a figure 8 inches wide with `count` panels, two per row."""
    columns = min(2, count)
    rows = -(-count // columns)
    figure, grid = plt.subplots(
        rows,
        columns,
        figsize=(8, 5.5 if rows == 1 else 3.6 * rows),
        squeeze=False,
        layout="constrained",
    )
    for axes in grid.flat[count:]:
        axes.set_visible(False)
    return figure, grid.flat


def log_axes(axes, values, xlabel):
    """Label a panel of `values` on log axes, without labels on minor ticks."""
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Warm runtime (s)")
    axes.set_title(panel_title(values))


def save(figure, colors, styles, path):
    """Save a figure with the legends of its colours and styles below its panels."""
    for handles, corner, x in ((colors, "upper left", 0), (styles, "upper right", 1)):
        figure.legend(handles=handles, loc=corner, bbox_to_anchor=(x, 0), frameon=False)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def style_of(value, color):
    """Return the style of a line of a device or dtype, in a colour.

    GPU and float32 lines are solid with filled markers, and CPU and float64
    lines dashed with hollow markers.
    """
    solid = value in SOLID
    return dict(
        color=color,
        linestyle="-" if solid else "--",
        markerfacecolor=color if solid else "none",
    )


def plot_runtimes(records, panel, line, path):
    """Plot the warm runtime of each line against the raster cells.

    At each raster, a line takes the fastest of its records that succeed, and
    joins the synthetic rasters; Maryland is a star. The tables give the
    records that did not succeed.

    **Arguments:**

    - `records`: The records to plot.
    - `panel`: The fields whose values give a panel, such as `("question",)`.
    - `line`: The fields whose values give a line. The last, the device or the
      dtype, sets its style: GPU and float32 lines are solid with filled
      markers, and CPU and float64 lines dashed with hollow markers. The others
      set its colour, and the legend gives their values, with the tier of a
      tool.
    - `path`: Where to save the figure.
    """
    panels = list(group_by(records, panel))
    colors = list(dict.fromkeys(tuple(r[f] for f in line[:-1]) for r in records))
    tiers = {tuple(r[f] for f in line[:-1]): r["tier"] for r in records}
    figure, grid = panels_figure(len(panels))
    for axes, (values, group) in zip(grid, panels):
        for of_line, records_of_line in group_by(group, line):
            color = f"C{colors.index(tuple(of_line.values())[:-1]) % 10}"
            style = style_of(of_line[line[-1]], color)
            ok = [record for record in records_of_line if record["status"] == "ok"]
            best = [
                min(of_raster, key=lambda record: record["median_s"])
                for _, of_raster in groupby(ok, key=lambda record: record["case"])
            ]
            synthetic = [record for record in best if side(record["case"])]
            axes.plot(
                [record["cells"] for record in synthetic],
                [record["median_s"] for record in synthetic],
                marker="o",
                **style,
            )
            for record in best:
                if not side(record["case"]):
                    star = style | {"linestyle": ""}
                    axes.plot(record["cells"], record["median_s"], "*", ms=12, **star)
        cells = [record["cells"] for record in group]
        axes.set_xlim(min(cells) / 2, max(cells) * 2)
        log_axes(axes, values, "Raster cells")
    drawn = {tuple(r[f] for f in line[:-1]) for r in records if r["status"] == "ok"}
    handles = [
        Line2D(
            [],
            [],
            color=f"C{index % 10}",
            label=" · ".join(map(name, key))
            + (f" · Tier {tiers[key]}" if tiers[key] else ""),
        )
        for index, key in enumerate(colors)
        if key in drawn
    ]
    styles = sorted({r[line[-1]] for r in records}, key=lambda v: v not in SOLID)
    styled = [
        Line2D(
            [],
            [],
            marker="o",
            label=style.upper() if line[-1] == "device" else style,
            **style_of(style, "gray"),
        )
        for style in styles
    ]
    if not all(side(record["case"]) for record in records):
        star = dict(linestyle="", marker="*", ms=12, color="gray", label="Maryland")
        styled.append(Line2D([], [], **star))
    save(figure, handles, styled, path)


def plot_errors(records, path):
    """Plot the warm runtime of each record of the solver view against its error.

    Only records with a measured error appear: the colour gives the solver, a
    filled marker float32 and a hollow one float64, and the shape the raster.
    Errors below `SMALLEST_ERROR` are drawn at it.
    """
    solvers = list(dict.fromkeys(record["solver"] for record in records))
    measured = [
        record
        for record in records
        if record["status"] == "ok" and record["agreement"] is not None
    ]
    panels = list(group_by(measured, ("question", "distance")))
    figure, grid = panels_figure(len(panels))
    for axes, (values, group) in zip(grid, panels):
        for record in group:
            color = f"C{solvers.index(record['solver']) % 10}"
            axes.plot(
                max(record["agreement"], SMALLEST_ERROR),
                record["median_s"],
                SHAPES[side(record["case"])],
                ms=7,
                color=color,
                markerfacecolor=color if record["dtype"] == "float32" else "none",
            )
        measure = group[0]["measure"]
        log_axes(axes, values, measure[0].upper() + measure[1:])
    handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="s",
            color=f"C{solvers.index(solver) % 10}",
            label=name(solver),
        )
        for solver in dict.fromkeys(record["solver"] for record in measured)
    ]
    gray = dict(linestyle="", color="gray")
    styled = [
        Line2D([], [], **gray, marker="o", label="float32"),
        Line2D([], [], **gray, marker="o", markerfacecolor="none", label="float64"),
    ]
    sides = sorted({side(r["case"]) for r in measured}, key=lambda s: s or np.inf)
    styled += [
        Line2D([], [], **gray, marker=SHAPES[s], label=f"{s}²" if s else "Maryland")
        for s in sides
    ]
    save(figure, handles, styled, path)


def maryland_pair(records):
    """Return the records of the Maryland figure, or `None` when one is missing.

    They are the JAXScape GPU record under the tool's `compared_with`, whose map
    the tool's maps are compared with, and the fastest setting of the tool that
    succeeds.
    """
    on_maryland = [
        record
        for record in of_question(MARYLAND["question"], records)
        if record["case"] == "maryland" and record["status"] == "ok"
    ]
    others = [record for record in on_maryland if record["tool"] == MARYLAND["tool"]]
    if not others:
        return None
    other = min(others, key=lambda record: record["median_s"])
    paired = ("JAXScape", "gpu", other["compared_with"])
    for record in on_maryland:
        if (record["tool"], record["device"], record["distance"]) == paired:
            return record, other
    return None


def quantiles(values):
    """Return the rank of each finite pixel as a quantile in [0, 1], else NaN."""
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
    figure, axes = plt.subplots(1, 2, figsize=(8, 4.6), layout="constrained")
    labels = (name(jaxscape["distance"]), MARYLAND["map"])
    for axis, record, label in zip(axes, (jaxscape, other), labels):
        axis.set_axis_off()
        axis.set_title(
            f"{record['tool']}, {record['device'].upper()}, {runtime(record)}\n{label}"
        )
        quantile = quantiles(np.load(map_path(run_dir, record)))
        image = axis.imshow(quantile, cmap=colormap, vmin=0, vmax=1)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Rank of the pixel (quantile)")
    figure.suptitle(f"Maryland · {RHO} = {agreement(other)}")
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def runtime(record):
    """Return the warm runtime of a record that succeeded."""
    return f"{record['median_s']:.3g} s"


def agreement(record):
    """Return the agreement of a record, Spearman's ρ or an error, or `missing`."""
    if record["agreement"] is None:
        return "missing"
    if record["measure"] == RHO:
        return f"{record['agreement']:.2f}"
    return f"{record['agreement']:.1e}"


def outcome(record):
    """Return the text of a record that succeeded, in a table.

    The text is its warm runtime, its agreement where it is measured, and its
    landscape total C where it is computed.
    """
    parts = [runtime(record)]
    if record["measure"] == RHO:
        text = agreement(record)
        parts.append("ρ missing" if text == "missing" else f"ρ = {text}")
    elif record["measure"]:
        parts.append(agreement(record))
    if record["total"] is not None:
        parts.append(f"C = {record['total']:.3g}")
    return " · ".join(parts)


def row(cells):
    """Return a row of a Markdown table."""
    return "| " + " | ".join(str(cell) for cell in cells) + " |"


def table(header, rows):
    """Return a Markdown table."""
    lines = [row(header), row(["---"] * len(header))]
    return "\n".join(lines + [row(cells) for cells in rows])


def collapsed(title, *blocks):
    """Return a note collapsed by default, with its title and Markdown blocks."""
    body = "\n\n".join(block for block in blocks if block)
    return f'??? note "{title}"\n\n' + textwrap.indent(body, "    ")


def outcome_table(records, fields):
    """Return the table of the outcome of each setting on each raster.

    **Arguments:**

    - `records`: The records of the table.
    - `fields`: The fields whose values make a setting, a row.

    **Returns:**

    The table, in Markdown, with one column per raster, followed by the
    numbered reasons of the cells that did not succeed. Such a cell gives its
    status and the number of its reason; the reasons are numbered in the order
    in which they first appear in the table.
    """
    cases = sorted({(record["cells"], record["case"]) for record in records})
    reasons = {}  # reason -> its number
    rows = []
    for values, group in group_by(records, fields):
        by_case = {}
        for record in group:
            text = outcome(record) if record["status"] == "ok" else record["status"]
            if record["status"] != "ok":
                number = reasons.setdefault(record["reason"], len(reasons) + 1)
                text = f"{text} ({number})"
            by_case[record["case"]] = text
        setting = [
            value.upper()
            if field == "device"
            else name(value)
            if field in ("distance", "solver")
            else value or ""
            for field, value in values.items()
        ]
        rows.append(setting + [by_case.get(case, "") for _, case in cases])
    header = [HEADERS[field] for field in fields]
    header += [case_label(case) for _, case in cases]
    listed = [
        f"{number}. `{str(reason).replace('`', "'")}`"
        for reason, number in reasons.items()
    ]
    return "\n\n".join([table(header, rows), "\n".join(listed)])


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


def jaxscape_setting(config, question, distance, setting):
    """Return a setting of JAXScape's entry of a question for a distance."""
    for entry in config["tools"]:
        if (entry["tool"], entry["question"]) == ("JAXScape", question):
            return entry["distances"][distance][setting]


def code(value):
    """Return a setting as inline code, under its name on the page."""
    return f"`{name(value)}`"


def either(values, joiner="or"):
    """Join values in prose, such as `a, b or c`."""
    values = list(values)
    return f" {joiner} ".join(filter(None, [", ".join(values[:-1]), values[-1]]))


def combination(entry):
    """Return the text of an entry of `fails_by_design`.

    For `{distance = "free_energy", dtype = "float32"}`, the text is
    `free energy in float32`.
    """
    words = []
    for field, joiner, show in (
        ("distance", "", name),
        ("solver", "with ", code),
        ("dtype", "in ", str),
        ("question", "in the ", lambda question: TASKS[question].lower()),
    ):
        if field in entry:
            values = entry[field] if isinstance(entry[field], list) else [entry[field]]
            words.append((joiner if words else "") + either(map(show, values)))
    return " ".join(words)


def introduction():
    """Return the introduction of the page."""
    return [
        "# Benchmark",
        "This page compares JAXScape with the software that ecologists use for "
        "connectivity analysis on three tasks, distance calculation, "
        "connectivity analysis and sensitivity analysis, and compares the "
        "linear solvers of JAXScape. [`records.csv`](records.csv) holds the "
        "record of every cell of the run, and [Setup](#setup) states the "
        "hardware, the timing protocol, the settings and the version of each "
        "tool.",
        "Each comparison carries a tier of comparability. Tier 1 computes the "
        "same quantity as JAXScape, and the largest relative error against a "
        "float64 reference measures its accuracy; only at Tier 1 do two "
        "runtimes measure the same computation. Tier 2 computes the same "
        "quantity on another graph or with another kernel, and Tier 3 a "
        "related quantity that answers the same question; both report "
        "Spearman's rank correlation ρ between the map of the tool and that of "
        "JAXScape.",
        "In the figures of the three tasks, a line gives the warm runtime of a "
        "tool and distance against the number of raster cells, at each raster "
        "that of its fastest setting that succeeded: solid with filled markers "
        "on the GPU, dashed with hollow markers on the CPU, and a star on "
        "Maryland. The table under each figure lists "
        "every setting, and the status and recorded reason of each cell that "
        "did not succeed.",
    ]


def distance_section(records, config):
    """Return the section of the distance calculation."""
    sites = config["distance"]["sites"]
    theta = config["distances"]["free_energy"]["theta"]
    return [
        "## Distance calculation",
        "The effective distance $d_{ij}$ between two habitat sites accounts for "
        "the landscape between them, and replaces Euclidean distance in "
        "dispersal kernels and in models of isolation by distance. Each tool "
        f"computes the distances among {sites} sites drawn at random from the "
        f"habitat pixels, across the whole raster: {sites * (sites - 1) // 2} "
        "pairs. Every tool builds the same graph: rook contiguity, with the "
        "mean resistance of two pixels as the resistance of their edge. Free "
        f"energy takes $\\theta = {theta}$. The error of a cell is the largest "
        "relative error over the pairs against a float64 reference: a sparse "
        "direct solve with `CholmodSolver` for resistance and free energy, and "
        "Dijkstra's algorithm in SciPy for least-cost paths.",
        "![Distance calculation: warm runtime against raster cells](distance.png)",
        collapsed(
            "Warm runtime (s) · largest relative error against the float64 reference",
            "Each cell gives the warm runtime and the largest relative error "
            "over the pairs of sites, `missing` without a reference.",
            outcome_table(records, COMPARISON),
        ),
    ]


def connectivity_section(records, config):
    """Return the section of the connectivity analysis."""
    windows = config["windows"]
    scale = f"{windows['dependency_range'] / 4:g}"

    def setting(name):
        return jaxscape_setting(config, "connectivity", "resistance", name)

    return [
        "## Connectivity analysis",
        "Connectivity measures how much habitat each pixel reaches through the "
        "landscape: Hanski's connectivity $S_i = \\sum_j K(d_{ij})\\, q_j$, "
        "with the habitat quality $q_j$ and the proximity $K$, and its "
        "landscape total $C = \\sum_i q_i S_i$, the quantity behind the "
        "equivalent connected area. In each call, JAXScape computes the map of "
        "$\\partial C / \\partial q_i$, which equals $2 S_i$ for symmetric "
        "distances, with `SensitivityAnalysis`, and the total $C$ with "
        "`ConnectivityAnalysis`. Both are moving-window analyses, on windows of "
        f"`dependency_range` {windows['dependency_range']} pixels with "
        f"`coarsening_factor` {windows['coarsening_factor']}, which gives "
        f"blocks of {windows['block_size']} pixels, with the proximity "
        f"$e^{{-r/{scale}}}$, $r$ in equivalent pixels; they build their graphs "
        "with rook contiguity, with the mean permeability of two pixels as the "
        f"conductance of their edge. Resistance takes {code(setting('method'))} "
        f"with {code(setting('solver'))}, the setting of the [connectivity "
        f"example]({EXAMPLE}), on tiles of up to {setting('batch_size'):,} block "
        "centres. The map of each other tool is compared with that of "
        "JAXScape's GPU cell under the distance of its row, over the pixels "
        "where both are defined; ConScape.jl, under free energy, is compared "
        "with JAXScape under least-cost path. Graphab gives the square of its "
        "equivalent connectivity, $\\sum_I \\sum_J a_I a_J "
        f"e^{{-d_{{IJ}}/{scale}}}$ over its habitat patches, as its total $C$.",
        "![Connectivity analysis: warm runtime against raster cells](connectivity.png)",
        collapsed(
            "Warm runtime (s) · Spearman ρ with JAXScape's map · landscape total C",
            "Each cell gives the warm runtime, Spearman's ρ between the map of "
            "the tool and that of JAXScape, `missing` without JAXScape's map or "
            "where ρ is undefined, and the landscape total $C$ of a tool that "
            "computes one.",
            outcome_table(records, COMPARISON),
        ),
    ]


def sensitivity_section(records, maryland):
    """Return the section of the sensitivity analysis."""
    normalized = NAMES["resistance, K = -R, normalized"]
    sections = [
        "## Sensitivity analysis",
        "The sensitivity $\\partial C / \\partial p_k$ of connectivity to the "
        "permeability $p_k$ of each pixel locates corridors and pinch points: "
        "the pixels whose loss of permeability reduces connectivity the most. "
        "JAXScape computes it with `SensitivityAnalysis`, with the windows and "
        "settings of the connectivity analysis; the other tools map a flow, "
        f"such as current, betweenness or visits. JAXScape's {normalized} is "
        "the map of the section [Comparison with Omniscape's normalized "
        f"current]({EXAMPLE}#comparison-with-omniscapes-normalized-current) of "
        "the connectivity example: under resistance with the proximity "
        "$K(R) = -R$, which weighs every pair of habitat pixels equally, "
        "$p_k^2\\, \\partial C / \\partial p_k = -\\partial C / \\partial r_k$ "
        "divided by $\\partial C / \\partial p_k$ on a landscape of "
        "permeability 1 with the same quality, and 0 where the latter is not "
        "positive; each call computes both sensitivities. Omniscape.jl "
        "computes its normalized current with the configuration of its "
        "Maryland example on the same windows, on a graph of eight neighbours, "
        f"and is compared with the {normalized}. ConScape.jl is compared with "
        "least-cost path, and each other tool with the distance of its row.",
        "![Sensitivity analysis: warm runtime against raster cells](sensitivity.png)",
        collapsed(
            "Warm runtime (s) · Spearman ρ with JAXScape's map",
            "Each cell gives the warm runtime and Spearman's ρ between the map "
            "of the tool and that of JAXScape, `missing` without JAXScape's map "
            "or where ρ is undefined.",
            outcome_table(records, COMPARISON),
        ),
    ]
    if maryland:
        jaxscape, other = maryland
        sections += [
            f"![Maryland: JAXScape's {name(jaxscape['distance'])} and "
            f"{other['tool']}'s {MARYLAND['map']}](maryland.png)",
            f"On Maryland, JAXScape's {name(jaxscape['distance'])} "
            f"({jaxscape['device'].upper()}, {runtime(jaxscape)}) and "
            f"{other['tool']}'s {MARYLAND['map']} "
            f"({other['device'].upper()}, `{other['solver']}`, "
            f"{runtime(other)}), each coloured by the rank of its pixels, with "
            f"barriers in gray: {RHO} = {agreement(other)}.",
        ]
    return sections


def solver_section(records, config):
    """Return the section of the solver view."""
    view = config["solver_view"]
    exact = {
        case
        for distances in view["questions"].values()
        for settings in distances.values()
        for case in settings.get("cases", [])
    }
    exact = either(case_label(case) for case in sorted(exact, key=side))
    approximate = [entry for entry in view["solvers"].values() if "method" in entry]
    methods = either(dict.fromkeys(code(entry["method"]) for entry in approximate))
    solvers = either((code(entry["solver"]) for entry in approximate), "and with")
    cases = dict.fromkeys(case for entry in approximate for case in entry["cases"])
    left_out = "; ".join(map(combination, view.get("fails_by_design", [])))
    exponent = f"$10^{{{np.log10(SMALLEST_ERROR):.0f}}}$"
    return [
        "## Solvers",
        "JAXScape solves the linear systems of resistance and free-energy "
        "distances with one of several solvers, in float32 or float64. Each "
        f"runs here on the {view['device'].upper()} in "
        f"{either(view['dtypes'], 'and')}: in the distance calculation on every "
        "raster of its distance, and in the connectivity and sensitivity "
        f"analyses under exact resistance at {exact}, and under {methods} with "
        f"{solvers} on {either(map(case_label, cases), 'and')}. `dense` is the "
        "dense solve, `solver=None`. The error of a cell is, in the distance "
        "calculation, the largest relative error against its float64 "
        "reference, and in the windowed analyses the relative L2 error of the "
        "map $m$, $\\lVert m - m_{\\mathrm{ref}} \\rVert_2 / \\lVert "
        "m_{\\mathrm{ref}} \\rVert_2$, against the map of exact resistance in "
        f"float64 with {code(view['reference_solver'])} on the same tiles, "
        f"measured at {exact} alone. A call stops after "
        f"{duration(view['timeout'])}, and under {methods} after "
        f"{duration(approximate[0].get('timeout', view['timeout']))}. These "
        f"combinations are not run: {left_out}.",
        "![Solvers: warm runtime against raster cells](solvers.png)",
        "Warm runtime of each solver against the raster cells: solid with "
        "filled markers in float32, dashed with hollow markers in float64, and "
        "a star on Maryland.",
        "![Solvers: warm runtime against error](solver-errors.png)",
        "Warm runtime of each cell with a measured error against its error: "
        "the colour gives the solver, a filled marker float32 and a hollow one "
        f"float64, and the shape the raster. Errors below {exponent} are drawn "
        f"at {exponent}.",
        collapsed(
            "Warm runtime (s) · error against the float64 reference",
            "Each cell gives the warm runtime, the error where it is measured, "
            "`missing` without a reference, and, in the connectivity analysis, "
            "the landscape total $C$.",
            outcome_table(records, SOLVER_VIEW),
        ),
    ]


def setup_section(records, config, environment):
    """Return the section of the setup: hardware, protocol, settings, versions."""
    run, landscape, maryland = config["run"], config["landscape"], config["maryland"]
    gpu, cpu, cores = environment["gpu"], environment["cpu"], environment["cores"]
    hardware = f"{cores} cores of {'an' if cpu[0] in 'AEIOU' else 'a'} {cpu}"
    if gpu:
        hardware = (
            f"one {gpu['name']} with {gpu['memory']}, driver {gpu['driver']} and "
            f"CUDA {gpu['cuda']}, and {hardware}"
        )
    habitat = (1 - landscape["habitat_threshold"]) * 100
    tasks = either((TASKS[q].lower() for q in maryland["questions"]), "and")
    part = ""
    if maryland.get("side"):
        part = f", of which the run takes the top-left {maryland['side']}² pixels"
    tools = {}
    for record in records:
        if record["view"] == "comparison":
            tool = tools.setdefault(record["tool"], (record["adapter"], set()))
            tool[1].add(record["question"])
    rows = [
        [
            tool,
            environment["tools"][tool],
            ", ".join(question for question in TASKS if question in asked),
            mapping(adapter),
        ]
        for tool, (adapter, asked) in tools.items()
    ]
    runtimes = ", ".join(f"{k} {v}" for k, v in environment["runtimes"].items())
    return [
        "## Setup",
        f"**Hardware.** The run takes {hardware}, under {environment['os']}. "
        "Each cell runs in its own process, pinned to these cores, with "
        f"{cores} threads for OpenMP, BLAS, Julia and R; XLA sizes its thread "
        "pools to the same cores.",
        "**Timing.** A warm-up call comes first and compiles the code of tools "
        "that compile. The warm runtime is the median of the timed calls that "
        f"follow it, {run['repeats']} per cell, or the time of a single call "
        "when the warm-up takes more than "
        f"{duration(run['single_call_after'])}. Every call redoes the work that "
        "depends on the landscape: graph assembly, factorization, multigrid "
        "hierarchy and solver state. A call stops after "
        f"{duration(run['timeout'])} (`timeout`). Within a setting, synthetic "
        "rasters run from the smallest up, and after a failure, a timeout or "
        "running out of memory (`oom`) on one, the larger ones are `skipped`; "
        "Maryland always runs.",
        "**Landscapes.** Each synthetic raster draws a Gaussian random field "
        "with a Gaussian covariance of correlation length "
        f"{landscape['correlation_length']} pixels, on a periodic grid with a "
        "fixed seed, and normalizes its ranks to [0, 1]. Habitat quality "
        "equals the field where it is at least "
        f"{landscape['habitat_threshold']}, and 0 elsewhere: {habitat:.0f} % "
        "of the pixels are habitat. Permeability is $10^{-2(1 - f)}$ for the "
        "field $f$, from 0.01 to 1, and resistance is its inverse. A larger "
        f"raster covers a larger area at the same grain. The {tasks} also run "
        "on the Maryland case of Omniscape's example: the 2016 land cover of "
        "Frederick, Maryland, from the National Land Cover Database, 694 × 725 "
        f"pixels of 30 m{part}. Each class takes the resistance of Omniscape's "
        "reclassification table; medium- and high-intensity development are "
        "barriers, and forest, the classes of resistance 1, is habitat of "
        "quality 1.",
        f"**Versions.** {runtimes}. The table gives the version of each tool, "
        "its tasks, and how its adapter maps the shared parameters onto it.",
        collapsed(
            "Tools: version, tasks and parameter mapping",
            table(["Tool", "Version", "Tasks", "Mapping"], rows),
        ),
    ]


def page(records, config, environment, maryland=None):
    """Return the Benchmark page of a run.

    **Arguments:**

    - `records`: The records of the run.
    - `config`: The configuration of the run, which gives the settings that the
      page states.
    - `environment`: The environment snapshot of the run.
    - `maryland`: The records of the Maryland figure, when the run directory
      has it.

    **Returns:**

    The page, in Markdown.
    """
    solvers = solver_view(records)
    comparison = [record for record in records if record["view"] == "comparison"]
    sections = [
        *introduction(),
        *distance_section(of_question("distance", comparison), config),
        *connectivity_section(of_question("connectivity", comparison), config),
        *sensitivity_section(of_question("sensitivity", comparison), maryland),
        *(solver_section(solvers, config) if solvers else []),
        *setup_section(records, config, environment),
    ]
    return "\n\n".join(section for section in sections if section) + "\n"


if __name__ == "__main__":
    main()
