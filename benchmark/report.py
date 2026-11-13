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
import re
import textwrap
from itertools import groupby, product
from pathlib import Path

import matplotlib
import numpy as np
from run import map_path, read_records, RHO
from scipy.stats import rankdata


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator


HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs" / "benchmark"  # the Benchmark page of the docs
EXAMPLE = "../examples/connectivity_analysis/connectivity_analysis.ipynb"

# The questions of the benchmark, and the title of the section of each.
QUESTIONS = {
    "distance": "Distance calculation",
    "connectivity": "Connectivity analysis",
    "sensitivity": "Sensitivity analysis",
}
# The names of distances and labels in tables and figures, where they differ.
NAMES = {
    "lcp": "least-cost path",
    "free_energy": "free energy",
    "resistance, K = -R, normalized": "normalized sensitivity (K = −R)",
}
# The solvers of JAXScape 0.0.16 that JAXScape 0.0.18 renamed, under their names
# in 0.0.18, which the page gives.
RENAMED = {"GridMultigridSolver": "GeometricMultigridCGSolver"}
# The questions as the configuration of the run of JAXScape 0.0.16 names them,
# under their names.
OLD_QUESTIONS = {"E1": "distance", "E2": "connectivity", "E3": "sensitivity"}
# The solvers that JAXScape removed, whose records the page leaves out.
REMOVED = {"PyAMGSolver"}
# The names of distances in prose.
PROSE = {
    "resistance": "resistance distance",
    "lcp": "least-cost path distance",
    "free_energy": "free-energy distance",
}
FAILURES = ("failed", "oom", "timeout")  # the statuses of a cell that stopped
# What JAXScape did on a raster where it stopped, by status, in prose.
STOPPED = {"oom": "ran out of memory", "timeout": "exceeded the time limit"}

LINE = ("tool", "distance", "device")  # a line of a figure of the comparison
SOLID = {"gpu", "float32"}  # the lines of the figures drawn solid, others dashed
# The marker of each raster, from the smallest; circles mark the dtype.
MARKERS = "D^vPX*"
SMALLEST_ERROR = 1e-16  # errors below it are drawn at it, on a log scale
# The Maryland figure: JAXScape's map of a question next to a tool's map.
MARYLAND = {
    "question": "sensitivity",
    "tool": "Omniscape.jl",
    "map": "normalized current",
}
BLUES = "#cde2fb #9ec5f4 #6da7ec #3987e5 #256abf #184f95 #0d366b".split()
# Figures are saved at DPI and shown at PIXELS_PER_INCH CSS pixels per inch of
# figure, so that their 9 pt text is 12.5 px high on the page, whatever their
# size. A figure of one panel is about 600 px wide, and the figures of the
# solver view, of four panels, about 720 px, the width of the docs page.
DPI = 200
PIXELS_PER_INCH = 100
ONE_PANEL = (6.0, 3.0)
PANEL_HEIGHT = 2.5
FOUR_PANELS_WIDTH = 7.2
plt.rcParams.update({"font.size": 9, "legend.fontsize": 8})


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
    records = [record for record in records if record["solver"] not in REMOVED]
    config = renamed(json.loads((run_dir / "config.json").read_text()))
    environment = json.loads((run_dir / "environment.json").read_text())

    comparison = [record for record in records if record["view"] == "comparison"]
    for question in QUESTIONS:
        if of_question(question, comparison):
            path = run_dir / f"{question}.png"
            plot_runtimes(of_question(question, comparison), ("question",), LINE, path)
    solvers = solver_view(records)
    if solvers:
        panel, line = ("question", "distance"), ("solver", "dtype")
        plot_runtimes(solvers, panel, line, run_dir / "solvers.png")
        plot_errors(solvers, run_dir / "solver-errors.png")
    pair = maryland_pair(comparison, config)
    if pair and all(map_path(run_dir, record).exists() for record in pair):
        plot_maryland(run_dir, *pair, run_dir / "maryland.png")
    maryland = pair if (run_dir / "maryland.png").exists() else None
    text = page(records, config, environment, maryland, run_dir)
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


def renamed(value):
    """Return a configuration, or a value of it, with its questions under their names.

    The run of JAXScape 0.0.16 names its questions as `OLD_QUESTIONS` gives
    them, in keys and in values.
    """
    if isinstance(value, dict):
        return {OLD_QUESTIONS.get(k, k): renamed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [renamed(item) for item in value]
    return OLD_QUESTIONS.get(value, value) if isinstance(value, str) else value


def of_question(question, records):
    """Return the records of a question."""
    return [record for record in records if record["question"] == question]


def solver_view(records):
    """Return the records of the solver view, in the order of `QUESTIONS`."""
    solvers = [record for record in records if record["view"] == "solver"]
    return sorted(solvers, key=lambda record: list(QUESTIONS).index(record["question"]))


def name(value):
    """Return the name in tables and figures of a distance, label or solver.

    A solver that JAXScape 0.0.18 renamed takes its name in 0.0.18.
    """
    value = NAMES.get(value, value or "")
    return re.sub(r"\w+", lambda word: RENAMED.get(word[0], word[0]), value)


def prose(distance):
    """Return the name in prose of a distance or label, such as `resistance distance`."""
    return PROSE.get(distance, name(distance))


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


def fastest(records, case):
    """Return the fastest of the records on a case."""
    on_case = [record for record in records if record["case"] == case]
    return min(on_case, key=lambda record: record["median_s"])


def succeeded(records):
    """Return the records that succeeded."""
    return [record for record in records if record["status"] == "ok"]


def panel_title(values):
    """Return the title of a panel, such as `Distance calculation, resistance`."""
    return ", ".join(
        QUESTIONS[value] if field == "question" else name(value)
        for field, value in values.items()
    )


def panels_figure(count):
    """Return a figure with `count` panels: one, or two per row."""
    if count == 1:
        figure, axes = plt.subplots(figsize=ONE_PANEL, layout="constrained")
        return figure, [axes]
    rows = -(-count // 2)
    figure, grid = plt.subplots(
        rows,
        2,
        figsize=(FOUR_PANELS_WIDTH, PANEL_HEIGHT * rows),
        squeeze=False,
        layout="constrained",
    )
    for axes in grid.flat[count:]:
        axes.set_visible(False)
    return figure, grid.flat


def log_axes(axes, xlabel):
    """Put a panel on log axes, without labels on minor ticks."""
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.xaxis.set_minor_formatter(NullFormatter())
    axes.yaxis.set_minor_formatter(NullFormatter())
    axes.set_xlabel(xlabel)
    axes.set_ylabel("Warm runtime (s)")


def raster_axis(axes, records):
    """Mark the synthetic rasters of `records` on the x axis, by their side."""
    sides = sorted({side(record["case"]) for record in records} - {None})
    axes.xaxis.set_minor_locator(NullLocator())
    axes.set_xticks([s * s for s in sides], [f"{s}²" for s in sides])
    cells = [record["cells"] for record in records]
    axes.set_xlim(min(cells) / 2, max(cells) * 2)


def runtime_axis(axes, labels, records):
    """Put a panel of one raster on a log runtime axis, with a row per label.

    The first label is the top row. Each label stands above its row, inside the
    panel, and breaks after each comma, and the runtime axis names the raster
    of `records`.
    """
    axes.set_xscale("log")
    runtimes = [record["median_s"] for record in records]
    axes.set_xlim(min(runtimes) / 2, max(runtimes) * 2)
    raster = case_label(records[0]["case"])
    axes.set_xlabel(f"Warm runtime on the {raster} raster (s)")
    axes.set_yticks([])
    axes.set_ylim(len(labels) - 0.5, -1)
    for row, label in enumerate(labels):
        axes.axhline(row, color="0.9", linewidth=0.8, zorder=0)
        axes.annotate(
            label.replace(", ", ",\n"),
            (0, row),
            xycoords=axes.get_yaxis_transform(),
            xytext=(4, 5),
            textcoords="offset points",
            va="bottom",
        )


def save(figure, colors, styles, path, count):
    """Save a figure with the legends of its colours and styles.

    The legends stand right of a figure of one panel, one above the other, and
    below the panels of a larger figure, side by side.
    """
    if count == 1:
        gap = Line2D([], [], linestyle="", label=" ")
        handles = colors + [gap] + styles
        figure.legend(handles=handles, loc="outside right upper", frameon=False)
    else:
        for handles, corner, x in ((colors, "upper left", 0), (styles, "upper right", 1)):
            figure.legend(
                handles=handles, loc=corner, bbox_to_anchor=(x, 0), frameon=False
            )
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
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
    """Plot the warm runtime of each line against the raster.

    At each raster, a line takes the fastest of its records that succeed, and
    joins the synthetic rasters; Maryland is a star. A panel whose records that
    succeed are all on one raster has a row per line instead, in the order of
    the legend, with a dot at the warm runtime of the line, and its runtime
    axis names the raster. A figure of one panel has no title: the section of
    the page that shows it names its question.

    **Arguments:**

    - `records`: The records to plot.
    - `panel`: The fields whose values give a panel, such as `("question",)`.
    - `line`: The fields whose values give a line. The last, the device or the
      dtype, sets its style: GPU and float32 lines are solid with filled
      markers, and CPU and float64 lines dashed with hollow markers. The others
      set its colour, and the legend gives their values, with the tier of a
      tool when the tools of the figure differ in tier.
    - `path`: Where to save the figure.
    """
    panels = list(group_by(records, panel))
    keys = list(dict.fromkeys(tuple(r[f] for f in line[:-1]) for r in records))
    tiers = {tuple(r[f] for f in line[:-1]): r["tier"] for r in records}
    figure, grid = panels_figure(len(panels))
    for axes, (values, group) in zip(grid, panels):
        ok = succeeded(group)
        one_raster = len({record["case"] for record in ok}) == 1
        on_panel = {tuple(r[f] for f in line[:-1]) for r in ok}
        rows = [key for key in keys if key in on_panel]
        for of_line, records_of_line in group_by(ok, line):
            key = tuple(of_line.values())[:-1]
            style = style_of(of_line[line[-1]], f"C{keys.index(key) % 10}")
            cases = dict.fromkeys(record["case"] for record in records_of_line)
            best = [fastest(records_of_line, case) for case in cases]
            if one_raster:
                dot = style | {"linestyle": ""}
                axes.plot(best[0]["median_s"], rows.index(key), "o", ms=6, **dot)
                continue
            synthetic = [record for record in best if side(record["case"])]
            axes.plot(
                [record["cells"] for record in synthetic],
                [record["median_s"] for record in synthetic],
                marker="o",
                ms=4,
                **style,
            )
            for record in best:
                if not side(record["case"]):
                    star = style | {"linestyle": ""}
                    axes.plot(record["cells"], record["median_s"], "*", ms=9, **star)
        if one_raster:
            runtime_axis(axes, [", ".join(map(name, key)) for key in rows], ok)
        else:
            log_axes(axes, "Raster (pixels)")
            raster_axis(axes, group)
        if len(panels) > 1:
            axes.set_title(panel_title(values))
    drawn = {tuple(r[f] for f in line[:-1]) for r in succeeded(records)}
    tiered = len({tiers[key] for key in drawn} - {None}) > 1
    handles = [
        Line2D(
            [],
            [],
            color=f"C{index % 10}",
            label=", ".join(map(name, key))
            + (f", Tier {tiers[key]}" if tiered and tiers[key] else ""),
        )
        for index, key in enumerate(keys)
        if key in drawn
    ]
    styles = sorted({r[line[-1]] for r in records}, key=lambda v: v not in SOLID)
    styled = [
        Line2D(
            [],
            [],
            marker="o",
            ms=4,
            label=style.upper() if line[-1] == "device" else style,
            **style_of(style, "gray"),
        )
        for style in styles
    ]
    if not all(side(record["case"]) for record in records):
        star = dict(linestyle="", marker="*", ms=9, color="gray", label="Maryland")
        styled.append(Line2D([], [], **star))
    save(figure, handles, styled, path, len(panels))


def plot_errors(records, path):
    """Plot the warm runtime of each record of the solver view against its error.

    Only records with a measured error appear: the colour gives the solver, a
    filled marker float32 and a hollow one float64, and the shape the raster.
    Errors below `SMALLEST_ERROR` are drawn at it.
    """
    solvers = list(dict.fromkeys(record["solver"] for record in records))
    measured = [
        record for record in succeeded(records) if record["agreement"] is not None
    ]
    sides = sorted({side(r["case"]) for r in measured}, key=lambda s: s or np.inf)
    shapes = {s: MARKERS[index % len(MARKERS)] for index, s in enumerate(sides)}
    panels = list(group_by(measured, ("question", "distance")))
    figure, grid = panels_figure(len(panels))
    for axes, (values, group) in zip(grid, panels):
        for record in group:
            color = f"C{solvers.index(record['solver']) % 10}"
            axes.plot(
                max(record["agreement"], SMALLEST_ERROR),
                record["median_s"],
                shapes[side(record["case"])],
                ms=6,
                color=color,
                markerfacecolor=color if record["dtype"] == "float32" else "none",
            )
        measure = group[0]["measure"]
        log_axes(axes, measure[0].upper() + measure[1:])
        if len(panels) > 1:
            axes.set_title(panel_title(values))
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
    styled += [
        Line2D([], [], **gray, marker=shapes[s], label=f"{s}²" if s else "Maryland")
        for s in sides
    ]
    save(figure, handles, styled, path, len(panels))


def compared_with(record, config):
    """Return the distance, or label, of JAXScape that a record is compared with.

    It is the `compared_with` of the record or, when the record leaves it out,
    of the tool's entry in the configuration, and by default the distance of
    the record.
    """
    if record["compared_with"]:
        return record["compared_with"]
    for entry in config["tools"]:
        if (entry["tool"], entry["question"]) == (record["tool"], record["question"]):
            return entry.get("compared_with") or record["distance"]
    return record["distance"]


def maryland_pair(records, config):
    """Return the records of the Maryland figure, or `None` when one is missing.

    They are the JAXScape GPU record under the tool's `compared_with`, whose map
    the tool's maps are compared with, and the fastest setting of the tool that
    succeeds.
    """
    on_maryland = [
        record
        for record in succeeded(of_question(MARYLAND["question"], records))
        if record["case"] == "maryland"
    ]
    others = [record for record in on_maryland if record["tool"] == MARYLAND["tool"]]
    if not others:
        return None
    other = fastest(others, "maryland")
    paired = ("JAXScape", "gpu", compared_with(other, config))
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
    figure, axes = plt.subplots(1, 2, figsize=(6.0, 3.0), layout="constrained")
    labels = (name(jaxscape["distance"]), MARYLAND["map"])
    for axis, record, label in zip(axes, (jaxscape, other), labels):
        axis.set_axis_off()
        axis.set_title(f"{record['tool']}\n{label}")
        quantile = quantiles(np.load(map_path(run_dir, record)))
        image = axis.imshow(quantile, cmap=colormap, vmin=0, vmax=1)
    figure.colorbar(image, ax=axes, shrink=0.8, label="Rank of the pixel (quantile)")
    figure.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def seconds(value):
    """Return a duration in seconds to three significant digits, such as `1700`."""
    return f"{float(f'{value:.3g}'):g}"


def runtime(record):
    """Return the warm runtime of a record that succeeded, such as `5.48 s`."""
    return f"{seconds(record['median_s'])} s"


def agreement(record):
    """Return the agreement of a record, such as `ρ = 0.93` or `error 3.2e-03`.

    The text is empty when the agreement is missing: without a reference, or
    where Spearman's ρ is undefined.
    """
    if record["agreement"] is None:
        return ""
    if record["measure"] == RHO:
        return f"ρ = {record['agreement']:.2f}"
    return f"error {record['agreement']:.1e}"


def factor(ratio):
    """Return a ratio of runtimes, with two significant digits below 10."""
    return f"{ratio:.2g}" if ratio < 10 else f"{ratio:.0f}"


def entry(record):
    """Return the entry of a record in a table.

    It is the warm runtime and the agreement of a record that succeeded, and
    the status of one that did not.
    """
    if record["status"] != "ok":
        return record["status"]
    return ", ".join(filter(None, (runtime(record), agreement(record))))


def setting(record):
    """Return the setting of a record, such as `GPU, float32, CuDSSSolver`."""
    values = (record["device"].upper(), record["dtype"], name(record["solver"]))
    return ", ".join(filter(None, values))


# The columns that name a setting in the tables of the comparison and of the
# solver view: the header of each, and its text for a record.
COMPARISON = {
    "Tool": lambda record: record["tool"],
    "Distance": lambda record: name(record["distance"]),
    "Setting": setting,
}
SOLVER_VIEW = {
    "Question": lambda record: QUESTIONS[record["question"]].lower(),
    "Distance": lambda record: name(record["distance"]),
    "dtype": lambda record: record["dtype"],
    "Solver": lambda record: name(record["solver"]),
}


def comparison_columns(records, config):
    """Return the columns that name a setting in the tables of a question.

    They are those of `COMPARISON`, whose column of distances is headed
    "Distance or label" when a record holds a label in it, such as
    `resistance, K = -R, normalized`.
    """
    if all(record["distance"] in config["distances"] for record in records):
        return COMPARISON
    return {
        "Distance or label" if header == "Distance" else header: text
        for header, text in COMPARISON.items()
    }


def row(cells):
    """Return a row of a Markdown table."""
    return "| " + " | ".join(str(cell).replace("|", "\\|") for cell in cells) + " |"


def table(header, rows):
    """Return a Markdown table."""
    lines = [row(header), row(["---"] * len(header))]
    return "\n".join(lines + [row(cells) for cells in rows])


def bullets(items):
    """Return a Markdown list of the items that are not empty."""
    return "\n".join(f"- {item}" for item in items if item)


def collapsed(title, *blocks):
    """Return a note collapsed by default, with its title and Markdown blocks."""
    body = "\n\n".join(block for block in blocks if block)
    return f'??? note "{title}"\n\n' + textwrap.indent(body, "    ")


def picture(run_dir, path, alt):
    """Return the HTML of a figure of the run directory, at its width on the page.

    The width is `PIXELS_PER_INCH` pixels per inch of the figure, read from
    the PNG header, so that the text of every figure has the same size.
    """
    pixels = int.from_bytes((run_dir / path).read_bytes()[16:20], "big")
    width = round(pixels / DPI * PIXELS_PER_INCH)
    return f'<img src="{path}" alt="{alt}" width="{width}">'


def outcome_table(records, columns):
    """Return the table of the outcome of each setting on each raster.

    **Arguments:**

    - `records`: The records of the table.
    - `columns`: The columns that name a setting, as `COMPARISON`.

    **Returns:**

    The table, in Markdown, with one row per setting and one column per
    raster, whose entries are given by `entry`.
    """
    cases = sorted({(record["cells"], record["case"]) for record in records})
    rows = {}
    for record in sorted(records, key=lambda record: record["cells"]):
        named = tuple(text(record) for text in columns.values())
        rows.setdefault(named, {})[record["case"]] = entry(record)
    order = list(dict.fromkeys(tuple(t(r) for t in columns.values()) for r in records))
    body = [
        [*named, *(rows[named].get(case, "") for _, case in cases)] for named in order
    ]
    return table([*columns, *(case_label(case) for _, case in cases)], body)


def reason(record):
    """Return the recorded reason of a record that did not succeed, in a table.

    For an out-of-memory error, it is the size of the allocation that failed.
    """
    text = record["reason"] or ""
    size = re.search(r"allocate ([\d.]+) ?([KMGT]i?B)", text)
    if size:
        return f"allocation of {size[1]} {size[2]} failed"
    return f"`{text.replace('`', "'")}`" if text else ""


def failures(records, columns, fold=10):
    """Return the section of the settings that did not complete a raster.

    **Arguments:**

    - `records`: The records of a question or of the solver view.
    - `columns`: The columns that name a setting, as `COMPARISON`.
    - `fold`: The number of rows above which the table is collapsed.

    **Returns:**

    The heading and the table of the records that stopped, with their status
    and recorded reason, or nothing when every record succeeded or was
    skipped.
    """
    order = list(dict.fromkeys(tuple(t(r) for t in columns.values()) for r in records))
    stopped = sorted(
        (record for record in records if record["status"] in FAILURES),
        key=lambda r: (order.index(tuple(t(r) for t in columns.values())), r["cells"]),
    )
    if not stopped:
        return []
    header = [*columns, "Raster", "Status", "Recorded reason"]
    rows = [
        [
            *(text(record) for text in columns.values()),
            case_label(record["case"]),
            f"`{record['status']}`",
            reason(record),
        ]
        for record in stopped
    ]
    listed = table(header, rows)
    if len(rows) > fold:
        listed = collapsed(f"{len(rows)} settings", listed)
    return ["### Settings that did not complete", listed]


def speedups(records, config):
    """Return the records that give the speed-up of JAXScape over each tool.

    For each other tool and distance of a question, on the largest synthetic
    raster on which the tool and JAXScape on the GPU both succeed, under the
    distance or label of JAXScape that the tool is compared with: the fastest
    record of each there.

    **Returns:**

    A list of pairs of the record of the tool and that of JAXScape.
    """
    ok = [record for record in succeeded(records) if side(record["case"])]
    others = [record for record in ok if record["tool"] != "JAXScape"]
    pairs = []
    for _, group in group_by(others, ("tool", "distance")):
        target = ("JAXScape", "gpu", compared_with(group[0], config))
        jaxscape = [r for r in ok if (r["tool"], r["device"], r["distance"]) == target]
        shared = {r["case"] for r in group} & {r["case"] for r in jaxscape}
        if shared:
            case = max(shared, key=side)
            pairs.append((fastest(group, case), fastest(jaxscape, case)))
    return pairs


def ratio(pair):
    """Return the speed-up of JAXScape in a pair of records of `speedups`."""
    tool, jaxscape = pair
    return tool["median_s"] / jaxscape["median_s"]


def speedup_finding(pairs, by_distance):
    """Return the sentence that states the speed-ups of JAXScape over the tools.

    **Arguments:**

    - `pairs`: The pairs of records of `speedups`.
    - `by_distance`: Whether to name the distance of each tool, as in the
      distance calculation, or its tier.
    """
    if not pairs:
        return ""

    def named(pair):
        tool = pair[0]
        return f"{tool['tool']}, {prose(tool['distance'])}" if by_distance else tool["tool"]

    def detail(pair):
        tool = pair[0]
        return prose(tool["distance"]) if by_distance else f"Tier {tool['tier']}"

    clauses = []
    for word, chosen, scale in (
        ("faster", [p for p in pairs if ratio(p) >= 1], lambda p: ratio(p)),
        ("slower", [p for p in pairs if ratio(p) < 1], lambda p: 1 / ratio(p)),
    ):
        if not chosen:
            continue
        low, high = min(chosen, key=scale), max(chosen, key=scale)
        tools = list(dict.fromkeys(pair[0]["tool"] for pair in chosen))
        if len(chosen) == 1:
            clause = f"{word} than {tools[0]} ({detail(low)}), by a factor of"
            clauses.append(f"{clause} {factor(scale(low))}")
            continue
        who = "every other tool" if len(chosen) == len(pairs) else either(tools, "and")
        clauses.append(
            f"{word} than {who}, by a factor of {factor(scale(low))} "
            f"({named(low)}) to {factor(scale(high))} ({named(high)})"
        )
    return (
        "On the largest synthetic raster that JAXScape on the GPU and the other "
        f"tool both completed, JAXScape is {', and '.join(clauses)}."
    )


def error_finding(records):
    """Return the sentence that states the largest errors of the distance calculation."""
    measured = [r for r in succeeded(records) if r["agreement"] is not None]
    jaxscape = [r for r in measured if r["tool"] == "JAXScape"]
    others = [r for r in measured if r["tool"] != "JAXScape"]
    if not jaxscape:
        return ""
    worst = max(jaxscape, key=lambda record: record["agreement"])
    text = (
        "The largest relative error with respect to the float64 reference is at "
        f"most {worst['agreement']:.1e} for JAXScape, under "
        f"{prose(worst['distance'])} in {worst['dtype']}"
    )
    if others:
        largest = max(record["agreement"] for record in others)
        text += f", and at most {largest:.1e} for the other tools"
    return text + "."


def rho_finding(pairs):
    """Return the sentence that states Spearman's ρ of the tools of `speedups`."""
    measured = [pair for pair in pairs if pair[0]["agreement"] is not None]
    if not measured:
        return ""
    low, high = (f(measured, key=lambda pair: pair[0]["agreement"]) for f in (min, max))
    values = [f"{pair[0]['agreement']:.2f} ({pair[0]['tool']})" for pair in (low, high)]
    span = values[0] if low is high else f"ranges from {values[0]} to {values[1]}"
    verb = "is " if low is high else ""
    return (
        f"On those rasters, {RHO} between the map of a tool and that of JAXScape "
        f"{verb}{span}."
    )


def limits_finding(records):
    """Return the sentence that states where JAXScape on the GPU stopped."""
    stopped = [
        record
        for record in records
        if (record["tool"], record["device"]) == ("JAXScape", "gpu")
        and record["status"] in FAILURES
    ]
    by_status = {}
    for record in stopped:
        where = f"under {prose(record['distance'])} on {case_label(record['case'])}"
        by_status.setdefault(STOPPED.get(record["status"], "failed"), []).append(where)
    if not by_status:
        return ""
    clauses = [f"{status} {either(where, 'and')}" for status, where in by_status.items()]
    return f"On the GPU, JAXScape {either(clauses, 'and')}."


def speedup_table(pairs, by_distance):
    """Return the table of the speed-up of JAXScape over each tool, in Markdown.

    **Arguments:**

    - `pairs`: The pairs of records of `speedups`.
    - `by_distance`: Whether to give the distance and the errors of each pair,
      as in the distance calculation, or the tier of the tool, the distance or
      label of JAXScape that it is compared with, and Spearman's ρ.
    """
    if not pairs:
        return ""
    runtimes = ["Raster", "Warm runtime of the tool", "Warm runtime of JAXScape"]
    if by_distance:
        header = ["Tool", "Distance", *runtimes, "Speed-up"]
        header += ["Error of the tool", "Error of JAXScape"]
    else:
        header = ["Tool", "Tier", "Compared with JAXScape under", *runtimes]
        header += ["Speed-up", RHO]
    rows = []
    for tool, jaxscape in pairs:
        times = [case_label(tool["case"]), runtime(tool), runtime(jaxscape)]
        speedup = factor(ratio((tool, jaxscape)))
        if by_distance:
            errors = [agreement(record).removeprefix("error ") for record in (tool, jaxscape)]
            rows.append([tool["tool"], name(tool["distance"]), *times, speedup, *errors])
        else:
            rho = agreement(tool).removeprefix("ρ = ")
            paired = name(jaxscape["distance"])
            rows.append([tool["tool"], tool["tier"], paired, *times, speedup, rho])
    return table(header, rows)


def comparison_results(records, config, run_dir, by_distance):
    """Return the finding and the results of a question of the comparison."""
    question = records[0]["question"]
    pairs = speedups(records, config)
    findings = [speedup_finding(pairs, by_distance)]
    findings += [error_finding(records) if by_distance else rho_finding(pairs)]
    findings.append(limits_finding(records))
    measure = "error" if by_distance else RHO
    missing = (
        "An entry without an error has no reference."
        if by_distance
        else "The maps of JAXScape are the references, and an entry of another "
        "tool without ρ has no map of JAXScape to compare with, or ρ is undefined."
    )
    return [
        " ".join(filter(None, findings)),
        "### Results",
        picture(run_dir, f"{question}.png", f"{QUESTIONS[question]}: warm runtime"),
        speedup_table(pairs, by_distance),
        collapsed(
            f"Warm runtime and {measure} of every setting on every raster",
            missing,
            outcome_table(records, comparison_columns(records, config)),
        ),
    ]


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
    `free-energy distance in float32`.
    """
    words = []
    for field, joiner, show in (
        ("distance", "", prose),
        ("solver", "with ", code),
        ("dtype", "in ", str),
        ("question", "in the ", lambda question: QUESTIONS[question].lower()),
    ):
        if field in entry:
            values = entry[field] if isinstance(entry[field], list) else [entry[field]]
            words.append((joiner if words else "") + either(map(show, values)))
    return " ".join(words)


def not_run(entry, records, view):
    """Return the parts of an entry of `fails_by_design` that no record ran.

    An entry without a question matches each question of the solver view
    `view` that computes one of its distances, or every question without a
    distance; it names them when they are several. The entry is split into its
    combinations, which take one value of each of its fields, and a record ran
    a combination when it holds each of its values. The combinations that no
    record ran, but those of a solver of `REMOVED`, are joined back into
    entries, one per value of the fields but the last that holds a list.
    """
    distances = entry.get("distance", [])
    distances = distances if isinstance(distances, list) else [distances]
    questions = [
        question
        for question, computed in view["questions"].items()
        if not distances or set(distances) & set(computed)
    ]
    if "question" not in entry and len(questions) > 1:
        entry = {**entry, "question": questions}
    fields = list(entry)
    lists = [field for field in fields if isinstance(entry[field], list)]
    joined = lists[-1:]  # the last field that holds a list, if any
    values = [entry[field] if field in lists else [entry[field]] for field in fields]
    parts = {}
    for chosen in product(*values):
        part = dict(zip(fields, chosen))
        if part.get("solver") in REMOVED or any(
            all(record[f] == part[f] for f in fields) for record in records
        ):
            continue
        key = tuple(value for field, value in part.items() if field not in joined)
        parts.setdefault(key, part | {field: [] for field in joined})
        for field in joined:
            parts[key][field].append(part[field])
    return list(parts.values())


def introduction():
    """Return the introduction of the page."""
    return [
        "# Benchmark",
        "This page compares JAXScape with the software that ecologists use for "
        "connectivity analysis on three questions: the distance calculation, the "
        "connectivity analysis and the sensitivity analysis. For each tool, it "
        "reports the warm runtime and the agreement of the results with a "
        "reference. The last section compares the linear solvers of JAXScape. "
        "[`records.csv`](records.csv) holds the record of every cell of the run: "
        "one setting of a tool on one raster.",
    ]


def setup_section(records, config, environment):
    """Return the section of the setup: hardware, software, protocol, measures."""
    run, landscape, maryland = config["run"], config["landscape"], config["maryland"]
    gpu, cpu, cores = environment["gpu"], environment["cpu"], environment["cores"]
    comparison = [record for record in records if record["view"] == "comparison"]
    solvers = solver_view(records)
    hardware = []
    if gpu:
        hardware.append(
            f"**GPU**: one {gpu['name']} with {gpu['memory']}, driver "
            f"{gpu['driver']}, CUDA {gpu['cuda']}."
        )
    article = "an" if cpu[0] in "AEIOU" else "a"
    hardware.append(
        f"**CPU**: {cores} cores of {article} {cpu}, under {environment['os']}. "
        f"Each cell runs in its own process, pinned to these cores, with {cores} "
        "threads for OpenMP, BLAS, Julia and R; XLA sizes its thread pools to "
        "the same cores."
    )
    languages = ", ".join(f"{k} {v}" for k, v in environment["runtimes"].items())
    hardware.append(f"**Languages and libraries**: {languages}.")

    tools = {}
    for record in comparison:
        tool = tools.setdefault(record["tool"], (record["adapter"], set()))
        tool[1].add(record["question"])
    versions = [
        [
            tool,
            environment["tools"][tool],
            ", ".join(QUESTIONS[q].lower() for q in QUESTIONS if q in asked),
        ]
        for tool, (_, asked) in tools.items()
    ]
    mappings = [[tool, mapping(adapter)] for tool, (adapter, _) in tools.items()]

    sides = sorted({side(r["case"]) for r in comparison} - {None})
    habitat = (1 - landscape["habitat_threshold"]) * 100
    part = ""
    if maryland.get("side"):
        part = f", of which the run takes the top-left {maryland['side']}² pixels"
    analyses = either((QUESTIONS[q].lower() for q in maryland["questions"]), "and")
    landscapes = [
        f"**Synthetic rasters** of {either((f'{s}²' for s in sides), 'and')} "
        "pixels. The benchmark draws a Gaussian random field $f$ with a Gaussian "
        f"covariance of correlation length {landscape['correlation_length']} "
        "pixels, on a periodic grid with a fixed seed, and normalizes its ranks "
        "to [0, 1]. The habitat quality is $f$ where $f \\geq "
        f"{landscape['habitat_threshold']}$, and 0 elsewhere: {habitat:.0f} % of "
        "the pixels are habitat. The permeability is $10^{-2(1 - f)}$, from 0.01 "
        "to 1, and the resistance is its inverse. A larger raster covers a "
        "larger area at the same grain.",
        f"**Maryland**, for the {analyses}: the land cover of Omniscape's "
        "example, the 2016 National Land Cover Database of Frederick, Maryland, "
        f"694 × 725 pixels of 30 m{part}. The resistance of each land-cover "
        "class is that of Omniscape's reclassification table; medium- and "
        "high-intensity development are barriers, and forest, the classes of "
        "resistance 1, is habitat of quality 1.",
    ]
    timing = [
        f"**Warm runtime**: the median time of the {run['repeats']} timed "
        f"call{'s' if run['repeats'] != 1 else ''} of a cell, which follow a "
        "warm-up call that compiles the code of the tools that compile; the time "
        "of a single timed call when the warm-up call takes more than "
        f"{duration(run['single_call_after'])}.",
        "**Work of a call**: every call redoes the work that depends on the "
        "landscape: graph assembly, factorization, multigrid hierarchy and "
        "solver state.",
        "**Speed-up of JAXScape** over a tool under a distance: the warm "
        "runtime of the tool divided by that of JAXScape on the GPU, each in its "
        "fastest setting on the largest synthetic raster that both completed, "
        "with JAXScape under the same distance in the distance calculation, and "
        "under the distance or label of the column \"Compared with JAXScape "
        "under\" in the connectivity and sensitivity analyses.",
        f"**Time limit**: a call stops after {duration(run['timeout'])}"
        + (
            ", and in the section of the linear solvers of JAXScape after the "
            "time limit stated in its Method"
            if solvers
            else ""
        )
        + ".",
        "**Status of a cell that did not complete**: `timeout` when a call "
        "exceeds its time limit, `oom` when a call runs out of memory, and "
        "`failed` on another error. A setting runs on the synthetic rasters from "
        "the smallest up, and is `skipped` on the rasters larger than one where "
        "it stopped; Maryland always runs.",
    ]
    rho = f"{RHO} between the map of the tool and that of JAXScape"
    tiers = [
        ["1", "the quantity of JAXScape, on the same graph"]
        + ["largest relative error with respect to a float64 reference"],
        ["2", "the quantity of JAXScape, on another graph or with another proximity"]
        + [rho],
        ["3", "a related quantity that answers the same question", rho],
    ]
    reading = [
        "**Figures of runtime**: each line passes through the warm runtimes of "
        "the fastest setting of a tool, distance and device that succeeded on "
        "each synthetic raster, solid on the GPU and dashed on the CPU"
        + (
            "; in the section of the linear solvers, a line is a solver and "
            "dtype, solid in float32 and dashed in float64"
            if solvers
            else ""
        )
        + ". A star marks Maryland.",
        "**Tables**: an entry is the warm runtime and the agreement of a cell "
        "that succeeded, or the status of a cell that did not.",
    ]
    return [
        "## Setup",
        "### Hardware and software",
        bullets(hardware),
        "### Tools",
        table(["Tool", "Version", "Questions"], versions),
        renaming(records),
        collapsed("Parameter mapping of each tool", table(["Tool", "Mapping"], mappings)),
        "### Landscapes",
        bullets(landscapes),
        "### Timing",
        bullets(timing),
        "### Comparability tiers",
        table(["Tier", "The tool computes", "Agreement measure"], tiers),
        "Only at Tier 1 do the tool and JAXScape perform the same computation.",
        "### Figures and tables",
        bullets(reading),
    ]


def renaming(records):
    """Return the sentence that states the solvers of the run that `RENAMED` renames.

    It is empty when the records name none of them.
    """
    words = set(re.findall(r"\w+", " ".join(r["solver"] or "" for r in records)))
    old = [solver for solver in RENAMED if solver in words]
    if not old:
        return ""
    pairs = (f"`{solver}` as {code(solver)}" for solver in old)
    return (
        "The page names the solvers of this run as JAXScape 0.0.18 does: "
        f"{either(pairs, 'and')}."
    )


def mapping(adapter):
    """Return the header comment of an adapter, which states its mapping."""
    lines = []
    for line in (HERE / adapter).read_text().splitlines():
        if line.startswith("#!"):
            continue
        if not line.startswith("#"):
            break
        lines.append(line.lstrip("#").strip())
    return " ".join(line for line in lines if line)


def distance_section(records, config, run_dir):
    """Return the section of the distance calculation."""
    sites = config["distance"]["sites"]
    theta = config["distances"]["free_energy"]["theta"]
    tiers = {record["tier"] for record in records} - {None}
    sides = {d: values["sides"] for d, values in config["distances"].items()}
    every = set().union(*sides.values())
    fewer = [
        f"{prose(d)} on {either((f'{s}²' for s in sorted(of_d)), 'and')} alone"
        for d, of_d in sides.items()
        if set(of_d) != every and any(r["distance"] == d for r in records)
    ]
    method = [
        "**Quantity**: the distance $d_{ij}$ on a graph between habitat sites "
        "$i$ and $j$: resistance distance, least-cost path distance, and "
        f"free-energy distance with $\\theta = {theta}$. Ecologists use it in "
        "place of Euclidean distance in dispersal kernels and in models of "
        "isolation by distance.",
        f"**Sites**: {sites} sites drawn at random from the habitat pixels of the "
        f"raster, {sites * (sites - 1) // 2} pairs.",
        "**Graph**: every tool builds the same graph: rook contiguity, with the "
        "mean resistance of two pixels as the resistance of their edge.",
        f"**Rasters**: {'; '.join(fewer)}." if fewer else "",
        "**Error**: the largest relative error over the pairs of sites with "
        "respect to a float64 reference computed on the CPU: the linear systems "
        "of resistance and free-energy distances are solved with JAXScape's "
        "`CholmodSolver`, and least-cost path distances are computed with "
        "Dijkstra's algorithm in SciPy.",
        f"**Tier**: every tool is at Tier {tiers.pop()}." if len(tiers) == 1 else "",
    ]
    return [
        "## Distance calculation",
        *comparison_results(records, config, run_dir, by_distance=True),
        *failures(records, comparison_columns(records, config)),
        "### Method",
        bullets(method),
    ]


def connectivity_section(records, config, run_dir):
    """Return the section of the connectivity analysis."""
    windows = config["windows"]
    scale = f"{windows['dependency_range'] / 4:g}"
    block = windows["block_size"]

    def setting(name):
        return jaxscape_setting(config, "connectivity", "resistance", name)

    method = [
        "**Quantity**: Hanski's connectivity $S_i = \\sum_j K(d_{ij})\\, q_j$ of "
        "each pixel $i$, with the habitat quality $q_j$ of pixel $j$ and the "
        "proximity $K$, and its landscape total $C = \\sum_i q_i S_i$, the "
        "square of the equivalent connected area.",
        "**JAXScape**: in each call, `SensitivityAnalysis` computes the map of "
        "$\\partial C / \\partial q_i$, equal to $2 S_i$ for a symmetric "
        "distance.",
        "**Windows**: windowed analyses with `dependency_range` "
        f"{windows['dependency_range']} pixels and `coarsening_factor` "
        f"{windows['coarsening_factor']}, that is, blocks of {block} × "
        f"{block} pixels, under the proximity $K(d) = e^{{-d/{scale}}}$, with $d$ "
        "in equivalent pixels.",
        "**Graph**: rook contiguity, with the mean permeability of two pixels as "
        "the conductance of their edge. JAXScape's windowed analyses raise "
        "permeability below the permeability floor, a small fraction of the "
        "largest permeability, to the floor"
        + (
            ". On Maryland, the other tools treat the barriers, of permeability "
            "0, as impassable, and JAXScape's analyses still cross the barrier "
            "pixels: the edge between a barrier pixel and a pixel of "
            "permeability $p$ has a conductance of about $p/2$."
            if "maryland" in config
            else "."
        ),
        f"**Resistance distance**: {code(setting('method'))} with "
        f"{code(setting('solver'))}, the setting of the [connectivity "
        f"example]({EXAMPLE}), on tiles of up to {setting('batch_size'):,} block "
        "centres.",
        f"**Agreement**: {RHO} between the map of a tool and the map of JAXScape "
        "on the GPU under the distance or label of the column \"Compared with "
        "JAXScape under\", over the pixels where both maps are defined.",
    ]
    return [
        "## Connectivity analysis",
        *comparison_results(records, config, run_dir, by_distance=False),
        *failures(records, comparison_columns(records, config)),
        "### Method",
        bullets(method),
    ]


def sensitivity_section(records, config, run_dir, maryland):
    """Return the section of the sensitivity analysis."""
    normalized = NAMES["resistance, K = -R, normalized"]
    results = comparison_results(records, config, run_dir, by_distance=False)
    if maryland:
        jaxscape, other = maryland
        results += [
            picture(
                run_dir,
                "maryland.png",
                f"Maryland: JAXScape's {name(jaxscape['distance'])} and "
                f"{other['tool']}'s {MARYLAND['map']}",
            ),
            f"Rank of each pixel of Maryland in JAXScape's "
            f"{name(jaxscape['distance'])} and in {other['tool']}'s "
            f"{MARYLAND['map']} (`{other['solver']}`), with the pixels without a "
            "value in gray."
            + (
                f" {RHO} between the two maps is {other['agreement']:.2f}."
                if other["agreement"] is not None
                else ""
            ),
        ]
    method = [
        "**Quantity**: the sensitivity $\\partial C / \\partial p_k$ of $C$ with "
        "respect to the permeability $p_k$ of each pixel $k$. The pixels of "
        "largest sensitivity are corridors and pinch points: the pixels where a "
        "decrease in permeability gives the largest decrease in $C$.",
        "**JAXScape**: `SensitivityAnalysis` computes the sensitivity, with the "
        "windows and settings of the connectivity analysis. The other tools "
        "compute a flow: a current, a betweenness or a number of visits.",
        f"**{normalized[0].upper() + normalized[1:]}**: the map of the section "
        "[Comparison with Omniscape's normalized current]"
        f"({EXAMPLE}#comparison-with-omniscapes-normalized-current) of the "
        "connectivity example. Under resistance distance with the proximity "
        "$K(R) = -R$, under which every pair of habitat pixels has the same "
        "weight, it is the ratio of $-\\partial C / \\partial r_k = p_k^2\\, "
        "\\partial C / \\partial p_k$, the sensitivity of $C$ with respect to "
        "the resistance $r_k$ of pixel $k$, to its value on a landscape of "
        "permeability 1 with the same quality, and 0 where the latter is not "
        "positive. Each call computes both sensitivities.",
        "**Omniscape.jl**: computes its normalized current with the "
        "configuration of its Maryland example, with `radius` equal to "
        "`dependency_range`, the same block size, and a graph of eight "
        "neighbours.",
        "**Agreement**: as in the connectivity analysis.",
    ]
    return [
        "## Sensitivity analysis",
        *results,
        *failures(records, comparison_columns(records, config)),
        "### Method",
        bullets(method),
    ]


def fastest_solvers(records, approximate):
    """Return the tables of the fastest solver and of the approximations.

    **Arguments:**

    - `records`: The records of the solver view.
    - `approximate`: The labels of the solvers under an approximation of the
      distance, such as Spielman's, and their entries in the configuration.

    **Returns:**

    The table of the fastest solver of each question, distance and dtype among
    those that solve without an approximation, on the largest synthetic raster
    on which one succeeded, compared with the next fastest; and the table of
    each approximation on the raster of the row of its question, distance and
    dtype, or else its largest synthetic raster, with its speed-up over the
    fastest solver there. Either is empty without a record.
    """
    exact, rows = {}, []
    for values, group in group_by(records, ("question", "distance", "dtype")):
        ok = [
            record
            for record in succeeded(group)
            if side(record["case"]) and record["solver"] not in approximate
        ]
        if not ok:
            continue
        case = max(ok, key=lambda record: record["cells"])["case"]
        ranked = sorted(
            (record for record in ok if record["case"] == case),
            key=lambda record: record["median_s"],
        )
        best, after = ranked[0], ranked[1:2]
        exact[tuple(values.values())] = best
        rows.append(
            [
                QUESTIONS[values["question"]].lower(),
                name(values["distance"]),
                values["dtype"],
                case_label(case),
                name(best["solver"]),
                runtime(best),
                agreement(best).removeprefix("error "),
                *(
                    [name(after[0]["solver"]), factor(ratio((after[0], best)))]
                    if after
                    else ["", ""]
                ),
            ]
        )
    header = ["Question", "Distance", "dtype", "Raster", "Fastest solver"]
    header += ["Warm runtime", "Error", "Next fastest solver", "Speed-up over it"]
    fastest = table(header, rows) if rows else ""
    rows = []
    fields = ("question", "distance", "dtype", "solver")
    for values, group in group_by(records, fields):
        ok = [r for r in succeeded(group) if side(r["case"])]
        if values["solver"] not in approximate or not ok:
            continue
        best = exact.get(tuple(values.values())[:-1])
        cases = {record["case"]: record for record in ok}
        record = cases.get(best and best["case"]) or max(ok, key=lambda r: r["cells"])
        compared = best if best and best["case"] == record["case"] else None
        entry = approximate[values["solver"]]
        rows.append(
            [
                QUESTIONS[values["question"]].lower(),
                name(values["distance"]),
                values["dtype"],
                case_label(record["case"]),
                name(entry["method"]),
                name(entry["solver"]),
                runtime(record),
                agreement(record).removeprefix("error "),
                factor(ratio((compared, record))) if compared else "",
            ]
        )
    header = ["Question", "Distance", "dtype", "Raster", "Approximation", "Solver"]
    header += ["Warm runtime", "Error", "Speed-up over the fastest solver"]
    return fastest, table(header, rows) if rows else ""


def solver_section(records, config, run_dir):
    """Return the section of the solver view."""
    view = config["solver_view"]
    approximate = {
        label: entry for label, entry in view["solvers"].items() if "method" in entry
    }
    windowed = [record for record in records if record["question"] != "distance"]
    ran = [record for record in windowed if record["solver"] in approximate]
    exact = [record for record in windowed if record["solver"] not in approximate]

    def cases(records):
        found = sorted({(r["cells"], r["case"]) for r in records})
        return either((case_label(case) for _, case in found), "and")

    used = [approximate[label] for label in dict.fromkeys(r["solver"] for r in ran)]
    methods = either(dict.fromkeys(code(entry["method"]) for entry in used))
    solvers = either((code(entry["solver"]) for entry in used), "and with")
    rasters = "**Rasters**: in the distance calculation, every raster of its distance"
    windows = []
    if exact:
        windows.append((cases(exact), "exact resistance distance"))
    if ran:
        windows.append((cases(ran), f"{methods} with {solvers}"))
    if len(windows) == 2 and windows[0][0] == windows[1][0]:
        under = either((what for _, what in windows), "and under")
        rasters += f"; in the windowed analyses, {windows[0][0]}, under {under}"
    elif windows:
        under = (f"{found} under {what}" for found, what in windows)
        rasters += f"; in the windowed analyses, {either(under, 'and')}"
    left_out = "; ".join(
        combination(part)
        for entry in view.get("fails_by_design", [])
        for part in not_run(entry, records, view)
    )
    limit = f"**Time limit**: a call stops after {duration(view['timeout'])}"
    longer = used[0].get("timeout", view["timeout"]) if used else view["timeout"]
    if longer != view["timeout"]:
        limit += f", and under {methods} after {duration(longer)}"
    exponent = f"$10^{{{np.log10(SMALLEST_ERROR):.0f}}}$"
    cholmod = any(record["solver"] == "CholmodSolver" for record in records)
    cpu = " `CholmodSolver` factorizes and solves on the CPU." if cholmod else ""
    method = [
        "**Solvers**: JAXScape solves the linear systems of resistance and "
        "free-energy distances with one of several solvers, on the "
        f"{view['device'].upper()}, in {either(view['dtypes'], 'and')}. `dense` "
        f"is the dense solve, `solver=None`.{cpu}",
        rasters + ".",
        "**Error**: in the distance calculation, the largest relative error "
        "with respect to the float64 reference of the distance calculation; in "
        "the windowed analyses, the relative L2 error $\\lVert m - "
        "m_{\\mathrm{ref}} \\rVert_2 / \\lVert m_{\\mathrm{ref}} \\rVert_2$ of "
        "the map $m$ with respect to the map $m_{\\mathrm{ref}}$ of exact "
        f"resistance distance in float64 with {code(view['reference_solver'])} "
        "on the same tiles"
        + (f", measured on {cases(exact)} alone" if exact else "")
        + f". In the figure of error, errors below {exponent} are drawn at "
        f"{exponent}.",
        limit + ".",
        f"**Not run**: {left_out}." if left_out else "",
    ]
    fastest, approximations = fastest_solvers(records, approximate)
    return [
        "## Linear solvers of JAXScape",
        "The fastest solver of each question, distance and dtype, without an "
        "approximation of the distance, on the largest synthetic raster on "
        "which one succeeded:"
        if fastest
        else "",
        fastest,
        "The approximations of resistance distance, each on the raster of its "
        "row above, or else on its largest synthetic raster:"
        if approximations
        else "",
        approximations,
        "### Results",
        picture(run_dir, "solvers.png", "Linear solvers: warm runtime"),
        picture(run_dir, "solver-errors.png", "Linear solvers: warm runtime and error"),
        collapsed(
            "Warm runtime and error of every solver on every raster",
            "An entry without an error has no reference.",
            outcome_table(records, SOLVER_VIEW),
        ),
        *failures(records, SOLVER_VIEW),
        "### Method",
        bullets(method),
    ]


def page(records, config, environment, maryland=None, run_dir=None):
    """Return the Benchmark page of a run.

    **Arguments:**

    - `records`: The records of the run.
    - `config`: The configuration of the run, which gives the settings that the
      page states.
    - `environment`: The environment snapshot of the run.
    - `maryland`: The records of the Maryland figure, when the run directory
      has it.
    - `run_dir`: The run directory, whose figures give their width on the
      page.

    **Returns:**

    The page, in Markdown.
    """
    solvers = solver_view(records)
    comparison = [record for record in records if record["view"] == "comparison"]
    sections = [*introduction(), *setup_section(records, config, environment)]
    for question, section in (
        ("distance", distance_section),
        ("connectivity", connectivity_section),
        ("sensitivity", lambda *a: sensitivity_section(*a, maryland)),
    ):
        if of_question(question, comparison):
            sections += section(of_question(question, comparison), config, run_dir)
    if solvers:
        sections += solver_section(solvers, config, run_dir)
    return "\n\n".join(section for section in sections if section) + "\n"


if __name__ == "__main__":
    main()
