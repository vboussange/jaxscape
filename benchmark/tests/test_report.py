import json
import re
import shutil
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixture"
SECTIONS = (
    "## Setup",
    "## Distance calculation",
    "## Connectivity analysis",
    "## Sensitivity analysis",
    "## Linear solvers of JAXScape",
)


@pytest.fixture
def report(script, tmp_path):
    """Return a function that reports a copy of the fixture run.

    The function takes the arguments of `report.py` and `edit`, a function that
    changes the configuration of the run, and returns the run directory.
    """

    def report(*arguments, edit=None):
        run_dir = tmp_path / "run"
        if not run_dir.exists():
            shutil.copytree(FIXTURE, run_dir)
        if edit:
            config = json.loads((run_dir / "config.json").read_text())
            edit(config)
            (run_dir / "config.json").write_text(json.dumps(config))
        result = script("report.py", run_dir, *arguments)
        assert result.returncode == 0, result.stderr
        return run_dir

    return report


def page(run_dir):
    return (run_dir / "index.md").read_text()


def section(page, heading):
    """Return the section of `page` under a `##` heading, up to the next one."""
    after = page.split(heading + "\n", 1)[1]
    return after.split("\n## ", 1)[0]


def collapsed(section):
    """Return the title and the body of each collapsed note of a section."""
    notes = re.findall(r'^\?\?\? note "(.*)"\n\n((?:    .*\n|\n)*)', section, re.M)
    return [(title, re.sub(r"^    ", "", body, flags=re.M)) for title, body in notes]


def table_row(text, *cells):
    """Return the one table row of `text` that starts with `cells`."""
    start = "| " + " | ".join(cells) + " |"
    rows = [line for line in text.splitlines() if line.startswith(start)]
    assert len(rows) == 1, (start, rows)
    return [cell.strip() for cell in rows[0].strip("|").split("|")]


def full_table(text):
    """Return the collapsed table of every setting of a section."""
    [table] = [body for title, body in collapsed(text) if "every" in title]
    return table


GMG = "GPU, float32, GeometricMultigridCGSolver"


def test_page_has_the_setup_then_a_section_per_question_and_the_solvers(report):
    text = page(report())

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)


def test_page_states_the_release_once_in_the_setup(report):
    text = page(report())

    assert text.count("0.0.18") == 1
    tools = section(text, "## Setup")
    assert table_row(tools, "JAXScape", "0.0.18")


def test_page_joins_names_without_dots_and_drops_the_landscape_total(report):
    text = page(report())

    assert "·" not in text
    assert "effective distance" not in text.lower()
    assert not re.search(r"C = \d", text)


def test_report_reads_the_records_and_writes_no_other_table_of_them(report):
    run_dir = report()

    assert "](records.csv)" in page(run_dir)
    assert not (run_dir / "results.csv").exists()


def test_report_writes_a_figure_per_question_and_two_of_the_solvers(report):
    run_dir = report()
    text = page(run_dir)

    for heading, figure in zip(
        SECTIONS[1:],
        ("distance.png", "connectivity.png", "sensitivity.png", "solvers.png"),
    ):
        assert re.search(rf'<img src="{figure}" alt="[^"]+" width="\d+">', section(text, heading))
        assert (run_dir / figure).read_bytes().startswith(b"\x89PNG")
    assert '<img src="solver-errors.png"' in section(text, SECTIONS[4])
    assert (run_dir / "solver-errors.png").read_bytes().startswith(b"\x89PNG")


def test_figures_are_shown_at_one_scale_and_the_solver_figures_widest(report):
    text = page(report())

    widths = dict(re.findall(r'<img src="([\w-]+)\.png" alt="[^"]+" width="(\d+)">', text))
    assert all(int(width) <= 760 for width in widths.values())
    assert int(widths["solvers"]) > int(widths["distance"])


def test_each_table_of_every_setting_is_collapsed_and_states_its_quantities(report):
    text = page(report())

    titles = {
        heading: [title for title, _ in collapsed(section(text, heading))]
        for heading in SECTIONS[1:]
    }
    assert titles == {
        "## Distance calculation": [
            "Warm runtime and error of every setting on every raster"
        ],
        "## Connectivity analysis": [
            "Warm runtime and Spearman's ρ of every setting on every raster"
        ],
        "## Sensitivity analysis": [
            "Warm runtime and Spearman's ρ of every setting on every raster"
        ],
        "## Linear solvers of JAXScape": [
            "Warm runtime and error of every solver on every raster"
        ],
    }


def test_distance_section_states_the_speed_up_errors_and_limits_first(report):
    text = section(page(report()), "## Distance calculation")
    finding = text.split("\n### ")[0]

    assert (
        "JAXScape is faster than every other tool, by a factor of 4.7 "
        "(ConScape.jl, free-energy distance) to 15 (Circuitscape.jl, resistance "
        "distance)." in finding
    )
    assert "at most 3.4e-06 for JAXScape" in finding
    assert (
        "On the GPU, JAXScape ran out of memory under resistance distance on "
        "4096² and failed under least-cost path distance on 256²." in finding
    )
    # The fastest setting of Circuitscape.jl on 1024², the largest raster that
    # it shares with JAXScape on the GPU, is cg+amg.
    assert table_row(text, "Circuitscape.jl", "resistance", "1024²") == [
        "Circuitscape.jl",
        "resistance",
        "1024²",
        "6 s",
        "0.41 s",
        "15",
        "5.1e-07",
        "3.4e-06",
    ]


def test_distance_table_lists_each_setting_with_its_runtime_and_error(report):
    table = full_table(section(page(report()), "## Distance calculation"))

    gpu = table_row(table, "JAXScape", "resistance", GMG)
    assert gpu[3:] == ["0.061 s, error 2.1e-06", "0.41 s, error 3.4e-06", "oom", "skipped"]
    cpu = table_row(table, "JAXScape", "resistance", "CPU, float32, GeometricMultigridCGSolver")
    assert cpu[3:5] == ["0.39 s", "timeout"]
    free_energy = table_row(table, "JAXScape", "free energy", "GPU, float64, CuDSSSolver")
    assert free_energy[3:] == ["0.53 s, error 1.0e-09", "", "", ""]
    amg = table_row(table, "Circuitscape.jl", "resistance", "CPU, float64, cg+amg")
    assert amg[3:] == ["0.9 s, error 4.2e-07", "6 s, error 5.1e-07", "", ""]
    conscape = table_row(table, "ConScape.jl", "free energy", "CPU, float64")
    assert conscape[3:] == ["2.5 s, error 2.0e-14", "", "", ""]


def test_each_setting_that_stopped_shows_its_status_and_recorded_reason(report):
    text = section(page(report()), "## Distance calculation")
    stopped = text.split("### Settings that did not complete\n", 1)[1]

    assert table_row(stopped, "JAXScape", "resistance", GMG)[3:] == [
        "4096²",
        "`oom`",
        "allocation of 9.1 GiB failed",
    ]
    assert table_row(stopped, "JAXScape", "least-cost path", "GPU, float32")[3:] == [
        "256²",
        "`failed`",
        "`ValueError: shapes do not match`",
    ]
    cpu = "CPU, float32, GeometricMultigridCGSolver"
    assert table_row(stopped, "JAXScape", "resistance", cpu)[3:] == [
        "1024²",
        "`timeout`",
        "`a call exceeded 3600 s`",
    ]
    # Skipped cells follow from the protocol, and are not listed.
    assert "`skipped`" not in stopped


def test_connectivity_table_lists_the_rank_correlation_of_each_tool(report):
    table = full_table(section(page(report()), "## Connectivity analysis"))

    assert table_row(table, "Tool") == [
        "Tool",
        "Distance",
        "Setting",
        "256²",
        "Maryland",
        "1024²",
        "4096²",
    ]
    resistance = table_row(table, "JAXScape", "resistance", GMG)
    assert resistance[3:5] == ["50 s", "200 s"]
    eco = table_row(table, "eco-connectivity", "least-cost path", "CPU, float32")
    assert eco[3:6] == ["0.2 s, ρ = 0.98", "1 s, ρ = 0.81", "timeout"]
    graphab = table_row(table, "Graphab", "least-cost path", "CPU, float64")
    assert graphab[3:5] == ["4 s, ρ = -0.06", "9.5 s"]


def test_sensitivity_section_gives_the_speed_up_tier_and_rank_correlation(report):
    text = section(page(report()), "## Sensitivity analysis")
    label = "normalized sensitivity (K = −R)"

    assert (
        "JAXScape is slower than every other tool, by a factor of 8.8 "
        "(Omniscape.jl) to 22 (samc)." in text
    )
    # Omniscape.jl is compared with JAXScape's normalized map, in its fastest
    # setting on 256², cholmod.
    assert table_row(text, "Omniscape.jl", "3") == [
        "Omniscape.jl",
        "3",
        label,
        "256²",
        "8 s",
        "70 s",
        "0.11",
        "0.43",
    ]
    table = full_table(text)
    assert table_row(table, "Tool")[:3] == ["Tool", "Distance or label", "Setting"]
    gpu = table_row(table, "JAXScape", label, GMG)
    assert gpu[3:5] == ["70 s", "300 s"]
    amg = table_row(table, "Omniscape.jl", "resistance", "CPU, float64, cg+amg")
    assert amg[3:5] == ["10 s, ρ = 0.41", "37 s, ρ = 0.62"]
    assert "K = -R" not in text
    assert re.search(
        r"\[[^]]*\]\(\.\./examples/connectivity_analysis/"
        r"connectivity_analysis\.ipynb#[a-z-]+\)",
        text,
    )


def test_report_draws_the_maryland_figure_and_states_its_correlation(report):
    run_dir = report()
    text = section(page(run_dir), "## Sensitivity analysis")

    assert (run_dir / "maryland.png").read_bytes().startswith(b"\x89PNG")
    assert '<img src="maryland.png"' in text
    # The fastest Omniscape setting that succeeds on Maryland is cg+amg, and
    # JAXScape's map is that of the GPU cell that Omniscape is compared with.
    assert (
        "Rank of each pixel of Maryland in JAXScape's normalized sensitivity "
        "(K = −R) and in Omniscape.jl's normalized current (`cg+amg`), with "
        "barriers in gray. Spearman's ρ between the two maps is 0.62." in text
    )


def test_report_without_the_maps_keeps_the_maryland_figure(report):
    run_dir = report()
    figure = (run_dir / "maryland.png").read_bytes()
    shutil.rmtree(run_dir / "maps")  # the maps of a run are not committed

    report()

    assert (run_dir / "maryland.png").read_bytes() == figure
    assert '<img src="maryland.png"' in page(run_dir)


def test_solver_section_names_the_fastest_solver_and_lists_every_solver(report):
    text = section(page(report()), "## Linear solvers of JAXScape")
    fastest = text.split("\n### ")[0]

    assert table_row(fastest, "distance calculation", "resistance", "float32") == [
        "distance calculation",
        "resistance",
        "float32",
        "4096²",
        "GeometricMultigridCGSolver",
        "5.1 s",
        "4.0e-06",
        "",
        "",
    ]
    assert table_row(fastest, "distance calculation", "resistance", "float64")[4:] == [
        "CuDSSSolver",
        "0.3 s",
        "1.0e-12",
        "",
        "",
    ]
    table = full_table(text)
    assert table_row(table, "Question") == [
        "Question",
        "Distance",
        "dtype",
        "Solver",
        "256²",
        "1024²",
        "4096²",
    ]
    rows = {
        ("distance calculation", "resistance", "float32", "GeometricMultigridCGSolver"): [
            "0.03 s, error 2.0e-06",
            "0.35 s, error 3.0e-06",
            "5.1 s, error 4.0e-06",
        ],
        ("distance calculation", "free energy", "float64", "CuDSSSolver"): [
            "0.35 s, error 4.7e-14",
            "",
            "",
        ],
        ("connectivity analysis", "resistance", "float32", "GeometricMultigridCGSolver"): [
            "91 s, error 3.1e-04",
            "",
            "",
        ],
        (
            "connectivity analysis",
            "resistance",
            "float64",
            "SpielmanApproximation(epsilon=0.15), CuDSSSolver",
        ): ["1.5 s, error 4.0e-02", "12 s", ""],
    }
    for configuration, cells in rows.items():
        assert table_row(table, *configuration)[4:] == cells


def test_solver_section_states_the_combinations_that_no_record_ran(report):
    def leave_out_cudss(config):
        config["solver_view"]["fails_by_design"].append(
            {"distance": "free_energy", "solver": ["CuDSSSolver", "CholmodSolver"]}
        )

    text = section(page(report(edit=leave_out_cudss)), "## Linear solvers of JAXScape")

    # AMJaxCGSolver in float32 and CuDSSSolver on free-energy distance have
    # records, and the page does not list them.
    assert (
        "**Not run**: free-energy distance in float32; free-energy distance with "
        "`GeometricMultigridCGSolver` or `AMJaxCGSolver`; free-energy distance with "
        "`CholmodSolver`." in text
    )


def test_setup_states_the_hardware_protocol_and_versions_of_the_run(report):
    text = page(report())
    setup = section(text, "## Setup")

    assert (
        "NVIDIA GeForce RTX 3090 with 24576 MiB, driver 580.178.04, CUDA 13.0"
        in setup
    )
    assert "16 cores of an AMD EPYC 7742 64-Core Processor" in setup
    assert "Ubuntu 22.04.5 LTS" in setup
    assert "the median time of the 5 timed calls of a cell" in setup
    assert "when the warm-up call takes more than 5 min" in setup
    assert (
        "**Time limit**: a call stops after 1 h, and in the section of the linear "
        "solvers of JAXScape after the time limit stated in its Method." in setup
    )
    assert "`timeout` when a call exceeds its time limit" in setup
    assert "the fastest setting of a tool, distance and device" in setup
    assert "Python 3.13.5, JAX 0.12.0.1, Julia 1.13.3" in setup
    jaxscape = table_row(setup, "JAXScape", "0.0.18")
    assert jaxscape[2] == "distance calculation, connectivity analysis, sensitivity analysis"
    assert table_row(setup, "Omniscape.jl", "0.6.4", "sensitivity analysis")
    [(title, mappings)] = collapsed(setup)
    assert title == "Parameter mapping of each tool"
    assert "rook contiguity" in table_row(mappings, "JAXScape")[1]
    # The page names no host and no GPU of a cluster.
    for detail in ("gpunode", "GPU-b8590203", "GPU 5", "80-95"):
        assert detail not in text


def test_page_states_the_settings_of_the_run(report):
    text = page(report())

    assert "`dependency_range` 100 pixels and `coarsening_factor` 0.2" in text
    assert "blocks of 21 × 21 pixels" in text
    assert (
        "**Time limit**: a call stops after 15 min, and under "
        "`SpielmanApproximation(epsilon=0.15)` after 1 h." in text
    )


def test_page_of_a_smoke_run_states_the_smoke_settings(report):
    def smoke(config):
        config["run"].update(repeats=1, timeout=600)
        config["windows"].update(dependency_range=10, block_size=3)
        config["solver_view"]["timeout"] = 120

    text = page(report(edit=smoke))

    assert "the median time of the 1 timed call of a cell" in text
    assert "**Time limit**: a call stops after 10 min" in text
    assert "`dependency_range` 10 pixels and `coarsening_factor` 0.2" in text
    assert "blocks of 3 × 3 pixels" in text
    assert "a call stops after 2 min, and under" in text


def test_report_of_rasters_of_any_side_draws_every_figure(report, tmp_path):
    # Smoke runs take 32² rasters, a side that the published runs do not have.
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    records = run_dir / "records.csv"
    records.write_text(records.read_text().replace("synthetic-256,65536", "synthetic-32,1024"))

    report()

    assert (run_dir / "solver-errors.png").read_bytes().startswith(b"\x89PNG")
    assert "32²" in page(run_dir)


def test_report_points_the_docs_link_at_the_run_only_when_asked(report, tmp_path):
    link = tmp_path / "docs" / "benchmark"
    link.parent.mkdir()
    link.symlink_to("../elsewhere")

    run_dir = report()
    assert link.readlink() == Path("../elsewhere")

    report("--publish", link)
    assert link.resolve() == run_dir.resolve()
    assert not link.readlink().is_absolute()
