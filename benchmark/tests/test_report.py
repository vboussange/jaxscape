import json
import re
import shutil
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixture"
SECTIONS = (
    "## Distance calculation",
    "## Connectivity analysis",
    "## Sensitivity analysis",
    "## Solvers",
    "## Setup",
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


def test_page_has_a_section_per_task_then_the_solvers_and_the_setup(report):
    text = page(report())

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings == list(SECTIONS)


def test_introduction_names_no_release_and_setup_takes_it_from_the_snapshot(report):
    text = page(report())

    assert "0.0.18" not in text.split(SECTIONS[0])[0]
    [(_, tools)] = collapsed(section(text, "## Setup"))
    assert table_row(tools, "JAXScape", "0.0.18")


def test_report_reads_the_records_and_writes_no_other_table_of_them(report):
    run_dir = report()

    assert "](records.csv)" in page(run_dir)
    assert not (run_dir / "results.csv").exists()


def test_report_writes_a_figure_per_task_and_two_of_the_solvers(report):
    run_dir = report()
    text = page(run_dir)

    for heading, figure in zip(
        SECTIONS,
        ("distance.png", "connectivity.png", "sensitivity.png", "solvers.png"),
    ):
        assert f"]({figure})" in section(text, heading)
        assert (run_dir / figure).read_bytes().startswith(b"\x89PNG")
    assert "](solver-errors.png)" in section(text, "## Solvers")
    assert (run_dir / "solver-errors.png").read_bytes().startswith(b"\x89PNG")


def test_each_table_is_collapsed_and_states_its_quantities_and_units(report):
    text = page(report())

    titles = {
        heading: [title for title, _ in collapsed(section(text, heading))]
        for heading in SECTIONS[:4]
    }
    assert titles == {
        "## Distance calculation": [
            "Warm runtime (s) · largest relative error against the float64 reference"
        ],
        "## Connectivity analysis": [
            "Warm runtime (s) · Spearman ρ with JAXScape's map · landscape total C"
        ],
        "## Sensitivity analysis": [
            "Warm runtime (s) · Spearman ρ with JAXScape's map"
        ],
        "## Solvers": ["Warm runtime (s) · error against the float64 reference"],
    }
    # Every table is in a collapsed note.
    assert not [line for line in text.splitlines() if line.startswith("|")]


def test_distance_table_lists_each_setting_with_its_runtime_and_error(report):
    [(_, table)] = collapsed(section(page(report()), "## Distance calculation"))

    gpu = table_row(
        table, "JAXScape", "resistance", "GPU", "float32", "GridMultigridCGSolver"
    )
    assert gpu[-4:][:2] == ["0.061 s · 2.1e-06", "0.41 s · 3.4e-06"]
    cpu = table_row(
        table, "JAXScape", "resistance", "CPU", "float32", "GridMultigridCGSolver"
    )
    assert cpu[-4:][:2] == ["0.39 s · missing", "timeout (3)"]
    free_energy = table_row(
        table, "JAXScape", "free energy", "GPU", "float64", "CuDSSSolver"
    )
    assert free_energy[-4:] == ["0.53 s · 1.0e-09", "", "", ""]
    amg = table_row(table, "Circuitscape.jl", "resistance", "CPU", "float64", "cg+amg")
    assert amg[-4:] == ["0.9 s · 4.2e-07", "6 s · 5.1e-07", "", ""]
    conscape = table_row(table, "ConScape.jl", "free energy", "CPU", "float64", "")
    assert conscape[-4:] == ["2.5 s · 2.0e-14", "", "", ""]


def test_a_cell_that_did_not_succeed_shows_its_status_and_recorded_reason(report):
    [(_, table)] = collapsed(section(page(report()), "## Distance calculation"))

    gpu = table_row(
        table, "JAXScape", "resistance", "GPU", "float32", "GridMultigridCGSolver"
    )
    oom, skipped = gpu[-2:]
    assert re.fullmatch(r"oom \(\d+\)", oom)
    assert re.fullmatch(r"skipped \(\d+\)", skipped)
    reasons = dict(re.findall(r"^(\d+)\. `(.*)`$", table, re.M))
    assert reasons[oom[5:-1]] == (
        "jaxlib.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying "
        "to allocate 9.1GiB."
    )
    assert reasons[skipped[9:-1]] == "after oom at synthetic-4096"
    lcp = table_row(table, "JAXScape", "least-cost path", "GPU", "float32", "")
    assert lcp[-4:-2] == ["failed (5)", "skipped (6)"]
    assert reasons["3"] == "a call exceeded 3600 s"
    assert reasons["5"] == "ValueError: shapes do not match"


def test_connectivity_table_lists_the_rank_correlation_and_the_landscape_total(
    report,
):
    [(_, table)] = collapsed(section(page(report()), "## Connectivity analysis"))

    assert table_row(table, "Tool")[-5:] == [
        "Tier",
        "256²",
        "Maryland",
        "1024²",
        "4096²",
    ]
    resistance = table_row(table, "JAXScape", "resistance", "GPU", "float32")
    assert resistance[-4:-2] == ["50 s · C = 1.2e+05", "200 s · C = 4.1e+05"]
    eco = table_row(table, "eco-connectivity", "least-cost path", "CPU", "float32")
    assert eco[-5:-1] == ["2", "0.2 s · ρ = 0.98", "1 s · ρ = 0.81", "timeout (1)"]
    graphab = table_row(table, "Graphab", "least-cost path", "CPU", "float64", "")
    assert graphab[-5:-2] == [
        "3",
        "4 s · ρ = -0.06 · C = 7.3e+04",
        "9.5 s · ρ missing · C = 2.6e+05",
    ]


def test_sensitivity_section_labels_the_normalized_map_and_links_its_example(
    report,
):
    text = section(page(report()), "## Sensitivity analysis")
    [(_, table)] = collapsed(text)
    label = "normalized sensitivity (K = −R)"

    gpu = table_row(table, "JAXScape", label, "GPU", "float32")
    assert gpu[-2:] == ["70 s", "300 s"]
    amg = table_row(table, "Omniscape.jl", "resistance", "CPU", "float64", "cg+amg")
    assert amg[-3:] == ["3", "10 s · ρ = 0.41", "37 s · ρ = 0.62"]
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
    assert "](maryland.png)" in text
    # The fastest Omniscape setting that succeeds on Maryland is cg+amg, and
    # JAXScape's map is that of the GPU cell that Omniscape is compared with.
    assert (
        "On Maryland, JAXScape's normalized sensitivity (K = −R) (GPU, 300 s) "
        "and Omniscape.jl's normalized current (CPU, `cg+amg`, 37 s)" in text
    )
    assert "ρ = 0.62" in text


def test_report_without_the_maps_keeps_the_maryland_figure(report):
    run_dir = report()
    figure = (run_dir / "maryland.png").read_bytes()
    shutil.rmtree(run_dir / "maps")  # the maps of a run are not committed

    report()

    assert (run_dir / "maryland.png").read_bytes() == figure
    assert "](maryland.png)" in page(run_dir)


def test_solver_table_names_the_solvers_of_the_current_api(report):
    [(_, table)] = collapsed(section(page(report()), "## Solvers"))

    assert table_row(table, "Task") == [
        "Task",
        "Distance",
        "dtype",
        "Solver",
        "256²",
        "1024²",
        "4096²",
    ]
    rows = {
        ("distance", "resistance", "float32", "GridMultigridCGSolver"): [
            "0.03 s · 2.0e-06",
            "0.35 s · 3.0e-06",
            "5.1 s · 4.0e-06",
        ],
        ("distance", "free energy", "float64", "CuDSSSolver"): [
            "0.35 s · 4.7e-14",
            "",
            "",
        ],
        ("connectivity", "resistance", "float32", "GridMultigridCGSolver"): [
            "91 s · 3.1e-04 · C = 1.2e+05",
            "",
            "",
        ],
        (
            "connectivity",
            "resistance",
            "float64",
            "SpielmanApproximation(epsilon=0.15), CuDSSSolver",
        ): ["1.5 s · 4.0e-02 · C = 1.2e+05", "12 s · C = 4.6e+05", ""],
        ("sensitivity", "resistance", "float64", "PyAMGCGSolver"): [
            "timeout (1)",
            "",
            "",
        ],
    }
    for configuration, cells in rows.items():
        assert table_row(table, *configuration)[4:] == cells


def test_solver_section_states_the_combinations_that_do_not_run(report):
    text = section(page(report()), "## Solvers")

    assert (
        "These combinations are not run: free energy in float32; free energy with "
        "`GridMultigridCGSolver`, `AMJaxCGSolver` or `PyAMGCGSolver`; "
        "`AMJaxCGSolver` in float32." in text
    )


def test_setup_states_the_hardware_protocol_and_versions_of_the_run(report):
    text = page(report())
    setup = section(text, "## Setup")

    assert (
        "NVIDIA GeForce RTX 3090 with 24576 MiB, driver 580.178.04 and CUDA 13.0"
        in setup
    )
    assert "16 cores of an AMD EPYC 7742 64-Core Processor" in setup
    assert "Ubuntu 22.04.5 LTS" in setup
    assert "the timed calls that follow it, 5 per cell" in setup
    assert "when the warm-up takes more than 5 min" in setup
    assert "A call stops after 1 h" in setup
    assert "Python 3.13.5, JAX 0.12.0.1, Julia 1.13.3" in setup
    [(title, tools)] = collapsed(setup)
    assert title == "Tools: version, tasks and parameter mapping"
    jaxscape = table_row(tools, "JAXScape", "0.0.18")
    assert jaxscape[2] == "distance, connectivity, sensitivity"
    assert "rook contiguity" in jaxscape[3]
    assert table_row(tools, "Omniscape.jl", "0.6.4", "sensitivity")
    # The page names no host and no GPU of a cluster.
    for detail in ("gpunode", "GPU-b8590203", "GPU 5", "80-95"):
        assert detail not in text


def test_page_states_the_settings_of_the_run(report):
    text = page(report())

    assert "`dependency_range` 100 pixels with `coarsening_factor` 0.2" in text
    assert "blocks of 21 pixels" in text
    assert (
        "A call stops after 15 min, and under "
        "`SpielmanApproximation(epsilon=0.15)` after 1 h." in text
    )


def test_page_of_a_smoke_run_states_the_smoke_settings(report):
    def smoke(config):
        config["run"].update(repeats=1, timeout=600)
        config["windows"].update(dependency_range=10, block_size=3)
        config["solver_view"]["timeout"] = 120

    text = page(report(edit=smoke))

    assert "the timed calls that follow it, 1 per cell" in text
    assert "A call stops after 10 min" in text
    assert "`dependency_range` 10 pixels with `coarsening_factor` 0.2" in text
    assert "blocks of 3 pixels" in text
    assert "A call stops after 2 min, and under" in text


def test_report_points_the_docs_link_at_the_run_only_when_asked(report, tmp_path):
    link = tmp_path / "docs" / "benchmark"
    link.parent.mkdir()
    link.symlink_to("../elsewhere")

    run_dir = report()
    assert link.readlink() == Path("../elsewhere")

    report("--publish", link)
    assert link.resolve() == run_dir.resolve()
    assert not link.readlink().is_absolute()
