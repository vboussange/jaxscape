import csv
import json
import shutil
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixture"


@pytest.fixture
def report(script, tmp_path):
    """Return a function that reports a copy of the fixture run, with the
    changes `edit` makes to its configuration, and returns its directory."""

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


def table_row(page, *cells):
    """Return the one table row of `page` that starts with `cells`."""
    start = "| " + " | ".join(cells) + " |"
    rows = [line for line in page.splitlines() if line.startswith(start)]
    assert len(rows) == 1, (start, rows)
    return [cell.strip() for cell in rows[0].strip("|").split("|")]


def section(page, heading):
    """Return the section of `page` under `heading`, up to the next heading."""
    after = page.split(heading + "\n", 1)[1]
    return after.split("\n#", 1)[0]


def test_report_writes_the_csv_with_one_row_per_record(report):
    run_dir = report()

    with open(run_dir / "results.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 56
    assert {row["status"] for row in rows} == {
        "ok",
        "failed",
        "timeout",
        "oom",
        "skipped",
    }
    assert {row["total"] for row in rows if row["tool"] == "Graphab"} == {
        "73000.0",
        "260000.0",
    }


def test_report_writes_one_figure_per_question(report):
    run_dir = report()

    for figure in ("e1.png", "e2.png", "e3.png"):
        assert (run_dir / figure).read_bytes().startswith(b"\x89PNG")


def test_e1_table_lists_each_configuration_with_its_runtime_and_error_or_status(
    report,
):
    e1 = section(page(report()), "### E1. Isolation between populations")

    gpu = table_row(
        e1, "JAXScape", "resistance", "GPU", "float32", "GridMultigridSolver"
    )
    assert gpu[-4:] == ["0.061 s · 2.1e-06", "0.41 s · 3.4e-06", "oom", "skipped"]
    cpu = table_row(
        e1, "JAXScape", "resistance", "CPU", "float32", "GridMultigridSolver"
    )
    assert cpu[-4:] == ["0.39 s · missing", "timeout", "skipped", ""]
    lcp = table_row(e1, "JAXScape", "lcp", "GPU", "float32", "")
    assert lcp[-4:] == ["failed", "skipped", "", ""]
    free_energy = table_row(
        e1, "JAXScape", "free_energy", "GPU", "float64", "CuDSSSolver"
    )
    assert free_energy[-4:] == ["0.53 s · 1.0e-09", "", "", ""]
    amg = table_row(e1, "Circuitscape.jl", "resistance", "CPU", "float64", "cg+amg")
    assert amg[-4:] == ["0.9 s · 4.2e-07", "6 s · 5.1e-07", "", ""]
    cholmod = table_row(
        e1, "Circuitscape.jl", "resistance", "CPU", "float64", "cholmod"
    )
    assert cholmod[-4:] == ["0.4 s · 3.0e-13", "oom", "", ""]
    conscape = table_row(e1, "ConScape.jl", "free_energy", "CPU", "float64", "")
    assert conscape[-4:] == ["2.5 s · 2.0e-14", "", "", ""]


def test_e2_table_lists_the_rank_correlation_and_the_landscape_total(report):
    e2 = section(page(report()), "### E2. Habitat connectivity")

    assert table_row(e2, "Tool")[-5:] == ["Tier", "256²", "Maryland", "1024²", "4096²"]
    resistance = table_row(e2, "JAXScape", "resistance", "GPU", "float32")
    assert resistance[-5:] == [
        "",
        "50 s · total 1.2e+05",
        "200 s · total 4.1e+05",
        "timeout",
        "skipped",
    ]
    cpu = table_row(e2, "JAXScape", "resistance", "CPU")
    assert cpu[-4:] == ["oom", "oom", "skipped", ""]
    conscape = table_row(e2, "ConScape.jl", "free_energy", "CPU", "float64", "")
    assert conscape[-5:] == ["3", "15 s · 0.91", "", "", ""]
    eco = table_row(e2, "eco-connectivity", "lcp", "CPU", "float32", "")
    assert eco[-5:] == ["2", "0.2 s · 0.98", "1 s · 0.81", "timeout", ""]
    graphab = table_row(e2, "Graphab", "lcp", "CPU", "float64", "")
    assert graphab[-5:] == [
        "3",
        "4 s · -0.06 · total 7.3e+04",
        "9.5 s · missing · total 2.6e+05",
        "",
        "",
    ]
    assert "Spearman's ρ" in e2


def test_e3_table_lists_the_rank_correlation_of_each_omniscape_setting(report):
    e3 = section(page(report()), "### E3. Corridors and pinch points")

    assert table_row(e3, "Tool")[-2:] == ["256²", "Maryland"]
    assert table_row(e3, "JAXScape", "lcp", "GPU")[-2:] == ["failed", "failed"]
    amg = table_row(e3, "Omniscape.jl", "resistance", "CPU", "float64", "cg+amg")
    assert amg[-3:] == ["3", "10 s · 0.41", "37 s · 0.62"]
    cholmod = table_row(e3, "Omniscape.jl", "resistance", "CPU", "float64", "cholmod")
    assert cholmod[-3:] == ["3", "8 s · 0.43", "timeout"]


def test_report_writes_the_maryland_figure_with_its_correlation_and_runtimes(
    report,
):
    run_dir = report()
    e3 = section(page(run_dir), "### E3. Corridors and pinch points")

    assert (run_dir / "maryland.png").read_bytes().startswith(b"\x89PNG")
    assert "](maryland.png)" in e3
    # The fastest Omniscape setting that succeeds on Maryland is cg+amg, and
    # JAXScape's map is that of its GPU cell.
    assert "ρ = 0.62" in e3
    assert "240 s" in e3
    assert "37 s" in e3


def test_report_without_the_maps_keeps_the_maryland_figure(report):
    run_dir = report()
    figure = (run_dir / "maryland.png").read_bytes()
    shutil.rmtree(run_dir / "maps")  # the maps of a run are not committed

    report()

    assert (run_dir / "maryland.png").read_bytes() == figure
    assert "](maryland.png)" in page(run_dir)


def test_solver_table_lists_each_solver_with_its_runtime_and_error_or_status(
    report,
):
    solvers = section(page(report()), "## Solvers")

    assert table_row(solvers, "Question") == [
        "Question",
        "Distance",
        "dtype",
        "Solver",
        "256²",
        "1024²",
        "4096²",
    ]
    rows = {
        ("E1", "resistance", "float32", "GridMultigridSolver"): [
            "0.03 s · 2.0e-06",
            "0.35 s · 3.0e-06",
            "5.1 s · 4.0e-06",
        ],
        ("E1", "resistance", "float64", "CuDSSSolver"): [
            "0.3 s · 1.0e-12",
            "timeout",
            "",
        ],
        ("E1", "resistance", "float32", "AMJaxCGSolver"): ["failed", "skipped", ""],
        ("E1", "resistance", "float32", "dense"): ["oom", "skipped", ""],
        ("E1", "free_energy", "float64", "CuDSSSolver"): ["0.35 s · 4.7e-14", "", ""],
        ("E2", "resistance", "float32", "GridMultigridSolver"): [
            "91 s · 3.1e-04 · total 1.2e+05",
            "",
            "",
        ],
        ("E3", "resistance", "float32", "GridMultigridSolver"): [
            "95 s · missing",
            "",
            "",
        ],
        ("E3", "resistance", "float64", "PyAMGSolver"): ["timeout", "", ""],
    }
    for configuration, cells in rows.items():
        assert table_row(solvers, *configuration)[4:] == cells


def test_solver_section_lists_the_combinations_that_fail_by_design(report):
    solvers = section(page(report()), "## Solvers")

    assert (
        "leaves out the combinations that fail by design: `free_energy` in "
        "`float32`; `free_energy` with `AMJaxCGSolver` or `PyAMGSolver`; "
        "`AMJaxCGSolver` in `float32` on `E2` or `E3`." in solvers
    )


def test_report_writes_the_solver_figure(report):
    run_dir = report()

    assert (run_dir / "solvers.png").read_bytes().startswith(b"\x89PNG")
    assert "](solvers.png)" in page(run_dir)


def test_page_states_the_settings_of_the_run(report):
    text = page(report())

    assert "the timed calls that follow it, 5 per cell" in text
    assert "when the warm-up takes more than 5 min" in text
    assert "A call stops after 1 h." in text
    assert "`dependency_range` 100 and `coarsening_factor` 0.2" in text
    assert "blocks of 21 pixels" in text
    assert "a call stops after 15 min" in text
    assert "E2 with resistance on 256²" in text


def test_page_of_a_smoke_run_states_the_smoke_settings(report):
    def smoke(config):
        config["run"].update(repeats=1, timeout=600)
        config["windows"].update(dependency_range=10, block_size=3)
        config["solver_view"]["timeout"] = 120
        config["solver_view"]["questions"]["E2"]["resistance"]["sides"] = [32]

    text = page(report(edit=smoke))

    assert "the timed calls that follow it, 1 per cell" in text
    assert "A call stops after 10 min." in text
    assert "`dependency_range` 10 and `coarsening_factor` 0.2" in text
    assert "blocks of 3 pixels" in text
    assert "a call stops after 2 min" in text
    assert "E2 with resistance on 32²" in text


def test_report_points_the_docs_link_at_the_run_only_when_asked(report, tmp_path):
    link = tmp_path / "docs" / "benchmark"
    link.parent.mkdir()
    link.symlink_to("../elsewhere")

    run_dir = report()
    assert link.readlink() == Path("../elsewhere")

    report("--publish", link)
    assert link.resolve() == run_dir.resolve()
    assert not link.readlink().is_absolute()


def test_page_has_its_sections_and_references_the_figure(report):
    text = page(report())

    for heading in (
        "## Questions",
        "## Comparison",
        "### E1. Isolation between populations",
        "### E2. Habitat connectivity",
        "### E3. Corridors and pinch points",
        "## Solvers",
        "## Environment",
        "## Parameter mappings",
        "## Excluded tools",
    ):
        assert heading + "\n" in text
    for figure in ("e1.png", "e2.png", "e3.png"):
        assert f"]({figure})" in text
    assert "](results.csv)" in text
    assert table_row(text, "GPU")[1] == (
        "NVIDIA GeForce RTX 3090 (GPU 5, GPU-b8590203-5a3b-fdba-68c9-0142ee891fb0), "
        "driver 580.178.04, 24576 MiB"
    )
    assert table_row(text, "Cores")[1] == "80-95"
    assert table_row(text, "Tools")[1] == (
        "JAXScape 0.0.16, Circuitscape.jl 6.1.0, ConScape.jl 0.3.0, gdistance "
        "1.6.5, skimage.graph.MCP 0.26.0, EcoScape 0.2.5, samc 4.2.1, "
        "eco-connectivity 2.5.0, Graphab 3.0.8, Omniscape.jl 0.6.4"
    )
    mapping = table_row(text, "JAXScape", "E1, E2, E3")
    assert mapping[2] == "`adapters/jaxscape.py`"
    assert "rook contiguity" in mapping[3]
    circuitscape = table_row(text, "Circuitscape.jl", "E1")
    assert circuitscape[2] == "`adapters/circuitscape.jl`"
    assert "connect_using_avg_resistances" in circuitscape[3]
    conscape = table_row(text, "ConScape.jl", "E1, E2")
    assert conscape[2] == "`adapters/conscape.jl`"
    assert "AverageWeight" in conscape[3]
    assert "connected_habitat" in conscape[3]
    ecoscape = table_row(text, "EcoScape", "E3")
    assert ecoscape[2] == "`adapters/ecoscape.py`"
    assert "dispersal" in ecoscape[3]
    samc = table_row(text, "samc", "E3")
    assert samc[2] == "`adapters/samc.R`"
    assert "absorption" in samc[3]
    eco = table_row(text, "eco-connectivity", "E2")
    assert eco[2] == "`adapters/eco_connectivity.py`"
    assert "exp(-d/25)" in eco[3]
    graphab = table_row(text, "Graphab", "E2")
    assert graphab[2] == "`adapters/graphab.py`"
    assert "F_I + a_I" in graphab[3]
    omniscape = table_row(text, "Omniscape.jl", "E3")
    assert omniscape[2] == "`adapters/omniscape.jl`"
    assert "`radius` is the shared `dependency_range`" in omniscape[3]
    assert table_row(text, "Linkage Mapper")[1] == "Needs ArcGIS."
