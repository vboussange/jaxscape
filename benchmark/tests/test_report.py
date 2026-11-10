import csv
import shutil
import subprocess
import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixture"


def report(tmp_path):
    run_dir = tmp_path / "run"
    shutil.copytree(FIXTURE, run_dir)
    subprocess.run(
        [sys.executable, "report.py", str(run_dir)], cwd=BENCHMARK, check=True
    )
    return run_dir


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


def test_report_writes_the_csv_with_one_row_per_record(tmp_path):
    run_dir = report(tmp_path)

    with open(run_dir / "results.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 33
    assert {row["status"] for row in rows} == {
        "ok",
        "failed",
        "timeout",
        "oom",
        "skipped",
    }


def test_report_writes_one_figure_per_question(tmp_path):
    run_dir = report(tmp_path)

    for figure in ("e1.png", "e2.png", "e3.png"):
        assert (run_dir / figure).read_bytes().startswith(b"\x89PNG")


def test_e1_table_lists_each_configuration_with_its_value_or_status(tmp_path):
    page = section(
        (report(tmp_path) / "index.md").read_text(),
        "### E1. Isolation between populations",
    )

    gpu = table_row(
        page, "JAXScape", "resistance", "GPU", "float32", "GridMultigridSolver"
    )
    assert gpu[-4:] == ["2.1e-06", "3.4e-06", "oom", "skipped"]
    cpu = table_row(
        page, "JAXScape", "resistance", "CPU", "float32", "GridMultigridSolver"
    )
    assert cpu[-4:] == ["missing", "timeout", "skipped", ""]
    lcp = table_row(page, "JAXScape", "lcp", "GPU", "float32", "")
    assert lcp[-4:] == ["failed", "skipped", "", ""]
    free_energy = table_row(
        page, "JAXScape", "free_energy", "GPU", "float64", "CuDSSSolver"
    )
    assert free_energy[-4:] == ["1.0e-09", "", "", ""]
    amg = table_row(page, "Circuitscape.jl", "resistance", "CPU", "float64", "cg+amg")
    assert amg[-4:] == ["4.2e-07", "5.1e-07", "", ""]
    cholmod = table_row(
        page, "Circuitscape.jl", "resistance", "CPU", "float64", "cholmod"
    )
    assert cholmod[-4:] == ["3.0e-13", "oom", "", ""]
    conscape = table_row(page, "ConScape.jl", "free_energy", "CPU", "float64", "")
    assert conscape[-4:] == ["2.0e-14", "", "", ""]


def test_e2_and_e3_tables_list_each_configuration_with_its_status(tmp_path):
    page = (report(tmp_path) / "index.md").read_text()
    e2 = section(page, "### E2. Habitat connectivity")
    e3 = section(page, "### E3. Corridors and pinch points")

    assert table_row(e2, "Tool")[-5:] == ["Tier", "256²", "Maryland", "1024²", "4096²"]
    resistance = table_row(
        e2, "JAXScape", "resistance", "GPU", "float32", "GridMultigridSolver"
    )
    assert resistance[-5:] == ["", "ok", "ok", "timeout", "skipped"]
    assert table_row(e2, "JAXScape", "lcp", "CPU")[-4:] == ["ok", "ok", "", ""]
    cpu = table_row(e2, "JAXScape", "resistance", "CPU")
    assert cpu[-4:] == ["oom", "skipped", "skipped", ""]
    eco = table_row(e2, "eco-connectivity", "lcp", "CPU", "float32", "")
    assert eco[-5:] == ["2", "0.98", "0.81", "timeout", ""]
    graphab = table_row(e2, "Graphab", "lcp", "CPU", "float64", "")
    assert graphab[-5:] == ["3", "-0.06", "missing", "", ""]
    assert "Spearman's ρ" in e2
    assert table_row(e3, "Tool")[-2:] == ["256²", "Maryland"]
    assert table_row(e3, "JAXScape", "resistance", "GPU")[-2:] == ["ok", "ok"]
    assert table_row(e3, "JAXScape", "lcp", "GPU")[-2:] == ["failed", "skipped"]


def test_page_has_its_sections_and_references_the_figure(tmp_path):
    page = (report(tmp_path) / "index.md").read_text()

    for heading in (
        "## Questions",
        "## Comparison",
        "### E1. Isolation between populations",
        "### E2. Habitat connectivity",
        "### E3. Corridors and pinch points",
        "## Environment",
        "## Parameter mappings",
        "## Excluded tools",
    ):
        assert heading + "\n" in page
    for figure in ("e1.png", "e2.png", "e3.png"):
        assert f"]({figure})" in page
    assert "](results.csv)" in page
    assert (
        table_row(page, "GPU")[1]
        == "NVIDIA GeForce RTX 3090, driver 580.178.04, 24576 MiB"
    )
    assert table_row(page, "Cores")[1] == "48-63"
    assert (
        table_row(page, "Tools")[1]
        == "JAXScape 0.0.16, Circuitscape.jl 6.1.0, ConScape.jl 0.3.0, "
        "eco-connectivity 2.5.0, Graphab 3.0.8"
    )
    mapping = table_row(page, "JAXScape", "E1, E2, E3")
    assert mapping[2] == "`adapters/jaxscape.py`"
    assert "rook contiguity" in mapping[3]
    circuitscape = table_row(page, "Circuitscape.jl", "E1")
    assert circuitscape[2] == "`adapters/circuitscape.jl`"
    assert "connect_using_avg_resistances" in circuitscape[3]
    conscape = table_row(page, "ConScape.jl", "E1")
    assert conscape[2] == "`adapters/conscape.jl`"
    assert "AverageWeight" in conscape[3]
    eco = table_row(page, "eco-connectivity", "E2")
    assert eco[2] == "`adapters/eco_connectivity.py`"
    assert "exp(-d/25)" in eco[3]
    graphab = table_row(page, "Graphab", "E2")
    assert graphab[2] == "`adapters/graphab.py`"
    assert "F_I + a_I" in graphab[3]
    assert table_row(page, "Linkage Mapper")[1] == "Needs ArcGIS."
