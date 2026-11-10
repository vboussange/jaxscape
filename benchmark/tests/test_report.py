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


def test_report_writes_the_csv_with_one_row_per_record(tmp_path):
    run_dir = report(tmp_path)

    with open(run_dir / "results.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 10
    assert {row["status"] for row in rows} == {
        "ok",
        "failed",
        "timeout",
        "oom",
        "skipped",
    }


def test_report_writes_the_e1_figure(tmp_path):
    run_dir = report(tmp_path)

    assert (run_dir / "e1.png").read_bytes().startswith(b"\x89PNG")


def test_e1_table_lists_each_configuration_with_its_value_or_status(tmp_path):
    page = (report(tmp_path) / "index.md").read_text()

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


def test_page_has_its_sections_and_references_the_figure(tmp_path):
    page = (report(tmp_path) / "index.md").read_text()

    for heading in (
        "## Questions",
        "## Comparison",
        "### E1. Isolation between populations",
        "## Environment",
        "## Parameter mappings",
        "## Excluded tools",
    ):
        assert heading + "\n" in page
    assert "](e1.png)" in page
    assert "](results.csv)" in page
    assert (
        table_row(page, "GPU")[1]
        == "NVIDIA GeForce RTX 3090, driver 580.178.04, 24576 MiB"
    )
    assert table_row(page, "Cores")[1] == "48-63"
    assert table_row(page, "Tools")[1] == "JAXScape 0.0.16"
    mapping = table_row(page, "JAXScape", "E1")
    assert mapping[2] == "`adapters/jaxscape.py`"
    assert "rook contiguity" in mapping[3]
    assert table_row(page, "Linkage Mapper")[1] == "Needs ArcGIS."
