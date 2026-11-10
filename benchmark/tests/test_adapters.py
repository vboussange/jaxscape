import json
import os
import subprocess
import sys
from pathlib import Path


BENCHMARK = Path(__file__).resolve().parents[1]
CORES = ",".join(str(core) for core in sorted(os.sched_getaffinity(0))[:2])
CONFIG = f"""
[run]
timeout = 600
repeats = 1
single_call_after = 300
cores = "{CORES}"
gpu = ""
smoke_side = 16

[landscape]
correlation_length = 4
habitat_threshold = 0.7
seed = 1

[E1]
sites = 6
seed = 2

[distances.resistance]
sides = [16]

[distances.lcp]
sides = [16]

[[tools]]
tool = "skimage.graph.MCP"
question = "E1"
adapter = "adapters/skimage.py"
devices = ["cpu"]
tier = 1

[tools.distances.lcp]

[[tools]]
tool = "gdistance"
question = "E1"
adapter = "adapters/gdistance.R"
devices = ["cpu"]
tier = 1

[tools.distances.resistance]

[tools.distances.lcp]
"""


def test_tier_1_adapters_agree_with_the_float64_reference(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(CONFIG)
    subprocess.run(
        [sys.executable, "run.py", tmp_path / "run", "--config", config],
        cwd=BENCHMARK,
        check=True,
    )
    lines = (tmp_path / "run" / "records.jsonl").read_text().splitlines()
    records = {(r["tool"], r["distance"]): r for r in map(json.loads, lines)}

    assert {cell: (r["status"], r["reason"]) for cell, r in records.items()} == {
        ("skimage.graph.MCP", "lcp"): ("ok", None),
        ("gdistance", "resistance"): ("ok", None),
        ("gdistance", "lcp"): ("ok", None),
    }
    errors = {cell: r["agreement"] for cell, r in records.items()}
    assert max(errors.values()) < 1e-12, errors


E2_CONFIG = f"""
[run]
timeout = 600
repeats = 1
single_call_after = 300
cores = "{CORES}"
gpu = ""
smoke_side = 16

[landscape]
correlation_length = 3
habitat_threshold = 0.7
seed = 1

[E1]
sites = 6
seed = 2

[distances.lcp]
sides = [64]

[windows]
dependency_range = 20
coarsening_factor = 0.2

[[tools]]
tool = "JAXScape"
question = "E2"
adapter = "adapters/jaxscape.py"
devices = ["cpu"]

[tools.distances.lcp]
dtype = "float32"
batch_size = 64

[[tools]]
tool = "eco-connectivity"
question = "E2"
adapter = "adapters/eco_connectivity.py"
devices = ["cpu"]
tier = 2

[tools.distances.lcp]

[[tools]]
tool = "Graphab"
question = "E2"
adapter = "adapters/graphab.py"
devices = ["cpu"]
tier = 3

[tools.distances.lcp]
"""


def test_e2_adapters_rank_pixels_as_jaxscape_does(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text(E2_CONFIG)
    subprocess.run(
        [sys.executable, "run.py", tmp_path / "run", "--config", config],
        cwd=BENCHMARK,
        check=True,
    )
    lines = (tmp_path / "run" / "records.jsonl").read_text().splitlines()
    records = {r["tool"]: r for r in map(json.loads, lines)}

    assert {tool: (r["status"], r["reason"]) for tool, r in records.items()} == {
        "JAXScape": ("ok", None),
        "eco-connectivity": ("ok", None),
        "Graphab": ("ok", None),
    }
    assert records["eco-connectivity"]["files"] is True
    assert records["Graphab"]["files"] is True
    assert records["eco-connectivity"]["agreement"] > 0.85
    assert records["Graphab"]["agreement"] > 0.4
