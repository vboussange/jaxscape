"""Join original/corrected benchmark records without changing source artifacts."""

import csv
import hashlib
import json
from pathlib import Path


def join(root):
    root = Path(root)
    original = json.loads((root / "original/benchmark_results.json").read_text())
    correction = json.loads(
        (root / "inverse-correction/benchmark_results.json").read_text()
    )
    key = lambda record: (record["task"], record["scenario"], record["tool"])
    replacement = {key(record): record for record in correction["records"]}
    assert len(original["records"]) == 44
    assert len(replacement) == 2
    assert all(
        task == "inverse_landscape_genetics" and "JAXAMG" in tool
        for task, _, tool in replacement
    )
    assert set(replacement).issubset({key(record) for record in original["records"]})
    assert (
        original["environment"]["dependency_versions"]
        == correction["environment"]["dependency_versions"]
    )
    payload = dict(original)
    payload["records"] = [
        replacement.get(key(record), record) for record in original["records"]
    ]
    payload["provenance"] = {"kind": "joined_report", "manifest": "manifest.json"}
    manifest = {"kind": "joined_report", "source_runs": {}, "record_sources": []}
    for label in ["original", "inverse-correction"]:
        source = json.loads((root / label / "manifest.json").read_text())
        assert source["source_status"] == ""
        assert source["environment_overrides"]["CUDA_VISIBLE_DEVICES"] == "0"
        manifest["source_runs"][label] = {
            "manifest": f"{label}/manifest.json",
            "source_commit": source["source_commit"],
            "sha256": {
                name: hashlib.sha256((root / label / name).read_bytes()).hexdigest()
                for name in [
                    "benchmark_results.json",
                    "benchmark_results.csv",
                    "manifest.json",
                    "run.log",
                ]
            },
        }
    for record in payload["records"]:
        manifest["record_sources"].append(
            {
                "task": record["task"],
                "scenario": record["scenario"],
                "tool": record["tool"],
                "run": "inverse-correction"
                if key(record) in replacement
                else "original",
            }
        )
    (root / "benchmark_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (root / "benchmark_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "tool",
                "software",
                "scenario",
                "status",
                "median_seconds",
                "timings_seconds",
                "metrics",
                "note",
            ],
        )
        writer.writeheader()
        for record in payload["records"]:
            row = dict(record)
            for name in ["metrics", "timings_seconds"]:
                row[name] = json.dumps(row[name])
            writer.writerow(row)
    return payload, manifest


if __name__ == "__main__":
    join(Path(__file__).parent)
