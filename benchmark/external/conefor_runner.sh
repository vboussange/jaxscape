#!/usr/bin/env bash
set -euo pipefail
CONEFOR_BIN="$1"
TASK="$2"
OUTPUT_JSON="$3"
python - <<'PY' "$CONEFOR_BIN" "$TASK" "$OUTPUT_JSON"
import json
import shutil
import sys
from pathlib import Path

conefor_bin, task, output_json = sys.argv[1:4]
status = "skipped"
note = f"Conefor adapter is configured for '{task}' but requires project-specific input wiring for {conefor_bin}."
if shutil.which(conefor_bin) or Path(conefor_bin).exists():
    note = f"Conefor executable detected at {conefor_bin}, but the benchmark input export should be customised before running this adapter."
Path(output_json).write_text(json.dumps({"status": status, "timings_seconds": [], "median_seconds": None, "note": note}, indent=2))
PY
