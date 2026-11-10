"""What a Python adapter shares with the runner.

The runner calls an adapter with the path of a cell file, a JSON object with:

- `question`, `distance`, `device`: what the cell computes, and where;
- `settings`: the tool's settings for the distance, with `dtype` and `solver`;
- `parameters`: the shared parameters of the distance, such as `theta`;
- `windows`: the `dependency_range` and `coarsening_factor` of E2 and E3;
- `case`: a directory with `quality.npy`, `permeability.npy` and, for E1,
  `sites.npy`, the row and column of each site;
- `repeats`, `single_call_after`: the timing protocol;
- `progress`, `output`, `result`: the files that the adapter writes.

The adapter touches `progress` at the start of each call, saves the output of
its last call in `output` (`.npy`) and writes `result`, a JSON object with its
`version`, `files` (whether its calls read or write files), `warmup_s` and
`times_s`. Adapters in other languages follow the same protocol.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np


def read_cell():
    """Return the cell file named on the command line, and the arrays of its case."""
    cell = json.loads(Path(sys.argv[1]).read_text())
    case = Path(cell["case"])
    arrays = {path.stem: np.load(path) for path in case.glob("*.npy")}
    return cell, arrays


def time_calls(cell, call):
    """Time `call`, a function that returns once its output is computed.

    One warm-up call comes first, then `repeats` timed calls, or at most one
    when the warm-up took longer than `single_call_after` seconds.

    **Returns:**

    The output of the last call, the warm-up time and the times of the timed
    calls, in seconds.
    """

    def timed():
        Path(cell["progress"]).touch()
        start = time.perf_counter()
        output = call()
        return output, time.perf_counter() - start

    output, warmup = timed()
    repeats = cell["repeats"]
    if warmup > cell["single_call_after"]:
        repeats = min(repeats, 1)
    times = []
    for _ in range(repeats):
        output, elapsed = timed()
        times.append(elapsed)
    return output, warmup, times


def write_result(cell, output, warmup, times, version, files=False):
    np.save(cell["output"], np.asarray(output))
    result = {"version": version, "files": files, "warmup_s": warmup, "times_s": times}
    Path(cell["result"]).write_text(json.dumps(result))
