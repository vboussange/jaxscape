# A fake tool for the runner's tests. For the distance calculation, it returns
# the Euclidean distances among the sites, and for the windowed analyses the
# quality, as a map, or minus the quality with `reversed`, both times 1 +
# `error`; with `off_graph`, its map is the exponential of the quality, NaN on
# the first pixel, as on a pixel off a tool's graph, and with `nan`, its output
# is NaN. It raises on the cases of `fail_on`, and outlasts the timeout on the
# cases of `sleep_on`.
import time
from pathlib import Path

import numpy as np
from adapters.protocol import read_cell, time_calls, write_result


cell, arrays = read_cell()
settings = cell["settings"]
case = Path(cell["case"]).name
if case in settings.get("fail_on", []):
    raise RuntimeError(f"fake failure on {case}")


def call():
    if case in settings.get("sleep_on", []):
        time.sleep(600)
    if cell["question"] == "distance":
        sites = arrays["sites"]
        output = np.linalg.norm(sites[:, None] - sites[None, :], axis=-1)
    elif settings.get("off_graph"):
        output = np.exp(arrays["quality"])
        output[0, 0] = np.nan
    else:
        output = -arrays["quality"] if settings.get("reversed") else arrays["quality"]
    scale = np.nan if settings.get("nan") else 1 + settings.get("error", 0)
    return output * scale


write_result(cell, *time_calls(cell, call))
