# A fake tool for the runner's tests: it returns the Euclidean distances among
# the sites for E1 and the quality for other questions, raises on rasters of
# side `fail_from` and larger, and outlasts the timeout on rasters of side
# `sleep_from` and larger.
import time

import numpy as np
from adapters.protocol import read_cell, time_calls, write_result


cell, arrays = read_cell()
side = arrays["permeability"].shape[0]
settings = cell["settings"]
if side >= settings.get("fail_from", np.inf):
    raise RuntimeError(f"fake failure on a raster of side {side}")


def call():
    if side >= settings.get("sleep_from", np.inf):
        time.sleep(600)
    if cell["question"] != "E1":
        return arrays["quality"]
    sites = arrays["sites"]
    return np.linalg.norm(sites[:, None] - sites[None, :], axis=-1)


write_result(cell, *time_calls(cell, call), version="1.0")
