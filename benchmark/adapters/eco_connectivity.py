# eco-connectivity computes the connectedness of each pixel i, with
# `option=1`: the sum over pixels j != i of exp(-d(i, j) / 25) q_j, divided by
# the same sum with quality 1 and every weight 1. Its numerator is Hanski's S_i;
# its denominator, the reach of an intact landscape, is constant away from the
# raster's edges. d is a least-cost distance over 8 neighbours, in which a step
# costs its length times the weight of the pixel it enters, here its resistance,
# 1 / permeability: the resistance raster is (r - 1) / (r_max - 1), and
# `max_cost` is r_max, the largest resistance. Quality is the condition
# raster. Barriers, of permeability 0, are masked, and the map is NaN there.
# Pixels are 1 km, so `lambdas=[25]` gives the kernel exp(-d/25) in pixels.
# The multi-resolution graph keeps the default `window_size`, `outer_window`
# and `window_mode`, and the default `levels` up to the first whose reach,
# `outer_window` times the level in pixels, covers `dependency_range`.
# `n_threads` is the number of cores of the cell. The tool reads GeoTIFFs:
# each call writes the two rasters to /dev/shm and computes the map from them.
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
import rasterio
from adapters.protocol import read_cell, time_calls, write_result
from connectivity import connectedness
from rasterio.transform import from_origin


OUTER_WINDOW = 9  # the default of `connectedness`

cell, arrays = read_cell()
if cell["question"] != "E2":
    raise ValueError("eco-connectivity computes the habitat connectivity of E2")
permeability = arrays["permeability"]
barrier = permeability == 0
resistance = 1 / np.where(barrier, 1, permeability)
largest = resistance.max()
rasters = {
    "condition": np.where(barrier, np.nan, arrays["quality"]),
    "resistance": np.where(barrier, np.nan, (resistance - 1) / (largest - 1)),
}
levels = [2]
while OUTER_WINDOW * levels[-1] < cell["windows"]["dependency_range"]:
    levels.append(2 * levels[-1])
rows, columns = permeability.shape
profile = dict(
    driver="GTiff",
    height=rows,
    width=columns,
    count=1,
    dtype="float32",
    nodata=np.nan,
    crs="EPSG:32633",
    transform=from_origin(500_000, 1000 * rows, 1000, 1000),
)
work = Path(cell["output"]).parent
files = {name: str(work / f"{name}.tif") for name in rasters}


def call():
    for name, raster in rasters.items():
        with rasterio.open(files[name], "w", **profile) as file:
            file.write(raster.astype("float32"), 1)
    return connectedness(
        condition_file=files["condition"],
        resistance_file=files["resistance"],
        lambdas=[25.0],
        max_cost=largest,
        levels=levels,
        option=1,
        n_threads=len(os.sched_getaffinity(0)),
    )


write_result(
    cell, *time_calls(cell, call), version=version("eco-connectivity"), files=True
)
