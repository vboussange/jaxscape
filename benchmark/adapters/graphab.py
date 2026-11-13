# Graphab builds a graph of habitat patches, the clusters of habitat pixels
# (quality > 0) connected through 4 neighbours. Its links are least-cost paths
# between patches over 8 neighbours, in which a step costs its length times the
# mean resistance, 1 / permeability, of its two pixels (`extcost`), and
# barriers, of permeability 0, are no data. Links and graph keep the paths of
# cost up to `dependency_range` (`maxcost`, `threshold`, `topo=complete`),
# where the proximity falls to exp(-4). `d=25` and `p=exp(-1)` give the
# proximity exp(-d/25) on the cost of the shortest path in the graph, which
# adds the costs of crossing patches. The capacity a_I of a patch is its area
# in pixels. Each call computes the equivalent connectivity EC of the graph,
# whose square, the sum over patches I and J of a_I a_J exp(-d_IJ/25), is the
# landscape total, and the flux F_I = sum over patches J != I of a_J
# exp(-d_IJ/25) (`beta=1`). The map is F_I + a_I, the patch analogue of
# Hanski's S_i, which adds the patch's own term that F leaves out, on the
# pixels of each patch, and NaN elsewhere. `-proc` is the number of cores of
# the cell. Graphab reads and writes files: each call writes the land and
# resistance rasters to /dev/shm, creates a new project from them, and reads
# the patches and EC back.
import math
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import landscape
import numpy as np
import rasterio
from adapters.protocol import read_cell, time_calls, write_result
from rasterio.transform import from_origin


HERE = Path(__file__).resolve().parents[1]
JAR = next((HERE / "java").glob("graphab-*.jar"))  # the jar of install.sh
KERNEL = ["d=25", f"p={math.exp(-1)}"]

cell, arrays = read_cell()
resistance = landscape.resistance(arrays["permeability"])
barrier = np.isnan(resistance)
land = np.where(barrier, -1, np.where(arrays["quality"] > 0, 1, 2)).astype("int32")
rows, columns = land.shape
profile = dict(
    driver="GTiff",
    height=rows,
    width=columns,
    count=1,
    crs="EPSG:32633",
    transform=from_origin(500_000, rows, 1, 1),
)
cost = str(cell["windows"]["dependency_range"])
work = Path(cell["output"]).parent / "graphab"
java = [
    str(HERE / "java" / "jre" / "bin" / "java"),
    "-Djava.awt.headless=true",
    f"-Djava.util.prefs.userRoot={work.parent / 'java-preferences'}",
    "-XX:-UsePerfData",
    "-jar",
    str(JAR),
    "-proc",
    str(len(os.sched_getaffinity(0))),
]


def graphab(*arguments):
    subprocess.run([*java, *arguments], check=True, stdout=subprocess.DEVNULL)


def call():
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    with rasterio.open(
        work / "land.tif", "w", dtype="int32", nodata=-1, **profile
    ) as file:
        file.write(land, 1)
    with rasterio.open(
        work / "resistance.tif", "w", dtype="float32", **profile
    ) as file:
        file.write(np.where(barrier, 1, resistance).astype("float32"), 1)
    graphab("--create", "project", str(work / "land.tif"), "nodata=-1", f"dir={work}")
    graphab(
        "--project",
        str(work / "project" / "project.xml"),
        *("--habitat", "name=habitat", "codes=1"),
        *("--linkset", "distance=cost", "name=links", "topo=complete"),
        *(f"maxcost={cost}", f"extcost={work / 'resistance.tif'}"),
        *("--graph", "name=graph", f"threshold={cost}"),
        *("--gmetric", "EC", *KERNEL),
        *("--lmetric", "F", *KERNEL, "beta=1"),
    )
    patches = work / "project" / "habitat"
    with rasterio.open(patches / "patches.tif") as file:
        ids = file.read(1)[1:-1, 1:-1]  # a border of one pixel pads the patches
    with closing(sqlite3.connect(patches / "patches.gpkg")) as database:
        flux = next(
            column
            for _, column, *_ in database.execute("PRAGMA table_info(patches)")
            if column.startswith("f_")
        )
        values = dict(database.execute(f'SELECT Id, "{flux}" + capacity FROM patches'))
    output = np.full(ids.shape, np.nan)
    habitat = ids > 0
    output[habitat] = [values[int(i)] for i in ids[habitat]]
    # EC.txt holds a header line and a line of values, whose last is EC.
    equivalent_connectivity = float(
        (work / "project" / "EC.txt").read_text().splitlines()[1].split("\t")[-1]
    )
    return output, equivalent_connectivity**2


(output, total), warmup, times = time_calls(cell, call)
write_result(cell, output, warmup, times, files=True, total=total)
