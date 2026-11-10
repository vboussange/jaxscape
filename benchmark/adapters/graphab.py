# Graphab builds a graph of habitat patches, the clusters of habitat pixels
# (quality > 0) connected through 4 neighbours. Its links are least-cost paths
# between patches over 8 neighbours, in which a step costs its length times
# the mean resistance, 1 / permeability, of its two pixels (`extcost`), and
# barriers, of permeability 0, are no data. Links and graph keep the paths of
# cost up to `dependency_range` (`maxcost`, `threshold`, `topo=complete`),
# where the kernel falls to exp(-4). `d=25` and `p=exp(-1)` give the kernel
# exp(-d/25) on the cost of the shortest path in the graph, which adds the
# costs of crossing patches. The capacity a_I of a patch is its area in
# pixels. Each call computes the PC of the graph, the landscape total, and the
# flux F_I = sum over patches J != I of a_J exp(-d_IJ/25) (`beta=1`). The map
# is F_I + a_I, the patch analogue of Hanski's S_i, which adds the patch's own
# term that F leaves out, on the pixels of each patch, and NaN elsewhere.
# `-proc` is the number of cores of the cell. Graphab reads and writes files:
# each call writes the land and resistance rasters to /dev/shm, creates a new
# project from them, and reads the patches back.
import math
import os
import shutil
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path

import numpy as np
import rasterio
from adapters.protocol import read_cell, time_calls, write_result
from rasterio.transform import from_origin


HERE = Path(__file__).resolve().parents[1]
JAR = next((HERE / "java").glob("graphab-*.jar"))  # the jar of install.sh
KERNEL = ["d=25", f"p={math.exp(-1)}"]

cell, arrays = read_cell()
if cell["question"] != "E2":
    raise ValueError("Graphab computes the habitat connectivity of E2")
quality, permeability = arrays["quality"], arrays["permeability"]
land = np.where(permeability == 0, -1, np.where(quality > 0, 1, 2)).astype("int32")
resistance = (1 / np.where(permeability == 0, 1, permeability)).astype("float32")
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
        file.write(resistance, 1)
    graphab("--create", "project", str(work / "land.tif"), "nodata=-1", f"dir={work}")
    graphab(
        "--project",
        str(work / "project" / "project.xml"),
        *("--habitat", "name=habitat", "codes=1"),
        *("--linkset", "distance=cost", "name=links", "topo=complete"),
        *(f"maxcost={cost}", f"extcost={work / 'resistance.tif'}"),
        *("--graph", "name=graph", f"threshold={cost}"),
        *("--gmetric", "PC", *KERNEL),
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
    return output


version = JAR.stem.removeprefix("graphab-")
write_result(cell, *time_calls(cell, call), version=version, files=True)
