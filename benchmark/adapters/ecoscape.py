# EcoScape (`ecoscape-connectivity`) simulates the repopulation of the habitat,
# the pixels of quality > 0, from seeds drawn at random in it, and maps its
# flow: the gradient of the repopulation with respect to the permeability of
# each pixel, times that permeability. Its edge rule is its own: at each step,
# individuals spread to the 8 neighbours of a pixel, survive a coin toss, and
# are multiplied by the permeability of their destination. The dispersal
# distance is the shared `dependency_range`, in pixels, which is also the
# border of its tiles and their padding, so that its map covers the raster.
# Quality enters only as habitat. The other settings are EcoScape's defaults:
# 400 simulations, 4 seeds per square of side 2 `dependency_range` + 1, tiles
# of 1000 pixels and no gap crossing, with the `batch_size` of the settings and
# a fixed random seed. EcoScape tiles a raster through GeoTIFFs only: every
# call writes the habitat and permeability to /dev/shm, runs on the GPU, and
# reads back the flow, which EcoScape writes with its connectivity. Its map is
# compared with JAXScape's under least-cost path distances: repopulation
# follows the most permeable neighbour, the maximum over neighbours, as a
# least-cost path does.
from pathlib import Path

import numpy as np
import rasterio
from adapters.protocol import read_cell, time_calls, write_result
from ecoscape_connectivity import compute_connectivity


cell, arrays = read_cell()
dispersal = cell["windows"]["dependency_range"]
work = Path(cell["output"]).parent
habitat, permeability, connectivity, flow = (
    work / f"{name}.tif" for name in ("habitat", "permeability", "connectivity", "flow")
)


def write(path, raster):
    profile = dict(driver="GTiff", count=1, dtype="float32", crs="EPSG:3857")
    with rasterio.open(
        path, "w", height=raster.shape[0], width=raster.shape[1], **profile
    ) as file:
        file.write(raster.astype(np.float32), 1)


def call():
    write(habitat, arrays["quality"] > 0)
    write(permeability, arrays["permeability"])
    compute_connectivity(
        habitat_fn=str(habitat),
        permeability_fn=str(permeability),
        connectivity_fn=str(connectivity),
        flow_fn=str(flow),
        dispersal=dispersal,
        border_size=dispersal,
        padding_size=dispersal,
        batch_size=cell["settings"]["batch_size"],
        random_seed=1,
        device="cuda" if cell["device"] == "gpu" else "cpu",
    )
    with rasterio.open(flow) as file:
        decibels = file.read(1)
    return 10 ** (decibels / 20) - 1  # EcoScape writes 20 log10(1 + flow)


write_result(cell, *time_calls(cell, call), files=True)
