# skimage.graph.MCP_Geometric finds least-cost paths on the resistance raster,
# 1 / permeability, with rook contiguity (`fully_connected=False`). The cost of
# a step is the mean resistance of its two pixels, 1 / `landscape.conductance`,
# the edge cost of JAXScape. Each call builds the MCP object, and searches from
# each site until it reaches all the others.
import numpy as np
import skimage
from adapters.protocol import read_cell, time_calls, write_result
from skimage.graph import MCP_Geometric


cell, arrays = read_cell()
if (cell["question"], cell["distance"]) != ("E1", "lcp"):
    raise ValueError("skimage.graph.MCP computes the least-cost distances of E1")
resistance = 1 / arrays["permeability"]
sites = [tuple(map(int, site)) for site in arrays["sites"]]


def call():
    search = MCP_Geometric(resistance, fully_connected=False)
    distances = np.empty((len(sites), len(sites)))
    for row, site in enumerate(sites):
        costs, _ = search.find_costs([site], ends=sites)
        distances[row] = [costs[end] for end in sites]
    return distances


write_result(cell, *time_calls(cell, call), version=skimage.__version__)
