# skimage.graph.MCP_Flexible finds least-cost paths on the permeability raster
# with rook contiguity (`fully_connected=False`). Its `travel_cost`, which the
# class lets subclasses override, charges 1 / conductance for a step between
# two pixels, with the conductance that `graph` in the configuration sets; the
# cost that MCP counts for the start pixel is subtracted. Each call builds the
# MCP object on the permeability, and searches from each site until it reaches
# all the others.
import numpy as np
import skimage
from adapters.protocol import conductance, read_cell, time_calls, write_result
from skimage.graph import MCP_Flexible


cell, arrays = read_cell()
if (cell["question"], cell["distance"]) != ("E1", "lcp"):
    raise ValueError("skimage.graph.MCP computes the least-cost distances of E1")
permeability = arrays["permeability"]
sites = [tuple(map(int, site)) for site in arrays["sites"]]
edge = conductance(cell)


class LeastCost(MCP_Flexible):
    def travel_cost(self, old_cost, new_cost, offset_length):
        return 1 / edge(old_cost, new_cost)


def call():
    search = LeastCost(permeability, fully_connected=False)
    distances = np.empty((len(sites), len(sites)))
    for row, site in enumerate(sites):
        costs, _ = search.find_costs([site], ends=sites)
        distances[row] = [costs[end] - costs[site] for end in sites]
    return distances


write_result(cell, *time_calls(cell, call), version=skimage.__version__)
