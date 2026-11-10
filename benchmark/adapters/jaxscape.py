# JAXScape builds a `GridGraph` of the permeability raster with rook
# contiguity, the conductance of an edge being the mean permeability of its two
# pixels, as its windowed analyses do. Least-cost and free-energy distances
# take 1 / conductance as the edge cost, and the free energy takes the shared
# `theta`. E1 is a nodes-to-nodes query among the sites. The distance runs with
# the solver and dtype of its settings, and every call builds the graph and the
# solver state.
import jax
from adapters.protocol import read_cell, time_calls, write_result


cell, arrays = read_cell()
settings = cell["settings"]
if settings["dtype"] == "float64":
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxscape
from jaxscape import (
    FreeEnergyDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    solvers,
)


def average(x, y):
    return (x + y) / 2


solver = (
    {"solver": getattr(solvers, settings["solver"])()} if "solver" in settings else {}
)
distance = {
    "resistance": lambda: ResistanceDistance(**solver),
    "lcp": lambda: LCPDistance(),
    "free_energy": lambda: FreeEnergyDistance(
        theta=cell["parameters"]["theta"], **solver
    ),
}[cell["distance"]]()

if cell["question"] != "E1":
    raise ValueError(f"the JAXScape adapter has no question {cell['question']}")
permeability = jnp.asarray(arrays["permeability"], dtype=settings["dtype"])
sites = jnp.asarray(arrays["sites"])


def call():
    graph = GridGraph(permeability, fun=average)
    return distance(graph, nodes=sites).block_until_ready()


write_result(cell, *time_calls(cell, call), version=jaxscape.__version__)
