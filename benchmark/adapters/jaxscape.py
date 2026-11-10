# JAXScape builds a `GridGraph` of the permeability raster with rook
# contiguity, the conductance of an edge being the harmonic mean of the
# permeabilities of its two pixels (`landscape.conductance`), so that its
# resistance is their mean resistance. Least-cost and free-energy distances
# take 1 / conductance as the edge cost, and the free energy takes the shared
# `theta`. E1 is a nodes-to-nodes query among the sites. E2 runs
# `SensitivityAnalysis(var="quality")` for the map and `ConnectivityAnalysis`
# for the total, and E3 `SensitivityAnalysis(var="permeability")`, with the
# shared `dependency_range` and `coarsening_factor`, the default proximity and
# the `batch_size` of the settings; these windowed analyses build their graphs
# with JAXScape's own edge rule, the mean permeability of the two pixels. The
# distance runs with the solver, its `rtol`, the resistance `method`, its
# `epsilon`, and the dtype of its settings, and every call builds the graphs
# and the solver states.
import jax
from adapters.protocol import read_cell, time_calls, write_result


cell, arrays = read_cell()
settings = cell["settings"]
if settings["dtype"] == "float64":
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxscape
from jaxscape import (
    ConnectivityAnalysis,
    FreeEnergyDistance,
    GridGraph,
    LCPDistance,
    ResistanceDistance,
    SensitivityAnalysis,
    solvers,
)
from landscape import conductance


def options(*names):
    return {name: settings[name] for name in names if name in settings}


solver = (
    {"solver": getattr(solvers, settings["solver"])(**options("rtol"))}
    if "solver" in settings
    else {}
)
method = (
    {"method": getattr(jaxscape, settings["method"])(**options("epsilon"))}
    if "method" in settings
    else {}
)
distance = {
    "resistance": lambda: ResistanceDistance(**solver, **method),
    "lcp": lambda: LCPDistance(),
    "free_energy": lambda: FreeEnergyDistance(
        theta=cell["parameters"]["theta"], **solver
    ),
}[cell["distance"]]()
permeability = jnp.asarray(arrays["permeability"], dtype=settings["dtype"])
windows = dict(
    quality_raster=jnp.asarray(arrays["quality"], dtype=settings["dtype"]),
    permeability_raster=permeability,
    distance=distance,
    batch_size=settings.get("batch_size"),
    **cell["windows"],
)


def isolation():
    graph = GridGraph(permeability, fun=conductance)
    return distance(graph, nodes=jnp.asarray(arrays["sites"])).block_until_ready()


def connectivity():
    total = ConnectivityAnalysis(**windows).run()
    habitat = SensitivityAnalysis(**windows).run("quality")
    return jax.block_until_ready((habitat, total))[0]


def corridors():
    return SensitivityAnalysis(**windows).run("permeability").block_until_ready()


call = {"E1": isolation, "E2": connectivity, "E3": corridors}[cell["question"]]
write_result(cell, *time_calls(cell, call), version=jaxscape.__version__)
