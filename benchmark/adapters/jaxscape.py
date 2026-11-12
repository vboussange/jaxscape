# JAXScape builds a `GridGraph` of the permeability raster with rook
# contiguity, the conductance of an edge being the harmonic mean of the
# permeabilities of its two pixels (`landscape.conductance`), so that its
# resistance is their mean resistance. Least-cost and free-energy distances
# take 1 / conductance as the edge cost, and the free energy takes the shared
# `theta`. E1 is a nodes-to-nodes query among the sites. E2 runs
# `SensitivityAnalysis(var="quality")` for the map and `ConnectivityAnalysis`
# for the landscape total, sum_i q_i S_i, and E3
# `SensitivityAnalysis(var="permeability")`, with the shared `dependency_range`
# and `coarsening_factor`, the default proximity and the `batch_size` of the
# settings; these windowed analyses build their graphs with JAXScape's own edge
# rule, the mean permeability of the two pixels. A `proximity` setting, a
# function of the distance in Python such as `lambda d: -d`, replaces the
# default proximity. With `normalized`, the map of E3 is the normalized
# sensitivity to resistance of the connectivity example: p^2 dC/dp divided by
# dC/dp on a flat landscape of permeability 1 and the same quality, and 0 where
# the latter is not positive, both computed in each call. The distance runs
# with the solver, the resistance `method` and the dtype of its settings, such as
# `GridMultigridSolver(rtol=3e-4)` (`dense` is the dense solve,
# `solver=None`), and every call builds the graphs and the solver states.
import ast

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


def construct(setting, module):
    """Return the object of `module` that a setting names.

    A setting such as `GridMultigridSolver(rtol=3e-4)` calls the class with its
    keywords, and `dense` gives `None`.
    """
    if setting == "dense":
        return None
    call = ast.parse(setting if "(" in setting else f"{setting}()", mode="eval").body
    keywords = {
        keyword.arg: ast.literal_eval(keyword.value) for keyword in call.keywords
    }
    return getattr(module, call.func.id)(**keywords)


solver = (
    {"solver": construct(settings["solver"], solvers)} if "solver" in settings else {}
)
method = (
    {"method": construct(settings["method"], jaxscape)} if "method" in settings else {}
)
distance = {
    "resistance": lambda: ResistanceDistance(**solver, **method),
    "lcp": lambda: LCPDistance(),
    "free_energy": lambda: FreeEnergyDistance(
        theta=cell["parameters"]["theta"], **solver
    ),
}[cell["distance"]]()
permeability = jnp.asarray(arrays["permeability"], dtype=settings["dtype"])
proximity = (
    {"proximity": eval(settings["proximity"])} if "proximity" in settings else {}
)
windows = dict(
    quality_raster=jnp.asarray(arrays["quality"], dtype=settings["dtype"]),
    permeability_raster=permeability,
    distance=distance,
    batch_size=settings.get("batch_size"),
    dependency_range=cell["windows"].get("dependency_range"),
    coarsening_factor=cell["windows"].get("coarsening_factor"),
    **proximity,
)


def isolation():
    graph = GridGraph(permeability, fun=conductance)
    return distance(graph, nodes=jnp.asarray(arrays["sites"])).block_until_ready()


def connectivity():
    total = ConnectivityAnalysis(**windows).run()
    habitat = SensitivityAnalysis(**windows).run("quality")
    return jax.block_until_ready((habitat, total))


def corridors():
    return SensitivityAnalysis(**windows).run("permeability").block_until_ready()


def normalized_corridors():
    # p^2 dC/dp over dC/dp on the flat landscape, as the connectivity example
    # computes it.
    flat = {**windows, "permeability_raster": jnp.ones_like(permeability)}
    sensitivity = SensitivityAnalysis(**windows).run("permeability")
    flat_sensitivity = SensitivityAnalysis(**flat).run("permeability")
    positive = flat_sensitivity > 0
    ratio = permeability**2 * sensitivity / jnp.where(positive, flat_sensitivity, 1)
    return jnp.where(positive, ratio, 0).block_until_ready()


e3 = normalized_corridors if settings.get("normalized") else corridors
call = {"E1": isolation, "E2": connectivity, "E3": e3}[cell["question"]]
output, warmup, times = time_calls(cell, call)
output, total = output if cell["question"] == "E2" else (output, None)
write_result(cell, output, warmup, times, total=total)
