# The float64 reference of the distance calculation, with respect to which the
# runner measures the error of every record of the distance calculation: the
# distances among the sites on the graph of `adapters/jaxscape.py`, in float64
# on the CPU. The linear systems of resistance and free-energy distances are
# solved with `CholmodSolver`, a sparse direct solver; least-cost path
# distances are computed with Dijkstra's algorithm in SciPy, with the edge cost
# 1 / conductance.
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from adapters.protocol import read_cell, time_calls, write_result
from jaxscape import FreeEnergyDistance, GridGraph, ResistanceDistance
from jaxscape.solvers import CholmodSolver
from landscape import conductance
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


cell, arrays = read_cell()
permeability = jnp.asarray(arrays["permeability"])
sites = arrays["sites"]


def least_cost():
    adjacency = GridGraph(permeability, fun=conductance).get_adjacency_matrix()
    rows, columns = np.asarray(adjacency.indices).T
    weights = np.asarray(adjacency.data)
    edges = weights > 0
    cost = coo_matrix(
        (1 / weights[edges], (rows[edges], columns[edges])), adjacency.shape
    )
    vertices = np.ravel_multi_index(tuple(sites.T), permeability.shape)
    return dijkstra(cost.tocsr(), indices=vertices)[:, vertices]


def solve(distance):
    return lambda: np.asarray(
        distance()(GridGraph(permeability, fun=conductance), nodes=sites)
    )


call = {
    "resistance": solve(lambda: ResistanceDistance(solver=CholmodSolver())),
    "lcp": least_cost,
    "free_energy": solve(
        lambda: FreeEnergyDistance(
            theta=cell["parameters"]["theta"], solver=CholmodSolver()
        )
    ),
}[cell["distance"]]

write_result(cell, *time_calls(cell, call))
