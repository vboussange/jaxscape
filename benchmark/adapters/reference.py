# The float64 reference of E1, against which the runner measures every E1
# record: the distances among the sites on the graph of `adapters/jaxscape.py`,
# in float64 on the CPU. Resistance and free-energy distances solve with
# `CholmodSolver`, a sparse direct solver; least-cost distances come from
# Dijkstra's algorithm in SciPy, with the edge cost 1 / conductance.
import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jaxscape
import numpy as np
from adapters.protocol import read_cell, time_calls, write_result
from jaxscape import FreeEnergyDistance, GridGraph, ResistanceDistance
from jaxscape.solvers import CholmodSolver
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


cell, arrays = read_cell()
permeability = jnp.asarray(arrays["permeability"])
sites = arrays["sites"]


def average(x, y):
    return (x + y) / 2


def least_cost():
    adjacency = GridGraph(permeability, fun=average).get_adjacency_matrix()
    rows, columns = np.asarray(adjacency.indices).T
    conductance = np.asarray(adjacency.data)
    edges = conductance > 0
    cost = coo_matrix(
        (1 / conductance[edges], (rows[edges], columns[edges])), adjacency.shape
    )
    vertices = np.ravel_multi_index(tuple(sites.T), permeability.shape)
    return dijkstra(cost.tocsr(), indices=vertices)[:, vertices]


def solve(distance):
    return lambda: np.asarray(
        distance()(GridGraph(permeability, fun=average), nodes=sites)
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

write_result(cell, *time_calls(cell, call), version=jaxscape.__version__)
