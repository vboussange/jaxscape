from collections.abc import Callable
from typing import Union

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from jax.scipy.linalg import inv

from jaxscape.distance import AbstractDistance
from jaxscape.graph import AbstractGraph
from jaxscape.utils import mapnz


# here the _cost function poses a problem
# you should debug this with https://docs.kidger.site/equinox/all-of-equinox/
# another option would be to NOT subclass GridGraph as eqx.Module


class RSPDistance(AbstractDistance):
    """
    Randomized shortest path distance. Requires the temperature parameter
    `theta` and `cost`, which can be either a `jax.experimental.sparse.BCOO`
    matrix or a function that will be used to map all non zero element of
    the adjacency matrix to create the cost matrix. `cost` defaults to the
    well adapted movement cost function `lambda x: -jnp.log(x))`.

    !!! warning
        This distance metric is experimental and may change in future releases.

    !!! example
        ```python
        from jaxscape import RSPDistance

        distance = RSPDistance(theta=0.01, cost=lambda x: -jnp.log(x))
        dist = distance(grid)
        ```
    """

    theta: Array
    _cost: Union[Callable[[Array], Array], BCOO]

    def __init__(
        self,
        theta: Union[float, Array],
        cost: Union[Callable[[Array], Array], BCOO] = lambda x: -jnp.log(x),
    ):
        self._cost = cost
        self.theta = theta

    def cost_matrix(self, graph: AbstractGraph) -> BCOO:
        if callable(self._cost):
            return mapnz(graph.get_adjacency_matrix(), self._cost)
        else:
            return self._cost

    @eqx.filter_jit
    def nodes_to_nodes_distance(self, graph: AbstractGraph, nodes: Array) -> Array:
        distances = self.all_pairs_distance(graph)
        return distances[nodes[:, None], nodes[None, :]]

    @eqx.filter_jit
    def sources_to_targets_distance(
        self, graph: AbstractGraph, sources: Array, targets: Array
    ) -> Array:
        distances = self.all_pairs_distance(graph)
        return distances[sources[:, None], targets[None, :]]

    @eqx.filter_jit
    def all_pairs_distance(self, graph: AbstractGraph) -> Array:
        A = graph.get_adjacency_matrix()
        C = self.cost_matrix(graph)
        return rsp_distance(self.theta, A, C)


def fundamental_matrix(W: BCOO) -> Array:
    # normalised graph laplacian
    L = sparse.eye(W.shape[0], dtype=W.dtype, index_dtype=W.indices.dtype) - W
    return inv(L.todense())


@eqx.filter_jit
def rsp_distance(theta: Union[float, Array], A: BCOO, C: BCOO) -> Array:
    row_sum = A.sum(1).todense()[:, None]
    Prw = A / row_sum  # random walk probability
    W = Prw * jnp.exp(-theta * C.todense())  # TODO: to optimze
    Z = fundamental_matrix(W)

    C_weighted = C * W
    C̄ = Z @ (C_weighted @ Z)
    C̄ = jnp.where(Z != 0, C̄ / Z, jnp.inf)  # Set to infinity where Z is zero

    diag_C̄ = jnp.diag(C̄)
    C̄ = C̄ - diag_C̄[None, :]

    return C̄
