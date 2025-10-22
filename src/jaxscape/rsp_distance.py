import jax
import jax.numpy as jnp
from jax.scipy.linalg import inv
from jax.experimental import sparse
from jaxscape.distance import AbstractDistance
from jaxscape.utils import mapnz
from typing import Callable, Union
import equinox as eqx
from jax.experimental.sparse import BCOO
# here the _cost function poses a problem
# you should debug this with https://docs.kidger.site/equinox/all-of-equinox/
# another option would be to NOT subclass GridGraph as eqx.Module

class RSPDistance(AbstractDistance):
    theta: jnp.ndarray
    _cost: Union[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray]
    
    def __init__(self, theta, cost=lambda x: -jnp.log(x)):
        """
        Randomized shortest path distance. Requires the temperature parameter
        `theta` and `cost`, which can be either a `jax.experimental.sparse.BCOO`
        matrix or a function that will be used to map all non zero element of
        the adjacency matrix to create the cost matrix. `cost` defaults to the
        well adapted movement cost function `lambda x: -jnp.log(x))`.
        """
        self._cost = cost
        self.theta = theta
        
    def cost_matrix(self, grid):
        if callable(self._cost):
            return mapnz(grid.get_adjacency_matrix(), self._cost)
        else:
            return self._cost
        
    @eqx.filter_jit
    def nodes_to_nodes_distance(self, grid, nodes):
        distances = self.all_pairs_distance(grid)
        return distances[nodes[:, None], nodes[None, :]]

    @eqx.filter_jit
    def sources_to_targets_distance(self, grid, sources, targets):
        distances = self.all_pairs_distance(grid)
        return distances[sources[:, None], targets[None, :]]

    @eqx.filter_jit
    def all_pairs_distance(self, grid):
        A = grid.get_adjacency_matrix()
        C = self.cost_matrix(grid)
        return rsp_distance(self.theta, A, C)
    
def fundamental_matrix(W):
    # normalised graph laplacian
    L = sparse.eye(W.shape[0], dtype=W.dtype, index_dtype=W.indices.dtype) - W
    return inv(L.todense())

@eqx.filter_jit
def rsp_distance(theta: float, A: BCOO, C: BCOO):
    row_sum = A.sum(1).todense()[:, None]
    Prw = A / row_sum  # random walk probability
    W = Prw * jnp.exp(-theta * C.todense()) # TODO: to optimze
    Z = fundamental_matrix(W)

    C_weighted = C * W
    C̄ = Z @ (C_weighted @ Z)
    C̄ = jnp.where(Z != 0, C̄ / Z, jnp.inf)  # Set to infinity where Z is zero

    diag_C̄ = jnp.diag(C̄)
    C̄ = C̄ - diag_C̄[None, :]

    return C̄
