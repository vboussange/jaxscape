import jax
import jax.numpy as jnp
from jax.scipy.linalg import inv
from jax.experimental import sparse
from jaxscape.utils import well_adapted_movement
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
    _cost: Union[Callable[jnp.ndarray, jnp.ndarray], jnp.ndarray]

    def __init__(self, theta, cost=well_adapted_movement):
        """
        Requires `cost`,
        which can be either a `jax.experimental.sparse.BCOO` matrix or a
        function taking an adjacency matrix of type
        `jax.experimental.sparse.BCOO` and returning a
        `jax.experimental.sparse.BCOO` cost matrix. `cost` defaults to
        `well_adapted_movement` function.
        """
        self._cost = cost
        self.theta = theta
        
    def cost_matrix(self, grid):
        if callable(self._cost):
            return self._cost(grid.get_adjacency_matrix())
        else:
            return self._cost
        
    @eqx.filter_jit
    def __call__(self, grid):
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
