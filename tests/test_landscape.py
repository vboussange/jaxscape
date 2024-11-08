import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax import random
import jax.random as jr
from jaxscape.gridgraph import GridGraph
from jaxscape.rastergraph import Landscape

def test_Landscape():
    permeability_raster = jnp.ones((10, 10)) 
    activities = jnp.ones(permeability_raster.shape, dtype=bool)

    grid = GridGraph(activities=activities, 
                    vertex_weights=permeability_raster)
    A = grid.get_adjacency_matrix()
    permeability_raster = permeability_raster.at[1,1].set(0.)
    land = Landscape(habitat_quality=permeability_raster, proximity=A, activities=grid.activities)
    func = land.equivalent_connected_habitat()
    assert func == (A.sum()/2 - 4) # we lose 1 vertex connected to 4 neighbors, as its quality is set 0.
