from jaxscape.gridgraph import GridGraph
import jax.numpy as jnp
from jax import jit
import equinox as eqx


class RasterGraph(eqx.Module):
    gridgraph: GridGraph
    x_coords: jnp.ndarray
    y_coords: jnp.ndarray
        
    def __init__(self, gridgraph, x_coords, y_coords):
        """
        An ExplicitGridGraph with `lat` `lon` accessors.
        """
        # assert int(activities.sum()) == proximity.shape[0], "The number of nodes in the graph defined by `proximity` should correspond to the number of active vertices defined in `activities`"
        assert len(x_coords) == gridgraph.width
        assert len(y_coords) == gridgraph.height

        self.gridgraph = gridgraph
        self.x_coords = x_coords  # Longitude values
        self.y_coords = y_coords  # Latitude values

    def index(self, lon, lat):
        # Find the nearest index in x (longitude) and y (latitude) coordinates
        if not jnp.all((self.x_coords.min() <= lon) & (lon <= self.x_coords.max())):
            raise ValueError(f"Longitude {lon} is out of bounds ({self.x_coords.min()}, {self.x_coords.max()}).")

        # Find the nearest index in x (longitude) and y (latitude) coordinates
        if not jnp.all((self.y_coords.min() <= lat) & (lat <= self.y_coords.max())):
            raise ValueError(f"Latitude {lat} is out of bounds ({self.y_coords.min()}, {self.y_coords.max()}).")

        i = jnp.abs(self.x_coords.reshape(-1, 1) - lon).argmin(axis=0)  # Find closest longitude index
        j = jnp.abs(self.y_coords.reshape(-1, 1) - lat).argmin(axis=0)  # Find closest latitude index
        return i, j
    
    # TODO: not sure we need those, but we keep it just in case
    def iloc(self, i, j):
        """
        Index-based access to the data.
        """
        return self.gridgraph.vertex_weights[i, j]

    def loc(self, lon, lat):
        """
        Coordinate-based access to the data.
        """
        i, j = self.index(lon, lat)
        return self.iloc(i, j)