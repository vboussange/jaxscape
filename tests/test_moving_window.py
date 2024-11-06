import jax.numpy as jnp
from jax import grad
from math import ceil
import matplotlib.pyplot as plt
from connectax.moving_window import WindowOperation, run_analysis
import xarray as xr
from pathlib import Path
from connectax.euclidean_distance import EuclideanDistance
from connectax.rsp_distance import RSPDistance

# Example usage
if __name__ == "__main__":
    # Load dataset and set up parameters
    D_km = 1.0
    D = D_km * 20 # TODO: to be changed
    sp_name = "Salmo trutta"
    
    path = Path("data/large_extent_habitat_suitability.nc")
    with xr.open_dataset(path, engine="netcdf4", decode_coords="all") as da: 
        habitat_suitability = da[sp_name] / 100
        da.close()
    jax_raster = jnp.array(habitat_suitability.data[0,:,:])
    jax_raster = jnp.nan_to_num(jax_raster, nan=0.)

    window_op = WindowOperation(raster_data = jax_raster, 
                                window_size = 40, 
                                buffer_size = int(3 * D_km))
    
    
    distance = EuclideanDistance(res=1.)
    output_array = run_analysis(window_op, D, distance)
    
    theta = jnp.array(0.01)
    distance = RSPDistance(theta)
    output_array = run_analysis(window_op, D, distance)
    # output_array = run_analysis(window_op, D, RSPDistance.rsp_distance, theta=theta)
