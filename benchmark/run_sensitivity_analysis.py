import jax
import jax.numpy as jnp
from jax import jit, grad
from jax.experimental.sparse import BCOO

import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

from scipy.sparse.csgraph import connected_components

from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
from jaxscape.utils import BCOO_to_sparse, get_largest_component_label
from jaxscape.moving_window import WindowOperation
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.rsp_distance import RSPDistance

import matplotlib.pyplot as plt


# Calculate the latitudinal and longitudinal resolution of a raster
def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution       

# Filter valid activities based on connected components
def get_valid_activities(hab_qual, activities):
    grid = GridGraph(activities, hab_qual)
    A = grid.get_adjacency_matrix()
    Anp = BCOO_to_sparse(A)
    _, labels = connected_components(Anp, directed=True, connection="strong")
    label = get_largest_component_label(labels)
    vertex_belongs_to_largest_component_node = labels == label
    return grid.node_values_to_array(vertex_belongs_to_largest_component_node) == True

# Perform sensitivity analysis across multiple windows
def run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance, cut_off):
    sensitivity_raster = jnp.full_like(habitat_quality_raster, jnp.nan)
    for x_start, y_start, hab_qual in tqdm(window_op.iterate_windows(habitat_quality_raster), total=window_op.nb_steps, desc="Running sensitivity analysis"):
        activities = hab_qual > 0
        valid_activities = get_valid_activities(hab_qual, activities)
        nb_active = int(valid_activities.sum())
        
        def compute_connectivity_in_window(hab_qual):
            gridgraph = GridGraph(activities=valid_activities, 
                                vertex_weights=hab_qual,
                                nb_active = nb_active)
            dist = distance(gridgraph)
            proximity = jnp.exp(-dist / D) > cut_off # we would ideally convert it to a BCOO matrix, but it is not jit compatible
            landscape = ExplicitGridGraph(vertex_weights = hab_qual, 
                                          adjacency_matrix= proximity, 
                                          activities = valid_activities, 
                                          nb_active = nb_active)
            return landscape.equivalent_connected_habitat()

        # Compute the gradient of connectivity with respect to habitat quality
        grad_connectivity_in_window = jit(grad(compute_connectivity_in_window))
                                        
        
        # Call the gradient function on the window's habitat quality
        sensitivity_raster_window = grad_connectivity_in_window(hab_qual)
        sensitivity_raster = window_op.update_raster_from_window(x_start, y_start, sensitivity_raster, sensitivity_raster_window)
    
    return sensitivity_raster

# Load habitat suitability data from a NetCDF file
def load_habitat_suitability(sp_name, path_ncfile=Path("data/large_extent_habitat_suitability.nc")):
    with xr.open_dataset(path_ncfile, engine="netcdf4", decode_coords="all") as da: 
        habitat_suitability = da[sp_name] / 100
    res = calculate_resolution(da)
    return jnp.array(habitat_suitability.data[0,:,:]), res

# Example usage
if __name__ == "__main__":
    sp_name = "Salmo trutta"
    D_km = jnp.array(4.0)
    
    params_computation = {"window_size": 20, "cut_off": 0.1}
    habitat_quality_raster, res = load_habitat_suitability(sp_name)
    plt.imshow(jnp.where(~jnp.isnan(habitat_quality_raster), habitat_quality_raster, 0))
    plt.show()
    
    D = D_km * 1000 # converting into meter, unit of res
    assert np.isclose(res[0], res[1], atol=1.0), "Cannot handle different x, y resolution"

    window_op = WindowOperation(shape=habitat_quality_raster.shape, 
                                window_size=params_computation["window_size"], 
                                buffer_size=int(3 * D_km * 1e3/ res[0]))
    
    distance = EuclideanDistance(res=1.0)
    sensitivity_raster = run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance, params_computation["cut_off"])

    plt.imshow(sensitivity_raster)
    plt.show()
    
    # theta = jnp.array(0.01)
    # alpha = jnp.array(21.0) 
    # D = D_km / alpha # need to convert D in ecological distance
    # distance = RSPDistance(theta=theta)
    # sensitivity_raster = run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance, params_computation["cut_off"])
