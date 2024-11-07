import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np  # for NaN handling, not used in heavy computations
import matplotlib.pyplot as plt
from connectax.gridgraph import GridGraph
from connectax.utils import BCOO_to_sparse, get_largest_component_label
from connectax.landscape import Landscape
import jax
from jax.experimental.sparse import BCOO
from tqdm import tqdm
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix      
import jax.numpy as jnp
from jax import grad
from math import ceil
import matplotlib.pyplot as plt
from connectax.moving_window import WindowOperation
import xarray as xr
from pathlib import Path
from connectax.euclidean_distance import EuclideanDistance
from connectax.rsp_distance import RSPDistance

def calculate_resolution(raster):
    lat_resolution = abs(raster.y.diff(dim='y').mean().values)
    lon_resolution = abs(raster.x.diff(dim='x').mean().values)
    return lat_resolution, lon_resolution       

def get_valid_activities(hab_qual, activities):
    # TODO: the best would be to avoid transfer between numpy and jax array
    grid = GridGraph(activities, hab_qual)
    A = grid.get_adjacency_matrix()
    Anp = BCOO_to_sparse(A)
    _, labels = connected_components(Anp, directed=True, connection="strong")
    label = get_largest_component_label(labels)
    vertex_belongs_to_largest_component_node = labels == label
    activities_pruned = grid.node_values_to_raster(vertex_belongs_to_largest_component_node)
    activities_pruned = activities_pruned == True
    return activities_pruned

def run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance, cut_off):
    """Performs the sensitivity analysis on each valid window.
    `D` must be expressed in the unit of habitat quality in `window_op`.
    """
    sensitivity_raster = jnp.full_like(habitat_quality_raster, jnp.nan)
    for x_start, y_start, hab_qual in tqdm(window_op.iterate_windows(habitat_quality_raster), total=window_op.nb_steps, desc="Running Analysis"):
        # Build grid graph and calculate Euclidean distances
        activities = hab_qual > 0
        valid_activities = get_valid_activities(hab_qual, activities) # this cannot be jit compiled, because it calls `connected_components` from scipy

        # TODO: we should jit the whole block below instead of jitting at each iteration
        def connectivity(hab_qual):
            # TODO: need to iterate through the connected components
            # for now, we only take the largest component, but we could build a loop here
            gridgraph = GridGraph(activities=valid_activities, 
                                    vertex_weights=hab_qual)
            dist = distance(gridgraph)
            proximity = BCOO.fromdense(jnp.exp(-dist / D) > cut_off)
            landscape = Landscape(hab_qual, proximity, valid_activities)
            func = landscape.functional_habitat()
            return func
    
        grad_connectivity = grad(connectivity)
        sensitivity_raster_window = grad_connectivity(hab_qual)

        sensitivity_raster = window_op.update_raster_from_window(x_start, y_start, sensitivity_raster, sensitivity_raster_window)

    return sensitivity_raster

def load_habitat_suitability(sp_name, path_ncfile = Path("data/large_extent_habitat_suitability.nc")):
    with xr.open_dataset(path_ncfile, engine="netcdf4", decode_coords="all") as da: 
        habitat_suitability = da[sp_name] / 100
        da.close()
    res = calculate_resolution(da)
    jax_raster = jnp.array(habitat_suitability.data[0,:,:])
    # jax_raster = jnp.nan_to_num(jax_raster, nan=0.)
    return jax_raster, res

# Example usage
if __name__ == "__main__":

    sp_name = "Salmo trutta"
    D_km = jnp.array(1.)
    
    
    params_computation = {"window_size": 40, 
                          "cut_off":  0.1}
    alpha = jnp.array(21.)
    habitat_quality_raster, res = load_habitat_suitability(sp_name)
    D = D_km / alpha
    assert np.isclose(res[0], res[1], atol=1.), "Cannot handle different x, y resolution"

    window_op = WindowOperation(shape = habitat_quality_raster.shape, 
                                window_size = params_computation["window_size"], 
                                buffer_size = int(3 * D_km / res[0]))
    
    distance = EuclideanDistance(res=1.)
    sensitivity_raster = run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance, params_computation["cut_off"])
    # import maptlotlib.pyplot as plt
    # plt.imshow(sensitivity_raster)
    
    # theta = jnp.array(0.01)
    # distance = RSPDistance(theta)
    # output_array = run_sensitivity_analysis(habitat_quality_raster, window_op, D, distance, params_computation["cut_off"])
    # output_array = run_analysis(window_op, D, RSPDistance.rsp_distance, theta=theta)
