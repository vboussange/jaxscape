import jax
import jax.numpy as jnp
from jaxscape.rsp_distance import RSPDistance
from jaxscape.resistance_distance import ResistanceDistance
from jaxscape.euclidean_distance import EuclideanDistance
from jaxscape.gridgraph import GridGraph, ExplicitGridGraph
import matplotlib.pyplot as plt


# Define a habitat suitability raster
habitat_suitability = jnp.array(
    [
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 2, 1],
        [0, 0, 0, 0, 0, 1, 1, 1],
    ],
    dtype="float32",
)
plt.imshow(habitat_suitability)
plt.show()
activities = habitat_suitability > 0

# `equivalent_connected_habitat` calculation
# We first need to calculate a distance, 
# that we transform into an ecological proximity
def calculate_ech(habitat_quality):
    grid = GridGraph(activities=activities, vertex_weights=habitat_quality)
    dist = distance(grid)
    # scaling
    dist = dist / dist.max()
    proximity = jnp.exp(-dist / D)
    landscape = ExplicitGridGraph(activities=activities, 
                                  vertex_weights=habitat_quality, 
                                  adjacency_matrix=proximity)
    ech = landscape.equivalent_connected_habitat()
    return ech
grad_ech = jax.grad(calculate_ech)

# derivative of w.r.t pixel habitat suitability 
# represents pixel contribution to landscape connectivity
D = 0.1  # dispersal distance
distance = RSPDistance(theta = jnp.array(1.))
sensitivities = grad_ech(habitat_suitability)
plt.imshow(sensitivities)
plt.show()

distance = ResistanceDistance()
sensitivities = grad_ech(habitat_suitability)
plt.imshow(sensitivities)
plt.show()


distance = EuclideanDistance(res=1)
sensitivities = grad_ech(habitat_suitability)
plt.imshow(sensitivities)
plt.show()


# real example
# habitat_suitability_raster_path = "data/small_extent_habitat_suitability.csv"
# habitat_suitability = jnp.array(np.loadtxt(habitat_suitability_raster_path, delimiter=","))
# plt.imshow(habitat_suitability)
# plt.show()

# # Filter valid activities based on connected components
# from jaxscape.utils import BCOO_to_sparse, get_largest_component_label
# from scipy.sparse.csgraph import connected_components
# def get_valid_activities(hab_qual, activities):
#     grid = GridGraph(activities, hab_qual)
#     A = grid.get_adjacency_matrix()
#     Anp = BCOO_to_sparse(A)
#     _, labels = connected_components(Anp, directed=True, connection="strong")
#     label = get_largest_component_label(labels)
#     vertex_belongs_to_largest_component_node = labels == label
#     return grid.node_values_to_array(vertex_belongs_to_largest_component_node) == True

# valid_activities = get_valid_activities(habitat_suitability, activities)
# habitat_suitability = habitat_suitability * valid_activities
# np.savetxt(habitat_suitability_raster_path, habitat_suitability, delimiter=",")
