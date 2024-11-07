import jax
import jaxscape


D = 1. # dispersal distance
theta = jnp.array(0.01)
distance = jaxscape.RSPDistance(theta)

def calculate_ech(habitat_quality):
    grid = jaxscape.GridGraph(activities=activities, vertex_weights=habitat_quality)
    dist = distance(grid)
    proximity = jnp.exp(-dist / D)
    landscape = jaxscape.Landscape(habitat_quality, proximity)
    ech = landscape.equivalent_connected_habitat()
    return ech
    
grad_ech = grad(calculate_ech)
sensitivities = grad_ech(habitat_quality)

plt.imshow(sensitivities)