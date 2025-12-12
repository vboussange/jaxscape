import jax.numpy as jnp
import matplotlib.pyplot as plt
import networkx as nx
from scipy.ndimage import gaussian_filter
from jaxscape import GridGraph
from jaxscape.rsp_distance import RSPDistance
import jax
from matplotlib.colors import LinearSegmentedColormap
COLORS_BR = ["#f72585","#b5179e","#7209b7","#560bad","#480ca8","#3a0ca3","#3f37c9","#4361ee","#4895ef","#4cc9f0"]
# check https://coolors.co/palettes/popular/gradient
CMAP_BR = LinearSegmentedColormap.from_list("species_richness", COLORS_BR[::-1])

# Generate a spatially autocorrelated raster

# Define a larger habitat suitability raster
habitat_suitability = jnp.array(
    [
        [1, 1, 1, 0, 0, 0, 0,],
        [1, 2, 1, 0, 0, 0, 0,],
        [1, 1, 1, 0, 0, 0, 0,],
        [0, 0, 1, 1, 0, 0, 0,],
        [0, 0, 0, 1, 1, 1, 1,],
        [0, 0, 0, 0, 1, 2, 1,],
        [0, 0, 0, 0, 1, 1, 1,],
    ],
    dtype="float32"
)

raster_size = habitat_suitability.shape

# Create a grid graph
G = nx.grid_2d_graph(raster_size[0], raster_size[1])

# Define positions for nodes at the center of each raster cell
pos = {(x, y): (y , raster_size[0] - x -1) for x, y in G.nodes()}  # Offset for pixel centers

# Define node and edge colors for visualization
node_colors = ['#f72585' if (x + y) % 2 == 0 else '#4cc9f0' for x, y in G.nodes()]
edge_colors = ['#555555' if i % 2 == 0 else '#999999' for i in range(len(G.edges()))]

# Plot the raster with autocorrelation
plt.figure(figsize=(8, 8))
plt.imshow(habitat_suitability, cmap=CMAP_BR, interpolation='nearest')
# plt.colorbar(label='Habitat suitability')
plt.axis('off')
plt.savefig("Habitat_suitability.pdf", transparent=True, dpi=300)
# plt.title("Spatially Autocorrelated Raster with Grid Overlay")

# Overlay the grid graph on the raster
nx.draw_networkx_nodes(G, pos, node_color='#999999', node_size=400, alpha=0.8)
nx.draw_networkx_edges(G, pos, edge_color='#555555', width=1, alpha=0.7)
# nx.draw_networkx_labels(G, pos, font_size=8, font_color='white', font_weight='bold')

# Remove axis for cleaner look
plt.savefig("Habitat_suitability_with_graph.pdf", transparent=True, dpi=300)

D = jnp.array(1.)  # dispersal distance
theta = jnp.array(10.)
distance = RSPDistance(theta)

activities = habitat_suitability > 0

# Equivalent connected habitat calculation.
# We first need to calculate a distance, 
# that we transform into an ecological proximity
def calculate_ech(habitat_quality):
    grid = GridGraph(activities=activities, grid=habitat_quality)
    dist = distance(grid)
    K = jnp.exp(-dist / D)
    q = grid.get_active_vertices_weights()
    ech = jnp.sqrt(q @ (K @ q)) 
    return ech

# derivative of w.r.t pixel habitat suitability 
# represents pixel contribution to landscape connectivity
grad_ech = jax.grad(calculate_ech)
sensitivities = grad_ech(habitat_suitability)
plt.figure(figsize=(8, 8))
plt.imshow(sensitivities)
plt.colorbar(label='Sensitivity')
plt.savefig("sensitivities.png", transparent=True, dpi=300)

# Plot the sensitivities on the grid graph
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(habitat_suitability, cmap='viridis', interpolation='nearest')
# plt.colorbar(label='Habitat suitability')
ax.axis('off')

sensitivity_values = sensitivities[::-1,:].ravel()
node_colors = sensitivity_values
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=100, cmap='plasma', alpha=0.8, ax=ax)
nx.draw_networkx_edges(G, pos, edge_color='#555555', width=1, alpha=0.7, ax=ax)

# # Add color bar for sensitivity values
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=sensitivities.min(), vmax=sensitivities.max()))
plt.colorbar(sm, label='Sensitivity', ax=ax)

fig.savefig("Habitat_suitability_with_sensitivities.png", transparent=True, dpi=300)
