# Graphs

Distances are computed on a graph, from its edge weights read as
conductances. `GridGraph` builds the graph of a raster: one vertex per pixel,
and edges between neighbouring pixels, whose weights `fun` computes from the
pixel values. The grid graphs of one raster shape and contiguity can share a
`GridTopology`, which holds their edges. `Graph` holds any graph, given by its
weighted adjacency matrix.

::: jaxscape.graph.AbstractGraph
    options:
      members:
        - get_adjacency_matrix

::: jaxscape.graph.Graph
    options:
      members:
        - get_adjacency_matrix

::: jaxscape.graph.GridGraph
    options:
      members:
        - coord_to_index
        - index_to_coord
        - node_values_to_array
        - array_to_node_values

::: jaxscape.graph.GridTopology
    options:
      members: false
