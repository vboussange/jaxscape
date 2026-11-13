# Distances

A distance object (`AbstractDistance`) takes a graph and returns the distances between some of its
vertices. `LCPDistance`, `ResistanceDistance` and `FreeEnergyDistance` compute
them from the edge weights, which they read as conductances;
`EuclideanDistance` computes them from the pixel coordinates of a `GridGraph`.
Every distance is called in the same way: `distance(graph, nodes=nodes)` for
the distances among `nodes`, `distance(graph, sources=sources,
targets=targets)` for the distances from `sources` to `targets`, and
`distance(graph)` for the distances between all pairs of vertices.
`ResistanceDistance` and `FreeEnergyDistance` solve linear systems, with a
solver chosen as [Linear solvers](linear_solvers.md) describes.

## Interface

`AbstractDistance` defines the queries of every distance.
`AbstractSolverDistance` adds the solver state of the distances that solve
linear systems.

::: jaxscape.distance.AbstractDistance
    options:
      heading_level: 3
      members:
        - reinit
        - sources_to_windows_distance
        - exact
        - debiased_proximity

::: jaxscape.distance.AbstractSolverDistance
    options:
      heading_level: 3
      members:
        - system_matrix
        - init

## Implemented distances

### Euclidean distance

The straight-line distance between pixels, in pixels. It does not depend on
the edge weights.

::: jaxscape.euclidean_distance.EuclideanDistance
    options:
      members: false
      inherited_members: false

### Least-cost path distance

The smallest sum of the edge costs `1 / a` along a path between two vertices,
for the conductance `a` of each edge.

::: jaxscape.lcp_distance.LCPDistance
    options:
      members:
        - sources_to_windows_distance

### Resistance distance

The effective resistance between two vertices when each edge is a resistor of
conductance `a`. Its `method` is `ExactResistance`, which computes it exactly,
or `SpielmanApproximation`, which estimates it from random projections.

::: jaxscape.resistance_distance.ResistanceDistance
    options:
      members:
        - sources_to_windows_distance
        - exact
        - debiased_proximity

::: jaxscape.resistance_distance.ExactResistance
    options:
      members: false
      inherited_members: false

::: jaxscape.resistance_distance.SpielmanApproximation
    options:
      members: false
      inherited_members: false

### Free-energy distance

A distance set by the inverse temperature `theta`, between the least-cost path
distance and the resistance distance: it approaches the least-cost path
distance as `theta` grows, and a multiple of the resistance distance as
`theta` tends to 0.

::: jaxscape.free_energy_distance.FreeEnergyDistance
    options:
      members: false
      inherited_members: false
