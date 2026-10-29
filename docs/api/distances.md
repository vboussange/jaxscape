# Distance metrics

::: jaxscape.distance.AbstractDistance
    options:
      members: false

::: jaxscape.euclidean_distance.EuclideanDistance
    options:
      members: false

::: jaxscape.lcp_distance.LCPDistance
    options:
      members: false

::: jaxscape.resistance_distance.ResistanceDistance
    options:
      members:
        - init
        - all_pairs_distance_and_state
        - nodes_to_nodes_distance_and_state
        - sources_to_targets_distance_and_state

::: jaxscape.resistance_distance.ExactResistance
    options:
      members: false

::: jaxscape.resistance_distance.SpielmanApproximation
    options:
      members: false

!!! warning "Removed in 0.0.14"

    `RSPDistance` is removed. `FreeEnergyDistance` replaces it, with edge costs
    equal to inverse conductances and symmetric first-hitting free energies as
    output. Its values differ from the expected path costs that `RSPDistance`
    returned.

::: jaxscape.free_energy_distance.FreeEnergyDistance
    options:
      members: false
