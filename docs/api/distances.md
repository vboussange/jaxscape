# Distance metrics

::: jaxscape.distance.AbstractDistance
    options:
      members:
        - reinit
        - sources_to_windows_distance
        - exact
        - debiased_proximity

::: jaxscape.distance.AbstractSolverDistance
    options:
      members:
        - system_matrix
        - init

::: jaxscape.euclidean_distance.EuclideanDistance
    options:
      members: false

::: jaxscape.lcp_distance.LCPDistance
    options:
      members:
        - sources_to_windows_distance

::: jaxscape.resistance_distance.ResistanceDistance
    options:
      members:
        - sources_to_windows_distance
        - exact
        - debiased_proximity

::: jaxscape.resistance_distance.ExactResistance
    options:
      members: false

::: jaxscape.resistance_distance.SpielmanApproximation
    options:
      members: false

::: jaxscape.free_energy_distance.FreeEnergyDistance
    options:
      members: false
