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

!!! warning "Changed in 0.0.16"

    `ResistanceDistance()` solves with `GridMultigridSolver()` by default, where
    earlier versions took dense solves. Its queries raise on graphs other than
    grid graphs, and on disconnected graphs: `ResistanceDistance(solver=None)`
    takes the dense solves of earlier versions.

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

!!! warning "Removed in 0.0.14"

    `RSPDistance` is removed. `FreeEnergyDistance` replaces it, with edge costs
    equal to inverse conductances and symmetric first-hitting free energies as
    output. Its values differ from the expected path costs that `RSPDistance`
    returned.

::: jaxscape.free_energy_distance.FreeEnergyDistance
    options:
      members: false
