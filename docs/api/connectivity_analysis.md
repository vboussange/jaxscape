# Connectivity analysis

`ConnectivityAnalysis` computes the connectivity of a landscape raster: the sum
of the proximities of its pairs of pixels, weighted by their habitat quality,
where the proximity of two pixels is a decreasing function of their distance.
`SensitivityAnalysis` computes the derivatives of this connectivity with
respect to the quality and the permeability of each pixel. Both compute
distances window by window, so that their memory does not grow with the size
of the raster. The window operations serve to build other definitions of
connectivity (see [Window operations](window_operations.md)).

## Windowed analyses

`ConnectivityAnalysis` and `SensitivityAnalysis` subclass `WindowedAnalysis`:
they share its arguments, and its windows, coarsening, tiles, padding and
permeability floor.

::: jaxscape.windowed_analysis.WindowedAnalysis
    options:
      heading_level: 3
      members: false

## Connectivity

::: jaxscape.connectivity_analysis.ConnectivityAnalysis
    options:
      heading_level: 3
      members:
        - run
      show_bases: false

## Sensitivity

::: jaxscape.connectivity_analysis.SensitivityAnalysis
    options:
      heading_level: 3
      members:
        - run
      show_bases: false
