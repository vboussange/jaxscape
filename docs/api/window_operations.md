# Window operations

`WindowOperation` divides a raster into square core windows, each with a
buffer of the neighbouring pixels that a computation on the core depends on,
and writes the results of a computation on each window back into a raster. It
serves to build window-based computations other than the analyses of
[Connectivity analysis](connectivity_analysis.md).
`padding` pads a raster to a shape that `WindowOperation` accepts.

::: jaxscape.window_operation.WindowOperation
    options:
      heading_level: 2
      members:
        - nb_steps
        - extract_total_window
        - extract_core_window
        - update_raster_with_core_window
        - update_raster_with_window
        - lazy_iterator
        - eager_iterator

::: jaxscape.utils.padding
    options:
      heading_level: 2
