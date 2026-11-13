# Window operations

`WindowOperation` divides a raster into square core windows that tile it. The
total window of a core window adds a buffer of the neighbouring pixels on which
a computation on the core window depends. `WindowOperation` extracts the total
windows of a raster and writes the results of a computation on each total
window back into a raster, so that computations other than the analyses of
[Connectivity analysis](connectivity_analysis.md) run as moving-window
operations.
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
