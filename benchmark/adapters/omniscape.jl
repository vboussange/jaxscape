# Omniscape.jl computes the normalized current of E3 with the configuration of
# its Maryland example: `radius` is the shared `dependency_range`, and
# `block_size` is `ceil(dependency_range * coarsening_factor)`, made odd as
# JAXScape makes its blocks, 21 pixels. Resistance is 1 / permeability, with
# the barriers, of permeability 0, as NoData, and the source strength of a
# pixel is its quality; on Maryland, these are the resistances of the example's
# reclassification table and the forest sources of its `source_from_resistance`
# with `r_cutoff` 1. The map is the cumulative current divided by the flow
# potential (`calc_normalized_current`, `calc_flow_potential`), compared with
# JAXScape's sensitivity under resistance, the distance of Omniscape's
# circuits. Omniscape keeps its other defaults: eight neighbours, the
# conductance of an edge being the mean permeability of its two pixels, divided
# by √2 on diagonals; artifact correction; double precision; the solver of the
# settings, with the windows solved in parallel on the Julia threads of the
# cell. Each call runs `run_omniscape` on arrays in memory, which redoes the
# sources, the targets and every window solve; it reads and writes no file,
# and only creates an empty project directory in /dev/shm.
include("protocol.jl")
using Omniscape

cell, arrays = read_cell()
windows = cell["windows"]
resistance = Matrix{Union{Missing, Float64}}(
    map(p -> p > 0 ? 1 / p : missing, arrays["permeability"])
)
sources = Matrix{Union{Missing, Float64}}(arrays["quality"])
block_size = ceil(Int, windows["dependency_range"] * windows["coarsening_factor"])
config = Dict{String, String}(
    "radius" => string(windows["dependency_range"]),
    "block_size" => string(block_size + 1 - block_size % 2),
    "calc_normalized_current" => "true",
    "calc_flow_potential" => "true",
    "solver" => cell["settings"]["solver"],
)

calls = 0
function call()
    global calls += 1
    # Omniscape creates its project directory, and a new one when it exists.
    project = joinpath(dirname(cell["output"]), "omniscape-$calls")
    outputs = run_omniscape(
        merge(config, Dict("project_name" => project)), copy(resistance);
        source_strength = copy(sources),
    )
    outputs === nothing && error("Omniscape rejected its configuration")
    cumulative, flow_potential, normalized = outputs
    return Float64.(coalesce.(normalized, NaN))
end

write_result(cell, time_calls(cell, call)...; version = string(pkgversion(Omniscape)))
