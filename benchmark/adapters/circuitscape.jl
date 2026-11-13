# Circuitscape.jl computes the pairwise resistances among the sites of the
# distance calculation, in its `pairwise` scenario, with the permeability
# raster as the conductance of each pixel. `connect_four_neighbors_only` gives
# rook contiguity, and `connect_using_avg_resistances` makes the resistance of
# an edge the mean resistance of its two pixels, as `landscape.conductance`
# does. The solver is `cg+amg` or `cholmod`, in double precision, with
# `parallelize` on the Julia threads of the cell. Circuitscape.jl has no public
# in-memory API for this scenario: each call builds its `RasterData` in memory
# and runs the internals that `compute` runs after reading files
# (`build_problem`, `single_ground_all_pairs`), and writes only its table of
# resistances, to /dev/shm. Every call rebuilds the graph, the Laplacian and
# the AMG hierarchy or Cholesky factor.
include("protocol.jl")
using Circuitscape

cell, arrays = read_cell()
conductance = Float64.(arrays["permeability"])
sites = arrays["sites"] .+ 1
nrows, ncols = size(conductance)
config = Circuitscape.init_config()
Circuitscape.update!(
    config,
    Dict(
        "scenario" => "pairwise",
        "solver" => cell["settings"]["solver"],
        "connect_four_neighbors_only" => "True",
        "connect_using_avg_resistances" => "True",
        "parallelize" => "True",
        "output_file" => joinpath(dirname(cell["output"]), "circuitscape.out"),
        "suppress_messages" => "True",
    ),
)

function call()
    data = Circuitscape.RasterData(
        conductance,
        Matrix{Int}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        Matrix{Float64}(undef, 0, 0),
        (sites[:, 1], sites[:, 2], collect(1:size(sites, 1))),
        Matrix{Float64}(undef, 0, 0),
        Circuitscape.IncludeExcludePairs(Int),
        Circuitscape.RasterMeta(ncols, nrows, 0.0, 0.0, 1.0, -9999.0, Float64[0], ""),
    )
    cfg = Circuitscape.CSConfig(config)
    Circuitscape.update_logging!(cfg)
    problem = Circuitscape.build_problem(data, cfg)
    table = Circuitscape.single_ground_all_pairs(problem, cfg)
    ids = Int.(table[2:end, 1])  # row and column 1 hold the ids of the sites
    resistances = similar(table, length(ids), length(ids))
    resistances[ids, ids] = table[2:end, 2:end]
    return resistances
end

write_result(cell, time_calls(cell, call)...; files = true)
