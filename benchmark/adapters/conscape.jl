# ConScape.jl computes the free-energy distances among the E1 sites, with the
# shared `theta`. Its graph has rook contiguity (`N4`), the affinity of an edge
# being the harmonic mean of the permeabilities of its two pixels
# (`AverageWeight`), as `landscape.conductance` gives, and the cost of an edge
# 1 / affinity (`Inv`). The sites are the targets of a `GridRSP`. ConScape's
# free-energy distance is directed, and the adapter returns the mean of the two
# directions between two sites, as JAXScape's `FreeEnergyDistance` does. Every
# call rebuilds the affinities, the `Grid` and the `GridRSP`, with its sparse LU
# factorization, which runs on the BLAS threads of the cell.
include("protocol.jl")
using ConScape, Logging, SparseArrays

cell, arrays = read_cell()
permeability = arrays["permeability"]
sites = CartesianIndex.(arrays["sites"][:, 1] .+ 1, arrays["sites"][:, 2] .+ 1)
nrows, ncols = size(permeability)
targets = sparse(first.(Tuple.(sites)), last.(Tuple.(sites)), 1.0, nrows, ncols)

function call()
    affinities = ConScape.graph_matrix_from_raster(
        permeability; neighbors = ConScape.N4, weight = ConScape.AverageWeight
    )
    grid = with_logger(NullLogger()) do  # `Grid` logs its pruning
        ConScape.Grid(
            nrows,
            ncols;
            affinities,
            costs = ConScape.Inv(),
            source_qualities = ones(nrows, ncols),
            target_qualities = targets,
        )
    end
    distances = ConScape.free_energy_distance(ConScape.GridRSP(grid; θ = cell["parameters"]["theta"]))
    # Rows of `distances` are the nodes of the grid, and columns its target nodes.
    nodes = indexin(sites, grid.id_to_grid_coordinate_list)
    directed = distances[nodes, indexin(nodes, ConScape._targetidx_and_nodes(grid)[2])]
    return (directed + directed') / 2
end

write_result(cell, time_calls(cell, call)...; version = string(pkgversion(ConScape)))
