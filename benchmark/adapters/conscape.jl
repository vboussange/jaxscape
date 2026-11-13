# ConScape.jl builds its graph with rook contiguity (`N4`), the affinity of an
# edge being the harmonic mean of the permeabilities of its two pixels
# (`AverageWeight`), as `landscape.conductance` gives, and the cost of an edge
# 1 / affinity (`Inv`). Its free energy takes the shared `theta`. The distance
# calculation computes the free-energy distances among the sites, the targets
# of a `GridRSP`. ConScape's free-energy distance is directed, and the adapter
# returns the mean of the two directions between two sites, as JAXScape's
# `FreeEnergyDistance` does. The connectivity and sensitivity analyses run on
# the whole raster, without windows, with targets at landmarks every b pixels
# (`coarse_graining`), each holding the quality of its block, where b is the
# `block_size` of the windows, JAXScape's block size: 21. The connectivity
# analysis returns Hanski's S_i, the connected habitat of each pixel at source
# quality 1 (`connected_habitat`): the sum over landmarks of their quality
# times exp(-d / (dependency_range / 4)), exp(-d / 25), for the free-energy
# distance d to the landmark. The sensitivity analysis returns the
# quality-weighted betweenness (`betweenness_qweighted`). JAXScape has no
# free-energy map, and the configuration compares these maps with JAXScape's
# under least-cost distance (`compared_with = "lcp"`). Free energy runs from a
# multiple of resistance distance, as theta falls to 0, to least-cost distance,
# as theta grows. On the synthetic raster of 256 pixels a side, for every theta
# from 1e-5 to 1.5e-2, it ranks the pairs of sites of the distance calculation
# as least-cost distance does (Spearman's rho 0.98 to 0.99, against 0.92 to
# 0.95 with resistance distance), and at the shared theta, 0.01, ConScape's
# maps of connectivity and sensitivity correlate more with the maps of
# JAXScape's GPU cells under least-cost distance (rho 0.93 and 0.84) than under
# resistance distance (0.87 and 0.72). Every call rebuilds the affinities, the
# `Grid`, the landmarks and the `GridRSP`, with its sparse LU factorization,
# which runs on the BLAS threads of the cell.
include("protocol.jl")
using ConScape, Logging, SparseArrays

cell, arrays = read_cell()
quality, permeability = arrays["quality"], arrays["permeability"]
nrows, ncols = size(permeability)
θ = cell["parameters"]["theta"]

function grid(; qualities...)
    # An edge out of a pixel of permeability 0 has affinity 0, and `Grid`
    # rejects its cost unless the edge is dropped.
    affinities = dropzeros!(
        ConScape.graph_matrix_from_raster(
            permeability; neighbors = ConScape.N4, weight = ConScape.AverageWeight
        ),
    )
    return with_logger(NullLogger()) do  # `Grid` logs its pruning
        ConScape.Grid(nrows, ncols; affinities, costs = ConScape.Inv(), qualities...)
    end
end

function isolation()
    sites = CartesianIndex.(arrays["sites"][:, 1] .+ 1, arrays["sites"][:, 2] .+ 1)
    targets = sparse(first.(Tuple.(sites)), last.(Tuple.(sites)), 1.0, nrows, ncols)
    grid_ = grid(source_qualities = ones(nrows, ncols), target_qualities = targets)
    distances = ConScape.free_energy_distance(ConScape.GridRSP(grid_; θ))
    # Rows of `distances` are the nodes of the grid, and columns its target nodes.
    nodes = indexin(sites, grid_.id_to_grid_coordinate_list)
    directed = distances[nodes, indexin(nodes, ConScape._targetidx_and_nodes(grid_)[2])]
    return (directed + directed') / 2
end

landmarks() = ConScape.coarse_graining(grid(qualities = quality), cell["windows"]["block_size"])

function connectivity()
    grid_ = grid(source_qualities = ones(nrows, ncols), target_qualities = landmarks())
    scale = cell["windows"]["dependency_range"] / 4
    return ConScape.connected_habitat(
        ConScape.GridRSP(grid_; θ);
        connectivity_function = ConScape.free_energy_distance,
        distance_transformation = d -> exp(-d / scale),
    )
end

function corridors()
    grid_ = grid(source_qualities = quality, target_qualities = landmarks())
    return ConScape.betweenness_qweighted(ConScape.GridRSP(grid_; θ))
end

call = Dict(
    "distance" => isolation,
    "connectivity" => connectivity,
    "sensitivity" => corridors,
)[cell["question"]]
write_result(cell, time_calls(cell, call)...)
