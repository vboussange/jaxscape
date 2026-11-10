# samc builds a spatial absorbing Markov chain on the resistance raster, 1 /
# permeability, with 4 neighbours (rook contiguity) and the transition function
# 1 / mean(x): an individual moves to a neighbour in proportion to the harmonic
# mean of the permeabilities of the two pixels, `landscape.conductance`, and
# barriers, of permeability 0, are left out of the chain. The absorption of
# every pixel is (4 / `dependency_range`)^2 per step, so that an individual
# moves about `dependency_range` / 4 pixels before it is absorbed, the length
# over which JAXScape's default proximity decays. Individuals are released in
# proportion to quality, and the map is `visitation`: the expected number of
# visits to each pixel before absorption. The solver is samc's `method`, whose
# `direct` method factorizes the chain with a single thread. Every call builds
# the chain, on which samc caches its factorization. Its map is compared with
# JAXScape's under resistance distances: an absorbing random walk is the random
# walk behind resistance distance.
source("adapters/protocol.R")
suppressPackageStartupMessages(library(samc))

cell <- read_cell()
resistance <- 1 / cell$arrays$permeability
barrier <- !is.finite(resistance)
resistance[barrier] <- NA
absorption <- matrix((4 / cell$windows$dependency_range)^2, nrow(resistance), ncol(resistance))
absorption[barrier] <- NA
quality <- cell$arrays$quality
quality[barrier] <- NA
model <- list(fun = "1/mean(x)", dir = 4, sym = TRUE)

call <- function() {
  chain <- samc(resistance, absorption, model = model, options = list(method = cell$settings$solver))
  map(chain, visitation(chain, init = quality))
}

write_result(cell, time_calls(cell, call))
