# gdistance builds a transition layer of the permeability raster with 4
# neighbours (rook contiguity), the conductance between two pixels being the
# mean or the harmonic mean of their permeabilities, as `graph` in the
# configuration sets it. Resistance distances are `commuteDistance` divided by
# the volume of the graph, the sum of its conductances in both directions, and
# least-cost distances are `costDistance`, whose edge cost is 1 / conductance.
# Pixels have side 1, so `geoCorrection` leaves the layer unchanged and is left
# out. Every call builds the transition layer, whose conductances depend on the
# permeability.
source("adapters/protocol.R")
suppressPackageStartupMessages(library(gdistance))

cell <- read_cell()
permeability <- cell$arrays$permeability
sites <- cell$arrays$sites
landscape <- raster(permeability, xmn = 0, xmx = ncol(permeability), ymn = 0, ymx = nrow(permeability))
# The centres of the sites' pixels; the first row of the raster is at the top.
xy <- cbind(sites[, 2] + 0.5, nrow(permeability) - sites[, 1] - 0.5)
conductance <- list(mean = mean, harmonic = function(x) 1 / mean(1 / x))[[cell$graph$conductance]]
layer <- function() transition(landscape, conductance, directions = 4, symm = TRUE)

call <- list(
  resistance = function() {
    graph <- layer()
    as.matrix(commuteDistance(graph, xy)) / sum(transitionMatrix(graph))
  },
  lcp = function() as.matrix(costDistance(layer(), xy))
)[[cell$distance]]

write_result(cell, time_calls(cell, call), version = format(packageVersion("gdistance")))
