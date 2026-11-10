# gdistance builds a transition layer of the resistance raster, 1 /
# permeability, with 4 neighbours (rook contiguity) and the transition function
# 1 / mean(x): the conductance between two pixels is the reciprocal of their
# mean resistance, `landscape.conductance`. Resistance distances are
# `commuteDistance` divided by the volume of the graph, the sum of its
# conductances in both directions, and least-cost distances are
# `costDistance`, whose edge cost is 1 / conductance. Pixels have side 1, so
# `geoCorrection` leaves the layer unchanged and is left out. Every call builds
# the transition layer, whose conductances depend on the resistance.
source("adapters/protocol.R")
suppressPackageStartupMessages(library(gdistance))

cell <- read_cell()
resistance <- 1 / cell$arrays$permeability
sites <- cell$arrays$sites
landscape <- raster(resistance, xmn = 0, xmx = ncol(resistance), ymn = 0, ymx = nrow(resistance))
# The centres of the sites' pixels; the first row of the raster is at the top.
xy <- cbind(sites[, 2] + 0.5, nrow(resistance) - sites[, 1] - 0.5)
layer <- function() transition(landscape, function(x) 1 / mean(x), directions = 4, symm = TRUE)

call <- list(
  resistance = function() {
    graph <- layer()
    as.matrix(commuteDistance(graph, xy)) / sum(transitionMatrix(graph))
  },
  lcp = function() as.matrix(costDistance(layer(), xy))
)[[cell$distance]]

write_result(cell, time_calls(cell, call), version = format(packageVersion("gdistance")))
