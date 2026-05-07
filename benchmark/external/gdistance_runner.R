args <- commandArgs(trailingOnly = TRUE)
mode <- args[[1]]
case_path <- args[[2]]
repeats <- as.integer(args[[3]])
output_path <- args[[4]]

coerce_numeric_matrix <- function(value) {
  if (is.matrix(value)) {
    return(matrix(as.numeric(value), nrow = nrow(value), ncol = ncol(value)))
  }
  matrix(as.numeric(unlist(value)), nrow = length(value), byrow = TRUE)
}

write_payload <- function(payload) {
  jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
}

if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("gdistance", quietly = TRUE) ||
    !requireNamespace("raster", quietly = TRUE) ||
    !requireNamespace("sp", quietly = TRUE)) {
  write_payload(list(
    status = "skipped",
    timings_seconds = list(),
    median_seconds = NULL,
    metrics = list(),
    note = "Install the local R benchmark library with gdistance, raster, sp, and jsonlite to run this adapter."
  ))
  quit(save = "no")
}

case_payload <- jsonlite::read_json(case_path, simplifyVector = TRUE)
cost_values <- coerce_numeric_matrix(case_payload$raster)
cost_raster <- raster::raster(cost_values, xmn = 0, xmx = ncol(cost_values), ymn = 0, ymx = nrow(cost_values))
transition_layer <- gdistance::transition(cost_raster, function(x) 1 / mean(x), directions = 4)

point_matrix <- coerce_numeric_matrix(case_payload$points)
sample_coords <- cbind(
  x = point_matrix[, 2] + 0.5,
  y = nrow(cost_values) - point_matrix[, 1] - 0.5
)
sample_locales <- sp::SpatialPoints(sample_coords)
origin_coord <- sample_coords[1, , drop = FALSE]
goal_coords <- sample_coords[2:nrow(sample_coords), , drop = FALSE]

pairwise_commute_distance <- function() {
  sample_count <- nrow(sample_coords)
  distances <- matrix(0, nrow = sample_count, ncol = sample_count)
  for (source_index in seq_len(sample_count - 1)) {
    for (target_index in seq.int(source_index + 1, sample_count)) {
      pair_coords <- sample_coords[c(source_index, target_index), , drop = FALSE]
      pair_distance <- as.matrix(gdistance::commuteDistance(transition_layer, sp::SpatialPoints(pair_coords)))[1, 1]
      distances[source_index, target_index] <- pair_distance
      distances[target_index, source_index] <- pair_distance
    }
  }
  distances
}

sum_shortest_path_rasters <- function() {
  accumulator <- matrix(0, nrow = nrow(cost_values), ncol = ncol(cost_values))
  for (goal_index in seq_len(nrow(goal_coords))) {
    path_lines <- gdistance::shortestPath(
      transition_layer,
      origin_coord,
      goal_coords[goal_index, , drop = FALSE],
      output = "SpatialLines"
    )
    path_raster <- raster::rasterize(path_lines, cost_raster, field = 1, background = 0)
    accumulator <- accumulator + raster::as.matrix(path_raster)
  }
  accumulator
}

sum_passage_rasters <- function() {
  accumulator <- matrix(0, nrow = nrow(cost_values), ncol = ncol(cost_values))
  for (goal_index in seq_len(nrow(goal_coords))) {
    passage_layer <- gdistance::passage(
      transition_layer,
      origin_coord,
      goal_coords[goal_index, , drop = FALSE],
      theta = 0,
      totalNet = "total",
      output = "RasterLayer"
    )
    accumulator <- accumulator + raster::as.matrix(passage_layer)
  }
  accumulator
}

run_mode <- switch(
  mode,
  least_cost_path = function() gdistance::costDistance(transition_layer, sample_locales),
  resistance_distance = pairwise_commute_distance,
  least_cost_centrality = sum_shortest_path_rasters,
  resistance_centrality = sum_passage_rasters,
  stop(sprintf("Unsupported gdistance benchmark mode: %s", mode))
)

warmup <- run_mode()
timings <- numeric(repeats)
last_result <- warmup
for (repeat_index in seq_len(repeats)) {
  start_time <- proc.time()[["elapsed"]]
  last_result <- run_mode()
  timings[[repeat_index]] <- proc.time()[["elapsed"]] - start_time
}

metrics <- list()
note <- switch(
  mode,
  least_cost_path = "Least-cost distances computed with gdistance::costDistance on a 4-neighbour transition graph.",
  resistance_distance = "Commute times computed with gdistance::commuteDistance on a 4-neighbour conductance graph.",
  least_cost_centrality = "Single-origin shortest-path incidence computed by summing gdistance::shortestPath transition rasters over all destinations.",
  resistance_centrality = "Single-origin random-walk passage centrality computed by summing gdistance::passage(..., totalNet = \"total\") rasters over all destinations."
)

if (mode == "least_cost_path" || mode == "resistance_distance") {
  metrics$distance_matrix <- unname(as.matrix(last_result))
} else {
  metrics$centrality_raster <- unname(last_result)
}

write_payload(list(
  status = "ok",
  timings_seconds = as.list(unname(timings)),
  median_seconds = stats::median(timings),
  metrics = metrics,
  note = note
))
