args <- commandArgs(trailingOnly = TRUE)
case_path <- args[[1]]
repeats <- as.integer(args[[2]])
output_path <- args[[3]]

coerce_numeric_matrix <- function(value) {
  if (is.matrix(value)) {
    return(matrix(as.numeric(value), nrow = nrow(value), ncol = ncol(value)))
  }
  matrix(as.numeric(unlist(value)), nrow = length(value), byrow = TRUE)
}

if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("samc", quietly = TRUE)) {
  payload <- list(
    status = "skipped",
    timings_seconds = list(),
    median_seconds = NULL,
    note = "Install the local R benchmark library with samc and jsonlite to run this adapter."
  )
  jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  quit(save = "no")
}

score_surface <- function(resistance_data, absorption_data, init_data) {
  rw_model <- list(fun = function(x) 1 / mean(x), dir = 4, sym = TRUE)
  samc_obj <- samc::samc(resistance_data, absorption_data, model = rw_model)
  init_prob <- init_data / sum(init_data)
  samc::survival(samc_obj, init_prob)
}

finite_difference_gradient <- function(resistance_data, absorption_data, init_data, epsilon = 1e-2) {
  gradient <- matrix(0, nrow = nrow(resistance_data), ncol = ncol(resistance_data))
  for (idx in seq_along(resistance_data)) {
    plus <- resistance_data
    minus <- resistance_data
    plus[idx] <- plus[idx] * (1 + epsilon)
    minus[idx] <- max(1e-3, minus[idx] * (1 - epsilon))
    delta <- plus[idx] - minus[idx]
    gradient[idx] <- (score_surface(plus, absorption_data, init_data) - score_surface(minus, absorption_data, init_data)) / delta
  }
  gradient
}

case_payload <- jsonlite::read_json(case_path, simplifyVector = TRUE)
permeability <- coerce_numeric_matrix(case_payload$raster)
resistance_data <- 1 / pmax(permeability, 1e-3)
absorption_data <- matrix(0.03, nrow = nrow(resistance_data), ncol = ncol(resistance_data))
absorption_data[c(1, nrow(absorption_data)), ] <- 0.08
absorption_data[, c(1, ncol(absorption_data))] <- 0.08
init_data <- matrix(0, nrow = nrow(resistance_data), ncol = ncol(resistance_data))
point_matrix <- coerce_numeric_matrix(case_payload$points)
for (row_index in seq_len(nrow(point_matrix))) {
  row <- point_matrix[row_index, 1] + 1
  col <- point_matrix[row_index, 2] + 1
  init_data[row, col] <- init_data[row, col] + 1
}

invisible(finite_difference_gradient(resistance_data, absorption_data, init_data))
timings <- numeric(repeats)
for (repeat_index in seq_len(repeats)) {
  start_time <- proc.time()[["elapsed"]]
  finite_difference_gradient(resistance_data, absorption_data, init_data)
  timings[[repeat_index]] <- proc.time()[["elapsed"]] - start_time
}

payload <- list(
  status = "ok",
  timings_seconds = as.list(unname(timings)),
  median_seconds = stats::median(timings),
  note = "Finite-difference sensitivity of SAMC mean survival time over the sample-point initial distribution."
)
jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
