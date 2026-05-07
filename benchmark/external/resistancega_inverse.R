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
    !requireNamespace("ResistanceGA", quietly = TRUE) ||
    !requireNamespace("raster", quietly = TRUE) ||
    !requireNamespace("sp", quietly = TRUE)) {
  payload <- list(
    status = "skipped",
    timings_seconds = list(),
    median_seconds = NULL,
    note = "Install the local R benchmark library with ResistanceGA, raster, and sp to run this adapter."
  )
  jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  quit(save = "no")
}

case_payload <- jsonlite::read_json(case_path, simplifyVector = TRUE)
permeability <- coerce_numeric_matrix(case_payload$raster)
resistance_values <- 1 / pmax(permeability, 1e-3)
resistance_raster <- raster::raster(resistance_values, xmn = 0, xmx = ncol(resistance_values), ymn = 0, ymx = nrow(resistance_values))

point_matrix <- coerce_numeric_matrix(case_payload$points)
sample_coords <- cbind(
  x = point_matrix[, 2] + 0.5,
  y = nrow(resistance_values) - point_matrix[, 1] - 0.5
)
sample_locales <- sp::SpatialPoints(sample_coords)
n_samples <- nrow(sample_coords)
benchmark_threads <- suppressWarnings(as.integer(Sys.getenv("BENCHMARK_THREADS", "4")))
if (is.na(benchmark_threads) || benchmark_threads < 1) {
  benchmark_threads <- 4L
}

seed_inputs <- ResistanceGA::gdist.prep(
  n.Pops = n_samples,
  samples = sample_locales,
  method = "costDistance"
)
response_vector <- as.vector(ResistanceGA::Run_gdistance(gdist.inputs = seed_inputs, r = resistance_raster, scl = FALSE))

fit_error_metrics <- function(result, target_vector) {
  if (is.null(result$cd) || length(result$cd) == 0) {
    return(list())
  }

  fitted_matrix <- result$cd[[1]]
  fitted_vector <- as.vector(fitted_matrix[lower.tri(fitted_matrix)])
  residual <- fitted_vector - target_vector
  rmse <- sqrt(mean(residual ^ 2))
  reference_scale <- sqrt(mean(target_vector ^ 2))
  ga_result <- if (!is.null(result$ga) && length(result$ga) > 0) result$ga[[1]] else NULL
  iteration_count <- if (!is.null(ga_result)) ga_result@iter else NA_integer_
  iteration_limit <- if (!is.null(ga_result)) ga_result@maxiter else NA_integer_
  converged <- !is.null(ga_result) && ga_result@iter < ga_result@maxiter

  list(
    rmse = unname(rmse),
    relative_rmse = unname(if (reference_scale > 0) rmse / reference_scale else rmse),
    aicc = if (!is.null(result$AICc) && nrow(result$AICc) > 0) unname(result$AICc$AICc[[1]]) else NULL,
    converged = converged,
    iteration_count = iteration_count,
    iteration_limit = iteration_limit
  )
}

run_once <- function() {
  results_dir <- tempfile(pattern = "jaxscape-rga-")
  dir.create(results_dir, recursive = TRUE)
  gdist_inputs <- ResistanceGA::gdist.prep(
    n.Pops = n_samples,
    response = response_vector,
    samples = sample_locales,
    method = "costDistance"
  )
  ga_inputs <- ResistanceGA::GA.prep(
    ASCII.dir = resistance_raster,
    Results.dir = paste0(results_dir, "/"),
    method = "LL",
    select.trans = list("M"),
    seed = 7,
    parallel = benchmark_threads,
    pop.size = 12,
    maxiter = 3,
    run = 1,
    max.cont = 25
  )
  result <- ResistanceGA::SS_optim(
    gdist.inputs = gdist_inputs,
    GA.inputs = ga_inputs,
    diagnostic_plots = FALSE,
    dist_mod = FALSE,
    null_mod = FALSE
  )
  list(
    run_time = result$Run.Time,
    metrics = fit_error_metrics(result, response_vector)
  )
}

warmup <- run_once()
timings <- numeric(repeats)
last_run <- warmup
for (repeat_index in seq_len(repeats)) {
  start_time <- proc.time()[["elapsed"]]
  last_run <- run_once()
  timings[[repeat_index]] <- proc.time()[["elapsed"]] - start_time
}

payload <- list(
  status = "ok",
  timings_seconds = as.list(unname(timings)),
  median_seconds = stats::median(timings),
  metrics = last_run$metrics,
  note = "Fixed-budget ResistanceGA calibration against the synthetic least-cost response generated from the benchmark raster."
)
jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
