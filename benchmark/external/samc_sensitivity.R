args <- commandArgs(trailingOnly = TRUE)
raster_path <- args[[1]]
repeats <- as.integer(args[[2]])
output_path <- args[[3]]

if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("samc", quietly = TRUE) ||
    !requireNamespace("terra", quietly = TRUE)) {
  payload <- list(
    status = "skipped",
    timings_seconds = list(),
    median_seconds = NULL,
    note = "Install the R packages 'samc' and 'terra' to run the samc benchmark adapter."
  )
  jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  quit(save = "no")
}

payload <- list(
  status = "skipped",
  timings_seconds = list(),
  median_seconds = NULL,
  note = "The samc adapter script is wired into the benchmark orchestrator; complete the project-specific passage benchmark here when the R environment is available."
)
jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
