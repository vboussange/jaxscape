args <- commandArgs(trailingOnly = TRUE)
raster_path <- args[[1]]
repeats <- as.integer(args[[2]])
output_path <- args[[3]]

if (!requireNamespace("jsonlite", quietly = TRUE) ||
    !requireNamespace("ResistanceGA", quietly = TRUE) ||
    !requireNamespace("raster", quietly = TRUE)) {
  payload <- list(
    status = "skipped",
    timings_seconds = list(),
    median_seconds = NULL,
    note = "Install the R packages 'ResistanceGA' and 'raster' to run the ResistanceGA benchmark adapter."
  )
  jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
  quit(save = "no")
}

payload <- list(
  status = "skipped",
  timings_seconds = list(),
  median_seconds = NULL,
  note = "The ResistanceGA adapter script is wired into the benchmark orchestrator; complete the project-specific inverse benchmark here when the R environment is available."
)
jsonlite::write_json(payload, output_path, auto_unbox = TRUE, pretty = TRUE, null = "null")
