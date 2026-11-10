# What an R adapter shares with the runner: the protocol of
# `adapters/protocol.py`. R adapters run with `Rscript` in the benchmark
# directory, whose `.Rprofile` loads the renv library of `renv.lock`.

# Return a `.npy` array of numbers, in its shape.
read_npy <- function(path) {
  file <- file(path, "rb")
  on.exit(close(file))
  version <- as.integer(readBin(file, "raw", 8)[7])
  length <- readBin(file, "integer", size = if (version == 1) 2 else 4, endian = "little")
  header <- rawToChar(readBin(file, "raw", length))
  type <- regmatches(header, regexpr("<[fi][0-9]+", header))
  shape <- as.integer(strsplit(sub(".*'shape': \\((.*)\\).*", "\\1", header), ",")[[1]])
  values <- readBin(
    file,
    if (substr(type, 2, 2) == "f") "double" else "integer",
    prod(shape),
    size = as.integer(substring(type, 3)),
    endian = "little"
  )
  if (grepl("'fortran_order': True", header)) array(values, shape) else aperm(array(values, rev(shape)))
}

# Write a matrix of numbers as a float64 `.npy` array.
write_npy <- function(x, path) {
  header <- sprintf("{'descr': '<f8', 'fortran_order': True, 'shape': (%d, %d), }", nrow(x), ncol(x))
  padded <- ceiling((10 + nchar(header) + 1) / 64) * 64 - 10
  header <- paste0(header, strrep(" ", padded - nchar(header) - 1), "\n")
  file <- file(path, "wb")
  on.exit(close(file))
  writeBin(c(as.raw(0x93), charToRaw("NUMPY"), as.raw(c(1, 0))), file)
  writeBin(nchar(header), file, size = 2, endian = "little")
  writeBin(charToRaw(header), file)
  writeBin(as.double(x), file, size = 8, endian = "little")
}

# Return the cell file named on the command line, with the arrays of its case
# in `arrays`.
read_cell <- function() {
  cell <- jsonlite::read_json(commandArgs(trailingOnly = TRUE)[1])
  paths <- list.files(cell$case, "\\.npy$", full.names = TRUE)
  cell$arrays <- setNames(lapply(paths, read_npy), sub("\\.npy$", "", basename(paths)))
  cell
}

# Time `call`, a function that returns once its output is computed: one
# warm-up call, then `repeats` timed calls, or at most one when the warm-up
# took longer than `single_call_after` seconds. Return the output of the last
# call, the warm-up time and the times of the timed calls, in seconds.
time_calls <- function(cell, call) {
  timed <- function() {
    file.create(cell$progress)
    start <- proc.time()[["elapsed"]]
    output <- call()
    list(output = output, time = proc.time()[["elapsed"]] - start)
  }
  last <- warmup <- timed()
  repeats <- if (warmup$time > cell$single_call_after) min(cell$repeats, 1) else cell$repeats
  times <- numeric(0)
  for (i in seq_len(repeats)) {
    last <- timed()
    times <- c(times, last$time)
  }
  list(output = last$output, warmup = warmup$time, times = times)
}

# Save the output of the last call, and write the result of the cell; `files`
# states whether the calls read or write files.
write_result <- function(cell, timing, files = FALSE) {
  write_npy(timing$output, cell$output)
  result <- list(files = files, warmup_s = timing$warmup, times_s = I(timing$times))
  jsonlite::write_json(result, cell$result, auto_unbox = TRUE, digits = NA)
}
