args <- commandArgs(trailingOnly = TRUE)
lib_path <- if (length(args) >= 1) args[[1]] else Sys.getenv("R_LIBS_USER")

if (!nzchar(lib_path)) {
  stop("Provide a target R library directory as the first argument or set R_LIBS_USER.")
}

dir.create(lib_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib_path, .libPaths()))

options(
  repos = c(CRAN = "https://cloud.r-project.org"),
  timeout = max(600, getOption("timeout")),
  Ncpus = max(1L, parallel::detectCores(logical = TRUE) - 1L)
)

append_env_path <- function(current_value, new_entries) {
  entries <- Filter(nzchar, c(new_entries, strsplit(current_value, .Platform$path.sep, fixed = TRUE)[[1]]))
  paste(unique(entries), collapse = .Platform$path.sep)
}

configure_homebrew_gfortran_runtime <- function() {
  if (Sys.info()[["sysname"]] != "Darwin") {
    return(invisible(NULL))
  }

  brew <- Sys.which("brew")
  if (!nzchar(brew)) {
    return(invisible(NULL))
  }

  gcc_prefix <- tryCatch(
    system2(brew, c("--prefix", "gcc"), stdout = TRUE, stderr = FALSE),
    error = function(...) character()
  )
  if (length(gcc_prefix) == 0L || !nzchar(gcc_prefix[[1]])) {
    return(invisible(NULL))
  }

  shared_dir <- file.path(gcc_prefix[[1]], "lib", "gcc", "current")
  runtime_dirs <- Sys.glob(file.path(gcc_prefix[[1]], "lib", "gcc", "*", "gcc", "aarch64-apple-darwin*", "*"))
  runtime_dirs <- runtime_dirs[file.exists(file.path(runtime_dirs, "libemutls_w.a"))]
  search_dirs <- unique(c(runtime_dirs, shared_dir))
  search_dirs <- search_dirs[file.exists(search_dirs)]
  if (length(search_dirs) == 0L) {
    return(invisible(NULL))
  }

  ldflags <- trimws(paste(Sys.getenv("LDFLAGS"), paste(sprintf("-L%s", search_dirs), collapse = " ")))
  flibs <- trimws(paste(paste(sprintf("-L%s", search_dirs), collapse = " "), Sys.getenv("FLIBS")))

  Sys.setenv(
    LIBRARY_PATH = append_env_path(Sys.getenv("LIBRARY_PATH"), search_dirs),
    LDFLAGS = ldflags,
    FLIBS = flibs
  )

  message(sprintf("Configured Homebrew GCC runtime search path: %s", paste(search_dirs, collapse = ", ")))
}

configure_homebrew_gfortran_runtime()

lock_dirs <- Sys.glob(file.path(lib_path, "00LOCK*"))
if (length(lock_dirs) > 0) {
  message(sprintf("Removing stale R install locks from %s", normalizePath(lib_path)))
  unlink(lock_dirs, recursive = TRUE, force = TRUE)
}

install_if_missing <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) == 0) {
    return(invisible(NULL))
  }

  message(sprintf("Installing CRAN packages: %s", paste(missing, collapse = ", ")))
  if (Sys.info()[["sysname"]] == "Darwin") {
    tryCatch(
      {
        install.packages(missing, quiet = FALSE, type = "binary")
        return(invisible(NULL))
      },
      error = function(error) {
        message(sprintf("Binary R packages are unavailable here (%s). Falling back to source installs.", conditionMessage(error)))
      }
    )
  }
  install.packages(missing, quiet = FALSE)
}

install_if_missing(c("jsonlite", "remotes", "samc"))

if (!requireNamespace("ResistanceGA", quietly = TRUE)) {
  message("Installing ResistanceGA from GitHub with hard dependencies only.")
  remotes::install_github(
    "wpeterman/ResistanceGA",
    upgrade = "never",
    dependencies = NA,
    build_vignettes = FALSE,
    quiet = FALSE
  )
}

required_packages <- c("jsonlite", "samc", "ResistanceGA", "gdistance", "raster", "sp")
missing_required <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_required) > 0) {
  stop(sprintf("Failed to install required R benchmark packages: %s", paste(missing_required, collapse = ", ")))
}

message(sprintf("Installed R benchmark packages into %s", normalizePath(lib_path)))