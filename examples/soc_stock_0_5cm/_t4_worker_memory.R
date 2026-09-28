# ══════════════════════════════════════════════════════════════════════════════
# T4 -- where a map worker's memory goes: its working set after every phase of
# every step, on full-width 250 m rows, under three settings
#
# THE QUESTION.
#
# T3's third run: a worker's PEAK working set grew ~4.3 GB a unit (14.4 ->
# 27.7 GB over 4 units of one step each; 17.1 -> 25.7 over 3 with 15
# threads) -- with the strips held to a few shapes and oneDNN's primitive
# caches capped, so not that, or not only that. ~4.3 GB is close to one
# full-width step's buffer (the 1.6 GB halo and the 3.7 GB rows), and on the
# 20 km grid, where a step's buffer is ~0.2 GB, the growth hardly shows:
# something a step reads may stay behind. A peak cannot say what. The
# CURRENT working set after each phase -- and after the full collection that
# ends every step -- can: a phase after which the level a collection leaves
# climbs, step after step, is the leak.
#
# THE SETTINGS, one hypothesis each, each on its own fresh rows:
#
#   as_is    the map as it runs
#   gdal_1   GDAL_NUM_THREADS = 1 -- GDAL >= 3.6 decodes the strips of one read
#            on threads; a leak there would belong to the reader
#   mkl_mm   MKL_DISABLE_FAST_MM = 1 -- MKL keeps buffers for reuse, sized to
#            the products it has run
#
# 1 worker x 15 threads (T3's fastest), 3 units of 32 rows, one step each,
# every band, both calibration sources.
#
# WHAT IT PRINTS: per setting, the level after each phase of each unit, and
# how much the level a full collection leaves grows from unit to unit. The
# checks only say the measurement is complete: what it means is read from the
# table.
#
# WHAT IT WRITES: three small maps under outputs/.../dsm_predict/t4_<setting>_
# <commit> (~0.4 GB each) and the traces in outputs/.../tuning/.../
# capability_sweep/t4_memory/.
#
# COST: ~20-25 min.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t4_worker_memory.R")
# ══════════════════════════════════════════════════════════════════════════════

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/cnn_architecture.R.
project_root <- (function() {
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
  }
  cand <- c(cand, getwd())
  for (d in cand) for (up in c(".", "..", "../..", "../../..")) {
    r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
    if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))
install_load_pkg(c("torch", "terra", "dplyr", "readr", "tibble", "purrr", "matrixStats",
                   "ps", "callr"))
pkgload::load_all(project_root)
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
data_dir     <- base("data", "processed")
patch_dir    <- base("outputs", "patches")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
sp_dir       <- base("outputs", "spatial_prediction")
maps_dir     <- file.path(sp_dir, "dsm_predict")
t4_dir       <- file.path(tuning_base, "capability_sweep", "t4_memory")
code_tag     <- .git_commit_at(project_root)
unit_rows    <- 32L
n_units      <- 3L

settings <- list(as_is  = character(0),
                 gdal_1 = c(GDAL_NUM_THREADS = "1"),
                 mkl_mm = c(MKL_DISABLE_FAST_MM = "1"))
# T3's runs mapped 896 rows from the first profile's; these follow them.
offset_rows <- env_int("soc_t4_offset_rows", 896L, min = 0L)

final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir  <- file.path(final_base, final_run_id)
summ       <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
cfg_row    <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)
calibration <- c(block = tuning_dir)
for (cand in file.path(tuning_base, c("soc_0_5cm_knndm_cfg003", "soc_0_5cm_design_knndm"))) {
  if (dir.exists(cand) &&
      !is.null(suppressMessages(cv_residuals_for_config(cand, cfg_row, required = FALSE)))) {
    calibration <- c(calibration, knndm = cand)
    break
  }
}

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = 3L,
  target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
  verbose      = FALSE)
rt <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
qc_path <- file.path(metadata_dir, "qc_table.csv")
g250 <- terra::rast(rt$raster_file[1])
n_col_250 <- as.integer(terra::ncol(g250))
r_first <- as.integer(terra::rowFromCell(g250, terra::cellFromXY(
  g250, cbind(data$store$meta$x[1], data$store$meta$y[1]))))

message("\n", strrep("=", 78))
message("T4 -- a map worker's working set, phase by phase, full-width 250 m rows")
message(strrep("=", 78))

required <- c("t4_01")
L <- check_ledger("T4")
traces <- list()
peaks <- list()
old_opts <- options(dsm.predict.trace_mem = TRUE)
for (k in seq_along(settings)) {
  nm <- names(settings)[k]
  r0 <- r_first - 3L + offset_rows + (k - 1L) * n_units * unit_rows
  message(sprintf("\n-- %s: rows %d-%d | extra environment: %s --", nm, r0,
                  r0 + n_units * unit_rows - 1L,
                  if (length(settings[[k]])) paste(sprintf("%s=%s", names(settings[[k]]), settings[[k]]),
                                                   collapse = " ") else "none"))
  options(dsm.predict.worker_env = settings[[k]])
  m <- dsm_predict(final_dir, data, rasters = rt, qc_table = qc_path, calibration = calibration,
                   extent = list(rows = c(r0, r0 + n_units * unit_rows - 1L), cols = c(1L, n_col_250)),
                   n_cores = 15L, threads_per_worker = 15L, unit_rows = unit_rows,
                   step_rows = unit_rows, probe = FALSE, output_dir = maps_dir,
                   run_id = sprintf("t4_%s_%s", nm, code_tag), verbose = FALSE)
  recs <- lapply(m$units$unit_id, function(u) readRDS(file.path(m$run_dir, "units", u, "done.rds")))
  traces[[nm]] <- dplyr::bind_rows(lapply(seq_along(recs), function(i) {
    tr <- recs[[i]]$mem_trace
    if (is.null(tr)) return(NULL)
    tr$unit <- i
    tr
  }))
  if (nrow(traces[[nm]]) > 0L) traces[[nm]]$setting <- nm
  peaks[[nm]] <- m$units$peak_gb
}
options(dsm.predict.worker_env = NULL)
options(old_opts)

tab <- dplyr::bind_rows(traces)
ledger_check(L, "t4_01", "every setting mapped every unit and traced every phase",
             nrow(tab) > 0L && all(vapply(names(settings), function(nm) {
               tr <- traces[[nm]]
               !is.null(tr) && nrow(tr) > 0L && length(unique(tr$unit)) == n_units &&
                 all(c("start", "read", "network_di", "bands", "write", "gc", "unit_end") %in% tr$phase)
             }, logical(1))),
             paste(vapply(names(settings), function(nm) sprintf("%s: %d mark(s)", nm,
                                                                 NROW(traces[[nm]])), character(1)),
                   collapse = " | "))

phases <- c("start", "read", "network_di", "bands", "write", "gc", "unit_end")
message("\n-- The working set (GB) after each phase: one row per unit (one step each) --")
for (nm in names(settings)) {
  tr <- traces[[nm]]
  if (is.null(tr) || nrow(tr) == 0L) next
  wide <- do.call(rbind, lapply(split(tr, tr$unit), function(u) {
    v <- stats::setNames(u$rss_gb[match(phases, u$phase)], phases)
    data.frame(setting = nm, unit = u$unit[1], t(round(v, 2)), check.names = FALSE)
  }))
  print(wide, row.names = FALSE)
  after_gc <- wide$gc
  message(sprintf("  %s: the level a full collection leaves, unit to unit: %s  (%+.2f GB a unit) | peaks %s GB",
                  nm, paste(sprintf("%.2f", after_gc), collapse = " -> "),
                  if (length(after_gc) > 1L) (after_gc[length(after_gc)] - after_gc[1]) / (length(after_gc) - 1L) else NA_real_,
                  paste(sprintf("%.1f", peaks[[nm]]), collapse = " / ")))
}

create_output_dirs(t4_dir)
safe_write_csv2(tab, file.path(t4_dir, "t4_traces.csv"))
v <- ledger_verdict(L, required, file.path(t4_dir, "t4_checks.csv"))
message("\nTraces: ", t4_dir)
