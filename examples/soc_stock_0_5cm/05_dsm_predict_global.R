# ══════════════════════════════════════════════════════════════════════════════
# 05 (package) -- the global 250 m map of SOC stock, 0-5 cm, with dsm_predict()
#
# WHAT IT MAKES. The final model (the newest final_* run, its 10 seeds) over
# the whole 250 m grid, 63,721 x 160,298 cells, in native units (t/ha):
#
#   ensemble_median, _mean, _sd, _mad, _min, _max   over the seeds
#   smeared_mean_<source>                 the conditional mean (Duan)
#   pi90_constant_* / pi90_level_di_*     90% conformal intervals, per source
#   di_<source>, aoa_<source>             dissimilarity index and AOA
#   valid_mask                            where the model predicted
#
# for two calibration sources: block CV (the tuning run's own folds) and
# kNNDM (folds at the distances the map predicts at; U1/U2). It replaced stage
# 05 (05_predict_spatial.R, tile by tile, with its 05a/05b/05c helpers) and
# stage 07's DI and AOA rasters, all removed on 2026-09-28. bands.csv, in the run's
# directory, says what every band is and calibration.csv every number that
# calibrated it.
#
# HOW LONG. T3's fifth run (commit 1d81e34) measured, on full-width rows near
# 47 N: 2 workers x 7 threads at 21,200 valid px/s -- ~33 h for the globe's
# ~2.3e9 valid pixels -- and 1 worker x 15 threads at 17,400 (~39 h). Each
# worker holds ~14.7 GB. This script takes 2 x 7 when two workers fit in the
# RAM dsm_predict() budgets (70% of what is free), else 1 x 15, which is then
# faster than one worker of 7 (11,600 px/s).
#
# RESUMABLE. The map is 249 units of 256 rows, read 32 rows at a time; a unit
# is finished when its record is written, after its files. If the run stops
# -- Windows restarting for an update, a power cut, Ctrl+C -- source this
# script again: it keeps every finished unit and maps the rest, and refuses
# to continue if anything that changes the numbers changed. For a run of a
# day and a half: pause Windows Update and set the power plan to never sleep.
#
# BEFORE IT MAPS: the disk's free space (below), the RAM (the work plan), and
# the probe -- 48 profiles through the whole chain, every seed's prediction
# against the one the final run stored; a mismatch stops the map before it
# starts.
#
# WRITES: outputs/.../spatial_prediction/.../dsm_predict/global_250m/ -- one
# GeoTIFF per band per unit (units/), one VRT per band over them, bands.csv,
# calibration.csv, prediction_manifest.csv, the workers' logs. T3 wrote ~42
# bytes per valid pixel over the 21 bands (DEFLATE): ~100 GB for the globe.
#
# SETTINGS (environment variables, all optional):
#   soc_global_run_id        the map's name, "global_250m"; another name, a new map
#   soc_global_threads       7 or 15 -- chosen from the free RAM when unset
#   soc_global_n_cores       15
#   soc_global_min_free_gb   150: the free disk the map needs to start (0 skips
#                            the check -- a decision, not a default)
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/05_dsm_predict_global.R")
# ══════════════════════════════════════════════════════════════════════════════

# The project root, found from where this file is: Rscript's --file, then the
# source() frame, then the working directory, climbing to R/cnn_architecture.R.
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
maps_dir     <- file.path(base("outputs", "spatial_prediction"), "dsm_predict")

run_id    <- env_chr("soc_global_run_id", "global_250m")
n_cores   <- env_int("soc_global_n_cores", 15L)
unit_rows <- 256L                # 249 units; a unit re-reads only its 14 halo rows
step_rows <- 32L                 # what T3 measured, and what two workers fit with

# ── the model and its calibration ─────────────────────────────────────────────
final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir  <- file.path(final_base, final_run_id)
summ       <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
cfg_row    <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)

# Both sources, or no map: a day and a half of mapping without the kNNDM
# intervals is not the map this script promises.
knndm_dir <- NULL
for (cand in file.path(tuning_base, c("soc_0_5cm_knndm_cfg003", "soc_0_5cm_design_knndm"))) {
  if (dir.exists(cand) &&
      !is.null(suppressMessages(cv_residuals_for_config(cand, cfg_row, required = FALSE)))) {
    knndm_dir <- cand
    break
  }
}
if (is.null(knndm_dir)) {
  stop("No kNNDM tuning run with cross-validated residuals for ", config_id, " under ",
       tuning_base, " (looked for soc_0_5cm_knndm_cfg003 and soc_0_5cm_design_knndm). ",
       "Run _u2_knndm_residuals.R first.", call. = FALSE)
}
calibration <- c(block = tuning_dir, knndm = knndm_dir)

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
n_row <- as.integer(terra::nrow(g250)); n_col <- as.integer(terra::ncol(g250))

# ── the layout: 2 x 7 when two workers fit, else 1 x 15 ──────────────────────
#
# dsm_predict() budgets 70% of the RAM free when it starts, and two workers
# of 7 threads need ~2 x 14.6 GB (the RAM model; T3 measured 14.7). When they
# do not fit, the work plan keeps one worker -- and one worker of 7 threads
# (11,600 px/s) is slower than one of 15 (17,400).
ram_free <- tryCatch(ps::ps_system_memory()$avail / 1e9, error = function(e) NA_real_)
two_fit  <- is.finite(ram_free) && 0.7 * ram_free >= 2 * 14.6
threads  <- env_int("soc_global_threads", if (two_fit) 7L else 15L)

# ── the disk ──────────────────────────────────────────────────────────────────
#
# ~42 bytes per valid pixel (T3), ~2.3e9 valid pixels: ~100 GB, and 150 GB
# free to start, scaled by the share of units not yet finished when resuming.
# A check that cannot measure stops: set soc_global_min_free_gb=0 to decide
# otherwise.
min_free <- env_int("soc_global_min_free_gb", 150L, min = 0L)
n_units  <- as.integer(ceiling(n_row / unit_rows))
n_done   <- length(Sys.glob(file.path(maps_dir, run_id, "units", "*", "done.rds")))
need_gb  <- min_free * max(0, n_units - n_done) / n_units
if (need_gb > 0) {
  create_output_dirs(maps_dir)
  drive <- substr(normalizePath(maps_dir, winslash = "/"), 1L, 3L)
  free_gb <- tryCatch({
    v <- as.numeric(ps::ps_disk_usage(drive)[["available"]])
    if (length(v) == 1L) v / 1e9 else NA_real_
  }, error = function(e) NA_real_)
  if (!is.finite(free_gb)) {
    stop("Could not read the free space on ", drive, " (ps::ps_disk_usage()). ",
         "Set soc_global_min_free_gb=0 to map without this check.", call. = FALSE)
  }
  if (free_gb < need_gb) {
    stop(sprintf("%.0f GB free on %s; the map needs ~%.0f GB to finish (%d of %d unit(s) left). ",
                 free_gb, drive, need_gb, n_units - n_done, n_units),
         "Free space, or set output_dir elsewhere.", call. = FALSE)
  }
}

message("\n", strrep("=", 78))
message("05 (package) -- the global 250 m map with dsm_predict()")
message(strrep("=", 78))
message("  final run   : ", final_dir, "  (", config_id, ", ", length(summ$seeds), " seeds)")
message("  calibration : block = ", basename(tuning_dir), " | knndm = ", basename(knndm_dir))
message("  grid        : ", format(n_row, big.mark = ","), " x ", format(n_col, big.mark = ","),
        " | ", n_units, " units of ", unit_rows, " rows in steps of ", step_rows,
        if (n_done > 0L) sprintf(" | %d already finished", n_done) else "")
message(sprintf("  layout      : %d thread(s) a worker on %d cores | RAM free %.1f GB%s",
                threads, n_cores, ram_free,
                if (two_fit) "" else " -- two workers of 7 do not fit, so one of 15"))
message("  output      : ", file.path(maps_dir, run_id))
message(strrep("=", 78), "\n")

m <- dsm_predict(final_dir, data, rasters = rt, qc_table = qc_path, calibration = calibration,
                 n_cores = n_cores, threads_per_worker = threads,
                 unit_rows = unit_rows, step_rows = step_rows, probe = TRUE,
                 output_dir = maps_dir, run_id = run_id)

message("\nThe map: one VRT per band in ", m$run_dir, " (bands.csv says what each is).")
