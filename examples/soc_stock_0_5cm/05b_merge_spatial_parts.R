source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("terra", "dplyr", "readr", "tibble", "purrr")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 05b — Merge per-worker spatial-prediction tiles
#
# Run this ONCE after every worker launched by 05_predict_spatial.R (with
# n_workers > 1) has finished. Each worker wrote a tile cropped to its own
# disjoint row range under outputs/.../raster/parts/; this script mosaics the
# tiles (per layer) into the final full-extent rasters in outputs/.../raster/,
# then runs the global plausibility checks that 05 deferred (a per-worker
# strip landing entirely on ocean/no-data legitimately has zero valid pixels,
# so those checks are only meaningful once on the assembled whole).
# ══════════════════════════════════════════════════════════════════════════════

# ── Settings — MUST match the values used to launch the workers in 05 ─────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"
config_id    <- "auto"
final_run_id <- "latest"

plausible_median_range <- c(1, 200)
plausible_hard_max     <- 1000

# ── Paths (mirrors 05_predict_spatial.R's resolution) ──────────────────────────

final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (identical(final_run_id, "latest")) {
  run_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^final_", run_dirs)]
  if (length(run_dirs) == 0) stop("No final model runs found in: ", final_model_base)
  final_run_id <- sort(run_dirs, decreasing = TRUE)[1]
  message("final_run_id resolved to: ", final_run_id)
}

final_run_dir <- file.path(final_model_base, final_run_id)

if (identical(config_id, "auto")) {
  tmp_summary_path <- file.path(final_run_dir, "comparison", "final_run_summary.rds")
  if (!file.exists(tmp_summary_path))
    stop("Nao foi possivel resolver config_id='auto': arquivo nao encontrado: ",
         tmp_summary_path)
  config_id <- readRDS(tmp_summary_path)$selected_cfgs$config_id[1]
  message("config_id resolved to: ", config_id)
}

output_dir        <- file.path(project_root, "outputs", "spatial_prediction",
                               "soc_stock_modeling", target_label, config_id)
output_raster_dir <- file.path(output_dir, "raster")
output_log_dir    <- file.path(output_dir, "log")
parts_dir         <- file.path(output_raster_dir, "parts")

if (!dir.exists(parts_dir)) stop("Parts directory not found: ", parts_dir,
                                 " — did any worker run with n_workers > 1?")

# ── Layers to merge ─────────────────────────────────────────────────────────────

layers <- tibble::tibble(
  key         = c("median", "mean", "sd", "mad", "min", "max", "mask"),
  file_suffix = c("ensemble_median_ton_ha", "ensemble_mean_ton_ha",
                  "ensemble_sd_ton_ha", "ensemble_mad_ton_ha",
                  "ensemble_min_ton_ha", "ensemble_max_ton_ha", "valid_mask"),
  datatype    = c("FLT4S", "FLT4S", "FLT4S", "FLT4S", "FLT4S", "FLT4S", "INT1U")
)

gdal_opts <- function(datatype) {
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L
  c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
    "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
}

merge_layer <- function(file_suffix, datatype) {
  pattern <- paste0("^", target_label, "_", config_id, "_", file_suffix, "_part.*\\.tif$")
  part_files <- list.files(parts_dir, pattern = pattern, full.names = TRUE)
  if (length(part_files) == 0) {
    message("  [SKIP] No part files found for layer '", file_suffix, "' (pattern: ", pattern, ")")
    return(NA_character_)
  }
  message(sprintf("  Merging %d part(s) for layer '%s'...", length(part_files), file_suffix))

  out_file <- file.path(output_raster_dir, paste0(target_label, "_", config_id, "_", file_suffix, ".tif"))
  if (file.exists(out_file)) file.remove(out_file)

  coll <- terra::sprc(lapply(part_files, terra::rast))
  terra::merge(coll, filename = out_file, overwrite = TRUE,
              datatype = datatype, gdal = gdal_opts(datatype))
  message("    -> ", out_file)
  out_file
}

message("\n── Merging worker tiles ─────────────────────────────────────────────")
written_files <- purrr::map2_chr(layers$file_suffix, layers$datatype, merge_layer)
names(written_files) <- layers$key

missing_layers <- layers$key[is.na(written_files)]
if (length(missing_layers) > 0) {
  stop("No part files found for layer(s): ", paste(missing_layers, collapse = ", "),
       " — did all workers finish before running this script?")
}

# ── Global sanity checks on the assembled map (deferred from 05) ──────────────
# Now meaningful: a per-worker strip could legitimately be all-ocean, but the
# FULL mosaic should not be.

f_median <- written_files[["median"]]
f_mask   <- written_files[["mask"]]

message("\n── Sanity checks (on the merged map) ───────────────────────────────")

r_median <- terra::rast(f_median)
r_mask   <- terra::rast(f_mask)

n_valid_total <- terra::global(r_mask, "sum", na.rm = TRUE)[1, 1]
n_cell        <- terra::ncell(r_mask)
global_mean_med <- terra::global(r_median, "mean", na.rm = TRUE)[1, 1]
global_min      <- terra::global(r_median, "min",  na.rm = TRUE)[1, 1]
global_max      <- terra::global(r_median, "max",  na.rm = TRUE)[1, 1]

message(sprintf("  Valid pixels         : %s (%.2f%% of grid)",
                format(n_valid_total, big.mark = ","), 100 * n_valid_total / n_cell))
message(sprintf("  Median map mean      : %.2f %s", global_mean_med, target_unit))
message(sprintf("  Median map range     : %.2f – %.2f %s", global_min, global_max, target_unit))

sanity_ok <- TRUE
if (is.na(n_valid_total) || n_valid_total == 0L) {
  sanity_ok <- FALSE
  message("  [FAIL] No valid pixels in the merged map.")
}
if (is.finite(global_mean_med) &&
    (global_mean_med < plausible_median_range[1] || global_mean_med > plausible_median_range[2])) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] Median-map mean %.2f outside plausible range [%g, %g].",
                  global_mean_med, plausible_median_range[1], plausible_median_range[2]))
}
if (is.finite(global_max) && global_max > plausible_hard_max) {
  message(sprintf("  [WARN] Max %.0f exceeds %g %s — inspect the high tail.",
                  global_max, plausible_hard_max, target_unit))
}
if (!sanity_ok) {
  message("  Sanity checks FAILED. Merged rasters were kept on disk for inspection ",
          "(not deleted automatically, unlike the per-worker checks in 05) — investigate before using them.")
} else {
  message("  All hard checks passed.")
}

# ── Combine per-worker block logs into one manifest, for reference ────────────

block_logs <- list.files(output_log_dir, pattern = "^prediction_block_summary_part.*\\.csv$",
                         full.names = TRUE)
if (length(block_logs) > 0) {
  combined <- dplyr::bind_rows(lapply(block_logs, readr::read_csv2, show_col_types = FALSE))
  safe_write_csv2(combined, file.path(output_log_dir, "prediction_block_summary_merged.csv"))
  message("\nCombined block log written: ",
          file.path(output_log_dir, "prediction_block_summary_merged.csv"))
}

message("\n── Merge complete ───────────────────────────────────────────────────")
message("  Final rasters written to: ", output_raster_dir)
message("  Worker tiles kept in:     ", parts_dir, " (delete manually once verified)")
