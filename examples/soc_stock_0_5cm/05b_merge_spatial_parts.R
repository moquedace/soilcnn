
# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R and in R/load_all.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/load_all.R.
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
    if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("terra", "dplyr", "readr", "tibble", "purrr", "stringr")
install_load_pkg(pkg)

rm(list = setdiff(ls(), "project_root"))  # keep the root found above
gc()

options(width = 200)

setwd(project_root)

source(file.path(project_root, "R", "utils.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 05b — Mosaics the 2D tiles produced by 05_predict_spatial.R
#
# Finds every file in raster/parts_2d/ with the suffix
# _rXXXofYYY_cXXXofZZZ.tif, groups them by layer (median, sd, etc.) and builds
# the final raster with terra::merge. Checks that the extent of the mosaic
# matches the original template.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"
config_id    <- "auto"
final_run_id <- "latest"

# ── Resolve config_id / final_run_id ──────────────────────────────────────────

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (identical(final_run_id, "latest")) {
  # By time, and only a FINISHED run -- see 05a_run_parallel.R for why the
  # alphabetical version handed out half-built directories.
  final_run_id <- latest_run_dir(
    final_model_base, prefix = "final_",
    require_file = file.path("comparison", "final_run_summary.rds"),
    label = "final_run_id")
}

if (identical(config_id, "auto")) {
  summary_path <- file.path(final_model_base, final_run_id, "comparison",
                            "final_run_summary.rds")
  if (!file.exists(summary_path)) stop("final_run_summary.rds not found.")
  config_id <- selected_config_id(readRDS(summary_path), final_run_id)
  message("config_id: ", config_id)
}

# ── Paths ──────────────────────────────────────────────────────────────────────

output_dir        <- file.path(project_root, "outputs", "spatial_prediction",
                               "soc_stock_modeling", target_label, config_id)
parts_dir         <- file.path(output_dir, "raster", "parts_2d")
output_raster_dir <- file.path(output_dir, "raster")
output_log_dir    <- file.path(output_dir, "log")

if (!dir.exists(parts_dir)) stop("2D parts folder not found: ", parts_dir)

# ── List the files by layer ────────────────────────────────────────────────────

all_parts <- list.files(parts_dir, pattern = "_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$",
                        full.names = TRUE)
if (length(all_parts) == 0) stop("No 2D tile found in: ", parts_dir)

message(sprintf("\n%d 2D tiles found in: %s", length(all_parts), parts_dir))

# Extracts the layer suffix (the part of the name between config_id_ and _rXXX)
layer_pattern <- paste0("^", target_label, "_", config_id, "_(.+)_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$")
file_suffixes <- unique(sub(layer_pattern, "\\1", basename(all_parts)))

message("Layers detected: ", paste(file_suffixes, collapse = ", "))

# ── Template of the full raster ────────────────────────────────────────────────

raster_table_file <- file.path(metadata_dir, "raster_table_used.csv")
if (!file.exists(raster_table_file)) stop("raster_table_used.csv not found.")
raster_table   <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
full_template  <- terra::rast(raster_table$raster_file[1])

# ── Per-layer merge function ───────────────────────────────────────────────────

merge_layer <- function(suffix) {
  pat <- paste0("^", target_label, "_", config_id, "_", suffix,
                "_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$")
  parts <- sort(list.files(parts_dir, pattern = pat, full.names = TRUE))

  if (length(parts) == 0) {
    stop("No tile for layer '", suffix, "' under ", parts_dir, call. = FALSE)
  }

  # THE FILENAMES DECLARE THE GRID, AND THE COUNT MUST MATCH IT. A 2x2 run
  # that lost a worker leaves three tiles; terra::merge() mosaics three tiles
  # without complaint and the hole is NA that looks like ocean. Tiles from two
  # different grids in one directory (a 1x1 left beside a 2x2) would merge
  # into a map that is right where they overlap and arbitrary where they do
  # not. Both are refused here, by name.
  g <- regmatches(basename(parts),
                  regexec("_r([0-9]+)of([0-9]+)_c([0-9]+)of([0-9]+)[.]tif$", basename(parts)))
  grids <- unique(vapply(g, function(m) paste0(m[3], "x", m[5]), character(1)))
  if (length(grids) != 1L) {
    stop("[", suffix, "] tiles from more than one shard grid are present (",
         paste(grids, collapse = ", "), ") in ", parts_dir,
         ".\n  Move the tiles of the grid you do not want out of the way.",
         call. = FALSE)
  }
  n_expected <- as.integer(g[[1]][3]) * as.integer(g[[1]][5])
  if (length(parts) != n_expected) {
    ids <- vapply(g, function(m) sprintf("r%s_c%s", m[2], m[4]), character(1))
    stop("[", suffix, "] the filenames declare a ", grids, " grid (", n_expected,
         " tiles) but ", length(parts), " tile(s) are present: ",
         paste(ids, collapse = ", "), ".\n  A shard did not finish; see the ",
         "worker logs, then re-run 05a (it resumes).", call. = FALSE)
  }

  message(sprintf("\n[%s] Mosaicking %d tiles (%s grid)...", suffix, length(parts), grids))
  t_start <- Sys.time()

  out_file <- file.path(output_raster_dir,
                        paste0(target_label, "_", config_id, "_", suffix, ".tif"))
  if (file.exists(out_file)) file.remove(out_file)

  datatype <- if (grepl("mask", suffix)) "INT1U" else "FLT4S"
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L

  rast_list <- lapply(parts, terra::rast)
  merged    <- terra::merge(
    terra::sprc(rast_list),
    filename  = out_file,
    overwrite = TRUE,
    datatype  = datatype,
    gdal      = c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
                  "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
  )

  dt <- Sys.time() - t_start
  message(sprintf("  -> %s  (%.1f %s)", basename(out_file),
                  as.numeric(dt), units(dt)))

  # Checks extent and dimensions
  # The file is already written when this is known; an error here, not a
  # warning, so nothing downstream (07, 99b) picks up a mosaic on the wrong
  # grid because the script "finished".
  if (!terra::compareGeom(merged, full_template, stopOnError = FALSE)) {
    stop("[", suffix, "] the mosaic's geometry differs from the template raster ",
         "-- the workers predicted on another grid. ", basename(out_file),
         " was written but must not be used.", call. = FALSE)
  }
  message("  Geometry OK.")

  out_file
}

merged_files <- purrr::map(file_suffixes, merge_layer)

# ── Sanity check on the median map ────────────────────────────────────────────

median_file <- merged_files[[which(file_suffixes == "ensemble_median_ton_ha")]]
if (!is.null(median_file) && file.exists(median_file)) {
  message("\n── Sanity check (full mosaic) ──────────────────────────────────")
  r_med   <- terra::rast(median_file)
  gstats  <- terra::global(r_med, c("min", "mean", "max"), na.rm = TRUE)
  g_mean  <- gstats[1, "mean"]
  g_max   <- gstats[1, "max"]

  message(sprintf("  Median map: min %.2f | mean %.2f | max %.2f  ton_ha",
                  gstats[1, "min"], g_mean, g_max))

  if (is.na(g_mean) || g_mean < 1 || g_mean > 200) {
    message("  [ATTENTION] Global mean outside the expected range [1, 200] -- check the mosaic.")
  } else {
    message("  [OK] Global mean within the plausible range.")
  }
  if (!is.na(g_max) && g_max > 1000) {
    message(sprintf("  [WARN] Max %.0f > 1000 -- check extreme pixels.", g_max))
  }
}

message("\n── 2D merge complete ─────────────────────────────────────────────")
message("  Final rasters: ", output_raster_dir)
message("  Part tiles: ", parts_dir, " (can be deleted after verification)")
