source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "terra",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "stringr",
  "matrixStats"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "metrics.R"))
source(file.path(project_root, "R", "cnn_architecture.R"))
source(file.path(project_root, "R", "tune_grid.R"))
source(file.path(project_root, "R", "train_cnn.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 05 — Spatial prediction (block-streaming, seed ensemble)
#
# Produces a wall-to-wall map of the target from the final CNN ensemble.
#
# Design principles (mirrors the robust streaming logic of the cnn09pp project,
# generalised to a single configurable window and a seed ensemble):
#
#   • BLOCK STREAMING of predictors. Never loads the full 187-layer stack into
#     RAM. Reads a horizontal strip of rows at a time, just wide enough to
#     extract the patch window for the centre rows in that strip. Scales to any
#     raster size / resolution. Only the single-band OUTPUT layers are held in
#     memory (cheap), so peak RAM is bounded by one strip of predictors.
#
#   • SCALED patches. 02_extract_patches.R now scales vals_matrix in-place
#     before building patches (z-score for continuous, /100 for percentages,
#     identity for dummies). Spatial prediction MUST apply the SAME transform
#     to every strip of raster values before passing patches to the model.
#     Parameters come from predictor_scaling.csv (training-split statistics
#     computed in script 01). NOT applying scaling here = wrong map.
#
#   • EXACT patch layout match. The (row, col) offset ordering and the reshape
#     to [N, C, H, W] replicate 02_extract_patches.R byte-for-byte, so patches
#     reach the model in the same spatial orientation they were trained on.
#
#   • SEED ENSEMBLE with a defensible aggregator. Each seed model predicts in
#     log1p space; we back-transform with expm1 and aggregate ACROSS SEEDS.
#       - MEDIAN is the primary map: it is invariant to the monotone expm1
#         (median(expm1(z)) = expm1(median(z))), robust to a single divergent
#         seed, and consistent with what SmoothL1 learns (a conditional median).
#       - MEAN is also written for comparison (note: mean in native space is
#         inflated vs. log space by Jensen's inequality — that gap is itself
#         diagnostic).
#       - Uncertainty = ensemble DISAGREEMENT (epistemic, from init/training
#         only). This is NOT a full predictive interval. SD and the more robust
#         MAD are both written; MIN/MAX give the raw envelope.
#
#   • SANITY GUARDS. Hard checks on channel alignment before predicting, and
#     plausibility checks on the output distribution after, to refuse to write a
#     nonsensical map silently.
# ══════════════════════════════════════════════════════════════════════════════

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

config_id    <- "cfg_012"     # winning config from 04_final_model.R
final_run_id <- "latest"      # "latest" -> most recent final_* run, or an explicit id
seeds        <- c(42L, 123L, 456L, 789L, 2025L)  # ensemble members (script 04)

# Which aggregation is reported as the headline map in the summary. Both median
# and mean rasters are always written; this only flags the recommended one.
ensemble_center <- "median"   # "median" (recommended) or "mean"

# Streaming / GPU parameters
output_block_rows <- 40L      # raster rows processed per strip (RAM vs. speed)
batch_size        <- 4096L    # patches per GPU forward pass

# Plausibility bounds for the post-prediction sanity check (native units).
# Wide on purpose — only meant to catch gross failures (e.g. unscaled inputs).
plausible_median_range <- c(1, 200)   # global median of the map should fall here
plausible_hard_max     <- 1000        # warn if any pixel exceeds this

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Paths ─────────────────────────────────────────────────────────────────────

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
patch_dir        <- file.path(project_root, "outputs", "patches",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

raster_table_file      <- file.path(metadata_dir, "raster_table_used.csv")
predictor_scaling_file <- file.path(metadata_dir, "predictor_scaling.csv")
patch_manifest_file    <- file.path(patch_dir, "patch_manifest.rds")

# Resolve "latest" final run
if (identical(final_run_id, "latest")) {
  run_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^final_", run_dirs)]
  if (length(run_dirs) == 0) stop("No final model runs found in: ", final_model_base)
  final_run_id <- sort(run_dirs, decreasing = TRUE)[1]
  message("final_run_id resolved to: ", final_run_id)
}

final_run_dir <- file.path(final_model_base, final_run_id)
model_dir     <- file.path(final_run_dir, config_id, "models")
summary_file  <- file.path(final_run_dir, "comparison", "final_run_summary.rds")

output_dir <- file.path(project_root, "outputs", "spatial_prediction",
                        "soc_stock_modeling", target_label, config_id)
output_raster_dir <- file.path(output_dir, "raster")
output_log_dir    <- file.path(output_dir, "log")

create_output_dirs(c(output_dir, output_raster_dir, output_log_dir))

# ── Validate inputs exist ─────────────────────────────────────────────────────

for (f in c(raster_table_file, predictor_scaling_file, summary_file)) {
  if (!file.exists(f)) stop("Required input not found: ", f)
}
if (!dir.exists(model_dir)) stop("Model directory not found: ", model_dir)

# ── Load predictor scaling parameters ────────────────────────────────────────

predictor_scaling <- readr::read_csv2(predictor_scaling_file, show_col_types = FALSE)

temperature_min_valid_celsius <- -100   # must match script 01

# Pre-compute vectors for fast per-strip application (channel order = predictor_cols order,
# validated below after raster_table is loaded).
# Applied to each strip of raw raster values before patch extraction.
apply_predictor_scaling <- function(mat, pred_names, scale_method,
                                    scale_center, scale_factor,
                                    temp_threshold) {
  for (i in seq_len(ncol(mat))) {
    x <- mat[, i]

    if (grepl("surface_temperature_celsius$", pred_names[i])) {
      x[!is.na(x) & is.finite(x) & x <= temp_threshold] <- NA_real_
    }

    if (scale_method[i] == "zscore_train") {
      x <- (x - scale_center[i]) / scale_factor[i]
    } else if (scale_method[i] == "percentage_0_100_to_0_1") {
      x[!is.na(x) & is.finite(x) & (x < 0 | x > 100)] <- NA_real_
      x <- x / 100
    }
    # "none_dummy_0_1": no transformation

    mat[, i] <- x
  }
  mat
}

# ── Load model config from the final-run summary ──────────────────────────────

final_summary <- readRDS(summary_file)

if (!config_id %in% final_summary$selected_cfgs$config_id) {
  stop("config_id '", config_id, "' not in final run summary. Available: ",
       paste(final_summary$selected_cfgs$config_id, collapse = ", "))
}

cfg <- dplyr::filter(final_summary$selected_cfgs, config_id == !!config_id)

window_sizes <- cfg$window_sizes[[1]]
if (length(window_sizes) != 1L) {
  stop("This prediction script handles single-branch configs only. ",
       "cfg ", config_id, " has window_sizes = ",
       paste(window_sizes, collapse = ", "),
       ". A dual-branch streaming variant is required for >1 window.")
}
window_size <- window_sizes[1]
half_w      <- (window_size - 1L) %/% 2L

message("\nConfig ", config_id,
        " | window ", window_size, "x", window_size,
        " | conv ", paste(cfg$conv_channels[[1]], collapse = "_"),
        " | embed ", cfg$embedding_dim,
        " | gate ", cfg$gate_type)

# ── Canonical predictor order + raster files ──────────────────────────────────
# raster_table_used.csv stores predictors in the EXACT order script 01/02 used
# (alphabetical), with their source file paths. This is the single source of
# truth for channel alignment.

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
if (!all(c("raster_file", "predictor") %in% names(raster_table))) {
  stop("raster_table_used.csv must contain 'raster_file' and 'predictor'.")
}

predictor_cols <- raster_table$predictor
n_channels     <- length(predictor_cols)
raster_files   <- raster_table$raster_file

missing_rasters <- raster_files[!file.exists(raster_files)]
if (length(missing_rasters) > 0) {
  print(missing_rasters)
  stop("Some predictor raster files no longer exist.")
}

# Align predictor_scaling to the EXACT channel order of raster_table / predictor_cols.
# Misalignment here = wrong scale applied to wrong channel = corrupted predictions.
predictor_scaling <- predictor_scaling |>
  dplyr::filter(predictor %in% predictor_cols) |>
  dplyr::arrange(match(predictor, predictor_cols))

if (!identical(predictor_scaling$predictor, predictor_cols)) {
  stop("predictor_scaling.csv channel order does not match raster_table_used.csv. ",
       "Re-check script 01 outputs.")
}

scale_method <- predictor_scaling$scaling_method
scale_center <- predictor_scaling$train_center
scale_factor <- predictor_scaling$train_scale

message("Predictor scaling loaded and aligned (", n_channels, " channels).")

# Cross-check against the patch manifest the model was actually trained on.
if (file.exists(patch_manifest_file)) {
  manifest <- readRDS(patch_manifest_file)
  manifest_predictors <- strsplit(manifest$predictor_cols_final, ";")[[1]]
  if (!identical(as.character(manifest_predictors), as.character(predictor_cols))) {
    stop("Predictor order in raster_table_used.csv does NOT match the patch ",
         "manifest the model was trained on. Channel alignment would be wrong. ",
         "Re-check 01/02 outputs.")
  }
  message("Channel alignment verified against patch manifest (", n_channels, " predictors).")
} else {
  message("WARNING: patch_manifest.rds not found — cannot cross-check channel order. ",
          "Proceeding with raster_table_used.csv order (", n_channels, " predictors).")
}

# ── Open raster stack (lazy) and check geometry ───────────────────────────────

rast_stack <- terra::rast(raster_files)
names(rast_stack) <- predictor_cols

geom_ok <- purrr::map_lgl(
  seq_len(terra::nlyr(rast_stack)),
  ~ terra::compareGeom(rast_stack[[1]], rast_stack[[.x]], stopOnError = FALSE)
)
if (!all(geom_ok)) {
  print(tibble::tibble(predictor = predictor_cols, geometry_ok = geom_ok) |>
          dplyr::filter(!geometry_ok))
  stop("Some predictor rasters do not share the same geometry.")
}

raster_template <- rast_stack[[1]]
r_nrow <- terra::nrow(rast_stack)
r_ncol <- terra::ncol(rast_stack)
n_cell <- terra::ncell(rast_stack)

message("Raster grid: ", r_nrow, " rows x ", r_ncol, " cols x ", n_channels, " layers")

# ── Load seed models ──────────────────────────────────────────────────────────

model_files <- file.path(model_dir, sprintf("seed%04d_best.pt", seeds))
have <- file.exists(model_files)
if (!any(have)) stop("No seed model files found in: ", model_dir)
if (!all(have)) {
  message("WARNING: missing seed files, using only the ", sum(have), " available:")
  print(model_files[!have])
}
seeds       <- seeds[have]
model_files <- model_files[have]
n_seeds     <- length(seeds)

message("\nLoading ", n_seeds, " seed model(s) for ", config_id, "...")
models <- purrr::map(seq_len(n_seeds), function(i) {
  m  <- build_cnn_from_config(cfg, n_channels)
  st <- torch::torch_load(model_files[i])
  m$load_state_dict(st)      # errors loudly if n_channels / architecture mismatch
  m$to(device = device)
  m$eval()
  m
})

# ── Helpers ───────────────────────────────────────────────────────────────────

#' Build a [n, C, w, w] patch array for the given centre pixels from a strip of
#' raster values read from rows [read_row_start .. read_row_start+read_nrows-1].
#'
#' Layout replicates 02_extract_patches.R EXACTLY:
#'   offsets = expand.grid(dr, dc) with dr (row offset) varying fastest;
#'   cells indexed row-major within the strip; reshape [n_pos, n, C] -> aperm ->
#'   [n, C, w, w] so dim3 = row offset, dim4 = col offset (PyTorch N,C,H,W).
#'
#' Returns list(arr, valid) where `valid` flags centres whose full window is
#' finite across all channels (centres with any NA/Inf are dropped from `arr`).
build_patch_array_block <- function(center_row, center_col,
                                    strip_values, read_row_start,
                                    n_cols, n_ch, w) {
  half <- (w - 1L) %/% 2L
  n    <- length(center_row)

  offsets <- expand.grid(dr = (-half):half, dc = (-half):half)
  n_pos   <- nrow(offsets)

  row_local <- center_row - read_row_start + 1L   # 1-based row within the strip

  # cell index into strip_values (row-major) for every (centre, offset)
  cell_mat <- matrix(0L, nrow = n, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cell_mat[, j] <- (row_local + offsets$dr[j] - 1L) * n_cols +
                     (center_col + offsets$dc[j])
  }

  all_cells <- as.vector(t(cell_mat))                  # pos-major within centre
  vals      <- strip_values[all_cells, , drop = FALSE] # (n*n_pos) x n_ch

  # validity: every position x every channel finite (base rowSums/colSums
  # accept logical matrices natively; TRUE counts as 1)
  row_finite  <- rowSums(!is.finite(vals)) == 0
  valid       <- colSums(matrix(row_finite, nrow = n_pos, ncol = n)) == n_pos

  if (!any(valid)) {
    return(list(arr = array(0, dim = c(0L, n_ch, w, w)), valid = valid))
  }

  valid_pos  <- which(valid)
  keep_rows  <- as.vector(vapply(
    valid_pos,
    function(p) ((p - 1L) * n_pos + 1L):(p * n_pos),
    integer(n_pos)
  ))
  vals_valid <- vals[keep_rows, , drop = FALSE]        # (n_valid*n_pos) x n_ch

  step1 <- array(vals_valid, dim = c(n_pos, length(valid_pos), n_ch)) # [pos, n, ch]
  step2 <- aperm(step1, c(2L, 3L, 1L))                                # [n, ch, pos]
  arr   <- array(step2, dim = c(length(valid_pos), n_ch, w, w))       # [n, ch, r, c]

  list(arr = arr, valid = valid)
}

#' Predict log1p values for a patch array with one model, batched on the GPU.
predict_one_model <- function(model, arr, device, batch_size) {
  n <- dim(arr)[1]
  if (n == 0L) return(numeric(0))
  out <- numeric(n)
  idx_groups <- split(seq_len(n), ceiling(seq_len(n) / batch_size))
  torch::with_no_grad({
    for (idx in idx_groups) {
      x <- torch::torch_tensor(arr[idx, , , , drop = FALSE],
                               dtype = torch::torch_float(), device = device)
      p <- model(x)                       # single-branch forward -> [b, 1], log1p
      out[idx] <- as.numeric(p$squeeze(2L)$to(device = "cpu"))
    }
  })
  out
}

# ── Output accumulators (single-band, row-major over all cells) ───────────────

out_median <- rep(NA_real_, n_cell)
out_mean   <- rep(NA_real_, n_cell)
out_sd     <- rep(NA_real_, n_cell)
out_mad    <- rep(NA_real_, n_cell)
out_min    <- rep(NA_real_, n_cell)
out_max    <- rep(NA_real_, n_cell)
out_mask   <- rep(0L,       n_cell)   # 1 = predicted, 0 = not (border/NA)

# ── Block-streaming prediction loop ───────────────────────────────────────────

block_starts <- seq(1L, r_nrow, by = output_block_rows)
block_log    <- vector("list", length(block_starts))

t0 <- proc.time()[["elapsed"]]
terra::readStart(rast_stack)

for (b in seq_along(block_starts)) {

  out_row_start <- block_starts[b]
  out_nrows     <- min(output_block_rows, r_nrow - out_row_start + 1L)
  out_row_end   <- out_row_start + out_nrows - 1L

  # centre rows in this block that have a full window inside the raster
  center_rows <- intersect(out_row_start:out_row_end,
                           (half_w + 1L):(r_nrow - half_w))

  n_req <- 0L; n_val <- 0L

  if (length(center_rows) > 0L) {
    read_row_start <- min(center_rows) - half_w
    read_row_end   <- max(center_rows) + half_w
    read_nrows     <- read_row_end - read_row_start + 1L

    strip_values <- terra::values(rast_stack, row = read_row_start,
                                  nrows = read_nrows, mat = TRUE)
    strip_values <- apply_predictor_scaling(strip_values, predictor_cols,
                                            scale_method, scale_center,
                                            scale_factor,
                                            temperature_min_valid_celsius)

    center_cols <- (half_w + 1L):(r_ncol - half_w)
    grid <- expand.grid(center_row = center_rows, center_col = center_cols)
    n_req <- nrow(grid)

    # process centres in chunks to bound the patch-array size
    chunk_id   <- ceiling(seq_len(n_req) / max(batch_size, 1L))
    chunk_list <- split(seq_len(n_req), chunk_id)

    for (ci in chunk_list) {
      cr <- grid$center_row[ci]
      cc <- grid$center_col[ci]

      pb <- build_patch_array_block(cr, cc, strip_values, read_row_start,
                                    r_ncol, n_channels, window_size)

      # global cell index (row-major) for ALL centres in this chunk
      cell_global <- (cr - 1L) * r_ncol + cc
      out_mask[cell_global] <- as.integer(pb$valid)

      if (dim(pb$arr)[1] == 0L) next

      # predict log1p with each seed -> [n_valid, n_seeds] native
      preds_native <- matrix(NA_real_, nrow = dim(pb$arr)[1], ncol = n_seeds)
      for (s in seq_len(n_seeds)) {
        plog <- predict_one_model(models[[s]], pb$arr, device, batch_size)
        preds_native[, s] <- pmax(expm1(plog), 0)
      }

      cell_valid <- cell_global[pb$valid]

      if (n_seeds >= 2L) {
        out_median[cell_valid] <- matrixStats::rowMedians(preds_native)
        out_mean[cell_valid]   <- rowMeans(preds_native)
        out_sd[cell_valid]     <- matrixStats::rowSds(preds_native)
        out_mad[cell_valid]    <- matrixStats::rowMads(preds_native)   # constant 1.4826
        out_min[cell_valid]    <- matrixStats::rowMins(preds_native)
        out_max[cell_valid]    <- matrixStats::rowMaxs(preds_native)
      } else {
        v <- preds_native[, 1]
        out_median[cell_valid] <- v
        out_mean[cell_valid]   <- v
        out_sd[cell_valid]     <- 0
        out_mad[cell_valid]    <- 0
        out_min[cell_valid]    <- v
        out_max[cell_valid]    <- v
      }
      n_val <- n_val + length(cell_valid)
    }
    rm(strip_values); gc()
  }

  block_log[[b]] <- tibble::tibble(
    block = b, row_start = out_row_start, row_end = out_row_end,
    n_centers = n_req, n_valid = n_val,
    n_invalid = n_req - n_val
  )

  if (b == 1L || b %% 10L == 0L || b == length(block_starts)) {
    el <- proc.time()[["elapsed"]] - t0
    message(sprintf("Block %d/%d | rows %d-%d | valid %s | %.0fs elapsed",
                    b, length(block_starts), out_row_start, out_row_end,
                    format(n_val, big.mark = ","), el))
  }
}

terra::readStop(rast_stack)
rm(models); gc()

total_time <- proc.time()[["elapsed"]] - t0
n_valid_total <- sum(out_mask == 1L)

# ── Sanity checks (refuse to write nonsense silently) ─────────────────────────

med_vals  <- out_median[out_mask == 1L]
global_med <- stats::median(med_vals, na.rm = TRUE)
global_max <- max(med_vals, na.rm = TRUE)
n_nonfinite <- sum(!is.finite(med_vals))

message("\n── Sanity checks ─────────────────────────────────────────────────")
message(sprintf("  Valid pixels        : %s (%.1f%% of grid)",
                format(n_valid_total, big.mark = ","), 100 * n_valid_total / n_cell))
message(sprintf("  Global median (map) : %.2f %s", global_med, target_unit))
message(sprintf("  Global max (map)    : %.2f %s", global_max, target_unit))

sanity_ok <- TRUE
if (n_valid_total == 0L) {
  sanity_ok <- FALSE
  message("  [FAIL] No valid pixels were predicted.")
}
if (n_nonfinite > 0L) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] %d non-finite predictions among valid pixels.", n_nonfinite))
}
if (is.finite(global_med) &&
    (global_med < plausible_median_range[1] || global_med > plausible_median_range[2])) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] Global median %.2f outside plausible range [%g, %g]. ",
                  global_med, plausible_median_range[1], plausible_median_range[2]),
          "Likely an input-scaling or alignment problem — NOT writing rasters.")
}
if (is.finite(global_max) && global_max > plausible_hard_max) {
  message(sprintf("  [WARN] Max %.0f exceeds %g %s — inspect the high tail.",
                  global_max, plausible_hard_max, target_unit))
}
if (!sanity_ok) {
  stop("Sanity checks failed. Rasters were NOT written. See messages above.")
}
message("  All hard checks passed.")

# ── Write output rasters ──────────────────────────────────────────────────────

write_layer <- function(values_vec, band_name, file_suffix, datatype = "FLT4S") {
  r <- terra::rast(raster_template)
  names(r) <- band_name
  terra::values(r) <- values_vec
  f <- file.path(output_raster_dir,
                 paste0(target_label, "_", config_id, "_", file_suffix, ".tif"))
  if (file.exists(f)) file.remove(f)
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L
  opts <- c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
            "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
  terra::writeRaster(r, f, overwrite = TRUE, datatype = datatype, gdal = opts)
  f
}

message("\nWriting rasters...")
f_median <- write_layer(out_median, "soc_pred_median_ton_ha", "ensemble_median_ton_ha")
f_mean   <- write_layer(out_mean,   "soc_pred_mean_ton_ha",   "ensemble_mean_ton_ha")
f_sd     <- write_layer(out_sd,     "soc_uncert_sd_ton_ha",   "ensemble_sd_ton_ha")
f_mad    <- write_layer(out_mad,    "soc_uncert_mad_ton_ha",  "ensemble_mad_ton_ha")
f_min    <- write_layer(out_min,    "soc_pred_min_ton_ha",    "ensemble_min_ton_ha")
f_max    <- write_layer(out_max,    "soc_pred_max_ton_ha",    "ensemble_max_ton_ha")
f_mask   <- write_layer(out_mask,   "valid_patch_mask",       "valid_mask", datatype = "INT1U")

# ── Metadata / manifests ──────────────────────────────────────────────────────

block_summary <- dplyr::bind_rows(block_log)
safe_write_csv2(block_summary, file.path(output_log_dir, "prediction_block_summary.csv"))

prediction_config <- tibble::tibble(
  target_label = target_label, target_unit = target_unit,
  config_id = config_id, final_run_id = final_run_id,
  window_size = window_size, n_channels = n_channels,
  n_seeds = n_seeds, seeds = paste(seeds, collapse = ";"),
  ensemble_center = ensemble_center,
  input_scaling = "predictor_scaling.csv_zscore_percentage_dummy",
  back_transform = "expm1_then_clip0",
  aggregation_space = "per_seed_expm1_then_aggregate_across_seeds",
  output_block_rows = output_block_rows, batch_size = batch_size,
  r_nrow = r_nrow, r_ncol = r_ncol, n_cell = n_cell,
  n_valid = n_valid_total, valid_fraction = n_valid_total / n_cell,
  global_median_ton_ha = global_med, global_max_ton_ha = global_max,
  runtime_min = total_time / 60,
  device = device$type, predicted_at = as.character(Sys.time())
)
safe_write_csv2(prediction_config, file.path(output_log_dir, "prediction_config.csv"))

# per-layer global stats (read back to confirm what was actually written)
raster_summary <- purrr::map_dfr(
  list(median = f_median, mean = f_mean, sd = f_sd, mad = f_mad,
       min = f_min, max = f_max, mask = f_mask),
  function(f) {
    r <- terra::rast(f)
    tibble::tibble(
      file = f,
      band = names(r),
      gmin  = terra::global(r, "min",  na.rm = TRUE)[1, 1],
      gmean = terra::global(r, "mean", na.rm = TRUE)[1, 1],
      gmax  = terra::global(r, "max",  na.rm = TRUE)[1, 1]
    )
  },
  .id = "layer"
)
safe_write_csv2(raster_summary, file.path(output_log_dir, "prediction_raster_summary.csv"))

safe_write_csv2(
  tibble::tibble(order = seq_len(n_channels), predictor = predictor_cols),
  file.path(output_log_dir, "predictor_order_used.csv")
)

# ── Report ────────────────────────────────────────────────────────────────────

message("\n── Spatial prediction complete ───────────────────────────────────")
message(sprintf("  Config / seeds   : %s / %d", config_id, n_seeds))
message(sprintf("  Valid pixels     : %s", format(n_valid_total, big.mark = ",")))
message(sprintf("  Runtime          : %.1f min", total_time / 60))
message(sprintf("  Headline map     : %s", ensemble_center))
message("\n  Central tendency (native %s), over valid pixels:", target_unit)
message(sprintf("    median map -> median %.2f | mean %.2f | max %.2f",
                stats::median(out_median[out_mask == 1L]),
                mean(out_median[out_mask == 1L]),
                max(out_median[out_mask == 1L])))
message(sprintf("    mean map   -> median %.2f | mean %.2f | max %.2f",
                stats::median(out_mean[out_mask == 1L]),
                mean(out_mean[out_mask == 1L]),
                max(out_mean[out_mask == 1L])))
message(sprintf("    (mean-map median exceeds median-map by %.2f %s = Jensen gap)",
                stats::median(out_mean[out_mask == 1L]) -
                  stats::median(out_median[out_mask == 1L]), target_unit))
message("\n  Ensemble disagreement (native %s, epistemic only):", target_unit)
message(sprintf("    SD  : median %.2f | max %.2f",
                stats::median(out_sd[out_mask == 1L]), max(out_sd[out_mask == 1L])))
message(sprintf("    MAD : median %.2f | max %.2f",
                stats::median(out_mad[out_mask == 1L]), max(out_mad[out_mask == 1L])))

message("\n  Rasters written to: ", output_raster_dir)
message("  Logs written to:    ", output_log_dir)
message("\n  RECOMMENDED map: ", basename(f_median),
        " (median is invariant to expm1, robust, matches the SmoothL1 target).")
