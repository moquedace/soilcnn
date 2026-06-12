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

config_id    <- "auto"        # "auto" -> rank-1 config from the final-run summary;
                              #           or set to an explicit id, e.g. "cfg_007"
final_run_id <- "latest"      # "latest" -> most recent final_* run, or an explicit id
seeds        <- c(42L, 123L, 456L, 789L, 2025L)  # ensemble members (script 04)

# Which aggregation is reported as the headline map in the summary. Both median
# and mean rasters are always written; this only flags the recommended one.
ensemble_center <- "median"   # "median" (recommended) or "mean"

# Streaming / GPU parameters
#
# Output is STREAMED to disk block-by-block (terra writeStart/writeValues/
# writeStop). The full-grid result is NEVER materialised in RAM — essential at
# fine resolution: a 250 m global grid has ~1.0e10 cells, so a single full-grid
# double vector needs ~82 GB and the 7 output layers together would need ~0.5 TB.
# Streaming bounds OUTPUT RAM to one block (tens of MB).
#
# The remaining RAM cost is the predictor STRIP (all C bands for the block's
# window), read once per block:
#   strip_bytes ≈ (output_block_rows + 2*half_w_max) * r_ncol * n_channels * 8
# max_strip_ram_gb auto-sizes output_block_rows to stay within that budget and
# avoids the std::bad_alloc that a too-large single strip would trigger at fine
# resolution (the same heap-fragmentation failure fixed in 02_extract_patches).
max_strip_ram_gb  <- 6        # per-strip predictor RAM budget in GB (NULL = use fixed rows below)
                              # IMPORTANT: pico de RAM real = 2 × strip, porque apply_predictor_scaling
                              # aciona copy-on-modify do R na primeira atribuicao mat[,i]<- dentro da funcao.
                              # Defina este valor <= RAM_livre / 2 para evitar OOM.
                              # Referencia para 187 bandas, ~149 k cols (1 row ≈ 223 MB):
                              #   6 GB → ~10 useful rows/block (strip 24 rows, pico ~12 GB) ← recomendado
                              #   8 GB → ~21 useful rows/block (strip 35 rows, pico ~16 GB)
                              #  16 GB → ~57 useful rows/block (strip 71 rows, pico ~32 GB) ← falha em 32 GB
output_block_rows <- 64L      # raster rows per strip; used only when max_strip_ram_gb is NULL
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

# Resolve "auto" config_id → rank-1 do final_run_summary.rds
if (identical(config_id, "auto")) {
  tmp_summary_path <- file.path(final_run_dir, "comparison", "final_run_summary.rds")
  if (!file.exists(tmp_summary_path))
    stop("Nao foi possivel resolver config_id='auto': arquivo nao encontrado: ",
         tmp_summary_path)
  config_id <- readRDS(tmp_summary_path)$selected_cfgs$config_id[1]
  message("config_id resolved to: ", config_id)
}

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

# Resolve "auto" config_id: pick the rank-1 config by mean test CCC across seeds.
# When 04 trains multiple configs, all_seed_results has one row per (config, seed);
# we rank by mean CCC descending and take the top config.
if (identical(config_id, "auto")) {
  if (nrow(final_summary$all_seed_results) > 0) {
    auto_rank <- final_summary$all_seed_results %>%
      dplyr::group_by(config_id) %>%
      dplyr::summarise(mean_ccc = mean(ccc, na.rm = TRUE), .groups = "drop") %>%
      dplyr::arrange(dplyr::desc(mean_ccc))
    config_id <- auto_rank$config_id[1]
    message("config_id resolved to: ", config_id,
            sprintf(" (mean test CCC %.4f across %d seeds)",
                    auto_rank$mean_ccc[1], sum(final_summary$all_seed_results$config_id == config_id)))
  } else {
    stop("config_id = 'auto' but final_run_summary$all_seed_results is empty.")
  }
}

if (!config_id %in% final_summary$selected_cfgs$config_id) {
  stop("config_id '", config_id, "' not in final run summary. Available: ",
       paste(final_summary$selected_cfgs$config_id, collapse = ", "))
}

cfg <- dplyr::filter(final_summary$selected_cfgs, config_id == !!config_id)

# Handles BOTH single-branch (1 window) and dual-branch (2 windows) configs.
# For dual-branch the two windows are read from the SAME predictor strip; a
# centre pixel is predicted only if ALL channels of BOTH windows are finite
# (the validity intersection), and the two patch arrays are fed to the two
# branches in window_sizes order (branch1 = window_sizes[1], branch2 = [2]).
window_sizes <- cfg$window_sizes[[1]]
n_branches   <- length(window_sizes)
if (!n_branches %in% c(1L, 2L)) {
  stop("cfg ", config_id, " has ", n_branches, " windows; expected 1 or 2.")
}
half_w_max   <- (max(window_sizes) - 1L) %/% 2L   # margins sized by the largest window

message("\nConfig ", config_id,
        " | window(s) ", paste(window_sizes, collapse = "x"), " (", n_branches, "-branch)",
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
predictor_scaling <- predictor_scaling %>%
  dplyr::filter(predictor %in% predictor_cols) %>%
  dplyr::arrange(match(predictor, predictor_cols))

if (!identical(predictor_scaling$predictor, predictor_cols)) {
  stop("predictor_scaling.csv channel order does not match raster_table_used.csv. ",
       "Re-check script 01 outputs.")
}

scale_method <- predictor_scaling$scaling_method
scale_center <- predictor_scaling$train_mean
scale_factor <- predictor_scaling$train_sd

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
  print(tibble::tibble(predictor = predictor_cols, geometry_ok = geom_ok) %>%
          dplyr::filter(!geometry_ok))
  stop("Some predictor rasters do not share the same geometry.")
}

raster_template <- rast_stack[[1]]
r_nrow <- terra::nrow(rast_stack)
r_ncol <- terra::ncol(rast_stack)
n_cell <- terra::ncell(rast_stack)

message("Raster grid: ", r_nrow, " rows x ", r_ncol, " cols x ", n_channels, " layers")

# ── Resolve output_block_rows from the per-strip RAM budget ───────────────────
# half_w_max (largest window) sets the strip margin. The predictor strip is the
# dominant allocation; size it to stay within max_strip_ram_gb.

bytes_per_strip_row <- as.numeric(r_ncol) * n_channels * 8   # all bands, one row

if (!is.null(max_strip_ram_gb)) {
  output_block_rows <- max(1L, as.integer(
    floor(max_strip_ram_gb * 1e9 / bytes_per_strip_row) - 2L * half_w_max
  ))
  message(sprintf(
    "Auto output_block_rows = %d  (strip budget %.1f GB -> %.2f GB/strip, %d blocks)",
    output_block_rows, max_strip_ram_gb,
    (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9,
    as.integer(ceiling(r_nrow / output_block_rows))
  ))
} else {
  message(sprintf(
    "output_block_rows = %d  (%.2f GB/strip, %d blocks)",
    output_block_rows,
    (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9,
    as.integer(ceiling(r_nrow / output_block_rows))
  ))
}

# Guard: a too-large single strip is the std::bad_alloc risk (heap can't find a
# big enough contiguous block). Warn well before that point so the user can lower
# output_block_rows / set max_strip_ram_gb.
strip_gb <- (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9
if (strip_gb > 8) {
  message(sprintf(
    "  WARNING: each predictor strip is ~%.1f GB. At fine resolution a single ",
    strip_gb),
    "contiguous read this large can fail with std::bad_alloc. Lower ",
    "output_block_rows or set max_strip_ram_gb (e.g. 4).")
}

# Guard (opposite end): when the largest window is big relative to the RAM budget,
# output_block_rows collapses to a handful of rows while the strip still spans
# output_block_rows + 2*half_w_max rows. Each block then RE-READS its margin rows,
# so the strip/useful-rows ratio is the disk over-read factor. At fine resolution
# (250 m, 187 bands) a window-15 config with max_strip_ram_gb = 4 yields ~2 useful
# rows per ~16-row strip → ~8x more raster I/O than necessary. Correct, just slow.
overread_factor <- (output_block_rows + 2L * half_w_max) / output_block_rows
if (output_block_rows < 8L && half_w_max >= 2L) {
  message(sprintf(
    "  NOTE: only %d useful row(s) per block but the strip spans %d rows (margin %d each side) -> ~%.1fx disk over-read.",
    output_block_rows, output_block_rows + 2L * half_w_max, half_w_max, overread_factor))
  message(sprintf(
    "        This is correct but I/O-heavy for window %d at this resolution. If you have the RAM, raise max_strip_ram_gb (e.g. 16-32) to enlarge the block and cut re-reads.",
    max(window_sizes)))
}

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

#' Build one [n, C, w, w] patch array PER window for the given centre pixels,
#' sharing a single validity mask across all windows (the intersection).
#'
#' Works for 1 window (single branch) or 2 windows (dual branch). For dual
#' branch a centre is kept only if EVERY channel of BOTH windows is finite, so
#' both branches always receive the exact same set of samples.
#'
#' Layout replicates 02_extract_patches.R EXACTLY, per window:
#'   offsets = expand.grid(dr, dc) with dr (row offset) varying fastest;
#'   cells indexed row-major within the strip; reshape [n_pos, n, C] -> aperm ->
#'   [n, C, w, w] so dim3 = row offset, dim4 = col offset (PyTorch N,C,H,W).
#'
#' Returns list(arrays, valid): `arrays` is a list of patch arrays in
#' window_sizes order (each holding only the common-valid centres); `valid`
#' flags, over ALL input centres, which were kept.
build_patches_multi <- function(center_row, center_col, strip_values,
                                read_row_start, n_cols, n_ch, window_sizes) {
  n         <- length(center_row)
  row_local <- center_row - read_row_start + 1L   # 1-based row within the strip

  # ── Pass 1: per-window cell-index matrix + validity → common (AND) mask ─────
  # Keep only the cheap integer cell matrices; the (large) extracted values are
  # transient here and re-extracted for valid centres in pass 2.
  per_win      <- vector("list", length(window_sizes))
  valid_common <- rep(TRUE, n)

  for (k in seq_along(window_sizes)) {
    w    <- window_sizes[k]
    half <- (w - 1L) %/% 2L
    offsets <- expand.grid(dr = (-half):half, dc = (-half):half)
    n_pos   <- nrow(offsets)

    cell_mat <- matrix(0L, nrow = n, ncol = n_pos)
    for (j in seq_len(n_pos)) {
      cell_mat[, j] <- (row_local + offsets$dr[j] - 1L) * n_cols +
                       (center_col + offsets$dc[j])
    }

    row_finite <- rowSums(!is.finite(
      strip_values[as.vector(t(cell_mat)), , drop = FALSE])) == 0
    valid_w      <- colSums(matrix(row_finite, nrow = n_pos, ncol = n)) == n_pos
    valid_common <- valid_common & valid_w

    per_win[[k]] <- list(w = w, n_pos = n_pos, cell_mat = cell_mat)
  }

  arrays <- vector("list", length(window_sizes))

  if (!any(valid_common)) {
    for (k in seq_along(window_sizes)) {
      arrays[[k]] <- array(0, dim = c(0L, n_ch, per_win[[k]]$w, per_win[[k]]$w))
    }
    return(list(arrays = arrays, valid = valid_common))
  }

  # ── Pass 2: extract + reshape only the common-valid centres, per window ─────
  valid_pos <- which(valid_common)
  for (k in seq_along(window_sizes)) {
    w     <- per_win[[k]]$w
    n_pos <- per_win[[k]]$n_pos
    sub_cells  <- as.vector(t(per_win[[k]]$cell_mat[valid_pos, , drop = FALSE]))
    vals_valid <- strip_values[sub_cells, , drop = FALSE]   # (n_valid*n_pos) x C

    step1 <- array(vals_valid, dim = c(n_pos, length(valid_pos), n_ch)) # [pos, n, ch]
    step2 <- aperm(step1, c(2L, 3L, 1L))                                # [n, ch, pos]
    arrays[[k]] <- array(step2, dim = c(length(valid_pos), n_ch, w, w)) # [n, ch, r, c]
  }

  list(arrays = arrays, valid = valid_common)
}

#' Predict log1p values for a list of patch arrays (one per branch) with one
#' model, batched on the GPU. Single branch → list of length 1; dual → length 2.
predict_one_model <- function(model, arr_list, device, batch_size) {
  n <- dim(arr_list[[1]])[1]
  if (n == 0L) return(numeric(0))
  out <- numeric(n)
  idx_groups <- split(seq_len(n), ceiling(seq_len(n) / batch_size))
  torch::with_no_grad({
    for (idx in idx_groups) {
      tensors <- lapply(arr_list, function(a)
        torch::torch_tensor(a[idx, , , , drop = FALSE],
                            dtype = torch::torch_float(), device = device))
      p <- do.call(model, tensors)        # 1 or 2 branch inputs -> [b, 1], log1p
      out[idx] <- as.numeric(p$squeeze(2L)$to(device = "cpu"))
    }
  })
  out
}

# ── Per-block prediction helper ───────────────────────────────────────────────
#
# Predicts every interior centre pixel inside one block of raster rows and
# returns the ensemble statistics, WITHOUT materialising any full-grid object.
# Returns global (row-major) cell indices so the caller can map them to the
# block-local positions it writes to disk.
compute_block <- function(b_start) {
  out_nrows   <- min(output_block_rows, r_nrow - b_start + 1L)
  out_row_end <- b_start + out_nrows - 1L

  center_rows <- intersect(b_start:out_row_end,
                           (half_w_max + 1L):(r_nrow - half_w_max))

  empty <- list(out_nrows = out_nrows, n_req = 0L, n_val = 0L,
                cells_all = integer(0), valid = logical(0),
                cells_valid = integer(0),
                median = numeric(0), mean = numeric(0), sd = numeric(0),
                mad = numeric(0), min = numeric(0), max = numeric(0))
  if (length(center_rows) == 0L) return(empty)

  read_row_start <- min(center_rows) - half_w_max
  read_row_end   <- max(center_rows) + half_w_max
  read_nrows     <- read_row_end - read_row_start + 1L

  strip_values <- terra::values(rast_stack, row = read_row_start,
                                nrows = read_nrows, mat = TRUE)
  strip_values <- apply_predictor_scaling(strip_values, predictor_cols,
                                          scale_method, scale_center, scale_factor,
                                          temperature_min_valid_celsius)

  center_cols <- (half_w_max + 1L):(r_ncol - half_w_max)
  grid  <- expand.grid(center_row = center_rows, center_col = center_cols)
  n_req <- nrow(grid)

  cells_all <- (grid$center_row - 1L) * r_ncol + grid$center_col
  valid_all <- logical(n_req)

  # pre-allocate stat buffers to the max possible size; fill by pointer, trim
  med <- mean_ <- sd_ <- mad_ <- min_ <- max_ <- numeric(n_req)
  cells_valid <- integer(n_req)
  ptr <- 0L

  chunk_list <- split(seq_len(n_req), ceiling(seq_len(n_req) / max(batch_size, 1L)))
  for (ci in chunk_list) {
    cr <- grid$center_row[ci]
    cc <- grid$center_col[ci]

    pb <- build_patches_multi(cr, cc, strip_values, read_row_start,
                              r_ncol, n_channels, window_sizes)
    valid_all[ci] <- pb$valid
    if (dim(pb$arrays[[1]])[1] == 0L) next

    preds_native <- matrix(NA_real_, nrow = dim(pb$arrays[[1]])[1], ncol = n_seeds)
    for (s in seq_len(n_seeds)) {
      preds_native[, s] <- pmax(expm1(predict_one_model(models[[s]], pb$arrays,
                                                         device, batch_size)), 0)
    }

    cv  <- ((cr - 1L) * r_ncol + cc)[pb$valid]
    nv  <- length(cv)
    rng <- (ptr + 1L):(ptr + nv)

    if (n_seeds >= 2L) {
      med[rng]   <- matrixStats::rowMedians(preds_native)
      mean_[rng] <- rowMeans(preds_native)
      sd_[rng]   <- matrixStats::rowSds(preds_native)
      mad_[rng]  <- matrixStats::rowMads(preds_native)   # constant 1.4826
      min_[rng]  <- matrixStats::rowMins(preds_native)
      max_[rng]  <- matrixStats::rowMaxs(preds_native)
    } else {
      v <- preds_native[, 1]
      med[rng] <- v; mean_[rng] <- v; sd_[rng] <- 0
      mad_[rng] <- 0; min_[rng] <- v; max_[rng] <- v
    }
    cells_valid[rng] <- cv
    ptr <- ptr + nv
  }

  rm(strip_values); gc()
  keep <- seq_len(ptr)
  list(out_nrows = out_nrows, n_req = n_req, n_val = ptr,
       cells_all = cells_all, valid = valid_all,
       cells_valid = cells_valid[keep],
       median = med[keep], mean = mean_[keep], sd = sd_[keep],
       mad = mad_[keep], min = min_[keep], max = max_[keep])
}

# ── Open streaming writers (one per output layer) ─────────────────────────────
# Each layer is written incrementally with writeStart/writeValues/writeStop, so
# only one block of rows per layer is ever in RAM.

gdal_opts <- function(datatype) {
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L
  c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
    "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
}

open_writer <- function(band_name, file_suffix, datatype = "FLT4S") {
  f <- file.path(output_raster_dir,
                 paste0(target_label, "_", config_id, "_", file_suffix, ".tif"))
  if (file.exists(f)) file.remove(f)
  r <- terra::rast(raster_template)
  names(r) <- band_name
  terra::writeStart(r, f, overwrite = TRUE, datatype = datatype,
                    gdal = gdal_opts(datatype))
  list(rast = r, file = f)
}

w_median <- open_writer("soc_pred_median_ton_ha", "ensemble_median_ton_ha")
w_mean   <- open_writer("soc_pred_mean_ton_ha",   "ensemble_mean_ton_ha")
w_sd     <- open_writer("soc_uncert_sd_ton_ha",   "ensemble_sd_ton_ha")
w_mad    <- open_writer("soc_uncert_mad_ton_ha",  "ensemble_mad_ton_ha")
w_min    <- open_writer("soc_pred_min_ton_ha",    "ensemble_min_ton_ha")
w_max    <- open_writer("soc_pred_max_ton_ha",    "ensemble_max_ton_ha")
w_mask   <- open_writer("valid_patch_mask",       "valid_mask", datatype = "INT1U")
writers  <- list(w_median, w_mean, w_sd, w_mad, w_min, w_max, w_mask)

abort_and_cleanup <- function(msg) {
  for (w in writers) try(terra::writeStop(w$rast), silent = TRUE)
  files <- vapply(writers, function(w) w$file, character(1))
  suppressWarnings(file.remove(files[file.exists(files)]))
  stop(msg, call. = FALSE)
}

# ── Block-streaming prediction loop (compute → write → discard) ───────────────

block_starts <- seq(1L, r_nrow, by = output_block_rows)
block_log    <- vector("list", length(block_starts))

# Running stats on the headline (median) map — avoids holding all pixels in RAM.
run_n         <- 0
run_sum       <- 0
run_min       <- Inf
run_max       <- -Inf
run_nonfinite <- 0
probe_done    <- FALSE

t0 <- Sys.time()

for (b in seq_along(block_starts)) {
  bs <- block_starts[b]
  cb <- compute_block(bs)

  blk_len    <- cb$out_nrows * r_ncol
  blk_median <- rep(NA_real_, blk_len)
  blk_mean   <- rep(NA_real_, blk_len)
  blk_sd     <- rep(NA_real_, blk_len)
  blk_mad    <- rep(NA_real_, blk_len)
  blk_min    <- rep(NA_real_, blk_len)
  blk_max    <- rep(NA_real_, blk_len)
  blk_mask   <- rep(0L,       blk_len)

  if (cb$n_req > 0L) {
    base <- (bs - 1L) * r_ncol                       # global→block-local offset
    blk_mask[cb$cells_all - base] <- as.integer(cb$valid)

    if (cb$n_val > 0L) {
      lv <- cb$cells_valid - base
      blk_median[lv] <- cb$median
      blk_mean[lv]   <- cb$mean
      blk_sd[lv]     <- cb$sd
      blk_mad[lv]    <- cb$mad
      blk_min[lv]    <- cb$min
      blk_max[lv]    <- cb$max

      # update running stats on the median map
      run_n         <- run_n + cb$n_val
      run_sum       <- run_sum + sum(cb$median[is.finite(cb$median)])
      run_min       <- min(run_min, min(cb$median, na.rm = TRUE))
      run_max       <- max(run_max, max(cb$median, na.rm = TRUE))
      run_nonfinite <- run_nonfinite + sum(!is.finite(cb$median))

      # fail-fast probe on the first block with predictions: catches the gross
      # "unscaled input" failure (predictions in the thousands) before spending
      # hours on the full grid.
      if (!probe_done) {
        probe_mean <- mean(cb$median[is.finite(cb$median)])
        if (!is.finite(probe_mean) || probe_mean > plausible_hard_max) {
          abort_and_cleanup(sprintf(
            "Fail-fast probe: first-block mean of median map = %.1f %s (> %g). Likely an input-scaling or channel-alignment problem. Partial rasters deleted.",
            probe_mean, target_unit, plausible_hard_max))
        }
        probe_done <- TRUE
      }
    }
  }

  terra::writeValues(w_median$rast, blk_median, bs, cb$out_nrows)
  terra::writeValues(w_mean$rast,   blk_mean,   bs, cb$out_nrows)
  terra::writeValues(w_sd$rast,     blk_sd,     bs, cb$out_nrows)
  terra::writeValues(w_mad$rast,    blk_mad,    bs, cb$out_nrows)
  terra::writeValues(w_min$rast,    blk_min,    bs, cb$out_nrows)
  terra::writeValues(w_max$rast,    blk_max,    bs, cb$out_nrows)
  terra::writeValues(w_mask$rast,   blk_mask,   bs, cb$out_nrows)

  block_log[[b]] <- tibble::tibble(
    block = b, row_start = bs, row_end = bs + cb$out_nrows - 1L,
    n_centers = cb$n_req, n_valid = cb$n_val,
    n_invalid = cb$n_req - cb$n_val
  )

  if (b == 1L || b %% 25L == 0L || b == length(block_starts)) {
    el <- Sys.time() - t0
    message(sprintf("Block %d/%d | rows %d-%d | valid %s | elapsed: %.2f %s",
                    b, length(block_starts), bs, bs + cb$out_nrows - 1L,
                    format(cb$n_val, big.mark = ","), el, units(el)))
  }
}

for (w in writers) terra::writeStop(w$rast)
rm(models); gc()

total_time    <- Sys.time() - t0
n_valid_total <- run_n

# ── Sanity checks (delete rasters if the written map is implausible) ───────────
# Exact global median is not held in RAM; we gate on the running MEAN of the
# median map (sufficient to catch gross failures) and report an approximate
# median sampled back from the written raster.

global_mean_med <- if (run_n > 0) run_sum / run_n else NA_real_
global_max      <- run_max

message("\n── Sanity checks ─────────────────────────────────────────────────")
message(sprintf("  Valid pixels         : %s (%.2f%% of grid)",
                format(n_valid_total, big.mark = ","), 100 * n_valid_total / n_cell))
message(sprintf("  Median map mean      : %.2f %s", global_mean_med, target_unit))
message(sprintf("  Median map range     : %.2f – %.2f %s", run_min, run_max, target_unit))

f_median <- w_median$file
f_mean   <- w_mean$file
f_sd     <- w_sd$file
f_mad    <- w_mad$file
f_min    <- w_min$file
f_max    <- w_max$file
f_mask   <- w_mask$file
written_files <- c(f_median, f_mean, f_sd, f_mad, f_min, f_max, f_mask)

sanity_ok <- TRUE
if (n_valid_total == 0L) {
  sanity_ok <- FALSE
  message("  [FAIL] No valid pixels were predicted.")
}
if (run_nonfinite > 0L) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] %d non-finite values in the median map.", run_nonfinite))
}
if (is.finite(global_mean_med) &&
    (global_mean_med < plausible_median_range[1] ||
     global_mean_med > plausible_median_range[2])) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] Median-map mean %.2f outside plausible range [%g, %g].",
                  global_mean_med, plausible_median_range[1], plausible_median_range[2]))
}
if (is.finite(global_max) && global_max > plausible_hard_max) {
  message(sprintf("  [WARN] Max %.0f exceeds %g %s — inspect the high tail.",
                  global_max, plausible_hard_max, target_unit))
}
if (!sanity_ok) {
  suppressWarnings(file.remove(written_files[file.exists(written_files)]))
  stop("Sanity checks failed. Written rasters were DELETED. See messages above.")
}
message("  All hard checks passed.")

# Approximate global median, sampled from the written raster (cheap, bounded RAM)
global_med <- tryCatch({
  samp <- terra::spatSample(terra::rast(f_median), size = 2e5,
                            method = "regular", na.rm = TRUE, values = TRUE)
  stats::median(samp[[1]], na.rm = TRUE)
}, error = function(e) NA_real_)
message(sprintf("  Global median (sampled, n=2e5): %.2f %s", global_med, target_unit))

# ── Metadata / manifests ──────────────────────────────────────────────────────

block_summary <- dplyr::bind_rows(block_log)
safe_write_csv2(block_summary, file.path(output_log_dir, "prediction_block_summary.csv"))

prediction_config <- tibble::tibble(
  target_label = target_label, target_unit = target_unit,
  config_id = config_id, final_run_id = final_run_id,
  window_sizes = paste(window_sizes, collapse = "x"), n_branches = n_branches,
  n_channels = n_channels,
  n_seeds = n_seeds, seeds = paste(seeds, collapse = ";"),
  ensemble_center = ensemble_center,
  input_scaling = "predictor_scaling.csv_zscore_percentage_dummy",
  back_transform = "expm1_then_clip0",
  aggregation_space = "per_seed_expm1_then_aggregate_across_seeds",
  output_block_rows = output_block_rows, batch_size = batch_size,
  r_nrow = r_nrow, r_ncol = r_ncol, n_cell = n_cell,
  n_valid = n_valid_total, valid_fraction = n_valid_total / n_cell,
  global_median_ton_ha = global_med, global_max_ton_ha = global_max,
  runtime_min = as.numeric(total_time, units = "mins"),
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
message(sprintf("  Runtime          : %.2f %s", total_time, units(total_time)))
message(sprintf("  Headline map     : %s", ensemble_center))

# Stats are read back from the written rasters (raster_summary) — nothing is held
# in RAM. terra::global gives mean/min/max per layer; the median map's median is
# the sampled estimate computed above.
rs_get <- function(layer, col) raster_summary[[col]][raster_summary$layer == layer]

message("\n  Central tendency (native ", target_unit, "), over valid pixels:")
message(sprintf("    median map -> median ~%.2f | mean %.2f | max %.2f",
                global_med, rs_get("median", "gmean"), rs_get("median", "gmax")))
message(sprintf("    mean map   ->            mean %.2f | max %.2f",
                rs_get("mean", "gmean"), rs_get("mean", "gmax")))
message(sprintf("    (mean-map mean exceeds median-map mean by %.2f %s = Jensen gap)",
                rs_get("mean", "gmean") - rs_get("median", "gmean"), target_unit))
message("\n  Ensemble disagreement (native ", target_unit, ", epistemic only):")
message(sprintf("    SD  : mean %.2f | max %.2f",
                rs_get("sd",  "gmean"), rs_get("sd",  "gmax")))
message(sprintf("    MAD : mean %.2f | max %.2f",
                rs_get("mad", "gmean"), rs_get("mad", "gmax")))

message("\n  Rasters written to: ", output_raster_dir)
message("  Logs written to:    ", output_log_dir)
message("\n  RECOMMENDED map: ", basename(f_median),
        " (median is invariant to expm1, robust, matches the SmoothL1 target).")
