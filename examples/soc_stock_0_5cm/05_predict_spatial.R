project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "terra",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "stringr",
  "matrixStats",
  "ps"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# One source() instead of several, in a dependency order that is not
# guessable. See R/load_all.R.
source(file.path(project_root, "R", "load_all.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 05 — Spatial prediction 2D-tiled (block-streaming, seed ensemble)
#
# Each worker receives a RECTANGLE of the raster (a slice of rows AND
# columns) rather than only a row strip. This keeps per-process RAM
# proportional to the tile's column fraction, allowing more concurrent
# workers per machine.
#
# (Historical note: an earlier 1D, row-strip-only version of this script
# was retired after production use showed it needed ~7-8 GB RAM/shard at
# 187 bands x ~160k columns, limiting concurrency. The 2D tiling here
# reduces that to ~2 GB/shard at n_col_shards=4, enough headroom to run
# ~8 workers concurrently on the same RAM budget.)
#
# Each worker writes a tile covering exactly its own rectangle (no NA
# padding). 05b_merge_spatial_parts.R mosaics all tiles at the end.
#
# CLI: Rscript 05_predict_spatial.R <row_shard_id> <col_shard_id>
#                                   <n_row_shards>  <n_col_shards>
#                                   [max_concurrent]
# ══════════════════════════════════════════════════════════════════════════════

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

# -- Arguments: CLI (Rscript) OR environment variables (source() in a console) -
#
# The rm(list=ls()) above erases any variable set before the source(), so an
# environment variable is the only way to pass a parameter into a source()d
# run: Sys.setenv() survives rm(list=ls()), a workspace object does not.
row_shard_id   <- 1L
col_shard_id   <- 1L
n_row_shards   <- 1L
n_col_shards   <- 1L
max_concurrent <- 1L   # only used to compute threads_per_worker

.cli_args <- commandArgs(trailingOnly = TRUE)
if (length(.cli_args) >= 4L) {
  row_shard_id  <- as.integer(.cli_args[1])
  col_shard_id  <- as.integer(.cli_args[2])
  n_row_shards  <- as.integer(.cli_args[3])
  n_col_shards  <- as.integer(.cli_args[4])
  if (length(.cli_args) >= 5L) max_concurrent <- as.integer(.cli_args[5])
} else if (nzchar(Sys.getenv("SOC_ROW_SHARD_ID"))) {
  row_shard_id  <- as.integer(Sys.getenv("SOC_ROW_SHARD_ID"))
  col_shard_id  <- as.integer(Sys.getenv("SOC_COL_SHARD_ID"))
  n_row_shards  <- as.integer(Sys.getenv("SOC_N_ROW_SHARDS"))
  n_col_shards  <- as.integer(Sys.getenv("SOC_N_COL_SHARDS"))
  if (nzchar(Sys.getenv("SOC_MAX_CONCURRENT"))) {
    max_concurrent <- as.integer(Sys.getenv("SOC_MAX_CONCURRENT"))
  }
}

stopifnot(
  n_row_shards >= 1L, n_col_shards >= 1L,
  row_shard_id >= 1L, row_shard_id <= n_row_shards,
  col_shard_id >= 1L, col_shard_id <= n_col_shards,
  max_concurrent >= 1L
)

n_total_shards <- n_row_shards * n_col_shards
is_partitioned <- n_total_shards > 1L

# ── WHICH RASTERS TO PREDICT ON ───────────────────────────────────────────────
#
# NULL (default) = the same rasters the patches were cut from, listed in
# raster_table_used.csv. That is the only setting whose map is comparable to
# the validation metrics.
#
# A directory = predict on THOSE rasters instead, matching predictors by file
# name. The point is a cheap end-to-end pass: a coarse grid turns a prediction
# measured in days into one measured in minutes, which is what makes it
# possible to exercise stage 05 at all during development.
#
# WHAT IT DOES NOT DO IS RESAMPLE. The window is counted in PIXELS, so the same
# 15 x 15 patch that spans 3.75 km at 250 m spans 300 km at 20 km: the network
# is shown a neighbourhood it was never trained on, and the values it returns
# are not the values it would return on the training grid. The wiring is what
# such a run proves -- never the map. The block below says so out loud, records
# the resolution it actually ran at, and refuses to be silent about it.
predict_raster_dir <- NULL
if (nzchar(Sys.getenv("SOC_PREDICT_RASTER_DIR"))) {
  predict_raster_dir <- Sys.getenv("SOC_PREDICT_RASTER_DIR")
}

config_id    <- "auto"
final_run_id <- "latest"
# Resolved below from final_run_summary$seeds -- the real list of seeds stage
# 04 trained. NOT fixed here: a hardcoded value goes silently stale the moment
# 04 changes how many seeds it uses, and it has. 04 went from 5 to 10 and this
# value did not follow.
seeds        <- NULL
ensemble_center <- "median"

# -- Streaming and RAM ---------------------------------------------------------
#
# With 2-D tiling, bytes_per_strip_row uses strip_ncol (the tile's columns plus
# its margins) rather than the whole r_ncol -> a larger output_block_rows ->
# fewer blocks -> less overhead.
max_strip_ram_gb  <- 4        # RAM budget per strip, per shard
output_block_rows <- 64L      # used only when max_strip_ram_gb is NULL

# A CAP ON BLOCK ROWS, INDEPENDENT OF THE STRIP BUDGET.
#
# The real RSS per shard is dominated by patch arrays accumulating on R's heap
# (vals_valid + step1 + step2 for the 9x9 and 15x15 branches), proportional to
# the number of VALID pixels per block -- not to the strip. With
# output_block_rows = 52 and dense blocks (109k valid per block) RSS reached
# 34 GB, which makes max_concurrent > 1 impossible. Capping at 16 rows cuts
# valid-per-block roughly threefold: ~10-15 GB expected on dense shards.
max_output_block_rows <- 16L  # never more than this, however cheap the strip

# batch_size dominates the RAM peak per CHUNK inside build_patches_multi: the
# window-validity check indexes strip_values for the whole chunk, creating a
# temporary matrix of ~batch_size * n_window_positions * n_channels * 8 bytes.
#
# That is independent of output_block_rows, so batch_size is the knob that
# actually bounds the per-chunk peak. At batch_size = 4096 with a 15x15 window
# (225 positions) over 187 channels, that check alone reached ~1.4 GB, repeated
# every chunk -- which is what produced the observed 30-37 GB spikes.
batch_size        <- 512L

plausible_median_range <- c(1, 200)
plausible_hard_max     <- 1000

# Threads split by max_concurrent (passed on the CLI by the orchestrator)
threads_per_worker <- max(1L, parallel::detectCores() %/% max_concurrent)
device <- setup_torch_device(n_threads = threads_per_worker, use_cuda = TRUE)

# ── Paths ─────────────────────────────────────────────────────────────────────

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
patch_dir        <- file.path(project_root, "outputs", "patches",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

raster_table_file      <- file.path(metadata_dir, "raster_table_used.csv")
qc_table_file          <- file.path(metadata_dir, "qc_table.csv")
patch_manifest_file    <- file.path(patch_dir, "patch_manifest.rds")

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
    stop("Could not resolve config_id = 'auto': ", tmp_summary_path)
  config_id <- readRDS(tmp_summary_path)$selected_cfgs$config_id[1]
  message("config_id resolved to: ", config_id)
}

model_dir    <- file.path(final_run_dir, config_id, "models")
summary_file <- file.path(final_run_dir, "comparison", "final_run_summary.rds")

# The scaling is part of the FITTED MODEL, written next to its weights by
# stage 04. It used to be read from a global table written by stage 01 -- a
# second copy of the same fact, free to drift, and it did: once patches were
# stored raw and scaling became per-fold, the map was being built with one set
# of constants while the network had been trained with another. Silently.
#
# It is built HERE, not with the other paths above, because it needs both
# final_run_dir and config_id -- and both can be "latest"/"auto" and are only
# resolved a few lines up. Built any earlier it referred to objects that did
# not exist yet.
predictor_scaling_file <- file.path(final_run_dir, config_id,
                                    "predictor_scaling.csv")

output_dir        <- file.path(project_root, "outputs", "spatial_prediction",
                               "soc_stock_modeling", target_label, config_id)
output_raster_dir <- file.path(output_dir, "raster")
output_log_dir    <- file.path(output_dir, "log")

create_output_dirs(c(output_dir, output_raster_dir, output_log_dir))

# ── Validate inputs ────────────────────────────────────────────────────────────

for (f in c(raster_table_file, predictor_scaling_file, summary_file)) {
  if (!file.exists(f)) stop("Required input not found: ", f)
}
if (!dir.exists(model_dir)) stop("Model directory not found: ", model_dir)

# ── Predictor scaling ─────────────────────────────────────────────────────────

predictor_scaling <- readr::read_csv2(predictor_scaling_file,
                                      show_col_types = FALSE)
qc_table <- readr::read_csv2(qc_table_file, show_col_types = FALSE)

# QC and scaling both come from the pipeline's own functions (R/preprocess.R),
# applied here in the same order stage 02 applied them: QC first, then the
# affine transform. This script used to carry its own hand-written copy of both
# -- a temperature threshold and a percentage clamp written out again, with the
# predictor name matched by a regex. Two copies of one rule drift, and when
# they do the map is wrong in a way no check can see.
apply_predictor_scaling <- function(mat, pred_names, scaling, qc) {
  qc_rows <- qc[match(pred_names, qc$predictor), , drop = FALSE]
  for (i in seq_len(ncol(mat))) {
    mat[, i] <- qc_band_values(mat[, i], qc_rows[i, ])
  }
  scale_patches_matrix(mat, scaling)
}

# ── Model config ──────────────────────────────────────────────────────────────

final_summary <- readRDS(summary_file)

if (identical(config_id, "auto")) {
  if (nrow(final_summary$all_seed_results) > 0) {
    auto_rank <- final_summary$all_seed_results %>%
      dplyr::group_by(config_id) %>%
      dplyr::summarise(mean_ccc = mean(ccc, na.rm = TRUE), .groups = "drop") %>%
      dplyr::arrange(dplyr::desc(mean_ccc))
    config_id <- auto_rank$config_id[1]
    message("config_id resolved to: ", config_id,
            sprintf(" (mean test CCC %.4f)", auto_rank$mean_ccc[1]))
  } else {
    stop("config_id = 'auto' but final_run_summary$all_seed_results is empty.")
  }
}

if (!config_id %in% final_summary$selected_cfgs$config_id) {
  stop("config_id '", config_id, "' not in final run summary.")
}

if (is.null(seeds)) {
  seeds <- final_summary$seeds
  if (is.null(seeds) || length(seeds) == 0L) {
    stop("Could not resolve seeds from final_run_summary$seeds: ", summary_file)
  }
  message("seeds resolved from final_run_summary.rds: ", paste(seeds, collapse = ", "),
          " (", length(seeds), " seed(s))")
}

cfg          <- dplyr::filter(final_summary$selected_cfgs, config_id == !!config_id)
window_sizes <- cfg$window_sizes[[1]]
n_branches   <- length(window_sizes)
if (!n_branches %in% c(1L, 2L)) stop("cfg ", config_id, " has ", n_branches, " windows.")
half_w_max   <- (max(window_sizes) - 1L) %/% 2L

message("\nConfig ", config_id,
        " | window(s) ", paste(window_sizes, collapse = "x"), " (", n_branches, "-branch)",
        " | conv ", paste(cfg$conv_channels[[1]], collapse = "_"),
        " | embed ", cfg$embedding_dim,
        " | gate ", cfg$gate_type)

# ── Raster table ──────────────────────────────────────────────────────────────

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
if (!all(c("raster_file", "predictor") %in% names(raster_table))) {
  stop("raster_table_used.csv must contain 'raster_file' and 'predictor'.")
}

predictor_cols <- raster_table$predictor
n_channels     <- length(predictor_cols)
raster_files   <- raster_table$raster_file

# -- predicting on a different grid -------------------------------------------
#
# Matching is by FILE NAME, and the predictor ORDER is the training order, kept
# exactly: the channel order is the contract tying channel i to band i, and the
# alphabetical order a directory listing returns is not that contract.
if (!is.null(predict_raster_dir)) {
  if (!dir.exists(predict_raster_dir)) {
    stop("predict_raster_dir does not exist: ", predict_raster_dir)
  }
  avail <- list.files(predict_raster_dir, pattern = "\\.(tif|tiff)$",
                      full.names = TRUE, ignore.case = TRUE)
  remapped <- file.path(predict_raster_dir, basename(raster_files))
  absent   <- raster_table$predictor[!remapped %in% avail]
  if (length(absent) > 0) {
    stop("predict_raster_dir is missing ", length(absent), " of the ",
         n_channels, " predictors the model needs, among them: ",
         paste(utils::head(absent, 8), collapse = ", "),
         "\nPredicting without a channel is not possible: the network has a ",
         "weight for every one of them.")
  }
  raster_files <- remapped
}

missing_rasters <- raster_files[!file.exists(raster_files)]
if (length(missing_rasters) > 0) {
  print(missing_rasters)
  stop("Some predictor raster files no longer exist.")
}

predictor_scaling <- predictor_scaling %>%
  dplyr::filter(predictor %in% predictor_cols) %>%
  dplyr::arrange(match(predictor, predictor_cols))

if (!identical(as.character(predictor_scaling$predictor), predictor_cols)) {
  stop("The model's predictor_scaling.csv is in a different channel order ",
       "than raster_table_used.csv. Predicting under that mismatch would feed ",
       "the network one channel while the map is built from another.")
}
if (!identical(as.character(qc_table$predictor), predictor_cols)) {
  stop("qc_table.csv is in a different channel order than ",
       "raster_table_used.csv.")
}

message("Model scaling and QC rules loaded and aligned (", n_channels,
        " channels), from: ", final_run_dir)

if (file.exists(patch_manifest_file)) {
  manifest <- readRDS(patch_manifest_file)
  manifest_predictors <- strsplit(manifest$predictor_cols_final, ";")[[1]]
  if (!identical(as.character(manifest_predictors), as.character(predictor_cols))) {
    stop("Predictor order mismatch vs patch manifest.")
  }
  message("Channel alignment verified against patch manifest.")
} else {
  message("WARNING: patch_manifest.rds not found.")
}

# ── Open raster stack ─────────────────────────────────────────────────────────

rast_stack <- terra::rast(raster_files)
names(rast_stack) <- predictor_cols

# -- the scale the map is actually being built at -----------------------------
#
# Compared against the store's own record, not against a number written here:
# a hardcoded expectation is wrong for everyone but this example.
.res_now <- terra::res(rast_stack)[1]
.res_trained <- if (file.exists(patch_manifest_file)) {
  m <- readRDS(patch_manifest_file)
  if ("cell_size" %in% names(m)) suppressWarnings(as.numeric(m$cell_size[1])) else NA_real_
} else NA_real_

message(sprintf("\nPrediction grid: %.8f per pixel", .res_now))
if (!is.na(.res_trained) && abs(.res_now - .res_trained) > 1e-9) {
  ratio <- .res_now / .res_trained
  message(strrep("!", 78))
  message(sprintf(
    paste0("THE MODEL WAS TRAINED AT %.8f AND IS PREDICTING AT %.8f (%.1fx).\n",
           "The window is counted in PIXELS, so a %d x %d patch now covers\n",
           "%.1fx the ground it covered in training. This map exercises the\n",
           "pipeline; it does NOT carry the accuracy the validation reported."),
    .res_trained, .res_now, ratio,
    max(window_sizes), max(window_sizes), ratio))
  message(strrep("!", 78))
}

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

# -- 2-D partition -----------------------------------------------------------
# Each worker covers a rectangle [my_row_start:my_row_end,
# my_col_start:my_col_end]. READING includes a half_w_max margin on every side,
# because the edge pixels' patches reach into it. Only the inner rectangle is
# WRITTEN to the output tile -- the margin belongs to the neighbouring shard.

row_bounds   <- floor(seq(1, r_nrow + 1, length.out = n_row_shards + 1L))
my_row_start <- row_bounds[row_shard_id]
my_row_end   <- row_bounds[row_shard_id + 1L] - 1L

col_bounds   <- floor(seq(1, r_ncol + 1, length.out = n_col_shards + 1L))
my_col_start <- col_bounds[col_shard_id]
my_col_end   <- col_bounds[col_shard_id + 1L] - 1L

tile_nrow <- my_row_end   - my_row_start   + 1L
tile_ncol <- my_col_end   - my_col_start   + 1L
tile_ncell <- tile_nrow * tile_ncol

part_suffix <- if (is_partitioned) {
  sprintf("_r%03dof%03d_c%03dof%03d", row_shard_id, n_row_shards, col_shard_id, n_col_shards)
} else {
  ""
}

if (is_partitioned) {
  message(sprintf(
    "Shard [%d/%d row, %d/%d col]: rows %d-%d (%s), cols %d-%d (%s)",
    row_shard_id, n_row_shards, col_shard_id, n_col_shards,
    my_row_start, my_row_end, format(tile_nrow, big.mark = ","),
    my_col_start, my_col_end, format(tile_ncol, big.mark = ",")))
}

# Columns the strip reads: the tile plus its margin, clamped to the raster
read_col_start <- max(1L,      my_col_start - half_w_max)
read_col_end   <- min(r_ncol,  my_col_end   + half_w_max)
strip_ncol     <- read_col_end - read_col_start + 1L

# -- output_block_rows, from the RAM budget ----------------------------------
# With 2-D tiling strip_ncol << r_ncol, so bytes_per_strip_row is far smaller
# -> a larger output_block_rows -> less I/O overhead than a 1-D scheme that
# reads a whole raster row per strip.

bytes_per_strip_row <- as.numeric(strip_ncol) * n_channels * 8

if (!is.null(max_strip_ram_gb)) {
  output_block_rows <- max(1L, as.integer(
    floor(max_strip_ram_gb * 1e9 / bytes_per_strip_row) - 2L * half_w_max
  ))
} else {
  # no auto-sizing: the fixed value above is used as given
}
# Apply the cap: whatever the strip budget allows, never exceed
# max_output_block_rows. This is what stops the prediction RSS (patch arrays x
# valid pixels per block) from exploding on dense shards, where the strip would
# be cheap enough to justify blocks the heap cannot afford.
output_block_rows <- min(output_block_rows, max_output_block_rows)

message(sprintf(
  "output_block_rows = %d (cap=%d | strip budget %.1f GB | strip_ncol %d vs r_ncol %d -> %.2f GB/strip)",
  output_block_rows, max_output_block_rows,
  if (!is.null(max_strip_ram_gb)) max_strip_ram_gb else NA_real_,
  strip_ncol, r_ncol,
  (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9
))

strip_gb <- (output_block_rows + 2L * half_w_max) * bytes_per_strip_row / 1e9
if (strip_gb > 8) {
  message(sprintf("  WARNING: strip ~%.1f GB. Considere reduzir max_strip_ram_gb.", strip_gb))
}

# ── Seed models ───────────────────────────────────────────────────────────────

model_files <- file.path(model_dir, sprintf("seed%04d_best.pt", seeds))
have <- file.exists(model_files)
if (!any(have)) stop("No seed model files found in: ", model_dir)
if (!all(have)) {
  message("WARNING: missing seeds, using only ", sum(have), " available.")
}
seeds       <- seeds[have]
model_files <- model_files[have]
n_seeds     <- length(seeds)

message("\nLoading ", n_seeds, " seed model(s) for ", config_id, "...")
models <- purrr::map(seq_len(n_seeds), function(i) {
  m  <- build_cnn_from_config(cfg, n_channels)
  st <- torch::torch_load(model_files[i])
  m$load_state_dict(st)
  m$to(device = device)
  m$eval()
  m
})

# ── Helpers ───────────────────────────────────────────────────────────────────

# build_patches_multi recebe center_col LOCAL (1-based dentro da strip de leitura),
# not the global column. strip_values has strip_ncol columns, not r_ncol.
#
# A geometria (indexacao de celula e montagem do array) vive em R/patches.R,
# shared with stage 02's training extraction -- it is the ONLY implementation,
# so the two paths cannot diverge in silence. The cost rule still holds:
# strip_values is indexed ONCE per window branch, and that result feeds both the
# validity check and the final assembly (reshape and slice only, never a second
# copy out of strip_values).
build_patches_multi <- function(center_row, center_col_local, strip_values,
                                read_row_start, strip_ncol_arg, n_ch, window_sizes) {
  n         <- length(center_row)
  row_local <- center_row - read_row_start + 1L

  per_win      <- vector("list", length(window_sizes))
  valid_common <- rep(TRUE, n)

  # Stage 1: gather each window ONCE. patch_gather() indexes strip_values a
  # single time and reports validity from that same pass, so the array is
  # built later by reshape/slice only -- never by re-indexing the strip.
  for (k in seq_along(window_sizes)) {
    cm <- patch_cell_index(row_local, center_col_local, strip_ncol_arg,
                           window_sizes[k])
    g  <- patch_gather(strip_values, cm, n_ch)
    valid_common <- valid_common & g$valid
    per_win[[k]] <- g$values
  }

  # Stage 2: only now, with validity intersected across ALL windows, cut the
  # arrays to the surviving centres.
  valid_pos <- which(valid_common)
  arrays <- lapply(seq_along(window_sizes), function(k) {
    patch_finish(per_win[[k]], valid_pos, n_ch, window_sizes[k])
  })

  list(arrays = arrays, valid = valid_common)
}

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
      p <- do.call(model, tensors)
      out[idx] <- as.numeric(p$squeeze(2L)$to(device = "cpu"))
    }
  })
  out
}

# ── compute_block ─────────────────────────────────────────────────────────────
# Predict the tile rectangle's pixels for one block of rows.
# Diferenças vs 05:
#   - center_cols limitados a my_col_start:my_col_end
#   - terra::values reads only read_col_start:read_col_end (strip_ncol columns)
#   - the center_col passed to build_patches_multi and quick_ok is LOCAL to the
#     strip, never global -- the one index that has to be converted
#   - cells_valid are GLOBAL, (row-1)*r_ncol+col, converted to tile-local later
#     no loop principal (fora desta funcao)

compute_block <- function(b_start) {
  out_nrows   <- min(output_block_rows, my_row_end - b_start + 1L)
  out_row_end <- b_start + out_nrows - 1L

  center_rows <- intersect(b_start:out_row_end,
                           (half_w_max + 1L):(r_nrow - half_w_max))

  # Tile columns, clamped by the global margins
  cc_start <- max(my_col_start, half_w_max + 1L)
  cc_end   <- min(my_col_end,   r_ncol - half_w_max)

  empty <- list(out_nrows = out_nrows, n_req = 0L, n_val = 0L,
                cells_all = integer(0), valid = logical(0),
                cells_valid = integer(0),
                median = numeric(0), mean = numeric(0), sd = numeric(0),
                mad = numeric(0), min = numeric(0), max = numeric(0),
                t_read = 0, t_scale = 0, t_predict = 0, n_quick_ok = 0L)
  if (length(center_rows) == 0L || cc_start > cc_end) return(empty)

  read_row_start <- min(center_rows) - half_w_max
  read_row_end   <- max(center_rows) + half_w_max
  read_nrows     <- read_row_end - read_row_start + 1L

  .t_read_start <- Sys.time()
  # Read the strip's 2-D slice: only this tile's columns, plus the margin
  strip_values <- terra::values(rast_stack,
                                row   = read_row_start, nrows = read_nrows,
                                col   = read_col_start, ncols = strip_ncol,
                                mat   = TRUE)
  t_read <- as.numeric(Sys.time() - .t_read_start, units = "secs")

  .t_scale_start <- Sys.time()
  strip_values <- apply_predictor_scaling(strip_values, predictor_cols,
                                          predictor_scaling, qc_table)
  t_scale <- as.numeric(Sys.time() - .t_scale_start, units = "secs")

  .t_predict_start <- Sys.time()
  center_cols <- cc_start:cc_end
  grid  <- expand.grid(center_row = center_rows, center_col = center_cols)
  n_req <- nrow(grid)

  # GLOBAL indices, for the cells_all / cells_valid returned to the caller
  cells_all <- (grid$center_row - 1L) * r_ncol + grid$center_col
  valid_all <- logical(n_req)

  med <- mean_ <- sd_ <- mad_ <- min_ <- max_ <- numeric(n_req)
  cells_valid <- integer(n_req)
  ptr <- 0L
  n_quick_ok <- 0L

  chunk_list <- split(seq_len(n_req), ceiling(seq_len(n_req) / max(batch_size, 1L)))
  n_chunks <- length(chunk_list)
  .heartbeat_last   <- Sys.time()
  heartbeat_every_s <- 120

  for (chunk_idx in seq_along(chunk_list)) {
    ci <- chunk_list[[chunk_idx]]

    if (as.numeric(Sys.time() - .heartbeat_last, units = "secs") >= heartbeat_every_s) {
      message(sprintf(
        "    [heartbeat] chunk %s/%s | quick_ok so far %s | valid so far %s | %.1f min into this block",
        format(chunk_idx, big.mark = ","), format(n_chunks, big.mark = ","),
        format(n_quick_ok, big.mark = ","), format(ptr, big.mark = ","),
        as.numeric(Sys.time() - .t_predict_start, units = "mins")))
      .heartbeat_last <- Sys.time()
    }

    cr <- grid$center_row[ci]
    cc <- grid$center_col[ci]   # GLOBAL columns

    # quick_ok: indices local to the strip (strip_ncol columns)
    row_local_c <- cr - read_row_start + 1L
    col_local_c <- cc - read_col_start + 1L   # column local to the strip
    center_idx  <- (row_local_c - 1L) * strip_ncol + col_local_c
    quick_ok    <- rowSums(!is.finite(strip_values[center_idx, , drop = FALSE])) == 0
    n_quick_ok  <- n_quick_ok + sum(quick_ok)

    if (!any(quick_ok)) next

    # build_patches_multi takes the column LOCAL to the strip
    cc_local <- cc - read_col_start + 1L
    pb <- build_patches_multi(cr[quick_ok], cc_local[quick_ok], strip_values,
                              read_row_start, strip_ncol, n_channels, window_sizes)
    valid_sub      <- logical(length(ci))
    valid_sub[quick_ok] <- pb$valid
    valid_all[ci]  <- valid_sub
    if (dim(pb$arrays[[1]])[1] == 0L) next

    preds_native <- matrix(NA_real_, nrow = dim(pb$arrays[[1]])[1], ncol = n_seeds)
    for (s in seq_len(n_seeds)) {
      preds_native[, s] <- pmax(expm1(predict_one_model(models[[s]], pb$arrays,
                                                         device, batch_size)), 0)
    }

    # cells_valid: índices GLOBAIS
    cv  <- ((cr[quick_ok] - 1L) * r_ncol + cc[quick_ok])[pb$valid]
    nv  <- length(cv)
    rng <- (ptr + 1L):(ptr + nv)

    if (n_seeds >= 2L) {
      med[rng]   <- matrixStats::rowMedians(preds_native)
      mean_[rng] <- rowMeans(preds_native)
      sd_[rng]   <- matrixStats::rowSds(preds_native)
      mad_[rng]  <- matrixStats::rowMads(preds_native)
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

  t_predict <- as.numeric(Sys.time() - .t_predict_start, units = "secs")
  rm(strip_values); gc()
  keep <- seq_len(ptr)
  list(out_nrows = out_nrows, n_req = n_req, n_val = ptr,
       cells_all = cells_all, valid = valid_all,
       cells_valid = cells_valid[keep],
       median = med[keep], mean = mean_[keep], sd = sd_[keep],
       mad = mad_[keep], min = min_[keep], max = max_[keep],
       t_read = t_read, t_scale = t_scale, t_predict = t_predict,
       n_quick_ok = n_quick_ok)
}

# ── Template do tile e writers ─────────────────────────────────────────────────

gdal_opts <- function(datatype) {
  predictor <- if (grepl("^INT|^UINT|^BYTE", datatype)) 2L else 3L
  c("COMPRESS=DEFLATE", paste0("PREDICTOR=", predictor),
    "TILED=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512")
}

if (is_partitioned) {
  full_ext    <- as.vector(terra::ext(raster_template))
  xres_full   <- terra::xres(raster_template)
  yres_full   <- terra::yres(raster_template)
  worker_ymax <- full_ext[["ymax"]] - (my_row_start - 1L) * yres_full
  worker_ymin <- full_ext[["ymax"]] - my_row_end * yres_full
  worker_xmin <- full_ext[["xmin"]] + (my_col_start - 1L) * xres_full
  worker_xmax <- full_ext[["xmin"]] + my_col_end * xres_full
  worker_ext  <- terra::ext(worker_xmin, worker_xmax, worker_ymin, worker_ymax)
  worker_template <- terra::crop(raster_template, worker_ext, snap = "near")
} else {
  worker_template <- raster_template
}

output_raster_dir_w <- if (is_partitioned)
  file.path(output_raster_dir, "parts_2d") else output_raster_dir
create_output_dirs(output_raster_dir_w)

open_writer <- function(band_name, file_suffix, datatype = "FLT4S") {
  f <- file.path(output_raster_dir_w,
                 paste0(target_label, "_", config_id, "_", file_suffix, part_suffix, ".tif"))
  if (file.exists(f)) file.remove(f)
  r <- terra::rast(worker_template)
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

# ── Loop principal ─────────────────────────────────────────────────────────────

diag_log_every <- 1L
.proc_handle   <- ps::ps_handle()

block_starts <- seq(my_row_start, my_row_end, by = output_block_rows)
block_log    <- vector("list", length(block_starts))

run_n <- 0; run_sum <- 0; run_min <- Inf; run_max <- -Inf
run_nonfinite <- 0; probe_done <- FALSE

t0 <- Sys.time()

for (b in seq_along(block_starts)) {
  bs <- block_starts[b]
  cb <- compute_block(bs)

  # blk_len: block rows x TILE columns, not the whole raster's columns
  blk_len    <- cb$out_nrows * tile_ncol
  blk_median <- rep(NA_real_, blk_len)
  blk_mean   <- rep(NA_real_, blk_len)
  blk_sd     <- rep(NA_real_, blk_len)
  blk_mad    <- rep(NA_real_, blk_len)
  blk_min    <- rep(NA_real_, blk_len)
  blk_max    <- rep(NA_real_, blk_len)
  blk_mask   <- rep(0L, blk_len)

  if (cb$n_req > 0L) {
    # Convert GLOBAL indices (row-1)*r_ncol+col into tile-block indices
    g2tile <- function(cells) {
      row_g       <- (cells - 1L) %/% r_ncol + 1L
      col_g       <- (cells - 1L) %% r_ncol + 1L
      row_in_blk  <- row_g - bs + 1L
      col_in_tile <- col_g - my_col_start + 1L
      (row_in_blk - 1L) * tile_ncol + col_in_tile
    }

    blk_mask[g2tile(cb$cells_all)] <- as.integer(cb$valid)

    if (cb$n_val > 0L) {
      tv <- g2tile(cb$cells_valid)
      blk_median[tv] <- cb$median
      blk_mean[tv]   <- cb$mean
      blk_sd[tv]     <- cb$sd
      blk_mad[tv]    <- cb$mad
      blk_min[tv]    <- cb$min
      blk_max[tv]    <- cb$max

      run_n         <- run_n + cb$n_val
      run_sum       <- run_sum + sum(cb$median[is.finite(cb$median)])
      run_min       <- min(run_min, min(cb$median, na.rm = TRUE))
      run_max       <- max(run_max, max(cb$median, na.rm = TRUE))
      run_nonfinite <- run_nonfinite + sum(!is.finite(cb$median))

      if (!probe_done) {
        probe_mean <- mean(cb$median[is.finite(cb$median)])
        if (!is.finite(probe_mean) || probe_mean > plausible_hard_max) {
          abort_and_cleanup(sprintf(
            "Fail-fast probe: first-block mean = %.1f %s (> %g).",
            probe_mean, target_unit, plausible_hard_max))
        }
        probe_done <- TRUE
      }
    }
  }

  # Row local to this worker's tile (1-based)
  bs_local <- bs - my_row_start + 1L

  terra::writeValues(w_median$rast, blk_median, bs_local, cb$out_nrows)
  terra::writeValues(w_mean$rast,   blk_mean,   bs_local, cb$out_nrows)
  terra::writeValues(w_sd$rast,     blk_sd,     bs_local, cb$out_nrows)
  terra::writeValues(w_mad$rast,    blk_mad,    bs_local, cb$out_nrows)
  terra::writeValues(w_min$rast,    blk_min,    bs_local, cb$out_nrows)
  terra::writeValues(w_max$rast,    blk_max,    bs_local, cb$out_nrows)
  terra::writeValues(w_mask$rast,   blk_mask,   bs_local, cb$out_nrows)

  rss_mb <- ps::ps_memory_info(.proc_handle)[["rss"]] / 1e6

  block_log[[b]] <- tibble::tibble(
    block = b, row_start = bs, row_end = bs + cb$out_nrows - 1L,
    col_start = my_col_start, col_end = my_col_end,
    n_centers = cb$n_req, n_quick_ok = cb$n_quick_ok, n_valid = cb$n_val,
    n_invalid = cb$n_req - cb$n_val,
    t_read = cb$t_read, t_scale = cb$t_scale, t_predict = cb$t_predict,
    rss_mb = rss_mb
  )

  if (b %% diag_log_every == 0L || b == 1L || b == length(block_starts)) {
    message(sprintf(
      "Block %d/%d | rows %d-%d cols %d-%d | centers %s quick_ok %s valid %s | read %.1fs scale %.1fs predict %.1fs | RSS %.0f MB",
      b, length(block_starts), bs, bs + cb$out_nrows - 1L,
      my_col_start, my_col_end,
      format(cb$n_req, big.mark = ","), format(cb$n_quick_ok, big.mark = ","),
      format(cb$n_val, big.mark = ","), cb$t_read, cb$t_scale, cb$t_predict, rss_mb))
  }

  if (b == 1L || b %% 25L == 0L || b == length(block_starts)) {
    el <- Sys.time() - t0
    message(sprintf("  └─ cumulative elapsed: %.2f %s", as.numeric(el), units(el)))
    safe_write_csv2(dplyr::bind_rows(block_log[seq_len(b)]),
                    file.path(output_log_dir,
                              paste0("prediction_block_summary_partial", part_suffix, ".csv")))
  }
}

for (w in writers) terra::writeStop(w$rast)
rm(models); gc()

total_time    <- Sys.time() - t0
n_valid_total <- run_n

# ── Sanity checks ──────────────────────────────────────────────────────────────

global_mean_med <- if (run_n > 0) run_sum / run_n else NA_real_
global_max      <- run_max

message("\n── Sanity checks ─────────────────────────────────────────────────")
message(sprintf("  Valid pixels         : %s (%.2f%% do tile)",
                format(n_valid_total, big.mark = ","), 100 * n_valid_total / tile_ncell))
message(sprintf("  Median map mean      : %.2f %s", global_mean_med, target_unit))
message(sprintf("  Median map range     : %.2f – %.2f %s", run_min, run_max, target_unit))

f_median <- w_median$file; f_mean <- w_mean$file; f_sd <- w_sd$file
f_mad    <- w_mad$file;    f_min  <- w_min$file;  f_max <- w_max$file
f_mask   <- w_mask$file
written_files <- c(f_median, f_mean, f_sd, f_mad, f_min, f_max, f_mask)

sanity_ok <- TRUE
if (!is_partitioned && n_valid_total == 0L) {
  sanity_ok <- FALSE
  message("  [FAIL] No valid pixels.")
}
if (run_nonfinite > 0L) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] %d non-finite values in median map.", run_nonfinite))
}
if (!is_partitioned && is.finite(global_mean_med) &&
    (global_mean_med < plausible_median_range[1] ||
     global_mean_med > plausible_median_range[2])) {
  sanity_ok <- FALSE
  message(sprintf("  [FAIL] Mean %.2f fora do range plausivel.", global_mean_med))
}
if (is.finite(global_max) && global_max > plausible_hard_max) {
  message(sprintf("  [WARN] Max %.0f > %g %s.", global_max, plausible_hard_max, target_unit))
}
if (is_partitioned && n_valid_total == 0L) {
  message("  [INFO] Tile has no valid pixels (ocean, most likely) -- not an error.")
}
if (!sanity_ok) {
  suppressWarnings(file.remove(written_files[file.exists(written_files)]))
  stop("Sanity checks failed. Rasters deletados.")
}
message("  All hard checks passed.")

global_med <- tryCatch({
  samp <- terra::spatSample(terra::rast(f_median), size = 2e5,
                            method = "regular", na.rm = TRUE, values = TRUE)
  stats::median(samp[[1]], na.rm = TRUE)
}, error = function(e) NA_real_)
message(sprintf("  Global median (sampled): %.2f %s", global_med, target_unit))

# ── Manifests ─────────────────────────────────────────────────────────────────

block_summary <- dplyr::bind_rows(block_log)
safe_write_csv2(block_summary,
                file.path(output_log_dir, paste0("prediction_block_summary", part_suffix, ".csv")))

device_type <- device$type

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
  tile_nrow = tile_nrow, tile_ncol = tile_ncol, tile_ncell = tile_ncell,
  row_shard_id = row_shard_id, col_shard_id = col_shard_id,
  n_row_shards = n_row_shards, n_col_shards = n_col_shards,
  strip_ncol = strip_ncol,
  n_valid = n_valid_total, valid_fraction = n_valid_total / tile_ncell,
  global_median_ton_ha = global_med, global_max_ton_ha = global_max,
  runtime_min = as.numeric(total_time, units = "mins"),
  # device_type, read before the tibble: a column called `device` shadows the
  # torch device for every argument AFTER it, so this is safe only while it
  # stays last. Read here, it stays safe wherever it moves.
  device = device_type, predicted_at = as.character(Sys.time())
)
safe_write_csv2(prediction_config,
                file.path(output_log_dir, paste0("prediction_config", part_suffix, ".csv")))

raster_summary <- purrr::map_dfr(
  list(median = f_median, mean = f_mean, sd = f_sd, mad = f_mad,
       min = f_min, max = f_max, mask = f_mask),
  function(f) {
    r <- terra::rast(f)
    tibble::tibble(
      file = f, band = names(r),
      gmin  = terra::global(r, "min",  na.rm = TRUE)[1, 1],
      gmean = terra::global(r, "mean", na.rm = TRUE)[1, 1],
      gmax  = terra::global(r, "max",  na.rm = TRUE)[1, 1]
    )
  },
  .id = "layer"
)
safe_write_csv2(raster_summary,
                file.path(output_log_dir, paste0("prediction_raster_summary", part_suffix, ".csv")))

if (row_shard_id == 1L && col_shard_id == 1L) {
  safe_write_csv2(
    tibble::tibble(order = seq_len(n_channels), predictor = predictor_cols),
    file.path(output_log_dir, "predictor_order_used.csv")
  )
}

# ── Report ─────────────────────────────────────────────────────────────────────

message("\n── Spatial prediction complete (2D tile) ────────────────────────")
message(sprintf("  Shard            : row %d/%d, col %d/%d",
                row_shard_id, n_row_shards, col_shard_id, n_col_shards))
message(sprintf("  Config / seeds   : %s / %d", config_id, n_seeds))
message(sprintf("  Valid pixels     : %s", format(n_valid_total, big.mark = ",")))
message(sprintf("  Runtime          : %.2f %s",
                as.numeric(total_time), units(total_time)))
message(sprintf("  strip_ncol       : %d (vs r_ncol %d -> %.1fx less RAM)",
                strip_ncol, r_ncol, r_ncol / strip_ncol))

rs_get <- function(layer, col) raster_summary[[col]][raster_summary$layer == layer]
message("\n  Central tendency (", target_unit, "):")
message(sprintf("    median map -> median ~%.2f | mean %.2f | max %.2f",
                global_med, rs_get("median", "gmean"), rs_get("median", "gmax")))
message(sprintf("    mean map   ->            mean %.2f | max %.2f",
                rs_get("mean", "gmean"), rs_get("mean", "gmax")))

message("\n  Rasters: ", output_raster_dir_w)
message("  Logs:    ", output_log_dir)
