source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "terra",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "janitor"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"

predictor_raster_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_20000m"

# All window sizes to extract (in pixels, must be odd).
# Larger window = more spatial context. The maximum window here determines
# which profiles survive the edge filter — profiles too close to the raster
# boundary for the largest window are excluded for ALL windows.
# Having all sizes extracted now means you can freely tune window_sizes later
# without re-running this script.
window_sizes_to_extract <- c(3L, 5L, 7L)

# ── Paths ─────────────────────────────────────────────────────────────────────

input_data_dir     <- file.path(project_root, "data",    "processed", "soc_stock_modeling", target_label)
input_metadata_dir <- file.path(project_root, "outputs", "metadata",  "soc_stock_modeling", target_label)
output_patch_dir   <- file.path(project_root, "outputs", "patches",   "soc_stock_modeling", target_label)
output_metadata_patch_dir <- file.path(input_metadata_dir, "patches")

create_output_dirs(c(output_patch_dir, output_metadata_patch_dir))

# ── Read scaled datasets and scaling table ────────────────────────────────────

message("Reading scaled datasets...")

read_split <- function(role) {
  readr::read_csv2(
    file.path(input_data_dir, paste0(role, "_scaled.csv")),
    show_col_types = FALSE
  )
}

train_scaled      <- read_split("train")
validation_scaled <- read_split("validation")
test_scaled       <- read_split("test")

predictor_scaling <- readr::read_csv2(
  file.path(input_metadata_dir, "predictor_scaling.csv"),
  show_col_types = FALSE
)

predictor_cols <- predictor_scaling$predictor
n_channels     <- length(predictor_cols)

message("Predictors: ", n_channels)
message("Train rows: ", nrow(train_scaled),
        " | Validation: ", nrow(validation_scaled),
        " | Test: ", nrow(test_scaled))

# ── Load raster stack ─────────────────────────────────────────────────────────
# Only .tif files at the root level (ignores subdirectories like 'nused')

message("\nLoading raster stack...")
t0 <- Sys.time()

raster_files <- list.files(
  predictor_raster_dir,
  pattern    = "\\.tif$",
  full.names  = TRUE,
  recursive   = FALSE
)

raster_names <- janitor::make_clean_names(
  tools::file_path_sans_ext(basename(raster_files))
)

# Keep only rasters that are in predictor_cols (same set as script 01)
keep_idx   <- raster_names %in% predictor_cols
raster_files <- raster_files[keep_idx]
raster_names <- raster_names[keep_idx]

# Sort to match predictor_cols order (critical for channel alignment)
order_idx    <- match(predictor_cols, raster_names)
raster_files <- raster_files[order_idx]
raster_names <- raster_names[order_idx]

stopifnot(identical(raster_names, predictor_cols))

rast_stack <- terra::rast(raster_files)
names(rast_stack) <- raster_names

n_rows_rast <- terra::nrow(rast_stack)
n_cols_rast <- terra::ncol(rast_stack)

message("Raster grid: ", n_rows_rast, " rows × ", n_cols_rast, " cols × ",
        terra::nlyr(rast_stack), " layers")

# Load all values into RAM — with 64 GB this is fine.
# Result: (n_rows_rast * n_cols_rast) × n_channels matrix.
# Channels are in predictor_cols order.
message("Loading raster values into RAM (this is the slow step)...")
vals_matrix <- terra::values(rast_stack)

message("Values loaded: ", format(object.size(vals_matrix), units = "MB"),
        " in ", round(difftime(Sys.time(), t0, units = "secs"), 1), "s")

# ── Apply predictor scaling to raw raster values ──────────────────────────────
# Scales vals_matrix in-place (one column = one predictor channel) using the
# same parameters computed from the training split in script 01.
# After this step, vals_matrix holds the SAME representation that the CNN will
# see during training — which is what spatial prediction must replicate too.
#
# Temperature QC is applied before z-scoring: values at or below
# temperature_min_valid_celsius are set to NA (same threshold as script 01).
# Dummy predictors (scaling_method == "none_dummy_0_1") are left unchanged.

temperature_min_valid_celsius <- -100   # must match script 01

message("\nApplying predictor scaling to raster values matrix (", n_channels, " channels)...")
t_scale <- Sys.time()

for (i in seq_len(n_channels)) {
  method    <- predictor_scaling$scaling_method[i]
  pred_name <- predictor_scaling$predictor[i]
  x         <- vals_matrix[, i]

  if (grepl("surface_temperature_celsius$", pred_name)) {
    x[!is.na(x) & is.finite(x) & x <= temperature_min_valid_celsius] <- NA_real_
  }

  if (method == "zscore_train") {
    x <- (x - predictor_scaling$train_center[i]) / predictor_scaling$train_scale[i]
  } else if (method == "percentage_0_100_to_0_1") {
    x[!is.na(x) & is.finite(x) & (x < 0 | x > 100)] <- NA_real_
    x <- x / 100
  }
  # "none_dummy_0_1": no transformation

  vals_matrix[, i] <- x
}

message("Scaling applied in ",
        round(difftime(Sys.time(), t_scale, units = "secs"), 1), "s")

# ── Helper: per-window validity mask ──────────────────────────────────────────
#
# Returns a logical vector (length = length(row_ids)): TRUE if the w×w patch
# centered on that profile is fully inside the raster AND has no NA in any
# cell/channel. Used to build a COMMON valid set across all window sizes so
# every window array and the target vector share identical row order.
#
# Why a common set: the NA check can remove DIFFERENT profiles per window
# (e.g., a coastal profile may have a clean 3×3 but a 7×7 that reaches an
# ocean NA cell). Extracting each window on its own valid set would silently
# misalign the arrays. Computing the intersection up front prevents that.

window_valid_mask <- function(row_ids, col_ids, vals_mat,
                              n_rows_rast, n_cols_rast, w) {
  half_w <- (w - 1L) / 2L

  edge_ok <- !is.na(row_ids) & !is.na(col_ids) &
    row_ids - half_w >= 1L & row_ids + half_w <= n_rows_rast &
    col_ids - half_w >= 1L & col_ids + half_w <= n_cols_rast

  valid <- edge_ok
  idx   <- which(edge_ok)
  if (length(idx) == 0L) return(valid)

  v_rows  <- row_ids[idx]
  v_cols  <- col_ids[idx]
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)

  cell_mat <- matrix(0L, nrow = length(idx), ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cell_mat[, j] <- (v_rows + offsets$dr[j] - 1L) * n_cols_rast +
                      v_cols + offsets$dc[j]
  }

  # all_cells ordering: [prof1pos1..prof1posN, prof2pos1..prof2posN, ...]
  all_cells <- as.vector(t(cell_mat))
  na_mat    <- !is.finite(vals_mat[all_cells, , drop = FALSE])  # (n_idx*n_pos) × n_ch

  na_per_pos     <- rowSums(na_mat) > 0L                        # length n_idx*n_pos
  na_per_profile <- colSums(matrix(na_per_pos, nrow = n_pos)) > 0L  # length n_idx

  valid[idx[na_per_profile]] <- FALSE
  valid
}

# ── Helper: build patch array for an EXACT valid set ──────────────────────────
#
# Builds an (n × n_ch × w × w) array for the profiles flagged TRUE in `valid`,
# which the caller guarantees are inside the raster and NA-free.
#
# Spatial layout: patches[prof, ch, r, col] corresponds to row offset
# dr = r - half - 1 and col offset dc = col - half - 1 (audited against the
# expand.grid offset order). This matches PyTorch's (C, H, W) convention.

build_patch_array <- function(row_ids, col_ids, vals_mat, n_cols_rast, w, valid) {
  half_w <- (w - 1L) / 2L
  n_ch   <- ncol(vals_mat)
  idx    <- which(valid)
  n      <- length(idx)
  if (n == 0L) return(NULL)

  v_rows  <- row_ids[idx]
  v_cols  <- col_ids[idx]
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)

  cell_mat <- matrix(0L, nrow = n, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cell_mat[, j] <- (v_rows + offsets$dr[j] - 1L) * n_cols_rast +
                      v_cols + offsets$dc[j]
  }

  all_cells <- as.vector(t(cell_mat))
  vals      <- vals_mat[all_cells, , drop = FALSE]   # (n*n_pos) × n_ch

  step1 <- array(vals,  dim = c(n_pos, n, n_ch))     # [pos, prof, ch]
  step2 <- aperm(step1, c(2L, 3L, 1L))               # [prof, ch, pos]
  array(step2, dim = c(n, n_ch, w, w))               # [prof, ch, row, col]
}

# ── Process one split ─────────────────────────────────────────────────────────

process_split <- function(scaled_df, role, vals_mat,
                           n_rows_rast, n_cols_rast,
                           window_sizes, predictor_cols) {
  message("\n── Processing split: ", role, " (", nrow(scaled_df), " profiles) ──")

  # Convert coordinates to raster row/col
  coords  <- as.matrix(scaled_df[, c("x", "y")])
  cells   <- terra::cellFromXY(rast_stack, coords)
  row_ids <- terra::rowFromCell(rast_stack, cells)
  col_ids <- terra::colFromCell(rast_stack, cells)

  # One COMMON valid set across all windows: a profile is kept only if EVERY
  # requested window is inside the raster AND NA-free. Guarantees that all
  # window arrays and the target vector are row-aligned.
  valid_common <- rep(TRUE, nrow(scaled_df))
  for (w in window_sizes) {
    vw <- window_valid_mask(row_ids, col_ids, vals_mat,
                            n_rows_rast, n_cols_rast, w)
    valid_common <- valid_common & vw
    message("  window ", w, "×", w, " valid: ", sum(vw),
            "  | running intersection: ", sum(valid_common))
  }

  n_valid <- sum(valid_common)
  message("  Final valid profiles: ", n_valid, " / ", nrow(scaled_df),
          " (", round(100 * (nrow(scaled_df) - n_valid) / nrow(scaled_df), 1),
          "% removed)")

  # Build each window array on the SAME valid set → identical N and row order
  patch_list <- list()
  for (w in window_sizes) {
    t_w <- Sys.time()
    key <- paste0("x_", w, "x", w, "_array")
    patch_list[[key]] <- build_patch_array(row_ids, col_ids, vals_mat,
                                           n_cols_rast, w, valid_common)
    message("  built ", key, " in ",
            round(difftime(Sys.time(), t_w, units = "secs"), 1), "s")
  }

  # Target and metadata for the common valid set (same order as the arrays)
  meta_valid <- scaled_df[valid_common, ] |>
    dplyr::select(profile_id, sample_id, dataset_role, x, y,
                  target_native, target_log1p) |>
    dplyr::rename(target_transform = target_log1p)

  c(patch_list, list(y = meta_valid$target_transform, meta = meta_valid))
}

# ── Run all splits ────────────────────────────────────────────────────────────

t_all <- Sys.time()

patches <- list(
  train      = process_split(train_scaled,      "train",      vals_matrix, n_rows_rast, n_cols_rast, window_sizes_to_extract, predictor_cols),
  validation = process_split(validation_scaled, "validation", vals_matrix, n_rows_rast, n_cols_rast, window_sizes_to_extract, predictor_cols),
  test       = process_split(test_scaled,       "test",       vals_matrix, n_rows_rast, n_cols_rast, window_sizes_to_extract, predictor_cols)
)

message("\nTotal extraction time: ",
        round(difftime(Sys.time(), t_all, units = "mins"), 1), " min")

# ── Manifest ──────────────────────────────────────────────────────────────────

manifest <- tibble::tibble(
  target_label            = target_label,
  n_channels              = n_channels,
  window_sizes_extracted  = paste(window_sizes_to_extract, collapse = ", "),
  n_train_valid           = nrow(patches$train$meta),
  n_validation_valid      = nrow(patches$validation$meta),
  n_test_valid            = nrow(patches$test$meta),
  n_train_input           = nrow(train_scaled),
  n_validation_input      = nrow(validation_scaled),
  n_test_input            = nrow(test_scaled),
  pct_train_removed       = round(100 * (nrow(train_scaled)      - nrow(patches$train$meta))      / nrow(train_scaled),      1),
  pct_validation_removed  = round(100 * (nrow(validation_scaled) - nrow(patches$validation$meta)) / nrow(validation_scaled), 1),
  pct_test_removed        = round(100 * (nrow(test_scaled)       - nrow(patches$test$meta))       / nrow(test_scaled),       1),
  input_scaling           = "predictor_scaling.csv",
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  predictor_cols_final    = paste(predictor_cols, collapse = ";"),
  raster_nrow             = n_rows_rast,
  raster_ncol             = n_cols_rast
)

message("\n── Manifest ──────────────────────────────")
print(manifest[, c("n_train_valid", "n_validation_valid", "n_test_valid",
                   "pct_train_removed", "pct_validation_removed", "pct_test_removed")],
      width = Inf)

# Array size report
message("\n── Array sizes ───────────────────────────")
for (role in c("train", "validation", "test")) {
  for (w in window_sizes_to_extract) {
    key <- paste0("x_", w, "x", w, "_array")
    arr <- patches[[role]][[key]]
    if (!is.null(arr)) {
      sz <- format(object.size(arr), units = "MB")
      message("  ", role, " ", key, ": ", paste(dim(arr), collapse = " × "), "  (", sz, ")")
    }
  }
}

# ── Save ──────────────────────────────────────────────────────────────────────

message("\nSaving patches...")
safe_save_rds(patches,  file.path(output_patch_dir, "patches_all_splits.rds"),  compress = FALSE)
safe_save_rds(manifest, file.path(output_patch_dir, "patch_manifest.rds"),      compress = FALSE)
safe_write_csv2(manifest, file.path(output_metadata_patch_dir, "patch_manifest.csv"))

for (role in c("train", "validation", "test")) {
  safe_write_csv2(
    patches[[role]]$meta,
    file.path(output_metadata_patch_dir, paste0("meta_", role, ".csv"))
  )
}

message("\nPatches saved to: ", output_patch_dir)
message("Done.")
