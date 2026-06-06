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

# ── Helper: vectorised patch extraction ───────────────────────────────────────
#
# Returns an array of shape (n_valid × n_channels × w × w) with scaled values,
# or NULL if no profiles pass the edge/NA filter.
#
# The spatial layout of the patch (H × W dimensions) follows:
#   row offset: -half .. 0 .. +half  (top to bottom)
#   col offset: -half .. 0 .. +half  (left to right)
# which matches PyTorch's (C, H, W) convention when converted to tensor.
#
# All arithmetic is done in RAM using matrix indexing — no R loop per profile.

extract_patches_vectorised <- function(row_ids, col_ids, vals_mat,
                                        n_rows_rast, n_cols_rast, w) {
  half_w <- (w - 1L) / 2L
  n_ch   <- ncol(vals_mat)

  # Edge filter: profile must be at least half_w cells from every border
  valid <- !is.na(row_ids) & !is.na(col_ids) &
    row_ids - half_w >= 1L & row_ids + half_w <= n_rows_rast &
    col_ids - half_w >= 1L & col_ids + half_w <= n_cols_rast

  n_valid <- sum(valid)
  if (n_valid == 0L) return(list(patches = NULL, valid_mask = valid))

  v_rows <- row_ids[valid]
  v_cols <- col_ids[valid]

  # Offsets for all w×w patch positions
  # expand.grid: dr varies slowest → patch rows top-to-bottom,
  # dc varies fastest → patch cols left-to-right
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)   # = w * w

  # Cell-index matrix: n_valid × n_pos
  # cell = (row - 1) * n_cols_rast + col
  cell_mat <- matrix(0L, nrow = n_valid, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cell_mat[, j] <- (v_rows + offsets$dr[j] - 1L) * n_cols_rast +
                      v_cols + offsets$dc[j]
  }

  # Single large extraction: (n_valid * n_pos) × n_ch
  # Rows of cell_mat are in profile order, so as.vector(t(cell_mat))
  # gives: all positions for profile 1, then profile 2, etc.
  all_cells <- as.vector(t(cell_mat))
  all_vals  <- vals_mat[all_cells, , drop = FALSE]  # (n_valid*n_pos) × n_ch

  # Check for any NA across spatial patch for each profile:
  # NA in any cell → exclude profile
  has_na <- apply(
    matrix(!is.finite(all_vals), nrow = n_pos, ncol = n_valid * n_ch),
    2L,
    any
  )
  # Reshape has_na: we need per-profile, not per (profile × channel)
  # Actually let's re-check: all_vals rows: [pos1_prof1, pos2_prof1, ..., pos_n_prof1, pos1_prof2, ...]
  # Check per profile: any NA in its n_pos * n_ch values
  na_matrix  <- !is.finite(all_vals)   # (n_valid * n_pos) × n_ch
  # Group by profile: profile i → rows ((i-1)*n_pos + 1) .. (i*n_pos)
  na_per_profile <- vapply(seq_len(n_valid), function(i) {
    rows_i <- ((i - 1L) * n_pos + 1L):(i * n_pos)
    any(na_matrix[rows_i, , drop = FALSE])
  }, logical(1L))

  # Update valid mask
  valid_indices       <- which(valid)
  valid[valid_indices[na_per_profile]] <- FALSE

  n_clean <- sum(!na_per_profile)
  if (n_clean == 0L) return(list(patches = NULL, valid_mask = valid))

  clean_rows_in_allvals <- rep(!na_per_profile, each = n_pos)
  clean_vals <- all_vals[clean_rows_in_allvals, , drop = FALSE]  # (n_clean*n_pos) × n_ch

  # Reshape to (n_clean × n_ch × w × w):
  # array(clean_vals, dim = c(n_pos, n_clean, n_ch))[pos, prof, ch] = clean_vals[(prof-1)*n_pos+pos, ch] ✓
  # Then aperm to (n_clean, n_ch, n_pos) then reshape to (n_clean, n_ch, w, w)
  step1   <- array(clean_vals, dim = c(n_pos, n_clean, n_ch))  # [pos, prof, ch]
  step2   <- aperm(step1, c(2L, 3L, 1L))                       # [prof, ch, pos]
  patches <- array(step2, dim = c(n_clean, n_ch, w, w))        # [prof, ch, row, col]

  list(patches = patches, valid_mask = valid)
}

# ── Process one split ─────────────────────────────────────────────────────────

process_split <- function(scaled_df, role, vals_mat,
                           n_rows_rast, n_cols_rast,
                           window_sizes, predictor_cols) {
  message("\n── Processing split: ", role, " (", nrow(scaled_df), " profiles) ──")

  # Convert coordinates to raster row/col
  coords <- as.matrix(scaled_df[, c("x", "y")])
  cells  <- terra::cellFromXY(rast_stack, coords)
  row_ids <- terra::rowFromCell(rast_stack, cells)
  col_ids <- terra::colFromCell(rast_stack, cells)

  # Determine valid profiles: use the maximum window for the strictest edge filter.
  # A profile valid for max_w is valid for all smaller windows.
  max_w    <- max(window_sizes)
  half_max <- (max_w - 1L) / 2L

  valid_base <- !is.na(row_ids) & !is.na(col_ids) &
    row_ids - half_max >= 1L & row_ids + half_max <= n_rows_rast &
    col_ids - half_max >= 1L & col_ids + half_max <= n_cols_rast

  # Extract patches for each window size (only for base-valid profiles)
  # All windows share the same valid set so arrays are aligned.
  patch_list  <- list()
  valid_final <- valid_base

  for (w in window_sizes) {
    message("  Extracting ", w, "×", w, " patches...")
    t_w <- Sys.time()

    # Pass only base-valid rows to the extractor
    res <- extract_patches_vectorised(
      row_ids     = ifelse(valid_base, row_ids, NA_integer_),
      col_ids     = ifelse(valid_base, col_ids, NA_integer_),
      vals_mat    = vals_mat,
      n_rows_rast = n_rows_rast,
      n_cols_rast = n_cols_rast,
      w           = w
    )

    valid_final <- res$valid_mask   # narrowed by NA check
    key         <- paste0("x_", w, "x", w, "_array")
    patch_list[[key]] <- res$patches

    message("    Done in ", round(difftime(Sys.time(), t_w, units = "secs"), 1), "s",
            " | valid profiles: ", sum(valid_final), " / ", nrow(scaled_df))
  }

  # All arrays must have the same N — use the final valid mask (strictest intersection)
  n_valid <- sum(valid_final)
  message("  Final valid profiles: ", n_valid, " / ", nrow(scaled_df),
          " (", round(100 * (nrow(scaled_df) - n_valid) / nrow(scaled_df), 1),
          "% removed)")

  # Trim arrays to final valid set (some NA check may have narrowed further)
  # valid_base profiles that survived NA check are in valid_final
  valid_in_base <- which(valid_final)[seq_len(n_valid)]  # index into scaled_df

  for (key in names(patch_list)) {
    arr <- patch_list[[key]]
    if (!is.null(arr) && dim(arr)[1] != n_valid) {
      patch_list[[key]] <- arr[seq_len(n_valid), , , , drop = FALSE]
    }
  }

  # Target and metadata for valid profiles only
  meta_valid <- scaled_df[valid_final, ] |>
    dplyr::select(profile_id, sample_id, dataset_role, x, y,
                  target_native, target_log1p) |>
    dplyr::rename(target_transform = target_log1p)

  y_vec <- meta_valid$target_transform

  c(patch_list, list(y = y_vec, meta = meta_valid))
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
