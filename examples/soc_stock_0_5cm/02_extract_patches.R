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

predictor_raster_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"

# All window sizes to extract (in pixels, must be odd).
# Larger window = more spatial context. The maximum window here determines
# which profiles survive the edge filter — profiles too close to the raster
# boundary for the largest window are excluded for ALL windows.
# Having all sizes extracted now means you can freely tune window_sizes later
# without re-running this script.
window_sizes_to_extract <- c(3L, 5L, 7L)

# ── Memory / performance settings ─────────────────────────────────────────────
#
# The raster stack is never loaded entirely into RAM. Instead, horizontal strips
# of `chunk_nrows` rows are read, scaled, and used for patch extraction, then
# discarded before the next strip is loaded. This bounds peak RAM to one chunk
# (plus a buffer of half_w_max rows on each side).
#
# RAM per chunk (approximate):
#   bytes = (chunk_nrows + w_max - 1) × n_cols × n_channels × 8
#
# Examples with 187 channels:
#   Resolution  n_cols   chunk_nrows=100   chunk_nrows=500   chunk_nrows=2000
#   20 km        1800        ~0.3 GB           ~1.3 GB            ~5.0 GB
#   5  km        7200        ~1.0 GB           ~5.2 GB           ~20.7 GB
#   1  km       36000        ~5.2 GB          ~25.9 GB           [too large]
#
# Rule of thumb: set chunk_nrows so that one chunk uses ≤ 25% of your free RAM.
# Setting chunk_nrows = NULL auto-computes from max_ram_gb (see below).
chunk_nrows <- 200L

# max_ram_gb: if not NULL, chunk_nrows is computed automatically so each chunk
# stays within this memory budget. Overrides chunk_nrows when set.
# Example: max_ram_gb = 8 will size chunks to use ≤ 8 GB each.
max_ram_gb  <- NULL

# temperature QC threshold — must match script 01
temperature_min_valid_celsius <- -100

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

train_sdd         <- read_split("train")
validation_scaled <- read_split("validation")
test_scaled       <- read_split("test")

predictor_scaling <- readr::read_csv2(
  file.path(input_metadata_dir, "predictor_scaling.csv"),
  show_col_types = FALSE
)

predictor_cols <- predictor_scaling$predictor
n_channels     <- length(predictor_cols)

message("Predictors: ", n_channels)
message("Train rows: ", nrow(train_sdd),
        " | Validation: ", nrow(validation_scaled),
        " | Test: ", nrow(test_scaled))

# ── Load raster stack (metadata only — values are read chunk by chunk) ────────

message("\nOpening raster stack (metadata only)...")
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
keep_idx     <- raster_names %in% predictor_cols
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
        terra::nlyr(rast_stack), " layers  (opened in ",
        round(difftime(Sys.time(), t0, units = "secs"), 1), "s)")

# ── Resolve chunk_nrows ───────────────────────────────────────────────────────

half_w_max <- (max(window_sizes_to_extract) - 1L) / 2L

if (!is.null(max_ram_gb)) {
  bytes_per_row <- as.numeric(n_cols_rast) * n_channels * 8
  chunk_nrows   <- max(1L, as.integer(
    floor(max_ram_gb * 1e9 / bytes_per_row) - 2L * half_w_max
  ))
  message(sprintf(
    "Auto chunk_nrows = %d  (budget %.1f GB → %.2f GB/chunk incl. buffer)",
    chunk_nrows, max_ram_gb,
    (chunk_nrows + 2L * half_w_max) * bytes_per_row / 1e9
  ))
} else {
  bytes_per_chunk <- as.numeric(chunk_nrows + 2L * half_w_max) *
                     n_cols_rast * n_channels * 8
  message(sprintf(
    "chunk_nrows = %d  (%.2f GB per chunk incl. buffer)",
    chunk_nrows, bytes_per_chunk / 1e9
  ))
}

# ── Helper: apply channel-wise scaling to a raw sub-matrix ───────────────────
#
# mat: (n_cells × n_channels) — modified in place.
# Same logic as script 01: z-score for continuous, /100 for percentages,
# identity for dummies. Temperature QC applied before z-score.

.scale_matrix <- function(mat, pred_scaling, temp_min) {
  n_ch <- ncol(mat)
  for (i in seq_len(n_ch)) {
    method    <- pred_scaling$scaling_method[i]
    pred_name <- pred_scaling$predictor[i]
    x         <- mat[, i]

    if (grepl("surface_temperature_celsius$", pred_name)) {
      x[!is.na(x) & is.finite(x) & x <= temp_min] <- NA_real_
    }

    if (method == "zscore_train") {
      x <- (x - pred_scaling$train_mean[i]) / pred_scaling$train_sd[i]
    } else if (method == "percentage_0_100_to_0_1") {
      x[!is.na(x) & is.finite(x) & (x < 0 | x > 100)] <- NA_real_
      x <- x / 100
    }
    # "none_dummy_0_1": no transformation

    mat[, i] <- x
  }
  mat
}

# ── Helper: NA check within a chunk sub-matrix ────────────────────────────────
#
# For profiles whose center rows are LOCAL to sub_mat, checks whether any cell
# of their w×w patch contains a non-finite value across ALL channels.
# local_rows: row positions within sub_mat (1-indexed).
# Returns logical vector: TRUE = at least one NA/Inf in patch → invalid.

.na_check_chunk <- function(local_rows, col_ids, sub_mat, n_cols_rast, w) {
  half_w  <- (w - 1L) / 2L
  n       <- length(local_rows)
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)

  cell_mat <- matrix(0L, nrow = n, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cell_mat[, j] <- (local_rows + offsets$dr[j] - 1L) * n_cols_rast +
                      col_ids + offsets$dc[j]
  }

  all_cells  <- as.vector(t(cell_mat))
  na_mat     <- !is.finite(sub_mat[all_cells, , drop = FALSE])  # (n*n_pos) × n_ch
  na_per_pos <- rowSums(na_mat) > 0L                            # length n*n_pos
  colSums(matrix(na_per_pos, nrow = n_pos)) > 0L                # TRUE = invalid
}

# ── Helper: extract patch array from a chunk sub-matrix ───────────────────────
#
# local_rows: row positions within sub_mat. col_ids: global (unchanged).
# Returns (n × n_ch × w × w) array. Caller guarantees all patches are valid.

.extract_patches_chunk <- function(local_rows, col_ids, sub_mat, n_cols_rast, w) {
  half_w  <- (w - 1L) / 2L
  n_ch    <- ncol(sub_mat)
  n       <- length(local_rows)
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)

  cell_mat <- matrix(0L, nrow = n, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cell_mat[, j] <- (local_rows + offsets$dr[j] - 1L) * n_cols_rast +
                      col_ids + offsets$dc[j]
  }

  all_cells <- as.vector(t(cell_mat))
  vals      <- sub_mat[all_cells, , drop = FALSE]   # (n*n_pos) × n_ch

  step1 <- array(vals,  dim = c(n_pos, n, n_ch))    # [pos, prof, ch]
  step2 <- aperm(step1, c(2L, 3L, 1L))              # [prof, ch, pos]
  array(step2, dim = c(n, n_ch, w, w))              # [prof, ch, row, col]
}

# ── Process one split (chunked) ───────────────────────────────────────────────
#
# Reads the raster in strips of chunk_nrows rows. For each strip:
#   1. Reads strip + half_w_max-row buffer from disk.
#   2. Applies channel-wise scaling.
#   3. NA-checks patches of all window sizes for profiles in this strip.
#   4. Extracts patches only for profiles that passed all checks.
#   5. Frees the strip and GC before the next strip.
#
# The common valid set (profiles valid across ALL window sizes) is built
# incrementally: a profile is marked invalid as soon as any window's NA check
# fails. This is equivalent to the intersection computed in the original
# single-pass version.

process_split <- function(scaled_df, role,
                           n_rows_rast, n_cols_rast, n_ch,
                           window_sizes, chunk_nrows, half_w_max,
                           predictor_scaling, temperature_min_valid_celsius) {

  message("\n── Processing split: ", role, " (", nrow(scaled_df), " profiles) ──")
  n_profiles <- nrow(scaled_df)

  # Convert coordinates to raster row/col
  coords  <- as.matrix(scaled_df[, c("x", "y")])
  cells   <- terra::cellFromXY(rast_stack, coords)
  row_ids <- terra::rowFromCell(rast_stack, cells)
  col_ids <- terra::colFromCell(rast_stack, cells)

  # ── Phase 1: edge check (no data reading required) ──────────────────────────
  # A profile is edge-invalid if its largest window extends outside the raster.
  valid_common <- !is.na(row_ids) & !is.na(col_ids)
  for (w in window_sizes) {
    hw <- (w - 1L) / 2L
    valid_common <- valid_common &
      row_ids - hw >= 1L & row_ids + hw <= n_rows_rast &
      col_ids - hw >= 1L & col_ids + hw <= n_cols_rast
  }
  message("  After edge check: ", sum(valid_common), " / ", n_profiles,
          " profiles remain")

  if (!any(valid_common)) {
    stop("No valid profiles after edge check for split: ", role)
  }

  # ── Phase 2: pre-allocate patch arrays ────────────────────────────────────
  # Sized for ALL profiles; trimmed to valid ones at the end.
  # NA-initialised: any row not filled by a chunk means the profile was invalid.
  patch_list <- setNames(
    lapply(window_sizes, function(w) {
      array(NA_real_, dim = c(n_profiles, n_ch, w, w))
    }),
    paste0("x_", window_sizes, "x", window_sizes, "_array")
  )

  # ── Phase 3: chunked read → scale → NA check → extract ──────────────────────
  chunk_starts <- seq(1L, n_rows_rast, by = chunk_nrows)
  n_chunks     <- length(chunk_starts)

  for (ci in seq_along(chunk_starts)) {
    cs <- chunk_starts[ci]
    ce <- min(cs + chunk_nrows - 1L, n_rows_rast)

    # Profiles whose center row falls in this chunk and passed edge check
    in_chunk  <- valid_common & row_ids >= cs & row_ids <= ce
    if (!any(in_chunk)) next

    n_in      <- sum(in_chunk)
    chunk_idx <- which(in_chunk)

    # Buffered read: extra half_w_max rows above and below the chunk
    read_start <- max(1L,          cs - half_w_max)
    read_end   <- min(n_rows_rast, ce + half_w_max)
    read_nrows <- read_end - read_start + 1L

    message(sprintf(
      "  [%d/%d] chunk rows %d-%d  |  reading %d rows (buffer ±%d)  |  %d profiles",
      ci, n_chunks, cs, ce, read_nrows, half_w_max, n_in
    ))

    # Read and scale this strip
    sub_mat <- terra::values(rast_stack, row = read_start, nrows = read_nrows,
                             mat = TRUE)
    sub_mat <- .scale_matrix(sub_mat, predictor_scaling,
                             temperature_min_valid_celsius)

    # Local row coordinates within sub_mat
    local_rows  <- row_ids[chunk_idx] - read_start + 1L
    chunk_cols  <- col_ids[chunk_idx]

    # NA check for every window → update valid_common incrementally
    for (w in window_sizes) {
      has_na <- .na_check_chunk(local_rows, chunk_cols, sub_mat, n_cols_rast, w)
      valid_common[chunk_idx[has_na]] <- FALSE
    }

    # Extract patches only for profiles still valid after the NA check
    still_valid  <- valid_common[chunk_idx]
    valid_idx    <- chunk_idx[still_valid]
    valid_lrows  <- local_rows[still_valid]
    valid_cols   <- chunk_cols[still_valid]

    if (any(still_valid)) {
      for (w in window_sizes) {
        key <- paste0("x_", w, "x", w, "_array")
        patch_list[[key]][valid_idx, , , ] <-
          .extract_patches_chunk(valid_lrows, valid_cols, sub_mat, n_cols_rast, w)
      }
    }

    rm(sub_mat)
    invisible(gc(verbose = FALSE))
  }

  # ── Phase 4: trim arrays to valid profiles only ────────────────────────────
  n_valid   <- sum(valid_common)
  valid_idx <- which(valid_common)

  message("  Final valid profiles: ", n_valid, " / ", n_profiles,
          " (", round(100 * (n_profiles - n_valid) / n_profiles, 1),
          "% removed)")
  message("  Window NA check results:")
  for (w in window_sizes) {
    message("    ", w, "×", w, " valid after trim: ", n_valid)
  }

  for (key in names(patch_list)) {
    patch_list[[key]] <- patch_list[[key]][valid_idx, , , , drop = FALSE]
  }

  meta_valid <- scaled_df[valid_common, ] %>%
    dplyr::select(profile_id, sample_id, dataset_role, x, y,
                  target_native, target_log1p) %>%
    dplyr::rename(target_transform = target_log1p)

  c(patch_list, list(y = meta_valid$target_transform, meta = meta_valid))
}

# ── Run all splits ────────────────────────────────────────────────────────────

t_all <- Sys.time()

patches <- list(
  train = process_split(
    train_sdd, "train",
    n_rows_rast, n_cols_rast, n_channels,
    window_sizes_to_extract, chunk_nrows, half_w_max,
    predictor_scaling, temperature_min_valid_celsius
  ),
  validation = process_split(
    validation_scaled, "validation",
    n_rows_rast, n_cols_rast, n_channels,
    window_sizes_to_extract, chunk_nrows, half_w_max,
    predictor_scaling, temperature_min_valid_celsius
  ),
  test = process_split(
    test_scaled, "test",
    n_rows_rast, n_cols_rast, n_channels,
    window_sizes_to_extract, chunk_nrows, half_w_max,
    predictor_scaling, temperature_min_valid_celsius
  )
)

message("\nTotal extraction time: ",
        round(difftime(Sys.time(), t_all, units = "mins"), 1), " min")

# ── Manifest ──────────────────────────────────────────────────────────────────

manifest <- tibble::tibble(
  target_label                  = target_label,
  n_channels                    = n_channels,
  window_sizes_extracted        = paste(window_sizes_to_extract, collapse = ", "),
  chunk_nrows_used              = chunk_nrows,
  n_train_valid                 = nrow(patches$train$meta),
  n_validation_valid            = nrow(patches$validation$meta),
  n_test_valid                  = nrow(patches$test$meta),
  n_train_input                 = nrow(train_sdd),
  n_validation_input            = nrow(validation_scaled),
  n_test_input                  = nrow(test_scaled),
  pct_train_removed             = round(100 * (nrow(train_sdd)         - nrow(patches$train$meta))      / nrow(train_sdd),         1),
  pct_validation_removed        = round(100 * (nrow(validation_scaled) - nrow(patches$validation$meta)) / nrow(validation_scaled), 1),
  pct_test_removed              = round(100 * (nrow(test_scaled)       - nrow(patches$test$meta))       / nrow(test_scaled),       1),
  input_scaling                 = "predictor_scaling.csv",
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  predictor_cols_final          = paste(predictor_cols, collapse = ";"),
  raster_nrow                   = n_rows_rast,
  raster_ncol                   = n_cols_rast
)

message("\n── Manifest ──────────────────────────────")
print(manifest[, c("n_train_valid", "n_validation_valid", "n_test_valid",
                   "pct_train_removed", "pct_validation_removed", "pct_test_removed",
                   "chunk_nrows_used")],
      width = Inf)

# Array size report
message("\n── Array sizes ───────────────────────────")
for (role in c("train", "validation", "test")) {
  for (w in window_sizes_to_extract) {
    key <- paste0("x_", w, "x", w, "_array")
    arr <- patches[[role]][[key]]
    if (!is.null(arr)) {
      sz <- format(object.size(arr), units = "MB")
      message("  ", role, " ", key, ": ",
              paste(dim(arr), collapse = " × "), "  (", sz, ")")
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
