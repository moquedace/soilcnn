source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
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
#
# A window's physical extent is window_size × raster resolution, so re-pick it
# whenever the resolution changes. At 250 m:
#   3×3  ≈ 0.75 km  (immediate neighbourhood, land cover, micro-relief)
#   9×9  ≈ 2.25 km  (hillslope, drainage, soil–landscape position)
#   15×15 ≈ 3.75 km (local landscape, catchment context, local climate)
window_sizes_to_extract <- c(3L, 9L, 15L)

# ── Memory / performance settings ─────────────────────────────────────────────
#
# Strategy: read ONE BAND AT A TIME per spatial chunk.
# This eliminates the std::bad_alloc caused by allocating all bands together
# (n_rows × n_cols × n_channels) which for fine-resolution global rasters can
# exceed 20+ GB in a single contiguous block — a block that heap fragmentation
# makes unavailable after many alloc/free cycles, even with plenty of total RAM.
#
# Memory per band read:
#   bytes = (chunk_nrows + 2 × half_w_max) × n_cols × 8
#
# Examples (single band, no n_channels factor):
#   Resolution  n_cols    chunk_nrows=500   chunk_nrows=2000
#   20 km        1800       ~0.015 GB         ~0.056 GB
#   5  km        7200       ~0.058 GB         ~0.230 GB
#   250 m       160000      ~1.3 GB           ~5.0 GB
#   1  km        36000      ~0.29 GB          ~1.15 GB
#
# Peak RAM ≈ one band strip + the pre-allocated patch arrays. The patch arrays
# (n_profiles × n_channels × Σ(w²) × 8 bytes, summed over splits) usually
# dominate the strip, but stay modest: ~5 GB for windows 3/5/7 and ~17 GB for
# 3/9/15 at ~37k profiles and 187 channels. The exact figure is computed and
# printed at runtime below; it depends only on profile count and the chosen
# windows, NOT on raster resolution.
#
# chunk_nrows trade-off:
#   Larger → fewer spatial chunks → fewer band reads → faster total runtime.
#   Smaller → smaller per-band allocation → less heap pressure.
#
# For 250 m global rasters (n_cols ≈ 160 000):
#   chunk_nrows = 500 → ~1.3 GB per band read  (recommended)
#   chunk_nrows = 200 → ~0.5 GB per band read  (very safe, more I/O calls)
#   chunk_nrows = 100 → ~0.26 GB per band read (maximum safety)
#
# Set chunk_nrows = NULL and max_ram_gb to auto-compute from a per-band budget.
chunk_nrows <- 200

# max_ram_gb: auto-sizes chunk_nrows so each SINGLE-BAND strip stays within
# this limit. Overrides chunk_nrows when not NULL.
# Note: this is PER BAND (not all bands combined), so values like 1-4 are fine.
max_ram_gb  <- NULL

# temperature QC threshold — must match script 01
temperature_min_valid_celsius <- -100

# ── Paths ─────────────────────────────────────────────────────────────────────

input_data_dir     <- file.path(project_root, "data",    "processed", "soc_stock_modeling", target_label)
input_metadata_dir <- file.path(project_root, "outputs", "metadata",  "soc_stock_modeling", target_label)
output_patch_dir   <- file.path(project_root, "outputs", "patches",   "soc_stock_modeling", target_label)
output_metadata_patch_dir <- file.path(input_metadata_dir, "patches")

create_output_dirs(c(output_patch_dir, output_metadata_patch_dir))

# ── Read split point tables and scaling table ─────────────────────────────────
# We read the *_raw.csv split files, not *_scaled.csv: this script only needs
# each profile's coordinates (x, y) and target, then re-extracts the predictor
# PATCHES directly from the rasters and scales them itself (.scale_band_vec).
# The pre-scaled predictor columns in *_scaled.csv would just be ignored, so the
# raw file is the honest input here. Scaling parameters come from
# predictor_scaling.csv (training-split statistics from script 01).

message("Reading split point tables...")

read_split <- function(role) {
  readr::read_csv2(
    file.path(input_data_dir, paste0(role, "_raw.csv")),
    show_col_types = FALSE
  )
}

train_pts      <- read_split("train")
validation_pts <- read_split("validation")
test_pts       <- read_split("test")

predictor_scaling <- readr::read_csv2(
  file.path(input_metadata_dir, "predictor_scaling.csv"),
  show_col_types = FALSE
)

predictor_cols <- predictor_scaling$predictor
n_channels     <- length(predictor_cols)

message("Predictors: ", n_channels)
message("Train rows: ", nrow(train_pts),
        " | Validation: ", nrow(validation_pts),
        " | Test: ", nrow(test_pts))

# ── Open raster stack (metadata only — values read band-by-band) ──────────────

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

keep_idx     <- raster_names %in% predictor_cols
raster_files <- raster_files[keep_idx]
raster_names <- raster_names[keep_idx]

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
        sprintf("%.2f %s", Sys.time() - t0, units(Sys.time() - t0)), ")")

# ── Resolve chunk_nrows ───────────────────────────────────────────────────────

half_w_max <- (max(window_sizes_to_extract) - 1L) / 2L

bytes_per_band_row <- as.numeric(n_cols_rast) * 8   # single band, double precision

if (!is.null(max_ram_gb)) {
  chunk_nrows <- max(1L, as.integer(
    floor(max_ram_gb * 1e9 / bytes_per_band_row) - 2L * half_w_max
  ))
  message(sprintf(
    "Auto chunk_nrows = %d  (per-band budget %.1f GB → %.2f GB/band-read)",
    chunk_nrows, max_ram_gb,
    (chunk_nrows + 2L * half_w_max) * bytes_per_band_row / 1e9
  ))
} else {
  message(sprintf(
    "chunk_nrows = %d  (%.3f GB per single-band read incl. buffer)",
    chunk_nrows,
    (chunk_nrows + 2L * half_w_max) * bytes_per_band_row / 1e9
  ))
}

# Patch arrays held in RAM: every split keeps one array per window for ALL its
# profiles, dims [n_profiles, n_channels, w, w]. The final `patches` list holds
# train + validation + test simultaneously and is saved uncompressed. This is
# the dominant RAM cost — far larger than one band strip — and it scales with
# the number of windows, NOT with raster resolution.
patch_array_gb <- (nrow(train_pts) + nrow(validation_pts) + nrow(test_pts)) *
  n_channels * sum(window_sizes_to_extract^2) * 8 / 1e9

message(sprintf(
  "Total spatial chunks: %d  |  Band reads per chunk-with-profiles: %d",
  ceiling(n_rows_rast / chunk_nrows), n_channels
))
message(sprintf(
  "Peak RAM ~ %.2f GB (one band strip) + ~%.1f GB (patch arrays, all splits, windows %s)",
  (chunk_nrows + 2L * half_w_max) * bytes_per_band_row / 1e9,
  patch_array_gb,
  paste(window_sizes_to_extract, collapse = "/")
))
if (patch_array_gb > 25) {
  message(sprintf(
    "  NOTE: patch arrays need ~%.0f GB. To reduce, set window_sizes_to_extract to only the window(s) you will train on.",
    patch_array_gb
  ))
}

# ── Helper: scale a single-band vector ────────────────────────────────────────
#
# Applies the same QC and transform used in script 01 for band i.
# vec: numeric vector of length read_nrows × n_cols (row-major, single band).

.scale_band_vec <- function(vec, band_idx, pred_scaling, temp_min) {
  method    <- pred_scaling$scaling_method[band_idx]
  pred_name <- pred_scaling$predictor[band_idx]

  if (grepl("surface_temperature_celsius$", pred_name)) {
    vec[!is.na(vec) & is.finite(vec) & vec <= temp_min] <- NA_real_
  }

  if (method == "zscore_train") {
    vec <- (vec - pred_scaling$train_mean[band_idx]) / pred_scaling$train_sd[band_idx]
  } else if (method == "percentage_0_100_to_0_1") {
    # Clamp into [0, 100] instead of discarding as NA: these are continuous
    # interpolated surfaces (PNV classes, clay mineralogy) that legitimately
    # overshoot slightly past 0/100 near sharp spatial transitions. Treating
    # that as missing data amplifies into large gaps once the CNN's
    # full-window validity rule invalidates the whole patch around each
    # discarded pixel. Genuine NA/Inf pass through untouched.
    finite_idx <- !is.na(vec) & is.finite(vec)
    vec[finite_idx] <- pmin(pmax(vec[finite_idx], 0), 100)
    vec <- vec / 100
  }
  # "none_dummy_0_1": no transformation

  vec
}

# ── Helper: build cell-index matrix for a window ──────────────────────────────
#
# Returns an (n_profiles × n_pos) integer matrix of LINEAR cell indices within
# a band strip (row-major, local coordinates).
# local_rows: 1-based row positions within the strip.
# col_ids:    global (unchanged across strips).

.cell_index_mat <- function(local_rows, col_ids, n_cols_rast, w) {
  half_w  <- (w - 1L) / 2L
  n       <- length(local_rows)
  offsets <- expand.grid(dr = (-half_w):half_w, dc = (-half_w):half_w)
  n_pos   <- nrow(offsets)

  cm <- matrix(0L, nrow = n, ncol = n_pos)
  for (j in seq_len(n_pos)) {
    cm[, j] <- (local_rows + offsets$dr[j] - 1L) * n_cols_rast +
                col_ids + offsets$dc[j]
  }
  cm
}

# ── Process one split (band-by-band chunked) ───────────────────────────────────
#
# For each spatial chunk (rows cs:ce) that contains at least one profile:
#   1. Pre-compute cell index matrices for each window — once per chunk.
#   2. Loop over all 187 bands:
#      a. Read one band strip: (read_nrows × n_cols) values ≈ few hundred MB.
#      b. Scale the strip.
#      c. Vectorised: extract patch values for all profiles in the chunk.
#      d. NA check → update valid_common.
#      e. Store in patch_list[profiles, band, spatial positions].
#      f. rm + gc() — releases the band strip before reading the next.
#   3. After all chunks: trim patch arrays to valid profiles only.
#
# Peak RAM at any step: one band strip + patch_list (the latter dominates, see
# the runtime estimate printed above). Never holds more than one band's worth of
# raster rows in memory at a time — that is what fixes the std::bad_alloc.

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

  # ── Phase 1: edge check (no data reading required) ────────────────────────
  valid_common <- !is.na(row_ids) & !is.na(col_ids)
  for (w in window_sizes) {
    hw <- (w - 1L) / 2L
    valid_common <- valid_common &
      row_ids - hw >= 1L & row_ids + hw <= n_rows_rast &
      col_ids - hw >= 1L & col_ids + hw <= n_cols_rast
  }
  message("  After edge check: ", sum(valid_common), " / ", n_profiles)

  if (!any(valid_common)) stop("No valid profiles after edge check: ", role)

  # ── Phase 2: pre-allocate patch arrays ──────────────────────────────────────
  # NA-initialised; trimmed at the end. Sized for ALL profiles (not just valid)
  # so that chunk_idx can be used as direct indices without remapping.
  patch_list <- setNames(
    lapply(window_sizes, function(w) array(NA_real_, dim = c(n_profiles, n_ch, w, w))),
    paste0("x_", window_sizes, "x", window_sizes, "_array")
  )

  # ── Phase 3: spatial chunk loop → band loop ──────────────────────────────────
  chunk_starts       <- seq(1L, n_rows_rast, by = chunk_nrows)
  n_chunks           <- length(chunk_starts)
  n_chunks_processed <- 0L

  for (ci in seq_along(chunk_starts)) {
    cs <- chunk_starts[ci]
    ce <- min(cs + chunk_nrows - 1L, n_rows_rast)

    in_chunk  <- valid_common & row_ids >= cs & row_ids <= ce
    if (!any(in_chunk)) next

    n_in      <- sum(in_chunk)
    chunk_idx <- which(in_chunk)
    n_chunks_processed <- n_chunks_processed + 1L

    read_start <- max(1L,          cs - half_w_max)
    read_end   <- min(n_rows_rast, ce + half_w_max)
    read_nrows <- read_end - read_start + 1L

    message(sprintf(
      "  [chunk %d/%d] rows %d-%d  |  %d profiles  |  reading %d rows × %d bands",
      ci, n_chunks, cs, ce, n_in, read_nrows, n_ch
    ))

    local_rows <- row_ids[chunk_idx] - read_start + 1L
    chunk_cols <- col_ids[chunk_idx]

    # Pre-compute cell index matrices once per spatial chunk (reused for every band)
    cell_mats <- lapply(window_sizes, .cell_index_mat,
                        local_rows = local_rows,
                        col_ids    = chunk_cols,
                        n_cols_rast = n_cols_rast)
    names(cell_mats) <- paste0("x_", window_sizes, "x", window_sizes, "_array")

    # ── Band loop: read one band at a time ────────────────────────────────────
    for (i in seq_len(n_ch)) {

      # Single-band read: (read_nrows × 1) matrix → flatten to vector
      # Memory: read_nrows × n_cols × 8 bytes (no n_channels factor)
      band_vec <- as.vector(
        terra::values(rast_stack[[i]], row = read_start, nrows = read_nrows)
      )

      band_vec <- .scale_band_vec(band_vec, i, predictor_scaling,
                                   temperature_min_valid_celsius)

      # For each window: vectorised NA check + patch storage
      for (wi in seq_along(window_sizes)) {
        w    <- window_sizes[wi]
        key  <- paste0("x_", w, "x", w, "_array")
        cm   <- cell_mats[[key]]      # n_in × n_pos
        n_pos <- w * w

        all_idx  <- as.vector(t(cm))          # n_in × n_pos → flat
        vals_vec <- band_vec[all_idx]         # extract all patch cells at once

        # NA check: mark profiles with any non-finite cell as invalid
        if (anyNA(vals_vec) || !all(is.finite(vals_vec))) {
          bad <- colSums(!is.finite(
            matrix(vals_vec, nrow = n_pos, ncol = n_in)
          )) > 0L
          valid_common[chunk_idx[bad]] <- FALSE
        }

        # Store band i for all profiles in chunk.
        # Profiles later found invalid will be trimmed in Phase 4.
        # Reshape: flat n_in*n_pos → [w, w, n_in] → permute → [n_in, w, w]
        patch_list[[key]][chunk_idx, i, , ] <- aperm(
          array(vals_vec, dim = c(w, w, n_in)),
          c(3L, 1L, 2L)
        )
      }

      rm(band_vec)
      # band_vec is reassigned (in place) every iteration, so only one ever
      # lives at a time; calling gc() every band just wastes time. Returning
      # freed memory to the OS every ~10 bands is enough to keep the heap tidy.
      if (i %% 10L == 0L) invisible(gc(verbose = FALSE))
    }

    invisible(gc(verbose = FALSE))
  }

  message(sprintf("  Spatial chunks processed: %d / %d  (rest had no profiles)",
                  n_chunks_processed, n_chunks))

  # ── Phase 4: trim patch arrays to valid profiles only ─────────────────────
  n_valid   <- sum(valid_common)
  valid_idx <- which(valid_common)

  message("  Final valid profiles: ", n_valid, " / ", n_profiles,
          " (", round(100 * (n_profiles - n_valid) / n_profiles, 1), "% removed)")

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
    train_pts, "train",
    n_rows_rast, n_cols_rast, n_channels,
    window_sizes_to_extract, chunk_nrows, half_w_max,
    predictor_scaling, temperature_min_valid_celsius
  ),
  validation = process_split(
    validation_pts, "validation",
    n_rows_rast, n_cols_rast, n_channels,
    window_sizes_to_extract, chunk_nrows, half_w_max,
    predictor_scaling, temperature_min_valid_celsius
  ),
  test = process_split(
    test_pts, "test",
    n_rows_rast, n_cols_rast, n_channels,
    window_sizes_to_extract, chunk_nrows, half_w_max,
    predictor_scaling, temperature_min_valid_celsius
  )
)

message(sprintf("\nTotal extraction time: %.2f %s",
                Sys.time() - t_all, units(Sys.time() - t_all)))

# ── Manifest ──────────────────────────────────────────────────────────────────

manifest <- tibble::tibble(
  target_label                  = target_label,
  n_channels                    = n_channels,
  window_sizes_extracted        = paste(window_sizes_to_extract, collapse = ", "),
  chunk_nrows_used              = chunk_nrows,
  n_train_valid                 = nrow(patches$train$meta),
  n_validation_valid            = nrow(patches$validation$meta),
  n_test_valid                  = nrow(patches$test$meta),
  n_train_input                 = nrow(train_pts),
  n_validation_input            = nrow(validation_pts),
  n_test_input                  = nrow(test_pts),
  pct_train_removed             = round(100 * (nrow(train_pts)         - nrow(patches$train$meta))      / nrow(train_pts),         1),
  pct_validation_removed        = round(100 * (nrow(validation_pts) - nrow(patches$validation$meta)) / nrow(validation_pts), 1),
  pct_test_removed              = round(100 * (nrow(test_pts)       - nrow(patches$test$meta))       / nrow(test_pts),       1),
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
