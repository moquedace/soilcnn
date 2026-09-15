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
  "janitor"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "patches.R"))
source(file.path(project_root, "R", "preprocess.R"))
source(file.path(project_root, "R", "dataset.R"))
source(file.path(project_root, "R", "diagnostics.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 02 — Patch extraction: ALL points, ONCE, RAW
#
# What changed and why
# --------------------
# This script used to run three times (one per split) and to bake the z-score
# into the stored patches. Both are gone:
#
#   • one pass instead of three   — the splits are disjoint subsets of points
#     over the SAME raster, so reading all 187 bands three times was pure
#     repetition. One pass cuts that I/O by 3x.
#
#   • raw instead of scaled       — mu and sigma are ESTIMATED from training
#     points, so with k folds there are k scalings of the same patches. Baking
#     one in ties the patches to one split for ever and makes cross-validation
#     cost k re-extractions. Scaling now happens at tensor-build time
#     (R/preprocess.R), one broadcast per fold.
#
# QC still happens here, because it is fold-INDEPENDENT: a physically
# impossible value is impossible regardless of who is in the training set. The
# rules come from qc_table.csv, written by 01 — no longer duplicated as
# literals in two scripts that had to be kept in sync by hand.
#
# Output: one file per window, float32, all surviving points. 03 loads only
# the windows its grid actually uses.
# ══════════════════════════════════════════════════════════════════════════════

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"

predictor_raster_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"

# Window sizes to extract (pixels, must be odd). The largest decides which
# profiles survive the edge filter — profiles too close to the raster boundary
# for the largest window are excluded for ALL windows, so every window sees
# exactly the same set of points.
#
# A window's physical extent is window_size x raster resolution, so re-pick it
# whenever the resolution changes. At 250 m:
#   3x3   ~ 0.75 km  (immediate neighbourhood, land cover, micro-relief)
#   9x9   ~ 2.25 km  (hillslope, drainage, soil-landscape position)
#   15x15 ~ 3.75 km  (local landscape, catchment context, local climate)
window_sizes_to_extract <- c(3L, 9L, 15L)

# Raster rows read at a time. Peak RAM per band read is
#   (chunk_nrows + 2 * half_w_max) * n_cols * 8 bytes
# and does NOT scale with channel count — which is what avoids the
# std::bad_alloc a whole-stack read triggers on a fine global raster.
#
# Raised from 200 to 1000. The old value came from an era when the whole
# 187-band stack was read at once and memory was the binding constraint;
# reading ONE band at a time costs ~0.3 GB at 200 rows, so 200 was leaving
# the real cost on the table: 269 chunks x 181 bands = 48,689 separate GDAL
# reads, each with its own open/seek/decode overhead, and each re-reading the
# 2*half_w_max margin rows. At 1000 rows the strip is ~1.3 GB -- irrelevant
# on this machine -- and the chunk count drops roughly five-fold.
chunk_nrows <- 1000

# Auto-size chunk_nrows so each SINGLE-BAND strip stays within this budget.
# Overrides chunk_nrows when not NULL.
max_ram_gb <- NULL

# Verification is by FILE SIZE, computed from the shape -- it never loads the
# tensor back. An earlier version read each file and compared cell values; it
# reported a false failure on a perfectly good 6 GB file and aborted after 5
# hours of extraction. Two lessons, both encoded below:
#   • a sanity check that can false-positive must never abort expensive work
#   • loading a multi-GB tensor just to check it is itself a risk -- it is
#     allocated outside the R heap, so gc() does not return it promptly, and
#     doing that inside an already-pressured session can kill it outright
size_tolerance_bytes <- 100000L   # RDS header

# ── Paths ─────────────────────────────────────────────────────────────────────

input_data_dir     <- file.path(project_root, "data",    "processed", "soc_stock_modeling", target_label)
input_metadata_dir <- file.path(project_root, "outputs", "metadata",  "soc_stock_modeling", target_label)
output_patch_dir   <- file.path(project_root, "outputs", "patches",   "soc_stock_modeling", target_label)
output_metadata_patch_dir <- file.path(input_metadata_dir, "patches")

create_output_dirs(c(output_patch_dir, output_metadata_patch_dir))

# ── Read points, types and QC rules ───────────────────────────────────────────
# Channel ORDER is set by predictor_type_table.csv and must survive every step
# from here to prediction. A silent reordering would feed the network one
# channel while the map is built from another, and nothing downstream would
# notice — so it is asserted at each handoff, never assumed.

message("Reading points, predictor types and QC rules...")

points_all <- readr::read_csv2(
  file.path(input_data_dir, "full_modeling_dataset_raw.csv"),
  show_col_types = FALSE
)

type_table <- readr::read_csv2(
  file.path(input_metadata_dir, "predictor_type_table.csv"),
  show_col_types = FALSE
)

qc_table <- readr::read_csv2(
  file.path(input_metadata_dir, "qc_table.csv"),
  show_col_types = FALSE
)

# Read only to record WHICH target this store was built for. Two stores that
# differ by nothing but the target look identical on disk, and the patches
# really are identical -- it is the stored targets that are not.
target_config <- readr::read_csv2(
  file.path(input_metadata_dir, "target_config.csv"),
  show_col_types = FALSE
)

predictor_cols <- type_table$predictor
n_channels     <- length(predictor_cols)
n_points       <- nrow(points_all)

stopifnot(
  identical(qc_table$predictor, predictor_cols),
  all(predictor_cols %in% names(points_all))
)

message("Points : ", n_points, " | Channels: ", n_channels)
message("Types  : ", sum(type_table$is_dummy), " dummy | ",
        sum(type_table$is_percentage), " percentage | ",
        sum(!type_table$is_dummy & !type_table$is_percentage), " continuous")
message("QC     : ", sum(!is.na(qc_table$na_below)), " with an NA floor | ",
        sum(!is.na(qc_table$clamp_lower)), " clamped into [0, 100]")

# ── Open raster stack (metadata only) ─────────────────────────────────────────

message("\nOpening raster stack (metadata only)...")
t0 <- Sys.time()

raster_files <- list.files(predictor_raster_dir, pattern = "\\.tif$",
                           full.names = TRUE, recursive = FALSE)
raster_names <- janitor::make_clean_names(
  tools::file_path_sans_ext(basename(raster_files))
)

keep_idx     <- raster_names %in% predictor_cols
raster_files <- raster_files[keep_idx]
raster_names <- raster_names[keep_idx]

order_idx <- match(predictor_cols, raster_names)
if (anyNA(order_idx)) {
  stop("Missing raster(s) for: ",
       paste(predictor_cols[is.na(order_idx)], collapse = ", "))
}
raster_files <- raster_files[order_idx]
raster_names <- raster_names[order_idx]

stopifnot(identical(raster_names, predictor_cols))

rast_stack <- terra::rast(raster_files)
names(rast_stack) <- raster_names

n_rows_rast <- terra::nrow(rast_stack)
n_cols_rast <- terra::ncol(rast_stack)

message("Raster grid: ", n_rows_rast, " rows x ", n_cols_rast, " cols x ",
        terra::nlyr(rast_stack), " layers  (opened in ",
        sprintf("%.2f %s", Sys.time() - t0, units(Sys.time() - t0)), ")")

# ── Resolve chunk_nrows and report the RAM plan ───────────────────────────────

half_w_max         <- (max(window_sizes_to_extract) - 1L) %/% 2L
bytes_per_band_row <- as.numeric(n_cols_rast) * 8

if (!is.null(max_ram_gb)) {
  chunk_nrows <- max(1L, as.integer(
    floor(max_ram_gb * 1e9 / bytes_per_band_row) - 2L * half_w_max
  ))
}

strip_gb <- (chunk_nrows + 2L * half_w_max) * bytes_per_band_row / 1e9

# The patch arrays dominate. They are built in double (R has no native float),
# then converted to float32 one window at a time and released, so the peak is
# "all doubles + the largest float", not "all doubles + all floats".
gb_double  <- n_points * n_channels * sum(window_sizes_to_extract^2) * 8 / 1e9
gb_f32_max <- n_points * n_channels * max(window_sizes_to_extract)^2 * 4 / 1e9

message(sprintf("\nchunk_nrows = %d  (%.2f GB per single-band read)",
                chunk_nrows, strip_gb))
message(sprintf("Patch arrays : %.1f GB in double during extraction", gb_double))
message(sprintf("Peak RAM     : ~%.1f GB (all doubles + largest float32 while converting)",
                gb_double + gb_f32_max))
message(sprintf("On disk      : ~%.1f GB float32, one file per window",
                gb_double / 2))

# ── Phase 1: edge check ───────────────────────────────────────────────────────
# Once, for every point, against the LARGEST window, so all windows share the
# same surviving set.

coords  <- as.matrix(points_all[, c("x", "y")])
cells   <- terra::cellFromXY(rast_stack, coords)
row_ids <- terra::rowFromCell(rast_stack, cells)
col_ids <- terra::colFromCell(rast_stack, cells)

valid_common <- !is.na(row_ids) & !is.na(col_ids)
for (w in window_sizes_to_extract) {
  ok_w <- patch_centre_in_bounds(row_ids, col_ids, n_rows_rast, n_cols_rast, w)
  ok_w[is.na(ok_w)] <- FALSE
  valid_common <- valid_common & ok_w
}

message("\nAfter edge check: ", sum(valid_common), " / ", n_points,
        "  (", round(100 * (n_points - sum(valid_common)) / n_points, 2),
        "% too close to the raster edge)")

if (!any(valid_common)) stop("No points survived the edge check.")

# ── Phase 2: allocate ─────────────────────────────────────────────────────────

patch_keys <- paste0("w", sprintf("%02d", window_sizes_to_extract))
patch_list <- setNames(
  lapply(window_sizes_to_extract,
         function(w) array(NA_real_, dim = c(n_points, n_channels, w, w))),
  patch_keys
)

# Blame matrix: which CHANNEL invalidated which point. This is the safeguard
# the pipeline was missing. The full-window rule turns a single NA pixel into a
# hole of up to 15x15 around it, so one sparse-NA channel can empty a map — it
# has happened here: coverage of a test tile was 0.72% until the percentage
# clamp was fixed, and it only surfaced after a full 250 m run. valid_common is
# an AND over 187 channels, so on its own it can never say WHICH one did it.
# ~28 MB at 37k points x 187 channels; cheap insurance.
blame <- matrix(FALSE, nrow = n_points, ncol = n_channels)

# ── Phase 3: chunk loop -> band loop ──────────────────────────────────────────

chunk_starts <- seq(1L, n_rows_rast, by = chunk_nrows)
n_chunks     <- length(chunk_starts)
n_done       <- 0L
t_extract    <- Sys.time()

# HOW MUCH OF THE RASTER THIS RUN WILL ACTUALLY READ.
#
# The reading, not the point count, is what stage 02 costs: one strip per band
# per chunk, and a chunk with no point in it is skipped entirely. So a 10%
# subsample does NOT make this ten times faster -- it makes it faster by
# whatever fraction of the raster its points happen to miss, which depends on
# how they are spread, not on how many there are.
#
# Printed before the loop so the number is available when deciding whether to
# wait, rather than inferred from watching the progress lines. chunk_nrows is
# the lever: smaller chunks skip more empty raster but issue more GDAL calls
# (see the note where chunk_nrows is set -- going 200 -> 1000 was a win on
# FULL data, where almost every chunk has a point and shrinking saves nothing).
.rows_planned <- 0L
.chunks_hit   <- 0L
for (.cs in chunk_starts) {
  .ce <- min(.cs + chunk_nrows - 1L, n_rows_rast)
  if (!any(valid_common & !is.na(row_ids) & row_ids >= .cs & row_ids <= .ce)) next
  .chunks_hit   <- .chunks_hit + 1L
  .rows_planned <- .rows_planned +
    (min(n_rows_rast, .ce + half_w_max) - max(1L, .cs - half_w_max) + 1L)
}
message(sprintf(
  "\nReading plan: %d of %d chunks hold a point -- %s of %s raster rows (%.1f%%)",
  .chunks_hit, n_chunks, format(.rows_planned, big.mark = ","),
  format(n_rows_rast, big.mark = ","), 100 * .rows_planned / n_rows_rast))
message(sprintf("  %d chunk(s) x %d band(s) = %s strip reads",
                .chunks_hit, n_channels,
                format(.chunks_hit * n_channels, big.mark = ",")))
rm(.rows_planned, .chunks_hit, .cs, .ce)

for (ci in seq_along(chunk_starts)) {
  cs <- chunk_starts[ci]
  ce <- min(cs + chunk_nrows - 1L, n_rows_rast)

  in_chunk <- valid_common & !is.na(row_ids) & row_ids >= cs & row_ids <= ce
  if (!any(in_chunk)) next

  n_in      <- sum(in_chunk)
  chunk_idx <- which(in_chunk)
  n_done    <- n_done + 1L

  read_start <- max(1L,          cs - half_w_max)
  read_end   <- min(n_rows_rast, ce + half_w_max)
  read_nrows <- read_end - read_start + 1L

  message(sprintf(
    "  [chunk %d/%d] rows %d-%d | %d points | reading %d rows x %d bands",
    ci, n_chunks, cs, ce, n_in, read_nrows, n_channels))

  local_rows <- row_ids[chunk_idx] - read_start + 1L
  chunk_cols <- col_ids[chunk_idx]

  # Cell indices are geometry, not data: computed once per chunk, reused for
  # all 187 bands. patch_cell_index() lives in R/patches.R, shared with the
  # prediction side so the two can never drift apart.
  cell_mats <- setNames(
    lapply(window_sizes_to_extract, function(w) {
      patch_cell_index(local_rows, chunk_cols, n_cols_rast, w)
    }),
    patch_keys
  )

  for (i in seq_len(n_channels)) {

    band_vec <- as.vector(
      terra::values(rast_stack[[i]], row = read_start, nrows = read_nrows)
    )

    # QC only. Scaling is fold-dependent and happens at tensor-build time.
    band_vec <- qc_band_values(band_vec, qc_table[i, ])

    for (wi in seq_along(window_sizes_to_extract)) {
      key <- patch_keys[wi]

      pb <- patch_band_assemble(band_vec, cell_mats[[key]],
                                window_sizes_to_extract[wi])

      if (!all(pb$valid)) {
        bad <- chunk_idx[!pb$valid]
        valid_common[bad] <- FALSE
        blame[bad, i]     <- TRUE
      }

      patch_list[[key]][chunk_idx, i, , ] <- pb$array
    }

    rm(band_vec)
    if (i %% 10L == 0L) invisible(gc(verbose = FALSE))
  }

  invisible(gc(verbose = FALSE))
}

message(sprintf("\nChunks processed: %d / %d  (rest had no points)  in %.2f %s",
                n_done, n_chunks,
                as.numeric(Sys.time() - t_extract),
                units(Sys.time() - t_extract)))

# ── Phase 4: who invalidated what ─────────────────────────────────────────────
# Read this table before anything else. A channel high in `n_sole_cause` is one
# you can drop to recover exactly that many points — and, more to the point,
# the same channel will punch the same holes in the map later, where it costs
# weeks instead of minutes.

n_valid   <- sum(valid_common)
valid_idx <- which(valid_common)
n_blamed  <- sum(rowSums(blame) > 0)

blame_report <- tibble::tibble(
  predictor     = predictor_cols,
  type          = dplyr::case_when(type_table$is_dummy      ~ "dummy",
                                   type_table$is_percentage ~ "percentage",
                                   TRUE                     ~ "continuous"),
  n_invalidated = as.integer(colSums(blame)),
  n_sole_cause  = as.integer(colSums(blame & (rowSums(blame) == 1L)))
) %>%
  dplyr::mutate(
    pct_invalidated = round(100 * n_invalidated / n_points, 3),
    pct_sole_cause  = round(100 * n_sole_cause  / n_points, 3)
  ) %>%
  dplyr::arrange(dplyr::desc(n_invalidated))

safe_write_csv2(blame_report,
                file.path(output_metadata_patch_dir, "channel_invalidation.csv"))

message("\n── Which channels invalidated points (full-window rule) ──")
message("  Window rule lost : ", format(n_blamed, big.mark = ","),
        " point(s) to non-finite values")
message("  Final valid      : ", format(n_valid, big.mark = ","), " / ",
        format(n_points, big.mark = ","),
        "  (", round(100 * (n_points - n_valid) / n_points, 2), "% removed)")

top_blame <- dplyr::filter(blame_report, n_invalidated > 0L)
if (nrow(top_blame) == 0L) {
  message("  No channel invalidated a single point.")
} else {
  message("\n  Channels responsible (top 15):")
  print_wide(dplyr::slice_head(top_blame, n = 15), n = Inf)

  if (top_blame$pct_invalidated[1] > 1) {
    message("\n  WARNING: '", top_blame$predictor[1], "' alone invalidated ",
            top_blame$pct_invalidated[1], "% of points.")
    message("  A channel with sparse NA does far more damage to the MAP than ",
            "to the training set: the")
    message("  full-window rule turns each NA pixel into a hole of up to ",
            max(window_sizes_to_extract), "x", max(window_sizes_to_extract),
            " around it. Check its coverage")
    message("  over land BEFORE committing to a full prediction run.")
  }
}

if (n_valid == 0L) stop("No points valid after the window rule.")

# ── Phase 5: persist ──────────────────────────────────────────────────────────
# Order matters and is deliberate:
#
#   1. metadata FIRST. patch_meta.csv defines WHICH points survived, and every
#      window file must line up with it row for row. Writing it before the
#      tensors means a crash mid-write leaves a store that can still be
#      completed, instead of 5 hours of extraction with nothing to anchor it.
#   2. then one window at a time, largest first, releasing each before the next
#      so the peak is bounded by the largest single window, not by all of them.
#   3. a window already on disk at the right size is SKIPPED. Re-running after
#      a crash then costs the extraction but not the writes.

meta_valid <- points_all[valid_idx, ] %>%
  # No role column: the store says WHERE the points are and WHAT they are
  # worth, never who trains. That is what lets the split change for free --
  # see the fold plans in R/resample.R.
  dplyr::select(profile_id, sample_id, x, y,
                target_native, target_log1p) %>%
  dplyr::rename(target_transform = target_log1p)

safe_write_csv2(meta_valid, file.path(output_patch_dir, "patch_meta.csv"))
safe_write_csv2(blame_report,
                file.path(output_metadata_patch_dir, "channel_invalidation.csv"))
message("\nMetadata written first (", nrow(meta_valid), " points) -- the window ",
        "files are aligned to it.")

# Expected size from the shape. float32 = 4 bytes; the serialisation adds a
# small header, hence the tolerance.

# Amostra para inspecao visual (99b). Gravada AQUI porque os arrays ja estao
# na memoria: custa alguns MB e zero tempo. A alternativa -- o 99b recarregar
# o store inteiro so para tirar 6 patches -- custava minutos e ~12 GB de RAM
# para olhar 0,016% do dado.
set.seed(42)
sample_idx <- sort(sample(n_valid, min(6L, n_valid)))
patch_sample <- list(
  meta    = meta_valid[sample_idx, ],
  windows = setNames(
    lapply(seq_along(window_sizes_to_extract), function(wi) {
      patch_list[[patch_keys[wi]]][valid_idx[sample_idx], , , , drop = FALSE]
    }), patch_keys),
  predictors = predictor_cols
)
safe_save_rds(patch_sample, file.path(output_patch_dir, "patch_sample.rds"),
              compress = TRUE)
message("
Amostra de ", length(sample_idx), " patches gravada para inspecao ",
        "visual (", round(file.size(file.path(output_patch_dir,
        "patch_sample.rds")) / 1e6, 1), " MB)")

saved <- tibble::tibble()

for (wi in order(window_sizes_to_extract, decreasing = TRUE)) {
  w   <- window_sizes_to_extract[wi]
  key <- patch_keys[wi]
  f   <- patch_window_path(output_patch_dir, w)
  exp_b <- patch_window_bytes(n_valid, n_channels, w)

  # already done? skip. This is what makes a re-run after a crash cheap.
  if (file.exists(f) && abs(file.size(f) - exp_b) < size_tolerance_bytes) {
    message(sprintf("  %-20s already on disk, size matches -- skipping (%.2f GB)",
                    basename(f), file.size(f) / 1e9))
    saved <- dplyr::bind_rows(saved, tibble::tibble(
      window = w, file = basename(f), gb = round(file.size(f) / 1e9, 2),
      status = "kept"))
    patch_list[[key]] <- NULL
    invisible(gc(verbose = FALSE))
    next
  }

  # saveRDS of the plain array -- NOT torch_save of a tensor. torch_save() in
  # this torch build corrupts silently above 2^31 bytes; it produced a
  # patches_w15.pt of exactly the right size with 4.3 GB of zeros in the tail.
  # See the note in R/dataset.R. The float32 conversion moved to load time.
  res <- save_patch_window(patch_list[[key]][valid_idx, , , , drop = FALSE],
                           output_patch_dir, w)
  patch_list[[key]] <- NULL
  invisible(gc(verbose = FALSE))

  message(sprintf("  %-20s written %.2f GB  %s", res$file, res$gb,
                  if (res$ok) "[size ok]" else
                    sprintf("[SIZE MISMATCH: expected %.2f GB]", res$exp_gb)))

  saved <- dplyr::bind_rows(saved, tibble::tibble(
    window = w, file = res$file, gb = res$gb,
    status = if (res$ok) "written" else "size_mismatch"))
}

# Report, do not abort. Everything extracted is already on disk by this point;
# throwing an error here would destroy nothing but would hide what succeeded.
bad <- dplyr::filter(saved, status == "size_mismatch")
if (nrow(bad) > 0L) {
  message("\n  WARNING: ", nrow(bad), " file(s) have an unexpected size: ",
          paste(bad$file, collapse = ", "))
  message("  Everything else is written. Re-run this script to rewrite only ",
          "the bad one(s) -- the rest will be skipped.")
}

# ── Manifest ──────────────────────────────────────────────────────────────────
# predictor_cols_final is the contract: every later stage asserts against it
# rather than re-deriving the channel order from a directory listing.

manifest <- tibble::tibble(
  target_label         = target_label,
  store_complete       = nrow(saved) == length(window_sizes_to_extract) &&
                         all(saved$status != "size_mismatch"),
  n_channels           = n_channels,
  n_points_input       = n_points,
  n_points_valid       = n_valid,
  pct_removed          = round(100 * (n_points - n_valid) / n_points, 2),
  windows_extracted    = paste(window_sizes_to_extract, collapse = ", "),
  storage              = "rds_double_one_file_per_window",
  scaling_applied      = FALSE,
  qc_applied           = "qc_table.csv",
  chunk_nrows_used     = chunk_nrows,
  predictor_cols_final = paste(predictor_cols, collapse = ";"),

  # THE SPEC THIS STORE WAS BUILT UNDER.
  #
  # Three things force a re-extraction, and only three: the predictors, the
  # windows and the target. Recording them here lets check_store_spec() refuse,
  # in seconds, a configuration the store cannot serve -- instead of the
  # mismatch surfacing as a wrong result, or as five hours of work discovered
  # to be for nothing.
  #
  # The predictor LIST is recorded, not a hash of it: comparing lists costs the
  # same and lets the error name which predictors differ.
  target_col           = target_config$target_col[1],
  target_transform     = "log1p",
  cell_size            = terra::res(rast_stack)[1],

  raster_nrow          = n_rows_rast,
  raster_ncol          = n_cols_rast,
  extracted_at         = as.character(Sys.time())
)

safe_save_rds(manifest, file.path(output_patch_dir, "patch_manifest.rds"),
              compress = FALSE)
safe_write_csv2(manifest, file.path(output_metadata_patch_dir, "patch_manifest.csv"))
safe_write_csv2(saved,    file.path(output_metadata_patch_dir, "patch_files.csv"))

message("\n── Manifest ──────────────────────────────")
message("  Patches are RAW (no scaling). 03 applies the fold's scaling at")
message("  tensor-build time via fit_scaling() / scale_patches().")
print_wide(dplyr::select(manifest, n_points_valid, pct_removed,
                         windows_extracted, n_channels, storage,
                         scaling_applied, store_complete))
print_wide(saved)

if (!manifest$store_complete[1]) {
  message("\n  Store INCOMPLETE. Re-run this script: finished windows are ",
          "skipped by size, so only what is missing costs anything.")
}

message("\nPatches saved to: ", output_patch_dir)
message("Done.")
