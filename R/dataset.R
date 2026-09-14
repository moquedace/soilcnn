# Patch store: loading raw patches and building a fold's tensors
#
# Sits between patch extraction (which writes raw, unscaled patches, one file
# per window) and the training loop (which needs scaled tensors split into
# train/validation/test).
#
# The split arrives as an INDEX — a list of integer row positions into the
# patch store — not as a property of the stored data. That is the change that
# makes cross-validation possible: the same 8.6 GB of patches serve any number
# of folds, each with its own scaling, at the cost of one broadcast per fold.
# `split_index_from_meta()` below is the degenerate case (a single fixed
# holdout, read from the dataset_role column) and is what the pipeline uses
# today; a spatial-fold constructor plugs into exactly the same slot.
#
# Channel ORDER is the contract that ties the whole pipeline together. It is
# asserted at every handoff rather than assumed, because a silent reordering
# would feed the network one channel while the map is built from another, and
# nothing downstream would notice.

# ── naming ────────────────────────────────────────────────────────────────────

#' Storage key for a window size. Also the filename stem in the patch store.
patch_window_key <- function(window_size) {
  paste0("w", sprintf("%02d", as.integer(window_size)))
}

# ── window file I/O ──────────────────────────────────────────────────
#
# Windows are stored as plain R arrays via saveRDS, NOT as torch tensors.
#
# That is a deliberate step back. An earlier version of this code stored
# float32 tensors with torch_save() to halve the files (8.6 GB instead of 17).
# torch_save() in R torch 0.17.0 breaks above 2^31 bytes: measured, a tensor of
# 2,147,479,648 bytes writes correctly and one of 2,147,487,648 bytes kills the
# session. Worse, in one case it produced a file of the RIGHT SIZE whose tail
# was entirely zeros -- silent corruption that passes a "no non-finite values"
# check, because zero is finite.
#
# saveRDS has no such limit and is what this pipeline used for a 17 GB object
# before. The disk saved was never worth it: the machine has terabytes free,
# and the failure mode of the alternative is a plausible-looking wrong map.
#
# Kept from the tensor version: ONE FILE PER WINDOW. That was the real win --
# a grid that never uses 15x15 should not load 12 GB for it -- and it costs
# nothing to keep.
#
# The float32 conversion still happens, just later: on load, once, into the
# tensor the model actually consumes.

#' Path of a window file inside a patch store.
patch_window_path <- function(patch_dir, window_size) {
  file.path(patch_dir, paste0("patches_", patch_window_key(window_size), ".rds"))
}

#' Expected size on disk of one window, in bytes (double, uncompressed).
#'
#' Used to spot a truncated or half-written file without reading it back --
#' reading a multi-GB file just to check it is itself a risk.
patch_window_bytes <- function(n_points, n_channels, window_size) {
  as.numeric(n_points) * n_channels * window_size * window_size * 8
}

#' Write one window array to the store.
#'
#' @param arr         Array [n_points, n_channels, w, w], double.
#' @param patch_dir   Store directory.
#' @param window_size Odd integer.
#' @return A one-row tibble describing what was written.
save_patch_window <- function(arr, patch_dir, window_size) {
  f <- patch_window_path(patch_dir, window_size)
  d <- dim(arr)
  if (length(d) != 4L || d[3] != window_size || d[4] != window_size) {
    stop("array must be [n, channels, ", window_size, ", ", window_size, "], got ",
         paste(d, collapse = " x "), call. = FALSE)
  }

  if (file.exists(f)) file.remove(f)
  saveRDS(arr, f, compress = FALSE)

  got <- file.size(f)
  exp <- patch_window_bytes(d[1], d[2], window_size)

  tibble::tibble(
    window = window_size, file = basename(f),
    gb = round(got / 1e9, 2), exp_gb = round(exp / 1e9, 2),
    # RDS adds a small header; anything further off means a partial write
    ok = abs(got - exp) < 100000
  )
}

#' Read one window and hand back the float32 tensor the model consumes.
#'
#' The double array is released before returning, so the caller is left
#' holding half the memory it took to get there.
load_patch_window <- function(patch_dir, window_size, expect_points = NULL,
                              expect_channels = NULL) {
  f <- patch_window_path(patch_dir, window_size)
  if (!file.exists(f)) stop("Missing window file: ", f, call. = FALSE)

  arr <- readRDS(f)
  d   <- dim(arr)
  if (length(d) != 4L) {
    stop(basename(f), " is not a 4D array (got ", length(d), " dims).",
         call. = FALSE)
  }
  if (!is.null(expect_points) && d[1] != expect_points) {
    stop(basename(f), " has ", d[1], " points, expected ", expect_points,
         " -- this window was not built from the same point set as the rest ",
         "of the store.", call. = FALSE)
  }
  if (!is.null(expect_channels) && d[2] != expect_channels) {
    stop(basename(f), " has ", d[2], " channels, expected ", expect_channels,
         call. = FALSE)
  }
  if (d[3] != window_size || d[4] != window_size) {
    stop(basename(f), " holds a ", d[3], "x", d[4], " window, expected ",
         window_size, "x", window_size, call. = FALSE)
  }

  x <- torch::torch_tensor(arr, dtype = torch::torch_float())
  rm(arr); gc(verbose = FALSE)
  x
}

# ── loading ───────────────────────────────────────────────────────────────────

#' Load a raw (unscaled) patch store from disk.
#'
#' Only the windows asked for are read. That is the point of one file per
#' window: a grid that never uses 15x15 should not pay 6 GB of RAM for it.
#'
#' @param patch_dir    Directory holding patches_wNN.pt, patch_meta.csv and
#'   patch_manifest.rds.
#' @param window_sizes Integer vector of windows to load. NULL loads all the
#'   windows the manifest says were extracted.
#' @param verbose      Print what was loaded and how big it is.
#' @return list(windows, meta, manifest, predictors, n_channels)
load_patch_store <- function(patch_dir, window_sizes = NULL, verbose = TRUE) {

  manifest_path <- file.path(patch_dir, "patch_manifest.rds")
  meta_path     <- file.path(patch_dir, "patch_meta.csv")
  for (f in c(manifest_path, meta_path)) {
    if (!file.exists(f)) {
      stop("Patch store incomplete, missing: ", f,
           "\nA patch store is written by the extraction step and must contain ",
           "patches_<window>.pt, patch_meta.csv and patch_manifest.rds.",
           call. = FALSE)
    }
  }

  manifest <- readRDS(manifest_path)
  meta     <- readr::read_csv2(meta_path, show_col_types = FALSE)

  if (isTRUE(manifest$scaling_applied[1])) {
    stop("This patch store was written WITH scaling already applied, so it ",
         "is tied to a single split and cannot be resampled. Re-extract the ",
         "patches without scaling (QC only) -- see make_qc_table() and ",
         "qc_band_values().", call. = FALSE)
  }

  predictors <- strsplit(manifest$predictor_cols_final[1], ";")[[1]]
  n_channels <- manifest$n_channels[1]
  stopifnot(length(predictors) == n_channels)

  available <- as.integer(trimws(
    strsplit(manifest$windows_extracted[1], ",")[[1]]
  ))
  if (is.null(window_sizes)) window_sizes <- available

  missing_w <- setdiff(window_sizes, available)
  if (length(missing_w) > 0L) {
    stop("Window(s) ", paste(missing_w, collapse = ", "),
         " are not in this store. Available: ",
         paste(available, collapse = ", "), call. = FALSE)
  }

  if (nrow(meta) != manifest$n_points_valid[1]) {
    stop("patch_meta.csv has ", nrow(meta), " rows but the manifest says ",
         manifest$n_points_valid[1], " valid points.", call. = FALSE)
  }

  if (verbose) {
    message("Loading patch store: ", nrow(meta), " points x ", n_channels,
            " channels | windows ", paste(window_sizes, collapse = ", "))
  }

  windows <- list()
  for (w in window_sizes) {
    # load_patch_window() asserts the shape against the store's own meta,
    # so a window built from a different point set is caught here rather
    # than as a cryptic conv error minutes into training.
    x <- load_patch_window(patch_dir, w,
                           expect_points   = nrow(meta),
                           expect_channels = n_channels)
    windows[[patch_window_key(w)]] <- x
    if (verbose) {
      message(sprintf("  %s: %s  (%.2f GB as float32)", patch_window_key(w),
                      paste(as.integer(x$shape), collapse = " x "),
                      prod(as.numeric(x$shape)) * 4 / 1e9))
    }
  }

  list(windows = windows, meta = meta, manifest = manifest,
       predictors = predictors, n_channels = n_channels,
       window_sizes = window_sizes)
}

# ── split as an index ─────────────────────────────────────────────────────────

#' Read the fixed holdout out of the meta table's dataset_role column.
#'
#' The single-holdout case, kept explicit so the call site reads the same as it
#' will when a resampling constructor takes its place.
#'
#' @param meta  The patch store's meta tibble.
#' @param roles Role names to extract, in order.
#' @return Named list of integer row positions into `meta`.
split_index_from_meta <- function(meta,
                                  roles = c("train", "validation", "test")) {
  if (!"dataset_role" %in% names(meta)) {
    stop("meta has no dataset_role column.", call. = FALSE)
  }
  idx <- lapply(roles, function(r) which(meta$dataset_role == r))
  names(idx) <- roles

  empty <- roles[lengths(idx) == 0L]
  if (length(empty) > 0L) {
    stop("No rows for role(s): ", paste(empty, collapse = ", "), call. = FALSE)
  }

  # Disjoint and complete: a point in two splits is leakage, a point in none is
  # silently discarded data. Both are worth failing on.
  all_idx <- unlist(idx, use.names = FALSE)
  if (anyDuplicated(all_idx)) {
    stop("Split index is not disjoint -- some rows appear in more than one role.",
         call. = FALSE)
  }
  if (length(all_idx) != nrow(meta)) {
    message("NOTE: ", nrow(meta) - length(all_idx),
            " row(s) of the patch store belong to no role and will be unused.")
  }
  idx
}

# ── aligning point values to the store ────────────────────────────────────────

#' Line up the point-value table with the patch store, row for row.
#'
#' Extraction drops points whose window was not fully valid, so the point table
#' has MORE rows than the patch store. Scaling must be estimated from the points
#' that actually train, which means matching on sample_id rather than trusting
#' row order.
#'
#' @param points Point-value table (see .point_contract).
#' @param meta   The patch store's meta tibble.
#' @return `points` reordered and subset to exactly match `meta` row for row.
align_points_to_meta <- function(points, meta) {
  check_point_contract(points, need = "sample_id", what = "points")
  check_point_contract(meta,   need = "sample_id", what = "patch store meta")
  pos <- match(meta$sample_id, points$sample_id)
  if (anyNA(pos)) {
    stop(sum(is.na(pos)), " point(s) in the patch store have no matching row ",
         "in the point table -- were the points and the patches built from ",
         "different data?", call. = FALSE)
  }
  out <- points[pos, , drop = FALSE]
  stopifnot(identical(out$sample_id, meta$sample_id))
  out
}

# ── building a fold's tensors ─────────────────────────────────────────────────

#' Build the scaled, split tensors for one fold.
#'
#' Scaling is estimated from `index$train` ONLY. Everything else — validation,
#' test, and at prediction time the whole map — is transformed with those same
#' constants, never with its own.
#'
#' @param store        From load_patch_store().
#' @param points       Point values, aligned via align_points_to_meta().
#' @param type_table   Tibble with predictor / is_dummy / is_percentage, in
#'   channel order.
#' @param index        From split_index_from_meta() or a fold constructor.
#' @param window_sizes Windows to include (default: all loaded).
#' @param scaling      Optional precomputed scaling; when NULL it is fitted
#'   from the fold's training rows, which is the point of the whole design.
#' @return list(cache, scaling) where `cache[[role]][[key]]` is a tensor and
#'   `cache[[role]]$y` the target column.
build_fold_cache <- function(store, points, type_table, index,
                             window_sizes = NULL, scaling = NULL,
                             verbose = TRUE) {

  if (is.null(window_sizes)) window_sizes <- store$window_sizes

  # Channel order must agree across the three sources that describe it.
  if (!identical(as.character(type_table$predictor), as.character(store$predictors))) {
    stop("type_table and the patch manifest disagree on channel order. Both ",
         "must list the same predictors in the same sequence -- rebuild them ",
         "from the same source.", call. = FALSE)
  }
  if (!all(store$predictors %in% names(points))) {
    stop("Point table is missing channel(s): ",
         paste(setdiff(store$predictors, names(points)), collapse = ", "),
         call. = FALSE)
  }
  if (nrow(points) != nrow(store$meta)) {
    stop("points has ", nrow(points), " rows, store has ", nrow(store$meta),
         " -- run align_points_to_meta() first.", call. = FALSE)
  }

  if (is.null(scaling)) {
    scaling <- fit_scaling(points, type_table, index$train)
  }
  if (any(scaling$degenerate)) {
    bad <- scaling$predictor[scaling$degenerate]
    stop("Degenerate scaling (zero or non-finite sd) on this fold's training ",
         "rows for: ", paste(bad, collapse = ", "),
         "\nA channel constant within THIS fold cannot be z-scored.",
         call. = FALSE)
  }

  roles <- names(index)
  cache <- setNames(vector("list", length(roles)), roles)
  for (r in roles) cache[[r]] <- list()

  for (w in window_sizes) {
    key <- patch_window_key(w)
    x   <- store$windows[[key]]
    if (is.null(x)) stop("Window ", w, " is not loaded in this store.",
                         call. = FALSE)

    # Clone before scaling in place: the store's raw tensor must survive, since
    # the next fold needs it unscaled. The clone is freed as soon as the
    # per-role slices are taken.
    xs <- scale_patches(x$clone(), scaling, inplace = TRUE)
    for (r in roles) {
      cache[[r]][[key]] <- xs[index[[r]], , , , drop = FALSE]
    }
    rm(xs); gc(verbose = FALSE)

    if (verbose) {
      message(sprintf("  %s scaled and split: %s",
                      key,
                      paste(sprintf("%s=%d", roles, lengths(index)),
                            collapse = " ")))
    }
  }

  check_point_contract(store$meta, need = "target_transform",
                       what = "patch store meta")
  y_all <- as.numeric(store$meta$target_transform)
  for (r in roles) {
    cache[[r]]$y <- torch::torch_tensor(
      y_all[index[[r]]], dtype = torch::torch_float()
    )$view(c(-1L, 1L))
  }

  list(cache = cache, scaling = scaling, index = index)
}

#' Metadata rows for each role, in the same order as the cached tensors.
#'
#' predict_loader() assumes row i of the loader is row i of this table, so it
#' must be sliced with exactly the same index used for the tensors.
fold_points_valid <- function(store, index) {
  setNames(lapply(names(index), function(r) store$meta[index[[r]], , drop = FALSE]),
           names(index))
}
