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
# A fold plan (R/resample.R) is what the pipeline uses
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
           "\nA patch store is written by stage 02 and must contain ",
           "patches_wNN.rds, patch_meta.csv and patch_manifest.rds.",
           call. = FALSE)
    }
  }

  manifest <- readRDS(manifest_path)
  meta     <- safe_read_csv2(meta_path)

  # THE STORE'S OWN VERDICT IS READ. Stage 02 writes store_complete = FALSE when
  # a window failed to save, prints a warning, and stops -- and this loader
  # then opened the store anyway, because nothing here looked. A store that
  # declared itself incomplete trained a model.
  if ("store_complete" %in% names(manifest) &&
      !isTRUE(as.logical(manifest$store_complete[1]))) {
    stop("This patch store's own manifest says it is INCOMPLETE (a window did ",
         "not save; see patch_files.csv in the store).\n  Re-run stage 02 for ",
         patch_dir, call. = FALSE)
  }

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

# The split used to be read from a `dataset_role` column on the store's meta.
# It is not any more, and the reader was removed rather than left lying about:
# the store carries points, coordinates and values, and WHO TRAINS is decided
# by a fold plan (R/resample.R) in seconds. That is what stops a change of
# split strategy from costing a re-extraction.

# ── the store's spec, and refusing a configuration it cannot serve ────────────
#
# Three things force a re-extraction: the PREDICTORS, the WINDOWS and the
# TARGET. Everything else about a run -- the split, the folds, the buffer, the
# seeds, the grid -- is decided downstream and costs seconds to change.
#
# So those three are recorded in the manifest when the store is built, and
# compared against what the current configuration asks for before any of it is
# used. A package user who changes a window gets a clear failure in seconds,
# naming the fix, rather than a silently wrong result or an extraction they
# discover was wasted.

#' What a patch store was built under.
#'
#' @param store From load_patch_store().
#' @return list(predictors, windows, target_col, target_transform, cell_size),
#'   with NA for anything a store written before this was recorded.
store_spec <- function(store) {
  m <- store$manifest
  get1 <- function(nm) if (nm %in% names(m)) m[[nm]][1] else NA
  list(
    predictors       = store$predictors,
    # What the STORE HOLDS, not the subset this session loaded: store$window_sizes
    # is the `window_sizes =` argument, so asking the store what it has through
    # that field would only ever echo the question back.
    windows          = as.integer(trimws(
      strsplit(as.character(m$windows_extracted[1]), ",")[[1]])),
    target_col       = get1("target_col"),
    target_transform = get1("target_transform"),
    cell_size        = suppressWarnings(as.numeric(get1("cell_size")))
  )
}

#' Refuse a configuration the store cannot serve.
#'
#' Reports EVERY mismatch it finds, not just the first: discovering one, fixing
#' it, re-running and discovering the next is the slow way to learn there were
#' three.
#'
#' @param store       From load_patch_store().
#' @param predictors  Channel names the current configuration expects.
#' @param windows     Window sizes the current grid needs.
#' @param target_col  Target column name, or NULL to skip.
#' @param cell_size   Raster resolution now, or NULL to skip.
#' @param strict      TRUE (default) stops; FALSE returns the messages, which
#'   is what a reporting script wants.
check_store_spec <- function(store, predictors = NULL, windows = NULL,
                             target_col = NULL, cell_size = NULL,
                             strict = TRUE) {
  spec <- store_spec(store)
  bad  <- character(0)

  if (!is.null(predictors)) {
    gone  <- setdiff(predictors, spec$predictors)
    extra <- setdiff(spec$predictors, predictors)
    if (length(gone) || length(extra)) {
      bad <- c(bad, sprintf(
        paste0("PREDICTORS differ: the store holds %d, this configuration ",
               "expects %d.%s%s\n  -> re-extract (stage 02), or restore the ",
               "predictor set this store was built with."),
        length(spec$predictors), length(predictors),
        if (length(gone))  sprintf("\n  missing from the store: %s",
                                   paste(utils::head(gone, 6), collapse = ", ")) else "",
        if (length(extra)) sprintf("\n  in the store but not expected: %s",
                                   paste(utils::head(extra, 6), collapse = ", ")) else ""))
    } else if (!identical(as.character(predictors), as.character(spec$predictors))) {
      bad <- c(bad, paste0(
        "PREDICTOR ORDER differs from the store's. The order is the contract ",
        "that ties channels to bands -- feeding the network one channel while ",
        "the map is built from another produces no error at all.",
        "\n  -> rebuild predictor_type_table.csv from the same source."))
    }
  }

  if (!is.null(windows)) {
    miss <- setdiff(as.integer(windows), as.integer(spec$windows))
    if (length(miss)) {
      bad <- c(bad, sprintf(
        paste0("WINDOWS %s are not in this store, which holds %s.",
               "\n  -> re-extract (stage 02) with the wider set, or change ",
               "the grid to stay inside it."),
        paste(miss, collapse = ", "), paste(spec$windows, collapse = ", ")))
    }
  }

  if (!is.null(target_col) && !is.na(spec$target_col) &&
      !identical(as.character(target_col), as.character(spec$target_col))) {
    bad <- c(bad, sprintf(
      "TARGET differs: the store was built for '%s', this run wants '%s'.%s",
      spec$target_col, target_col,
      "\n  -> re-extract (stage 02): the patches are the same, but the stored targets are not."))
  }

  if (!is.null(cell_size) && !is.na(spec$cell_size) &&
      abs(cell_size - spec$cell_size) > 1e-9) {
    bad <- c(bad, sprintf(
      paste0("RESOLUTION differs: the store was built at %.8f, the rasters ",
             "now read %.8f.\n  -> the predictor directory changed. Every ",
             "window, buffer and block size is in these units."),
      spec$cell_size, cell_size))
  }

  if (length(bad) == 0L) return(invisible(character(0)))
  msg <- paste0("This patch store cannot serve this configuration:\n\n",
                paste0("  * ", bad, collapse = "\n\n"))
  if (strict) stop(msg, call. = FALSE) else return(bad)
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

  # THE VALUES HAVE TO LINE UP; THE STORAGE TYPE DOES NOT.
  #
  # This was stopifnot(identical(...)), and identical() compares type as well
  # as value. In this pipeline both sides arrive from read_csv2() as doubles,
  # so it passed -- but a point table built in R carries integer sample_ids
  # against a store read from CSV, and the check then failed on 1L vs 1 with
  # the message "identical(...) is not TRUE", which names nothing and suggests
  # nothing.
  #
  # An id is a label. Compared numerically when both sides are numbers, and as
  # text otherwise -- never as.character() on numbers, because as.character(1e5)
  # is "1e+05" while as.character(100000L) is "100000", and a framework that
  # breaks above 99,999 points is worse than one that breaks loudly.
  same <- if (is.numeric(out$sample_id) && is.numeric(meta$sample_id)) {
    isTRUE(all.equal(as.numeric(out$sample_id), as.numeric(meta$sample_id)))
  } else {
    identical(as.character(out$sample_id), as.character(meta$sample_id))
  }
  if (!same) {
    bad <- which(as.character(out$sample_id) != as.character(meta$sample_id))
    stop("The aligned point table does not line up with the patch store: ",
         length(bad), " row(s) differ, first at position ", bad[1],
         " (points has '", out$sample_id[bad[1]], "', the store expects '",
         meta$sample_id[bad[1]], "').\n  sample_id must identify the same ",
         "observation in both, and it is what every fold index refers to.",
         call. = FALSE)
  }
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
#' @param index        One fold of a fold_plan: named list of row positions.
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
# ── the same fold, read as a table ────────────────────────────────────────────
#
# WHY A SECOND VIEW AND NOT A SECOND EXTRACTION.
#
# A Random Forest on the centre pixel is the classic digital soil mapping
# baseline, and until it is measured the CNN's CCC has no scale on it. But it
# must be measured on THE SAME FOLD -- same training rows, same scaling, same
# buffer -- or the comparison is between two experiments rather than two
# models, and the difference between them is unattributable.
#
# So the table is DERIVED from the tensors the fold cache already holds. It
# costs one pass over data that is already in memory, already scaled by this
# fold's training rows, already split. The alternative -- a separate tabular
# extraction from the rasters -- is a second code path producing numbers that
# only look like the first one's.
#
# WHAT THE FEATURES ARE, AND WHAT EACH ONE IS FOR.
#
#   centre        the value at the point. The classic baseline.
#   window_mean   the per-channel mean over each window. Context WITHOUT
#                 spatial structure: the same neighbourhood the convolution
#                 sees, with the arrangement thrown away.
#
# The second is the interesting one. If a Random Forest on centre + window
# means matches the CNN, then the convolution is doing averaging and its
# structure is worth nothing -- a falsifiable claim, measured cheaply, under
# folds that are identical by construction rather than by intention.

#' Read a fold cache as feature matrices.
#'
#' @param cache      The `cache` element of build_fold_cache().
#' @param predictors Channel names, in the store's order.
#' @param windows    Window sizes to summarise, or NULL for every one present.
#' @param features   Any of "centre", "window_mean".
#' @return Named list by role: list(x = matrix, y = numeric). Column names are
#'   `<predictor>` for the centre and `<predictor>_mean_w<W>` for the means.
fold_table_view <- function(cache, predictors, windows = NULL,
                            features = c("centre", "window_mean")) {
  features <- match.arg(features, several.ok = TRUE)
  roles    <- names(cache)

  have_keys <- setdiff(names(cache[[roles[1]]]), "y")
  if (is.null(windows)) {
    windows <- sort(as.integer(sub("^w", "", have_keys)))
  }
  keys <- patch_window_key(sort(as.integer(windows)))
  if (!all(keys %in% have_keys)) {
    stop("The cache does not hold window(s) ",
         paste(setdiff(keys, have_keys), collapse = ", "), call. = FALSE)
  }

  # The centre pixel is the SAME value in every window -- concentric patches
  # around one point -- so it is taken from the smallest, which is the cheapest
  # tensor to index. Verified rather than assumed, on the first role only: it
  # is a statement about the extraction geometry, and if it is false there
  # every downstream number is wrong in a way no metric would reveal.
  key_small <- keys[1]

  centre_of <- function(x) {
    hw <- as.integer(x$shape[3])
    c_ <- (hw %/% 2L) + 1L               # torch is 1-indexed
    as.matrix(x[, , c_, c_]$to(device = "cpu"))
  }

  out <- setNames(vector("list", length(roles)), roles)
  for (r in roles) {
    blocks <- list()

    if ("centre" %in% features) {
      m <- centre_of(cache[[r]][[key_small]])
      colnames(m) <- predictors
      blocks[["centre"]] <- m

      if (length(keys) > 1L) {
        # The geometry check, on one role and one window, costs one extra
        # indexing operation and catches a patch store whose windows were cut
        # around different points.
        m2 <- centre_of(cache[[r]][[keys[length(keys)]]])
        if (!isTRUE(all.equal(m[1:min(50L, nrow(m)), , drop = FALSE],
                              m2[1:min(50L, nrow(m2)), , drop = FALSE],
                              check.attributes = FALSE))) {
          stop("The centre pixel differs between window ", keys[1], " and ",
               keys[length(keys)], " for role '", r, "'. Concentric patches ",
               "around the same point must share their centre -- this store ",
               "was not cut that way, and every table feature built from it ",
               "would describe a different location than the tensors do.",
               call. = FALSE)
        }
      }
    }

    if ("window_mean" %in% features) {
      for (k in keys) {
        # A 1x1 window has no neighbourhood: its mean IS its centre, and a
        # duplicated column is a column a tree can split on twice for free.
        if (as.integer(sub("^w", "", k)) <= 1L) next
        mm <- as.matrix(cache[[r]][[k]]$mean(dim = c(3L, 4L))$to(device = "cpu"))
        colnames(mm) <- paste0(predictors, "_mean_", k)
        blocks[[paste0("mean_", k)]] <- mm
      }
    }

    if (length(blocks) == 0L) {
      stop("No features requested.", call. = FALSE)
    }
    out[[r]] <- list(
      x = do.call(cbind, blocks),
      y = as.numeric(as.matrix(cache[[r]]$y$to(device = "cpu")))
    )
  }
  out
}

fold_points_valid <- function(store, index) {
  setNames(lapply(names(index), function(r) store$meta[index[[r]], , drop = FALSE]),
           names(index))
}
