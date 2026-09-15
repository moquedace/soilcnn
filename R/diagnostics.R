# Pipeline diagnostics
#
# Checks that answer questions the rest of the pipeline cannot answer about
# itself. Three of the four exist because a failure they would have caught
# actually happened here.
#
# The unit tests (tests/) prove the CODE is right, on synthetic data, in
# seconds. These prove THIS RUN is right, on the real data. Different
# questions, both cheap, and neither substitutes for the other.

# ── 1. cross-check: patch centres against the point table ─────────────────────
#
# The strongest check in this file, and the one nothing was doing.
#
# The centre cell of every patch MUST equal the value the point table holds for
# that predictor at that point -- they are the same cell of the same raster,
# reached by two completely independent routes:
#
#   point table : terra::extract() on a SpatVector of coordinates
#   patch store : cellFromXY() -> row/col -> patch_cell_index() -> array
#
# A CRS mismatch, a row/col swap, an off-by-one, a channel reordering, a stale
# raster directory -- any of them breaks this equality. Synthetic tests prove
# the index algebra is correct; this proves the algebra was applied to the
# right place in the real raster.
#
# Cost: reads the SMALLEST window only (3x3 is ~0.5 GB), since the centre is
# the same cell in every window.

#' Compare patch centres with the point-table values.
#'
#' @param patch_dir  Patch store directory.
#' @param points     Point table, already aligned to the store's meta (see
#'   align_points_to_meta()).
#' @param predictors Channel names, in channel order.
#' @param window     Which window to read (default: the smallest available).
#' @param tol        Absolute tolerance. Both routes are double precision doing
#'   the same arithmetic, so agreement should be exact; the tolerance exists
#'   only to absorb CSV round-tripping of the point table.
#' @return list(ok, n_points, n_channels, n_mismatch, worst, by_channel)
check_patch_centres <- function(patch_dir, points, predictors,
                                window = NULL, tol = 1e-6) {

  manifest <- readRDS(file.path(patch_dir, "patch_manifest.rds"))
  available <- as.integer(trimws(strsplit(manifest$windows_extracted[1], ",")[[1]]))
  if (is.null(window)) window <- min(available)

  arr <- readRDS(patch_window_path(patch_dir, window))
  d   <- dim(arr)
  if (d[1] != nrow(points)) {
    stop("Patch store has ", d[1], " points but the point table has ",
         nrow(points), " -- align them first.", call. = FALSE)
  }
  if (d[2] != length(predictors)) {
    stop("Patch store has ", d[2], " channels but ", length(predictors),
         " predictor names were given.", call. = FALSE)
  }

  centre <- (window + 1L) %/% 2L
  got    <- arr[, , centre, centre, drop = TRUE]   # [n_points, n_channels]
  rm(arr); gc(verbose = FALSE)

  want <- as.matrix(points[, predictors, drop = FALSE])

  diff_abs <- abs(got - want)
  diff_abs[is.na(got) & is.na(want)] <- 0          # NA on both sides agrees
  bad <- !is.na(diff_abs) & diff_abs > tol
  bad[is.na(got) != is.na(want)] <- TRUE           # NA on one side only

  by_channel <- tibble::tibble(
    predictor  = predictors,
    n_mismatch = as.integer(colSums(bad)),
    worst_diff = apply(diff_abs, 2, function(z) if (all(is.na(z))) NA_real_
                       else max(z, na.rm = TRUE))
  ) %>%
    dplyr::filter(n_mismatch > 0L) %>%
    dplyr::arrange(dplyr::desc(n_mismatch))

  list(
    ok         = sum(bad) == 0L,
    window     = window,
    n_points   = d[1],
    n_channels = d[2],
    n_cells    = prod(dim(bad)),
    n_mismatch = sum(bad),
    worst      = suppressWarnings(max(diff_abs, na.rm = TRUE)),
    by_channel = by_channel
  )
}

# ── 2. overlap between splits: two different things, never conflated ──────────
#
# ONE OF THESE IS A DEFECT. THE OTHER IS NOT.
#
#   same raster cell   Two points in the same pixel have a patch that is
#                      IDENTICAL, bit for bit. The model can be scored on an
#                      input it was trained on. This is a defect under ANY
#                      split -- random, spatial, grouped -- because it is not
#                      about geography, it is about duplicate inputs.
#
#   patches overlap    Two nearby points share some of their surrounding
#                      pixels. This is NOT a defect. It is what neighbouring
#                      samples look like. Under a RANDOM split it is the very
#                      condition being measured -- "how well does this predict
#                      at new points drawn from the same spatial distribution"
#                      -- so calling it leakage misstates the question the
#                      split was chosen to answer.
#
#                      It becomes worth looking at only when the plan claims to
#                      be SPATIAL, where the point of the exercise is to score
#                      on ground the model has not seen. There it describes how
#                      well the separation held; it is still not a defect.
#
# The `matters` column carries that distinction so no caller has to remember
# it, and so the 99 warns on the first and merely reports the second.

#' Quantify how much of one split shares raster cells with another.
#'
#' @param row_ids,col_ids Integer raster row/col of every point.
#' @param split           Character vector of split labels, same length.
#' @param windows         Window sizes to report shared pixels for.
#' @param reference       Split whose points count as "seen in training".
#' @return A tibble, one row per (split, criterion), with a `matters` flag:
#'   TRUE for identical patches (a defect under any split) and FALSE for shared
#'   pixels between neighbours (not a defect -- see the note above).
spatial_overlap_report <- function(row_ids, col_ids, split,
                                   windows = c(3L, 9L, 15L),
                                   reference = "train") {
  stopifnot(length(row_ids) == length(col_ids),
            length(row_ids) == length(split))

  is_ref <- split == reference
  if (!any(is_ref)) {
    stop("No points in the reference split '", reference, "'.", call. = FALSE)
  }

  # Exact-cell collision: a hash of (row, col) is enough and is O(n).
  ref_cell <- unique(paste(row_ids[is_ref], col_ids[is_ref], sep = "_"))

  out <- list()
  for (s in setdiff(unique(split), reference)) {
    sel <- split == s
    n_s <- sum(sel)

    same <- sum(paste(row_ids[sel], col_ids[sel], sep = "_") %in% ref_cell)
    out[[length(out) + 1L]] <- tibble::tibble(
      split = s, criterion = "identical patch (same raster cell)",
      window = NA_integer_, matters = TRUE,
      n = same, pct = round(100 * same / n_s, 2), n_split = n_s)

    # Shared pixels: two patches of width w have a cell in common exactly when
    # their centres are within w-1 cells in BOTH axes -- a Chebyshev square.
    #
    # This used to bucket by HALF the window and test the 9 neighbours, with no
    # exact check. Two errors in one: the radius was half what the geometry
    # needs, so real overlaps were missed, and nothing verified the candidates
    # the buckets returned. Caught by asserting against a layout whose answer
    # was known -- three points placed exactly two cells away were reported as
    # not sharing anything.
    #
    # Now the buckets NARROW the search and an exact test DECIDES it, the same
    # shape apply_buffer() uses.
    for (w in windows) {
      hw   <- max(1L, w - 1L)
      rb   <- row_ids %/% hw
      cb   <- col_ids %/% hw
      ref_i <- which(is_ref)
      sel_i <- which(sel)
      ref_by <- split(ref_i, paste(rb[ref_i], cb[ref_i], sep = "_"))
      key_s  <- paste(rb[sel_i], cb[sel_i], sep = "_")

      hit <- rep(FALSE, n_s)
      for (bk in unique(key_s)) {
        rows <- which(key_s == bk)
        rc   <- as.integer(strsplit(bk, "_", fixed = TRUE)[[1]])
        cand <- unlist(ref_by[paste(rep(rc[1] + (-1:1), each = 3L),
                                    rep(rc[2] + (-1:1), times = 3L),
                                    sep = "_")], use.names = FALSE)
        if (length(cand) == 0L) next
        cr <- row_ids[cand]; cc <- col_ids[cand]
        hit[rows] <- vapply(rows, function(i) {
          any(abs(cr - row_ids[sel_i[i]]) <= hw &
              abs(cc - col_ids[sel_i[i]]) <= hw)
        }, logical(1))
      }
      out[[length(out) + 1L]] <- tibble::tibble(
        split = s, criterion = paste0("shares pixels (", w, "x", w, ")"),
        window = w, matters = FALSE,
        n = sum(hit), pct = round(100 * sum(hit) / n_s, 2), n_split = n_s)
    }
  }
  dplyr::bind_rows(out)
}

# ── 3. run snapshots: what changed since last time? ───────────────────────────
#
# In a refactor the question asked after every run is "did anything move?", and
# answering it meant scrolling back through old output by hand. A snapshot per
# run turns that into a diff.
#
# "Everything identical" is the result you usually want, and it is the one that
# is hardest to confirm by eye.

#' Record a named set of scalar values for this run.
#'
#' @param values  Named list/vector of scalars (numeric or character).
#' @param dir     Where snapshots live.
#' @param label   Snapshot name; defaults to a timestamp.
write_run_snapshot <- function(values, dir, label = NULL) {
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  if (is.null(label)) label <- format(Sys.time(), "%Y%m%d_%H%M%S")

  # unname(): vapply() over a named list returns a NAMED vector, and those
  # names ride along into the tibble column. Harmless to `==`, but it makes the
  # written and the in-memory value fail identical() -- a difference in
  # attributes reading as a difference in content.
  snap <- tibble::tibble(
    key   = names(values),
    value = unname(vapply(values, function(z) as.character(z[1]), character(1)))
  )
  f <- file.path(dir, paste0("snapshot_", label, ".csv"))
  readr::write_csv2(snap, f)
  invisible(f)
}

# Snapshots are compared as TEXT, so reading one has to give back text.
#
# Without this, read_csv2() guesses the type: under a ";"/"," locale it reads
# the POINT in "31.190645" as a THOUSANDS separator and returns 31190645. The
# written value and the value read back then differ by formatting alone, and
# the diff reports a change where nothing changed -- precisely the noise this
# mechanism exists to remove.
.read_snapshot <- function(path) {
  suppressMessages(
    readr::read_csv2(path,
                     col_types = readr::cols(.default = readr::col_character()))
  )
}

#' Compare this run's values with the most recent earlier snapshot.
#'
#' @return list(has_previous, previous_file, diff) where `diff` is a tibble of
#'   every key with old value, new value and a status: `=`, `changed`, `new`,
#'   `gone`.
compare_run_snapshot <- function(values, dir, exclude = character(0)) {
  prev_files <- sort(list.files(dir, pattern = "^snapshot_.*\\.csv$",
                                full.names = TRUE), decreasing = TRUE)
  now <- tibble::tibble(
    key = names(values),
    new = unname(vapply(values, function(z) as.character(z[1]), character(1)))
  )

  if (length(prev_files) == 0L) {
    return(list(has_previous = FALSE, previous_file = NA_character_,
                diff = dplyr::mutate(now, old = NA_character_,
                                     status = "new")))
  }

  prev <- .read_snapshot(prev_files[1])
  d <- dplyr::full_join(
    dplyr::rename(prev, old = value), now, by = "key"
  ) %>%
    dplyr::mutate(
      status = dplyr::case_when(
        is.na(old)  ~ "new",
        is.na(new)  ~ "gone",
        old == new  ~ "=",
        TRUE        ~ "changed"
      )
    ) %>%
    dplyr::filter(!key %in% exclude) %>%
    dplyr::select(key, old, new, status)

  list(has_previous = TRUE, previous_file = basename(prev_files[1]), diff = d)
}

#' Print a snapshot comparison, changes first.
# cat(), not message(): message() writes to stderr and print() to stdout, and
# in the RStudio console the two land on the SAME line ("...changed:# A
# tibble"). Report output has to leave through one channel to keep its order.
print_snapshot_diff <- function(cmp, n_show = 40L) {
  if (!cmp$has_previous) {
    cat("  Sem snapshot anterior -- este vira a referencia.
")
    return(invisible(NULL))
  }
  cat("  Comparando com: ", cmp$previous_file, "
", sep = "")

  changed <- dplyr::filter(cmp$diff, status != "=")
  if (nrow(changed) == 0L) {
    cat("  TUDO IDENTICO ao run anterior (", nrow(cmp$diff), " valores).
",
        sep = "")
  } else {
    cat("  ", nrow(changed), " de ", nrow(cmp$diff), " valores mudaram:
",
        sep = "")
    print_wide(dplyr::slice_head(changed, n = n_show), n = Inf)
    unchanged <- sum(cmp$diff$status == "=")
    if (unchanged > 0L) cat("  (", unchanged, " inalterado(s))
", sep = "")
  }
  invisible(changed)
}

# ── How much does choosing the best epoch flatter the validation metric? ──────
#
# THE QUESTION.
#
# Early stopping picks the epoch with the lowest validation loss, and the
# validation metric is then read AT THAT EPOCH -- from the same data that chose
# it. The number reported is therefore the minimum of a noisy sequence, and the
# minimum of a noisy sequence is below its mean by construction. Some of the
# validation score is selection, not skill.
#
# This matters here beyond tidiness: it is the argument for whether a separate
# early-stopping split is needed. If the bias is small, the current design
# stands and 15% of the training data stays in training. If it is large, the
# ranking between configs may partly be a ranking of who got the luckiest
# epoch.
#
# THE ESTIMATOR, AND WHAT IT IS NOT.
#
# Around the chosen epoch the trajectory has flattened -- that is what triggers
# the patience counter. Treating the epochs in that plateau as exchangeable
# draws from one distribution, the optimism of taking their minimum is
#
#     mean(plateau) - min(plateau)
#
# It costs nothing: the histories are already on disk, and NOTHING IS
# RETRAINED.
#
# What it is NOT: an unbiased estimate of the bias in CCC. It is measured in
# LOSS units, on the plateau only, and it assumes the plateau is flat rather
# than still descending -- an assumption that inflates the estimate whenever
# training had not really converged. It is a screening measurement: a clear
# "small" is trustworthy, a clear "large" is a reason to run the real
# experiment (a third split), and a borderline answer means run it too.

#' Estimate the optimism of early stopping from run histories.
#'
#' @param history_dir Directory of *_history.csv written by a run.
#' @param plateau     How many epochs around the chosen one to treat as
#'   exchangeable. Defaults to 20; it should be no larger than the patience
#'   used, or the window reaches back into the part still descending.
#' @param loss_col    Column holding the validation loss.
#' @return A tibble, one row per unit, plus the attribute "summary".
early_stopping_bias <- function(history_dir, plateau = 20L,
                                loss_col = "val_loss") {
  files <- list.files(history_dir, pattern = "_history\.csv$", full.names = TRUE)
  if (length(files) == 0L) {
    stop("No *_history.csv in: ", history_dir,
         "\nThis reads the per-epoch histories a run already wrote; it ",
         "retrains nothing.", call. = FALSE)
  }

  rows <- lapply(files, function(f) {
    h <- suppressWarnings(readr::read_csv2(f, show_col_types = FALSE))
    if (!loss_col %in% names(h) || nrow(h) < 3L) return(NULL)
    v <- as.numeric(h[[loss_col]])
    keep <- is.finite(v)
    v <- v[keep]
    if (length(v) < 3L) return(NULL)

    best_i <- which.min(v)
    # The window is centred on the chosen epoch and clipped to the trajectory.
    # Centred rather than trailing: the epochs just BEFORE the minimum are as
    # much part of the plateau as those after, and using only what came after
    # biases the window towards the tail where the loss may be rising again.
    half <- max(1L, plateau %/% 2L)
    lo   <- max(1L, best_i - half)
    hi   <- min(length(v), best_i + half)
    w    <- v[lo:hi]

    tibble::tibble(
      unit_id      = sub("_history\.csv$", "", basename(f)),
      n_epochs     = length(v),
      best_epoch   = best_i,
      best_loss    = v[best_i],
      plateau_n    = length(w),
      plateau_mean = mean(w),
      plateau_sd   = stats::sd(w),
      # The two numbers the question turns on.
      bias_abs     = mean(w) - v[best_i],
      bias_rel     = (mean(w) - v[best_i]) / abs(mean(w)),
      # Did training actually flatten? A plateau still descending steeply makes
      # the estimate an upper bound rather than an estimate, and saying so is
      # better than reporting a number that looks the same either way.
      still_descending = (mean(w[seq_len(length(w) %/% 2L)]) -
                          mean(w[(length(w) %/% 2L + 1L):length(w)])) >
                         2 * stats::sd(w)
    )
  })

  out <- dplyr::bind_rows(rows)
  if (nrow(out) == 0L) {
    stop("No usable histories found (need >= 3 finite epochs of '",
         loss_col, "').", call. = FALSE)
  }

  attr(out, "summary") <- list(
    n_units          = nrow(out),
    median_bias_abs  = stats::median(out$bias_abs, na.rm = TRUE),
    median_bias_rel  = stats::median(out$bias_rel, na.rm = TRUE),
    max_bias_rel     = max(out$bias_rel, na.rm = TRUE),
    pct_descending   = 100 * mean(out$still_descending, na.rm = TRUE),
    plateau          = plateau
  )
  out
}

#' Print the verdict from early_stopping_bias().
print_early_stopping_bias <- function(bias, threshold_rel = 0.02) {
  s <- attr(bias, "summary")
  cat("\n-- Optimism of early stopping (from histories, nothing retrained) --\n")
  cat(sprintf("  units                 : %d\n", s$n_units))
  cat(sprintf("  plateau window        : %d epochs around the chosen one\n",
              s$plateau))
  cat(sprintf("  median bias           : %.6f loss (%.2f%% of the plateau mean)\n",
              s$median_bias_abs, 100 * s$median_bias_rel))
  cat(sprintf("  worst unit            : %.2f%%\n", 100 * s$max_bias_rel))
  cat(sprintf("  still descending      : %.0f%% of units\n", s$pct_descending))

  if (s$pct_descending > 25) {
    cat("\n  -> A quarter or more of the units had not flattened. These\n",
        "     numbers are an UPPER BOUND, not an estimate: raise patience or\n",
        "     n_epochs before reading anything into them.\n", sep = "")
  } else if (s$median_bias_rel < threshold_rel) {
    cat("\n  -> SMALL. Choosing the epoch on the validation fold costs about\n",
        "     this much, and it is not worth a third split: carving one would\n",
        "     take 15% out of TRAINING to remove a bias of this size.\n", sep = "")
  } else {
    cat("\n  -> LARGE ENOUGH TO MATTER. Part of each config's validation score\n",
        "     is the luck of its best epoch. Worth the real experiment: a\n",
        "     separate early-stopping split, scored on the untouched fold.\n", sep = "")
  }
  invisible(bias)
}
