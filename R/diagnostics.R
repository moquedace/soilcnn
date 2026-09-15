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
    print(dplyr::slice_head(changed, n = n_show), n = Inf, width = Inf)
    unchanged <- sum(cmp$diff$status == "=")
    if (unchanged > 0L) cat("  (", unchanged, " inalterado(s))
", sep = "")
  }
  invisible(changed)
}
