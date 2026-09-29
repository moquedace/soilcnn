# ── kNNDM: folds shaped by where the map will actually be predicted ───────────
#
# THE ARGUMENT AGAINST BLOCKS, MADE BY SOMEONE WHO BUILT THE BLOCKS.
#
# `spatial_folds()` cuts the ground into squares and buffers the edges. It works,
# it is understandable from memory, and it has one weakness that no amount of
# care removes: the block size and the buffer width are CHOICES, and nothing in
# the data says whether they were the right ones. This project picked a block
# size by measuring fold balance -- a criterion of convenience, not of validity.
#
# kNNDM (Linnenbrink, Milà, Ludwig & Meyer, 2024, Geosci. Model Dev. 17,
# 5897-5912) starts somewhere better: THE RIGHT VALIDATION DEPENDS ON WHERE YOU
# WILL PREDICT. It builds folds so that the distribution of distances from a
# validation point to its nearest training point imitates the distribution of
# distances from a PREDICTION pixel to its nearest training point. It minimises
# the Wasserstein statistic W between those two empirical distributions.
#
# The elegant consequence, and the reason it is worth the dependency: when the
# samples are well spread over the prediction area, kNNDM converges by itself to
# ordinary random k-fold. It does not IMPOSE spatial separation when separation
# is not what prediction will face -- which is exactly the correct criticism of
# blind blocking, and exactly what a buffer cannot decide for itself.
#
# WHY THE PROJECTION IS NOT A DETAIL HERE.
#
# kNNDM compares distances, so the distances have to mean something. These
# points are global lon/lat, where a degree is 111 km at the equator and 20 km
# at 70 degrees north. Worse, the nearest-neighbour search is O(n log n) with a
# kd-tree on a plane and O(n^2) in time AND memory on a sphere -- 31,000 points
# is 9.6e8 pairs, about 7.7 GB for one double matrix.
#
# So the coordinates are projected to an EQUAL-AREA projection (Mollweide by
# default) before anything is measured. Equal-area rather than conformal because
# what is being compared is distance across the whole domain, and an equal-area
# projection keeps the comparison honest at the scale the folds are cut at.
# It was measured on the real coordinates rather than assumed
# (_measure_knndm.R, a one-off removed on 2026-09-28; its numbers are in
# docs/project_log.md).
#
# WHY THERE IS NO BUFFER ARGUMENT, AND WHEN THAT IS WRONG.
#
# A buffer exists to stop a validation point sitting next to a training point.
# kNNDM's whole premise is that whether that is a problem depends on prediction:
# this map is predicted wall to wall, so prediction pixels DO sit next to
# training points, and forcing them apart in validation would measure a
# scenario that never happens. Adding a buffer here would undo the method.
#
# What a buffer was also catching, and what kNNDM does not address, is two
# profiles in the SAME raster cell: identical input, different target, one
# scored on what the other trained. That is a defect under any plan and is
# handled where it belongs -- `spatial_overlap_report()` in R/diagnostics.R.

.need_knndm <- function() {
  miss <- c("CAST", "sf")[!c(requireNamespace("CAST", quietly = TRUE),
                             requireNamespace("sf", quietly = TRUE))]
  if (length(miss)) {
    stop("knndm_folds() needs the package(s): ", paste(miss, collapse = ", "),
         ".\n  install.packages(c(", paste(sprintf('"%s"', miss),
                                           collapse = ", "), "))\n",
         "  The reference implementation is used rather than a re-derivation: ",
         "a\n  published CV method re-coded is a method that quietly differs ",
         "from the paper.", call. = FALSE)
  }
  invisible(TRUE)
}

#' Coordinates in a projection where distances mean something.
#'
#' @param x,y   Coordinates.
#' @param crs   The CRS they are in. 4326 (lon/lat WGS84) by default.
#' @param to    Target projection. Mollweide by default: equal-area, global.
#' @return A two-column matrix of projected coordinates, in metres.
#' @noRd
project_xy <- function(x, y, crs = 4326,
                       to = "+proj=moll +lon_0=0 +datum=WGS84 +units=m") {
  .need_knndm()
  pts <- sf::st_as_sf(data.frame(x = as.numeric(x), y = as.numeric(y)),
                      coords = c("x", "y"), crs = crs)
  if (is.null(to) || identical(to, FALSE)) return(sf::st_coordinates(pts))
  sf::st_coordinates(sf::st_transform(pts, to))
}

#' Fold plan from k-fold Nearest Neighbour Distance Matching.
#'
#' @param meta       Store metadata: needs x, y and sample_id.
#' @param k          Number of folds.
#' @param predpoints Where the map will be predicted: a data frame with x and y,
#'   or an sf object. REQUIRED unless `modeldomain` is passed through `...` --
#'   the method has no meaning without it, so it is not defaulted.
#' @param test_ids   Frozen test sample_ids. Preferred over `hold_out_test`
#'   whenever several plans are to be compared on the same held-out data.
#' @param hold_out_test Carve a test set by running kNNDM with k + 1 folds and
#'   holding one out. FALSE by default: a test set that changes with k is not a
#'   frozen test set.
#' @param crs        CRS of `meta$x`/`meta$y`. 4326 by default.
#' @param project_to Projection used for the distance comparison, or NULL to
#'   use the coordinates as they are (correct only if already projected).
#' @param seed       Seed, for the sampling kNNDM does internally.
#' @param ...        Passed to CAST::knndm() -- `maxp`, `clustering`,
#'   `samplesize`, `modeldomain`, `space`.
#' @return A fold_plan.
#' @export
knndm_folds <- function(meta, k = 5L, predpoints = NULL, test_ids = NULL,
                        hold_out_test = FALSE, crs = 4326,
                        project_to = "+proj=moll +lon_0=0 +datum=WGS84 +units=m",
                        seed = 42L, ...) {
  # ARGUMENTS FIRST, DEPENDENCIES SECOND.
  #
  # These checks are true whether or not CAST is installed, so running them
  # first gives a person with a typo the useful message instead of an install
  # instruction -- and lets the argument contract be tested on a machine that
  # does not have the optional dependency.
  if (!all(c("x", "y") %in% names(meta))) {
    stop("meta needs x and y columns.", call. = FALSE)
  }
  k <- as.integer(k)
  if (k < 2L) stop("k must be at least 2.", call. = FALSE)

  dots <- list(...)
  if (is.null(predpoints) && is.null(dots$modeldomain)) {
    stop("kNNDM needs to know WHERE THE MAP WILL BE PREDICTED -- that is the ",
         "whole\n  method. Pass `predpoints` (a sample of the prediction ",
         "pixels; a few\n  thousand regularly spaced is plenty) or ",
         "`modeldomain` (an sf polygon).\n",
         "  Defaulting it would turn kNNDM into an expensive random split.",
         call. = FALSE)
  }

  .need_knndm()

  n  <- nrow(meta)
  xy <- project_xy(meta$x, meta$y, crs = crs, to = project_to)
  out_crs <- if (is.null(project_to)) crs else project_to
  tp <- sf::st_as_sf(data.frame(X = xy[, 1], Y = xy[, 2]),
                     coords = c("X", "Y"), crs = out_crs)

  # Defined after `tp` on purpose: it reprojects an sf onto the SAME frame the
  # training points ended up in. Comparing distances measured in two different
  # projections is the silent version of comparing them in two different units.
  as_sf <- function(obj) {
    if (inherits(obj, "sf")) return(sf::st_transform(obj, sf::st_crs(tp)))
    if (!all(c("x", "y") %in% names(obj))) {
      stop("`predpoints` needs x and y columns, or must be an sf object.",
           call. = FALSE)
    }
    m <- project_xy(obj$x, obj$y, crs = crs, to = project_to)
    sf::st_as_sf(data.frame(X = m[, 1], Y = m[, 2]), coords = c("X", "Y"),
                 crs = out_crs)
  }
  pp <- if (!is.null(predpoints)) as_sf(predpoints) else NULL

  run_knndm <- function(idx, kk) {
    args <- c(list(tpoints = tp[idx, ], k = as.integer(kk)), dots)
    if (!is.null(pp)) args$predpoints <- pp
    res <- with_local_seed(seed, do.call(CAST::knndm, args))
    cl  <- as.integer(res$clusters)
    if (length(cl) != length(idx)) {
      stop("CAST::knndm() returned ", length(cl), " fold labels for ",
           length(idx), " points.", call. = FALSE)
    }
    list(clusters = cl, W = res$W)
  }

  # ── the test set ────────────────────────────────────────────────────────────
  #
  # Frozen ids are preferred and come first. Otherwise, one kNNDM fold out of
  # k + 1 becomes the test set -- which is more principled than it looks: that
  # fold is already shaped so its distance-to-training distribution matches
  # prediction, which is precisely what a test set should be.
  #
  # The fold chosen is the one closest in size to the mean, deterministically.
  # An arbitrary index would make the test set depend on CAST's internal
  # labelling, and a draw would make it depend on a second seed.
  W_test <- NA_real_
  if (!is.null(test_ids)) {
    if (!"sample_id" %in% names(meta)) {
      stop("meta needs sample_id to apply a frozen test set.", call. = FALSE)
    }
    pos <- match(as.character(test_ids), as.character(meta$sample_id))
    if (anyNA(pos)) {
      stop(sum(is.na(pos)), " frozen test sample_id(s) are not in meta.",
           call. = FALSE)
    }
    test <- sort(pos)
  } else if (isTRUE(hold_out_test)) {
    first <- run_knndm(seq_len(n), k + 1L)
    sizes <- as.integer(table(factor(first$clusters, levels = seq_len(k + 1L))))
    pick  <- which.min(abs(sizes - mean(sizes)))
    test  <- which(first$clusters == pick)
    W_test <- first$W
  } else {
    test <- integer(0)
  }

  pool <- setdiff(seq_len(n), test)
  if (length(pool) < k) {
    stop("Only ", length(pool), " point(s) left after the test set, for k = ",
         k, " folds.", call. = FALSE)
  }

  main <- run_knndm(pool, k)
  assignment <- main$clusters

  asg <- tibble::tibble(sample_id = meta$sample_id[pool], fold = assignment)

  plan <- .new_fold_plan(
    .folds_from_assignment(assignment, pool, test, k),
    "knndm_folds",
    list(k = k, test_frac = round(length(test) / n, 4),
         n_test = length(test), seed = seed,
         # W IS THE QUALITY OF THE PLAN, not decoration. It is the Wasserstein
         # distance between the CV and the prediction distance distributions:
         # small means the folds imitate prediction, large means they do not and
         # the CV estimate is about a scenario that will not occur.
         #
         # It is in the COORDINATE UNITS (metres here), so it is comparable
         # between plans over the same points and meaningless between datasets
         # or projections. Read it against the spacing of your own data -- a W
         # of 100 km is small over a continent and enormous over a farm.
         W = signif(main$W, 4),
         W_test_split = if (is.finite(W_test)) signif(W_test, 4) else NA_real_,
         projection = if (is.null(project_to)) "none (already projected)"
                      else project_to),
    meta, assignment = asg
  )
  # WHAT A REFIT NEEDS TO CUT ITS VALIDATION THE SAME WAY. The final model
  # stops on a validation set carved by the tuning plan's own criterion
  # (refit_split()), and kNNDM's criterion IS the prediction points, in the
  # frame their distances were measured in. Kept beside the plan rather than
  # in params, which print.fold_plan() prints: a table of points is not a
  # parameter. A plan made before this (2026-09-29) has none, and a refit of
  # it must be given the points.
  plan$knndm <- list(predpoints = predpoints, crs = crs, project_to = project_to,
                     args = dots)
  plan
}

#' A regular sample of the prediction area, as kNNDM's `predpoints`.
#'
#' kNNDM needs the places the map will cover, not the places it was trained on.
#' A few thousand regularly spaced pixels describe that well enough -- the
#' method compares DISTRIBUTIONS of distances, and a regular sample estimates
#' one cheaply. A random sample would work too and is noisier for the same size.
#'
#' @param raster A SpatRaster (terra) covering the prediction area.
#' @param size   How many points to draw.
#' @return A data frame with x and y, in the raster's own CRS.
#' @export
prediction_sample <- function(raster, size = 5000L) {
  if (!requireNamespace("terra", quietly = TRUE)) {
    stop("prediction_sample() needs the 'terra' package.", call. = FALSE)
  }
  s <- terra::spatSample(raster, size = size, method = "regular",
                         na.rm = TRUE, xy = TRUE, values = FALSE)
  s <- as.data.frame(s)
  if (nrow(s) == 0L) {
    stop("The regular sample came back empty -- every cell drawn was NA. ",
         "Pass a raster that covers the prediction area.", call. = FALSE)
  }
  tibble::tibble(x = s$x, y = s$y)
}
