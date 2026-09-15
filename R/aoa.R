# ── Where the map should be believed ──────────────────────────────────────────
#
# WHY THIS EXISTS.
#
# A prediction map has a value at every pixel, including pixels whose predictor
# combination the model never saw. Nothing in the raster distinguishes the two,
# and the cross-validated CCC printed beside the map does not apply to the
# second kind at all: it was estimated where training data exists.
#
# Meyer & Pebesma (2021, Methods in Ecology and Evolution 12:1620-1633) make
# this operational. A dissimilarity index (DI) measures how far a pixel is, in
# PREDICTOR space, from the training data; the area of applicability (AOA) is
# the part of the map whose DI stays below a threshold derived from the
# cross-validated training data itself. Their recommendation is not that the
# AOA is nice to have -- it is that a map should not be published without it.
#
# HOW IT IS COMPUTED HERE.
#
#   1. Predictors are scaled. This framework already scales per fold, so the
#      SAME scaling that trained the model is the one the distances use --
#      which matters, because a distance in unscaled space is dominated by
#      whichever channel happens to have the largest units.
#   2. avg_dist is the mean pairwise distance WITHIN the training data. It is
#      the unit the DI is expressed in, which is what makes a DI comparable
#      between datasets rather than being an arbitrary number of metres.
#   3. DI of a new point = (distance to its nearest training point) / avg_dist.
#   4. The threshold is the outlier-removed maximum of the CROSS-VALIDATED
#      training DI: for each training point, the distance to the nearest
#      training point that is NOT in its fold. That is the key idea -- it asks
#      how dissimilar a point can be and still have been predicted well during
#      cross-validation, rather than picking a number.
#
# WHICH FEATURE SPACE, FOR A CNN.
#
# The model consumes patches, not points, so "predictor space" needs a choice.
# This uses the CENTRE PIXEL of each patch: it is the space a soil scientist
# reasons about, it is exactly the space the RF baseline lives in, and it makes
# the DI of the CNN's map comparable with the DI of the baseline's. A patch
# distance would be defensible too and is not offered, because a number nobody
# can interpret is worse than one that is only mostly right.
#
# WHAT THIS IS NOT.
#
# It is not an uncertainty estimate. A pixel inside the AOA is one whose
# predictor combination resembles the training data -- not one whose prediction
# is guaranteed accurate. Outside the AOA, the cross-validation error simply
# does not apply, and the honest report is "not applicable", not a wider
# interval.

#' Summarise a training set for dissimilarity-index computation.
#'
#' Computed once and reused for every prediction chunk: the training data does
#' not change, and the pairwise mean is the expensive part.
#'
#' @param x_train  Numeric matrix of SCALED training predictors (n x p), in the
#'   model's channel order.
#' @param weights  Optional per-predictor weights, e.g. variable importance.
#'   NULL gives every predictor the same weight, which is what Meyer & Pebesma
#'   fall back to when no importance is available.
#' @param max_pairs Cap on the number of point pairs used to estimate
#'   avg_dist. The full n^2 is unnecessary for a mean and quadratic in memory;
#'   a large random sample estimates it to more precision than the quantity
#'   deserves.
#' @param seed     Draw seed for that sample.
#' @return An object of class "di_reference".
di_reference <- function(x_train, weights = NULL, max_pairs = 2e6, seed = 42L) {
  x_train <- as.matrix(x_train)
  if (!is.numeric(x_train)) stop("x_train must be numeric.", call. = FALSE)
  n <- nrow(x_train); p <- ncol(x_train)
  if (n < 2L) stop("di_reference() needs at least 2 training rows.",
                   call. = FALSE)

  if (is.null(weights)) {
    weights <- rep(1, p)
  } else {
    if (length(weights) != p) {
      stop("weights has ", length(weights), " entries for ", p,
           " predictors.", call. = FALSE)
    }
    if (any(!is.finite(weights)) || any(weights < 0)) {
      stop("weights must be finite and non-negative.", call. = FALSE)
    }
    if (sum(weights) == 0) stop("weights are all zero.", call. = FALSE)
    # Normalised to mean 1 so the DI keeps its scale whatever the importance
    # units are -- a weighting that also rescales is two changes at once.
    weights <- weights * p / sum(weights)
  }

  xw <- sweep(x_train, 2L, sqrt(weights), "*")

  # A channel that is constant over the training rows contributes nothing to
  # any distance and would contribute nothing to a new point's either. Left in
  # rather than dropped: dropping would change p and make two references built
  # from the same data incomparable.
  avg <- with_local_seed(seed, {
    if (n <= 1500L) {
      mean(stats::dist(xw))
    } else {
      # Sampling PAIRS, not points: a sample of points estimates the mean
      # pairwise distance of the sample, which is the same quantity only if
      # the sample is representative -- sampling pairs needs no such argument.
      m <- min(as.integer(max_pairs), n * (n - 1L) / 2L)
      i <- sample.int(n, m, replace = TRUE)
      j <- sample.int(n, m, replace = TRUE)
      keep <- i != j
      mean(sqrt(rowSums((xw[i[keep], , drop = FALSE] -
                         xw[j[keep], , drop = FALSE])^2)))
    }
  })

  if (!is.finite(avg) || avg <= 0) {
    stop("The mean pairwise distance in the training data is ", avg,
         " -- every training row is identical in predictor space, so no ",
         "dissimilarity can be expressed relative to it.", call. = FALSE)
  }

  structure(list(x = xw, weights = weights, avg_dist = avg,
                 n = n, p = p),
            class = "di_reference")
}

#' Nearest-neighbour distance from each row of x to the training data.
#'
#' Split out because both the DI of new data and the threshold need it, and a
#' second implementation of the same search is a second chance to compute a
#' different thing.
#'
#' @param ref     From di_reference().
#' @param x       Numeric matrix, same columns and scaling as the training data.
#' @param exclude Optional integer vector, one per row of x, giving a training
#'   FOLD to exclude from that row's search. Used for the threshold.
#' @param folds_train Fold label of each training row; required with `exclude`.
#' @param chunk   Rows of x handled at a time. Distance is computed exactly,
#'   which is n_train * p per row; the chunk bounds the temporary matrix.
.di_nn_dist <- function(ref, x, exclude = NULL, folds_train = NULL,
                        chunk = 2000L) {
  x <- as.matrix(x)
  if (ncol(x) != ref$p) {
    stop("x has ", ncol(x), " columns; the reference was built from ", ref$p,
         ".", call. = FALSE)
  }
  xw  <- sweep(x, 2L, sqrt(ref$weights), "*")
  out <- numeric(nrow(xw))

  use_fnn <- is.null(exclude) && requireNamespace("FNN", quietly = TRUE)

  for (s in seq(1L, nrow(xw), by = chunk)) {
    e   <- min(s + chunk - 1L, nrow(xw))
    blk <- xw[s:e, , drop = FALSE]

    if (use_fnn) {
      # A kd-tree is not asymptotically better in this many dimensions, but
      # FNN's implementation is compiled and still beats the R loop below.
      out[s:e] <- as.numeric(FNN::get.knnx(ref$x, blk, k = 1L)$nn.dist[, 1])
    } else {
      d2 <- outer(rowSums(blk^2), rowSums(ref$x^2), "+") -
            2 * tcrossprod(blk, ref$x)
      if (!is.null(exclude)) {
        # Mask out the training rows in the same fold, so a point is never its
        # own nearest neighbour and never borrows one it trained beside.
        for (r in seq_len(nrow(blk))) {
          d2[r, folds_train == exclude[s + r - 1L]] <- Inf
        }
      }
      # Numerical floor: the expansion can return a tiny negative for a point
      # compared with itself, and sqrt() of that is NaN.
      #
      # rowMins where available: apply(d2, 1, min) is an R-level loop over a
      # chunk x n_train matrix, and over a raster that is millions of rows.
      # Same answer, compiled.
      mins <- if (requireNamespace("matrixStats", quietly = TRUE)) {
        matrixStats::rowMins(d2)
      } else {
        apply(d2, 1L, min)
      }
      out[s:e] <- sqrt(pmax(mins, 0))
    }
  }
  out
}

#' The dissimilarity index of new data.
#'
#' @param ref From di_reference().
#' @param x   Numeric matrix of new rows, scaled with the MODEL's scaling.
#' @return Numeric vector, one DI per row. 0 means identical to a training
#'   point; 1 means as far from the training data as two training points are
#'   from each other on average.
dissimilarity_index <- function(ref, x, chunk = 2000L) {
  stopifnot(inherits(ref, "di_reference"))
  .di_nn_dist(ref, x, chunk = chunk) / ref$avg_dist
}

#' The DI threshold that separates the area of applicability.
#'
#' Derived from the CROSS-VALIDATED training data: for each training row, the
#' distance to the nearest training row in a DIFFERENT fold. That set is the
#' dissimilarity the model actually coped with during cross-validation, so the
#' outlier-removed maximum of it is the largest dissimilarity for which the
#' reported error has been demonstrated.
#'
#' @param ref   From di_reference().
#' @param folds Integer fold label per training row, in the same order as the
#'   matrix the reference was built from.
#' @param k_iqr Outlier rule: threshold = Q75 + k_iqr * IQR. 1.5 is Tukey's
#'   fence and the value Meyer & Pebesma use.
#' @return The threshold, with the cross-validated DI attached as "cv_di".
aoa_threshold <- function(ref, folds, k_iqr = 1.5) {
  stopifnot(inherits(ref, "di_reference"))
  folds <- as.integer(folds)
  if (length(folds) != ref$n) {
    stop("folds has ", length(folds), " entries for ", ref$n,
         " training rows.", call. = FALSE)
  }
  if (length(unique(folds)) < 2L) {
    stop("The threshold is derived from distances ACROSS folds; with one ",
         "fold there are none. Pass the fold labels of a k-fold plan.",
         call. = FALSE)
  }

  d  <- .di_nn_dist(ref, ref$x / rep(sqrt(ref$weights), each = ref$n),
                    exclude = folds, folds_train = folds)
  cv <- d / ref$avg_dist

  q  <- stats::quantile(cv, probs = c(0.25, 0.75), na.rm = TRUE)
  th <- as.numeric(q[2] + k_iqr * (q[2] - q[1]))
  attr(th, "cv_di") <- cv
  attr(th, "k_iqr") <- k_iqr
  th
}

#' Is each row inside the area of applicability?
#'
#' @param di        From dissimilarity_index().
#' @param threshold From aoa_threshold().
#' @return Logical vector. TRUE means the cross-validated error applies here.
inside_aoa <- function(di, threshold) as.numeric(di) <= as.numeric(threshold)

#' Report what an AOA covers.
print_aoa <- function(di, threshold, label = "prediction area") {
  inside <- inside_aoa(di, threshold)
  cat("\n-- Area of applicability --\n")
  kq <- attr(threshold, "k_iqr")
  if (is.null(kq)) kq <- 1.5
  cat(sprintf("  DI threshold          : %.4f  (Q75 + %g x IQR of the ",
              as.numeric(threshold), kq),
      "cross-validated training DI)\n", sep = "")
  cat(sprintf("  %-21s : %s of %s cells inside (%.1f%%)\n", label,
              format(sum(inside, na.rm = TRUE), big.mark = ","),
              format(length(inside), big.mark = ","),
              100 * mean(inside, na.rm = TRUE)))
  cat(sprintf("  DI quantiles          : min %.2f | median %.2f | ",
              min(di, na.rm = TRUE), stats::median(di, na.rm = TRUE)))
  cat(sprintf("q95 %.2f | max %.2f\n",
              stats::quantile(di, 0.95, na.rm = TRUE), max(di, na.rm = TRUE)))
  if (mean(inside, na.rm = TRUE) < 0.5) {
    cat("\n  -> LESS THAN HALF THE MAP IS INSIDE. The cross-validated error\n",
        "     does not describe the rest: those pixels hold predictor\n",
        "     combinations the model never met. Report the map WITH this\n",
        "     mask, not the headline metric alone.\n", sep = "")
  }
  invisible(inside)
}
