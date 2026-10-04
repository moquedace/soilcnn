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
#      cross-validation, rather than picking a number. A point no fold held
#      out -- the training side of a holdout -- was never predicted, so it has
#      no such DI: it is a neighbour, not a measurement.
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
#' @examples
#' set.seed(1)
#' x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
#' ref <- di_reference(x_train)
#' ref
#' @export
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
#'   NA marks a row never held out: no row's search excludes it.
#' @param chunk   Rows of x handled at a time. Distance is computed exactly,
#'   which is n_train * p per row; the chunk bounds the temporary matrix.
#' @noRd
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
        # which(): a row never held out has fold NA, and is masked for no one.
        for (r in seq_len(nrow(blk))) {
          d2[r, which(folds_train == exclude[s + r - 1L])] <- Inf
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
#' @param chunk Rows of `x` compared at a time. It bounds the memory a call
#'   takes and does not change the result.
#' @examples
#' set.seed(1)
#' x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
#' ref <- di_reference(x_train)
#' dissimilarity_index(ref, rbind(c(0, 0), c(4, 4)))   # near the data, and far from it
#' @export
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
#' A row never held out -- the training side of a single split -- has fold
#' `NA`: it is a neighbour to every fold, and has no cross-validated DI of its
#' own, since no model predicted it.
#'
#' @param ref   From di_reference().
#' @param folds Integer fold label per training row, in the same order as the
#'   matrix the reference was built from; `NA` for a row never held out.
#' @param k_iqr Outlier rule: threshold = Q75 + k_iqr * IQR. 1.5 is Tukey's
#'   fence and the value Meyer & Pebesma use.
#' @return The threshold, with the cross-validated DI attached as "cv_di"
#'   (`NA` for the rows never held out).
#' @examples
#' set.seed(1)
#' x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
#' ref <- di_reference(x_train)
#' th <- aoa_threshold(ref, folds = rep(1:5, 20))
#' as.numeric(th)                 # the threshold
#' summary(attr(th, "cv_di"))     # the cross-validated DI it was taken from
#' @export
aoa_threshold <- function(ref, folds, k_iqr = 1.5) {
  stopifnot(inherits(ref, "di_reference"))
  folds <- as.integer(folds)
  if (length(folds) != ref$n) {
    stop("folds has ", length(folds), " entries for ", ref$n,
         " training rows.", call. = FALSE)
  }
  held <- which(!is.na(folds))
  if (length(held) == 0L) {
    stop("No training row was held out (every fold label is NA): there is no ",
         "cross-validated distance to derive the threshold from.", call. = FALSE)
  }
  # A held-out row's neighbour must lie outside its fold: in another fold, or
  # among the rows never held out. One fold and nothing beside it has none.
  if (length(unique(folds[held])) < 2L && length(held) == ref$n) {
    stop("The threshold is derived from distances ACROSS folds; with one ",
         "fold there are none. Pass the fold labels of a k-fold plan, or NA ",
         "for the rows a single split never held out.", call. = FALSE)
  }

  # The rows go back unweighted, because .di_nn_dist() weights what it is
  # given. A channel of weight ZERO cannot be divided back -- 0 / 0 -- and that
  # NaN made every cross-validated DI NaN: the first map given an importance
  # with a zero in it (tests/test_predict.R, section 8) had no calibration
  # point left. Its value is multiplied by zero again, so any finite one will
  # do; dividing by 1 leaves the zero it holds.
  sw <- sqrt(ref$weights)
  x  <- ref$x / rep(ifelse(sw > 0, sw, 1), each = ref$n)
  cv <- rep(NA_real_, ref$n)
  cv[held] <- .di_nn_dist(ref, x[held, , drop = FALSE], exclude = folds[held],
                          folds_train = folds) / ref$avg_dist

  # A THRESHOLD THAT IS NOT A NUMBER STOPS HERE. The zero-weight NaN above went
  # through quantile(na.rm = TRUE) as an empty set and came out NA: an AOA that
  # admits nothing, built without an error.
  if (!any(is.finite(cv[held]))) {
    stop("No held-out row has a finite cross-validated DI, so the AOA threshold ",
         "cannot be derived.", call. = FALSE)
  }
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
#' @examples
#' set.seed(1)
#' x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
#' ref <- di_reference(x_train)
#' threshold <- aoa_threshold(ref, folds = rep(1:5, 20))
#' inside_aoa(dissimilarity_index(ref, rbind(c(0, 0), c(4, 4))), threshold)
#' @export
inside_aoa <- function(di, threshold) as.numeric(di) <= as.numeric(threshold)

#' Report what an AOA covers.
#'
#' Prints the threshold, the share of cells inside it and the quantiles of
#' the DI, and says so plainly when less than half the map is inside.
#'
#' @param di        Dissimilarity index of the cells, from dissimilarity_index().
#' @param threshold From aoa_threshold().
#' @param label     What the cells are, for the report.
#' @return The logical mask of the cells inside (inside_aoa()), invisibly.
#' @examples
#' set.seed(1)
#' x_train <- matrix(rnorm(200), 100, 2)    # 100 training rows of 2 scaled predictors
#' ref <- di_reference(x_train)
#' threshold <- aoa_threshold(ref, folds = rep(1:5, 20))
#' print_aoa(dissimilarity_index(ref, matrix(rnorm(400, sd = 1.5), 200, 2)), threshold)
#' @export
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

#' Print a `di_reference`
#'
#' @param x   A `di_reference`, from [di_reference()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.di_reference <- function(x, ...) {
  cat("<di_reference> ", x$n, " training row(s) x ", x$p, " predictor(s)",
      if (length(unique(x$weights)) == 1L) " | unweighted" else " | importance-weighted",
      "\n  mean pairwise distance (the DI's unit): ",
      format(x$avg_dist, digits = 4), "\n", sep = "")
  invisible(x)
}

# ── A fitted model's reference, built once for the map, the AOA and the interval ─
#
# Stage 07 built this inline. The level-and-dissimilarity interval needs the
# SAME construction twice -- the calibration points' DI and every map pixel's
# -- and two copies of "what the model saw" would drift the first time either
# changed, which is the one way this interval can lose its guarantee without a
# single error: calibrated on one DI, applied with another.

#' The dissimilarity reference of a fitted model.
#'
#' The centre pixel of every point that trained or validated in the tuning plan,
#' QC'd and scaled with the MODEL's scaling (stage 07's construction). Each
#' point's fold is the one it was held out in, so its cross-validated DI is the
#' distance to the nearest point OUTSIDE its fold: the dissimilarity the model
#' that predicted it during cross-validation actually faced. That is the DI a
#' calibration residual comes with. A point that only ever trained -- the
#' training side of a holdout -- has fold `NA` and no cross-validated DI: it is
#' in the reference, a neighbour to every fold, and out of the threshold.
#'
#' @param points     Point table aligned to the store (align_points_to_meta()).
#' @param predictors Channel names, in the model's order.
#' @param qc_table   QC rules, in the same order.
#' @param scaling    The fitted model's predictor_scaling, in the same order.
#' @param plan       The tuning run's fold plan.
#' @param weights    NULL (every channel alike), or one non-negative weight per
#'   channel, in the same order -- an importance, from importance_weights(),
#'   so a point is unlike the training data in what the model uses (Meyer &
#'   Pebesma 2021).
#' @return An `aoa_reference`: the DI reference, the AOA threshold, and each
#'   used point's fold and cross-validated DI (`NA` for a point never held out).
#' @examples
#' ex <- example_landscape()
#' store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
#'                      windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
#'                      percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
#'                      overwrite = TRUE, verbose = FALSE)
#' data <- dsm_load(store, windows = integer(0), verbose = FALSE)
#' predictors <- as.character(data$store$predictors)
#' qc <- utils::read.csv2(file.path(data$patch_dir, "qc_table.csv"))
#' qc <- qc[match(predictors, qc$predictor), ]
#' # a fitted model's scaling is its predictor_scaling.csv; here, the profiles' own
#' x <- as.matrix(data$points[, predictors])
#' scaling <- data.frame(predictor = predictors, center = colMeans(x),
#'                       scale = apply(x, 2, sd))
#' plan <- spatial_folds(data$store$meta, k = 3, block_size = 0.05)
#' aref <- aoa_reference(data$points, predictors, qc, scaling, plan)
#' as.numeric(aref$threshold)     # beyond it, the cross-validated error does not apply
#' head(aref$cv)
#' @export
aoa_reference <- function(points, predictors, qc_table, scaling, plan, weights = NULL) {
  stopifnot(inherits(plan, "fold_plan"))
  if (!identical(as.character(scaling$predictor), as.character(predictors)) ||
      !identical(as.character(qc_table$predictor), as.character(predictors))) {
    stop("scaling, qc_table and predictors must list the same channels in the ",
         "same order: the DI is a distance in a space whose axes are the ",
         "channels, and a permuted axis gives a finite, wrong answer.", call. = FALSE)
  }
  train_mat <- as.matrix(points[, predictors, drop = FALSE])
  storage.mode(train_mat) <- "double"
  for (i in seq_along(predictors)) {
    train_mat[, i] <- qc_band_values(train_mat[, i], qc_table[i, ])
  }
  train_mat <- scale_patches_matrix(train_mat, scaling)

  # The fold a point was HELD OUT in: exactly the "not in its own fold" the
  # threshold is defined against.
  #
  # A POINT THAT ONLY EVER TRAINED HAS NO FOLD (NA). No model predicted it
  # during cross-validation, so it has no dissimilarity one coped with; it
  # trained the models of the folds that held it, so it neighbours their
  # points -- taken here as every fold's, the approximation the buffer already
  # makes (a point a fold's buffer dropped still neighbours that fold). The
  # first version gave it the first fold it trained in, which hid it from the
  # validation points of that very fold. On a holdout that is fold 1, the
  # validation rows' own, for every point: nothing was left across folds to
  # measure, and dsm_predict() stopped on the SOC 0-30 cm trial's holdout
  # design (2026-09-29). A k-fold plan partitions its pool, every point
  # validates once, and nothing there changes.
  fold_of <- rep(NA_integer_, nrow(points))
  for (j in seq_along(plan$folds)) fold_of[plan$folds[[j]]$validation] <- j
  used <- sort(unique(unlist(lapply(plan$folds, function(f) c(f$train, f$validation)))))
  used <- used[is.finite(rowSums(train_mat[used, , drop = FALSE]))]
  if (length(used) < 2L) {
    stop("Fewer than 2 usable training rows after QC: nothing to measure a ",
         "dissimilarity against.", call. = FALSE)
  }
  ref <- di_reference(train_mat[used, , drop = FALSE], weights = weights)
  th  <- aoa_threshold(ref, fold_of[used])
  structure(list(
    ref = ref, threshold = th,
    cv = tibble::tibble(sample_id = points$sample_id[used], fold = fold_of[used],
                        cv_di = as.numeric(attr(th, "cv_di"))),
    predictors = as.character(predictors), qc_table = qc_table, scaling = scaling),
    class = "aoa_reference")
}

#' The DI of new rows of RAW predictor values, QC'd and scaled as the reference was.
#'
#' @param aref   From aoa_reference().
#' @param values Matrix or data frame of raw values, columns in the model's order.
#' @return The DI of each row; NA where a channel is missing after QC.
#' @examples
#' ex <- example_landscape()
#' store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
#'                      windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
#'                      percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
#'                      overwrite = TRUE, verbose = FALSE)
#' data <- dsm_load(store, windows = integer(0), verbose = FALSE)
#' predictors <- as.character(data$store$predictors)
#' qc <- utils::read.csv2(file.path(data$patch_dir, "qc_table.csv"))
#' qc <- qc[match(predictors, qc$predictor), ]
#' # a fitted model's scaling is its predictor_scaling.csv; here, the profiles' own
#' x <- as.matrix(data$points[, predictors])
#' scaling <- data.frame(predictor = predictors, center = colMeans(x),
#'                       scale = apply(x, 2, sd))
#' plan <- spatial_folds(data$store$meta, k = 3, block_size = 0.05)
#' aref <- aoa_reference(data$points, predictors, qc, scaling, plan)
#' aoa_di(aref, data$points[1:5, predictors])
#' @export
aoa_di <- function(aref, values) {
  stopifnot(inherits(aref, "aoa_reference"))
  m <- as.matrix(values)
  if (ncol(m) != length(aref$predictors)) {
    stop("values has ", ncol(m), " column(s); the reference has ",
         length(aref$predictors), " channels.", call. = FALSE)
  }
  storage.mode(m) <- "double"
  for (i in seq_along(aref$predictors)) {
    m[, i] <- qc_band_values(m[, i], aref$qc_table[i, ])
  }
  m <- scale_patches_matrix(m, aref$scaling)
  keep <- is.finite(rowSums(m))
  di <- rep(NA_real_, nrow(m))
  if (any(keep)) di[keep] <- dissimilarity_index(aref$ref, m[keep, , drop = FALSE])
  di
}
