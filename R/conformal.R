# ── Calibrated uncertainty: split conformal prediction, and PICP ──────────────
#
# WHAT WAS WRONG WITH THE UNCERTAINTY MAP.
#
# Stage 04 trains one config under several seeds and reports the median and the
# spread of that ensemble. design_decisions.md §11 already recorded, honestly,
# that the spread is NOT a prediction interval: it measures how much the answer
# moves when the initialisation moves, which is a property of the optimiser, not
# of the soil. It says nothing about the irreducible noise, the model's bias, or
# anything the ensemble agrees on while being wrong together.
#
# The size of the mistake is visible in this project's own numbers: the CNN's
# spread between seeds is 0.038 CCC, while its MAE is around 17 t/ha against a
# median stock of 29.3. An uncertainty map drawn from the seed spread would
# promise a few t/ha where the error is an order of magnitude larger. A map that
# understates its uncertainty is worse than no map, because somebody acts on it.
#
# WHAT SPLIT CONFORMAL DOES INSTEAD.
#
#   1. hold out a calibration set the model never trained on
#   2. compute the absolute residuals there
#   3. take their (1 - alpha) quantile, q
#   4. every new prediction gets  pred +/- q
#
# and P(y in the interval) >= 1 - alpha. No normality, no assumption that the
# model is good, no asymptotics. The only requirement is EXCHANGEABILITY:
# calibration points and new points drawn from the same distribution. That is
# also exactly where it stops being true, which is why this file says so out
# loud rather than printing a number and moving on -- see the AOA (R/aoa.R):
# outside the area of applicability the exchangeability assumption fails, and
# the guarantee goes with it.
#
# THE FINITE-SAMPLE CORRECTION IS NOT A DETAIL. The quantile taken is the
# ceil((n+1)(1-alpha))-th smallest residual, not the plain sample quantile. With
# n = 100 and alpha = 0.1 that is the 91st value rather than the 90th. Using the
# ordinary quantile makes the coverage guarantee false by roughly 1/n, and the
# error is in the optimistic direction -- the one nobody checks.
#
# NORMALISED (LOCALLY ADAPTIVE) INTERVALS. A constant +/- q is honest but
# blunt: it is as wide over terrain the model knows well as over terrain it has
# never seen. Dividing each residual by a per-point DIFFICULTY score before
# taking the quantile gives intervals that scale with that score while keeping
# the coverage guarantee intact. The ensemble spread, useless as an interval, is
# a perfectly good difficulty score -- so the seed-to-seed variation finally
# earns its place, calibrated instead of trusted.
#
# AND THEN MEASURE IT. PICP -- the share of observations inside the nominal
# interval -- turns "uncertainty" from an adjective into a number that can be
# wrong. Promise 90%, deliver 61%, and the map is broken in a way that is
# visible. Most DSM papers publish an uncertainty map and never check it.

#' Calibrate a conformal interval from held-out residuals.
#'
#' @param obs,pred Observations and predictions on the CALIBRATION set -- data
#'   the model did not train on. Using training residuals produces intervals
#'   that are too narrow by exactly the amount the model overfits, and the
#'   result looks like a much better map.
#' @param alpha    Miscoverage rate: 0.1 asks for 90% coverage.
#' @param difficulty Optional per-point difficulty score (e.g. the ensemble
#'   spread). When given, intervals scale with it. Must be positive.
#' @param group    Optional group of each point -- a spatial block, a region,
#'   a profile. Given, every group weighs the same in the quantile, whatever
#'   its number of points: q is the smallest score at which the mean of the
#'   groups' empirical distributions reaches (1 - alpha)(1 + 1/m), m groups
#'   (Dunn, Wasserman & Ramdas 2023). The interval then speaks for a new point
#'   of a new group rather than for a point drawn from the pooled sample.
#' @return An object of class "conformal_cal".
#' @examples
#' set.seed(1)
#' obs  <- rlnorm(300, 3, 0.4)
#' pred <- obs * exp(rnorm(300, 0, 0.25))
#' cal  <- conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1)
#' cal
#' # the same residuals, every one of 30 groups weighing alike
#' conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1, group = rep(1:30, 5))
#' @export
conformal_calibrate <- function(obs, pred, alpha = 0.1, difficulty = NULL, group = NULL) {
  stopifnot(length(obs) == length(pred))
  if (!is.numeric(alpha) || length(alpha) != 1L || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a single number in (0, 1).", call. = FALSE)
  }
  keep <- is.finite(obs) & is.finite(pred)
  if (!is.null(difficulty)) {
    stopifnot(length(difficulty) == length(obs))
    keep <- keep & is.finite(difficulty) & difficulty > 0
  }
  if (!is.null(group)) {
    if (length(group) != length(obs)) {
      stop("group has ", length(group), " label(s) for ", length(obs), " point(s).",
           call. = FALSE)
    }
    keep <- keep & !is.na(group)
  }
  o <- as.numeric(obs)[keep]; p <- as.numeric(pred)[keep]
  n <- length(o)
  if (n < 2L) {
    stop("Only ", n, " usable calibration point(s). A conformal interval is a ",
         "quantile of residuals; there is nothing to take a quantile of.",
         call. = FALSE)
  }

  res <- abs(o - p)
  d   <- if (is.null(difficulty)) rep(1, n) else as.numeric(difficulty)[keep]
  score <- res / d
  g <- if (is.null(group)) NULL else as.character(group)[keep]

  # THE (n+1) CORRECTION. The guarantee is over the calibration set PLUS the new
  # point, so the rank is taken out of n+1, not n. k > n means this calibration
  # set is too small to certify this alpha at all: the threshold is
  # ceiling(1/alpha) - 1, so 90% needs 9 points and 95% needs 19. Below it the
  # honest answer is an infinite interval, not a finite one that quietly fails
  # to cover. By group, the same count applies to the groups.
  qk <- .conformal_quantile(score, alpha, g)
  if (!is.finite(qk$q)) {
    unit <- if (is.null(g)) "points" else "groups"
    warning("alpha = ", alpha, " needs at least ", ceiling(1 / alpha) - 1,
            " calibration ", unit, " to be certifiable; there are ", qk$m,
            ". The interval is infinite, which is the correct answer.",
            call. = FALSE)
  }

  structure(list(q = qk$q, alpha = alpha, n = n, k = qk$k,
                 normalised = !is.null(difficulty),
                 weighting = if (is.null(g)) "point" else "group",
                 n_groups = qk$m,
                 residuals = res, scores = score),
            class = "conformal_cal")
}

# ── WEIGHTING BY GROUP ────────────────────────────────────────────────────────
#
# THE POOLED QUANTILE IS A CHOICE OF TARGET. Profiles come in clusters: a few
# dense surveys and many places with one pit. Pooled point by point, the
# quantile is the dense surveys' -- the interval covers 90% of PROFILES, which
# is mostly a statement about the places that were sampled most. A map makes
# its promise about places.
#
# Dunn, Wasserman & Ramdas (2023, JASA 118:2491-2502) treat the groups as the
# exchangeable units of a two-layer model. Their pooled-CDF method gives every
# group the same weight -- each point 1/(points in its group) -- and is
# asymptotically valid in the number of groups. With the (1 + 1/m) factor
# taken here, q is also the limit, over infinitely many draws, of their
# repeated subsampling (one point per group per draw; Theorem 10), whose
# coverage is at least 1 - 2 alpha for a new point of a NEW group in finite
# samples: the p-value of a residual r there averages to
# (1 + sum_j S_j(r)) / (m + 1), S_j the share of group j at or above r, and
# accepting r while that is >= alpha is q's definition below. tests/
# test_conformal.R holds the two to each other.
#
# With one point per group it is the ordinary rank, ceiling((n + 1)(1 - alpha)).
.conformal_quantile <- function(score, alpha, group = NULL) {
  n <- length(score)
  if (is.null(group)) {
    k <- ceiling((n + 1) * (1 - alpha))
    return(list(q = if (k > n) Inf else sort(score)[k], k = as.integer(k), m = n))
  }
  sizes  <- table(group)
  m      <- length(sizes)
  w      <- 1 / as.numeric(sizes[as.character(group)])
  target <- (1 - alpha) * (m + 1)
  # The weights are sums of 1/n_j: compared with a tolerance, or a target met
  # exactly (m = 1/alpha - 1) fails by a rounding.
  tol <- 1e-9 * (m + 1)
  if (target > m + tol) return(list(q = Inf, k = NA_integer_, m = m))
  o   <- order(score)
  cum <- cumsum(w[o])
  i   <- which(cum >= target - tol)[1]
  list(q = score[o][i], k = as.integer(i), m = m)
}

#' Turn predictions into intervals.
#'
#' @param cal        A conformal_cal.
#' @param pred       Predictions to wrap.
#' @param difficulty Difficulty scores for these predictions. Required when the
#'   calibration was normalised, and refused when it was not -- mixing the two
#'   silently produces intervals with no guarantee at all.
#' @param lower_limit Floor for the lower bound, e.g. 0 for a stock. Clipping a
#'   bound at a physical limit can only INCREASE coverage, so the guarantee
#'   survives it.
#' @return A tibble: pred, lower, upper and width, one row per prediction.
#' @examples
#' set.seed(1)
#' obs  <- rlnorm(300, 3, 0.4)
#' pred <- obs * exp(rnorm(300, 0, 0.25))
#' cal  <- conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1)
#' iv <- conformal_interval(cal, pred[151:300], lower_limit = 0)
#' head(iv)
#' @export
conformal_interval <- function(cal, pred, difficulty = NULL,
                               lower_limit = -Inf) {
  stopifnot(inherits(cal, "conformal_cal"))
  if (inherits(cal, "conformal_scaled")) {
    stop("This calibration has a FITTED scale: use conformal_scaled_interval() ",
         "with the covariates it was fitted on.", call. = FALSE)
  }
  if (cal$normalised && is.null(difficulty)) {
    stop("This calibration is normalised, so it needs a difficulty score for ",
         "each prediction -- the same kind used to calibrate it.", call. = FALSE)
  }
  if (!cal$normalised && !is.null(difficulty)) {
    stop("This calibration is not normalised; a difficulty score here would ",
         "widen the intervals with no guarantee attached to the widening.",
         call. = FALSE)
  }
  d <- if (cal$normalised) as.numeric(difficulty) else 1
  half <- cal$q * d
  tibble::tibble(
    pred  = as.numeric(pred),
    lower = pmax(as.numeric(pred) - half, lower_limit),
    upper = as.numeric(pred) + half,
    width = pmin(as.numeric(pred) + half, Inf) -
            pmax(as.numeric(pred) - half, lower_limit)
  )
}

#' Prediction Interval Coverage Probability.
#'
#' The share of observations that fall inside their interval. If a 90% interval
#' covers 61%, the uncertainty map is wrong, and this is the line that says so.
#'
#' @param obs          Observations.
#' @param lower,upper  Interval bounds.
#' @return A one-row tibble: n, picp, mean_width, median_width.
#' @noRd
picp <- function(obs, lower, upper) {
  stopifnot(length(obs) == length(lower), length(obs) == length(upper))
  keep <- is.finite(obs) & is.finite(lower)
  o <- as.numeric(obs)[keep]
  lo <- as.numeric(lower)[keep]; up <- as.numeric(upper)[keep]
  inside <- o >= lo & o <= up
  tibble::tibble(
    n = length(o),
    picp = mean(inside),
    mean_width = mean(up - lo),
    median_width = stats::median(up - lo)
  )
}

#' Coverage, and whether it holds where it is needed.
#'
#' @param obs,lower,upper As in picp().
#' @param group  Optional grouping (a spatial block, a region, a soil class).
#' @param alpha  The nominal miscoverage, for the verdict.
#' @return An object of class "picp_report".
#' @examples
#' set.seed(1)
#' obs  <- rlnorm(300, 3, 0.4)
#' pred <- obs * exp(rnorm(300, 0, 0.25))
#' cal  <- conformal_calibrate(obs[1:150], pred[1:150], alpha = 0.1)
#' iv <- conformal_interval(cal, pred[151:300], lower_limit = 0)
#' # one width for every level: right on average, and by level?
#' picp_report(obs[151:300], iv$lower, iv$upper, alpha = 0.1,
#'             group = ifelse(pred[151:300] > median(pred), "high", "low"))
#' @export
picp_report <- function(obs, lower, upper, group = NULL, alpha = 0.1) {
  overall <- picp(obs, lower, upper)
  by_group <- NULL
  if (!is.null(group)) {
    stopifnot(length(group) == length(obs))
    d <- tibble::tibble(obs = as.numeric(obs), lower = as.numeric(lower),
                        upper = as.numeric(upper), group = group)
    d <- d[is.finite(d$obs) & is.finite(d$lower), , drop = FALSE]
    by_group <- d %>%
      dplyr::group_by(.data$group) %>%
      dplyr::summarise(n = dplyr::n(),
                       picp = mean(.data$obs >= .data$lower &
                                   .data$obs <= .data$upper),
                       mean_width = mean(.data$upper - .data$lower),
                       .groups = "drop") %>%
      dplyr::arrange(.data$picp)
  }
  structure(list(overall = overall, by_group = by_group, alpha = alpha),
            class = "picp_report")
}

#' Print a `picp_report`
#'
#' @param x   A `picp_report`, from [picp_report()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.picp_report <- function(x, ...) {
  target <- 1 - x$alpha
  o <- x$overall
  cat("\nInterval coverage (PICP)\n")
  cat(strrep("-", 58), "\n")
  cat(sprintf("  nominal   : %.0f%%\n", 100 * target))
  cat(sprintf("  observed  : %.1f%%  (%d point(s))\n", 100 * o$picp, o$n))
  cat(sprintf("  width     : mean %.2f | median %.2f\n",
              o$mean_width, o$median_width))

  # A conformal interval calibrated and measured on the SAME points will hit its
  # nominal rate almost exactly -- that is arithmetic, not evidence. The number
  # is worth something only on points the calibration never saw, so the verdict
  # is phrased to keep that distinction alive.
  gap <- o$picp - target
  if (abs(gap) <= 0.02) {
    cat("\n  -> Coverage matches the promise. If these are points the\n")
    cat("     calibration did not see, the uncertainty is calibrated.\n")
  } else if (gap < 0) {
    cat(sprintf("\n  -> UNDER-COVERAGE by %.1f points. The map promises more\n",
                -100 * gap))
    cat("     certainty than it delivers, which is the dangerous direction.\n")
  } else {
    cat(sprintf("\n  -> Over-coverage by %.1f points: the intervals are wider\n",
                100 * gap))
    cat("     than they need to be. Safe, but less useful.\n")
  }

  if (!is.null(x$by_group)) {
    # THE GUARANTEE IS MARGINAL, NOT CONDITIONAL. "90% overall" is compatible
    # with 99% over the easy half and 60% over the hard half -- and the hard
    # half is the part anyone actually needs the interval for. Splitting by
    # group is the cheapest way to see it, so it is not optional here.
    cat("\n  By group (worst coverage first):\n")
    print(utils::head(x$by_group, 10), n = 10)
    worst <- x$by_group$picp[1]
    if (is.finite(worst) && worst < target - 0.1) {
      cat(sprintf("\n     Worst group covers %.1f%% against a nominal %.0f%%.\n",
                  100 * worst, 100 * target))
      cat("     Conformal guarantees coverage ON AVERAGE, not everywhere. A\n")
      cat("     group this far below is where the exchangeability assumption\n")
      cat("     is breaking -- compare with the area of applicability.\n")
    }
  }
  invisible(x)
}

#' Print a `conformal_cal`
#'
#' @param x   A `conformal_cal`, from [conformal_calibrate()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.conformal_cal <- function(x, ...) {
  cat("\n<conformal_cal> ", sprintf("%.0f%% intervals", 100 * (1 - x$alpha)),
      if (x$normalised) " (normalised)" else " (constant width)", "\n", sep = "")
  cat(sprintf("  calibration points : %d\n", x$n))
  # A calibration made before the weighting existed has no field: by point.
  if (identical(x$weighting, "group")) {
    cat(sprintf("  weighting          : by group -- %d group(s), each weighing the same\n",
                x$n_groups))
    cat(sprintf("  level              : the groups' mean distribution at %.4f  (the (1 + 1/m) correction)\n",
                (1 - x$alpha) * (1 + 1 / x$n_groups)))
  } else {
    cat(sprintf("  rank used          : %d of %d  (the (n+1) correction)\n",
                x$k, x$n))
  }
  if (is.finite(x$q)) {
    cat(sprintf("  q                  : %.4f%s\n", x$q,
                if (x$normalised) " x difficulty" else ""))
  } else {
    cat("  q                  : Inf -- too few points to certify this alpha\n")
  }
  invisible(x)
}

#' Calibrate on the folds of a resampling run, then report coverage.
#'
#' The natural calibration set in this framework is the VALIDATION side of each
#' fold: it is held out of training, it is separated spatially with a buffer,
#' and it already exists. This walks the per-unit predictions a run wrote,
#' calibrates on one fold and measures coverage on the others, so the reported
#' PICP is always out-of-calibration.
#'
#' @param pred_obs A table with columns fold, obs, pred (as the runners write).
#' @param alpha    Miscoverage rate.
#' @param difficulty Optional column name holding a difficulty score.
#' @param group    Optional column name to break coverage down by.
#' @return A picp_report over the pooled out-of-calibration points.
#' @examples
#' set.seed(1)
#' pred_obs <- data.frame(fold = rep(1:5, each = 60), obs = rlnorm(300, 3, 0.4))
#' pred_obs$pred <- pred_obs$obs * exp(rnorm(300, 0, 0.25))
#' conformal_cv(pred_obs, alpha = 0.1)
#' @export
conformal_cv <- function(pred_obs, alpha = 0.1, difficulty = NULL,
                         group = NULL) {
  need <- c("fold", "obs", "pred")
  gone <- setdiff(need, names(pred_obs))
  if (length(gone)) {
    stop("pred_obs needs column(s): ", paste(gone, collapse = ", "),
         call. = FALSE)
  }
  folds <- sort(unique(pred_obs$fold))
  if (length(folds) < 2L) {
    stop("Cross-calibration needs at least 2 folds -- with one, the ",
         "calibration set and the evaluation set are the same points, and the ",
         "coverage comes out right by arithmetic rather than by evidence.",
         call. = FALSE)
  }

  out <- lapply(folds, function(f) {
    cal_rows <- pred_obs[pred_obs$fold != f, , drop = FALSE]
    new_rows <- pred_obs[pred_obs$fold == f, , drop = FALSE]
    cal <- conformal_calibrate(
      cal_rows$obs, cal_rows$pred, alpha = alpha,
      difficulty = if (!is.null(difficulty)) cal_rows[[difficulty]] else NULL)
    iv <- conformal_interval(
      cal, new_rows$pred,
      difficulty = if (!is.null(difficulty)) new_rows[[difficulty]] else NULL)
    tibble::tibble(fold = f, obs = new_rows$obs,
                   lower = iv$lower, upper = iv$upper,
                   group = if (!is.null(group)) new_rows[[group]] else NA)
  })
  all_rows <- dplyr::bind_rows(out)
  picp_report(all_rows$obs, all_rows$lower, all_rows$upper,
              group = if (!is.null(group)) all_rows$group else NULL,
              alpha = alpha)
}

# ── A width that follows the level and the dissimilarity ──────────────────────
#
# WHY THE CONSTANT INTERVAL IS NOT ENOUGH HERE.
#
# Measured on this project's test set (2026-09-26): the constant 90% interval,
# 79 t/ha wide everywhere, covered 94% of the lowest fifth of predictions and
# 77% of the highest. The error of a right-skewed stock grows with the stock,
# so one width is too wide where the soil holds little carbon and too narrow
# where it holds much -- right on average, wrong everywhere in particular.
#
# THE REMEDY, AND WHY IT KEEPS THE GUARANTEE.
#
# Regress |residual| on covariates that should predict how wrong the model is
# -- the predicted LEVEL, and the dissimilarity index (R/aoa.R), which says how
# far a point is from anything the model trained on -- and let the interval be
# q x that fitted scale. The fit is done on ONE half of the calibration points
# and q is taken on the OTHER. That separation is the whole point: the
# locally-weighted score |r| / scale is then exchangeable across the
# calibration half and a new point, so split conformal's finite-sample
# coverage holds exactly (Papadopoulos, Gammerman & Vovk 2008; Lei, G'Sell,
# Rinaldo, Tibshirani & Wasserman 2018, section 5.2). Fitted and calibrated on
# the same points, the fit would absorb the residuals the quantile measures,
# and the interval would come out narrow in exactly the way nobody checks.
#
# The earlier measurement (level only, fitted on half the calibration set):
# |r| ~ 8.59 + 0.258 x level gave 89/92/88/83/88% by fifth of the prediction,
# against the constant interval's 94/94/92/82/77. The dissimilarity index is
# the second covariate because it is the one thing a map pixel and a
# calibration point can be measured on the same way -- the seed spread is not
# (see stage 04's note: the calibration seeds are not the map's seeds).
#
# WHAT IT STILL DOES NOT FIX. Coverage is guaranteed over points exchangeable
# with the calibration set. Outside the area of applicability they are not,
# and no width is honest there -- that is what the AOA mask is for.

#' Fit the scale of an interval: |residual| on covariates that say how wrong.
#'
#' The scale of [conformal_scaled_calibrate()], fitted on its own: a + b1 x1 +
#' b2 x2 + ..., by least squares on |obs - pred|, with a floor. Fitted on
#' points the quantile is NOT taken on -- a calibration set's quantile, scaled
#' by a fit to the cross-validated residuals, keeps the whole calibration set
#' for q; fitted and calibrated on the same points, the fit would absorb the
#' residuals the quantile measures.
#'
#' @param obs,pred   Observations and predictions, NATIVE units.
#' @param covariates A data frame of scale covariates, one row per point, e.g.
#'   data.frame(level = pred, di = di).
#' @param floor_frac The scale is never below this share of the median
#'   |residual|. A linear fit can go to zero or below at the edge of the
#'   covariates' range, and a zero-width interval there would claim a
#'   certainty the data never gave.
#' @return A `conformal_scale`: the coefficients, the floor, and how well the
#'   fit explained |residual| (r2_fit).
#' @examples
#' set.seed(1)
#' level <- runif(300, 1, 4)
#' di    <- runif(300)                      # a dissimilarity index
#' obs   <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di))
#' pred  <- exp(level)
#' conformal_scale_fit(obs, pred, data.frame(level = pred, di = di))
#' @export
conformal_scale_fit <- function(obs, pred, covariates, floor_frac = 0.05) {
  covariates <- as.data.frame(covariates)
  if (length(obs) != length(pred) || nrow(covariates) != length(obs)) {
    stop("obs, pred and covariates must describe the same points: ",
         length(obs), ", ", length(pred), " and ", nrow(covariates), " rows.",
         call. = FALSE)
  }
  if (is.null(names(covariates)) || any(!nzchar(names(covariates)))) {
    stop("covariates needs named columns -- the names are how the map's ",
         "covariates are matched to the fitted coefficients.", call. = FALSE)
  }
  x_ok <- Reduce(`&`, lapply(covariates, function(v) is.finite(as.numeric(v))))
  keep <- is.finite(obs) & is.finite(pred) & x_ok
  o <- as.numeric(obs)[keep]
  p <- as.numeric(pred)[keep]
  X <- covariates[keep, , drop = FALSE]
  n <- length(o)
  if (n < ncol(X) + 3L) {
    stop("Only ", n, " usable point(s) to fit a scale on ", ncol(X), " covariate(s).",
         call. = FALSE)
  }
  res <- abs(o - p)
  fit <- stats::lm(abs_res ~ ., data = data.frame(abs_res = res, X))
  coef <- stats::coef(fit)
  coef[!is.finite(coef)] <- 0          # a covariate constant on these points
  floor <- floor_frac * stats::median(res)
  if (!is.finite(floor) || floor <= 0) floor <- .Machine$double.eps
  structure(list(coef = coef, floor = floor, terms = names(X), n_fit = n,
                 r2_fit = summary(fit)$r.squared, floor_frac = floor_frac),
            class = "conformal_scale")
}

#' Print a `conformal_scale`
#'
#' @param x   A `conformal_scale`, from [conformal_scale_fit()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.conformal_scale <- function(x, ...) {
  terms <- names(x$coef)[-1L]
  cat(sprintf("\n<conformal_scale> |residual| ~ %.4f%s\n", x$coef[1L],
              paste(sprintf(" %+.4f x %s", x$coef[terms], terms), collapse = "")))
  cat(sprintf("  fitted on %d point(s), R2 %.3f | floor %.4f (%.2f x the median |residual|)\n",
              x$n_fit, x$r2_fit, x$floor, x$floor_frac))
  invisible(x)
}

#' Calibrate an interval whose width is a fitted scale.
#'
#' @param obs,pred   Calibration observations and predictions, NATIVE units.
#' @param covariates A data frame of scale covariates, one row per point, e.g.
#'   data.frame(level = pred, di = di). The scale is a + b1 x1 + b2 x2 + ...,
#'   fitted by least squares on |obs - pred|.
#' @param alpha      Miscoverage rate: 0.1 asks for 90%.
#' @param fit_frac   Share of the points the scale is fitted on; the rest
#'   calibrate q. One half each is the textbook split. Not used when `scale`
#'   is given.
#' @param floor_frac The scale is never below this share of the median
#'   |residual| of the fitting half. A linear fit can go to zero or below at
#'   the edge of the covariates' range, and a zero-width interval there would
#'   claim a certainty the data never gave.
#' @param seed       Seed of the split.
#' @param scale      NULL: the scale is fitted on `fit_frac` of these points and
#'   q taken on the rest. A `conformal_scale` from [conformal_scale_fit()],
#'   fitted on OTHER points: every point here then calibrates q.
#' @param group      Optional group of each point, as in
#'   [conformal_calibrate()]: q weighs every group alike, and the split, when
#'   there is one, keeps a group on one side.
#' @return A `conformal_scaled` (also a `conformal_cal`).
#' @examples
#' set.seed(1)
#' level <- runif(300, 1, 4)
#' di    <- runif(300)                      # a dissimilarity index
#' obs   <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di))
#' pred  <- exp(level)
#' covariates <- data.frame(level = pred, di = di)
#' cal <- conformal_scaled_calibrate(obs[1:200], pred[1:200], covariates[1:200, ],
#'                                   alpha = 0.1)
#' cal
#' # the scale fitted on other points, and all 100 of these calibrating q
#' sc <- conformal_scale_fit(obs[1:200], pred[1:200], covariates[1:200, ])
#' conformal_scaled_calibrate(obs[201:300], pred[201:300], covariates[201:300, ],
#'                            scale = sc)
#' @export
conformal_scaled_calibrate <- function(obs, pred, covariates, alpha = 0.1,
                                       fit_frac = 0.5, floor_frac = 0.05,
                                       seed = 42L, scale = NULL, group = NULL) {
  covariates <- as.data.frame(covariates)
  if (length(obs) != length(pred) || nrow(covariates) != length(obs)) {
    stop("obs, pred and covariates must describe the same points: ",
         length(obs), ", ", length(pred), " and ", nrow(covariates), " rows.",
         call. = FALSE)
  }
  if (is.null(names(covariates)) || any(!nzchar(names(covariates)))) {
    stop("covariates needs named columns -- the names are how the map's ",
         "covariates are matched to the fitted coefficients.", call. = FALSE)
  }
  if (!is.null(scale) && !inherits(scale, "conformal_scale")) {
    stop("scale must be a conformal_scale, from conformal_scale_fit().", call. = FALSE)
  }
  if (is.null(scale) && (!is.numeric(fit_frac) || fit_frac <= 0 || fit_frac >= 1)) {
    stop("fit_frac must be in (0, 1).", call. = FALSE)
  }
  if (!is.null(group) && length(group) != length(obs)) {
    stop("group has ", length(group), " label(s) for ", length(obs), " point(s).",
         call. = FALSE)
  }
  x_ok <- Reduce(`&`, lapply(covariates, function(v) is.finite(as.numeric(v))))
  keep <- is.finite(obs) & is.finite(pred) & x_ok
  if (!is.null(group)) keep <- keep & !is.na(group)
  o <- as.numeric(obs)[keep]
  p <- as.numeric(pred)[keep]
  X <- covariates[keep, , drop = FALSE]
  g <- if (is.null(group)) NULL else as.character(group)[keep]
  n <- length(o)

  if (is.null(scale)) {
    if (n < 20L) {
      stop("Only ", n, " usable calibration point(s): the scale needs some to be ",
           "fitted on and the quantile needs others to be taken on.", call. = FALSE)
    }
    # By point, the draw it always was; by group, whole groups, so no group
    # is fitted on and calibrated on at once. Groups of one point each ARE
    # points, and take the points' draw: weighted by group they must give what
    # they give by point, not the same method over another random half (the
    # smoke of 2026-10-05: 0.707 by point and 0.902 by "group" in a design
    # without groups, all of the difference the half drawn).
    idx_fit <- if (is.null(g) || !anyDuplicated(g)) {
      with_local_seed(seed, sort(sample.int(n, max(2L, floor(fit_frac * n)))))
    } else {
      .draw_groups_for_frac(g, fit_frac, seed)
    }
    idx_cal <- setdiff(seq_len(n), idx_fit)
    if (length(idx_cal) < 2L || length(idx_fit) < ncol(X) + 3L) {
      stop("The split by group left ", length(idx_fit), " point(s) to fit the scale ",
           "and ", length(idx_cal), " to calibrate q: too few groups for a split.",
           call. = FALSE)
    }
    sc_fit <- conformal_scale_fit(o[idx_fit], p[idx_fit], X[idx_fit, , drop = FALSE],
                                  floor_frac = floor_frac)
  } else {
    gone <- setdiff(scale$terms, names(X))
    if (length(gone) > 0L) {
      stop("The scale was fitted on ", paste(scale$terms, collapse = ", "),
           "; covariates lacks ", paste(gone, collapse = ", "), ".", call. = FALSE)
    }
    idx_cal <- seq_len(n)
    sc_fit  <- scale
  }

  sc <- .conformal_scale(sc_fit$coef, sc_fit$floor, X[idx_cal, , drop = FALSE])
  cal <- conformal_calibrate(o[idx_cal], p[idx_cal], alpha = alpha, difficulty = sc,
                             group = if (is.null(g)) NULL else g[idx_cal])

  structure(c(unclass(cal), list(
    coef = sc_fit$coef, floor = sc_fit$floor, terms = sc_fit$terms,
    n_fit = sc_fit$n_fit, r2_fit = sc_fit$r2_fit,
    fit_frac = if (is.null(scale)) fit_frac else NA_real_,
    floor_frac = sc_fit$floor_frac, seed = seed, scale_given = !is.null(scale))),
    class = c("conformal_scaled", "conformal_cal"))
}

# The fitted scale, for new points. Computed from the coefficients and not
# with predict.lm(): the map applies it to billions of pixels, and a
# multiply-add needs no model frame.
.conformal_scale <- function(coef, floor, covariates) {
  terms <- names(coef)[-1L]
  gone <- setdiff(terms, names(covariates))
  if (length(gone) > 0L) {
    stop("The scale was fitted on ", paste(terms, collapse = ", "),
         "; these points lack ", paste(gone, collapse = ", "), ".", call. = FALSE)
  }
  X <- as.matrix(as.data.frame(covariates)[, terms, drop = FALSE])
  pmax(as.numeric(coef[1L] + X %*% coef[terms]), floor)
}

#' Turn predictions into intervals with a fitted scale.
#'
#' @param cal        From conformal_scaled_calibrate().
#' @param pred       Predictions, native units.
#' @param covariates The same covariates the scale was fitted on, for these
#'   predictions -- the level, and the dissimilarity index computed the same
#'   way as the calibration points' was.
#' @param lower_limit Floor of the lower bound (0 for a stock).
#' @return A tibble: pred, lower, upper and width, one row per prediction.
#' @examples
#' set.seed(1)
#' level <- runif(300, 1, 4)
#' di    <- runif(300)                      # a dissimilarity index
#' obs   <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di))
#' pred  <- exp(level)
#' covariates <- data.frame(level = pred, di = di)
#' cal <- conformal_scaled_calibrate(obs[1:200], pred[1:200], covariates[1:200, ],
#'                                   alpha = 0.1)
#' iv <- conformal_scaled_interval(cal, pred[201:300], covariates[201:300, ],
#'                                 lower_limit = 0)
#' picp_report(obs[201:300], iv$lower, iv$upper)
#' @export
conformal_scaled_interval <- function(cal, pred, covariates, lower_limit = -Inf) {
  stopifnot(inherits(cal, "conformal_scaled"))
  half <- cal$q * .conformal_scale(cal$coef, cal$floor, covariates)
  tibble::tibble(
    pred  = as.numeric(pred),
    lower = pmax(as.numeric(pred) - half, lower_limit),
    upper = as.numeric(pred) + half,
    width = pmin(as.numeric(pred) + half, Inf) - pmax(as.numeric(pred) - half, lower_limit))
}

#' Print a `conformal_scaled`
#'
#' @param x   A `conformal_scaled`, from [conformal_scaled_calibrate()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.conformal_scaled <- function(x, ...) {
  cat("\n<conformal_scaled> ", sprintf("%.0f%% intervals", 100 * (1 - x$alpha)),
      ", width = q x fitted scale\n", sep = "")
  terms <- names(x$coef)[-1L]
  cat(sprintf("  scale              : %.4f%s  (floor %.4f; R2 on the fit half %.3f)\n",
              x$coef[1L],
              paste(sprintf(" %+.4f x %s", x$coef[terms], terms), collapse = ""),
              x$floor, x$r2_fit))
  cat(sprintf("  points             : %d fitted the scale%s, %d calibrated q\n",
              x$n_fit, if (isTRUE(x$scale_given)) " (other points)" else "", x$n))
  if (identical(x$weighting, "group")) {
    cat(sprintf("  weighting          : by group -- %d group(s), each weighing the same\n",
                x$n_groups))
  } else {
    cat(sprintf("  rank used          : %d of %d  (the (n+1) correction)\n", x$k, x$n))
  }
  cat(sprintf("  q                  : %.4f x scale\n", x$q))
  invisible(x)
}

# ── CV+: the cross-validation's own models, at the new point ──────────────────
#
# WHAT THE CV CALIBRATION GETS WRONG, AND WHAT CV+ DOES INSTEAD.
#
# The cross-validated residuals R_i come from fold models; the interval they
# calibrate is put around another model -- the final one, refitted on more
# data. Barber, Candes, Ramdas & Tibshirani (2021, Ann. Stat. 49:486-507) keep
# the fold models at the new point as well: for each calibration point i, its
# fold's model predicts the new point, and the interval is
#
#   lower = the floor(alpha (n+1))-th smallest of  mu_k(i)(x) - R_i
#   upper = the ceil((1-alpha)(n+1))-th smallest of mu_k(i)(x) + R_i
#
# With exchangeable points and an algorithm fixed in advance, the coverage is at
# least 1 - 2 alpha - min{2(1 - 1/K)/(n/K + 1), (1 - K/n)/(K + 1)} >= 1 - 2 alpha
# - sqrt(2/n) (their Theorem 4) -- and close to 1 - alpha in practice. What it
# does not survive here: the configuration was chosen on these same folds, and
# the points cluster in space. Measured on the test set, as every interval is.
#
# THE ORDER STATISTIC, PIXEL BY PIXEL. The n values differ at every new point
# (each fold's model moves its share of them), so there is no single sorted
# vector to index. The count of values at or below t is a sum, over the folds,
# of how many of that fold's sorted scores are at or below (t - mu_k(x)) / d(x)
# -- findInterval(), vectorised over the pixels -- and the r-th smallest is the
# smallest t whose count reaches r. It lies between the folds' lowest and
# highest prediction plus the pooled r-th score (every value is at least its
# row's lowest mu plus its score, at most the highest), so bisection starts on
# a bracket as wide as the folds disagree, and the result is snapped to the
# largest value at or below the bracket's top: the order statistic itself, to
# a tolerance far below a float32 band. Python against brute force, 300 random
# draws with ties: 7e-8 at most (2026-10-05).

#' Calibrate a CV+ interval from the folds of a cross-validation.
#'
#' @param obs  Observations of the calibration points: every point validated
#'   once, in one fold.
#' @param pred Each point's out-of-fold prediction -- by the model of the fold
#'   that held it out (for a seed ensemble, the median of that fold's seeds).
#' @param fold The fold that held each point out.
#' @param alpha Miscoverage rate: 0.1 asks for 90%.
#' @param difficulty Optional positive scale of each point, for a normalised
#'   CV+: the residuals are divided by it, and [cv_plus_interval()] multiplies
#'   them back by the new point's.
#' @return A `cv_plus_cal`: each fold's sorted scores, the rank, the folds.
#' @examples
#' set.seed(1)
#' x <- runif(200)
#' fold <- rep(1:5, 40)
#' y <- 10 + 5 * x + rnorm(200)
#' # each fold's model: a line fitted without that fold
#' fits <- lapply(1:5, function(k) lm(y ~ x, data = data.frame(x, y)[fold != k, ]))
#' oof  <- vapply(seq_along(y), function(i)
#'   unname(predict(fits[[fold[i]]], data.frame(x = x[i]))), numeric(1))
#' cal <- cv_plus_calibrate(y, oof, fold, alpha = 0.1)
#' cal
#' # the five models at three new points, one column per fold
#' new <- data.frame(x = c(0.1, 0.5, 0.9))
#' cv_plus_interval(cal, sapply(fits, predict, newdata = new))
#' @export
cv_plus_calibrate <- function(obs, pred, fold, alpha = 0.1, difficulty = NULL) {
  n0 <- length(obs)
  if (length(pred) != n0 || length(fold) != n0) {
    stop("obs, pred and fold must describe the same points: ", n0, ", ",
         length(pred), " and ", length(fold), ".", call. = FALSE)
  }
  if (!is.numeric(alpha) || length(alpha) != 1L || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a single number in (0, 1).", call. = FALSE)
  }
  keep <- is.finite(obs) & is.finite(pred) & !is.na(fold)
  if (!is.null(difficulty)) {
    if (length(difficulty) != n0) {
      stop("difficulty has ", length(difficulty), " value(s) for ", n0, " point(s).",
           call. = FALSE)
    }
    keep <- keep & is.finite(difficulty) & difficulty > 0
  }
  o <- as.numeric(obs)[keep]
  p <- as.numeric(pred)[keep]
  f <- fold[keep]
  d <- if (is.null(difficulty)) rep(1, length(o)) else as.numeric(difficulty)[keep]
  n <- length(o)
  # The folds in their own order -- numbers as numbers, so fold 10 comes after
  # fold 9 -- which is the order an unnamed fold_pred's columns are read in.
  folds <- sort(unique(f))
  if (length(folds) < 2L) {
    stop("CV+ needs at least 2 folds: with one, no point's model differs from ",
         "another's, and the interval is a split interval with extra steps.",
         call. = FALSE)
  }
  res <- abs(o - p)
  score <- res / d
  by_fold <- lapply(folds, function(k) sort(score[f == k]))
  names(by_fold) <- as.character(folds)
  r <- ceiling((n + 1) * (1 - alpha))
  if (r > n) {
    warning("alpha = ", alpha, " needs at least ", ceiling(1 / alpha) - 1,
            " calibration points to be certifiable; there are ", n,
            ". The interval is infinite, which is the correct answer.",
            call. = FALSE)
  }
  structure(list(alpha = alpha, n = n, r = as.integer(r), folds = folds,
                 scores = by_fold, normalised = !is.null(difficulty),
                 residuals = res),
            class = "cv_plus_cal")
}

#' Turn the fold models' predictions into CV+ intervals.
#'
#' @param cal       A `cv_plus_cal`, from [cv_plus_calibrate()].
#' @param fold_pred A matrix, one row per new point and one column per fold:
#'   that fold's model at the point (for a seed ensemble, the median of the
#'   fold's seeds). Columns named by fold are matched by name; unnamed ones
#'   are read in the order of `cal$folds`.
#' @param difficulty The new points' scale, for a normalised calibration;
#'   refused for a plain one, as in [conformal_interval()].
#' @param lower_limit Floor of the lower bound, e.g. 0 for a stock.
#' @return A tibble: lower, upper and width, one row per new point.
#' @examples
#' set.seed(1)
#' fold <- rep(1:4, 25)
#' oof  <- rnorm(100, 20, 3)              # each point's out-of-fold prediction
#' obs  <- oof + rnorm(100, 0, 2)
#' cal  <- cv_plus_calibrate(obs, oof, fold, alpha = 0.1)
#' # the four fold models at two new points: they disagree a little
#' cv_plus_interval(cal, rbind(c(19.5, 20.2, 20.0, 20.4), c(30.1, 29.0, 29.8, 30.6)),
#'                  lower_limit = 0)
#' @export
cv_plus_interval <- function(cal, fold_pred, difficulty = NULL, lower_limit = -Inf) {
  stopifnot(inherits(cal, "cv_plus_cal"))
  mu <- as.matrix(fold_pred)
  storage.mode(mu) <- "double"
  K <- length(cal$folds)
  if (ncol(mu) != K) {
    stop("fold_pred has ", ncol(mu), " column(s); the calibration has ", K, " fold(s).",
         call. = FALSE)
  }
  if (!is.null(colnames(mu))) {
    pos <- match(as.character(cal$folds), colnames(mu))
    if (anyNA(pos)) {
      stop("fold_pred's columns do not name the calibration's folds (",
           paste(cal$folds, collapse = ", "), ").", call. = FALSE)
    }
    mu <- mu[, pos, drop = FALSE]
  }
  if (cal$normalised && is.null(difficulty)) {
    stop("This calibration is normalised, so it needs a difficulty score for ",
         "each new point -- the same kind used to calibrate it.", call. = FALSE)
  }
  if (!cal$normalised && !is.null(difficulty)) {
    stop("This calibration is not normalised; a difficulty score here would ",
         "widen the intervals with no guarantee attached to the widening.",
         call. = FALSE)
  }
  N <- nrow(mu)
  d <- if (cal$normalised) as.numeric(difficulty) else rep(1, N)
  if (length(d) != N) {
    stop("difficulty has ", length(d), " value(s) for ", N, " point(s).", call. = FALSE)
  }
  lower <- upper <- rep(NA_real_, N)
  ok <- is.finite(rowSums(mu)) & is.finite(d) & d > 0
  if (cal$r > cal$n) {
    lower[ok] <- -Inf
    upper[ok] <- Inf
  } else if (any(ok)) {
    m <- mu[ok, , drop = FALSE]
    upper[ok] <- .cv_plus_order_stat(m, d[ok], cal$scores, cal$r)
    # The floor(alpha (n+1))-th smallest of mu - R is minus the
    # ceil((1-alpha)(n+1))-th smallest of -mu + R: one routine, both bounds.
    lower[ok] <- -.cv_plus_order_stat(-m, d[ok], cal$scores, cal$r)
  }
  lower <- pmax(lower, lower_limit)
  tibble::tibble(lower = lower, upper = upper, width = upper - lower)
}

# The r-th smallest of {mu[, k] + d * scores[[k]][i]}, row by row (see above).
.cv_plus_order_stat <- function(mu, d, scores, r, tol = 1e-7, max_iter = 80L) {
  N <- nrow(mu)
  K <- ncol(mu)
  s_r <- sort(unlist(scores, use.names = FALSE))[r]
  lo <- matrixStats::rowMins(mu) + d * s_r
  hi <- matrixStats::rowMaxs(mu) + d * s_r
  # A value equal to t counts. (t - mu) / d is a hair below the score it came
  # from as often as not -- 12.299999999999997 against 12.3 -- so the
  # comparison has a tolerance; without it, an exact tie fell out of the count.
  at_or_below <- function(t, rows) {
    out <- integer(length(rows))
    for (k in seq_len(K)) {
      u <- (t - mu[rows, k]) / d[rows]
      out <- out + findInterval(u + 1e-9 * pmax(1, abs(u)), scores[[k]])
    }
    out
  }
  at_lo <- at_or_below(lo, seq_len(N)) >= r
  act <- which(!at_lo)
  for (it in seq_len(max_iter)) {
    act <- act[hi[act] - lo[act] > tol * pmax(1, abs(hi[act]))]
    if (length(act) == 0L) break
    mid <- (lo[act] + hi[act]) / 2
    up  <- at_or_below(mid, act) >= r
    hi[act[up]]  <- mid[up]
    lo[act[!up]] <- mid[!up]
  }
  snap <- rep(-Inf, N)
  for (k in seq_len(K)) {
    u   <- (hi - mu[, k]) / d
    idx <- findInterval(u + 1e-9 * pmax(1, abs(u)), scores[[k]])
    hit <- idx >= 1L
    snap[hit] <- pmax(snap[hit], mu[hit, k] + d[hit] * scores[[k]][idx[hit]])
  }
  ifelse(at_lo, lo, snap)
}

#' Print a `cv_plus_cal`
#'
#' @param x   A `cv_plus_cal`, from [cv_plus_calibrate()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.cv_plus_cal <- function(x, ...) {
  cat("\n<cv_plus_cal> ", sprintf("%.0f%% intervals", 100 * (1 - x$alpha)),
      if (x$normalised) " (normalised)" else "", "\n", sep = "")
  cat(sprintf("  calibration points : %d, in %d fold(s) (%s)\n", x$n, length(x$folds),
              paste(lengths(x$scores), collapse = ", ")))
  if (x$r > x$n) {
    cat("  rank               : too few points to certify this alpha -- infinite\n")
  } else {
    cat(sprintf("  rank used          : %d of %d, both bounds (Barber et al. 2021)\n",
                x$r, x$n))
  }
  invisible(x)
}

# ── Coverage on held-out points, by what the interval promised ────────────────
#
# One number per interval is how an under-covering corner hides: 90% overall
# is compatible with 99% where the soil is easy and 70% where it is not. So
# the coverage is reported over all points; as the mean over groups, each
# weighing the same -- what a group-weighted calibration promises; and within
# every stratum given (inside and outside the AOA, fifths of the predicted
# level, regions).
.coverage_rows <- function(obs, lower, upper, group = NULL, strata = list()) {
  obs <- as.numeric(obs); lower <- as.numeric(lower); upper <- as.numeric(upper)
  ok <- is.finite(obs) & !is.na(lower) & !is.na(upper)
  inside <- obs >= lower & obs <= upper
  width  <- upper - lower
  one <- function(type, name, sel) {
    tibble::tibble(stratum_type = type, stratum = as.character(name), n = sum(sel),
                   coverage = if (any(sel)) mean(inside[sel]) else NA_real_,
                   mean_width = if (any(sel)) mean(width[sel]) else NA_real_)
  }
  out <- list(one("all", "points", ok))
  if (!is.null(group)) {
    g <- as.character(group)
    sel <- ok & !is.na(g)
    if (any(sel)) {
      cov_g <- tapply(inside[sel], g[sel], mean)
      wid_g <- tapply(width[sel], g[sel], mean)
      out[[length(out) + 1L]] <- tibble::tibble(
        stratum_type = "all", stratum = "groups (each weighs the same)",
        n = length(cov_g), coverage = mean(cov_g), mean_width = mean(wid_g))
    }
  }
  for (nm in names(strata)) {
    s <- as.character(strata[[nm]])
    for (lv in sort(unique(s[ok & !is.na(s)]))) {
      out[[length(out) + 1L]] <- one(nm, lv, ok & !is.na(s) & s == lv)
    }
  }
  dplyr::bind_rows(out)
}

# ── Where the calibration residuals should come from ──────────────────────────
#
# A LESSON PAID FOR IN A RUN.
#
# Stage 04 first calibrated on the validation rows of its own refit split. That
# is held out of training, so it looked correct, and the coverage came back at
# 83.6% against a nominal 90%.
#
# The refit validation is ONE fold of k = 7 -- 449 points from one spatial
# region, and the easiest of the three sets by a wide margin (MAE 14.6 against
# 17.0 across the cross-validation and 18.2 on the test set). Calibrating on one
# block and measuring coverage on another is exchangeability failure by
# construction, and the interval came out too narrow by exactly that.
#
# The cross-validated residuals are the better calibration set, and the tuning
# run already wrote them: every point is predicted once as validation, across
# every fold, so the residual distribution spans the whole area rather than one
# corner of it. Recalibrating that way moved coverage from 83.6% to 87.8%.
#
# IT STILL UNDER-COVERS, AND THAT IS A RESULT RATHER THAN A BUG. The CV
# calibration is conservative by construction -- those models trained on ~53% of
# the points against the final model's 69%, so their residuals are larger -- and
# the interval is still short on the test set. What is left is the part
# conformal cannot fix: a held-out spatial block is not exchangeable with the
# blocks used to calibrate. That gap is worth reporting, and this framework
# reports it rather than tuning alpha until the number looks right.
#
# WHY THIS IS NOT SPLIT CONFORMAL, STRICTLY. The configuration was chosen on
# these same folds, and the residuals are applied to a refit on more data, so
# the calibration points are not a set nothing else touched. Selection on the
# folds biases the residuals small; the larger training set of the refit biases
# them large. Neither bias is bounded, so the finite-sample guarantee is gone
# and the coverage is an estimate, which is why it is always measured on the
# test rows. It stays the default because it costs nothing: no point leaves
# the training, no model is run again.
#
# THE TWO WITH A GUARANTEE ARE OFFERED BESIDE IT (2026-10-05). "split": a
# calibration set the plan carves beside the test set (calibration_frac),
# which nothing trains on or chooses with -- the guarantee above, paid for in
# training points. "cv_plus": CV+ (above), the fold models at every pixel --
# no point spent, paid for in network passes. dsm_final() calibrates every
# method the runs allow and checks every one on the test set, overall, by
# group, inside and outside the AOA and by level: which assumption breaks
# least on clustered profiles is the test set's to say.

#' Cross-validated residuals from a tuning run, for calibration.
#'
#' Every point appears once as validation, pooled over folds and averaged over
#' seeds, so the residual distribution covers the whole study area instead of
#' one fold's worth of it.
#'
#' @param run_dir   Tuning run directory (the one holding predictions/).
#' @param config_id Which config's predictions to read.
#' @param role      Which role to keep. "validation" is the point of this.
#' @return A tibble with sample_id, obs, pred (the seed ensemble's median), and
#'   n_seeds; or NULL when the run wrote no usable predictions.
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()    # a small fitted run, made once a session
#' res <- cv_residuals(run$fit$run_dir, run$final$selected_config_ids)
#' head(res)
#' conformal_calibrate(res$obs, res$pred, alpha = 0.1)
#' }
#' @export
cv_residuals <- function(run_dir, config_id, role = "validation") {
  pred_dir <- file.path(run_dir, "predictions")
  files <- list.files(pred_dir,
                      pattern = sprintf("^%s_f[0-9]+_s[0-9]+_pred_all\\.csv$",
                                        config_id),
                      full.names = TRUE)
  if (length(files) == 0L) return(NULL)

  rows <- purrr::map_dfr(files, function(f) {
    d <- suppressMessages(readr::read_csv2(f, show_col_types = FALSE))
    if (!all(c("sample_id", "dataset_role", "obs", "pred") %in% names(d))) {
      return(tibble::tibble())
    }
    dplyr::select(dplyr::filter(d, .data$dataset_role == role),
                  sample_id, obs, pred)
  })
  if (nrow(rows) == 0L) return(NULL)

  # The MEDIAN over seeds, matching what the map will be: the deployed
  # prediction is the ensemble median, so the residual being calibrated has to
  # be the ensemble's residual and not one seed's.
  rows %>%
    dplyr::group_by(.data$sample_id) %>%
    dplyr::summarise(obs = dplyr::first(.data$obs),
                     pred = stats::median(.data$pred),
                     n_seeds = dplyr::n(), .groups = "drop")
}

# ── Residuals from ANOTHER run: the same configuration, found by what it is ───
#
# The calibration source is an argument (block folds or kNNDM folds decide
# which job the interval is honest for -- see above), so the residuals often
# come from a run other than the one the config was selected in. And there a
# config_id means nothing: it is a label within ONE run. This project met it
# on 2026-09-18 -- the deployed cfg_003 and the kNNDM design run's cfg_003
# differ in three dropout fields. So the config is found by its
# hyperparameters, every one of them, and a run without an identical config
# gives no residuals at all rather than a neighbour's.

# Every hyperparameter of a grid row, as one string. config_id is a label and
# n_params a consequence, so neither enters.
.config_signature <- function(cfg_row) {
  keep <- sort(setdiff(names(cfg_row), c("config_id", "n_params")))
  paste(vapply(keep, function(p) {
    v <- cfg_row[[p]]
    if (is.list(v)) v <- v[[1]]
    paste0(p, "=", paste(format(v, digits = 15, trim = TRUE), collapse = "x"))
  }, character(1)), collapse = "|")
}

#' Cross-validated residuals of a configuration, from any tuning run.
#'
#' @param run_dir  A tuning run directory (tune_grid.rds, predictions/).
#' @param cfg_row  One row of a grid: the configuration, whatever it is called
#'   in that run.
#' @param required TRUE stops, saying why, when the run cannot serve the
#'   configuration; FALSE says why and returns NULL.
#' @return What cv_residuals() returns, with the run's own config_id attached
#'   as attribute "config_id".
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()    # a small fitted run, made once a session
#' grid <- readRDS(file.path(run$fit$run_dir, "tune_grid.rds"))
#' cfg  <- grid[grid$config_id == run$final$selected_config_ids, ]
#' # found by its hyperparameters, so the same call reads any other tuning run
#' head(cv_residuals_for_config(run$fit$run_dir, cfg))
#' }
#' @export
cv_residuals_for_config <- function(run_dir, cfg_row, required = TRUE) {
  fail <- function(reason) {
    if (required) stop(reason, call. = FALSE)
    message(reason)
    NULL
  }
  grid_path <- file.path(run_dir, "tune_grid.rds")
  if (!file.exists(grid_path)) return(fail(paste("No tune_grid.rds in", run_dir)))
  grid <- readRDS(grid_path)
  sig  <- .config_signature(cfg_row)
  same <- vapply(seq_len(nrow(grid)), function(i)
    identical(.config_signature(grid[i, , drop = FALSE]), sig), logical(1))
  if (!any(same)) {
    return(fail(paste0("No configuration in ", basename(run_dir), " has these ",
                       "hyperparameters -- a config_id names a different architecture ",
                       "in every run, so none is borrowed by name.")))
  }
  cid <- grid$config_id[which(same)[1]]
  res <- cv_residuals(run_dir, cid)
  if (is.null(res)) {
    return(fail(paste0(basename(run_dir), " holds the configuration (as ", cid,
                       ") but wrote no predictions for it.")))
  }
  attr(res, "config_id") <- cid
  res
}
