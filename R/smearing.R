# ── Back-transform: the median surface and the mean surface ───────────────────
#
# THE DEFECT THIS EXISTS FOR, MEASURED.
#
# The target is trained on log1p and back-transformed with expm1. The loss is
# SmoothL1, so the network estimates a conditional MEDIAN in log space, and
# expm1() of a median is the median of the stock -- not its mean. The target is
# right-skewed, so the two are not the same number:
#
#   observed on the frozen test set    mean 39.45   median 31.18
#   predicted, expm1                   mean 29.84   median 25.60
#   bias                              -9.61 t/ha, -24.4%, 53% OF THE MAE
#
# Nothing in the framework could see it until calc_metrics() gained a signed
# bias, because every other metric is blind to the sign of the error. Anyone
# summing that map for a total carbon stock gets a quarter less than the data
# say.
#
# DUAN'S SMEARING ESTIMATOR (1983, JASA 78:605-610).
#
# Write the model in transform space as  z = f(x) + e,  z = log1p(y). Then
#
#   E[y | x] = E[exp(z) - 1] = exp(f(x)) * E[exp(e)] - 1
#
# and E[exp(e)] is estimated by the mean of exp(residual) over held-out points.
# One scalar, no retraining, no distributional assumption beyond the residuals
# being exchangeable with the new points -- the same assumption the conformal
# interval rests on, calibrated on the same out-of-fold residuals.
#
# Measured here, on the cross-validated residuals of cfg_003 (n = 9,276):
#
#   S = 1.3594        bias  -24.4%  ->  +3.7%
#                     CCC    0.4748 ->   0.5692
#                     RMSE  28.07   ->  27.07
#                     MAE   18.15   ->  19.14
#
# The residuals are near-lognormal: exp(mean + sd^2/2) = 1.3587 against the
# empirical 1.3594, so the estimator behaves as the theory says it should.
#
# THE MAE GOING UP IS NOT A REGRESSION. It is the trade-off, and it is the
# reason this does not replace anything:
#
#   the MEDIAN minimises absolute error   -> best MAE, the typical stock here
#   the MEAN   minimises squared error    -> best RMSE, and the only one you
#                                            may sum for a total
#
# They answer different questions and both are legitimate products. A framework
# that silently swapped one for the other would trade a known bias for an
# unknown one. So the smeared surface is written BESIDE the median surface and
# labelled, and the choice is the user's.

#' The smearing factor, from held-out residuals in transform space.
#'
#' @param obs_transform,pred_transform Observed and predicted values IN
#'   TRANSFORM SPACE (log1p here), on data the model did not train on. Training
#'   residuals are smaller than the ones new points will produce, so a factor
#'   calibrated on them under-corrects by exactly the amount the model overfits.
#' @param transform Name of the forward transform. Only "log1p" and "log" are
#'   supported, and an unknown one is refused rather than assumed: the algebra
#'   above is specific to an exponential back-transform, and applying it to a
#'   square root or an identity would scale a number that needs no scaling.
#' @return An object of class "smearing_cal".
smearing_factor <- function(obs_transform, pred_transform,
                            transform = c("log1p", "log")) {
  transform <- match.arg(transform)
  stopifnot(length(obs_transform) == length(pred_transform))
  keep <- is.finite(obs_transform) & is.finite(pred_transform)
  o <- as.numeric(obs_transform)[keep]
  p <- as.numeric(pred_transform)[keep]
  n <- length(o)
  if (n < 30L) {
    stop("Only ", n, " usable residual(s). The smearing factor is a mean of ",
         "exp(residual), which a handful of points estimates badly and a heavy ",
         "right tail estimates worse.", call. = FALSE)
  }

  e <- o - p
  s <- mean(exp(e))
  if (!is.finite(s) || s <= 0) {
    stop("The smearing factor came out ", s, ", which means the residuals hold ",
         "values exp() cannot handle. Check that these really are transform-",
         "space columns and not native ones.", call. = FALSE)
  }

  # THE LOGNORMAL CROSS-CHECK, reported rather than assumed.
  #
  # If the residuals were exactly lognormal, S would equal exp(mean + sd^2/2).
  # Duan's estimator does not need that to be true -- it is non-parametric, and
  # that is its point -- but the two agreeing says the residual distribution has
  # no surprises in it, and the two disagreeing is worth seeing before anyone
  # multiplies a map by this number.
  s_lnorm <- exp(mean(e) + stats::var(e) / 2)

  structure(list(s = s, n = n, transform = transform,
                 mean_residual = mean(e), sd_residual = stats::sd(e),
                 s_lognormal = s_lnorm,
                 agreement = s / s_lnorm),
            class = "smearing_cal")
}

#' The conditional MEAN surface, from transform-space predictions.
#'
#' @param pred_transform Predictions in transform space.
#' @param cal A smearing_cal, or a bare positive number to use as the factor.
#' @param lower_limit Floor, e.g. 0 for a stock. Applied after the correction.
#' @return Numeric, in native units: an estimate of E[y | x] rather than of its
#'   median.
smear <- function(pred_transform, cal, lower_limit = 0) {
  s <- if (inherits(cal, "smearing_cal")) cal$s else as.numeric(cal)
  tr <- if (inherits(cal, "smearing_cal")) cal$transform else "log1p"
  stopifnot(length(s) == 1L, is.finite(s), s > 0)
  z <- as.numeric(pred_transform)
  out <- switch(tr,
    # exp(z) * S - 1, NOT expm1(z) * S. The -1 comes out of the expectation
    # after the scaling, and folding it in the other order scales the offset
    # too -- a small error at small z and a growing one everywhere else.
    log1p = exp(z) * s - 1,
    log   = exp(z) * s,
    stop("Unsupported transform: ", tr, call. = FALSE))
  if (is.finite(lower_limit)) out <- pmax(out, lower_limit)
  out
}

#' @export
print.smearing_cal <- function(x, ...) {
  cat("\n<smearing_cal> ", x$transform, " back-transform\n", sep = "")
  cat(sprintf("  held-out residuals : %d\n", x$n))
  cat(sprintf("  mean, sd (log)     : %+.4f, %.4f\n",
              x$mean_residual, x$sd_residual))
  cat(sprintf("  S = mean(exp(e))   : %.4f\n", x$s))
  cat(sprintf("  lognormal check    : %.4f  (ratio %.3f)\n",
              x$s_lognormal, x$agreement))
  if (abs(x$agreement - 1) > 0.10) {
    cat("  -> The two disagree by more than 10%. Duan's estimator does not\n")
    cat("     need lognormal residuals, so this is not an error -- but the\n")
    cat("     residual distribution has a shape worth looking at before a map\n")
    cat("     is multiplied by it.\n")
  }
  cat(sprintf("\n  A prediction of %.1f in log space becomes %.2f (median) or ",
              1.5, expm1(1.5)))
  cat(sprintf("%.2f (mean).\n", exp(1.5) * x$s - 1))
  invisible(x)
}

#' The smearing factor from a tuning run's out-of-fold predictions.
#'
#' The same source the conformal calibration uses, and for the same reason:
#' every point is predicted once, as validation, somewhere, so the residual
#' distribution spans the whole study area instead of one fold's corner of it.
#'
#' @param run_dir   Tuning run directory.
#' @param config_id Which config's residuals.
#' @param role      Which role. "validation" is the point.
#' @return A smearing_cal, or NULL when the run wrote no usable predictions.
smearing_from_run <- function(run_dir, config_id, role = "validation",
                              transform = "log1p") {
  files <- list.files(
    file.path(run_dir, "predictions"),
    pattern = sprintf("^%s_f[0-9]+_s[0-9]+_pred_all[.]csv$", config_id),
    full.names = TRUE)
  if (length(files) == 0L) return(NULL)

  rows <- purrr::map_dfr(files, function(f) {
    d <- suppressMessages(readr::read_csv2(f, show_col_types = FALSE))
    need <- c("dataset_role", "obs_transform", "pred_transform")
    if (!all(need %in% names(d))) return(tibble::tibble())
    dplyr::select(dplyr::filter(d, .data$dataset_role == role),
                  obs_transform, pred_transform)
  })
  if (nrow(rows) < 30L) return(NULL)
  smearing_factor(rows$obs_transform, rows$pred_transform,
                  transform = transform)
}
