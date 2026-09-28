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
# being exchangeable with the new points.
#
# Measured on this project's cross-validated ensemble residuals (3,092 points):
#
#   S = 1.3461        bias  -24.4%  ->  +2.7%
#                     RMSE  28.07   ->  27.00
#                     MAE   18.15   ->  19.02
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
#
# ── HOW GOOD IS THE SCALAR? NOT AS GOOD AS ONE DECIMAL PLACE SUGGESTS ─────────
#
# Two things were measured here and both bound what this correction can claim.
#
# 1. THE TARGET IS BRACKETED, NOT PINNED. Measuring S directly on the deployed
#    10-seed ensemble's own held-out residuals gives 1.2590 on the refit
#    validation fold (449 points) and 1.3890 on the test set (591 points) --
#    a 10% span, and it straddles every candidate. Residual VARIANCE shrinks
#    with ensemble size as theory says (sd 0.6698 one seed -> 0.6507 three ->
#    0.6393 ten), but the residual MEAN moves four times further and in
#    inconsistent directions. So an S quoted to four figures is false precision,
#    and a change of 1% in it is not a bias fix; it is a coin flip.
#
# 2. DUAN ASSUMES e IS INDEPENDENT OF x, AND HERE IT IS NOT. S by quintile of
#    the prediction, on the out-of-fold ensemble:
#
#      lowest predictions  1.795  1.382  1.258  1.142  1.154  highest
#
#    A single scalar is set largely by the low-prediction points and then
#    applied to the high-prediction pixels, which carry most of the carbon (the
#    top two quintiles hold 60% of the observed total). This is a real
#    assumption violation, not a rounding error: the estimator that actually
#    unbiases a map TOTAL -- the exp(f)-weighted mean of exp(e) -- comes out at
#    1.2370, 8% below the unweighted 1.3461, which is eight times the
#    calibration-set question that turned this up.
#
# TWO REFINEMENTS WERE TRIED AND BOTH ARE WORSE ON HELD-OUT GROUND.
# Calibrated out-of-fold, applied blind to the frozen test set:
#
#      S = 1.3461  global, unweighted        bias  +2.7%   RMSE 27.00
#      S = 1.2370  exp(f)-weighted for a total     -5.8%   RMSE 26.72
#      S by quintile of the prediction             -3.9%   RMSE 26.48
#
# The refinements move the bias further from zero, not closer. The reason is a
# mismatch neither of them models: the calibration models are trained on the CV
# folds (~53% of the points) and the deployed model on the refit split (~69%),
# so the calibration residuals are systematically larger and their structure
# does not transfer. That transfer gap dominates both refinements, which is why
# the plain scalar stays and the diagnostic below exists to keep the violation
# visible instead of assumed away.

#' The smearing factor, from held-out residuals in transform space.
#'
#' @param obs_transform,pred_transform Observed and predicted values IN
#'   TRANSFORM SPACE (log1p here), on data the model did not train on. Training
#'   residuals are smaller than the ones new points will produce, so a factor
#'   calibrated on them under-corrects by exactly the amount the model overfits.
#'   ONE ROW PER POINT: see smearing_from_run() for why seeds are collapsed
#'   first.
#' @param transform Name of the forward transform. Only "log1p" and "log" are
#'   supported, and an unknown one is refused rather than assumed: the algebra
#'   above is specific to an exponential back-transform, and applying it to a
#'   square root or an identity would scale a number that needs no scaling.
#' @return An object of class "smearing_cal".
#' @export
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

  # THE INDEPENDENCE CHECK, which is the assumption that licenses ONE scalar.
  #
  # Duan's derivation needs e independent of x. Nothing so far tests that, and
  # on this project it is false: S runs from 1.80 at the lowest predictions to
  # 1.15 at the highest. One number then over-corrects the low pixels and
  # under-corrects the high ones -- and the high ones carry most of the stock,
  # so it lands squarely on any total computed from the map.
  #
  # Reported, NOT corrected. Both obvious repairs (an exp(f)-weighted factor, a
  # per-quintile factor) were measured on held-out ground and both make the bias
  # worse, because the calibration models see less training data than the
  # deployed one and the structure does not transfer. So this prints the
  # violation and leaves the decision with whoever reads it.
  #
  # exp(f)-weighted S is carried alongside because it is the functional that
  # unbiases a SUM, and the mean surface is the one the docs say may be summed.
  bins <- if (n >= 250L) {
    q <- stats::quantile(p, probs = seq(0, 1, length.out = 6), na.rm = TRUE)
    idx <- cut(p, breaks = unique(q), include.lowest = TRUE, labels = FALSE)
    vapply(split(e, idx), function(z) mean(exp(z)), numeric(1))
  } else {
    numeric(0)
  }
  s_weighted <- sum(exp(p) * exp(e)) / sum(exp(p))

  structure(list(s = s, n = n, transform = transform,
                 mean_residual = mean(e), sd_residual = stats::sd(e),
                 s_lognormal = s_lnorm,
                 agreement = s / s_lnorm,
                 s_by_bin = bins,
                 s_weighted = s_weighted),
            class = "smearing_cal")
}

#' The conditional MEAN surface, from transform-space predictions.
#'
#' @param pred_transform Predictions in transform space.
#' @param cal A smearing_cal, or a bare positive number to use as the factor.
#' @param lower_limit Floor, e.g. 0 for a stock. Applied after the correction.
#' @return Numeric, in native units: an estimate of `E[y | x]` rather than of its
#'   median.
#' @export
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

#' Print a `smearing_cal`
#'
#' @param x   A `smearing_cal`, from [smearing_factor()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.smearing_cal <- function(x, ...) {
  cat("\n<smearing_cal> ", x$transform, " back-transform\n", sep = "")
  cat(sprintf("  held-out points    : %d\n", x$n))
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

  if (length(x$s_by_bin) > 1L) {
    cat(sprintf("  S by prediction    : %s\n",
                paste(sprintf("%.3f", x$s_by_bin), collapse = " ")))
    spread <- max(x$s_by_bin) / min(x$s_by_bin)
    if (spread > 1.25) {
      cat(sprintf("  -> S varies %.0f%% across the prediction range, so the residual\n",
                  100 * (spread - 1)))
      cat("     is NOT independent of the prediction and one scalar cannot suit\n")
      cat("     both ends. It over-corrects where S is small and under-corrects\n")
      cat("     where S is large. Since high predictions carry most of the\n")
      cat("     stock, this falls mainly on any TOTAL taken from the map.\n")
      cat(sprintf("     For reference, the factor that unbiases a sum is %.4f.\n",
                  x$s_weighted))
      cat("     Both obvious repairs were measured here and BOTH made the\n")
      cat("     held-out bias worse -- see the note at the top of R/smearing.R\n")
      cat("     before reaching for one.\n")
    }
  }

  cat(sprintf("\n  A prediction of %.1f in log space becomes %.2f (median) or ",
              1.5, expm1(1.5)))
  cat(sprintf("%.2f (mean).\n", exp(1.5) * x$s - 1))
  invisible(x)
}

#' The smearing factor from a tuning run's out-of-fold predictions.
#'
#' ONE RESIDUAL PER POINT, not one per (point, seed).
#'
#' The deployed prediction is the ensemble MEDIAN over seeds, so the residual
#' being calibrated has to be the ensemble's and not one member's. This is the
#' same reasoning cv_residuals() applies for the conformal interval, and for a
#' while this function claimed to use "the same out-of-fold residuals" while
#' quietly reading the per-seed rows instead: 9,276 rows for 3,092 points.
#'
#' Two consequences, and the second is the one that matters:
#'
#'   - exp() is convex, so mean(exp(e)) grows with var(e). A single member's
#'     residual is noisier than the ensemble's (sd 0.6698 against 0.6507 here),
#'     which inflated S by 0.99% -- almost exactly the 1.00% that the variance
#'     difference alone predicts.
#'
#'   - `n` read 9,276 when there were 3,092 exchangeable units, overstating the
#'     evidence threefold in the one line a reader would use to judge it.
#'
#' The 1% is NOT why this was changed. The target is bracketed between 1.259 and
#' 1.389 by two held-out measurements of the deployed ensemble itself, so no
#' choice inside that range is defensible as more accurate. It was changed
#' because the claim in the docs was false and the printed n was wrong.
#'
#' @param run_dir   Tuning run directory.
#' @param config_id Which config's residuals.
#' @param role      Which role. "validation" is the point.
#' @inheritParams smearing_factor
#' @return A smearing_cal, or NULL when the run wrote no usable predictions.
#' @export
smearing_from_run <- function(run_dir, config_id, role = "validation",
                              transform = "log1p") {
  files <- list.files(
    file.path(run_dir, "predictions"),
    pattern = sprintf("^%s_f[0-9]+_s[0-9]+_pred_all[.]csv$", config_id),
    full.names = TRUE)
  if (length(files) == 0L) return(NULL)

  rows <- purrr::map_dfr(files, function(f) {
    d <- suppressMessages(readr::read_csv2(f, show_col_types = FALSE))
    need <- c("sample_id", "dataset_role", "obs_transform", "pred_transform")
    if (!all(need %in% names(d))) return(tibble::tibble())
    dplyr::select(dplyr::filter(d, .data$dataset_role == role),
                  sample_id, obs_transform, pred_transform)
  })
  if (nrow(rows) == 0L) return(NULL)

  # The median in transform space IS the transform of the median: expm1 is
  # monotone, so this collapses to the same ensemble the map deploys.
  ens <- rows %>%
    dplyr::group_by(.data$sample_id) %>%
    dplyr::summarise(obs_transform = dplyr::first(.data$obs_transform),
                     pred_transform = stats::median(.data$pred_transform),
                     .groups = "drop")

  if (nrow(ens) < 30L) return(NULL)
  smearing_factor(ens$obs_transform, ens$pred_transform,
                  transform = transform)
}
