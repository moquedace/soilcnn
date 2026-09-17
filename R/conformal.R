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
#' @return An object of class "conformal_cal".
conformal_calibrate <- function(obs, pred, alpha = 0.1, difficulty = NULL) {
  stopifnot(length(obs) == length(pred))
  if (!is.numeric(alpha) || length(alpha) != 1L || alpha <= 0 || alpha >= 1) {
    stop("alpha must be a single number in (0, 1).", call. = FALSE)
  }
  keep <- is.finite(obs) & is.finite(pred)
  if (!is.null(difficulty)) {
    stopifnot(length(difficulty) == length(obs))
    keep <- keep & is.finite(difficulty) & difficulty > 0
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

  # THE (n+1) CORRECTION. The guarantee is over the calibration set PLUS the new
  # point, so the rank is taken out of n+1, not n. k > n means this calibration
  # set is too small to certify this alpha at all: the threshold is
  # ceiling(1/alpha) - 1, so 90% needs 9 points and 95% needs 19. Below it the
  # honest answer is an infinite interval, not a finite one that quietly fails
  # to cover.
  k <- ceiling((n + 1) * (1 - alpha))
  if (k > n) {
    warning("alpha = ", alpha, " needs at least ", ceiling(1 / alpha) - 1,
            " calibration points to be certifiable; there are ", n,
            ". The interval is infinite, which is the correct answer.",
            call. = FALSE)
    q <- Inf
  } else {
    q <- sort(score)[k]
  }

  structure(list(q = q, alpha = alpha, n = n, k = k,
                 normalised = !is.null(difficulty),
                 residuals = res, scores = score),
            class = "conformal_cal")
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
conformal_interval <- function(cal, pred, difficulty = NULL,
                               lower_limit = -Inf) {
  stopifnot(inherits(cal, "conformal_cal"))
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

#' @export
print.conformal_cal <- function(x, ...) {
  cat("\n<conformal_cal> ", sprintf("%.0f%% intervals", 100 * (1 - x$alpha)),
      if (x$normalised) " (normalised)" else " (constant width)", "\n", sep = "")
  cat(sprintf("  calibration points : %d\n", x$n))
  cat(sprintf("  rank used          : %d of %d  (the (n+1) correction)\n",
              x$k, x$n))
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
cv_residuals <- function(run_dir, config_id, role = "validation") {
  pred_dir <- file.path(run_dir, "predictions")
  files <- list.files(pred_dir,
                      pattern = sprintf("^%s_f[0-9]+_s[0-9]+_pred_all\.csv$",
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
