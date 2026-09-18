# Unit test: the three functions whose failure mode is "everything is fine"
#
# Every other test here covers something that BREAKS LOUDLY when it is wrong.
# These three do the opposite:
#
#   calc_metrics()         a wrong formula produces a plausible number, and it
#                          is the number every ranking is built on
#   spatial_overlap_report() under-reporting produces a clean 0%, which is
#                          exactly the answer that stops anyone looking
#   write/compare_run_snapshot()  a broken comparison says "everything
#                          identical", which is the result people want to see
#
# The pipeline already ASSERTS 0% identical patches per fold on the word of the
# second one, and had never verified it against a case with a known answer.
#
# Verified:
#   1. ccc() against hand-computed values, and against DescTools to 1e-12
#   2. every other metric against arithmetic done by hand
#   3. the degenerate cases: constant prediction, too few points, all NA
#   4. overlap counted on a layout whose answer is known by construction
#   5. the defect/context distinction survives (identical patch vs shared pixel)
#   6. snapshots round-trip, including the decimal that broke them once
#   7. a changed / new / removed value is each reported as what it is
#
# Run: source("D:/.../tests/test_reporting.R")    (CPU, no torch needed)

suppressMessages({
  library(tibble)
  library(dplyr)
  library(readr)
})

# -- project root: works under source() in the console AND under Rscript ------

root <- (function() {
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
  }
  cand <- c(cand, getwd())
  for (d in cand) {
    for (up in c(".", "..")) {
      r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
      if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
    }
  }
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "metrics.R"))
source(file.path(root, "R", "diagnostics.R"))

ok <- c()

# =============================================================================
# 1-3. calc_metrics()
# =============================================================================

obs  <- c(2, 4, 6,  8, 10, 12, 14, 16)
pred <- c(3, 3, 7,  7, 11, 11, 15, 15)
n    <- length(obs)

# -- hand-computed, with the population moments Lin (1989) defines ------------
mo <- mean(obs); mp <- mean(pred)
vo <- sum((obs - mo)^2) / n
vp <- sum((pred - mp)^2) / n
cv <- sum((obs - mo) * (pred - mp)) / n
ccc_hand <- 2 * cv / (vo + vp + (mo - mp)^2)

ok["ccc_matches_hand_computation"] <- isTRUE(all.equal(ccc(obs, pred), ccc_hand))
ok["ccc_is_symmetric"] <- isTRUE(all.equal(ccc(obs, pred), ccc(pred, obs)))
ok["ccc_perfect_agreement_is_one"] <- isTRUE(all.equal(ccc(obs, obs), 1))

# A model that predicts a CONSTANT agrees with nothing: variance of the
# prediction is zero, so the numerator is zero. Returning 0 -- not NA -- is the
# honest reading, and it is what stops a dead model from ranking as "missing".
ok["ccc_constant_prediction_is_zero"] <-
  isTRUE(all.equal(ccc(obs, rep(7, n)), 0))

# Perfect anti-correlation with the same spread gives -1.
ok["ccc_mirror_is_minus_one"] <- {
  m <- mean(obs)
  isTRUE(all.equal(ccc(obs, 2 * m - obs), -1))
}

ok["ccc_needs_two_points"] <- is.na(ccc(1, 1))
ok["ccc_all_na_is_na"]     <- is.na(ccc(rep(NA_real_, 5), rep(NA_real_, 5)))
ok["ccc_drops_na_pairs"]   <- isTRUE(all.equal(
  ccc(c(obs, NA), c(pred, 5)), ccc(obs, pred)))

# -- against DescTools, so this stays a reimplementation, not a variant -------
if (requireNamespace("DescTools", quietly = TRUE)) {
  ref <- as.numeric(DescTools::CCC(obs, pred, conf.level = 0.95)$rho.c$est)[1]
  ok["ccc_equals_DescTools"] <- abs(ccc(obs, pred) - ref) < 1e-12
  cat("  own ccc vs DescTools     : difference ",
      format(abs(ccc(obs, pred) - ref), scientific = TRUE, digits = 2),
      "\n", sep = "")
} else {
  cat("  DescTools absent -- comparison skipped (not a failure)\n")
}

# -- the rest of the metrics, each against arithmetic done by hand ------------
m <- calc_metrics(obs, pred)

ok["metrics_n"]    <- m$n == n
ok["metrics_ccc"]  <- isTRUE(all.equal(m$ccc, ccc_hand))
ok["metrics_mae"]  <- isTRUE(all.equal(m$mae, mean(abs(pred - obs))))
ok["metrics_rmse"] <- isTRUE(all.equal(m$rmse, sqrt(mean((pred - obs)^2))))
ok["metrics_r2"]   <- isTRUE(all.equal(m$r2, cor(obs, pred)^2))
ok["metrics_nse"]  <- isTRUE(all.equal(
  m$nse, 1 - sum((obs - pred)^2) / sum((obs - mo)^2)))
ok["metrics_rpd"]  <- isTRUE(all.equal(m$rpd, stats::sd(obs) / m$rmse))
ok["metrics_mqi"]  <- isTRUE(all.equal(m$mqi, (m$ccc * m$nse) / (m$mae / mo)))

# NSE is 0 for the mean-only model, by definition -- the anchor the whole
# metric is built around.
ok["nse_of_mean_model_is_zero"] <-
  isTRUE(all.equal(calc_metrics(obs, rep(mo, n))$nse, 0))

# Degenerate inputs give NA, not a number that looks usable.
ok["metrics_too_few_points"] <- is.na(calc_metrics(1, 1)$ccc)
ok["metrics_all_na"]         <- is.na(calc_metrics(rep(NA_real_, 5),
                                                   rep(NA_real_, 5))$rmse)

cat("  metrics checked          : CCC ", sprintf("%.4f", m$ccc),
    " | MAE ", sprintf("%.3f", m$mae),
    " | NSE ", sprintf("%.4f", m$nse),
    " | RPD ", sprintf("%.3f", m$rpd), "\n", sep = "")

# =============================================================================
# 4-5. spatial_overlap_report()
# =============================================================================
#
# A layout whose answer is known by construction:
#   - 10 training points on distinct cells, 100 apart
#   - 3 validation points on EXACTLY three of those cells  -> identical patch
#   - 2 validation points 2 cells away                     -> shares pixels only
#   - 5 validation points 10,000 cells away                -> neither

tr_r <- seq(0, 900, by = 100)
tr_c <- rep(0L, 10)

va_r <- c(tr_r[1:3],            # same cell as training
          tr_r[5] + 2, tr_r[6] + 2,   # two cells away
          rep(10000, 5))              # far away
va_c <- c(rep(0L, 5), rep(0L, 5))

rid  <- c(tr_r, va_r)
cid  <- c(tr_c, va_c)
role <- c(rep("train", 10), rep("validation", 10))

ov <- spatial_overlap_report(rid, cid, role, windows = c(3L, 9L))

same <- dplyr::filter(ov, matters)
ok["overlap_counts_identical_exactly"] <- same$n == 3L
ok["overlap_pct_is_right"]             <- isTRUE(all.equal(same$pct, 30))
ok["overlap_n_split_is_right"]         <- same$n_split == 10L

# 3x3 patches at a distance of 2 cells DO share pixels; at 100 they do not.
sh3 <- dplyr::filter(ov, window == 3L)
ok["overlap_3x3_counts_neighbours"] <- sh3$n == 5L   # 3 same-cell + 2 nearby

# 9x9 reaches 8 cells, and nothing here sits between 3 and 8 cells from a
# training point, so the count must NOT move. A purely bucket-based
# approximation grows here, because a wider bucket sweeps in points that are
# not actually within reach -- which is why the candidates get an exact test.
sh9 <- dplyr::filter(ov, window == 9L)
ok["overlap_9x9_does_not_inflate"] <- sh9$n == 5L

# And the far group must never be counted, at any window: 10,000 cells away is
# 10,000 cells away.
ok["overlap_ignores_distant_points"] <- all(dplyr::filter(ov, !matters)$n == 5L)

# THE DISTINCTION. Identical patches are a defect under any plan. Shared pixels
# between neighbours are NOT -- under a random split they are the condition
# being measured. Conflating them turns a legitimate random split into a
# reported failure.
ok["only_identical_patch_matters"] <-
  all(same$matters) && !any(dplyr::filter(ov, !matters)$matters)
ok["report_names_both_criteria"] <-
  any(grepl("identical patch", ov$criterion)) &&
  any(grepl("shares pixels",   ov$criterion))

# No training points at all in the reference split is a caller error, not a 0%.
ok["overlap_needs_a_reference"] <- inherits(
  try(spatial_overlap_report(rid, cid, rep("validation", 20)), silent = TRUE),
  "try-error")

cat("  overlap checked          : ", same$n, " of ", same$n_split,
    " in the same pixel (expected 3 of 10)\n", sep = "")

# =============================================================================
# 6-7. run snapshots
# =============================================================================

snap_dir <- file.path(tempdir(), "dlc_snap_test")
unlink(snap_dir, recursive = TRUE)

# The value that broke this once: as.character(31.190645) writes a decimal
# POINT, and read_csv2() under a comma-decimal locale reads that point as a
# THOUSANDS separator, returning 31190645. Four values showed as "changed" when
# nothing had changed -- the exact noise the mechanism exists to remove.
vals <- list(ccc = 31.190645, pct = 0.28, n = 36697L, label = "soc_0_5cm")

f1 <- write_run_snapshot(vals, snap_dir, label = "aaa")
ok["snapshot_file_written"] <- file.exists(f1)

cmp <- compare_run_snapshot(vals, snap_dir)
ok["snapshot_has_previous"]   <- isTRUE(cmp$has_previous)
ok["snapshot_all_identical"]  <- all(cmp$diff$status == "=")
ok["snapshot_decimal_survives"] <- {
  row <- dplyr::filter(cmp$diff, key == "ccc")
  identical(row$old, row$new) && grepl("31.19", row$old, fixed = TRUE)
}

# One value changed, one added, one removed -- each reported as what it is.
vals2 <- list(ccc = 31.190645, pct = 0.31, n = 36697L, extra = 1)
cmp2  <- compare_run_snapshot(vals2, snap_dir)
st    <- setNames(cmp2$diff$status, cmp2$diff$key)

ok["snapshot_detects_change"]   <- identical(unname(st["pct"]),   "changed")
ok["snapshot_detects_new"]      <- identical(unname(st["extra"]), "new")
ok["snapshot_detects_removed"]  <- identical(unname(st["label"]), "gone")
ok["snapshot_leaves_rest_alone"] <-
  identical(unname(st[c("ccc", "n")]), c("=", "="))

# With no previous snapshot there is nothing to compare against, and saying so
# is different from saying nothing changed.
empty_dir <- file.path(tempdir(), "dlc_snap_empty")
unlink(empty_dir, recursive = TRUE)
dir.create(empty_dir, recursive = TRUE, showWarnings = FALSE)
cmp3 <- compare_run_snapshot(vals, empty_dir)
ok["snapshot_no_previous_is_not_identical"] <-
  !cmp3$has_previous && all(cmp3$diff$status == "new")

unlink(snap_dir,  recursive = TRUE)
unlink(empty_dir, recursive = TRUE)

cat("  snapshot checked         : 31.190645 survives the write",
    " and read cycle\n", sep = "")


# =============================================================================
# THE SIGNED BIAS
#
# Every other metric here is blind to the sign of the error, and that blindness
# cost this project a real result: the final model under-predicted the test set
# by 9.61 t/ha -- 24.4% of the observed mean, and 53% OF THE MAE -- while CCC,
# MAE, RMSE, R2, NSE, RPD and MQI all reported numbers nobody questioned. A map
# summed for a total carbon stock would have been a quarter light.
#
# The cause is a modelling choice rather than a bug: a SmoothL1 loss on log1p
# estimates a conditional MEDIAN in log space, expm1() of which is the median of
# the stock and not its mean, and the target is right-skewed. That is worth
# knowing; what is not acceptable is that no metric could show it.
#
# So the property tested is not "the number is computed" but "a constant offset
# is visible in bias and invisible in the unsigned metrics".
# =============================================================================

bias_obs  <- c(10, 20, 30, 40, 50)
bias_pred <- bias_obs - 6                      # every prediction 6 units low
bm <- calc_metrics(bias_obs, bias_pred)

results["bias_is_signed_and_exact"] <- isTRUE(all.equal(bm$bias, -6))
results["bias_pct_is_relative_to_obs"] <-
  isTRUE(all.equal(bm$bias_pct, 100 * -6 / mean(bias_obs)))

# THE ASYMMETRY THAT MATTERS: over-prediction must come back with the other
# sign. A bias reported as abs() would be no better than MAE.
results["bias_sign_flips_with_direction"] <-
  isTRUE(all.equal(calc_metrics(bias_obs, bias_obs + 6)$bias, 6))

# ...and MAE cannot tell those two apart, which is the whole point.
results["mae_cannot_tell_the_direction"] <-
  isTRUE(all.equal(calc_metrics(bias_obs, bias_obs - 6)$mae,
                   calc_metrics(bias_obs, bias_obs + 6)$mae))

results["unbiased_predictions_give_zero_bias"] <-
  abs(calc_metrics(bias_obs, bias_obs + c(-2, 2, -2, 2, 0))$bias) < 1e-12

# The NA row must carry the same columns, or bind_rows() across a failed unit
# and a successful one invents columns and the comparison table changes shape.
results["na_row_has_the_bias_columns"] <- {
  z <- calc_metrics(c(1), c(1))          # n < 2 -> the NA branch
  all(c("bias", "bias_pct") %in% names(z)) && is.na(z$bias)
}

# And the real case, reproduced from the numbers on disk: a right-skewed target
# predicted at its conditional median.
set.seed(11)
skew_obs  <- exp(stats::rnorm(4000, log(30), 0.55))          # mean > median
skew_pred <- rep(stats::median(skew_obs), length(skew_obs))  # a perfect median
sm <- calc_metrics(skew_obs, skew_pred)
results["a_perfect_median_is_a_biased_mean"] <- sm$bias < -1
results["and_the_skew_is_what_causes_it"] <-
  mean(skew_obs) / stats::median(skew_obs) > 1.1

cat(sprintf("  signed bias              : a -6 offset reads %.1f (MAE reads %.1f either way)
",
            bm$bias, bm$mae))

.report(ok, "test_metrics_reporting")
