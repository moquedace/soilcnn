# Unit test: the interval covers what it promises
#
# WHY THIS FILE EXISTS.
#
# An uncertainty map is the one output nobody can check by eye. A prediction map
# that is wrong looks wrong next to a soil map; an uncertainty map that promises
# 90% and delivers 61% looks exactly like one that delivers 90%. So the coverage
# property has to be tested by simulation, against a distribution whose truth is
# known, rather than trusted because the formula was copied correctly.
#
# What is checked:
#   1. coverage, empirically, over many repetitions -- the guarantee itself
#   2. the (n+1) correction actually changes the answer, in the safe direction
#   3. too few calibration points gives an INFINITE interval, not a false one
#   4. normalised intervals vary in width and still cover
#   5. the marginal guarantee can hide a broken group, and picp_report shows it
#   6. every way of mixing normalised and constant calibration is refused
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_conformal.R")

suppressMessages({
  library(tibble)
  library(dplyr)
})

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
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
.load_framework(root)

ok <- logical(0)

# ── 1. THE GUARANTEE, measured ───────────────────────────────────────────────
#
# Coverage is a property of repetitions, not of one run, so it is measured over
# many. The expected rate is not 1 - alpha exactly but k/(n+1), which for
# n = 50 and alpha = 0.1 is 46/51 = 0.902 -- slightly conservative, by
# construction. Asserting against 0.902 rather than 0.9 is the difference
# between testing the theorem and testing a rounded memory of it.

set.seed(101)
n_cal <- 50L; n_new <- 200L; alpha <- 0.1
reps <- 300L
cov_rate <- vapply(seq_len(reps), function(i) {
  cal_obs  <- rnorm(n_cal, 30, 8)
  cal_pred <- cal_obs + rnorm(n_cal, 0, 5)      # an unbiased, noisy model
  cal <- conformal_calibrate(cal_obs, cal_pred, alpha = alpha)
  new_obs  <- rnorm(n_new, 30, 8)
  new_pred <- new_obs + rnorm(n_new, 0, 5)
  iv <- conformal_interval(cal, new_pred)
  mean(new_obs >= iv$lower & new_obs <= iv$upper)
}, numeric(1))

expected <- ceiling((n_cal + 1) * (1 - alpha)) / (n_cal + 1)
ok["coverage_matches_the_theorem"] <- abs(mean(cov_rate) - expected) < 0.01
ok["coverage_is_never_below_nominal_on_average"] <- mean(cov_rate) >= 1 - alpha

# A SKEWED, HEAVY-TAILED error must not change the answer. That is the whole
# selling point over a normal-theory interval: no distributional assumption.
set.seed(202)
cov_skew <- vapply(seq_len(reps), function(i) {
  cal_obs  <- rnorm(n_cal, 30, 8)
  cal_pred <- cal_obs + rexp(n_cal, rate = 0.2) - 5
  cal <- conformal_calibrate(cal_obs, cal_pred, alpha = alpha)
  new_obs  <- rnorm(n_new, 30, 8)
  new_pred <- new_obs + rexp(n_new, rate = 0.2) - 5
  iv <- conformal_interval(cal, new_pred)
  mean(new_obs >= iv$lower & new_obs <= iv$upper)
}, numeric(1))
ok["coverage_survives_a_skewed_error"] <- abs(mean(cov_skew) - expected) < 0.015

# ── 2. the (n+1) correction ──────────────────────────────────────────────────
#
# The plain sample quantile is the mistake that makes the guarantee false by
# about 1/n, in the optimistic direction. It has to be visibly different, and
# conformal's q has to be the LARGER of the two.

set.seed(303)
o <- rnorm(20, 10, 3); p <- o + rnorm(20, 0, 2)
cal20 <- conformal_calibrate(o, p, alpha = 0.1)
plain <- stats::quantile(abs(o - p), 0.9, names = FALSE)
ok["correction_uses_rank_n_plus_1"] <- cal20$k == ceiling(21 * 0.9)
ok["correction_is_not_the_plain_quantile"] <- cal20$q != plain
ok["correction_errs_on_the_wide_side"]     <- cal20$q >= plain

# ── 3. too few points is an infinite interval, not a false one ───────────────
#
# For 90% the threshold is ceiling(1/alpha) - 1 = 9 points. With 8 there is no
# finite interval that can carry the guarantee, and inventing one is the failure
# this branch exists to avoid.

ok["too_few_points_warns"] <- {
  w <- NULL
  withCallingHandlers(
    conformal_calibrate(rnorm(8), rnorm(8), alpha = 0.1),
    warning = function(cond) { w <<- conditionMessage(cond)
                               invokeRestart("muffleWarning") })
  is.character(w) && grepl("certifiable", w)
}
ok["too_few_points_gives_infinite_q"] <-
  is.infinite(suppressWarnings(
    conformal_calibrate(rnorm(8), rnorm(8), alpha = 0.1)$q))
# ...and exactly at the threshold it must NOT warn, or the boundary is off by
# one in the other direction and everyone gets infinite intervals.
ok["the_threshold_itself_is_fine"] <- {
  w <- NULL
  cc <- withCallingHandlers(
    conformal_calibrate(rnorm(9), rnorm(9), alpha = 0.1),
    warning = function(cond) { w <<- conditionMessage(cond)
                               invokeRestart("muffleWarning") })
  is.null(w) && is.finite(cc$q)
}
ok["two_points_is_not_enough_to_be_a_quantile"] <- inherits(
  try(conformal_calibrate(1, 1, alpha = 0.1), silent = TRUE), "try-error")

# ── 4. normalised intervals ──────────────────────────────────────────────────
#
# Heteroscedastic by construction: the noise grows with a known difficulty
# score. A constant-width interval then covers on average while covering badly
# at both ends -- too wide where the model is sure, too narrow where it is not.
# The normalised interval is the fix, and must keep the guarantee.

set.seed(404)
make_het <- function(n) {
  diff <- runif(n, 0.5, 4)
  obs  <- rnorm(n, 30, 8)
  pred <- obs + rnorm(n, 0, 3 * diff)
  tibble(obs = obs, pred = pred, diff = diff)
}
cal_h <- make_het(400L); new_h <- make_het(2000L)

cal_const <- conformal_calibrate(cal_h$obs, cal_h$pred, alpha = 0.1)
cal_norm  <- conformal_calibrate(cal_h$obs, cal_h$pred, alpha = 0.1,
                                 difficulty = cal_h$diff)
iv_const <- conformal_interval(cal_const, new_h$pred)
iv_norm  <- conformal_interval(cal_norm, new_h$pred, difficulty = new_h$diff)

ok["normalised_is_flagged_as_such"]   <- isTRUE(cal_norm$normalised)
ok["constant_width_is_constant"]      <- dplyr::n_distinct(round(iv_const$width, 9)) == 1L
ok["normalised_width_varies"]          <- dplyr::n_distinct(round(iv_norm$width, 6)) > 100L
ok["normalised_width_tracks_difficulty"] <-
  cor(iv_norm$width, new_h$diff) > 0.99

pc <- picp(new_h$obs, iv_const$lower, iv_const$upper)
pn <- picp(new_h$obs, iv_norm$lower, iv_norm$upper)
ok["both_cover_marginally"] <- pc$picp > 0.87 && pn$picp > 0.87

# THE POINT OF NORMALISING: conditional coverage. Split the new points into an
# easy and a hard half by difficulty; the constant interval must be visibly
# uneven across them and the normalised one much less so.
half <- new_h$diff > stats::median(new_h$diff)
gap_of <- function(iv) {
  a <- mean(new_h$obs[!half] >= iv$lower[!half] & new_h$obs[!half] <= iv$upper[!half])
  b <- mean(new_h$obs[half]  >= iv$lower[half]  & new_h$obs[half]  <= iv$upper[half])
  abs(a - b)
}
ok["constant_width_covers_unevenly"] <- gap_of(iv_const) > 0.10
ok["normalising_evens_the_coverage"] <- gap_of(iv_norm) < gap_of(iv_const) / 2

# ── 5. the marginal guarantee can hide a broken group ────────────────────────
#
# This is the scientific point of picp_report(), not a convenience: "90%
# overall" is compatible with 99% over the easy half and 60% over the hard half,
# and the hard half is the part anyone needs an interval for.

grp <- ifelse(half, "hard", "easy")
rep_const <- picp_report(new_h$obs, iv_const$lower, iv_const$upper,
                         group = grp, alpha = 0.1)
ok["report_gives_overall_coverage"] <- rep_const$overall$picp > 0.8
ok["report_breaks_down_by_group"]   <- nrow(rep_const$by_group) == 2L
ok["report_sorts_worst_group_first"] <-
  rep_const$by_group$picp[1] <= rep_const$by_group$picp[2]
ok["the_worst_group_is_the_hard_one"] <- rep_const$by_group$group[1] == "hard"
ok["marginal_hid_the_group_failure"] <-
  rep_const$overall$picp - rep_const$by_group$picp[1] > 0.08

# ── 6. refusals ──────────────────────────────────────────────────────────────

ok["normalised_cal_without_difficulty_is_refused"] <- inherits(
  try(conformal_interval(cal_norm, new_h$pred), silent = TRUE), "try-error")
ok["constant_cal_with_difficulty_is_refused"] <- inherits(
  try(conformal_interval(cal_const, new_h$pred, difficulty = new_h$diff),
      silent = TRUE), "try-error")
ok["alpha_outside_0_1_is_refused"] <- inherits(
  try(conformal_calibrate(o, p, alpha = 1.5), silent = TRUE), "try-error")

# A lower limit can only raise coverage, so it must never lower it -- and it
# must actually bite on a stock, where the interval would otherwise go negative.
iv_floor <- conformal_interval(cal_const, new_h$pred, lower_limit = 0)
ok["lower_limit_is_applied"]        <- all(iv_floor$lower >= 0)
ok["lower_limit_cannot_lose_cover"] <-
  picp(new_h$obs, iv_floor$lower, iv_floor$upper)$picp >= pc$picp

# ── 7. conformal_cv ──────────────────────────────────────────────────────────

cv_tbl <- bind_rows(lapply(1:3, function(f) {
  d <- make_het(300L); d$fold <- f; d
}))
rep_cv <- conformal_cv(cv_tbl, alpha = 0.1)
ok["cv_calibrates_out_of_fold"] <- rep_cv$overall$n == nrow(cv_tbl)
ok["cv_coverage_is_near_nominal"] <- abs(rep_cv$overall$picp - 0.9) < 0.04
ok["cv_normalised_runs"] <- {
  r <- conformal_cv(cv_tbl, alpha = 0.1, difficulty = "diff")
  abs(r$overall$picp - 0.9) < 0.04
}
# One fold means calibrating and measuring on the same points, where the
# coverage comes out right by arithmetic rather than by evidence.
ok["cv_refuses_a_single_fold"] <- inherits(
  try(conformal_cv(dplyr::filter(cv_tbl, fold == 1), alpha = 0.1),
      silent = TRUE), "try-error")
ok["cv_needs_its_columns"] <- inherits(
  try(conformal_cv(dplyr::select(cv_tbl, -obs)), silent = TRUE), "try-error")

# ── 6. a fitted scale: the level and the dissimilarity ───────────────────────
#
# The error grows with the stock AND with the distance from the training data,
# by construction -- the two things the map's interval is meant to follow. A
# constant width covers on average and badly at both ends of either axis; a
# scale fitted on the level alone evens out one axis; level + DI evens out
# both. Every one of them must keep the marginal guarantee, because the scale
# is fitted on one half of the calibration points and q is taken on the other.
set.seed(505)
make_ld <- function(n) {
  level <- runif(n, 5, 150)          # the predicted stock, t/ha
  di    <- rexp(n, rate = 3)         # mostly near the training data, a long tail
  obs   <- level + rnorm(n, 0, 2 + 0.15 * level + 25 * di)
  tibble(obs = obs, pred = level, level = level, di = di)
}
cal_ld <- make_ld(2000L); new_ld <- make_ld(20000L)

cs_const <- conformal_calibrate(cal_ld$obs, cal_ld$pred, alpha = 0.1)
cs_level <- conformal_scaled_calibrate(cal_ld$obs, cal_ld$pred, cal_ld[, "level"], alpha = 0.1)
cs_ldi   <- conformal_scaled_calibrate(cal_ld$obs, cal_ld$pred, cal_ld[, c("level", "di")],
                                       alpha = 0.1)
iv_c  <- conformal_interval(cs_const, new_ld$pred)
iv_l  <- conformal_scaled_interval(cs_level, new_ld$pred, new_ld[, "level"])
iv_ld <- conformal_scaled_interval(cs_ldi, new_ld$pred, new_ld[, c("level", "di")])
cover_by <- function(iv, g) tapply(new_ld$obs >= iv$lower & new_ld$obs <= iv$upper, g, mean)
fifths   <- function(v) cut(v, stats::quantile(v, 0:5 / 5), include.lowest = TRUE)
spread   <- function(v) max(v) - min(v)
by_di <- fifths(new_ld$di); by_lv <- fifths(new_ld$level)

ok["a_fitted_scale_keeps_the_marginal_coverage"] <-
  abs(picp(new_ld$obs, iv_l$lower, iv_l$upper)$picp - 0.9) < 0.03 &&
  abs(picp(new_ld$obs, iv_ld$lower, iv_ld$upper)$picp - 0.9) < 0.03
ok["the_scale_is_fitted_on_one_half_and_q_taken_on_the_other"] <-
  cs_ldi$n_fit == 1000L && cs_ldi$n == 1000L
ok["the_fit_finds_both_covariates"] <- cs_ldi$coef[["level"]] > 0 && cs_ldi$coef[["di"]] > 0
ok["level_and_di_even_out_coverage_along_the_di"] <-
  spread(cover_by(iv_ld, by_di)) < 0.5 * spread(cover_by(iv_c, by_di))
ok["and_along_the_level"] <-
  spread(cover_by(iv_ld, by_lv)) < 0.5 * spread(cover_by(iv_c, by_lv))
ok["the_level_alone_leaves_the_di_uneven"] <-
  spread(cover_by(iv_l, by_di)) > spread(cover_by(iv_ld, by_di))

# The guarantee is a finite-sample one: at n = 60 (30 to fit, 30 to calibrate)
# the coverage averaged over many draws must still reach the nominal level.
cov_small <- replicate(200L, {
  cal <- make_ld(60L); new <- make_ld(500L)
  cs <- conformal_scaled_calibrate(cal$obs, cal$pred, cal[, c("level", "di")], alpha = 0.1)
  iv <- conformal_scaled_interval(cs, new$pred, new[, c("level", "di")])
  mean(new$obs >= iv$lower & new$obs <= iv$upper)
})
ok["the_guarantee_holds_on_average_at_a_small_n"] <- mean(cov_small) >= 0.9 - 0.01

edge_points <- data.frame(level = c(-1e6, 0), di = c(-1e6, 0))
ok["the_scale_never_goes_below_its_floor"] <-
  all(.conformal_scale(cs_ldi$coef, cs_ldi$floor, edge_points) >= cs_ldi$floor)
ok["too_few_points_for_a_fitted_scale_are_refused"] <- inherits(
  try(conformal_scaled_calibrate(cal_ld$obs[1:10], cal_ld$pred[1:10], cal_ld[1:10, "level"]),
      silent = TRUE), "try-error")
ok["the_constant_interval_refuses_a_fitted_scale"] <- inherits(
  try(conformal_interval(cs_ldi, new_ld$pred), silent = TRUE), "try-error")
ok["a_map_without_the_di_is_refused"] <- inherits(
  try(conformal_scaled_interval(cs_ldi, new_ld$pred, new_ld[, "level"]), silent = TRUE),
  "try-error")

cat(sprintf("  coverage by DI fifth     : constant %s | level %s | level+DI %s\n",
            paste(sprintf("%.2f", cover_by(iv_c, by_di)), collapse = "/"),
            paste(sprintf("%.2f", cover_by(iv_l, by_di)), collapse = "/"),
            paste(sprintf("%.2f", cover_by(iv_ld, by_di)), collapse = "/")))
cat(sprintf("  coverage over %d reps    : %.4f (theorem says %.4f)\n",
            reps, mean(cov_rate), expected))
cat(sprintf("  skewed error             : %.4f\n", mean(cov_skew)))
cat(sprintf("  easy vs hard gap         : constant %.3f | normalised %.3f\n",
            gap_of(iv_const), gap_of(iv_norm)))

.report(ok, "test_conformal")
