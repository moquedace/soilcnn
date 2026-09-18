# Unit test: the back-transform recovers the mean without breaking the median
#
# WHY THIS FILE EXISTS.
#
# The final model under-predicted the frozen test set by 24.4% -- more than half
# its MAE -- because expm1() of a conditional median is a median, and the target
# is right-skewed. Duan's smearing estimator corrects it with one scalar.
#
# A correction factor is a number that multiplies a whole map. If it is wrong in
# the second decimal, every carbon total computed from that map is wrong by the
# same proportion, and nothing in the map looks unusual. So the tests here are
# built on distributions whose true conditional mean is known in closed form,
# rather than on the numbers this project happened to measure.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_smearing.R")

suppressMessages({
  library(tibble)
  library(dplyr)
  library(readr)
  library(purrr)
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
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "smearing.R"))

ok <- logical(0)

# ── 1. THE CLOSED FORM ────────────────────────────────────────────────────────
#
# Take y lognormal: log(y) = mu + e, e ~ N(0, sigma). Then the conditional
# median is exp(mu) and the conditional mean is exp(mu + sigma^2/2). A model
# that predicts mu exactly is a perfect median predictor and a biased mean
# predictor, by a factor of exactly exp(sigma^2/2) -- which is what the smearing
# factor must recover.
#
# This is the whole claim, tested against arithmetic rather than against the
# project's own numbers.

set.seed(101)
n     <- 20000L
mu    <- log(30)
sigma <- 0.55
e     <- stats::rnorm(n, 0, sigma)
y     <- exp(mu + e)                       # native, right-skewed
z     <- log(y)                            # transform space
zhat  <- rep(mu, n)                        # a PERFECT median predictor

cal <- smearing_factor(z, zhat, transform = "log")
expected_s <- exp(sigma^2 / 2)

ok["s_recovers_the_closed_form"] <- abs(cal$s - expected_s) < 0.02
ok["the_naive_back_transform_is_the_median"] <-
  abs(exp(mu) - stats::median(y)) / stats::median(y) < 0.02
ok["the_naive_back_transform_is_NOT_the_mean"] <-
  (mean(y) - exp(mu)) / mean(y) > 0.10

# and the corrected surface hits the mean
ok["smear_recovers_the_mean"] <-
  abs(mean(smear(zhat, cal, lower_limit = -Inf)) - mean(y)) / mean(y) < 0.02

# ── 2. log1p, WHICH IS WHAT THIS PROJECT ACTUALLY USES ───────────────────────
#
# The algebra differs: E[y] = exp(f(x)) * S - 1, and folding the -1 in before
# the scaling would scale the offset too. The error is small at small z and
# grows everywhere else, which is the worst shape a bug can have.

set.seed(202)
y1    <- expm1(stats::rnorm(n, log1p(30), 0.55))
y1    <- y1[y1 > 0]
z1    <- log1p(y1)
zhat1 <- rep(stats::median(z1), length(z1))
cal1  <- smearing_factor(z1, zhat1, transform = "log1p")

ok["log1p_smear_recovers_the_mean"] <-
  abs(mean(smear(zhat1, cal1, lower_limit = -Inf)) - mean(y1)) / mean(y1) < 0.03

# THE ORDER OF OPERATIONS, made explicit. expm1(z) * S is the wrong formula and
# is close enough to the right one to pass a casual eye.
ok["smear_is_not_expm1_times_s"] <- {
  right <- exp(zhat1[1]) * cal1$s - 1
  wrong <- expm1(zhat1[1]) * cal1$s
  abs(right - wrong) > 0.1 &&
    abs(smear(zhat1[1], cal1, lower_limit = -Inf) - right) < 1e-9
}

# ── 3. AN UNBIASED MODEL MUST NOT BE "CORRECTED" ─────────────────────────────
#
# If the residuals are symmetric and tiny, S is ~1 and the surface barely moves.
# A correction that always inflates would trade a known bias for an unknown one.
set.seed(303)
z2    <- stats::rnorm(5000, log1p(30), 0.02)
zhat2 <- z2 - stats::rnorm(5000, 0, 0.02)
cal2  <- smearing_factor(z2, zhat2)
ok["a_nearly_perfect_model_gets_a_factor_near_one"] <- abs(cal2$s - 1) < 0.01

# ── 4. THE LOGNORMAL CROSS-CHECK IS REPORTED, NOT ASSUMED ────────────────────
ok["lognormal_check_agrees_on_lognormal_residuals"] <-
  abs(cal$agreement - 1) < 0.02

# ...and must DISAGREE when the residuals are not lognormal, or it is not a
# check. A heavy discrete right tail is the case that matters, because that is
# what a few very high-carbon profiles look like.
set.seed(404)
e3   <- c(stats::rnorm(4900, 0, 0.1), stats::rnorm(100, 3, 0.1))
cal3 <- smearing_factor(e3 + 5, rep(5, length(e3)))
ok["lognormal_check_disagrees_on_a_heavy_tail"] <- abs(cal3$agreement - 1) > 0.10

# ── 5. REFUSALS ──────────────────────────────────────────────────────────────

# Too few residuals: a mean of exp() over a handful with a heavy tail is noise.
ok["too_few_residuals_is_refused"] <- inherits(
  try(smearing_factor(stats::rnorm(10), stats::rnorm(10)), silent = TRUE),
  "try-error")

# An unsupported transform must be refused, not assumed. The algebra is specific
# to an exponential back-transform; applying it to a square root would scale a
# number that needs no scaling, silently.
ok["an_unknown_transform_is_refused"] <- inherits(
  try(smearing_factor(z1, zhat1, transform = "sqrt"), silent = TRUE),
  "try-error")

# Native-space columns passed by mistake: exp() of a stock of 173 overflows to
# something absurd rather than quietly returning a plausible factor.
ok["native_columns_do_not_pass_silently"] <- {
  big <- smearing_factor(stats::runif(500, 1, 170), stats::runif(500, 1, 170))
  !is.finite(big$s) || big$s > 100
}

ok["lower_limit_is_applied"] <-
  all(smear(rep(-5, 10), cal1, lower_limit = 0) == 0)

# ── 6. THE TRADE-OFF IS REAL, AND THE TEST SAYS SO ───────────────────────────
#
# Correcting toward the mean must IMPROVE squared error and WORSEN absolute
# error. If a change improved both, it would not be a median-to-mean move and
# something else would be going on.
set.seed(505)
yt    <- expm1(stats::rnorm(8000, log1p(30), 0.6))
yt    <- yt[yt > 0]
zt    <- log1p(yt)
zhatt <- rep(stats::median(zt), length(zt))
calt  <- smearing_factor(zt, zhatt)
naive <- expm1(zhatt)
corr  <- smear(zhatt, calt, lower_limit = 0)
mae   <- function(p) mean(abs(yt - p))
rmse  <- function(p) sqrt(mean((yt - p)^2))

ok["correcting_improves_rmse"] <- rmse(corr) < rmse(naive)
ok["correcting_worsens_mae"]   <- mae(corr)  > mae(naive)
ok["correcting_removes_the_bias"] <-
  abs(mean(corr) - mean(yt)) < abs(mean(naive) - mean(yt)) / 3

# ── 7. THE RESIDUAL MUST BE THE ENSEMBLE'S, NOT ONE MEMBER'S ─────────────────
#
# The deployed prediction is the median over seeds, so the calibrated residual
# has to be the median's. smearing_from_run() used to read the per-seed rows --
# 3 rows per point -- while documenting itself as using the same residuals as
# the conformal interval, which collapses them. Two failures came with that:
#
#   n overstated the evidence by the number of seeds, in the one line a reader
#   uses to judge the factor; and exp() being convex, mean(exp(e)) grows with
#   var(e), so a noisier single-member residual inflates S.
#
# Built as a real run directory, because the defect was in the reading, and a
# fixture that hands the function a tidy data frame would test the fixture.

set.seed(606)
run_dir <- file.path(tempdir(), "smear_run_fixture")
pred_dir <- file.path(run_dir, "predictions")
dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)

n_pt <- 400L
truth <- stats::rnorm(n_pt, log1p(30), 0.5)
obs_t <- truth + stats::rnorm(n_pt, 0, 0.45)      # irreducible error
for (s in 1:3) {
  # each seed is the truth plus its OWN noise, so the median over the three is
  # a less noisy predictor than any one of them -- the whole point
  readr::write_csv2(
    tibble::tibble(sample_id = seq_len(n_pt),
                   dataset_role = "validation",
                   obs_transform = obs_t,
                   pred_transform = truth + stats::rnorm(n_pt, 0, 0.30)),
    file.path(pred_dir, sprintf("cfg_t_f1_s%d_pred_all.csv", s)))
}

cal_run <- smearing_from_run(run_dir, "cfg_t")

ok["from_run_returns_one_row_per_point"] <- identical(cal_run$n, n_pt)

# and the pooled-rows factor it replaced is LARGER, by convexity
pooled <- purrr::map_dfr(
  list.files(pred_dir, full.names = TRUE),
  ~ suppressMessages(readr::read_csv2(.x, show_col_types = FALSE)))
cal_pooled <- smearing_factor(pooled$obs_transform, pooled$pred_transform)

ok["pooling_seeds_inflates_the_factor"] <-
  cal_pooled$n == 3L * n_pt && cal_pooled$s > cal_run$s

# the inflation is the variance difference, not something else: for near-normal
# residuals the ratio is exp((var_pooled - var_ens) / 2)
ok["the_inflation_is_the_variance_difference"] <- {
  predicted <- exp((cal_pooled$sd_residual^2 - cal_run$sd_residual^2) / 2) *
    exp(cal_pooled$mean_residual - cal_run$mean_residual)
  abs(predicted - cal_pooled$s / cal_run$s) < 0.01
}

# ── 8. THE INDEPENDENCE ASSUMPTION IS REPORTED ───────────────────────────────
#
# Duan needs e independent of x, and one scalar is only defensible while that
# holds. On this project it does not, so the diagnostic has to actually fire --
# and, just as importantly, has to stay quiet when the assumption holds, or it
# is an alarm nobody reads.

set.seed(707)
p_ind <- stats::runif(2000, 2, 5)
e_ind <- stats::rnorm(2000, 0, 0.4)               # independent of p by design
cal_ind <- smearing_factor(p_ind + e_ind, p_ind)
ok["independent_residuals_give_a_flat_profile"] <-
  length(cal_ind$s_by_bin) == 5L &&
  max(cal_ind$s_by_bin) / min(cal_ind$s_by_bin) < 1.25

# ...and the sum-unbiasing factor coincides with the plain one when it holds,
# because the weights then carry no information
ok["weighted_equals_plain_when_independent"] <-
  abs(cal_ind$s_weighted / cal_ind$s - 1) < 0.05

set.seed(808)
p_dep <- stats::runif(2000, 2, 5)
e_dep <- stats::rnorm(2000, 0, 0.4) - 0.35 * (p_dep - 3.5)   # e shrinks with p
cal_dep <- smearing_factor(p_dep + e_dep, p_dep)
ok["dependent_residuals_are_detected"] <-
  max(cal_dep$s_by_bin) / min(cal_dep$s_by_bin) > 1.25
ok["the_profile_is_monotone_in_the_right_direction"] <-
  cal_dep$s_by_bin[1] > cal_dep$s_by_bin[5]

# the weighted factor must then DIFFER, since it is the one that unbiases a sum
# and the high-prediction points it weights carry a different S
ok["weighted_differs_when_dependent"] <-
  abs(cal_dep$s_weighted / cal_dep$s - 1) > 0.05

# a small calibration set gets no profile rather than a noisy one: five bins of
# 20 points each would produce a spread out of nothing and fire the warning
ok["a_small_set_reports_no_profile"] <- {
  set.seed(909)
  small_p <- stats::runif(100, 2, 5)
  length(smearing_factor(small_p + stats::rnorm(100, 0, 0.4),
                         small_p)$s_by_bin) == 0L
}

cat(sprintf("  closed form              : S = %.4f, exp(sigma^2/2) = %.4f\n",
            cal$s, expected_s))
cat(sprintf("  ensemble vs pooled rows  : n %d vs %d | S %.4f vs %.4f\n",
            cal_run$n, cal_pooled$n, cal_run$s, cal_pooled$s))
cat(sprintf("  independence check       : flat %.2fx | dependent %.2fx (fires above 1.25x)\n",
            max(cal_ind$s_by_bin) / min(cal_ind$s_by_bin),
            max(cal_dep$s_by_bin) / min(cal_dep$s_by_bin)))
cat(sprintf("  trade-off                : MAE %.2f -> %.2f | RMSE %.2f -> %.2f | bias %+.2f -> %+.2f\n",
            mae(naive), mae(corr), rmse(naive), rmse(corr),
            mean(naive) - mean(yt), mean(corr) - mean(yt)))

.report(ok, "test_smearing")
