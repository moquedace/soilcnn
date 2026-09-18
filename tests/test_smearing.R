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

cat(sprintf("  closed form              : S = %.4f, exp(sigma^2/2) = %.4f\n",
            cal$s, expected_s))
cat(sprintf("  trade-off                : MAE %.2f -> %.2f | RMSE %.2f -> %.2f | bias %+.2f -> %+.2f\n",
            mae(naive), mae(corr), rmse(naive), rmse(corr),
            mean(naive) - mean(yt), mean(corr) - mean(yt)))

.report(ok, "test_smearing")
