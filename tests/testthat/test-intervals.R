test_that("the conformal half-width is the k-th residual, k = ceiling((n + 1)(1 - alpha))", {
  cal <- conformal_calibrate(obs = rep(0, 19), pred = 1:19, alpha = 0.1)
  expect_equal(cal$k, 18)
  expect_equal(cal$q, 18)
  iv <- conformal_interval(cal, pred = 5, lower_limit = 0)
  expect_equal(iv$lower, 0)                # 5 - 18, floored at the limit
  expect_equal(iv$upper, 23)
})

test_that("too few calibration points give an infinite interval, and say so", {
  expect_warning(cal <- conformal_calibrate(obs = rep(0, 5), pred = 1:5, alpha = 0.1),
                 "certifiable")
  expect_equal(cal$q, Inf)
})

test_that("on exchangeable points the coverage is the one asked for", {
  set.seed(1)
  obs  <- rnorm(4000)
  pred <- obs + rnorm(4000)
  cal  <- conformal_calibrate(obs[1:2000], pred[1:2000], alpha = 0.1)
  iv   <- conformal_interval(cal, pred[2001:4000])
  cover <- picp_report(obs[2001:4000], iv$lower, iv$upper)$overall$picp
  expect_gt(cover, 0.87)
  expect_lt(cover, 0.93)
})

test_that("a normalised calibration refuses predictions without their difficulty", {
  cal <- conformal_calibrate(obs = rep(0, 19), pred = 1:19, alpha = 0.1, difficulty = rep(2, 19))
  expect_error(conformal_interval(cal, pred = 5), "difficulty")
  expect_equal(conformal_interval(cal, pred = 5, difficulty = 2)$upper, 5 + 2 * cal$q)
})

test_that("by group, one point per group is the ordinary rank", {
  set.seed(2)
  s <- rexp(60)
  by_point <- conformal_calibrate(obs = rep(0, 60), pred = s, alpha = 0.1)
  by_group <- conformal_calibrate(obs = rep(0, 60), pred = s, alpha = 0.1, group = 1:60)
  expect_equal(by_group$q, by_point$q)
  expect_equal(by_group$weighting, "group")
})

test_that("by group, q is where the groups' mean distribution reaches (1 - alpha)(1 + 1/m)", {
  # 10 groups: one of 91 points with residual 100, nine of one point each with
  # residuals 1..9. By point, 91% of the points sit at 100; by group, every
  # group weighs a tenth, and the level 0.9 x 1.1 = 0.99 is reached only at
  # the big group's 100 -- while 80% stops at 9: 0.8 x 1.1 = 0.88, and nine
  # groups of ten reach it at their largest residual.
  res <- c(rep(100, 91), 1:9)
  grp <- c(rep("big", 91), paste0("g", 1:9))
  cal90 <- conformal_calibrate(rep(0, 100), res, alpha = 0.1, group = grp)
  cal80 <- conformal_calibrate(rep(0, 100), res, alpha = 0.2, group = grp)
  expect_equal(cal90$q, 100)
  expect_equal(cal80$q, 9)
  expect_equal(cal80$n_groups, 10L)
  # by point, the same 80% is the big group's
  expect_equal(conformal_calibrate(rep(0, 100), res, alpha = 0.2)$q, 100)
})

test_that("by group, q is the largest residual the subsampling p-value accepts", {
  set.seed(3)
  res <- rexp(200)
  grp <- sample(1:25, 200, replace = TRUE)
  q <- conformal_calibrate(rep(0, 200), res, alpha = 0.1, group = grp)$q
  m <- length(unique(grp))
  pval <- function(r) (1 + sum(tapply(res >= r, grp, mean))) / (m + 1)
  accepted <- Filter(function(r) pval(r) >= 0.1, sort(res))
  expect_equal(q, max(unlist(accepted)))
})

test_that("too few groups give an infinite interval, and say so", {
  expect_warning(cal <- conformal_calibrate(rep(0, 40), 1:40, alpha = 0.1,
                                            group = rep(1:5, 8)), "groups")
  expect_equal(cal$q, Inf)
})

test_that("a scale fitted on other points leaves every point to calibrate q", {
  set.seed(4)
  level <- runif(300, 1, 4); di <- runif(300)
  obs <- exp(level + rnorm(300, 0, 0.1 + 0.3 * di)); pred <- exp(level)
  X <- data.frame(level = pred, di = di)
  sc <- conformal_scale_fit(obs[1:200], pred[1:200], X[1:200, ])
  cs <- conformal_scaled_calibrate(obs[201:300], pred[201:300], X[201:300, ], scale = sc)
  expect_equal(cs$n, 100L)
  expect_equal(unname(cs$coef), unname(sc$coef))
  expect_true(cs$scale_given)
  expect_error(conformal_scaled_calibrate(obs[201:300], pred[201:300], X[201:300, "level", drop = FALSE],
                                          scale = sc), "lacks")
})

test_that("by group, one point per group fits the scale on the half a point would", {
  # Groups of one point are points: the same half fits the scale and the same
  # half calibrates q, so the interval is the one by point -- not the same
  # method over another random half.
  set.seed(6)
  level <- runif(120, 1, 4); di <- runif(120)
  obs <- exp(level + rnorm(120, 0, 0.1 + 0.3 * di)); pred <- exp(level)
  X <- data.frame(level = pred, di = di)
  by_point <- conformal_scaled_calibrate(obs, pred, X, alpha = 0.1)
  by_group <- conformal_scaled_calibrate(obs, pred, X, alpha = 0.1, group = seq_along(obs))
  expect_equal(by_group$coef, by_point$coef)
  expect_equal(by_group$q, by_point$q)
  expect_equal(by_group$weighting, "group")
  # groups of two take whole groups, and so another half
  by_pair <- conformal_scaled_calibrate(obs, pred, X, alpha = 0.1, group = rep(1:60, each = 2))
  expect_false(isTRUE(all.equal(by_pair$coef, by_point$coef)))
})

test_that("CV+ is the order statistics of the fold models' values, both bounds", {
  set.seed(5)
  n_k <- c(30, 25, 40, 1, 34)
  fold <- rep(seq_along(n_k), n_k)
  oof  <- rnorm(sum(n_k), 20, 3)
  obs  <- oof + rnorm(sum(n_k), 0, 2)
  cal  <- cv_plus_calibrate(obs, oof, fold, alpha = 0.1)
  mu <- matrix(rnorm(8 * 5, 20, 1), 8, 5)
  iv <- cv_plus_interval(cal, mu)
  R <- abs(obs - oof)
  n <- length(R)
  for (j in 1:8) {
    up <- sort(mu[j, fold] + R)[ceiling(0.9 * (n + 1))]
    lo <- sort(mu[j, fold] - R)[floor(0.1 * (n + 1))]
    expect_equal(iv$upper[j], up, tolerance = 1e-7)
    expect_equal(iv$lower[j], lo, tolerance = 1e-7)
  }
  # columns named by fold are matched by name, in any order
  colnames(mu) <- as.character(1:5)
  expect_equal(cv_plus_interval(cal, mu[, 5:1])$upper, iv$upper)
})

test_that("CV+ normalised multiplies the scores back by the new point's scale", {
  set.seed(6)
  fold <- rep(1:4, 25)
  oof <- rnorm(100, 10)
  d <- runif(100, 0.5, 2)
  obs <- oof + d * rnorm(100)
  cal <- cv_plus_calibrate(obs, oof, fold, alpha = 0.2, difficulty = d)
  mu <- matrix(10, 3, 4)
  expect_error(cv_plus_interval(cal, mu), "difficulty")
  iv <- cv_plus_interval(cal, mu, difficulty = c(1, 2, 3))
  s <- sort(abs(obs - oof) / d)
  r <- ceiling(0.8 * 101)
  expect_equal(iv$upper, 10 + c(1, 2, 3) * s[r], tolerance = 1e-7)
})

test_that("CV+ with too few points is infinite, and with one fold refused", {
  expect_warning(cal <- cv_plus_calibrate(1:6, 1:6 + 0.5, rep(1:2, 3), alpha = 0.1),
                 "certifiable")
  iv <- cv_plus_interval(cal, matrix(0, 2, 2))
  expect_equal(iv$upper, c(Inf, Inf))
  expect_error(cv_plus_calibrate(1:20, 1:20, rep(1, 20)), "2 folds")
})
