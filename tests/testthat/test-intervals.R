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
