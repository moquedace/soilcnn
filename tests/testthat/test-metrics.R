test_that("ccc() is 1 for agreement, and an offset is disagreement though r is 1", {
  obs <- c(10, 20, 30, 40)
  expect_equal(ccc(obs, obs), 1)
  expect_lt(ccc(obs, obs + 5), 1)
  # 2 cov / (var + var + offset^2), the moments by n: 2 * 125 / (125 + 125 + 25)
  expect_equal(ccc(obs, obs + 5), 250 / 275)
})

test_that("calc_metrics() reports a constant error as the error it is, with its sign", {
  m <- calc_metrics(obs = c(1, 2, 3, 4), pred = c(2, 3, 4, 5))
  expect_equal(m$n, 4L)
  expect_equal(m$mae, 1)
  expect_equal(m$rmse, 1)
  expect_equal(m$bias, 1)          # mean(pred - obs): above the observations
  expect_equal(m$bias_pct, 40)     # of mean(obs) = 2.5
  expect_equal(m$ccc, 2 * 1.25 / (1.25 + 1.25 + 1))
})

test_that("calc_metrics() says NA, not a number, when there is nothing to measure", {
  m <- calc_metrics(obs = 1, pred = 2)
  expect_equal(m$n, 1L)
  expect_true(is.na(m$ccc) && is.na(m$mae))
})
