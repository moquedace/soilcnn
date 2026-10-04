smearing_fixture <- function() {
  p <- rep(log1p(c(5, 20, 50)), 20)            # predictions, in log1p space
  list(p = p, o = p + rep(c(-0.5, 0.5), 30))   # residuals of -0.5 and 0.5, half each
}

test_that("the global factor is Duan's: the mean of exp(residual)", {
  d <- smearing_fixture()
  cal <- smearing_factor(d$o, d$p)
  expect_equal(cal$s, cosh(0.5))
  # exp(z) S - 1, not expm1(z) S: the -1 comes out after the scaling
  expect_equal(smear(log1p(10), cal), 11 * cosh(0.5) - 1)
  expect_equal(smear(log1p(10), cosh(0.5)), 11 * cosh(0.5) - 1)   # a bare number too
})

test_that("a factor by level needs residuals enough to have one", {
  d <- smearing_fixture()
  cal <- smearing_factor(d$o, d$p)
  expect_error(smear(2, cal, method = "level"), "no factor by level")
  expect_error(smearing_factor(d$o[1:10], d$p[1:10]), "usable residual")
})

test_that("the mean is never asked to go below the floor", {
  d <- smearing_fixture()
  cal <- smearing_factor(d$o, d$p)
  expect_equal(smear(-5, cal, lower_limit = 0), 0)
})
