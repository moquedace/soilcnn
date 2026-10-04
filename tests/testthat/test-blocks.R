test_that("equal-area blocks: metres as given, degrees projected at their centre", {
  b <- equal_area_blocks(c(500, 900, 1500), c(500, 900, 500), size_km = 1, coords = "metres")
  expect_equal(as.character(b), c("0_0", "0_0", "1_0"))
  # 0.03, 0.05 and 0.10 degrees north of the centre: 3.3, 5.6 and 11.1 km
  d <- equal_area_blocks(c(-49, -49, -49), c(-19.97, -19.95, -19.90), size_km = 10,
                         centre = c(-49, -20))
  expect_equal(as.character(d), c("0_0", "0_0", "0_1"))
})

test_that("a constant error is the estimate, and every draw of the blocks agrees", {
  set.seed(1)
  obs <- runif(60, 10, 50)
  blk <- rep(1:12, each = 5)
  bb <- block_bootstrap(obs, obs + 2, blk, metric = c("mae", "bias"), n_boot = 200)
  mae <- bb[bb$metric == "mae" & bb$weights == "profile", ]
  expect_equal(mae$estimate, 2)
  expect_equal(c(mae$ci_low, mae$ci_high), c(2, 2))
  expect_equal(bb$estimate[bb$metric == "bias"], c(2, 2))   # by profile and by block
})

test_that("the same seed draws the same blocks, and a model against itself differs by nothing", {
  set.seed(1)
  obs  <- runif(60, 10, 50)
  pred <- obs + rnorm(60, 0, 5)
  blk  <- rep(1:12, each = 5)
  a <- block_bootstrap(obs, pred, blk, metric = "mae", n_boot = 200, seed = 3)
  b <- block_bootstrap(obs, pred, blk, metric = "mae", n_boot = 200, seed = 3)
  expect_identical(a$ci_low, b$ci_low)
  self <- block_bootstrap(obs, pred, blk, against = pred, metric = "mae", n_boot = 200)
  expect_equal(self$difference, c(0, 0))
  expect_equal(c(self$ci_low, self$ci_high), c(0, 0, 0, 0))
})

test_that("one block is refused: there is nothing to draw", {
  expect_error(block_bootstrap(1:10, 1:10 + 1, rep(1, 10), n_boot = 200), "one block")
})
