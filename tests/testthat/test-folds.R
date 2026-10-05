landscape_meta <- function() {
  ex <- example_landscape()
  data.frame(sample_id = seq_len(nrow(ex$profiles)), x = ex$profiles$x, y = ex$profiles$y)
}

test_that("spatial folds validate each pooled point once and keep the test set apart", {
  meta <- landscape_meta()
  plan <- spatial_folds(meta, k = 3, test_frac = 0.2, block_size = 0.05)
  val  <- unlist(lapply(plan$folds, `[[`, "validation"))
  test <- plan$folds[[1]]$test
  expect_equal(anyDuplicated(val), 0L)
  expect_length(intersect(val, test), 0)
  expect_setequal(c(val, test), seq_len(nrow(meta)))
  for (f in plan$folds) {
    expect_length(intersect(f$train, c(f$validation, f$test)), 0)
    expect_identical(f$test, test)
  }
})

test_that("the buffer leaves no training point within it of a validation point", {
  meta <- landscape_meta()
  buffer <- 7 * 0.0025                     # the widest window, in degrees
  plan <- spatial_folds(meta, k = 3, block_size = 0.05, buffer = buffer)
  for (f in plan$folds) {
    dx <- abs(outer(meta$x[f$train], meta$x[f$validation], "-"))
    dy <- abs(outer(meta$y[f$train], meta$y[f$validation], "-"))
    expect_true(all(pmax(dx, dy) >= buffer))   # Chebyshev: the patch is a square
  }
})

test_that("region folds never split a group", {
  ex <- example_landscape()
  plan <- region_folds(landscape_meta(), group = ex$profiles$survey, k = 3)
  for (f in plan$folds) {
    expect_length(intersect(ex$profiles$survey[f$train], ex$profiles$survey[f$validation]), 0)
  }
})

test_that("random folds partition the points, under their own seed only", {
  meta <- landscape_meta()
  set.seed(1)
  untouched <- runif(1)
  set.seed(1)
  a <- random_folds(meta, k = 4, seed = 7)
  expect_identical(runif(1), untouched)   # the session's random numbers left alone
  b <- random_folds(meta, k = 4, seed = 7)
  expect_identical(a$folds, b$folds)
  val <- unlist(lapply(a$folds, `[[`, "validation"))
  expect_equal(anyDuplicated(val), 0L)
  expect_setequal(val, seq_len(nrow(meta)))
})

test_that("a holdout's three sets are disjoint and cover the points", {
  meta <- landscape_meta()
  f <- holdout(meta, validation_frac = 0.2, test_frac = 0.2)$folds[[1]]
  expect_length(intersect(f$test, c(f$train, f$validation)), 0)
  expect_length(intersect(f$train, f$validation), 0)
  expect_setequal(c(f$train, f$validation, f$test), seq_len(nrow(meta)))
})

test_that("a calibration set is whole blocks, carved after the test set, in no fold", {
  meta <- landscape_meta()
  plan <- spatial_folds(meta, k = 3, test_frac = 0.2, block_size = 0.05,
                        calibration_frac = 0.2)
  cal  <- plan$calibration
  test <- plan$folds[[1]]$test
  expect_gt(length(cal), 0)
  expect_length(intersect(cal, test), 0)
  for (f in plan$folds) expect_length(intersect(cal, c(f$train, f$validation, f$test)), 0)
  # whole blocks: no block holds calibration points and others
  blk <- plan$group
  expect_length(intersect(unique(blk[cal]), unique(blk[-cal])), 0)
  # the test set does not move when a calibration set is asked for
  expect_identical(test, spatial_folds(meta, k = 3, test_frac = 0.2, block_size = 0.05)$folds[[1]]$test)
  # every point is in exactly one place
  pooled <- unlist(lapply(plan$folds, `[[`, "validation"))
  expect_setequal(c(pooled, test, cal), seq_len(nrow(meta)))
})

test_that("a frozen calibration set is taken as given, and never from the test set", {
  meta <- landscape_meta()
  plan <- spatial_folds(meta, k = 3, test_frac = 0.2, block_size = 0.05, calibration_frac = 0.2)
  ids  <- meta$sample_id[plan$calibration]
  again <- random_folds(meta, k = 3, test_ids = meta$sample_id[plan$folds[[1]]$test],
                        calibration_ids = ids)
  expect_setequal(again$calibration, plan$calibration)
  expect_error(random_folds(meta, k = 3, test_ids = meta$sample_id[plan$folds[[1]]$test],
                            calibration_ids = meta$sample_id[plan$folds[[1]]$test][1:3]),
               "test set")
})

test_that("the specs refuse held-out shares that leave nothing to train on", {
  expect_error(spatial_cv(test_frac = 0.5, calibration_frac = 0.5), "leave something")
  expect_error(random_cv(calibration_frac = 1.2), "fraction")
  expect_equal(holdout_cv(calibration_frac = 0.1)$calibration_frac, 0.1)
})
