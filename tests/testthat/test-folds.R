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
