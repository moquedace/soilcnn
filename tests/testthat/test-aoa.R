test_that("a training row is at DI 0, and a point far from them all is outside the area", {
  set.seed(1)
  x <- matrix(rnorm(200), 100, 2)
  ref <- di_reference(x)
  expect_true(all(dissimilarity_index(ref, x) < 1e-6))
  th <- aoa_threshold(ref, folds = rep(1:5, 20))
  expect_gt(as.numeric(th), 0)
  expect_length(attr(th, "cv_di"), 100)
  expect_false(inside_aoa(dissimilarity_index(ref, rbind(c(10, 10))), th))
  expect_true(inside_aoa(0, th))
})

test_that("the DI is in units of the mean distance between training rows", {
  x <- rbind(c(0, 0), c(1, 0))                 # one pair, one unit apart
  ref <- di_reference(x)
  expect_equal(ref$avg_dist, 1)
  expect_equal(dissimilarity_index(ref, rbind(c(3, 0))), 2)
})

test_that("a channel of weight zero does not count", {
  set.seed(1)
  x <- matrix(rnorm(200), 100, 2)
  ref <- di_reference(x, weights = c(1, 0))
  same_in_the_first <- rbind(c(x[1, 1], 100))  # row 1 in channel 1, far off in channel 2
  expect_lt(dissimilarity_index(ref, same_in_the_first), 1e-6)
})
