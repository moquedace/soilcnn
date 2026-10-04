test_that("a resampling spec refuses numbers that cannot be what was meant", {
  expect_error(spatial_cv(k = 1), "at least 2")
  expect_error(spatial_cv(test_frac = 15), "fraction")
  expect_error(holdout_cv(validation_frac = 0.6, test_frac = 0.5), "leave something")
  expect_identical(spatial_cv(k = 4)$kind, "spatial")
})

test_that("a spec prints a table by its size", {
  pts <- data.frame(x = c(-49.60, -49.55, -49.50), y = c(-20.10, -20.05, -20.00))
  expect_output(print(knndm_cv(k = 5, predpoints = pts)), "<3 rows of x, y>")
})

test_that("a model spec checks its functions, and the registry holds the three", {
  expect_error(model_spec("bad", "table", fit = function(x, y) 0,
                          predict = function(object, x) 0), "cfg")
  spec <- model_spec("mean_only", "table", fit = function(x, y, cfg, ...) list(mu = mean(y)),
                     predict = function(object, x, ...) rep(object$mu, nrow(x)))
  expect_s3_class(spec, "model_spec")
  expect_true(all(c("cnn", "rf", "mlp") %in% list_models()$name))
})

test_that("a target transform reads back what it wrote", {
  tr <- target_transform_spec("log1p")
  expect_equal(tr$inverse(tr$forward(c(0, 1, 41))), c(0, 1, 41))
  expect_error(target_transform_spec("sqrt"), "Unknown target transform")
})
