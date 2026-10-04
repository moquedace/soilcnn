# One store of the example landscape for the file: the extraction reads eight
# rasters, a second or two, and every test here reads the same store.
landscape_store <- local({
  made <- NULL
  function() {
    if (is.null(made)) {
      ex <- example_landscape()
      made <<- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                           windows = c(3, 7), out_dir = tempfile("landscape"),
                           percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
                           verbose = FALSE)
    }
    made
  }
})

test_that("the example landscape becomes a store that loads without torch's backend", {
  data <- dsm_load(landscape_store(), windows = integer(0), verbose = FALSE)
  expect_s3_class(data, "dsm_data")
  expect_equal(nrow(data$store$meta), nrow(example_landscape()$profiles))
  expect_equal(data$store$n_channels, 8L)
  expect_identical(data$transform$name, "log1p")
  expect_equal(data$transform$inverse(data$transform$forward(41)), 41)
  expect_equal(data$cell_size, 0.0025)
})

test_that("the geology dummies are found as one variable, and a grouping of yours applies", {
  data <- dsm_load(landscape_store(), windows = integer(0), verbose = FALSE)
  g <- importance_groups(data)
  geology <- grepl("^geology_", g$channel)
  expect_equal(sum(geology), 3L)
  expect_length(unique(g$variable[geology]), 1L)
  expect_true(all(g$rule[geology] == "one-hot set"))
  expect_true(all(g$rule[!geology] == "alone"))
  themes <- data.frame(channel = c("temperature", "precipitation"), variable = "climate")
  mine <- importance_groups(data, themes)
  expect_equal(mine$variable[mine$channel %in% themes$channel], c("climate", "climate"))
  expect_true(all(mine$rule[mine$channel %in% themes$channel] == "yours"))
})

test_that("a spec becomes a plan against the store's points", {
  data <- dsm_load(landscape_store(), windows = integer(0), verbose = FALSE)
  plan <- resolve_resampling(spatial_cv(k = 3, block_size = 0.05, test_frac = 0.2), data,
                             windows = c(3, 7), verbose = FALSE)
  expect_s3_class(plan, "fold_plan")
  expect_equal(plan$n_folds, 3L)
})
