# The network trained end to end: the small example run, on the example
# landscape, where libtorch is installed. CRAN's machines have no backend, and
# the run takes a minute, so it is skipped there.

test_that("the example run trains, refits and predicts", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed(), "no libtorch backend")
  run <- example_run(file.path(tempdir(), "soilcnn_test_network"))
  expect_true(dir.exists(run$fit$run_dir))
  expect_true(dir.exists(run$final$run_dir))
  res <- cv_residuals(run$fit$run_dir, run$final$selected_config_ids)
  expect_gt(nrow(res), 0)
  expect_true(all(is.finite(res$obs)))
})
