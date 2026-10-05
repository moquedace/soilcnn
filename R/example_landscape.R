# ── The example landscape the package ships ──────────────────────────────────
#
# A synthetic data set small enough to install with the package, made by
# data-raw/make_example_landscape.R (in the repository, not in the package):
# what the examples, the fast tests and a computing vignette run on. The
# script's header says what was put in it and why; in short, every part of the
# package has something to find in it -- a one-hot set, a percentage, clustered
# profiles, a right-skewed target, and a signal only a neighbourhood shows.

#' A small synthetic landscape, for the examples and the tests.
#'
#' 80 x 80 cells of 0.0025 degrees (about 250 m) in coordinates of
#' south-eastern Brazil -- the place is borrowed, the landscape is made up --
#' with eight predictors and 160 soil profiles, 90 of them in six survey
#' clusters. The soil organic carbon stock (t/ha, 0-30 cm) was drawn from
#' vegetation, clay, temperature, geology and the topographic position of the
#' cell in its 7 x 7 neighbourhood: a network that reads the neighbourhood has
#' something to find that the centre pixel does not show.
#'
#' @return A list: `raster_dir`, the folder of the eight GeoTIFFs (elevation,
#'   temperature, precipitation, ndvi, clay_pct, and geology as three 0/1
#'   dummies); `profiles`, a data frame of `profile_id`, `x`, `y`, `survey` and
#'   `soc_stock`; and `truth`, the formula the stock was drawn from.
#' @examples
#' ex <- example_landscape()
#' head(ex$profiles)
#' list.files(ex$raster_dir)
#' @export
example_landscape <- function() {
  dir <- system.file("extdata", "landscape", package = "soilcnn")
  if (!nzchar(dir) || !file.exists(file.path(dir, "profiles.csv"))) {
    stop("The example landscape is not in this copy of soilcnn (inst/extdata/landscape).",
         call. = FALSE)
  }
  list(raster_dir = file.path(dir, "rasters"),
       profiles = utils::read.csv(file.path(dir, "profiles.csv"), stringsAsFactors = FALSE),
       truth = paste("log(soc_stock) = 2.6 + 1.5 ndvi + 0.012 clay_pct - 0.03 (temperature - 20)",
                     "- 0.005 position + 0.25 basalt - 0.15 sandstone + a smooth field (sd 0.15)",
                     "+ noise (sd 0.25), where position is the elevation less its 7 x 7 mean"))
}

# ONE SMALL RUN, MADE ONCE A SESSION. The examples of the functions that need a
# fitted model -- dsm_final(), dsm_predict(), dsm_importance() and the rest --
# would each train one, and CRAN's checks run every example: a minute each. So
# they share this one, made at the first call and kept for the session, in
# tempdir(), on one core: two folds of one seed to tune, two seeds to refit, a
# small network for 30 epochs.

#' A small fitted run on the example landscape, made once a session.
#'
#' The store, a tuning run and a final model on [example_landscape()], in
#' `dir`: what the examples of the functions that need a fitted model start
#' from. Made at the first call -- a minute or two on one core -- and kept
#' for the session; later calls return it at once. It is an example, not a
#' model: 160 profiles, one small configuration, 30 epochs.
#'
#' @param dir Where the run is made. The default is in `tempdir()`, so it goes
#'   with the session.
#' @param verbose Report each step.
#' @return A list: `data` (from [dsm_load()]), `fit` (from [dsm_train()]),
#'   `final` (from [dsm_final()]) and `dir`.
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()
#' run$final
#' }
#' @export
example_run <- function(dir = file.path(tempdir(), "soilcnn_example"), verbose = FALSE) {
  key <- normalizePath(dir, winslash = "/", mustWork = FALSE)
  kept <- .pkg_state$example_runs[[key]]
  if (!is.null(kept) && dir.exists(kept$final$run_dir)) return(kept)
  # One thread for the tuning in this session, and the session's own count --
  # torch's, and the two variables set_torch_threads() writes -- put back
  # after: an example does not leave torch slower, nor the environment changed.
  threads_before <- torch::torch_get_num_threads()
  env_before <- Sys.getenv(c("OMP_NUM_THREADS", "MKL_NUM_THREADS"), unset = NA)
  on.exit({
    torch::torch_set_num_threads(threads_before)
    for (v in names(env_before)) {
      if (is.na(env_before[[v]])) Sys.unsetenv(v) else
        do.call(Sys.setenv, stats::setNames(list(env_before[[v]]), v))
    }
  }, add = TRUE)
  ex <- example_landscape()
  store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
                       windows = c(3, 7), out_dir = file.path(dir, "store"),
                       percentage = "^clay_pct$", transform = "log1p", n_cores = 1L,
                       overwrite = TRUE, verbose = verbose)
  data <- dsm_load(store, verbose = verbose)
  grid <- make_manual_tune_grid(window_sizes = list(c(3L, 7L)), conv_channels = list(c(8L)),
                                embedding_dim = 16L, base_lr = 0.005, batch_size = 16L,
                                dropout = 0, gate_type = "vector_featurewise",
                                use_residual = FALSE, use_se_block = FALSE)
  fit <- dsm_train(data, model = "cnn",
                   resampling = spatial_cv(k = 2L, block_size = 0.05, buffer = "auto",
                                           test_frac = 0.2),
                   tune_grid = grid, n_seeds = 1L, output_dir = file.path(dir, "tuning"),
                   run_id = "tuning", device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
                   n_epochs = 30L, patience = 10L, print_every = 100L, verbose = verbose)
  # 30% to stop the refit on: k = 3 blocks, so the 15 blocks of 160 points
  # split cleanly; at the default 15% (k = 7) the largest block outweighs a fold.
  final <- dsm_final(fit, seeds = 2L, n_cores = 1L, threads_per_unit = 1L,
                     validation_frac = 0.3,
                     training = list(n_epochs = 30L, patience = 10L, print_every = 100L),
                     output_dir = file.path(dir, "final_model"), run_id = "final",
                     verbose = verbose)
  run <- list(data = data, fit = fit, final = final, dir = key)
  if (is.null(.pkg_state$example_runs)) .pkg_state$example_runs <- list()
  .pkg_state$example_runs[[key]] <- run
  run
}
