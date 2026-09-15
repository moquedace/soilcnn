# ── The front end ─────────────────────────────────────────────────────────────
#
# WHAT THIS IS FOR.
#
# Everything under R/ is the engine, and the engine is not the product. A
# person who wants to fit a dual-branch CNN to their own soil data should not
# have to know that the scaling is fitted per fold, that the buffer is exact
# only under a Chebyshev metric, or that the patch store keeps one file per
# window. Those facts are why the results are trustworthy; they are not the
# interface.
#
# caret is the inspiration and not the template. What is worth copying is its
# SHAPE:
#
#   * one function does the fitting                      -> dsm_train()
#   * a small object describes the resampling            -> spatial_cv() etc.
#   * the model is named, not hand-built                 -> model = "cnn"
#   * tune_length asks for a budget, not a grid          -> tune_length = 30
#
# What is deliberately NOT copied is caret's habit of owning everything it
# touches. `trainControl()` carries 30 arguments because caret also does the
# preprocessing, the parallel backend, the sampling and the summary functions.
# Here the resampling spec carries the handful of things that decide WHO trains
# and WHO scores, and nothing else.
#
# THE ONE RULE THIS FILE FOLLOWS.
#
# Every default must be either obviously right or computed from the data --
# never a number someone once measured on another dataset. That is not a style
# preference: a block size measured on the full point set and carried into a
# 10% subsample is how one fold of this project's own dev run came to hold a
# third of the data. So `block_size = "auto"` measures, `buffer = "auto"`
# derives from the window and the resolution, and both say what they chose.

# ── loading: one call instead of six ──────────────────────────────────────────

#' Load everything a run needs, and check that it fits together.
#'
#' Replaces the six-step preamble every pipeline script used to repeat: open
#' the store, read the points, read the predictor types, align the points to
#' the store, read the raster resolution, and verify the store can serve this
#' configuration.
#'
#' @param patch_dir    Directory written by the extraction step.
#' @param points       Point table, or a path to the CSV written by stage 01.
#' @param type_table   Predictor types, or a path to its CSV.
#' @param windows      Windows to load. NULL loads every window the store has,
#'   which is the wrong default when the grid needs two of five -- pass the
#'   windows the grid actually uses and the store reads only those.
#' @param cell_size    Raster resolution in x/y units. NULL reads it from
#'   `raster_table`, which is the only source that cannot drift.
#' @param raster_table Path to raster_table_used.csv, for `cell_size`.
#' @param target_col   Expected target column, for the store lock.
#' @return A `dsm_data` object.
dsm_load <- function(patch_dir, points, type_table, windows = NULL,
                     cell_size = NULL, raster_table = NULL, target_col = NULL,
                     verbose = TRUE) {

  read_if_path <- function(z, what) {
    if (is.character(z) && length(z) == 1L) {
      if (!file.exists(z)) stop(what, " not found: ", z, call. = FALSE)
      readr::read_csv2(z, show_col_types = FALSE)
    } else z
  }
  points     <- read_if_path(points,     "Point table")
  type_table <- read_if_path(type_table, "Predictor type table")

  store  <- load_patch_store(patch_dir, windows, verbose = verbose)
  points <- align_points_to_meta(points, store$meta)

  # Read from the raster itself, never written by hand: block_size and buffer
  # are given in the SAME units as x/y, and getting that wrong produces either
  # an abort (harmless) or a split that only LOOKS spatial (not).
  if (is.null(cell_size) && !is.null(raster_table)) {
    rt <- read_if_path(raster_table, "raster_table_used.csv")
    r1 <- rt$raster_file[1]
    if (!is.null(r1) && file.exists(r1) &&
        requireNamespace("terra", quietly = TRUE)) {
      cell_size <- terra::res(terra::rast(r1))[1]
    }
  }
  if (is.null(cell_size)) {
    # The store recorded it at extraction time; that is a weaker source than
    # the raster (it says what WAS true, not what is) but far better than none.
    sp <- store_spec(store)
    if (!is.na(sp$cell_size)) cell_size <- sp$cell_size
  }

  check_store_spec(store,
                   predictors = type_table$predictor,
                   windows    = store$window_sizes,
                   target_col = target_col,
                   cell_size  = cell_size)

  out <- structure(
    list(store = store, points = points, type_table = type_table,
         cell_size = cell_size, patch_dir = patch_dir,
         target_col = target_col %||% store_spec(store)$target_col),
    class = "dsm_data")
  if (verbose) print(out)
  out
}

#' @export
print.dsm_data <- function(x, ...) {
  cat("<dsm_data>\n")
  cat("  points     : ", format(nrow(x$store$meta), big.mark = ","), "\n", sep = "")
  cat("  channels   : ", x$store$n_channels, "\n", sep = "")
  cat("  windows    : ", paste(x$store$window_sizes, collapse = ", "), "\n", sep = "")
  cat("  target     : ", x$target_col %||% "(not recorded)", "\n", sep = "")
  cat("  cell size  : ",
      if (is.null(x$cell_size)) "(unknown -- 'auto' buffers unavailable)"
      else format(x$cell_size, digits = 8), "\n", sep = "")
  invisible(x)
}

# ── resampling specs: what trainControl() is for ──────────────────────────────
#
# A SPEC IS NOT A PLAN. It says what kind of split is wanted; the plan is what
# you get when that is applied to a specific set of points. Keeping them apart
# is what lets `block_size = "auto"` mean "measure it when you see the data"
# instead of "guess now".

.resample_spec <- function(kind, ...) {
  structure(c(list(kind = kind), list(...)), class = "resample_spec")
}

#' Spatially blocked k-fold: whole blocks of ground go to one fold.
#'
#' @param k          Folds.
#' @param block_size "auto" measures it from the points (see
#'   suggest_block_size); a number is used as given, in x/y units.
#' @param buffer     "auto" is `max(window) * cell_size`, which is the exact
#'   distance at which two patches stop sharing a pixel under the Chebyshev
#'   metric; a number is used as given; NULL applies none.
#' @param test_frac  Share held out entirely, carved by the same criterion.
#' @param max_share  Balance constraint for "auto": the largest share of the
#'   points one block may hold.
spatial_cv <- function(k = 5L, block_size = "auto", buffer = "auto",
                       test_frac = 0.15, max_share = 0.10,
                       buffer_metric = c("chebyshev", "euclidean"),
                       seed = 42L) {
  .resample_spec("spatial", k = as.integer(k), block_size = block_size,
                 buffer = buffer, test_frac = test_frac, max_share = max_share,
                 buffer_metric = match.arg(buffer_metric), seed = seed)
}

#' Random k-fold. Ignores geography by construction.
#'
#' Right when the rows really are independent, and the cleanest way to MEASURE
#' what geography is worth: run it against spatial_cv() on the same points and
#' the gap is the spatial optimism.
random_cv <- function(k = 5L, test_frac = 0.15, group = "auto", seed = 42L) {
  .resample_spec("random", k = as.integer(k), test_frac = test_frac,
                 group = group, seed = seed)
}

#' A single train/validation/test split.
holdout_cv <- function(validation_frac = 0.15, test_frac = 0.15,
                       group = "auto", seed = 42L) {
  .resample_spec("holdout", validation_frac = validation_frac,
                 test_frac = test_frac, group = group, seed = seed)
}

#' Leave-region-out, on a grouping that already exists (biome, catchment, ...).
region_cv <- function(group, k = NULL, test_frac = 0.15, seed = 42L) {
  .resample_spec("region", group = group, k = k, test_frac = test_frac,
                 seed = seed)
}

#' @export
print.resample_spec <- function(x, ...) {
  cat("<resample_spec> ", x$kind, "\n", sep = "")
  for (nm in setdiff(names(x), "kind")) {
    v <- x[[nm]]
    if (is.null(v)) v <- "NULL"
    if (length(v) > 4L) v <- sprintf("<%d values>", length(v))
    cat(sprintf("  %-15s %s\n", nm, paste(format(v), collapse = ", ")))
  }
  invisible(x)
}

#' Turn a spec into a fold plan against real points.
#'
#' Exported because a plan is worth looking at before spending a night on it:
#' `plan <- resolve_resampling(spatial_cv(k = 5), data); print(plan)`.
#'
#' @param spec     From spatial_cv() and friends, or an existing fold_plan,
#'   which is returned unchanged.
#' @param data     From dsm_load().
#' @param test_ids Sample ids to force into the test set, so a frozen test set
#'   survives a change of method.
#' @param windows  Windows the grid will use, for `buffer = "auto"`.
resolve_resampling <- function(spec, data, test_ids = NULL, windows = NULL,
                               verbose = TRUE) {
  if (inherits(spec, "fold_plan")) return(spec)
  stopifnot(inherits(spec, "resample_spec"), inherits(data, "dsm_data"))
  meta <- data$store$meta
  if (is.null(windows)) windows <- data$store$window_sizes

  auto_buffer <- function(b) {
    if (is.null(b)) return(NULL)
    if (!identical(b, "auto")) return(as.numeric(b))
    if (is.null(data$cell_size)) {
      stop("buffer = \"auto\" needs the raster resolution, and this dsm_data ",
           "has none. Pass cell_size to dsm_load(), or give buffer a number ",
           "in the units of x/y.", call. = FALSE)
    }
    # max(window) * cell_size, and not half of it: two patches of width w share
    # a pixel when their centres are within w-1 cells in BOTH axes. That is a
    # SQUARE condition, so the exact separation distance is w * res under the
    # Chebyshev metric -- a circular buffer of the same radius lets the
    # diagonal escape.
    max(windows) * data$cell_size
  }

  plan <- switch(spec$kind,
    spatial = {
      bs <- spec$block_size
      if (identical(bs, "auto")) {
        choice <- suggest_block_size(meta, k = spec$k, max_share = spec$max_share)
        if (verbose) print_block_choice(choice)
        bs <- as.numeric(choice)
      }
      spatial_folds(meta, k = spec$k, test_frac = spec$test_frac,
                    block_size = as.numeric(bs),
                    buffer = auto_buffer(spec$buffer),
                    buffer_metric = spec$buffer_metric,
                    test_ids = test_ids, seed = spec$seed)
    },
    random  = random_folds(meta, k = spec$k, test_frac = spec$test_frac,
                           test_ids = test_ids, seed = spec$seed,
                           group = spec$group),
    holdout = holdout(meta, validation_frac = spec$validation_frac,
                      test_frac = spec$test_frac, test_ids = test_ids,
                      seed = spec$seed, group = spec$group),
    region  = region_folds(meta, group = spec$group, k = spec$k,
                           test_frac = spec$test_frac, test_ids = test_ids,
                           seed = spec$seed),
    stop("Unknown resampling kind: ", spec$kind, call. = FALSE)
  )
  # Proven against the data, not assumed from the constructor.
  check_fold_plan(plan, meta = meta)
  plan
}

# ── the one function ──────────────────────────────────────────────────────────

#' Fit and tune a model over a resampling plan.
#'
#' @param data       From dsm_load().
#' @param model      A registered model name ("cnn", "rf", "mlp", or anything
#'   registered with register_model()), or a model_spec.
#' @param resampling A resample_spec, or a fold_plan to use as given.
#' @param tune_grid  An explicit grid. NULL asks the model for one.
#' @param tune_length How many configurations to try when `tune_grid` is NULL.
#'   The same meaning as caret's: a budget, not a lattice.
#' @param n_seeds    Repetitions per (config, fold). One gives a ranking with
#'   no error bar, which is a ranking of luck as often as of skill.
#' @param transform  Inverse of the transform the target carries, e.g. expm1.
#' @param features   For tabular models: "centre", "window_mean", or both.
#' @param test_ids   Sample ids forced into the test set.
#' @param ...        Passed to the underlying runner (n_epochs, patience, ...).
#' @return The runner's result, plus the plan and the data it used.
dsm_train <- function(data, model = "cnn", resampling = spatial_cv(),
                      tune_grid = NULL, tune_length = 20L, n_seeds = 3L,
                      transform = identity,
                      features = c("centre", "window_mean"),
                      output_dir = "./outputs/tuning",
                      run_id = format(Sys.time(), "%Y%m%d_%H%M%S"),
                      base_seed = 42L, device = NULL, test_ids = NULL,
                      resume = TRUE, verbose = TRUE, ...) {

  stopifnot(inherits(data, "dsm_data"))
  if (is.character(model)) model <- get_model(model)
  stopifnot(inherits(model, "model_spec"))

  windows_needed <- if (!is.null(tune_grid) && "window_sizes" %in% names(tune_grid)) {
    sort(unique(unlist(tune_grid$window_sizes)))
  } else data$store$window_sizes

  plan <- resolve_resampling(resampling, data, test_ids = test_ids,
                             windows = windows_needed, verbose = verbose)

  if (identical(model$input, "patches")) {
    if (is.null(tune_grid)) {
      if (is.null(model$default_grid)) {
        stop("Model '", model$name, "' has no default grid; pass tune_grid.",
             call. = FALSE)
      }
      tune_grid <- model$default_grid(tune_length, base_seed)
    }
    # The grid may ask for windows the caller did not load. Said here, where
    # the fix is one argument, rather than inside the loader.
    need <- sort(unique(unlist(tune_grid$window_sizes)))
    if (!all(need %in% data$store$window_sizes)) {
      stop("This grid needs window(s) ",
           paste(setdiff(need, data$store$window_sizes), collapse = ", "),
           " but the data was loaded with ",
           paste(data$store$window_sizes, collapse = ", "),
           ".\n  Reload with dsm_load(..., windows = c(",
           paste(need, collapse = ", "), ")).", call. = FALSE)
    }
    if (is.null(device)) device <- setup_torch_device()
    res <- run_cnn_resample(
      tune_grid = tune_grid, store = data$store, points = data$points,
      type_table = data$type_table, plan = plan, transform = transform,
      output_dir = output_dir, device = device, run_id = run_id,
      base_seed = base_seed, n_seeds = n_seeds, resume = resume, ...)
  } else {
    res <- run_table_resample(
      model = model, tune_grid = tune_grid, store = data$store,
      points = data$points, type_table = data$type_table, plan = plan,
      features = features, transform = transform, output_dir = output_dir,
      run_id = run_id, base_seed = base_seed, n_seeds = n_seeds,
      tune_length = tune_length, device = device, resume = resume, ...)
  }

  res$model <- model$name
  res$data  <- data
  class(res) <- c("dsm_fit", class(res))
  if (verbose) print(res)
  res
}

#' @export
print.dsm_fit <- function(x, ...) {
  cat("\n<dsm_fit> model: ", x$model, " | ", x$plan$method, " | ",
      x$plan$n_folds, " fold(s)\n", sep = "")
  if (nrow(x$by_config) > 0L) {
    cat("\nTop configurations (mean +/- sd over repetitions):\n")
    print_wide(utils::head(dplyr::select(
      x$by_config, dplyr::any_of(c("rank", "config_id", "n_units",
                                   "val_ccc_mean", "val_ccc_sd", "val_ccc_se",
                                   "val_mae_mean", "n_failed"))), 5L), n = 5L)
    nf <- try(seed_noise_floor(x$comparison), silent = TRUE)
    if (!inherits(nf, "try-error") && is.finite(nf$median_sd)) {
      cat(sprintf(
        "\nSeed noise floor: %.4f -- a gap between configs smaller than this ",
        nf$median_sd), "is not evidence.\n", sep = "")
    }
  }
  cat("\nResults: ", x$run_dir, "\n", sep = "")
  invisible(x)
}
