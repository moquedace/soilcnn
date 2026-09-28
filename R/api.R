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
#' @param patch_dir    Directory written by the extraction step -- or the
#'   `dsm_store` dsm_prepare() returned, in which case nothing else is needed.
#' @param points       Point table, or a path to the CSV written by stage 01.
#'   NULL for a store written by dsm_prepare(), which carries its own copy.
#' @param type_table   Predictor types, or a path to its CSV. NULL as above.
#' @param windows      Windows to load. NULL loads every window the store has,
#'   which is the wrong default when the grid needs two of five -- pass the
#'   windows the grid actually uses and the store reads only those.
#' @param cell_size    Raster resolution in x/y units. NULL reads it from
#'   `raster_table`, which is the only source that cannot drift.
#' @param raster_table Path to raster_table_used.csv, for `cell_size`.
#' @param target_col   Expected target column, for the store lock.
#' @return A `dsm_data` object. `$transform` is the target transform the store
#'   was built under -- name, forward and inverse -- or NULL when the store did
#'   not record one.
dsm_load <- function(patch_dir, points = NULL, type_table = NULL, windows = NULL,
                     cell_size = NULL, raster_table = NULL, target_col = NULL,
                     verbose = TRUE) {

  # A STORE WRITTEN BY dsm_prepare() CARRIES ITS OWN TABLES, and its recipe
  # says where they are. That is what lets dsm_load(store) take one argument:
  # the six-step preamble this function replaced is not re-imposed on the user
  # as five paths to type. A store from before dsm_prepare() has no recipe,
  # and the explicit arguments stay required for it.
  if (inherits(patch_dir, "dsm_store")) patch_dir <- patch_dir$store_dir
  recipe_path <- file.path(patch_dir, "recipe.rds")
  recipe <- if (file.exists(recipe_path)) readRDS(recipe_path) else NULL
  if (is.null(points) || is.null(type_table)) {
    if (is.null(recipe)) {
      stop("`points` and `type_table` are required: ", patch_dir, " holds no ",
           "recipe.rds, so it was not written by dsm_prepare() and does not ",
           "carry its own tables.", call. = FALSE)
    }
    if (is.null(points))     points     <- file.path(patch_dir, recipe$files$points)
    if (is.null(type_table)) type_table <- file.path(patch_dir, recipe$files$type_table)
    if (is.null(raster_table) && is.null(cell_size)) {
      raster_table <- file.path(patch_dir, recipe$files$raster_table)
    }
    if (is.null(target_col)) target_col <- recipe$target
  }

  read_if_path <- function(z, what) {
    if (is.character(z) && length(z) == 1L) {
      if (!file.exists(z)) {
        stop(what, " not found: ", z, "\n  It is written by stage 01 ",
             "(examples/soc_stock_0_5cm/01_prepare_dataset.R); run that first, ",
             "or pass a data frame.", call. = FALSE)
      }
      safe_read_csv2(z)
    } else z
  }
  points     <- read_if_path(points,     "Point table")
  type_table <- read_if_path(type_table, "Predictor type table")

  # THE TYPE TABLE IS CHECKED AT THE DOOR. Without `predictor` the store-spec
  # check below is skipped (it compares against NULL), and a missing
  # is_dummy / is_percentage died inside dplyr::case_when() when the first fold
  # cache was built -- minutes in, with a message about a case_when clause.
  need <- c("predictor", "is_dummy", "is_percentage")
  gone <- setdiff(need, names(type_table))
  if (length(gone) > 0L) {
    stop("type_table is missing column(s): ", paste(gone, collapse = ", "),
         ".\n  It needs predictor (channel name, in store order), is_dummy and ",
         "is_percentage (logical) -- stage 01 writes predictor_type_table.csv.",
         call. = FALSE)
  }
  for (lc in c("is_dummy", "is_percentage")) {
    if (!is.logical(type_table[[lc]])) {
      stop("type_table$", lc, " must be logical, got ", class(type_table[[lc]])[1],
           ".", call. = FALSE)
    }
  }

  store  <- load_patch_store(patch_dir, windows, verbose = verbose)
  points <- align_points_to_meta(points, store$meta)

  # Read from the raster itself, never written by hand: block_size and buffer
  # are given in the SAME units as x/y, and getting that wrong produces either
  # an abort (harmless) or a split that only LOOKS spatial (not).
  if (is.null(cell_size) && !is.null(raster_table)) {
    # LOUD, NOT QUIET. The caller asked for the strongest source of cell_size
    # -- the raster itself. When that source is unusable this used to fall
    # through to the store manifest without a word, and the resolution lock
    # then compared the manifest with itself. Three ways it can be unusable,
    # each named.
    rt <- read_if_path(raster_table, "raster_table_used.csv")
    if (!"raster_file" %in% names(rt)) {
      stop("raster_table has no raster_file column: ", raster_table, call. = FALSE)
    }
    r1 <- rt$raster_file[1]
    if (!file.exists(r1)) {
      stop("raster_table names ", r1, ", which does not exist -- the predictor ",
           "directory moved, or this table was written on another machine.",
           "\n  Pass cell_size = <number> instead, or fix the path.", call. = FALSE)
    }
    if (!requireNamespace("terra", quietly = TRUE)) {
      stop("Reading cell_size from a raster needs the terra package. Install it, ",
           "or pass cell_size = <number>.", call. = FALSE)
    }
    cell_size <- terra::res(terra::rast(r1))[1]
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

  # THE TRANSFORM IS READ, NOT RE-TYPED. Stage 02 recorded "log1p" in the
  # manifest, and stages 04 and 05 then typed its inverse, expm1, by hand --
  # so a store built without the log would have been back-transformed with one
  # anyway. From here the inverse travels with the data. An unknown name stops
  # (target_transform_spec() says why): an inverse that cannot be looked up
  # would give every "native" metric in the wrong space.
  tr_name <- if ("target_transform" %in% names(store$manifest)) {
    as.character(store$manifest$target_transform[1])
  } else recipe$transform
  transform <- if (is.null(tr_name) || is.na(tr_name)) NULL else
    target_transform_spec(tr_name)

  out <- structure(
    list(store = store, points = points, type_table = type_table,
         cell_size = cell_size, patch_dir = patch_dir,
         target_col = target_col %||% store_spec(store)$target_col,
         transform = transform, recipe = recipe),
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
  cat("  transform  : ",
      if (is.null(x$transform)) "(not recorded)" else x$transform$name, "\n", sep = "")
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

# THE DOT COMES FIRST, AND IT HAS TO.
#
# This was `function(kind, ...)`, and R partially matches a named argument
# against any formal that comes BEFORE `...`. So `spatial_cv(k = 5)` called
# `.resample_spec("spatial", k = 5L, ...)`, `k` partial-matched `kind`, and the
# spec came out with `kind = 5L` while the string "spatial" fell into `...`.
#
# The failure that produced was worse than an error. `switch()` on a NUMERIC
# EXPR ignores the alternative names and returns the nth one, so
# resolve_resampling() ran the third branch -- holdout -- and returned a
# perfectly valid one-fold plan to someone who asked for spatial blocks. Only
# region_cv() crashed, because there k is NULL and switch(NULL) cannot pretend.
#
# A formal declared AFTER `...` can only be matched exactly, which is the rule
# that makes this impossible. The leading dot is belt and braces: no argument a
# constructor forwards will ever be called `.kind`.
.resample_spec <- function(..., .kind) {
  stopifnot(is.character(.kind), length(.kind) == 1L)
  structure(c(list(kind = .kind), list(...)), class = "resample_spec")
}

# THE NUMBERS A CONSTRUCTOR IS GIVEN, CHECKED WHERE THEY ARE GIVEN.
#
# as.integer(k) was the whole of it: k = 2.5 became 2 folds without a word,
# k = "5" became 5 by luck, and k = "five" became NA and failed deep in the
# fold assignment, naming an internal. A fraction had no check at all:
# test_frac = 15, meant as 15%, reached the fold code as fifteen times the
# data. A whole number of at least 2, and a fraction in [0, 1), or the
# constructor stops, naming itself.
.check_k <- function(k, what, allow_null = FALSE) {
  if (is.null(k) && allow_null) return(NULL)
  if (!is.numeric(k) || length(k) != 1L || !is.finite(k) || k != round(k) || k < 2) {
    stop(what, "(): k must be a whole number of at least 2 -- the number of folds; got ",
         if (is.null(k)) "NULL" else paste(format(k), collapse = ", "),
         if (is.character(k)) " (a string)" else "", ".", call. = FALSE)
  }
  as.integer(k)
}

.check_frac <- function(x, what, name, zero_ok = TRUE, one_ok = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) || x > 1 || x < 0 ||
      (!zero_ok && x == 0) || (!one_ok && x == 1)) {
    stop(what, "(): ", name, " must be a fraction in ", if (zero_ok) "[0, " else "(0, ",
         if (one_ok) "1]" else "1)", " -- 0.15 for 15%; got ",
         if (is.null(x)) "NULL" else paste(format(x), collapse = ", "), ".", call. = FALSE)
  }
  as.numeric(x)
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
  .resample_spec(.kind = "spatial", k = .check_k(k, "spatial_cv"), block_size = block_size,
                 buffer = buffer, test_frac = .check_frac(test_frac, "spatial_cv", "test_frac"),
                 max_share = .check_frac(max_share, "spatial_cv", "max_share", zero_ok = FALSE,
                                         one_ok = TRUE),
                 buffer_metric = match.arg(buffer_metric), seed = seed)
}

#' Folds matched to where the map will be predicted (kNNDM).
#'
#' The better-founded alternative to spatial_cv(). Blocks need a block size and
#' a buffer width, and nothing in the data says whether the chosen ones were
#' right. kNNDM instead shapes the folds so the distance from a validation point
#' to its nearest training point is distributed like the distance from a
#' PREDICTION pixel to its nearest training point.
#'
#' It therefore needs `predpoints`: a sample of where the map will be drawn.
#' There is no default, because a default would turn the method into an
#' expensive random split. prediction_sample(raster) produces one.
#'
#' When the samples are well spread over the prediction area it converges by
#' itself to ordinary random k-fold -- it does not impose separation that
#' prediction will not face. That is the property blocks cannot have.
#'
#' Needs the CAST and sf packages. See R/knndm.R for the projection question,
#' which is not optional on lon/lat data.
knndm_cv <- function(k = 5L, predpoints = NULL, hold_out_test = FALSE,
                     crs = 4326,
                     project_to = "+proj=moll +lon_0=0 +datum=WGS84 +units=m",
                     seed = 42L, ...) {
  .resample_spec(.kind = "knndm", k = .check_k(k, "knndm_cv"), predpoints = predpoints,
                 hold_out_test = hold_out_test, crs = crs,
                 project_to = project_to, seed = seed, extra = list(...))
}

#' Random k-fold. Ignores geography by construction.
#'
#' Right when the rows really are independent, and the cleanest way to MEASURE
#' what geography is worth: run it against spatial_cv() on the same points and
#' the gap is the spatial optimism.
random_cv <- function(k = 5L, test_frac = 0.15, group = "auto", seed = 42L) {
  .resample_spec(.kind = "random", k = .check_k(k, "random_cv"),
                 test_frac = .check_frac(test_frac, "random_cv", "test_frac"),
                 group = group, seed = seed)
}

#' A single train/validation/test split.
holdout_cv <- function(validation_frac = 0.15, test_frac = 0.15,
                       group = "auto", seed = 42L) {
  validation_frac <- .check_frac(validation_frac, "holdout_cv", "validation_frac", zero_ok = FALSE)
  test_frac <- .check_frac(test_frac, "holdout_cv", "test_frac")
  if (validation_frac + test_frac >= 1) {
    stop("holdout_cv(): validation_frac + test_frac must leave something to train on; got ",
         validation_frac, " + ", test_frac, ".", call. = FALSE)
  }
  .resample_spec(.kind = "holdout", validation_frac = validation_frac,
                 test_frac = test_frac, group = group, seed = seed)
}

#' Leave-region-out, on a grouping that already exists (biome, catchment, ...).
region_cv <- function(group, k = NULL, test_frac = 0.15, seed = 42L) {
  .resample_spec(.kind = "region", group = group, k = .check_k(k, "region_cv", allow_null = TRUE),
                 test_frac = .check_frac(test_frac, "region_cv", "test_frac"), seed = seed)
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

  # Guarded rather than trusted: switch() on a non-character EXPR silently
  # selects by POSITION, which is how a spatial request once came back as a
  # holdout. If kind is ever not one of the four, say so instead of resampling.
  if (!is.character(spec$kind) || length(spec$kind) != 1L) {
    stop("This resample_spec has no usable `kind` (got ",
         paste(class(spec$kind), collapse = "/"), " of length ",
         length(spec$kind), "). Build it with spatial_cv(), knndm_cv(), ",
         "random_cv(), holdout_cv() or region_cv().", call. = FALSE)
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
    knndm   = do.call(knndm_folds, c(
      list(meta = meta, k = spec$k, predpoints = spec$predpoints,
           test_ids = test_ids, hold_out_test = spec$hold_out_test,
           crs = spec$crs, project_to = spec$project_to, seed = spec$seed),
      spec$extra)),
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
#' @param tune_grid  An explicit grid. NULL asks the model for one -- for the
#'   CNN, drawn over the windows the loaded store holds: every window alone
#'   and every pair.
#' @param tune_length How many configurations to try when `tune_grid` is NULL.
#'   The same meaning as caret's: a budget, not a lattice.
#' @param n_seeds    Repetitions per (config, fold). One gives a ranking with
#'   no error bar, which is a ranking of luck as often as of skill.
#' @param transform  NULL (the default) uses the inverse of the transform the
#'   store was built under, which dsm_load() read (`data$transform`). A
#'   function is the inverse to use instead, and is refused if it disagrees
#'   with the store's.
#' @param clamp      c(lower, upper), the plausible range of the target in
#'   native units: every prediction is clipped into it before it is scored.
#'   c(0, Inf) suits a stock or a concentration. A target that can be
#'   negative -- a temperature, a log-ratio -- needs c(-Inf, Inf): clipped at
#'   zero it loses half its predictions, and the metrics still look plausible.
#'   Refused if the data itself falls outside it. The run keeps it, and
#'   dsm_final() refits with the same.
#' @param features   For tabular models: "centre", "window_mean", or both.
#' @param device     A torch device. NULL builds one with setup_torch_device().
#' @param n_cores    Cores for training: torch's threads for the CNN and the
#'   MLP, ranger's for the forest. NULL is the physical cores minus one (see
#'   resolve_cores()) -- except that a `device` passed in keeps the threads it
#'   was set up with unless n_cores is given too.
#' @param test_ids   Sample ids forced into the test set.
#' @param ...        Passed to the underlying runner (n_epochs, patience, ...).
#' @return The runner's result, plus the plan and the data it used.
dsm_train <- function(data, model = "cnn", resampling = spatial_cv(),
                      tune_grid = NULL, tune_length = 20L, n_seeds = 3L,
                      transform = NULL, clamp = c(0, Inf),
                      features = c("centre", "window_mean"),
                      output_dir = "./outputs/tuning",
                      run_id = format(Sys.time(), "%Y%m%d_%H%M%S"),
                      base_seed = 42L, device = NULL, n_cores = NULL,
                      test_ids = NULL, resume = TRUE, evaluate_test = FALSE,
                      verbose = TRUE, ...) {

  # THE DOOR. Each of these used to fail later and worse: a wrong `data`
  # died inside the store code, a caret-style resampling = "cv" died in a
  # stopifnot() naming an internal expression, and a misspelt training
  # argument travelled through `...` to fail at the FIRST UNIT -- after the plan
  # was checked and written and the first fold cache built -- for the CNN, or
  # to be swallowed by the table runners' own `...` for rf and mlp.
  if (!inherits(data, "dsm_data")) {
    stop("`data` must come from dsm_load(); got a ", class(data)[1], ".",
         call. = FALSE)
  }
  if (is.character(model)) model <- get_model(model)
  if (!inherits(model, "model_spec")) {
    stop("`model` must be a registered name (\"cnn\", \"rf\", \"mlp\", ...) or ",
         "a model_spec(); got a ", class(model)[1], ".", call. = FALSE)
  }
  if (!inherits(resampling, c("resample_spec", "fold_plan"))) {
    stop("`resampling` must be built with spatial_cv(), knndm_cv(), random_cv(), ",
         "holdout_cv() or region_cv(), or be a fold_plan -- got a ",
         class(resampling)[1], ". caret-style strings such as \"cv\" are not ",
         "accepted here.", call. = FALSE)
  }
  # The transform and the cores are checked at the door too: both cost a
  # second here, and the run's first unit to get wrong.
  transform <- .resolve_train_transform(transform, data, verbose = verbose)
  clamp <- .check_train_clamp(clamp, data)
  if (!is.null(n_cores)) n_cores <- suppressMessages(resolve_cores(n_cores, what = "training"))

  if (identical(model$input, "patches")) {
    dots <- names(list(...))
    internal <- c("cfg", "n_channels", "loaders", "points_valid", "transform",
                  "device", "clamp")
    allowed <- c(setdiff(names(formals(train_one_cnn)), internal), "release_store")
    bad <- setdiff(dots, allowed)
    if (length(bad) > 0L) {
      stop("dsm_train() does not know argument(s): ", paste(bad, collapse = ", "),
           ".\n  Training options for the CNN are: ",
           paste(sort(allowed), collapse = ", "), call. = FALSE)
    }
    # A grid given by hand is checked against the parameter space here,
    # before the plan: a missing or misspelt column used to surface at the
    # first unit, or never (.check_cnn_grid()).
    if (!is.null(tune_grid)) tune_grid <- .check_cnn_grid(tune_grid, verbose = verbose)
  }

  # The buffer is derived from the LARGEST window in play. With an explicit
  # grid that is the largest window the grid asks for; with tune_length the
  # grid does not exist yet -- it is drawn after the plan, because a default
  # grid may need the fold's data -- so every loaded window counts.
  #
  # That errs WIDE, and wide is the safe direction: a buffer larger than
  # necessary drops a few more training points, while one that is too small
  # leaves validation patches sharing pixels with training and reports a clean
  # zero for it.
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
      n_train_min <- min(vapply(plan$folds, function(f) length(f$train), integer(1)))
      tune_grid <- .draw_default_grid(model, tune_length, base_seed,
                                      data$store$window_sizes, n_train_min,
                                      verbose)
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
    # The threads belong to the session, not to the device: a device passed
    # in keeps whatever its caller set, unless n_cores says otherwise.
    if (is.null(device)) {
      device <- setup_torch_device(n_threads = n_cores)
    } else if (!is.null(n_cores)) {
      set_torch_threads(n_cores)
    }
    .train_clamp_record(file.path(output_dir, run_id), clamp, resume)
    res <- run_cnn_resample(
      tune_grid = tune_grid, store = data$store, points = data$points,
      type_table = data$type_table, plan = plan, transform = transform,
      output_dir = output_dir, device = device, run_id = run_id,
      base_seed = base_seed, n_seeds = n_seeds, resume = resume,
      evaluate_test = evaluate_test, clamp = clamp, ...)
  } else {
    # A torch model on the table path (the MLP) gets its threads the same way;
    # the forest reads n_cores itself, through the runner's `...`.
    if (!is.null(n_cores) && isNamespaceLoaded("torch")) set_torch_threads(n_cores)
    .train_clamp_record(file.path(output_dir, run_id), clamp, resume)
    res <- run_table_resample(
      model = model, tune_grid = tune_grid, store = data$store,
      points = data$points, type_table = data$type_table, plan = plan,
      features = features, transform = transform, output_dir = output_dir,
      run_id = run_id, base_seed = base_seed, n_seeds = n_seeds,
      tune_length = tune_length, device = device, resume = resume,
      evaluate_test = evaluate_test, clamp = clamp, n_cores = n_cores, ...)
  }

  res$clamp <- clamp
  res$model <- model$name
  res$data  <- data
  class(res) <- c("dsm_fit", class(res))
  if (verbose) print(res)
  res
}

# THE INVERSE TRAVELS WITH THE DATA. dsm_train() took `transform = identity`,
# and stage 03 passed expm1 by hand -- so a log1p store trained through the
# front door without that argument would have reported every "native" metric
# in log space: well-formed, plausible and wrong. NULL now means the store's
# own inverse. A function given explicitly is checked against it on a few
# values and refused if the two disagree; an equivalent written differently
# (function(z) exp(z) - 1) passes. With no transform recorded -- a store built
# by hand -- there is nothing to check against: NULL means identity, and the
# message says so.
.resolve_train_transform <- function(transform, data, verbose = TRUE) {
  rec <- data$transform
  if (is.null(transform)) {
    if (is.null(rec)) {
      if (verbose) {
        message("The store records no target transform: metrics are computed ",
                "on the target as stored.")
      }
      return(identity)
    }
    return(rec$inverse)
  }
  if (!is.function(transform)) {
    stop("`transform` must be NULL (the store's own inverse) or a function -- ",
         "the inverse of the target transform, such as expm1. Got a ",
         class(transform)[1], ".", call. = FALSE)
  }
  if (is.null(rec)) return(transform)
  probe <- c(0, 0.5, 1, 2.5, 5)
  want  <- rec$inverse(probe)
  got   <- suppressWarnings(tryCatch(as.numeric(transform(probe)),
                                     error = function(e) rep(NA_real_, length(probe))))
  if (length(got) != length(want) || !isTRUE(all.equal(got, want))) {
    stop("`transform` disagrees with the store. It was built under \"", rec$name,
         "\", whose inverse gives ", paste(signif(want, 4), collapse = ", "),
         " at ", paste(probe, collapse = ", "), "; the function passed gives ",
         paste(signif(got, 4), collapse = ", "), ".\n  Leave transform = NULL ",
         "to use the store's own: another inverse puts every native-unit metric ",
         "in the wrong space.", call. = FALSE)
  }
  transform
}

# THE CLAMP IS AT THE DOOR, NOT IN `...`. It is the one training argument that
# destroys predictions silently: every native prediction is clipped into it
# before it is scored, so a target that can be negative -- a temperature, a
# log-ratio -- clipped at the default zero loses half its predictions while
# the metrics still come out plausible. A formal now, checked for its shape,
# and checked against the data: an observed target outside the clamp is a
# clamp that would clip what the data itself says is possible.
.check_train_clamp <- function(clamp, data) {
  if (!is.numeric(clamp) || length(clamp) != 2L || anyNA(clamp) || clamp[1] > clamp[2]) {
    stop("clamp must be c(lower, upper) with lower <= upper and no NA: c(0, Inf) for a ",
         "stock or a concentration, c(-Inf, Inf) for a target that can be negative.",
         call. = FALSE)
  }
  y <- data$store$meta$target_native
  y <- y[is.finite(y)]
  out <- y < clamp[1] | y > clamp[2]
  if (any(out)) {
    stop(sprintf("clamp = c(%s, %s) would clip the data itself: %d of %d observed target value(s) lie outside it (the data runs from %s to %s). Every prediction is clipped into the clamp before it is scored -- widen it, e.g. clamp = c(-Inf, Inf).",
                 format(clamp[1]), format(clamp[2]), sum(out), length(y),
                 format(signif(min(y), 4)), format(signif(max(y), 4))),
         call. = FALSE)
  }
  as.numeric(clamp)
}

# The clamp a run scored its units with is part of the run. Resumed with
# another, its early units and its late ones would be two metrics in one
# table, and the selection and the calibration residuals would mix them. So
# it is written before the first unit, a resume with a different one is
# refused, and dsm_final() refits with it (.final_clamp()).
.train_clamp_record <- function(run_dir, clamp, resume) {
  f <- file.path(run_dir, "clamp.rds")
  if (isTRUE(resume) && file.exists(f)) {
    old <- as.numeric(readRDS(f))
    if (!identical(old, as.numeric(clamp))) {
      stop(sprintf("This run was started with clamp = c(%s, %s) and is being resumed with c(%s, %s): its units would be scored two ways. Resume with the same clamp, or start another run_id.",
                   format(old[1]), format(old[2]), format(clamp[1]), format(clamp[2])),
           call. = FALSE)
    }
    return(invisible(f))
  }
  create_output_dirs(run_dir)
  safe_save_rds(as.numeric(clamp), f, compress = FALSE)
  invisible(f)
}

# A default grid drawn over what the store and the plan can serve. The
# windows and the smallest fold go to the generator when it declares them (see
# default_grid in R/model_registry.R); a generator that does not is called as
# before. What was drawn is said, because a default nobody sees is a default
# nobody can question.
.draw_default_grid <- function(model, tune_length, seed, windows, n_train = NULL,
                               verbose = TRUE) {
  takes <- names(formals(model$default_grid))
  args  <- list(tune_length, seed)
  if ("windows" %in% takes) args$windows <- windows
  if ("n_train" %in% takes && !is.null(n_train)) args$n_train <- n_train
  grid <- do.call(model$default_grid, args)
  if (verbose) {
    said <- sprintf("%d config(s)", nrow(grid))
    if ("windows" %in% takes) {
      said <- c(said, paste0("over the store's windows ", paste(windows, collapse = ", ")))
    }
    if ("window_sizes" %in% names(grid)) {
      drawn <- vapply(unique(grid$window_sizes), paste, character(1), collapse = "+")
      said <- c(said, paste0("window options drawn: ", paste(drawn, collapse = ", ")))
    }
    if ("batch_size" %in% names(grid)) {
      said <- c(said, paste0("batch sizes: ", paste(sort(unique(grid$batch_size)), collapse = ", "),
                             if ("n_train" %in% takes && !is.null(n_train))
                               sprintf(" (smallest fold trains on %s points)",
                                       format(n_train, big.mark = ",")) else ""))
    }
    message("Default grid: ", paste(said, collapse = " | "))
  }
  grid
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
