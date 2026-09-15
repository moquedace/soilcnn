# Unit test: the model registry and the tabular view of a fold
#
# These two exist so the CNN's CCC has a scale on it. That makes their failure
# modes quiet ones:
#
#   a registry that accepts a broken spec fails minutes later, inside a fold
#   a table view built from the wrong window answers about another place
#   a feature matrix whose columns drift between fit and predict is a model
#     predicting confidently from the wrong covariates, with no error at all
#
# Verified:
#   1. model_spec() refuses a fit/predict with the wrong contract
#   2. the registry refuses a silent overwrite, and allows a declared one
#   3. fold_table_view() takes the centre from the right pixel
#   4. window means are the means, and a 1x1 window contributes no duplicate
#   5. the centre disagreement between windows is DETECTED, not averaged away
#   6. the RF spec fits and predicts through the registry's contract
#   7. a reordered prediction table is refused
#
# Run: source("D:/.../tests/test_model_registry.R")     (torch for tensors
#      only -- nothing is trained)

suppressMessages({
  library(tibble)
  library(dplyr)
})

# -- project root: works under source() in the console AND under Rscript ------

root <- (function() {
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
  }
  cand <- c(cand, getwd())
  for (d in cand) {
    for (up in c(".", "..")) {
      r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
      if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
    }
  }
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "metrics.R"))
source(file.path(root, "R", "resample.R"))
source(file.path(root, "R", "dataset.R"))
source(file.path(root, "R", "model_registry.R"))

ok <- c()

# =============================================================================
# 1. model_spec() checks the contract at declaration time
#
# Minutes into a fold is the wrong moment to learn that an argument is named
# `data` instead of `x`.
# =============================================================================

good_fit  <- function(x, y, cfg, ...) list(mean = mean(y))
good_pred <- function(object, x, ...) rep(object$mean, nrow(x))

ok["spec_accepts_a_correct_contract"] <- !inherits(
  tryCatch(model_spec("t1", "table", good_fit, good_pred),
           error = function(e) e), "error")

ok["spec_refuses_fit_without_x"] <- inherits(
  tryCatch(model_spec("t2", "table", function(data, y, cfg) NULL, good_pred),
           error = function(e) e), "error")

ok["spec_refuses_predict_without_object"] <- inherits(
  tryCatch(model_spec("t3", "table", good_fit, function(m, x) NULL),
           error = function(e) e), "error")

ok["spec_refuses_a_non_function"] <- inherits(
  tryCatch(model_spec("t4", "table", "not a function", good_pred),
           error = function(e) e), "error")

ok["spec_refuses_an_unknown_input"] <- inherits(
  tryCatch(model_spec("t5", "rasters", good_fit, good_pred),
           error = function(e) e), "error")

# =============================================================================
# 2. The registry
# =============================================================================

sp <- model_spec("test_dummy", "table", good_fit, good_pred,
                 description = "a constant")
register_model(sp, overwrite = TRUE)

ok["registered_model_comes_back"] <-
  identical(get_model("test_dummy")$name, "test_dummy")

# A silent replacement is how two different models answer to one name in the
# same session, with the results carrying no mark of which one ran.
ok["refuses_silent_overwrite"] <- inherits(
  tryCatch(register_model(sp), error = function(e) e), "error")
ok["allows_declared_overwrite"] <- !inherits(
  tryCatch(register_model(sp, overwrite = TRUE), error = function(e) e), "error")

ok["unknown_model_names_the_alternatives"] <- {
  m <- tryCatch(get_model("nope"), error = function(e) conditionMessage(e))
  is.character(m) && grepl("test_dummy", m)
}

lm_ <- list_models()
ok["list_models_reports_input"] <-
  identical(lm_$input[lm_$name == "test_dummy"], "table")

# =============================================================================
# 3-5. fold_table_view()
#
# A cache built by hand with values that make the right answer arithmetic:
# channel c of point i is filled with (i * 10 + c), and the centre pixel alone
# is overwritten with a marker. The mean and the centre are then two DIFFERENT
# known numbers, so a function that confuses them cannot pass both.
# =============================================================================

if (requireNamespace("torch", quietly = TRUE)) {

  N <- 6L; C <- 3L
  mk <- function(w, centre_marker = TRUE) {
    a <- array(0, dim = c(N, C, w, w))
    for (i in seq_len(N)) for (cc in seq_len(C)) a[i, cc, , ] <- i * 10 + cc
    if (centre_marker && w > 1L) {
      ctr <- (w %/% 2L) + 1L
      for (i in seq_len(N)) for (cc in seq_len(C)) {
        a[i, cc, ctr, ctr] <- -(i * 10 + cc)     # sign-flipped: unmistakable
      }
    }
    torch::torch_tensor(a, dtype = torch::torch_float())
  }

  idx <- list(train = 1:4, validation = 5:6)
  cache <- list(
    train = list(w03 = mk(3L)[1:4, , , , drop = FALSE],
                 w05 = mk(5L)[1:4, , , , drop = FALSE],
                 y   = torch::torch_tensor(as.numeric(1:4))$view(c(-1L, 1L))),
    validation = list(w03 = mk(3L)[5:6, , , , drop = FALSE],
                      w05 = mk(5L)[5:6, , , , drop = FALSE],
                      y   = torch::torch_tensor(as.numeric(5:6))$view(c(-1L, 1L)))
  )
  preds <- c("a", "b", "c")

  tv <- fold_table_view(cache, preds, features = c("centre", "window_mean"))

  ok["table_has_every_role"] <- identical(sort(names(tv)),
                                          c("train", "validation"))
  ok["table_row_counts_match_the_fold"] <-
    nrow(tv$train$x) == 4L && nrow(tv$validation$x) == 2L

  # centre + means for w03 and w05 = 3 * 3 = 9 columns
  ok["table_column_count"] <- ncol(tv$train$x) == 9L
  ok["table_centre_columns_named_after_predictors"] <-
    identical(colnames(tv$train$x)[1:3], preds)
  ok["table_mean_columns_carry_the_window"] <-
    identical(colnames(tv$train$x)[4:6], paste0(preds, "_mean_w03"))

  # 3. the CENTRE is the centre: the sign-flipped marker, not the fill
  ok["centre_is_the_centre_pixel"] <-
    isTRUE(all.equal(unname(tv$train$x[, 1]), -c(11, 21, 31, 41),
                     tolerance = 1e-5))
  ok["centre_is_per_channel"] <-
    isTRUE(all.equal(unname(tv$train$x[, 3]), -c(13, 23, 33, 43),
                     tolerance = 1e-5))

  # 4. the MEAN is the mean, and it is NOT the centre: for a 3x3 window, eight
  # cells hold v and one holds -v, so the mean is v * 7/9.
  v <- c(11, 21, 31, 41)
  ok["window_mean_w03_is_arithmetic"] <-
    isTRUE(all.equal(unname(tv$train$x[, 4]), v * 7 / 9, tolerance = 1e-5))
  # 5x5: 24 cells hold v, one holds -v -> v * 23/25
  ok["window_mean_w05_is_arithmetic"] <-
    isTRUE(all.equal(unname(tv$train$x[, 7]), v * 23 / 25, tolerance = 1e-5))

  ok["table_y_survives"] <- isTRUE(all.equal(tv$train$y, as.numeric(1:4)))

  # centre only, and the column count follows
  tv_c <- fold_table_view(cache, preds, features = "centre")
  ok["centre_only_is_C_columns"] <- ncol(tv_c$train$x) == 3L

  # A 1x1 window's mean IS its centre, and a duplicated column is one a tree
  # can split on twice for free.
  cache1 <- list(train = list(w01 = mk(1L, FALSE)[1:4, , , , drop = FALSE],
                              y = cache$train$y))
  tv1 <- fold_table_view(cache1, preds)
  ok["w01_contributes_no_duplicate_mean"] <- ncol(tv1$train$x) == 3L

  # 5. a store whose windows were cut around DIFFERENT points must be caught:
  # every table feature built from it describes another location than the
  # tensors do, and no metric would ever reveal it.
  bad <- cache
  bad$train$w05 <- bad$train$w05 + 1000
  ok["disagreeing_centres_are_refused"] <- inherits(
    tryCatch(fold_table_view(bad, preds), error = function(e) e), "error")

  ok["missing_window_is_refused"] <- inherits(
    tryCatch(fold_table_view(cache, preds, windows = c(3L, 9L)),
             error = function(e) e), "error")

  # ===========================================================================
  # 6-7. A real model through the registry's contract
  # ===========================================================================

  if (requireNamespace("randomForest", quietly = TRUE) ||
      requireNamespace("ranger", quietly = TRUE)) {
    source(file.path(root, "R", "baselines.R"))

    xs <- tv$train$x
    ys <- c(1.0, 2.0, 3.0, 4.0)
    g  <- rf_grid(1L, seed = 1L)
    g$n_trees <- 20L                     # a contract test, not a fit quality one

    spec <- get_model("rf")
    fitted <- spec$fit(x = xs, y = ys, cfg = g[1, ])
    p <- spec$predict(fitted, tv$validation$x)

    ok["rf_fits_through_the_spec"]  <- inherits(fitted, "rf_fitted")
    ok["rf_predicts_one_per_row"]   <- length(p) == nrow(tv$validation$x)
    ok["rf_predictions_are_finite"] <- all(is.finite(p))
    ok["rf_reports_its_size"]       <- {
      np <- spec$count_params(fitted); is.numeric(np) && np > 0
    }

    # 7. columns in another order: a forest reads features by POSITION, so this
    # is the case that produces a confident answer from the wrong covariates.
    shuffled <- tv$validation$x[, c(2:9, 1), drop = FALSE]
    ok["rf_refuses_reordered_columns"] <- inherits(
      tryCatch(spec$predict(fitted, shuffled), error = function(e) e), "error")

    cat("  RF backend               : ", fitted$backend, "\n", sep = "")
  } else {
    cat("  RF skipped               : neither ranger nor randomForest installed\n")
  }

  # ===========================================================================
  # 8. The caret adapter
  #
  # caret is borrowed for its model LIBRARY, never for its resampling. What
  # must hold is that a borrowed method arrives as an ordinary model_spec and
  # that nothing downstream can tell the difference.
  # ===========================================================================

  if (requireNamespace("caret", quietly = TRUE)) {
    source(file.path(root, "R", "caret_adapter.R"))

    ok["caret_lists_regression_methods"] <- {
      av <- caret_available("^rf$|^ranger$|^glmnet$")
      is.data.frame(av) && nrow(av) > 0
    }

    ok["caret_refuses_an_unknown_method"] <- inherits(
      tryCatch(caret_model_info("definitely_not_a_method"),
               error = function(e) e), "error")

    sp_c <- caret_spec("rf", name = "caret_rf")
    ok["caret_spec_is_a_model_spec"] <- inherits(sp_c, "model_spec")
    ok["caret_spec_consumes_a_table"] <- identical(sp_c$input, "table")

    # The generator needs real data, and says so rather than inventing some.
    ok["caret_default_grid_refuses_without_x"] <- inherits(
      tryCatch(sp_c$default_grid(3L, 42L), error = function(e) e), "error")

    if (requireNamespace("randomForest", quietly = TRUE)) {
      # 12 rows: caret::train needs enough to fit, and the point is the
      # contract, not the fit.
      set.seed(7)
      xc <- matrix(stats::rnorm(12 * 4), nrow = 12,
                   dimnames = list(NULL, c("p1", "p2", "p3", "p4")))
      yc <- as.numeric(xc[, 1] * 2 + stats::rnorm(12, sd = 0.1))

      gc_ <- sp_c$default_grid(2L, 42L, x = xc, y = yc)
      ok["caret_grid_has_config_id"] <- "config_id" %in% names(gc_)
      ok["caret_grid_has_the_methods_parameters"] <- "mtry" %in% names(gc_)
      ok["caret_grid_config_ids_are_unique"] <-
        !anyDuplicated(gc_$config_id)

      fc <- sp_c$fit(x = xc, y = yc, cfg = gc_[1, ])
      pc <- sp_c$predict(fc, xc)
      ok["caret_fits_through_the_spec"]  <- inherits(fc, "caret_fitted")
      ok["caret_predicts_one_per_row"]   <- length(pc) == nrow(xc)
      ok["caret_predictions_are_finite"] <- all(is.finite(pc))

      # caret is NOT allowed to hold the training data: at 30k x 360 that is
      # ~85 MB per unit, and a run holds many units.
      ok["caret_does_not_keep_the_data"] <- is.null(fc$fit$trainingData)

      ok["caret_refuses_reordered_columns"] <- inherits(
        tryCatch(sp_c$predict(fc, xc[, c(2, 1, 3, 4), drop = FALSE]),
                 error = function(e) e), "error")

      # A config row missing the method's parameters must fail AT FIT, naming
      # the fix -- not reach caret and come back as a cryptic tuneGrid error.
      ok["caret_refuses_a_grid_missing_parameters"] <- inherits(
        tryCatch(sp_c$fit(x = xc, y = yc,
                          cfg = tibble::tibble(config_id = "x", nonsense = 1)),
                 error = function(e) e), "error")

      cat("  caret adapter            : rf borrowed, fitted and predicted\n")
    }
  } else {
    cat("  caret adapter skipped    : caret not installed\n")
  }

} else {
  cat("  table view skipped       : torch not available\n")
}

.report(ok, "test_model_registry")
