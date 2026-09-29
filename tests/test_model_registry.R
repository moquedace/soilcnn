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
# Run: source("<package root>/tests/test_model_registry.R")     (torch for tensors
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
  stop("Project root not found. setwd() to the package root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
.load_framework(root)

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
  # 5a. RESUME MATCHES ON THE CONFIGURATION, NOT ON ITS NAME
  #
  # Both runners skip a unit whose unit_id is already in the comparison table,
  # and a unit_id is "<config_id>_f<fold>_s<seed>". That is sound only while
  # config_id denotes the same hyperparameters it denoted when the unit was
  # fitted. Fixing rf_grid() changed exactly that: rf_001 used to be
  # mtry_frac = 0.1 and is now 1/3. Resuming on the name alone would have
  # reported results for a forest that was never fitted.
  #
  # The second case here is the one that makes the guard dangerous rather than
  # merely useful. The grid holds window_sizes as a list-column c(5L, 7L); the
  # comparison row holds the string "5x7". Compared literally, every CNN unit
  # ever written is stale, and the guard silently triggers a full retrain --
  # hours of GPU time, and it looks exactly like a successful resume.
  # ===========================================================================

  grid_a <- tibble::tibble(config_id = c("rf_001", "rf_002"),
                           mtry_frac = c(1/3, 0.10),
                           min_node_size = c(5L, 5L))
  cmp_a  <- tibble::tibble(
    unit_id   = c("rf_001_f1_s1", "rf_001_f1_s2", "rf_002_f1_s1"),
    config_id = c("rf_001", "rf_001", "rf_002"),
    status    = "success",
    mtry_frac = c(0.10, 0.10, 0.10),          # the OLD rf_001, and rf_002 matches
    min_node_size = c(5L, 5L, 5L),
    val_ccc   = c(0.46, 0.47, 0.46))          # an outcome: not part of identity
  done_a <- cmp_a$unit_id

  keep_a <- suppressMessages(.resumable_units(done_a, cmp_a, grid_a))
  ok["resume_drops_a_renamed_config"] <-
    identical(sort(keep_a), "rf_002_f1_s1")

  # Nothing changed: nothing may be dropped. A guard that refits what is
  # already correct is indistinguishable, from the outside, from no resume.
  ok["resume_keeps_an_unchanged_config"] <- {
    g <- tibble::tibble(config_id = c("rf_001", "rf_002"),
                        mtry_frac = c(0.10, 0.10),
                        min_node_size = c(5L, 5L))
    identical(sort(suppressMessages(.resumable_units(done_a, cmp_a, g))),
              sort(done_a))
  }

  # THE EXPENSIVE FALSE POSITIVE: list-column vs flattened string.
  ok["resume_matches_across_representations"] <- {
    g <- tibble::tibble(config_id = c("cfg_001", "cfg_002"),
                        window_sizes  = list(c(5L, 7L), 7L),
                        conv_channels = list(c(32L, 64L), c(16L, 32L)),
                        base_lr = c(1e-4, 3e-4))
    cmp <- tibble::tibble(
      unit_id   = c("cfg_001_f1_s1", "cfg_002_f1_s1"),
      config_id = c("cfg_001", "cfg_002"),
      status    = "success",
      window_sizes  = c("5x7", "7"),
      conv_channels = c("32_64", "16_32"),
      base_lr = c(1e-4, 3e-4))
    identical(sort(suppressMessages(
      .resumable_units(cmp$unit_id, cmp, g))), sort(cmp$unit_id))
  }

  # ...and it must still SEE a real change hiding in that representation.
  ok["resume_sees_a_changed_window"] <- {
    g <- tibble::tibble(config_id = "cfg_001", window_sizes = list(c(5L, 9L)))
    cmp <- tibble::tibble(unit_id = "cfg_001_f1_s1", config_id = "cfg_001",
                          status = "success", window_sizes = "5x7")
    length(suppressMessages(.resumable_units(cmp$unit_id, cmp, g))) == 0L
  }

  # A cached config the grid no longer mentions is not stale -- it is simply
  # not asked for, and its unit_id can never be generated. Leave the record.
  ok["resume_ignores_configs_not_in_the_grid"] <- {
    g <- tibble::tibble(config_id = "rf_002", mtry_frac = 0.10,
                        min_node_size = 5L)
    "rf_001_f1_s1" %in% suppressMessages(.resumable_units(done_a, cmp_a, g))
  }

  ok["resume_survives_an_empty_history"] <-
    identical(.resumable_units(character(0), tibble::tibble(), grid_a),
              character(0))

  # ===========================================================================
  # 5b. THE GRIDS MUST NOT REPEAT THEMSELVES
  #
  # This is a post-mortem test. 03b ran four RF configs per family and three of
  # them were the SAME config: rf_grid() drew mtry_frac from four values with
  # replacement and never de-duplicated, so the run spent 75% of the forest
  # budget re-measuring one setting, and the "four-point" baseline the CNN was
  # compared against rested on two distinct forests.
  #
  # Worse, every draw landed on the smallest mtry (0.1p), well below the p/3
  # regression default -- so the baseline was not merely narrow, it was
  # HANDICAPPED, in the direction that flatters the CNN. A baseline that is
  # accidentally weak does not fail loudly; it quietly confirms the hypothesis.
  #
  # Hence three properties, checked for every tune_length a person might use:
  #   rows are distinct, the count is what was asked for, and the first row is
  #   the textbook default rather than an arbitrary corner of the space.
  # ===========================================================================

  ok["rf_grid_rows_are_distinct"] <- all(vapply(1:6, function(k) {
    g <- rf_grid(k, seed = 1L)
    nrow(unique(g[, c("mtry_frac", "min_node_size")])) == nrow(g)
  }, logical(1)))

  ok["rf_grid_gives_what_was_asked"] <- all(vapply(1:6, function(k) {
    nrow(rf_grid(k, seed = 1L)) == k
  }, logical(1)))

  # tune_length = 1 must be the forest everyone else would have fitted, so a
  # single-config baseline is a FAIR baseline and not a random one.
  ok["rf_grid_starts_at_the_default"] <- {
    g <- rf_grid(1L)
    isTRUE(all.equal(g$mtry_frac[1], 1/3)) && g$min_node_size[1] == 5L
  }

  # With n_features the fractions become counts, and two fractions can round to
  # one count on a narrow table. Distinct rows must stay distinct AS FITTED.
  ok["rf_grid_distinct_after_rounding"] <- all(vapply(c(9L, 20L, 181L, 724L),
    function(p) {
      g <- rf_grid(6L, n_features = p)
      nrow(unique(g[, c("mtry", "min_node_size")])) == nrow(g)
    }, logical(1)))

  ok["rf_grid_mtry_is_within_the_table"] <- {
    g <- rf_grid(6L, n_features = 181L); all(g$mtry >= 1L & g$mtry <= 181L)
  }

  # The MLP grid is a genuine random draw over four interacting axes -- that is
  # what random search is for -- but a repeated draw is still waste.
  ok["mlp_grid_rows_are_distinct"] <- all(vapply(c(2L, 4L, 8L), function(k) {
    g <- mlp_grid(k, seed = 7L)
    nrow(unique(g[, setdiff(names(g), "config_id")])) == nrow(g)
  }, logical(1)))

  ok["mlp_grid_ids_are_unique"] <- {
    g <- mlp_grid(8L, seed = 7L); !anyDuplicated(g$config_id)
  }

  # ===========================================================================
  # 6-7. A real model through the registry's contract
  # ===========================================================================

  if (requireNamespace("randomForest", quietly = TRUE) ||
      requireNamespace("ranger", quietly = TRUE)) {
    xs <- tv$train$x
    ys <- c(1.0, 2.0, 3.0, 4.0)
    g  <- rf_grid(1L, seed = 1L)
    g$n_trees <- 20L                     # a contract test, not a fit quality one

    spec <- get_model("rf")
    # suppressWarnings: randomForest objects to a response with five or fewer
    # unique values, and the fixture has four ON PURPOSE. What is being tested
    # is the spec's CONTRACT -- fit takes (x, y, cfg), predict takes
    # (object, x) -- not whether a forest of four points is any good.
    fitted <- suppressWarnings(spec$fit(x = xs, y = ys, cfg = g[1, ]))
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
