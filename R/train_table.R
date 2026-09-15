# ── Running a tabular model over a fold plan ──────────────────────────────────
#
# WHY A SECOND RUNNER, AND NOT A GENERALISED FIRST ONE.
#
# run_cnn_resample() could have been widened to take a model_spec and branch on
# `input`. It was not, deliberately.
#
# The CNN path carries epoch histories, gate analyses, per-quantile metrics,
# checkpoint files, DataLoaders and a device -- none of which a Random Forest
# has, and every one of which would become an `if` inside the one function this
# project cannot afford to destabilise. Five restarts of this project have all
# had the same cause: a defect introduced upstream of where it was noticed.
# Rewriting the trained path days before the definitive run is exactly that
# move.
#
# What the two runners DO share is the only thing that has to be shared: the
# fold plan, the seed discipline, and the SHAPE OF THE COMPARISON TABLE. That
# is what makes summarise_resamples(), seed_noise_floor() and one_se() work
# unchanged across model families, which is the whole point of measuring
# baselines at all.
#
# If a third tabular family arrives it costs a model_spec and nothing else.
# If the CNN path and this one ever converge on what they need, they can be
# merged then, against two real implementations instead of one imagined
# interface.

#' Build the prediction table for a tabular model, in the CNN's shape.
#'
#' Mirrors predict_loader() exactly, so downstream code cannot tell which
#' family produced a row -- and so a metric can never be computed one way for
#' one model and another way for another.
#'
#' @param pred_raw     Predictions in TRANSFORM space (what the model returns).
#' @param points_valid Metadata rows for this role, in the same order.
#' @param dataset_role "train", "validation" or "test".
#' @param transform    Inverse of the target transform, e.g. expm1.
#' @param clamp        Range applied in native space.
predict_table <- function(pred_raw, points_valid, dataset_role,
                          transform = identity, clamp = c(0, Inf)) {
  check_point_contract(points_valid, what = "points_valid")
  if (length(pred_raw) != nrow(points_valid)) {
    stop("Prediction length != metadata rows for split: ", dataset_role,
         call. = FALSE)
  }
  pred_native <- transform(pred_raw)
  if (is.finite(clamp[1])) pred_native <- pmax(pred_native, clamp[1])
  if (is.finite(clamp[2])) pred_native <- pmin(pred_native, clamp[2])
  obs_native <- as.numeric(points_valid$target_native)

  tibble::tibble(
    profile_id          = points_valid$profile_id,
    sample_id           = points_valid$sample_id,
    dataset_role        = dataset_role,
    obs                 = obs_native,
    pred                = pred_native,
    obs_transform       = as.numeric(points_valid$target_transform),
    pred_transform      = pred_raw,
    residual            = pred_native - obs_native,
    abs_error           = abs(pred_native - obs_native),
    residual_transform  = pred_raw - as.numeric(points_valid$target_transform),
    abs_error_transform = abs(pred_raw - as.numeric(points_valid$target_transform))
  )
}

#' Fit a registered tabular model over every fold, config and seed.
#'
#' @param model      A model_spec with input == "table", or its registered name.
#' @param tune_grid  Tibble of configs, one row each, with a config_id column.
#'   NULL uses the model's default_grid().
#' @param store,points,type_table,plan  As in run_cnn_resample().
#' @param features   Which tabular features to build: "centre", "window_mean".
#' @param windows    Windows to summarise, or NULL for every one loaded.
#' @param transform  Inverse of the target transform.
#' @param n_seeds    Repetitions per (config, fold). One gives no error bar.
#' @return list(comparison, by_config, run_dir, plan).
run_table_resample <- function(model, tune_grid = NULL, store, points,
                               type_table, plan,
                               features    = c("centre", "window_mean"),
                               windows     = NULL,
                               transform   = identity,
                               output_dir  = "./outputs/tuning",
                               run_id      = format(Sys.time(), "%Y%m%d_%H%M%S"),
                               base_seed   = 42L,
                               n_seeds     = 1L,
                               tune_length = 6L,
                               device      = NULL,
                               resume      = TRUE,
                               evaluate_test = FALSE,
                               clamp       = c(0, Inf),
                               ...) {

  if (is.character(model)) model <- get_model(model)
  stopifnot(inherits(model, "model_spec"), inherits(plan, "fold_plan"))
  if (!identical(model$input, "table")) {
    stop("run_table_resample() needs a model whose input is 'table'; '",
         model$name, "' consumes '", model$input, "'.", call. = FALSE)
  }
  n_seeds <- as.integer(n_seeds)
  stopifnot(n_seeds >= 1L)

  # THE GRID MAY NOT EXIST YET, AND THAT IS DELIBERATE.
  #
  # A default grid can need the real training data -- mtry is a fraction of
  # ncol(x), and glmnet's lambda path is computed from the values themselves.
  # The data only exists once the first fold's table is built, so generation is
  # deferred to there rather than faked here against a synthetic matrix.
  #
  # An explicit tune_grid skips all of this and is validated immediately: a
  # missing config_id should not be found one fold into the run.
  if (!is.null(tune_grid) && !"config_id" %in% names(tune_grid)) {
    stop("tune_grid must have a config_id column.", call. = FALSE)
  }
  if (is.null(tune_grid) && is.null(model$default_grid)) {
    stop("Model '", model$name, "' has no default_grid(); pass tune_grid.",
         call. = FALSE)
  }

  fold_sizes <- check_fold_plan(plan, meta = store$meta)
  message("\n-- Resampling plan --")
  print(plan)
  if (!is.null(tune_grid)) {
    message("\nModel: ", model$name, " | ", nrow(tune_grid), " config(s) x ",
            plan$n_folds, " fold(s) x ", n_seeds, " seed(s) = ",
            nrow(tune_grid) * plan$n_folds * n_seeds, " units")
  } else {
    message("\nModel: ", model$name, " | grid generated from fold 1's data")
  }

  run_dir <- file.path(output_dir, run_id)
  create_output_dirs(file.path(run_dir, c("comparison", "predictions",
                                          "metrics", "history")))

  # Written BEFORE any fitting: results whose folds cannot be reconstructed are
  # results that cannot be defended.
  safe_save_rds(plan, file.path(run_dir, "fold_plan.rds"), compress = FALSE)
  safe_write_csv2(dplyr::mutate(fold_sizes, method = plan$method),
                  file.path(run_dir, "fold_sizes.csv"))
  if (!is.null(tune_grid)) {
    safe_save_rds(tune_grid, file.path(run_dir, "tune_grid.rds"),
                  compress = FALSE)
  }

  comparison_csv <- file.path(run_dir, "comparison", "comparison_all.csv")
  comparison_rds <- file.path(run_dir, "comparison", "comparison_all.rds")
  comparison <- tibble::tibble()
  done_ids   <- character(0)
  if (resume && file.exists(comparison_rds)) {
    # The RDS, never the CSV: read_csv2() guesses types, and it has guessed
    # wrong here in two ways that each killed a run AFTER the training was
    # done. See the long note in run_cnn_tuning().
    comparison <- readRDS(comparison_rds)
    if ("unit_id" %in% names(comparison)) {
      done_ids <- comparison$unit_id[comparison$status == "success"]
    }
  }

  windows_needed <- if (is.null(windows)) store$window_sizes else windows
  grid_checked   <- FALSE

  for (j in seq_along(plan$folds)) {
    idx <- plan$folds[[j]]
    message("\n", strrep("=", 78))
    message("FOLD ", j, "/", plan$n_folds, " -- ",
            paste(sprintf("%s=%d", names(idx), lengths(idx)), collapse = " | "))
    message(strrep("=", 78))

    # The SAME cache the CNN would get: same training rows, same scaling fitted
    # on them. That identity is what makes the comparison a comparison.
    fold <- build_fold_cache(store, points, type_table, idx, windows_needed)
    tab  <- fold_table_view(fold$cache, store$predictors,
                            windows = windows_needed, features = features)
    pv   <- fold_points_valid(store, idx)

    # The tensors are not needed once the table exists, and they are the large
    # object: releasing them here is what lets a baseline run beside a CNN run.
    rm(fold); invisible(gc(verbose = FALSE))

    message("Table view: ", ncol(tab[[1]]$x), " features (",
            paste(features, collapse = " + "), ")")

    # Generated ONCE, from fold 1, and reused for every fold after it. A grid
    # regenerated per fold would tune a different set of configs on each one,
    # and the per-config means every decision is read from would then average
    # over configs that are not the same config.
    if (is.null(tune_grid)) {
      tune_grid <- model$default_grid(tune_length, base_seed,
                                      x = tab$train$x, y = tab$train$y)
      if (!"config_id" %in% names(tune_grid)) {
        stop("default_grid() for '", model$name,
             "' returned no config_id column.", call. = FALSE)
      }
      safe_save_rds(tune_grid, file.path(run_dir, "tune_grid.rds"),
                    compress = FALSE)
      message("Grid: ", nrow(tune_grid), " config(s) x ", plan$n_folds,
              " fold(s) x ", n_seeds, " seed(s) = ",
              nrow(tune_grid) * plan$n_folds * n_seeds, " units")
    }

    # The grid exists now (generated here, or supplied by the caller), so this
    # is the first moment the cached units can be checked against it. Once, on
    # the first fold: the grid does not change afterwards.
    if (!grid_checked) {
      done_ids <- .resumable_units(done_ids, comparison, tune_grid)
      grid_checked <- TRUE
    }

    for (i in seq_len(nrow(tune_grid))) {
      cfg <- tune_grid[i, ]
      for (seed_i in seq_len(n_seeds)) {

        # The same name shape as the CNN's, for the same reason: changing
        # n_seeds must not rename the units, or resuming retrains everything.
        unit_id <- sprintf("%s_f%d_s%d", cfg$config_id, j, seed_i)
        if (unit_id %in% done_ids) {
          message("  ", unit_id, " -- already fitted, skipping")
          next
        }

        # The seed depends on the REPETITION, never on the config: two configs
        # within one repetition then start from the same draw, and the spread
        # across repetitions measures luck rather than luck plus config.
        this_seed <- base_seed + seed_i - 1L
        set.seed(this_seed)
        if (requireNamespace("torch", quietly = TRUE)) {
          torch::torch_manual_seed(this_seed)
        }

        # READ OFF THE SPEC HERE, WHERE `model` CAN ONLY MEAN THE SPEC.
        #
        # tibble() evaluates its arguments in order and puts each finished
        # column into the data mask, so a column named `model` SHADOWS the
        # model_spec for every expression after it. `model = model$name`
        # created exactly that column, and the next line's
        # `model$count_params` then resolved `model` to a character vector and
        # died with "$ operator is invalid for atomic vectors" -- on the first
        # unit of the first baseline.
        #
        # Same shape as the print_block_choice() bug: a column name colliding
        # with a variable inside a tidy-eval context. The fix is not to rename
        # the column -- `model` is the right name for it -- but to stop reading
        # the spec once a column could be called that.
        model_name <- model$name

        message("\n-- ", unit_id, " (config ", i, "/", nrow(tune_grid),
                ", fold ", j, ", seed ", this_seed, ") --")

        t0 <- Sys.time()
        err_msg <- NA_character_
        fitted <- tryCatch(
          model$fit(x = tab$train$x, y = tab$train$y, cfg = cfg,
                    x_val = tab$validation$x, y_val = tab$validation$y,
                    device = device, ...),
          error = function(e) {
            message("  ERROR: ", conditionMessage(e))
            err_msg <<- conditionMessage(e)
            NULL
          })
        runtime_min <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

        # A unit that never reaches the table is indistinguishable from one
        # that was never run -- and resume filters on status == "success",
        # which implies a "failed" was meant to exist. Write it.
        if (nrow(comparison) > 0L) {
          comparison <- dplyr::filter(comparison, unit_id != .env$unit_id)
        }

        if (is.null(fitted)) {
          comparison <- dplyr::bind_rows(comparison, dplyr::bind_cols(
            tibble::tibble(
              unit_id = unit_id, config_id = cfg$config_id, model = model_name,
              fold = j, seed = this_seed, best_epoch = NA_integer_,
              runtime_min = round(runtime_min, 2), n_params = NA_integer_,
              n_features = ncol(tab$train$x),
              status = "failed", error_message = err_msg),
            dplyr::select(cfg, -config_id)))
          write_comparison(comparison, comparison_csv, comparison_rds)
          next
        }

        preds <- lapply(names(tab), function(r) {
          predict_table(model$predict(fitted, tab[[r]]$x),
                        pv[[r]], r, transform = transform, clamp = clamp)
        })
        pred_all <- dplyr::bind_rows(preds)
        perf_all <- pred_all %>%
          dplyr::group_by(.data$dataset_role) %>%
          dplyr::group_modify(~ calc_metrics(.x$obs, .x$pred)) %>%
          dplyr::ungroup()

        safe_write_csv2(pred_all, file.path(run_dir, "predictions",
                        paste0(unit_id, "_pred_all.csv")))
        safe_write_csv2(perf_all, file.path(run_dir, "metrics",
                        paste0(unit_id, "_perf.csv")))
        if (!is.null(fitted$history)) {
          safe_write_csv2(fitted$history, file.path(run_dir, "history",
                          paste0(unit_id, "_history.csv")))
        }

        val_perf  <- dplyr::filter(perf_all, .data$dataset_role == "validation")
    # ── The test set is NOT scored during tuning (evaluate_test) ──────────────
        #
        # A frozen test set is frozen only while nothing reads it. Scoring it on
        # every unit puts test_ccc in the comparison table beside val_ccc, and from
        # there it takes one glance to prefer the config that "also does well on
        # test" -- which is selection on the test set, done by a human instead of
        # an argmax, and it inflates the final number by exactly as much.
        #
        # The columns still EXIST, holding NA, so the table keeps one shape whether
        # the test was scored or not and every reader downstream is unchanged.
        # Stage 04 scores the test once, on the chosen config, which is the only
        # moment the number means what it is reported to mean.
        test_perf <- dplyr::filter(perf_all, .data$dataset_role == "test")
        if (!isTRUE(evaluate_test)) test_perf <- test_perf[0, , drop = FALSE]
        if (nrow(test_perf) == 0L) {
          # No test role in this plan: the columns still exist, holding NA, so
          # the table has ONE shape whatever the plan was.
          test_perf <- val_perf
          test_perf[] <- lapply(test_perf, function(z) z[NA_integer_])
        }
        val_metrics <- val_perf %>%
          dplyr::select(n, ccc, r2, mae, nse, rmse, rpd, mqi) %>%
          dplyr::rename_with(~ paste0("val_", .x))
        test_metrics <- test_perf %>%
          dplyr::select(n, ccc, r2, mae, nse, rmse, rpd, mqi) %>%
          dplyr::rename_with(~ paste0("test_", .x))

        # Computed before the tibble, for the same reason model_name is: this
        # is the expression that failed.
        n_params_val <- if (!is.null(model$count_params)) {
          tryCatch(as.integer(model$count_params(fitted)),
                   error = function(e) NA_integer_)
        } else NA_integer_

        comparison <- dplyr::bind_rows(comparison, dplyr::bind_cols(
          tibble::tibble(
            unit_id     = unit_id,
            config_id   = cfg$config_id,
            model       = model_name,
            fold        = j,
            seed        = this_seed,
            best_epoch  = if (!is.null(fitted$best_epoch))
                            as.integer(fitted$best_epoch) else NA_integer_,
            runtime_min = round(runtime_min, 2),
            n_params    = n_params_val,
            n_features  = ncol(tab$train$x),
            status      = "success",
            error_message = NA_character_),
          dplyr::select(cfg, -config_id),
          val_metrics, test_metrics))
        write_comparison(comparison, comparison_csv, comparison_rds)

        message(sprintf("  val CCC %.4f | MAE %.3f | %.2f min",
                        val_metrics$val_ccc, val_metrics$val_mae, runtime_min))

        rm(fitted); invisible(gc(verbose = FALSE))
      }
    }
    rm(tab); invisible(gc(verbose = FALSE))
  }

  by_config <- summarise_resamples(comparison)
  if (nrow(by_config) > 0L) {
    safe_write_csv2(by_config,
                    file.path(run_dir, "comparison", "comparison_by_config.csv"))
  }
  list(comparison = comparison, by_config = by_config,
       run_dir = run_dir, plan = plan)
}
