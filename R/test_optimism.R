# ── Scoring the whole grid on the test set, AFTER the choice is made ──────────
#
# WHY THIS FILE EXISTS AND WHY IT REFUSES TO RUN UNASKED.
#
# During tuning the test set is not scored at all (`evaluate_test = FALSE`).
# The reason is not fussiness: with 24 configs and 9 repetitions, the BEST of
# 216 noisy test scores is systematically higher than any one of them, and the
# number would sit on screen beside the selection metric at the exact moment
# somebody is choosing. Selection then happens through the reader, and inflates
# the result by as much as an argmax would.
#
# But "never score the test on the grid" is a rule about ORDER, not about
# arithmetic. Once the choice is locked, scoring every config on the test
# measures something else entirely, and something worth publishing:
#
#     SELECTION OPTIMISM = best test score in the grid
#                          - test score of the config validation actually chose
#
# That is the amount a paper would have overstated by choosing on the test set,
# measured rather than assumed. Almost nobody reports it. It also audits the
# validation scheme itself: a large gap says the validation is too noisy to
# separate configs, which is a fact about the METHOD -- and never a licence to
# swap in the config that won on test.
#
# The order is what makes it safe, so the order is ENFORCED rather than
# recommended: score_test_grid() refuses to run until freeze_selection() has
# written the choice to disk, and it prints both timestamps so the sequence is
# auditable afterwards by someone who was not in the room.
#
# Nothing is retrained. Every ingredient is already on disk after stage 03: the
# weights (models/<unit_id>_best.pt), the architecture (tune_grid.rds), which
# rows are test (fold_plan.rds), and the fold's scaling, rebuilt from the same
# index. The whole report is inference, which is why this can be an
# afterthought rather than a budget.

#' The short commit of the work tree a given directory belongs to, or NA.
#'
#' `-C` is the whole point: without it git answers for the process's working
#' directory, which has nothing to do with the file being written.
#' @noRd
.git_commit_at <- function(dir) {
  out <- tryCatch(
    suppressWarnings(system2("git", c("-C", shQuote(normalizePath(dir,
                                                                 winslash = "/",
                                                                 mustWork = FALSE)),
                                      "rev-parse", "--short", "HEAD"),
                             stdout = TRUE, stderr = FALSE)),
    error = function(e) character(0))
  if (length(out) == 0L || is.na(out[1]) || !nzchar(trimws(out[1]))) {
    return(NA_character_)
  }
  trimws(out[1])
}

#' Record which config was chosen, and when.
#'
#' Writes the selection into the run directory so a later test-set report can
#' prove the choice preceded it. Call it at the moment the choice is made.
#'
#' @param run_dir   Tuning run directory (the one holding comparison/).
#' @param config_id The chosen config (or configs, one per family).
#' @param rule      How it was chosen, e.g. "one_se" or "rank1".
#' @param metric    The selection metric.
#' @param note      Anything a reader would need to reconstruct the decision.
#' @return The selection record, invisibly.
#' @examples
#' run_dir <- file.path(tempdir(), "tuning_example")
#' freeze_selection(run_dir, "cfg_002", note = "one_se on the cross-validation")
#' # the same choice again is accepted; another is refused
#' try(freeze_selection(run_dir, "cfg_005"))
#' @export
freeze_selection <- function(run_dir, config_id, rule = "one_se",
                             metric = "val_ccc", note = NA_character_) {
  stopifnot(is.character(config_id), length(config_id) >= 1L)
  sel_dir <- file.path(run_dir, "comparison")
  create_output_dirs(sel_dir)
  path <- file.path(sel_dir, "selection.rds")

  # A SECOND FREEZE IS NOT A FREEZE. Overwriting silently would allow: score
  # the grid, dislike the answer, re-freeze, re-score -- the exact sequence
  # this file exists to prevent. Changing a locked selection must be a
  # deliberate act, visible in the shell history rather than in a rerun.
  if (file.exists(path)) {
    old <- readRDS(path)
    if (!identical(sort(old$config_id), sort(config_id))) {
      stop("This run already has a frozen selection (",
           paste(old$config_id, collapse = ", "), ", ", format(old$frozen_at),
           ").\n  Re-freezing a different config would destroy the ordering ",
           "the file exists to prove.\n  Delete ", path, " deliberately if the ",
           "first selection was genuinely wrong, and record why in the log.",
           call. = FALSE)
    }
    return(invisible(old))
  }

  rec <- list(
    config_id = config_id, rule = rule, metric = metric, note = note,
    frozen_at = Sys.time(),
    # The commit makes the claim checkable by a third party: the selection was
    # frozen against this state of the code, and git says when that state was.
    #
    # RESOLVED AGAINST THE RECORD'S OWN DIRECTORY, not the process's. This ran
    # `git rev-parse` with no -C, so it asked whatever directory R happened to
    # be sitting in -- and returned NA whenever that was outside the repo, which
    # a script doing setwd() elsewhere makes routine. The field then read NA
    # with nothing saying why, in the one place whose whole job is to let
    # someone else check the claim. It was noticed only because a test run
    # printed "config, rule, metric, time" where the run before had printed
    # "..., commit 3649fd8".
    git_commit = .git_commit_at(sel_dir))
  safe_save_rds(rec, path, compress = FALSE)
  message("Selection frozen: ", paste(config_id, collapse = ", "),
          " (", rule, " on ", metric, ") -> ", path)
  if (is.na(rec$git_commit)) {
    message("  NOTE: no git commit recorded -- ", sel_dir, " is not inside a ",
            "git work tree.\n  The record still fixes WHAT was chosen and ",
            "WHEN; it cannot fix against which state of the code.")
  }
  invisible(rec)
}

#' Score every trained unit of a tuning run on the held-out test set.
#'
#' @param run_dir   Tuning run directory.
#' @param data      A dsm_data (from dsm_load()), or a list with store, points
#'   and type_table.
#' @param transform NULL (the default) for the inverse of the transform the
#'   store was built under, as in dsm_train(); a function is the inverse to
#'   use instead, and is refused if it disagrees with the store's.
#' @param device    torch device.
#' @param config_ids Which configs to score. NULL means every config in the grid.
#' @param allow_unfrozen Escape hatch for teaching or for a run whose selection
#'   was recorded elsewhere. Not a default, and the report says it was used.
#' @return An object of class "test_optimism".
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()    # a small fitted run, made once a session
#' # every configuration on the test set, once the choice is frozen
#' score_test_grid(run$fit$run_dir, run$data, device = torch::torch_device("cpu"))
#' }
#' @export
score_test_grid <- function(run_dir, data, transform = NULL, device,
                            config_ids = NULL, allow_unfrozen = FALSE) {

  sel_path  <- file.path(run_dir, "comparison", "selection.rds")
  selection <- if (file.exists(sel_path)) readRDS(sel_path) else NULL
  if (is.null(selection) && !isTRUE(allow_unfrozen)) {
    stop("No frozen selection in this run.\n",
         "  Scoring the grid on the test set is only meaningful AFTER the ",
         "choice is locked --\n  before that it is selection on the test set ",
         "with extra steps.\n",
         "  Call freeze_selection(\"", run_dir, "\", <config_id>) first.",
         call. = FALSE)
  }

  store      <- data$store
  points     <- data$points
  type_table <- data$type_table
  if (is.null(store) || is.null(points) || is.null(type_table)) {
    stop("`data` must be a dsm_data, or a list with store, points and ",
         "type_table.", call. = FALSE)
  }
  # THE STORE'S INVERSE, AS dsm_train() TAKES IT. The default was identity:
  # on a log1p store, a call that left it out scored the test set in log
  # space -- a table of plausible numbers, in the wrong units, beside the
  # tuning table dsm_train() had scored in native ones.
  transform <- .resolve_train_transform(transform, data, verbose = FALSE)

  grid_path <- file.path(run_dir, "tune_grid.rds")
  plan_path <- file.path(run_dir, "fold_plan.rds")
  for (p in c(grid_path, plan_path)) {
    if (!file.exists(p)) stop("Missing ", p, call. = FALSE)
  }
  tune_grid <- readRDS(grid_path)
  plan      <- readRDS(plan_path)
  if (!is.null(config_ids)) {
    tune_grid <- tune_grid[tune_grid$config_id %in% config_ids, , drop = FALSE]
  }
  if (nrow(tune_grid) == 0L) stop("No configs to score.", call. = FALSE)
  if (length(plan$folds[[1]]$test) == 0L) {
    stop("This plan has no test set -- there is nothing to score.",
         call. = FALSE)
  }

  model_dir <- file.path(run_dir, "models")
  rows <- list()
  # One buffer for every fold's cache, as the training loop has (run_cnn_resample()).
  buffer <- if (.use_fold_buffer()) new_fold_buffer() else NULL

  for (j in seq_along(plan$folds)) {
    idx <- plan$folds[[j]]

    # THE FOLD'S OWN SCALING, REBUILT THE WAY TRAINING BUILT IT.
    #
    # Scaling is fitted on the training rows of the fold, so a model must see
    # test patches scaled by ITS fold's constants. Rebuilding the cache from
    # the same index reproduces them exactly; reading a global table instead
    # would be a second copy of the same fact, free to drift -- the defect that
    # once had stage 05 predicting with constants the network never saw.
    fold <- build_fold_cache(store, points, type_table, idx,
                             store$window_sizes, verbose = FALSE, buffer = buffer)
    pv   <- fold_points_valid(store, idx)

    for (i in seq_len(nrow(tune_grid))) {
      cfg   <- tune_grid[i, ]
      ckpts <- list.files(
        model_dir,
        pattern = sprintf("^%s_f%d_s[0-9]+_best[.]pt$", cfg$config_id, j),
        full.names = TRUE)

      for (ck in ckpts) {
        unit_id <- sub("_best[.]pt$", "", basename(ck))
        seed_i  <- as.integer(sub(".*_s([0-9]+)$", "\\1", unit_id))

        m <- build_cnn_from_config(cfg, store$n_channels)
        m$load_state_dict(torch::torch_load(ck))
        m$to(device = device)

        loaders <- .make_loaders_from_cache(fold$cache, cfg)
        pred <- predict_loader(m, loaders$test, pv$test, "test",
                               transform = transform, device = device)
        perf <- calc_metrics(pred$obs, pred$pred)

        rows[[length(rows) + 1L]] <- dplyr::bind_cols(
          tibble::tibble(unit_id = unit_id, config_id = cfg$config_id,
                         fold = j, seed_i = seed_i),
          dplyr::rename_with(perf, ~ paste0("test_", .x)))

        rm(m, loaders); invisible(gc(verbose = FALSE))
      }
    }
    rm(fold); invisible(gc(verbose = FALSE))
  }

  units <- dplyr::bind_rows(rows)
  if (nrow(units) == 0L) {
    stop("No checkpoints matched the grid in ", model_dir, call. = FALSE)
  }

  by_config <- units %>%
    dplyr::group_by(.data$config_id) %>%
    dplyr::summarise(
      n_units       = dplyr::n(),
      test_ccc_mean = mean(.data$test_ccc, na.rm = TRUE),
      test_ccc_sd   = stats::sd(.data$test_ccc, na.rm = TRUE),
      test_mae_mean = mean(.data$test_mae, na.rm = TRUE),
      .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(.data$test_ccc_mean))
  by_config$test_rank <- seq_len(nrow(by_config))

  structure(
    list(units = units, by_config = by_config,
         chosen = if (!is.null(selection)) selection$config_id else NA_character_,
         selection = selection, run_dir = run_dir,
         scored_at = Sys.time(), unfrozen = is.null(selection)),
    class = "test_optimism")
}

#' Say what the grid did on the test set, and what that does and does not mean.
#'
#' @param x   A `test_optimism`, from [score_test_grid()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.test_optimism <- function(x, ...) {
  cat("\nTest-set scores across the grid\n")
  cat(strrep("-", 66), "\n")
  if (isTRUE(x$unfrozen)) {
    cat("  WARNING: no frozen selection. These numbers cannot support any claim\n")
    cat("  about optimism -- nothing here proves the choice came first.\n\n")
  } else {
    cat("  selection frozen : ", format(x$selection$frozen_at),
        "  (", x$selection$rule, " on ", x$selection$metric, ")\n", sep = "")
    cat("  grid scored      : ", format(x$scored_at), "\n", sep = "")
  }
  print(x$by_config, n = Inf)

  if (!all(is.na(x$chosen)) && any(x$by_config$config_id %in% x$chosen)) {
    r    <- x$by_config[x$by_config$config_id %in% x$chosen, ][1, ]
    best <- x$by_config[1, ]
    gap  <- best$test_ccc_mean - r$test_ccc_mean
    cat("\n  chosen by validation : ", r$config_id,
        sprintf("   test CCC %.4f   (rank %d of %d on test)",
                r$test_ccc_mean, r$test_rank, nrow(x$by_config)),
        "\n", sep = "")
    cat("  best on test         : ", best$config_id,
        sprintf("   test CCC %.4f", best$test_ccc_mean), "\n", sep = "")
    cat(sprintf("  SELECTION OPTIMISM   : %+.4f CCC\n", gap))

    # The yardstick is the spread WITHIN a config, for the same reason the
    # family comparison uses it: an optimism smaller than the noise between
    # repetitions of one config is not a finding about selection at all, it is
    # that same noise seen from another angle.
    noise <- stats::median(x$by_config$test_ccc_sd, na.rm = TRUE)
    if (is.finite(noise)) {
      cat(sprintf("  typical sd within one config : %.4f\n", noise))
      if (gap <= noise) {
        cat("  -> Validation chose a config indistinguishable from the best on\n")
        cat("     test. Selecting on the test set would have bought nothing.\n")
      } else {
        cat("  -> The grid's best on test is ahead of the chosen config by more\n")
        cat("     than the noise. That measures how much a test-selected number\n")
        cat("     would have overstated the result. It is NOT a reason to swap:\n")
        cat("     swapping is what the measurement is measuring.\n")
      }
    }
  }
  invisible(x)
}
