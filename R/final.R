# ── The final model: the selected configuration, refitted under N seeds ──────
#
# WHAT THIS REPLACES.
#
# The SOC project's stage 04 did this for one dataset, in 850 lines of
# script: choose the configuration a tuning run supports, refit it on
# everything but the test set under ten seeds, and turn the ten into what the
# map needs -- the ensemble median, a calibrated interval, a smearing factor
# for the mean surface. dsm_final() is that stage with the dataset taken out,
# and it writes the same files in the same places, so dsm_predict() reads a
# dsm_final() run exactly as it reads a stage-04 one.
#
# HOW THE SEEDS ARE TRAINED, AND WHY THAT WAY.
#
# Two measurements decided it (2026-09-27):
#
#   T1  one unit uses this CPU poorly: tripling its threads from 5 to 15 made
#       the heavy configuration only 1.5x faster, and 30 threads (the
#       hyperthreads) gained nothing.
#   T2  three units of 5 threads side by side trained the same six units
#       1.52x faster than one unit of 15 in sequence -- and gave, unit for
#       unit, EXACTLY the numbers each gave alone. A unit's result depends on
#       its seed and its thread count, and on nothing else: not on its
#       neighbours, not on how many fit on the machine.
#
# So the seeds are trained side by side, each in its own R process, with a
# FIXED number of threads per unit (threads_per_unit, 5 by default). How many
# run at once follows from n_cores and the RAM, and changes only the time.
# The thread count per unit changes the numbers (T1: 0.067 CCC between counts
# in 12 epochs), so it is recorded with the run, like the seeds.
#
# EVERY UNIT IS TRAINED IN A SUBPROCESS, EVEN WITH ONE WORKER. OpenMP sizes its
# pool when torch loads; a unit trained in the user's session gets whatever
# thread setting that session happened to load torch with -- B6 measured the
# difference at 1.2e-4 CCC. A process started with OMP_NUM_THREADS already set
# gives the same answer whichever session asked for it.
#
# WHAT IT WRITES (the layout stage 05 reads):
#
#   <run>/comparison/final_run_summary.rds    the choice, the seeds, the results
#   <run>/comparison/all_seed_results_test.csv, config_summary_test.csv
#   <run>/<config>/models/seed%04d_best.pt    one checkpoint per seed
#   <run>/<config>/predictor_scaling.csv      the scaling the weights expect
#   <run>/<config>/ensemble_predictions.csv, conformal_90.rds, smearing.rds
#   <run>/run_spec.rds                       what the seeds' numbers depend on,
#                                            written before the first seed; a
#                                            resume is held to it
#
# and, new, the answer to "which CNN was chosen, exactly":
#
#   <run>/final_report.md                   how it was chosen, every
#                                           hyperparameter, the refit, the test
#   <run>/selected_hyperparameters.csv      one row per hyperparameter: its
#                                           value, whether the search varied it,
#                                           the values the grid tried

# The refit's schedule. Longer than tuning's: the configuration is known now,
# so there is no reason to hurry convergence -- tuning trades a little
# accuracy per config for covering the grid, and this stage does not. These
# are stage 04's values.
.final_training_defaults <- list(
  n_epochs             = 700L,
  patience             = 100L,
  es_min_delta         = 0.0003,
  warmup_start_lr      = 1e-5,
  lr_plateau_factor    = 0.5,
  lr_plateau_patience  = 30L,
  lr_plateau_min_delta = 0.0003,
  min_lr               = 1e-6,
  gradient_clip        = 1.0,
  print_every          = 10L,
  augment              = TRUE
)

# THE CLAMP IS THE TUNING RUN'S. Its units were scored with it -- the
# selection, and the cross-validated residuals the intervals are calibrated
# on -- so the refit clamps the same way, and so does a map (dsm_predict()
# reads the refit's). Given in `training`, it must agree. A tuning run that
# recorded none (stage 03, or before dsm_train() kept it) meant the default.
.final_clamp <- function(tuning_dir, given) {
  f <- file.path(tuning_dir, "clamp.rds")
  tuned <- if (file.exists(f)) as.numeric(readRDS(f)) else NULL
  if (is.null(given)) return(tuned %||% c(0, Inf))
  given <- as.numeric(given)
  if (!is.null(tuned) && !identical(tuned, given)) {
    stop(sprintf("training$clamp = c(%s, %s), but the tuning run scored its units -- the selection and the calibration residuals -- with c(%s, %s). Leave it out to refit with the tuning run's.",
                 format(given[1]), format(given[2]), format(tuned[1]), format(tuned[2])),
         call. = FALSE)
  }
  given
}

#' Refit the configuration a tuning run selects, under N seeds.
#'
#' @param tuning     A `dsm_fit` from dsm_train(), or the directory of a tuning
#'   run (it holds fold_plan.rds, tune_grid.rds and comparison/).
#' @param data       From dsm_load(): the store the tuning ran on. Taken from
#'   the dsm_fit when `tuning` is one. Any windows may be loaded -- the
#'   workers read the ones the selected configuration needs.
#' @param config     "auto" applies `rule`; config id(s) choose by hand, and
#'   the frozen record then says "manual".
#' @param rule       "one_se" (the simplest config within one standard error
#'   of the best) or "rank1" (the best mean).
#' @param metric     The selection metric, as in the tuning table.
#' @param seeds      A count (seeds 42, 43, ...) or the seeds themselves.
#' @param validation_frac Share of the non-test rows the refit stops on,
#'   carved by the tuning plan's own criterion (refit_split()): blocks, random
#'   rows, whole regions, or kNNDM against the same prediction points.
#' @param predpoints For a kNNDM tuning run made before its plan kept its
#'   prediction points (2026-09-28): those points, a data frame with x and y.
#'   NULL takes the plan's.
#' @param training   Overrides of the refit schedule (.final_training_defaults)
#'   -- any argument of train_one_cnn().
#' @param transform  NULL for the store's own inverse (see dsm_train()).
#' @param n_cores    Cores for the whole fit. NULL is the physical cores minus
#'   one.
#' @param threads_per_unit Threads each seed trains with. Part of the result
#'   (see the header), so it is recorded; 5 was measured best here (T1, T2).
#' @param max_ram_gb RAM the workers may use in total. NULL for 70% of what is
#'   available at the start (read with the ps package).
#' @param conformal_alpha Miscoverage levels of the intervals: 0.1 is 90%.
#' @param output_dir Where runs go. NULL for final_model/ beside the tuning
#'   run's directory.
#' @param run_id     NULL for final_<timestamp>. Give an existing one with
#'   resume = TRUE to finish an interrupted fit.
#' @param resume     Skip seeds whose checkpoint and record are already there.
#'   A resumed run is held to the settings it started with (run_spec.rds):
#'   another thread count, schedule, grid, split or scaling is refused, and so
#'   is a directory dsm_final() did not start.
#' @param verbose    Report progress, and print the result.
#' @return A `dsm_final`, printed with the report.
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()    # a small fitted run, made once a session
#' final <- dsm_final(run$fit, seeds = 2, n_cores = 1, threads_per_unit = 1,
#'                    training = list(n_epochs = 10, patience = 5), output_dir = tempdir(),
#'                    run_id = "final_example", verbose = FALSE)
#' final
#' }
#' @export
dsm_final <- function(tuning, data = NULL, config = "auto",
                      rule = c("one_se", "rank1"), metric = "val_ccc",
                      seeds = 10L, validation_frac = 0.15, predpoints = NULL,
                      training = list(),
                      transform = NULL, n_cores = NULL, threads_per_unit = 5L,
                      max_ram_gb = NULL, conformal_alpha = c(0.1, 0.05),
                      output_dir = NULL, run_id = NULL, resume = TRUE,
                      verbose = TRUE) {

  t_start <- Sys.time()
  say  <- function(...) if (verbose) message(...)
  rule <- match.arg(rule)

  # ── 0. the arguments, all checked before anything trains ──────────────────
  if (inherits(tuning, "dsm_fit")) {
    tuning_dir <- tuning$run_dir
    if (is.null(data)) data <- tuning$data
  } else if (is.character(tuning) && length(tuning) == 1L) {
    tuning_dir <- tuning
  } else {
    stop("`tuning` must be a dsm_fit from dsm_train() or the directory of a ",
         "tuning run; got a ", class(tuning)[1], ".", call. = FALSE)
  }
  tuning_dir <- normalizePath(tuning_dir, winslash = "/", mustWork = FALSE)
  for (f in c("fold_plan.rds", "tune_grid.rds",
              file.path("comparison", "comparison_ranked.csv"))) {
    if (!file.exists(file.path(tuning_dir, f))) {
      stop("Not a finished tuning run, missing ", f, ": ", tuning_dir, call. = FALSE)
    }
  }
  if (!inherits(data, "dsm_data")) {
    stop("`data` must come from dsm_load() -- the store the tuning ran on.",
         call. = FALSE)
  }
  seeds <- .final_seeds(seeds)
  internal <- c("cfg", "n_channels", "loaders", "points_valid", "transform",
                "device", "model_name", "on_epoch")
  bad <- setdiff(names(training), setdiff(names(formals(train_one_cnn)), internal))
  if (length(bad) > 0L) {
    stop("`training` has argument(s) train_one_cnn() does not take: ",
         paste(bad, collapse = ", "), call. = FALSE)
  }
  training  <- utils::modifyList(.final_training_defaults, training)
  training$clamp <- .final_clamp(tuning_dir, training$clamp)
  transform <- .resolve_train_transform(transform, data, verbose = verbose)
  n_cores   <- resolve_cores(n_cores, what = "the final fit")
  tpu <- suppressWarnings(as.integer(threads_per_unit))
  if (length(threads_per_unit) != 1L || is.na(tpu) || tpu < 1L || tpu != threads_per_unit) {
    stop("threads_per_unit must be a whole number >= 1.", call. = FALSE)
  }
  if (tpu > n_cores) {
    say("threads_per_unit = ", tpu, " is more than the ", n_cores, " core(s) this ",
        "fit may use; each seed trains with ", n_cores, " instead -- and its ",
        "numbers then differ from a ", tpu, "-thread fit of the same seed.")
    tpu <- n_cores
  }
  if (!is.numeric(conformal_alpha) || any(conformal_alpha <= 0 | conformal_alpha >= 1)) {
    stop("conformal_alpha must be numbers in (0, 1), e.g. c(0.1, 0.05).", call. = FALSE)
  }

  # ── 1. what the tuning run left ───────────────────────────────────────────
  tuning_plan <- readRDS(file.path(tuning_dir, "fold_plan.rds"))
  grid        <- readRDS(file.path(tuning_dir, "tune_grid.rds"))
  if (!all(c("window_sizes", "batch_size") %in% names(grid))) {
    stop("dsm_final() refits a CNN tuning run, and this run's grid has no ",
         "window_sizes / batch_size -- a table model's (rf, mlp) is fitted ",
         "differently.", call. = FALSE)
  }
  units_tbl   <- safe_read_csv2(file.path(tuning_dir, "comparison", "comparison_ranked.csv"))
  bc_path     <- file.path(tuning_dir, "comparison", "comparison_by_config.csv")
  by_config   <- if (file.exists(bc_path)) safe_read_csv2(bc_path) else NULL

  # ── 2. which configuration ────────────────────────────────────────────────
  #
  # CHOSEN HERE, FROZEN ONLY IN STEP 4, once nothing else can stop the call.
  # freeze_selection() refuses ever to record another choice in this tuning
  # run. Frozen here, as it was, a choice made by a call that then stopped --
  # a window the store lacks, a batch the refit cannot fill, a resume held to
  # other settings -- stayed frozen with no final model behind it, and the
  # corrected call was refused after it.
  sel <- .final_select(config, rule, metric, by_config, units_tbl, grid, say, verbose)
  ids <- sel$ids
  selected <- grid[grid$config_id %in% ids, , drop = FALSE]

  windows_needed <- sort(unique(unlist(selected$window_sizes)))
  available <- as.integer(trimws(strsplit(
    as.character(data$store$manifest$windows_extracted[1]), ",")[[1]]))
  if (!all(windows_needed %in% available)) {
    stop("The selected config(s) need window(s) ",
         paste(setdiff(windows_needed, available), collapse = ", "),
         ", which this store does not hold (it has ",
         paste(available, collapse = ", "), ").", call. = FALSE)
  }

  # ── 3. the refit split, carved by the tuning plan's own criterion ─────────
  #
  # Same test set, validation carved the same way the selection was: a model
  # selected on spatial folds that then stopped on a random validation set
  # would have changed the question between the two stages.
  refit <- refit_split(tuning_plan, data$store$meta, validation_frac = validation_frac,
                       predpoints = predpoints)
  index <- refit$folds[[1]]
  n_train <- length(index$train)
  too_big <- selected$batch_size > n_train
  if (any(too_big)) {
    stop("batch_size larger than the refit's ", n_train, " training point(s) in ",
         paste(selected$config_id[too_big], collapse = ", "), ": such a unit ",
         "would take no gradient step.", call. = FALSE)
  }
  if (verbose) {
    message("\n-- The refit split (from the tuning plan) --")
    print(refit)
  }

  # The scaling belongs to the fit, not to a seed, and not to a worker: fitted
  # ONCE here, from this split's training rows, handed to every worker, and
  # written beside the weights -- stage 05 scales the rasters with it.
  scaling <- fit_scaling(data$points, data$type_table, index$train)
  # Checked here, as build_fold_cache() checks it in every worker: there it
  # stops after the choice is frozen and after each worker has loaded.
  if (any(scaling$degenerate)) {
    stop("Degenerate scaling (zero or non-finite sd) on the refit's training rows for: ",
         paste(scaling$predictor[scaling$degenerate], collapse = ", "),
         ". A channel constant over the rows a model trains on cannot be z-scored.",
         call. = FALSE)
  }

  # ── 4. where it goes, what it depends on, and what is already done ────────
  output_dir <- output_dir %||% file.path(dirname(tuning_dir), "final_model")
  run_id     <- run_id %||% paste0("final_", format(Sys.time(), "%Y%m%d_%H%M%S"))
  run_dir    <- normalizePath(file.path(output_dir, run_id), winslash = "/", mustWork = FALSE)
  if (dir.exists(run_dir) && !isTRUE(resume)) {
    stop("A final run already exists at ", run_dir, ". Pass resume = TRUE to ",
         "finish it, or another run_id.", call. = FALSE)
  }
  # What a seed's numbers depend on besides the seed: recorded when the run
  # is new, and a resume that would compute the rest another way is refused
  # (.final_run_spec()). The same five values .resolve_train_transform()
  # probes the inverse at.
  .final_run_spec(run_dir, run_id, list(
    grid = grid, split = index, scaling = scaling, threads_per_unit = tpu,
    training = training, transform_probe = as.numeric(transform(c(0, 0.5, 1, 2.5, 5)))))
  # Everything that can stop the call has been checked; the choice is frozen
  # before the first seed trains, and so before any test score exists.
  freeze_selection(tuning_dir, ids, rule = sel$rule_applied, metric = metric,
                   note = paste("dsm_final() on", format(Sys.time())))
  create_output_dirs(c(run_dir, file.path(run_dir, "comparison")))
  for (cid in ids) {
    create_output_dirs(file.path(run_dir, cid, c("models", "history", "predictions",
                                                 "metrics", "gates", "units")))
    safe_write_csv2(scaling, file.path(run_dir, cid, "predictor_scaling.csv"))
  }

  units <- expand.grid(seed = seeds, config_id = ids, stringsAsFactors = FALSE)
  units <- tibble::tibble(config_id = units$config_id, seed = as.integer(units$seed))
  units$unit_id <- sprintf("%s_seed%04d", units$config_id, units$seed)
  done <- vapply(seq_len(nrow(units)), function(i)
    .final_unit_done(run_dir, units$config_id[i], units$seed[i]), logical(1))
  todo <- units[!done, , drop = FALSE]
  if (any(done)) say("\nResuming: ", sum(done), " of ", nrow(units),
                     " seed unit(s) already fitted, kept as they are.")

  # ── 5. train, side by side ────────────────────────────────────────────────
  run_info <- list(n_workers = 0L, threads_per_unit = tpu, peak_gb = NA_real_,
                   per_worker_gb_estimate = NA_real_)
  if (nrow(todo) > 0L) {
    run_info <- .final_train_units(
      todo = todo, selected = selected, data = data, index = index,
      scaling = scaling, windows = windows_needed, training = training,
      transform = transform, run_dir = run_dir, n_cores = n_cores,
      threads_per_unit = tpu, max_ram_gb = max_ram_gb, say = say)
  }

  # EVERY REQUESTED SEED, OR STOP. Stage 04's rule: a seed that failed must be
  # named here, where the run is still in front of someone -- not discovered
  # later as a map built from the survivors.
  status <- vapply(seq_len(nrow(units)), function(i)
    .final_unit_done(run_dir, units$config_id[i], units$seed[i]), logical(1))
  if (!all(status)) {
    lost <- units$unit_id[!status]
    msgs <- vapply(lost, function(u) {
      r <- .final_unit_record(run_dir, units$config_id[units$unit_id == u],
                              units$seed[units$unit_id == u])
      if (is.null(r)) "never trained" else as.character(r$error %||% r$status)
    }, character(1))
    stop("Seed unit(s) did not finish: ",
         paste(sprintf("%s (%s)", lost, msgs), collapse = "; "),
         ".\n  Worker logs: ", file.path(run_dir, "logs"),
         "\n  Fix the cause and call dsm_final() again with run_id = \"", run_id,
         "\" -- the finished seeds are kept.", call. = FALSE)
  }

  # ── 6. from N seeds to what the map needs ─────────────────────────────────
  all_seed_results <- tibble::tibble()
  per_config <- list()
  for (cid in ids) {
    asm <- .final_assemble_config(
      cfg_dir = file.path(run_dir, cid), cfg_out_dir = file.path(run_dir, cid),
      config_id = cid, seeds = seeds, tuning_dir = tuning_dir,
      conformal_alpha = conformal_alpha, transform_name = data$transform$name %||% "none",
      verbose = verbose)
    per_config[[cid]] <- asm
    all_seed_results <- dplyr::bind_rows(all_seed_results, asm$seed_rows)
  }
  config_summary <- .final_config_summary(all_seed_results)
  safe_write_csv2(all_seed_results, file.path(run_dir, "comparison", "all_seed_results_test.csv"))
  safe_write_csv2(config_summary,   file.path(run_dir, "comparison", "config_summary_test.csv"))
  .final_paired(all_seed_results, ids, run_dir, verbose)

  seeds_fitted <- sort(unique(all_seed_results$seed))
  hyper <- .final_hyper_table(selected, grid, training)
  safe_write_csv2(hyper, file.path(run_dir, "selected_hyperparameters.csv"))

  summary_rds <- list(
    # Stage 04's fields, unchanged: stage 05 reads them.
    selected_cfgs = selected, selected_config_ids = ids,
    selection_rule = sel$rule_applied, seeds = seeds, seeds_fitted = seeds_fitted,
    all_seed_results = all_seed_results, config_summary = config_summary,
    run_id = run_id, tuning_run_id = basename(tuning_dir),
    # The path as well as the name: dsm_predict() calibrates the map's
    # intervals on this run's residuals, and a name alone says where nothing is.
    tuning_dir = tuning_dir, tuning_dir_rel = .relative_path(tuning_dir, run_dir),
    # What makes a seed's numbers reproducible, beside the seed itself.
    threads_per_unit = tpu, n_workers = run_info$n_workers, training = training,
    validation_frac = validation_frac, refit_method = refit$method, n_train = n_train,
    n_validation = length(index$validation), n_test = length(index$test),
    torch_version = as.character(utils::packageVersion("torch")),
    r_version = R.version.string, git_commit = .git_commit_at(run_dir),
    fitted_by = "dsm_final()", finished_at = Sys.time())
  safe_save_rds(summary_rds, file.path(run_dir, "comparison", "final_run_summary.rds"),
                compress = FALSE)

  out <- structure(list(
    run_dir = run_dir, tuning_dir = tuning_dir, selected_config_ids = ids,
    selection = sel, selected = selected, seeds = seeds, hyper = hyper,
    all_seed_results = all_seed_results, config_summary = config_summary,
    per_config = per_config, by_config = by_config, units_tbl = units_tbl,
    grid = grid, training = training, threads_per_unit = tpu,
    n_workers = run_info$n_workers, peak_gb = run_info$peak_gb,
    per_worker_gb_estimate = run_info$per_worker_gb_estimate,
    split = c(train = n_train, validation = length(index$validation),
              test = length(index$test)),
    refit_method = refit$method,
    target = data$target_col, fitted_by = "dsm_final()",
    minutes = as.numeric(difftime(Sys.time(), t_start, units = "mins"))),
    class = "dsm_final")
  out$report_file <- .final_write_report(out)
  if (verbose) print(out)
  invisible(out)
}

#' Print a `dsm_final`
#'
#' @param x   A `dsm_final`, from [dsm_final()] or [dsm_report_final()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.dsm_final <- function(x, ...) {
  cat("\n<dsm_final> ", x$run_dir, "\n", sep = "")
  cat("  selected   : ", paste(x$selected_config_ids, collapse = ", "), "  (",
      x$selection$rule_applied, " on ", x$selection$metric, ")\n", sep = "")
  cat("  seeds      : ", length(x$seeds), "  (",
      if (is.na(x$threads_per_unit)) "thread count not recorded"
      else paste0(x$threads_per_unit, " thread(s) per seed"),
      if (isTRUE(x$n_workers > 0L)) sprintf(", %d side by side", x$n_workers) else "",
      ")\n", sep = "")
  cat("  split      : ", paste(sprintf("%s %d", names(x$split), x$split), collapse = " | "), "\n", sep = "")
  for (cid in x$selected_config_ids) {
    h <- x$hyper[x$hyper$config_id == cid & x$hyper$group != "final refit", , drop = FALSE]
    cat("\n  ", cid, " -- every hyperparameter of the selected CNN:\n", sep = "")
    for (i in seq_len(nrow(h))) {
      cat(sprintf("    %-16s %-22s %-7s %s\n", h$parameter[i], h$value[i],
                  if (!isTRUE(h$used[i])) "n/a" else if (isTRUE(h$searched[i])) "tuned" else "fixed",
                  if (!isTRUE(h$used[i])) "(not used by this network)"
                  else if (isTRUE(h$searched[i])) paste0("(tried: ", h$values_tried[i], ")") else ""))
    }
    s <- x$config_summary[x$config_summary$config_id == cid, , drop = FALSE]
    if (nrow(s) == 1L && is.finite(s$ccc_mean)) {
      cat(sprintf("\n    test, mean +/- sd over %d seeds: CCC %.4f +/- %.4f | MAE %.3f +/- %.3f | RMSE %.3f +/- %.3f\n",
                  s$n_seeds, s$ccc_mean, s$ccc_sd, s$mae_mean, s$mae_sd, s$rmse_mean, s$rmse_sd))
    }
  }
  cat("\n  report     : ", x$report_file, "\n", sep = "")
  invisible(x)
}

#' The declaration of a final model that already exists.
#'
#' dsm_final() writes final_report.md as it fits. A final model fitted by
#' stage 04, before dsm_final() existed, has the seeds and the files but no
#' declaration -- and retraining ten seeds to obtain one would also change
#' them (T1: the thread count changes the numbers). This writes it from what
#' is on disk, re-assembling the seeds the way P3 proved dsm_final() does,
#' and changes nothing that is there: it ADDS final_report.md and
#' selected_hyperparameters.csv to the run directory.
#'
#' @param run_dir    The final-model run (it holds comparison/final_run_summary.rds).
#' @param tuning_dir The tuning run it was selected from.
#' @param conformal_alpha As the run was calibrated; stage 04 used c(0.1, 0.05).
#' @param verbose    Print the result.
#' @return A `dsm_final` describing the run, printed with the declaration.
#' @examplesIf torch::torch_is_installed()
#' \donttest{
#' run <- example_run()    # a small fitted run, made once a session
#' # the declaration of a final run, written from what is on disk
#' dsm_report_final(run$final$run_dir, run$fit$run_dir)
#' }
#' @export
dsm_report_final <- function(run_dir, tuning_dir, conformal_alpha = c(0.1, 0.05),
                             verbose = TRUE) {
  summ_path <- file.path(run_dir, "comparison", "final_run_summary.rds")
  for (f in c(summ_path, file.path(tuning_dir, "tune_grid.rds"),
              file.path(tuning_dir, "comparison", "comparison_ranked.csv"))) {
    if (!file.exists(f)) stop("Not found: ", f, call. = FALSE)
  }
  summ      <- readRDS(summ_path)
  grid      <- readRDS(file.path(tuning_dir, "tune_grid.rds"))
  units_tbl <- safe_read_csv2(file.path(tuning_dir, "comparison", "comparison_ranked.csv"))
  bc_path   <- file.path(tuning_dir, "comparison", "comparison_by_config.csv")
  by_config <- if (file.exists(bc_path)) safe_read_csv2(bc_path) else NULL
  sel_path  <- file.path(tuning_dir, "comparison", "selection.rds")
  metric    <- if (file.exists(sel_path)) readRDS(sel_path)$metric else "val_ccc"
  ids       <- summ$selected_config_ids
  seeds     <- summ$seeds

  # The selection as it was made: the rule is the recorded one, and one_se's
  # tie count is recomputed from the table it was computed from.
  pick <- NULL
  if (identical(summ$selection_rule, "one_se") && !is.null(by_config)) {
    pick <- tryCatch(one_se(by_config, metric = metric, complexity = "n_params"),
                     error = function(e) NULL)
  }
  noise <- if (!is.null(by_config) && paste0(metric, "_sd") %in% names(by_config)) {
    stats::median(by_config[[paste0(metric, "_sd")]], na.rm = TRUE)
  } else NA_real_
  sel <- list(ids = ids, rule_applied = summ$selection_rule,
              rule_asked = summ$selection_rule, metric = metric, pick = pick,
              median_seed_sd = noise)

  per_config <- list()
  all_seed_results <- tibble::tibble()
  for (cid in ids) {
    per_config[[cid]] <- .final_assemble_config(
      cfg_dir = file.path(run_dir, cid), cfg_out_dir = NULL, config_id = cid,
      seeds = seeds, tuning_dir = tuning_dir, conformal_alpha = conformal_alpha,
      transform_name = if (file.exists(file.path(run_dir, cid, "smearing.rds"))) "log1p" else "none",
      verbose = FALSE, write = FALSE)
    all_seed_results <- dplyr::bind_rows(all_seed_results, per_config[[cid]]$seed_rows)
  }

  # The split, counted from what the first seed predicted: no store needed.
  pa <- safe_read_csv2(file.path(run_dir, ids[1], "predictions",
                                 sprintf("seed%04d_pred_all.csv", seeds[1])))
  n_role <- function(r) sum(pa$dataset_role == r)

  training <- summ$training %||% list()
  x <- structure(list(
    run_dir = run_dir, tuning_dir = tuning_dir, selected_config_ids = ids,
    selection = sel, selected = summ$selected_cfgs, seeds = seeds,
    hyper = .final_hyper_table(summ$selected_cfgs, grid, training),
    all_seed_results = all_seed_results,
    config_summary = .final_config_summary(all_seed_results),
    per_config = per_config, by_config = by_config, units_tbl = units_tbl,
    grid = grid, training = training,
    threads_per_unit = summ$threads_per_unit %||% NA_integer_,
    n_workers = summ$n_workers %||% 0L, peak_gb = NA_real_,
    per_worker_gb_estimate = NA_real_,
    split = c(train = n_role("train"), validation = n_role("validation"),
              test = n_role("test")),
    target = summ$target %||% basename(dirname(run_dir)),
    fitted_by = summ$fitted_by %||% "stage 04, before dsm_final() existed",
    minutes = NA_real_),
    class = "dsm_final")
  safe_write_csv2(x$hyper, file.path(run_dir, "selected_hyperparameters.csv"))
  x$report_file <- .final_write_report(x)
  if (verbose) print(x)
  invisible(x)
}

# ── helpers: the arguments ────────────────────────────────────────────────────

# A count means seeds 42, 43, ... -- the convention dsm_train() already uses
# (base_seed + i - 1), so a seed number means the same thing in both stages.
.final_seeds <- function(seeds) {
  s <- suppressWarnings(as.integer(seeds))
  if (length(seeds) == 1L && !is.na(s) && s >= 1L && s == seeds) {
    return(42L + seq_len(s) - 1L)
  }
  if (length(s) == 0L || anyNA(s) || any(s != seeds)) {
    stop("seeds must be a count or whole numbers, e.g. 10 or c(7, 28, 42).",
         call. = FALSE)
  }
  if (anyDuplicated(s)) {
    stop("seeds has duplicates: ", paste(unique(s[duplicated(s)]), collapse = ", "),
         ". Two runs under one seed are one run, twice.", call. = FALSE)
  }
  s
}

# ── helpers: the choice (stage 04's rules, unchanged) ─────────────────────────
#
# WHAT WAS ASKED FOR AND WHAT HAPPENED ARE TWO FACTS. `rule` is the request;
# rule_applied is what the code below did, and it is that one which gets
# frozen. They diverge when the ids are named by hand (no rule ran) and when
# one_se was asked for but the run did not record what it needs (rank 1 was
# taken instead) -- and the record must be able to tell those apart.
.final_select <- function(config, rule, metric, by_config, units_tbl, grid, say, verbose) {
  pick <- NULL
  if (identical(config, "auto")) {
    se_col <- paste0(metric, "_se")
    if (!is.null(by_config) && identical(rule, "one_se") &&
        se_col %in% names(by_config) && "n_params" %in% names(by_config)) {
      pick <- one_se(by_config, metric = metric, complexity = "n_params")
      if (verbose) print_one_se(pick, metric = metric)
      ids <- pick$config_id
      rule_applied <- "one_se"
    } else if (!is.null(by_config)) {
      if (identical(rule, "one_se")) {
        missing <- setdiff(c(se_col, "n_params"), names(by_config))
        say("\n  NOTE: one_se() needs ", paste(missing, collapse = " and "),
            ", which this tuning run did not record. Falling back to rank 1.")
        rule_applied <- "rank1_after_one_se_unavailable"
      } else {
        rule_applied <- "rank1"
      }
      ids <- by_config$config_id[by_config$rank == 1L]
    } else {
      # The per-UNIT table ranks one (config, fold, seed) per row: a lucky seed
      # can outrank a steady mean there, so the record names it apart.
      rule_applied <- "rank1_from_units"
      ids <- units_tbl$config_id[units_tbl$rank == 1L]
    }
  } else {
    ids <- as.character(config)
    rule_applied <- "manual"
    say("\nConfig(s) named by hand: ", paste(ids, collapse = ", "),
        " -- no selection rule was applied, and the frozen record says so.")
  }
  ids <- unique(ids)
  gone <- setdiff(ids, grid$config_id)
  if (length(gone) > 0L) {
    stop("config id(s) not in this run's grid: ", paste(gone, collapse = ", "),
         ". A config id is a label within ONE tuning run -- the same name in ",
         "another run is another architecture.", call. = FALSE)
  }

  # If the winner's margin is smaller than the seed noise, SAY SO where the
  # choice is made. Selecting anyway is legitimate; not knowing is not.
  noise <- NA_real_
  if (!is.null(by_config) && nrow(by_config) > 1L &&
      paste0(metric, "_sd") %in% names(by_config)) {
    top2 <- by_config[order(by_config$rank), , drop = FALSE][1:2, ]
    gap  <- top2[[paste0(metric, "_mean")]][1] - top2[[paste0(metric, "_mean")]][2]
    noise <- stats::median(by_config[[paste0(metric, "_sd")]], na.rm = TRUE)
    if (is.finite(gap) && is.finite(noise) && gap < noise) {
      say("\n  WARNING: the 1st config's margin over the 2nd (", round(gap, 4),
          ") is SMALLER than the typical sd between seeds (", round(noise, 4), ").")
      say("  The two are indistinguishable at this run's number of repetitions.")
    }
  }
  list(ids = ids, rule_applied = rule_applied, rule_asked = rule, metric = metric,
       pick = pick, median_seed_sd = noise)
}

# ── helpers: the units ────────────────────────────────────────────────────────

.final_unit_record_path <- function(run_dir, config_id, seed) {
  file.path(run_dir, config_id, "units", sprintf("seed%04d.rds", seed))
}
# Where a unit's files go, under the run: its configuration's directory -- or,
# for a refit that leaves channels out (refit_importance(), R/importance_refit.R),
# the directory its `dir` names, one per variable left out.
.final_unit_dir <- function(units) {
  if ("dir" %in% names(units)) units$dir else units$config_id
}
# The channels a unit leaves out: none, but in a refit_importance() unit.
.final_unit_left_out <- function(units, u) {
  if ("left_out" %in% names(units)) as.integer(units$left_out[[u]]) else integer(0)
}
.final_unit_record <- function(run_dir, config_id, seed) {
  p <- .final_unit_record_path(run_dir, config_id, seed)
  if (file.exists(p)) readRDS(p) else NULL
}
# Done means BOTH the record says success and the checkpoint is on disk: a
# record without weights is a unit that cannot be mapped.
.final_unit_done <- function(run_dir, config_id, seed) {
  r <- .final_unit_record(run_dir, config_id, seed)
  !is.null(r) && identical(r$status, "success") &&
    file.exists(file.path(run_dir, config_id, "models", sprintf("seed%04d_best.pt", seed)))
}

# WHAT A SEED'S NUMBERS DEPEND ON, BESIDES THE SEED, RECORDED BEFORE THE FIRST
# SEED TRAINS. A resume keeps every finished seed and trains the rest, so the
# rest must be computed as the finished ones were -- or one run holds seeds
# computed two ways, its summary records only the second, and nothing in the
# ensemble shows it. T1 measured what one of these alone does: another thread
# count moved the heavy configuration's val_ccc by 0.067 in 12 epochs.
#
#   grid              the tuning run's: a config id names a network only
#                     within one grid, and another grid's cfg_003 is another
#                     network under the same name
#   split, scaling    the refit's rows and the constants its inputs are scaled
#                     with: another store, point table or validation_frac
#                     changes them
#   threads_per_unit  T1
#   training          the schedule and the clamp; print_every only prints,
#                     and is not compared
#   transform_probe   the inverse at five values: the space of every metric
#
# The seeds are not in it (adding seeds to a run is what resume is for), and
# neither are the ids: a second configuration trains in its own directory.
#
# Written when the directory is new. One that exists without it was not
# started by dsm_final() -- a final model fitted by stage 04, whose seeds a
# fit into it would overwrite -- and it is refused whatever it holds.
.final_run_spec <- function(run_dir, run_id, spec) {
  f <- file.path(run_dir, "run_spec.rds")
  if (!file.exists(f)) {
    held <- if (dir.exists(run_dir)) {
      list.files(run_dir, recursive = TRUE, all.files = TRUE, no.. = TRUE)
    } else character(0)
    if (length(held) > 0L) {
      stop("run_id \"", run_id, "\" names a directory dsm_final() did not start: ",
           run_dir, " holds ", length(held), " file(s) and no run_spec.rds -- a final ",
           "model fitted by stage 04, or one from before dsm_final() recorded its ",
           "settings. A fit into it would overwrite its seeds. Give another run_id.",
           call. = FALSE)
    }
    create_output_dirs(run_dir)
    safe_save_rds(c(spec, list(written_at = Sys.time())), f, compress = FALSE)
    return(invisible(spec))
  }
  old  <- readRDS(f)
  said <- function(v) if (is.null(v)) "(not set)" else paste(format(v, trim = TRUE), collapse = ", ")
  what <- c(grid            = "the tuning run's grid (a config id names a network only within one)",
            split           = "the refit's rows (another store, test set or validation_frac)",
            scaling         = "the predictor scaling (another store or point table)",
            transform_probe = "the inverse transform")
  diffs <- character(0)
  for (k in names(spec)) {
    a <- old[[k]]
    b <- spec[[k]]
    if (identical(k, "training")) {
      a <- a[setdiff(names(a), "print_every")]
      b <- b[setdiff(names(b), "print_every")]
    }
    if (isTRUE(all.equal(a, b))) next
    diffs <- c(diffs, if (identical(k, "training")) {
      keys <- union(names(a), names(b))
      keys <- keys[!vapply(keys, function(z) isTRUE(all.equal(a[[z]], b[[z]])), logical(1))]
      sprintf("training$%s %s, now %s", keys,
              vapply(keys, function(z) said(a[[z]]), character(1)),
              vapply(keys, function(z) said(b[[z]]), character(1)))
    } else if (identical(k, "threads_per_unit")) {
      sprintf("threads_per_unit %s, now %s", said(a), said(b))
    } else {
      what[[k]]
    })
  }
  if (length(diffs) > 0L) {
    stop("run_id \"", run_id, "\" was started with other settings than this call gives:\n  ",
         paste(diffs, collapse = "\n  "),
         "\n  Its finished seeds and the rest would be computed two ways. Resume it with ",
         "the settings it started with (", f, "), or give another run_id.",
         call. = FALSE)
  }
  invisible(old)
}

# WHAT A WORKER NEEDS IN MEMORY, from T7's trace (2026-09-28): the deployed
# cfg_003 on the dev store, private memory phase by phase.
#
#   R and torch, before anything is read      1.1 GB
#   the fold cache                            half the windows as doubles
#   what training and the final evaluation    4.7 GB, held from the first
#   keep in mimalloc's pool                   unit on and reused after it
#
# and, while a window is read, its R array (doubles) beside the cache built so
# far. That is the worker's own peak when the array outweighs the pool -- the
# full data set's 15x15 window, ~11x the dev store's 1.21 GB, is ~13 GB
# against 4.7 -- and every worker passes through it once, at its start.
# The copies the cache used to leave behind -- 2.4 GB there, about 4x the
# cache -- are gone (build_fold_cache()), and with them T2's 7x rule. An
# estimate, and said to be one: the pool is the deployed configuration's (a
# heavier network keeps more), and every worker's real peak is measured and
# reported beside it.
#
#   steady   a worker from its cache on, training
#   peak     a worker at its own peak: training, or reading its largest window
.final_worker_gb <- function(store, windows) {
  per_window <- nrow(store$meta) * store$n_channels * as.numeric(windows)^2 * 8 / 1e9
  cache  <- sum(per_window) / 2
  steady <- 1.1 + cache + 4.7
  c(steady = steady, peak = max(steady, 1.1 + cache + max(per_window)))
}

# HOW MANY WORKERS THE RAM HOLDS. The workers read the store one at a time
# (.final_one_reader()), so at any moment at most one is at its peak and the
# rest train, or wait with nothing read: all of them at `steady` and one at
# `peak`. Until 2026-09-28 every worker was counted at its peak, because every
# worker read at once. On the full data set (windows 3 and 15), by this
# estimate a worker trains at ~13 GB and peaks at ~21 while it reads: a 40 GB
# budget held one worker counted that way, and holds two counted this way.
.final_workers_that_fit <- function(est, budget) {
  max(1L, 1L + as.integer(floor((budget - est[["peak"]]) / est[["steady"]])))
}

# ONE WORKER READS THE STORE AT A TIME. Reading a window holds its whole R
# array (doubles) beside the cache being cut from it: 1.21 GB for the dev
# store's 15x15 window, ~13 GB for the full data set's. Workers started
# together read together: three would hold three, ~40 GB for a moment on a
# machine of 68, and so the budget counted every worker at that peak. Nor are
# reads side by side faster: the store's files are read whole and
# uncompressed, and on the HDD they sit on here -- the rasters' -- one reader
# alone read ~150 MB/s and fifteen together ~99 MB/s in all (2026-09-27). A
# lock directory puts the reads in turn; the training after them, where the
# hours go, still runs side by side, and a unit's numbers never depended on
# which worker trained it or when (T2).
#
# The lock is a directory, as a unit's claim is: dir.create() makes it or finds
# it made, atomically. A worker that stops with an error releases it on the
# way out; one that dies outright cannot, so the process id written inside
# says whose it is, and the lock of a process that no longer exists is taken
# over -- or one crashed worker would leave every other waiting, and the fit
# would hang with nothing said. The id is written a moment after the directory
# is made; a lock that stays without one for `no_id_s` seconds (a minute) lost
# its worker in that moment. `expr` is evaluated only once the lock is held.
.final_one_reader <- function(claims_dir, expr, no_id_s = 60) {
  lock     <- file.path(claims_dir, ".reading")
  pid_file <- file.path(lock, "pid")
  said <- FALSE
  repeat {
    if (dir.create(lock, showWarnings = FALSE)) break
    holder <- suppressWarnings(as.integer(
      tryCatch(readLines(pid_file, warn = FALSE), error = function(e) character(0))[1]))
    # ps has no pid_exists() -- that is psutil's -- and the first version
    # called it, which R CMD check caught (2026-09-28): the ids there are.
    gone <- if (is.na(holder)) {
      isTRUE(difftime(Sys.time(), file.mtime(lock), units = "secs") > no_id_s)
    } else {
      !(holder %in% ps::ps_pids())
    }
    if (gone) {
      unlink(lock, recursive = TRUE)
      next
    }
    if (!said) {
      message("  waiting for another worker to finish reading the store",
              if (is.na(holder)) "" else paste0(" (process ", holder, ")"))
      said <- TRUE
    }
    Sys.sleep(1)
  }
  on.exit(unlink(lock, recursive = TRUE), add = TRUE)
  writeLines(as.character(Sys.getpid()), pid_file)
  expr
}

# `fill` is for a refit that leaves channels out (refit_importance()): each
# channel's training mean, scaled as the inputs are, which the left-out
# channels take. NULL for dsm_final(), whose units leave nothing out.
.final_train_units <- function(todo, selected, data, index, scaling, windows,
                               training, transform, run_dir, n_cores,
                               threads_per_unit, max_ram_gb, say, fill = NULL) {
  if (!requireNamespace("callr", quietly = TRUE)) {
    stop("dsm_final() trains every seed in its own R process and needs the ",
         "callr package. install.packages(\"callr\").", call. = FALSE)
  }
  loader <- .pkg_loader()          # the framework this session runs, for each worker

  est <- .final_worker_gb(data$store, windows)
  n_workers <- max(1L, min(n_cores %/% threads_per_unit, nrow(todo)))
  budget <- max_ram_gb
  if (is.null(budget) && requireNamespace("ps", quietly = TRUE)) {
    avail <- tryCatch(ps::ps_system_memory()$avail / 1e9, error = function(e) NA_real_)
    if (is.finite(avail)) budget <- 0.7 * avail
  }
  if (!is.null(budget)) {
    fits <- .final_workers_that_fit(est, budget)
    if (fits < n_workers) {
      say("  RAM caps the workers at ", fits, " of ", n_workers, " (", sprintf("%.1f", budget),
          " GB budget, ~", sprintf("%.1f", est[["steady"]]), " GB each, one at a time up to ",
          sprintf("%.1f", est[["peak"]]), " while it reads). The numbers do not ",
          "change -- only the time.")
      n_workers <- fits
    }
    if (est[["peak"]] > budget) {
      say("  WARNING: one worker is estimated at ", sprintf("%.1f", est[["peak"]]),
          " GB at its peak, against a budget of ", sprintf("%.1f", budget),
          " GB. It is started anyway.")
    }
  }
  say(sprintf(paste0("\nTraining %d seed unit(s): %d worker(s) side by side x %d thread(s), ",
                     "~%.1f GB each, up to %.1f while one reads the store -- they read one at a time (estimate)"),
              nrow(todo), n_workers, threads_per_unit, est[["steady"]], est[["peak"]]))

  claims_dir <- file.path(run_dir, ".claims")
  logs_dir   <- file.path(run_dir, "logs")
  unlink(claims_dir, recursive = TRUE)       # stale claims of a run that died
  create_output_dirs(c(claims_dir, logs_dir))

  job <- list(
    patch_dir = normalizePath(data$patch_dir, winslash = "/", mustWork = FALSE),
    points = data$points, type_table = data$type_table, windows = windows,
    cell_size = data$cell_size, target_col = data$target_col, index = index,
    scaling = scaling, cfgs = selected, units = todo, training = training,
    transform = transform, run_dir = run_dir, claims_dir = claims_dir,
    threads = threads_per_unit, fill = fill,
    # options(dsm.final.trace_mem = TRUE): every worker records its memory
    # after each phase and each epoch (see .final_worker()). T7 does.
    trace_mem = isTRUE(getOption("dsm.final.trace_mem", FALSE)))
  .pkg_check_portable(job, "dsm_final()")

  t0 <- Sys.time()
  procs <- lapply(seq_len(n_workers), function(w) {
    callr::r_bg(
      .final_worker_entry,
      args = list(job = c(job, list(worker = w)), loader = loader),
      env = c(callr::rcmd_safe_env(),
              OMP_NUM_THREADS = as.character(threads_per_unit),
              MKL_NUM_THREADS = as.character(threads_per_unit),
              # A diagnosis can add to the workers' environment, as dsm_predict()'s
              # can (T7 sets MIMALLOC_SHOW_STATS).
              getOption("dsm.final.worker_env", character(0))),
      stdout = file.path(logs_dir, sprintf("worker_%02d.log", w)), stderr = "2>&1",
      supervise = TRUE)
  })

  n_seen <- -1L
  repeat {
    alive  <- vapply(procs, function(p) p$is_alive(), logical(1))
    n_done <- sum(file.exists(.final_unit_record_path(run_dir, .final_unit_dir(todo), todo$seed)))
    if (n_done != n_seen) {
      say(sprintf("  %5.1f min | %d of %d unit(s) finished | %d worker(s) running",
                  as.numeric(difftime(Sys.time(), t0, units = "mins")), n_done,
                  nrow(todo), sum(alive)))
      n_seen <- n_done
    }
    if (!any(alive)) break
    Sys.sleep(5)
  }

  res  <- lapply(procs, function(p) tryCatch(p$get_result(), error = function(e) e))
  errs <- vapply(res, function(r) inherits(r, "error"), logical(1))
  if (any(errs)) {
    for (w in which(errs)) {
      say("  worker ", w, " stopped with an error: ", conditionMessage(res[[w]]),
          "\n    log: ", file.path(logs_dir, sprintf("worker_%02d.log", w)))
    }
  }
  peaks <- vapply(res[!errs], function(r) as.numeric(r$peak_gb %||% NA_real_), numeric(1))
  peak  <- if (any(is.finite(peaks))) max(peaks[is.finite(peaks)]) else NA_real_
  say(sprintf("  workers done in %.1f min | peak RAM per worker %.1f GB (estimated %.1f)",
              as.numeric(difftime(Sys.time(), t0, units = "mins")), peak, est[["peak"]]))
  unlink(claims_dir, recursive = TRUE)
  list(n_workers = n_workers, threads_per_unit = threads_per_unit, peak_gb = peak,
       per_worker_gb_estimate = est[["peak"]])
}

# What each worker process runs. At the top level on purpose, as prepare.R's
# band worker is: callr sends a function without its enclosing frame unless
# asked to, but a function defined inside .final_train_units() would still
# LOOK as if it carried that frame -- the store, the tensors -- and nothing
# here should depend on a detail of how another package serialises. The
# framework is loaded the way the session loaded it, and the worker taken
# from its namespace, where it is not exported.
.final_worker_entry <- function(job, loader) {
  ns <- loader$open(loader)
  get(".final_worker", envir = ns)(job)
}

# ONE WORKER: its own copy of the store and the fold cache, then units claimed
# one at a time until none is left. A claim is a directory: dir.create() either
# makes it or finds it made, atomically, so two workers never train the same
# seed -- and which worker trains which seed does not matter, because T2
# showed a unit's numbers do not depend on it.
.final_worker <- function(job) {
  set_torch_threads(job$threads)

  # THE WORKER'S MEMORY, PHASE BY PHASE AND EPOCH BY EPOCH, when the fit is
  # asked for it (options(dsm.final.trace_mem = TRUE); T7 does): the training
  # side of what T4 did for the map. T2 measured ~10 GB at the peak of a
  # worker holding 1.26 GB of windows, and a peak says neither which phase set
  # it nor whether the level climbs from epoch to epoch or unit to unit. Each
  # mark follows a collection, so a level is what is held rather than garbage
  # waiting; the private memory is read beside the working set because Windows
  # takes pages out of a working set when the PC needs them, and a leak hides
  # there (P5). Written after the setup and after every unit, so a worker that
  # dies leaves its trace up to its last unit.
  #
  # The phases: start (the framework loaded), store_loaded (the table: no
  # window is loaded here), fold_cache (every role of every window, read
  # from the store's files), then each unit's.
  t0 <- Sys.time()
  trace <- list()
  mark <- function(phase, unit_id = NA_character_, epoch = NA_integer_) {
    if (!isTRUE(job$trace_mem)) return(invisible(NULL))
    g <- gc(verbose = FALSE)
    trace[[length(trace) + 1L]] <<- data.frame(
      worker = job$worker, unit_id = unit_id, phase = phase, epoch = as.integer(epoch),
      seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
      rss_gb = .predict_rss_gb(), private_gb = .predict_private_gb(),
      peak_gb = .final_peak_gb(), r_heap_gb = sum(g[, 2]) / 1024,
      stringsAsFactors = FALSE)
    invisible(NULL)
  }
  save_trace <- function() {
    if (isTRUE(job$trace_mem) && length(trace) > 0L) {
      safe_save_rds(do.call(rbind, trace),
                    file.path(job$run_dir, "logs", sprintf("worker_%02d_mem_trace.rds", job$worker)),
                    compress = FALSE)
    }
  }

  mark("start")
  # THE TABLE ONLY. The fold cache reads each window from the store's file and
  # keeps only the roles cut from it; a window loaded here as a tensor would be
  # released once the cache existed, and mimalloc keeps a block that size
  # (T7: the store's raw windows were 0.61 GB of the 2.4 left behind).
  data <- dsm_load(job$patch_dir, points = job$points, type_table = job$type_table,
                   windows = integer(0), cell_size = job$cell_size,
                   target_col = job$target_col, verbose = FALSE)
  mark("store_loaded")
  fold <- .final_one_reader(job$claims_dir, build_fold_cache(
    data$store, data$points, data$type_table, job$index, job$windows,
    scaling = job$scaling, verbose = FALSE))
  mark("fold_cache")
  pv    <- fold_points_valid(data$store, job$index)
  n_ch  <- data$store$n_channels
  cache <- fold$cache
  rm(fold)
  save_trace()
  device <- torch::torch_device("cpu")

  for (u in seq_len(nrow(job$units))) {
    cid  <- job$units$config_id[u]
    seed <- job$units$seed[u]
    uid  <- job$units$unit_id[u]
    if (!dir.create(file.path(job$claims_dir, uid), showWarnings = FALSE)) next
    cfg <- job$cfgs[job$cfgs$config_id == cid, , drop = FALSE]
    unit_dir <- .final_unit_dir(job$units[u, , drop = FALSE])
    cfg_dir  <- file.path(job$run_dir, unit_dir)
    left_out <- .final_unit_left_out(job$units, u)
    message("\n-- [", cid, "] seed ", seed, " (worker ", job$worker, ", ",
            job$threads, " thread(s))",
            if (length(left_out)) sprintf(", %d channel(s) left out (%s)", length(left_out), unit_dir),
            " --")

    # LEFT OUT, for refit_importance(): the unit's channels take their training
    # mean in every window and role of the cache, and are put back once the
    # unit is done, so the next unit trains from the cache as it was read.
    # Before the seeds are set: it draws no random number, and the unit then
    # trains exactly as the run's seed did, but for those channels.
    kept <- .importance_leave_out(cache, left_out, job$fill)

    # Stage 04's order: the seeds, then the loaders -- the loaders shuffle
    # with torch's generator, so they must be built after it is seeded.
    set.seed(seed)
    torch::torch_manual_seed(seed)
    loaders <- .make_loaders_from_cache(cache, cfg)
    mark("unit_start", uid)
    # The hook reads memory and nothing else: no tensor, no RNG, so a traced
    # unit trains exactly as an untraced one (tests/test_final.R asserts it).
    on_epoch <- if (isTRUE(job$trace_mem)) function(epoch) mark("epoch", uid, epoch) else NULL
    result <- tryCatch(
      do.call(train_one_cnn, c(list(
        cfg = cfg, n_channels = n_ch, loaders = loaders, points_valid = pv,
        transform = job$transform, device = device,
        model_name = paste0(cid, "_seed", seed), on_epoch = on_epoch), job$training)),
      error = function(e) e)
    mark("unit_trained", uid)

    rec <- list(unit_id = uid, config_id = cid, seed = seed,
                threads = job$threads, worker = job$worker, finished_at = Sys.time())
    if (length(left_out)) rec$left_out <- left_out
    if (inherits(result, "error")) {
      rec$status <- "failed"
      rec$error  <- conditionMessage(result)
      message("  ERROR [", cid, "] seed ", seed, ": ", rec$error)
    } else {
      sl <- sprintf("seed%04d", seed)
      safe_torch_save(result$best_state, file.path(cfg_dir, "models", paste0(sl, "_best.pt")))
      safe_write_csv2(result$history,       file.path(cfg_dir, "history",     paste0(sl, "_history.csv")))
      safe_write_csv2(result$pred_all,      file.path(cfg_dir, "predictions", paste0(sl, "_pred_all.csv")))
      safe_write_csv2(result$perf_all,      file.path(cfg_dir, "metrics",     paste0(sl, "_perf.csv")))
      safe_write_csv2(result$perf_quantile, file.path(cfg_dir, "metrics",     paste0(sl, "_perf_quantile.csv")))
      if (!is.null(result$gate)) {
        safe_write_csv2(result$gate$summary,    file.path(cfg_dir, "gates", paste0(sl, "_gate_summary.csv")))
        safe_write_csv2(result$gate$by_profile, file.path(cfg_dir, "gates", paste0(sl, "_gate_profiles.csv")))
      }
      rec$status      <- "success"
      rec$best_epoch  <- result$best_epoch
      rec$runtime_min <- result$runtime_min
      message("  [", cid, "] seed ", seed, " | best_ep ", result$best_epoch,
              sprintf(" | %.1f min", result$runtime_min))
    }
    # The record LAST: its existence is what says the unit is finished.
    safe_save_rds(rec, .final_unit_record_path(job$run_dir, unit_dir, seed), compress = FALSE)
    .importance_put_back(cache, kept)
    rm(result, loaders)
    invisible(gc(verbose = FALSE))
    mark("unit_released", uid)
    save_trace()
  }
  list(worker = job$worker, peak_gb = .final_peak_gb())
}

# The process's peak working set on Windows, its resident size elsewhere.
.final_peak_gb <- function() {
  if (!requireNamespace("ps", quietly = TRUE)) return(NA_real_)
  mi <- tryCatch(ps::ps_memory_info(ps::ps_handle()), error = function(e) NULL)
  if (is.null(mi)) return(NA_real_)
  as.numeric(if ("peak_wset" %in% names(mi)) mi[["peak_wset"]] else mi[["rss"]]) / 1e9
}

# ── helpers: from N seeds to the ensemble, the interval, the mean surface ─────
#
# Stage 04's post-processing, read back from the per-seed files rather than
# from memory -- so it can also be run over a stage-04 run's files, which is
# how _p3_final_assembly_check.R proves it computes what 04 computed.
.final_assemble_config <- function(cfg_dir, cfg_out_dir, config_id, seeds,
                                   tuning_dir, conformal_alpha,
                                   transform_name = "log1p", verbose = TRUE,
                                   write = TRUE) {
  say <- function(...) if (verbose) message(...)
  if (write) create_output_dirs(cfg_out_dir)
  seed_rows <- list(); seed_preds <- list()
  for (s in seeds) {
    sl   <- sprintf("seed%04d", s)
    perf <- safe_read_csv2(file.path(cfg_dir, "metrics", paste0(sl, "_perf.csv")))
    hist <- safe_read_csv2(file.path(cfg_dir, "history", paste0(sl, "_history.csv")))
    rec  <- .final_unit_record(dirname(cfg_dir), config_id, s)
    best_epoch <- if (!is.null(rec$best_epoch)) rec$best_epoch else utils::tail(hist$best_epoch, 1L)
    seed_rows[[length(seed_rows) + 1L]] <- dplyr::filter(perf, .data$dataset_role == "test") %>%
      dplyr::mutate(config_id = config_id, seed = s,
                    best_epoch = best_epoch,
                    runtime_min = rec$runtime_min %||% NA_real_)
    pa <- safe_read_csv2(file.path(cfg_dir, "predictions", paste0(sl, "_pred_all.csv")))
    seed_preds[[length(seed_preds) + 1L]] <- pa %>%
      dplyr::filter(.data$dataset_role %in% c("validation", "test")) %>%
      dplyr::select(sample_id, dataset_role, obs, pred) %>%
      dplyr::mutate(seed = s)
  }
  seed_rows <- dplyr::bind_rows(seed_rows)
  preds <- dplyr::bind_rows(seed_preds)

  ens <- preds %>%
    dplyr::group_by(.data$sample_id, .data$dataset_role) %>%
    dplyr::summarise(obs = dplyr::first(.data$obs),
                     pred = stats::median(.data$pred),
                     spread = stats::sd(.data$pred),
                     n_seeds = dplyr::n(), .groups = "drop")
  if (write) safe_write_csv2(ens, file.path(cfg_out_dir, "ensemble_predictions.csv"))

  # WHICH RESIDUALS CALIBRATE (stage 04's finding): the tuning run's
  # cross-validated residuals, which cover every block, and not the refit's
  # own validation -- one fold from one region, the easiest of the three sets,
  # which gave 83.6% coverage against a nominal 90%. The refit's validation is
  # the fallback for a tuning run that wrote no predictions.
  cv_cal <- cv_residuals(tuning_dir, config_id)
  cal_source <- "cross-validated residuals (all folds)"
  cal_rows <- if (!is.null(cv_cal) && nrow(cv_cal) >= nrow(ens) / 4) {
    dplyr::mutate(cv_cal, spread = NA_real_)
  } else {
    cal_source <- "the refit's validation split (one fold)"
    dplyr::filter(ens, .data$dataset_role == "validation")
  }
  chk_rows <- dplyr::filter(ens, .data$dataset_role == "test")
  say("\nConformal calibration set: ", cal_source, " -- ", nrow(cal_rows), " point(s)")

  conformal <- list()
  if (nrow(cal_rows) >= 9L && nrow(chk_rows) > 0L) {
    for (a in conformal_alpha) {
      cal <- conformal_calibrate(cal_rows$obs, cal_rows$pred, alpha = a)
      iv  <- conformal_interval(cal, chk_rows$pred, lower_limit = 0)
      pr  <- picp_report(chk_rows$obs, iv$lower, iv$upper, alpha = a)
      if (verbose) {
        message("\n-- [", config_id, "] conformal, ", round(100 * (1 - a)), "% --")
        print(cal)
        print(pr)
      }
      # The level-and-dissimilarity interval is dsm_predict()'s (step 4): the
      # difficulty score must be computed the same way for a calibration point
      # and for a map pixel, which the seed spread here is not.
      if (write) {
        safe_save_rds(
          list(alpha = a, constant = cal, normalised = NULL,
               n_calibration = nrow(cal_rows), config_id = config_id,
               calibrated_on = "validation", checked_on = "test",
               calibration_source = cal_source),
          file.path(cfg_out_dir, sprintf("conformal_%02d.rds", round(100 * (1 - a)))),
          compress = FALSE)
      }
      conformal[[as.character(a)]] <- list(cal = cal, picp = pr)
    }
  } else {
    say("\n-- [", config_id, "] conformal skipped: ", nrow(cal_rows),
        " calibration point(s), ", nrow(chk_rows), " to check on.")
  }

  # The mean surface beside the median one (R/smearing.R): one scalar,
  # calibrated on the same out-of-fold residuals. Only a log1p model needs it.
  sm <- NULL
  smear_test <- NULL
  if (identical(transform_name, "log1p")) {
    sm <- smearing_from_run(tuning_dir, config_id)
    if (!is.null(sm)) {
      if (verbose) print(sm)
      if (write) safe_save_rds(sm, file.path(cfg_out_dir, "smearing.rds"), compress = FALSE)
      tst <- dplyr::filter(preds, .data$dataset_role == "test") %>%
        dplyr::group_by(.data$sample_id) %>%
        dplyr::summarise(obs = dplyr::first(.data$obs),
                         pred = stats::median(.data$pred), .groups = "drop")
      if (nrow(tst) > 0L) {
        corrected <- smear(log1p(tst$pred), sm, lower_limit = 0)
        smear_test <- dplyr::bind_rows(
          dplyr::mutate(calc_metrics(tst$obs, tst$pred), surface = "median (inverse)", .before = 1),
          dplyr::mutate(calc_metrics(tst$obs, corrected), surface = "mean (smeared)", .before = 1))
        smear_test$total_pct_vs_obs <- c(100 * (sum(tst$pred) / sum(tst$obs) - 1),
                                         100 * (sum(corrected) / sum(tst$obs) - 1))
        if (verbose) {
          message("\n-- [", config_id, "] back-transform, on the test set --")
          print(smear_test)
        }
      }
    } else {
      say("\nSmearing: the tuning run wrote no transform-space residuals, so no mean surface.")
    }
  }

  ens_test <- if (nrow(chk_rows) > 0L) calc_metrics(chk_rows$obs, chk_rows$pred) else NULL
  list(seed_rows = seed_rows, ensemble = ens, ensemble_test = ens_test,
       conformal = conformal, calibration_source = cal_source, smearing = sm,
       smear_test = smear_test)
}

# Mean +/- sd between seeds, per config -- stage 04's table.
.final_config_summary <- function(all_seed_results) {
  if (nrow(all_seed_results) == 0L) return(tibble::tibble())
  all_seed_results %>%
    dplyr::group_by(config_id) %>%
    dplyr::summarise(
      n_seeds   = dplyr::n(),
      ccc_mean  = mean(ccc),  ccc_sd  = stats::sd(ccc),
      r2_mean   = mean(r2),   r2_sd   = stats::sd(r2),
      mae_mean  = mean(mae),  mae_sd  = stats::sd(mae),
      nse_mean  = mean(nse),  nse_sd  = stats::sd(nse),
      rmse_mean = mean(rmse), rmse_sd = stats::sd(rmse),
      rpd_mean  = mean(rpd),  rpd_sd  = stats::sd(rpd),
      mqi_mean  = mean(mqi),  mqi_sd  = stats::sd(mqi),
      best_epoch_mean   = mean(best_epoch),
      runtime_min_total = sum(runtime_min),
      .groups = "drop") %>%
    dplyr::arrange(dplyr::desc(ccc_mean))
}

# Two configs, compared seed by seed: the same seed is the same draw, so the
# difference within a seed is the architecture's. More than two is a choice of
# pair this function will not make; it says so rather than skipping quietly.
.final_paired <- function(all_seed_results, ids, run_dir, verbose) {
  if (length(ids) > 2L && verbose) {
    message("\n-- Paired comparison skipped: ", length(ids), " configs. It compares ",
            "exactly two, seed by seed.")
  }
  if (length(ids) != 2L) return(invisible(NULL))
  a <- all_seed_results[all_seed_results$config_id == ids[1], c("seed", "ccc", "mae", "rmse", "mqi")]
  b <- all_seed_results[all_seed_results$config_id == ids[2], c("seed", "ccc", "mae", "rmse", "mqi")]
  p <- dplyr::inner_join(a, b, by = "seed", suffix = paste0("_", ids))
  for (m in c("ccc", "mae", "rmse", "mqi")) {
    p[[paste0("d_", m)]] <- p[[paste0(m, "_", ids[1])]] - p[[paste0(m, "_", ids[2])]]
  }
  safe_write_csv2(p, file.path(run_dir, "comparison", "paired_by_seed.csv"))
  if (verbose) {
    message(sprintf("\n-- Paired difference (%s - %s), mean over seeds: dCCC %+.4f | dMAE %+.3f",
                    ids[1], ids[2], mean(p$d_ccc), mean(p$d_mae)))
  }
  invisible(p)
}

# ── helpers: the declaration of the selected CNN ──────────────────────────────
#
# EVERY HYPERPARAMETER, AND WHETHER THE SEARCH CHOSE IT. A value in the chosen
# row is only "the optimum" if the grid offered alternatives: a parameter the
# grid held fixed was decided by whoever wrote the grid, not by the data. The
# table says which is which, and lists what was tried, so a reader can tell a
# learning rate the search picked from three from a batch size nobody varied.
.final_param_meaning <- c(
  window_sizes    = "patch size(s) in pixels: one is a single branch, two a dual branch with a gate",
  conv_channels   = "feature maps per convolution block; the count of blocks is the depth",
  use_residual    = "residual (skip) connection around each convolution block",
  use_se_block    = "squeeze-and-excitation channel attention",
  se_reduction    = "bottleneck ratio of the SE block",
  embedding_dim   = "size of each branch's embedding",
  embed_pool      = "how a branch reduces its map before the embedding: flatten keeps every cell, gap averages",
  conv_padding    = "same pads with zeros; valid_large keeps only measured pixels on the large branch",
  gate_type       = "how the two branches are fused",
  dropout         = "overall regularisation strength, mapped to the five dropout sites below",
  spatial_dropout = "derived from dropout (x 0.25)",
  embed_dropout   = "derived from dropout (always 0)",
  gate_dropout    = "derived from dropout (x 0.5)",
  head_dropout_1  = "derived from dropout (x 1)",
  head_dropout_2  = "derived from dropout (x 0.5)",
  base_lr         = "peak learning rate after warmup (Adam)",
  weight_decay    = "L2 penalty on the weights",
  batch_size      = "samples per gradient step",
  loss_fn         = "training loss, in the transformed space",
  warmup_epochs   = "epochs of linear learning-rate warmup",
  n_params        = "trainable parameters of the network (a consequence, not a choice)")

.final_param_group <- function(p) {
  if (p %in% c("window_sizes", "conv_channels", "use_residual", "use_se_block",
               "se_reduction", "embedding_dim", "embed_pool", "conv_padding",
               "gate_type", "n_params")) return("architecture")
  if (p %in% c("dropout", "spatial_dropout", "embed_dropout", "gate_dropout",
               "head_dropout_1", "head_dropout_2", "weight_decay")) return("regularisation")
  "optimisation"
}

.final_fmt_value <- function(v, p) {
  if (is.list(v)) v <- v[[1]]
  if (identical(p, "window_sizes")) return(paste(v, collapse = "x"))
  if (identical(p, "conv_channels")) return(paste(v, collapse = "_"))
  if (is.numeric(v)) return(format(v, digits = 6, trim = TRUE, scientific = abs(v) < 1e-3 && v != 0))
  as.character(v)
}

# A PARAMETER THE NETWORK DOES NOT USE WAS NOT CHOSEN. A single-branch
# network has no gate, so its gate_type is whatever the grid forces
# (no_gate_concat) and its gate dropout multiplies nothing; without an SE block
# the SE bottleneck ratio is never built. Declaring those "tuned" because the
# grid's other configs varied them -- as the first version of this table did
# for the deployed SOC model, a single 15x15 branch -- claims a choice nobody
# made. Returns NA when the parameter is used, the reason when it is not.
.final_param_unused <- function(p, cfg_row) {
  one_branch <- length(cfg_row$window_sizes[[1]]) < 2L
  if (p %in% c("gate_type", "gate_dropout") && one_branch) return("one branch, no gate")
  if (identical(p, "se_reduction") && !isTRUE(as.logical(cfg_row$use_se_block))) {
    return("no SE block")
  }
  NA_character_
}

.final_hyper_table <- function(selected, grid, training) {
  params <- setdiff(names(grid), "config_id")
  rows <- list()
  for (i in seq_len(nrow(selected))) {
    cid <- selected$config_id[i]
    for (p in params) {
      col   <- grid[[p]]
      tried <- unique(vapply(seq_along(col), function(k) .final_fmt_value(col[k], p), character(1)))
      if (is.numeric(col)) tried <- tried[order(as.numeric(unique(col)))]
      unused  <- .final_param_unused(p, selected[i, , drop = FALSE])
      meaning <- unname(.final_param_meaning[p]) %||% NA_character_
      if (!is.na(unused)) meaning <- paste0(meaning, " -- NOT USED by this network (", unused, ")")
      rows[[length(rows) + 1L]] <- tibble::tibble(
        config_id = cid, group = .final_param_group(p), parameter = p,
        value = .final_fmt_value(selected[[p]][i], p),
        used = is.na(unused),
        searched = is.na(unused) && length(tried) > 1L && !identical(p, "n_params"),
        values_tried = paste(tried, collapse = ", "),
        meaning = meaning)
    }
    for (p in names(training)) {
      rows[[length(rows) + 1L]] <- tibble::tibble(
        config_id = cid, group = "final refit", parameter = p,
        value = paste(format(training[[p]], trim = TRUE), collapse = ", "),
        used = TRUE, searched = FALSE, values_tried = NA_character_,
        meaning = "the refit's training schedule (set, not searched)")
    }
  }
  out <- dplyr::bind_rows(rows)
  out$meaning[is.na(out$meaning)] <- ""
  out
}

# ── helpers: the report ───────────────────────────────────────────────────────
.final_write_report <- function(x) {
  f <- function(v, d = 4) if (is.null(v) || length(v) == 0L || !is.finite(v)) "NA" else formatC(v, digits = d, format = "f")
  bc <- x$by_config
  L <- c(sprintf("# Final model -- %s", x$target %||% "target"), "",
         sprintf("Run `%s`, fitted by %s. This report written %s.", basename(x$run_dir),
                 x$fitted_by %||% "`dsm_final()`", format(Sys.time(), "%Y-%m-%d %H:%M")), "",
         "## How it was chosen", "",
         sprintf("- Tuning run: `%s` -- %d configuration(s) in the grid, %s unit(s) trained.",
                 basename(x$tuning_dir), nrow(x$grid), format(nrow(x$units_tbl), big.mark = ",")),
         sprintf("- Rule applied: **%s** on `%s` (asked: %s).", x$selection$rule_applied,
                 x$selection$metric, x$selection$rule_asked))
  if (!is.null(bc) && nrow(bc) > 0L) {
    m <- x$selection$metric
    best <- bc[order(bc$rank), , drop = FALSE][1, ]
    L <- c(L, sprintf("- Best mean: `%s`, %s = %s +/- %s over %s unit(s).", best$config_id, m,
                      f(best[[paste0(m, "_mean")]]), f(best[[paste0(m, "_sd")]]),
                      best$n_units %||% "?"))
    for (cid in x$selected_config_ids) {
      r <- bc[bc$config_id == cid, , drop = FALSE]
      if (nrow(r) == 1L) {
        L <- c(L, sprintf("- Selected: `%s`, %s = %s +/- %s (se %s), rank %s of %d%s.", cid, m,
                          f(r[[paste0(m, "_mean")]]), f(r[[paste0(m, "_sd")]]),
                          f(r[[paste0(m, "_se")]] %||% NA_real_), r$rank, nrow(bc),
                          if ("n_params" %in% names(r)) sprintf(", %s parameters", format(r$n_params, big.mark = ",")) else ""))
      }
    }
    if (!is.null(x$selection$pick)) {
      L <- c(L, sprintf("- one_se: %d config(s) within one standard error of the best; the simplest of them was taken.",
                        attr(x$selection$pick, "within_one_se")))
    }
    if (is.finite(x$selection$median_seed_sd)) {
      L <- c(L, sprintf("- Typical sd between seeds in the tuning run: %s -- a gap smaller than this is not evidence.",
                        f(x$selection$median_seed_sd)))
    }
  }
  for (cid in x$selected_config_ids) {
    h <- x$hyper[x$hyper$config_id == cid, , drop = FALSE]
    L <- c(L, "", sprintf("## `%s`: every hyperparameter", cid), "",
           "| group | parameter | value | chosen by | values tried | meaning |",
           "|---|---|---|---|---|---|")
    for (i in seq_len(nrow(h))) {
      by <- if (identical(h$group[i], "final refit")) "set for the refit"
            else if (!isTRUE(h$used[i])) "not used by this network"
            else if (isTRUE(h$searched[i])) "**the search**" else "fixed in the grid"
      L <- c(L, sprintf("| %s | `%s` | %s | %s | %s | %s |", h$group[i], h$parameter[i], h$value[i],
                        by, ifelse(is.na(h$values_tried[i]), "", h$values_tried[i]), h$meaning[i]))
    }
  }
  L <- c(L, "", "## The refit", "",
         sprintf("- Split from the tuning plan%s: %d training, %d validation (early stopping), %d test.",
                 if (is.null(x$refit_method)) "" else sprintf(", by its own criterion (`%s`)", x$refit_method),
                 x$split[["train"]], x$split[["validation"]], x$split[["test"]]),
         if (is.na(x$threads_per_unit)) {
           sprintf("- Seeds: %s. The thread count they trained with was not recorded by the stage that fitted them -- and a seed's numbers depend on it (T1).",
                   paste(x$seeds, collapse = ", "))
         } else {
           sprintf("- Seeds: %s -- each with **%d thread(s)**%s. A seed's numbers depend on its seed and its thread count, and on nothing else (T2).",
                   paste(x$seeds, collapse = ", "), x$threads_per_unit,
                   if (isTRUE(x$n_workers > 0L)) sprintf(", %d trained side by side", x$n_workers) else "")
         },
         if (length(x$training) == 0L) "- The refit's training schedule was not recorded by the stage that fitted it." else NULL,
         if (is.finite(x$peak_gb %||% NA_real_)) sprintf("- Peak RAM per worker: %.1f GB (estimated %.1f).", x$peak_gb, x$per_worker_gb_estimate) else NULL)
  for (cid in x$selected_config_ids) {
    s  <- x$config_summary[x$config_summary$config_id == cid, , drop = FALSE]
    pc <- x$per_config[[cid]]
    L <- c(L, "", sprintf("## `%s` on the test set", cid), "")
    if (nrow(s) == 1L) {
      L <- c(L, sprintf("- Per seed, mean +/- sd over %d: CCC %s +/- %s | MAE %s +/- %s | RMSE %s +/- %s | R2 %s +/- %s | best epoch %s on average.",
                        s$n_seeds, f(s$ccc_mean), f(s$ccc_sd), f(s$mae_mean, 3), f(s$mae_sd, 3),
                        f(s$rmse_mean, 3), f(s$rmse_sd, 3), f(s$r2_mean), f(s$r2_sd), f(s$best_epoch_mean, 1)))
    }
    if (!is.null(pc$ensemble_test)) {
      e <- pc$ensemble_test
      L <- c(L, sprintf("- The ensemble median: CCC %s | MAE %s | RMSE %s.", f(e$ccc), f(e$mae, 3), f(e$rmse, 3)))
    }
    for (a in names(pc$conformal)) {
      cc <- pc$conformal[[a]]
      L <- c(L, sprintf("- Conformal %d%%: half-width %s, from %d calibration point(s) (%s); coverage on the test set %s.",
                        round(100 * (1 - as.numeric(a))), f(cc$cal$q, 3), cc$cal$n,
                        pc$calibration_source, f(100 * cc$picp$overall$picp %||% NA_real_, 1)))
    }
    if (!is.null(pc$smearing)) {
      L <- c(L, sprintf("- Smearing factor: %s (the mean surface is the median one times it, on the original scale).",
                        f(pc$smearing$s)))
    }
  }
  L <- c(L, "", "## Files", "",
         "- `selected_hyperparameters.csv` -- the tables above, one row per hyperparameter",
         "- `comparison/final_run_summary.rds` -- the choice, the seeds, the threads, the results",
         "- `<config>/models/seed####_best.pt`, `<config>/predictor_scaling.csv` -- what a map needs",
         "- `<config>/ensemble_predictions.csv`, `conformal_##.rds`, `smearing.rds`")
  path <- file.path(x$run_dir, "final_report.md")
  writeLines(enc2utf8(L), path, useBytes = TRUE)
  path
}
