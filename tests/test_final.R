# Unit test: dsm_final() -- the selected configuration refitted under N seeds,
# side by side, and the declaration of every hyperparameter it was chosen with
#
# WHAT IS VERIFIED.
#
#   1. the layout stage 05 reads is written: the summary, one checkpoint per
#      seed, the scaling, the ensemble, the interval, the mean-surface factor
#   2. the choice is frozen in the TUNING run, as stage 04 froze it
#   3. SIDE BY SIDE CHANGES NO NUMBER: two workers of one thread and one worker
#      of one thread give every seed identical metrics, bit for bit -- T2's
#      finding, kept by the suite at the size of a test. The one worker runs
#      with the memory trace on, so the trace is held to it as well
#   4. the report declares every hyperparameter of the grid, says which the
#      search varied, and lists what it tried
#   5. an interrupted fit resumes without retraining a finished seed
#   6. the arguments that can only be wrong are refused before anything trains;
#      a resume only with the settings its run started with; and a refused
#      call freezes no choice
#   7. a final model dsm_final() did not start gets its declaration, and is
#      never fitted into
#
# The fixture: 96 points in 8 sites two degrees apart, 3 channels, window 3 --
# enough blocks that a spatial tuning plan and its refit split both have
# something to cut. Every unit trains for 3 epochs: this is about the wiring.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_final.R")
# (trains; ~1-2 min on CPU, most of it starting worker processes)

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)
  library(readr)
})

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
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
.load_framework(root)

ok <- c()
err <- function(expr) {
  e <- tryCatch({ suppressMessages(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(e)) "" else e
}

# ── the fixture: a store on disk, and a tuning run over it ───────────────────
set.seed(20260927)
n_site <- 8L; per <- 12L; n_pts <- n_site * per; n_ch <- 3L; win <- 3L
preds <- paste0("p", seq_len(n_ch))
CELL  <- 0.00224579811173295
site <- rep(seq_len(n_site), each = per)
x <- ((site - 1L) %% 4L) * 2 + stats::runif(n_pts, 0, 0.05)
y <- ((site - 1L) %/% 4L) * 2 + stats::runif(n_pts, 0, 0.05)
arr <- array(stats::rnorm(n_pts * n_ch * win * win), dim = c(n_pts, n_ch, win, win))
centre <- (win + 1L) %/% 2L
y_true <- 2 * arr[, 1, centre, centre] + stats::rnorm(n_pts, sd = 0.5)
y_true <- y_true - min(y_true) + 1
meta <- tibble::tibble(profile_id = seq_len(n_pts), sample_id = seq_len(n_pts),
                       x = x, y = y, target_native = y_true,
                       target_transform = log1p(y_true))

base      <- file.path(tempdir(), "dlc_final_test")
unlink(base, recursive = TRUE)                 # on the way IN: a leftover would be resumed
store_dir <- file.path(base, "store")
out_root  <- file.path(base, "out")
dir.create(store_dir, recursive = TRUE)
invisible(save_patch_window(arr, store_dir, win))
readr::write_csv2(meta, file.path(store_dir, "patch_meta.csv"))
saveRDS(tibble::tibble(
  scaling_applied = FALSE, predictor_cols_final = paste(preds, collapse = ";"),
  n_channels = n_ch, windows_extracted = as.character(win), n_points_valid = n_pts,
  target_col = "soc_stock", target_transform = "log1p", cell_size = CELL),
  file.path(store_dir, "patch_manifest.rds"))
points <- meta
for (j in seq_len(n_ch)) points[[preds[j]]] <- arr[, j, centre, centre]
type_table <- tibble::tibble(predictor = preds, is_dummy = FALSE, is_percentage = FALSE)

data <- suppressMessages(dsm_load(store_dir, points, type_table,
                                  target_col = "soc_stock", verbose = FALSE))

# Two configs that differ in ONE hyperparameter, so the report has one
# "tuned" row to find and every other row "fixed".
grid <- make_manual_tune_grid(
  window_sizes = list(c(win)), conv_channels = list(c(4L)),
  embedding_dim = c(8L, 16L), base_lr = 0.01, batch_size = 8L, dropout = 0.0,
  gate_type = "no_gate_concat", use_residual = FALSE, use_se_block = FALSE)

fit <- suppressMessages(dsm_train(
  data, model = "cnn",
  resampling = spatial_cv(k = 2L, block_size = 1, buffer = "auto", test_frac = 0.2),
  tune_grid = grid, n_seeds = 2L, output_dir = out_root, run_id = "tuning",
  device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
  n_epochs = 3L, patience = 3L, print_every = 100L, augment = FALSE, verbose = FALSE))

final_args <- list(
  seeds = c(1L, 2L, 3L), validation_frac = 0.34, threads_per_unit = 1L,
  training = list(n_epochs = 3L, patience = 3L, print_every = 100L, augment = FALSE),
  output_dir = file.path(out_root, "final_model"), verbose = FALSE)

# ── 1-2. two workers side by side ────────────────────────────────────────────
fin <- suppressMessages(do.call(dsm_final, c(list(fit, n_cores = 2L, run_id = "par"),
                                             final_args)))
rd  <- fin$run_dir
cid <- fin$selected_config_ids[1]
sm  <- readRDS(file.path(rd, "comparison", "final_run_summary.rds"))

ok["the_summary_stage_05_reads_is_written"] <-
  all(c("selected_cfgs", "selected_config_ids", "selection_rule", "seeds",
        "seeds_fitted", "all_seed_results", "config_summary", "run_id",
        "tuning_run_id") %in% names(sm))
ok["every_seed_has_its_checkpoint"] <-
  all(file.exists(file.path(rd, cid, "models", sprintf("seed%04d_best.pt", 1:3))))
ok["the_scaling_travels_with_the_weights"] <- {
  sc <- safe_read_csv2(file.path(rd, cid, "predictor_scaling.csv"))
  identical(as.character(sc$predictor), preds)
}
ok["the_ensemble_the_interval_and_the_mean_factor_are_written"] <-
  all(file.exists(file.path(rd, cid, c("ensemble_predictions.csv", "conformal_90.rds",
                                       "smearing.rds"))))
ok["the_run_records_its_threads_and_workers"] <-
  identical(sm$threads_per_unit, 1L) && identical(sm$n_workers, 2L) &&
  identical(sm$seeds_fitted, 1:3)
ok["the_choice_is_frozen_in_the_tuning_run"] <- {
  s <- readRDS(file.path(fit$run_dir, "comparison", "selection.rds"))
  identical(s$config_id, fin$selected_config_ids)
}

# ── 3. side by side changes no number ────────────────────────────────────────
#
# WITH THE MEMORY TRACE ON, for this fit only (T7 reads the trace): a traced
# worker must train exactly as an untraced one, and the identity below with
# the two-worker fit, which ran without it, is what says so.
old_opt <- options(dsm.final.trace_mem = TRUE)
fin1 <- suppressMessages(do.call(dsm_final, c(list(fit, config = cid, n_cores = 1L,
                                                   run_id = "seq"), final_args)))
options(old_opt)
ok["one_worker_is_what_one_worker_was_asked"] <- identical(fin1$n_workers, 1L)
ok["two_workers_and_one_give_identical_seeds"] <-
  identical(fin$all_seed_results$seed, fin1$all_seed_results$seed) &&
  identical(fin$all_seed_results$ccc, fin1$all_seed_results$ccc) &&
  identical(fin$all_seed_results$mae, fin1$all_seed_results$mae)
ok["and_an_identical_ensemble"] <- identical(
  safe_read_csv2(file.path(rd, cid, "ensemble_predictions.csv"))$pred,
  safe_read_csv2(file.path(fin1$run_dir, cid, "ensemble_predictions.csv"))$pred)
# Every phase once, every epoch of every seed (3 epochs, none stopped early:
# patience 3 cannot run out before a fourth), and a level at each.
tr1 <- readRDS(file.path(fin1$run_dir, "logs", "worker_01_mem_trace.rds"))
ok["the_trace_has_every_phase_and_every_epoch"] <-
  all(c("start", "store_loaded", "fold_cache", "unit_start",
        "epoch", "unit_trained", "unit_released") %in% tr1$phase) &&
  identical(as.integer(table(tr1$unit_id[tr1$phase == "epoch"])), rep(3L, 3L)) &&
  all(is.finite(tr1$rss_gb)) && all(is.finite(tr1$r_heap_gb))
ok["without_the_option_there_is_no_trace"] <-
  !file.exists(file.path(rd, "logs", "worker_01_mem_trace.rds"))

# ── 4. the declaration ───────────────────────────────────────────────────────
h <- safe_read_csv2(file.path(rd, "selected_hyperparameters.csv"))
ok["every_hyperparameter_of_the_grid_is_declared"] <-
  all(setdiff(names(grid), "config_id") %in% h$parameter)
ok["the_one_the_search_varied_is_marked_tuned"] <-
  isTRUE(h$searched[h$parameter == "embedding_dim"][1]) &&
  identical(as.character(h$values_tried[h$parameter == "embedding_dim"][1]), "8, 16")
ok["the_ones_it_did_not_are_marked_fixed"] <-
  !any(as.logical(h$searched[h$parameter %in% c("conv_channels", "base_lr", "batch_size")]))
# The fixture's configs are single-branch without SE: the gate and the SE
# ratio are not used, and must not be declared as chosen.
ok["what_the_network_does_not_use_is_declared_unused"] <-
  !any(as.logical(h$used[h$parameter %in% c("gate_type", "gate_dropout", "se_reduction")])) &&
  !any(as.logical(h$searched[h$parameter %in% c("gate_type", "gate_dropout", "se_reduction")]))
ok["the_refit_schedule_is_declared_too"] <-
  identical(as.character(h$value[h$parameter == "n_epochs"][1]), "3")
rep_txt <- readLines(fin$report_file, encoding = "UTF-8")
ok["the_report_names_the_config_and_every_parameter"] <-
  any(grepl(cid, rep_txt, fixed = TRUE)) &&
  all(vapply(setdiff(names(grid), "config_id"),
             function(p) any(grepl(paste0("`", p, "`"), rep_txt, fixed = TRUE)), logical(1)))
ok["the_report_says_how_it_was_chosen"] <-
  any(grepl("Rule applied", rep_txt, fixed = TRUE)) &&
  any(grepl("thread(s)", rep_txt, fixed = TRUE))

# ── 5. resume ────────────────────────────────────────────────────────────────
rec_before <- readRDS(.final_unit_record_path(rd, cid, 2L))
fin_r <- suppressMessages(do.call(dsm_final, c(list(fit, config = cid, n_cores = 2L,
                                                    run_id = "par", resume = TRUE),
                                               final_args)))
rec_after <- readRDS(.final_unit_record_path(rd, cid, 2L))
ok["a_resumed_fit_retrains_no_finished_seed"] <-
  identical(rec_before$finished_at, rec_after$finished_at) && identical(fin_r$n_workers, 0L) &&
  identical(fin_r$all_seed_results$ccc, fin$all_seed_results$ccc)

# ── 6. what can only be wrong is refused before anything trains ──────────────
refuse <- function(...) err(do.call(dsm_final, utils::modifyList(
  c(list(fit, n_cores = 1L), final_args), list(...))))
ok["duplicate_seeds_are_refused"]        <- grepl("duplicates", refuse(seeds = c(1L, 1L)))
ok["zero_threads_per_unit_is_refused"]   <- grepl("whole number", refuse(threads_per_unit = 0L))
ok["a_misspelt_training_argument_is_refused"] <-
  grepl("does not take", refuse(training = list(n_epoch = 3L)))
ok["a_config_not_in_the_grid_is_refused"] <- grepl("not in this run", refuse(config = "cfg_999"))
ok["an_existing_run_is_not_overwritten_silently"] <-
  grepl("already exists", refuse(run_id = "par", resume = FALSE))
ok["a_seed_count_means_42_onwards"] <- identical(.final_seeds(3L), 42:44)

# A RESUME IS HELD TO THE SETTINGS ITS RUN STARTED WITH (run_spec.rds). The
# thread count is T1's; the schedule is what every epoch runs; print_every
# only prints, and a resume that changes nothing else trains nothing.
ok["the_run_records_what_its_seeds_depend_on"] <- {
  sp <- readRDS(file.path(rd, "run_spec.rds"))
  identical(sp$threads_per_unit, 1L) && identical(sp$training$n_epochs, 3L) &&
    all(c("grid", "split", "scaling", "transform_probe") %in% names(sp))
}
ok["a_resume_with_another_thread_count_is_refused"] <-
  grepl("threads_per_unit 1, now 2",
        refuse(run_id = "par", n_cores = 2L, threads_per_unit = 2L), fixed = TRUE)
ok["a_resume_with_another_schedule_is_refused"] <-
  grepl("training$n_epochs 3, now 4",
        refuse(run_id = "par", training = list(n_epochs = 4L)), fixed = TRUE)
ok["a_resume_that_only_prints_otherwise_is_not_refused"] <- {
  fp <- suppressMessages(do.call(dsm_final, utils::modifyList(
    c(list(fit, config = cid, n_cores = 1L, run_id = "par"), final_args),
    list(training = list(print_every = 1L)))))
  identical(fp$n_workers, 0L) && identical(fp$all_seed_results$ccc, fin$all_seed_results$ccc)
}

# A REFUSED CALL FREEZES NO CHOICE. freeze_selection() never records another
# choice in a tuning run, so a choice frozen by a call that then stopped would
# outlive the refusal, and the corrected call would be refused after it. On a
# copy of the tuning run with nothing frozen, a resume the run's settings
# refuse must leave nothing frozen.
tcopy_root <- file.path(base, "tuning_copy")
dir.create(tcopy_root)
file.copy(fit$run_dir, tcopy_root, recursive = TRUE)
tcopy <- file.path(tcopy_root, basename(fit$run_dir))
unlink(file.path(tcopy, "comparison", "selection.rds"))
ok["a_refused_call_freezes_no_choice"] <-
  grepl("threads_per_unit 1, now 2", err(do.call(dsm_final, utils::modifyList(
    c(list(tcopy, data = data, config = cid, n_cores = 2L), final_args),
    list(run_id = "par", threads_per_unit = 2L)))), fixed = TRUE) &&
  !file.exists(file.path(tcopy, "comparison", "selection.rds"))

# THE CLAMP IS THE TUNING RUN'S. dsm_train() recorded the one it scored its
# units with -- the default here -- the refit took it without being told, and
# the summary a map reads carries it; another, given in `training`, is refused.
ok["the_tuning_run_keeps_its_clamp"] <-
  identical(readRDS(file.path(fit$run_dir, "clamp.rds")), c(0, Inf)) && identical(fit$clamp, c(0, Inf))
ok["the_refit_clamps_as_the_tuning_did"] <- identical(as.numeric(sm$training$clamp), c(0, Inf))
ok["a_refit_clamp_the_tuning_did_not_use_is_refused"] <-
  grepl("Leave it out", refuse(training = list(n_epochs = 3L, clamp = c(-Inf, Inf))))
ok["a_tuning_run_that_kept_no_clamp_meant_the_default"] <-
  identical(.final_clamp(file.path(out_root, "no_such_run"), NULL), c(0, Inf))

# ── 7. the declaration of a model fitted before dsm_final() existed ──────────
#
# A stage-04 run: the same files, but no record of threads, schedule or
# units, and no report. dsm_report_final() must write the declaration from
# what is on disk -- the same one dsm_final() wrote -- without retraining.
old <- file.path(base, "stage04_like")
dir.create(old)
file.copy(rd, old, recursive = TRUE)
old_rd <- file.path(old, basename(rd))
s04 <- readRDS(file.path(old_rd, "comparison", "final_run_summary.rds"))
s04[c("threads_per_unit", "n_workers", "training", "fitted_by", "validation_frac",
      "n_train", "n_validation", "n_test", "torch_version", "r_version",
      "git_commit", "finished_at")] <- NULL
saveRDS(s04, file.path(old_rd, "comparison", "final_run_summary.rds"))
unlink(file.path(old_rd, c("final_report.md", "selected_hyperparameters.csv", "run_spec.rds")))
unlink(file.path(old_rd, cid, "units"), recursive = TRUE)
rep04 <- suppressMessages(dsm_report_final(old_rd, fit$run_dir, verbose = FALSE))
txt04 <- readLines(rep04$report_file, encoding = "UTF-8")
ok["a_stage_04_run_gets_its_declaration_without_retraining"] <-
  file.exists(file.path(old_rd, "selected_hyperparameters.csv")) &&
  any(grepl("not recorded", txt04, fixed = TRUE)) &&
  all(file.exists(file.path(old_rd, cid, "models", sprintf("seed%04d_best.pt", 1:3))))
ok["it_declares_the_hyperparameters_dsm_final_declared"] <-
  identical(as.list(rep04$hyper), as.list(fin$hyper[fin$hyper$group != "final refit", ]))
ok["and_reports_the_same_test_results"] <-
  isTRUE(all.equal(rep04$config_summary$ccc_mean, fin$config_summary$ccc_mean)) &&
  isTRUE(all.equal(rep04$config_summary$mae_mean, fin$config_summary$mae_mean))
# NEVER FITTED INTO. Named as a run_id, a final model dsm_final() did not
# start -- the deployed one was fitted by stage 04 -- is refused before
# anything is written into it: a fit there would overwrite its seeds.
pt_file <- file.path(old_rd, cid, "models", "seed0001_best.pt")
pt_md5  <- unname(tools::md5sum(pt_file))
ok["a_directory_dsm_final_did_not_start_is_never_fitted_into"] <-
  grepl("did not start", err(do.call(dsm_final, utils::modifyList(
    c(list(fit, n_cores = 1L), final_args), list(output_dir = old, run_id = basename(rd)))))) &&
  identical(unname(tools::md5sum(pt_file)), pt_md5) &&
  !file.exists(file.path(old_rd, "run_spec.rds"))

# ── 8. calibration residuals of a configuration, found by what it is ─────────
#
# The interval's residuals may come from another run (block or kNNDM folds),
# where a config_id is only a label: the configuration must be found by its
# hyperparameters, and one that differs in any of them must get none.
sel_row <- grid[grid$config_id == cid, , drop = FALSE]
by_sig  <- suppressMessages(cv_residuals_for_config(fit$run_dir, sel_row))
same_cols <- c("sample_id", "obs", "pred")
ok["residuals_are_found_by_the_hyperparameters"] <-
  identical(attr(by_sig, "config_id"), cid) &&
  isTRUE(all.equal(as.data.frame(by_sig)[, same_cols],
                   as.data.frame(cv_residuals(fit$run_dir, cid))[, same_cols]))
renamed <- sel_row
renamed$config_id <- "called_something_else"
ok["a_renamed_config_is_the_same_config"] <-
  identical(attr(suppressMessages(cv_residuals_for_config(fit$run_dir, renamed)), "config_id"), cid)
changed <- sel_row
changed$base_lr <- 0.123
ok["a_config_with_other_hyperparameters_gets_none"] <-
  is.null(suppressMessages(cv_residuals_for_config(fit$run_dir, changed, required = FALSE))) &&
  grepl("No configuration", err(cv_residuals_for_config(fit$run_dir, changed)))

unlink(base, recursive = TRUE)

cat(sprintf("  fixture                  : %d points in %d sites | %d configs tuned\n",
            n_pts, n_site, nrow(grid)))
cat(sprintf("  selected                 : %s (%s)\n", cid, fin$selection$rule_applied))
cat(sprintf("  2 workers vs 1           : identical seeds -> %s\n",
            ok[["two_workers_and_one_give_identical_seeds"]]))

.report(ok, "test_final")
