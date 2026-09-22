
# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R and in R/load_all.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/load_all.R.
project_root <- (function() {
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
  for (d in cand) for (up in c(".", "..", "../..", "../../..")) {
    r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
    if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c(
  "torch",
  "coro",
  "dplyr",
  "tidyr",
  "readr",
  "tibble",
  "purrr",
  "DescTools"
)

install_load_pkg(pkg)

rm(list = setdiff(ls(), "project_root"))  # keep the root found above
gc()

options(width = 200)

setwd(project_root)

# One source() instead of ten, in a dependency order that is not guessable.
source(file.path(project_root, "R", "load_all.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

# Which tuning run to use. "latest" takes the most recent one automatically
# (sorted by the timestamp in the directory name). Or name a run_id.
tuning_run_id <- "latest"

# Which configs to train as the final model. Each is trained with EVERY seed.
#
# NULL = use rank 1 of the per-config ranking automatically -- the right choice
# for a first run at a new resolution, when nobody knows yet which wins.
#
# To compare the top N paired by seed: run 03, open
#   outputs/.../tuning/<run>/comparison/comparison_by_config.csv
# and list the winning config_ids here, e.g. c("cfg_007", "cfg_013").
#
# WARNING: NEVER reuse an ID from an older run. The same "cfg_004" is a
# DIFFERENT architecture in every tuning run -- the numbering is per run, so an
# ID copied across runs names something else entirely and nothing will say so.
#
# The seeds are the same for every config, which is what makes the comparison
# paired by seed rather than a comparison of who drew the luckier start.
selected_config_ids <- NULL

# How to choose when selected_config_ids is NULL.
#   "one_se" -- the simplest config within one standard error of the best
#   "rank1"  -- the best mean, whatever its spread
# See the long note at the selection itself for why one_se is the default.
# Nominal coverages for the conformal intervals. 0.1 -> 90%, 0.05 -> 95%.
# Each needs at least ceiling(1/alpha) - 1 calibration points to be certifiable
# (9 and 19 respectively).
conformal_alpha  <- c(0.1, 0.05)

selection_rule   <- "one_se"
selection_metric <- "val_ccc"

# Seeds. Each one is an independent training run from scratch, and the spread
# between them estimates how STABLE the training is -- not how good the model
# is. A publishable result should have a low spread (ideally under ~5% of the
# mean CCC); a large one means the number being reported is partly the draw.
#
# This is the same quantity seed_noise_floor() measures during tuning, at a
# larger sample: ten seeds here against three there.
seeds <- c(7, 28, 42L, 94, 123L, 333, 456L, 666, 789L, 2025L)

# ── Overrides, for driving this stage from outside ────────────────────────────
#
# THE rm(list = ls()) AT THE TOP IS WHY THESE ARE ENVIRONMENT VARIABLES.
#
# Every other example script can be driven by setting a variable before the
# source(). This one cannot: line 20 erases the workspace, deliberately, so that
# a stale object from a previous run can never leak into a final model.
# Sys.setenv() survives that erasure; a workspace object does not.
#
# Nothing here changes what the stage does by default -- unset, every one of
# them leaves the values above exactly as written. They exist so that a caller
# can run the REAL script rather than a copy of it: _b2_two_config_check.R
# points this stage at a throwaway copy of a tuning run, and a check that runs
# a duplicate of the code is a check of the duplicate.
#
#   soc_final_tuning_run_id   which tuning run to read ("latest", or a run id)
#   soc_final_config_ids      comma-separated config ids, e.g. "cfg_003,cfg_002"
#   soc_final_seeds           comma-separated integers, e.g. "7,28,42"
#
# Naming config ids here is the same act as naming them at the top of the file,
# so it takes the same path: selection_rule_applied becomes "manual" and the
# frozen record says so.
#
# env_chr()/env_csv() live in R/utils.R -- one reader, shared with 03 and 05a.

tuning_run_id       <- env_chr("soc_final_tuning_run_id", tuning_run_id)
selected_config_ids <- env_csv("soc_final_config_ids", selected_config_ids)
seeds               <- env_csv("soc_final_seeds", seeds, as_int = TRUE)

if (nzchar(Sys.getenv("soc_final_config_ids")) ||
    nzchar(Sys.getenv("soc_final_seeds")) ||
    nzchar(Sys.getenv("soc_final_tuning_run_id"))) {
  message("\n-- Driven by environment overrides --")
  message("  tuning run : ", tuning_run_id)
  message("  configs    : ",
          if (is.null(selected_config_ids)) "(by the selection rule)"
          else paste(selected_config_ids, collapse = ", "))
  message("  seeds      : ", paste(seeds, collapse = ", "),
          "  (", length(seeds), ")")
  message("  Unset these variables to return to the values written in this file.")
}

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Final training hyperparameters ────────────────────────────────────────────
# More epochs and more patience than during tuning: the configs are known now,
# so there is no reason to hurry convergence. Tuning trades a little accuracy
# per config for covering the grid; this stage does not.

training_args <- list(
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

# ── Paths ─────────────────────────────────────────────────────────────────────

patch_dir         <- file.path(project_root, "outputs", "patches",
                                "soc_stock_modeling", target_label)
data_dir          <- file.path(project_root, "data", "processed",
                                "soc_stock_modeling", target_label)
metadata_dir      <- file.path(project_root, "outputs", "metadata",
                                "soc_stock_modeling", target_label)

output_tuning_dir <- file.path(project_root, "outputs", "tuning",
                                "soc_stock_modeling", target_label)

# Resolve "latest" to the most recent tuning run
if (identical(tuning_run_id, "latest")) {
  tuning_run_id <- latest_run_dir(
    output_tuning_dir, prefix = "soc_",
    require_file = file.path("comparison", "comparison_ranked.csv"),
    label = "tuning_run_id")
}

tuning_dir   <- file.path(output_tuning_dir, tuning_run_id)
ranking_file <- file.path(tuning_dir, "comparison", "comparison_ranked.csv")
# THE PER-CONFIG TABLE IS THE ONE THAT DECIDES (mean +/- sd over the
# repetitions). comparison_ranked.csv has one row per UNIT (config x fold x
# seed), and reading its rank would let a lucky seed of a mediocre config
# outrank the steady mean of a good one -- which is precisely the mistake
# repetitions exist to prevent. Older runs (one seed, one fold) have no
# per-config file, and there the two tables coincide anyway.
byconfig_file <- file.path(tuning_dir, "comparison", "comparison_by_config.csv")
tune_grid_file <- file.path(tuning_dir, "tune_grid.rds")

run_id     <- paste0("final_", format(Sys.time(), "%Y%m%d_%H%M%S"))
output_dir <- file.path(project_root, "outputs", "final_model",
                        "soc_stock_modeling", target_label, run_id)

create_output_dirs(c(output_dir, file.path(output_dir, "comparison")))

# -- Validations -------------------------------------------------------------

if (!dir.exists(patch_dir))      stop("Patches not found: ", patch_dir)
if (!file.exists(ranking_file))  stop("Ranking not found: ",  ranking_file)
if (!file.exists(tune_grid_file)) stop("tune_grid.rds not found: ", tune_grid_file)

# -- Select the configs ------------------------------------------------------

ranking        <- readr::read_csv2(ranking_file, show_col_types = FALSE)
tune_grid_full <- readRDS(tune_grid_file)

by_config <- if (file.exists(byconfig_file)) {
  readr::read_csv2(byconfig_file, show_col_types = FALSE)
} else {
  NULL
}

# ── Which config goes to the final model ──────────────────────────────────────
#
# THE BEST MEAN IS OFTEN THE LUCKIEST DRAW.
#
# The tuning run reports a mean over repetitions and the standard error of that
# mean. When several configs sit inside one standard error of the winner, the
# ranking has not separated them -- it has ordered noise. Taking rank 1 anyway
# is a choice to carry the largest model in the tie, because complexity and
# luck correlate: more parameters means more ways to fit the validation fold.
#
# one_se() takes the SIMPLEST config whose mean is within one standard error of
# the best. It is the rule caret has had since the beginning (`oneSE`), it is
# Breiman's original 1-SE rule, and it exists here because this project's own
# noise floor showed the ranking does not separate: 0.038 CCC between seeds
# against gaps of 0.008 between families.
#
# When nothing ties, it returns the winner -- so it costs nothing when the
# ranking IS decisive, which is the property that makes it safe as a default.
#
# selection_rule = "rank1" restores the old behaviour for someone who wants it.
# WHAT WAS ASKED FOR, AND WHAT ACTUALLY HAPPENED, ARE TWO DIFFERENT FACTS.
#
# selection_rule holds the REQUEST and is a constant set at the top of this
# file. selection_rule_applied holds what the code below actually did, and it is
# that one which gets frozen into selection.rds.
#
# They diverge in two real cases, and the record used to report the request in
# both: a hand-written selected_config_ids skips the rule altogether, and a
# requested one_se that cannot be applied falls back to rank 1 (the branch below
# even prints a note saying so). selection.rds is the file whose whole job is to
# say how the final model was chosen -- the published selection optimism of
# +0.0000 means nothing if that field describes a rule nobody ran.
selection_rule_applied <- NULL

if (is.null(selected_config_ids)) {
  selected_config_ids <- if (!is.null(by_config) &&
                             identical(selection_rule, "one_se") &&
                             paste0(selection_metric, "_se") %in% names(by_config) &&
                             "n_params" %in% names(by_config)) {
    pick <- one_se(by_config, metric = selection_metric, complexity = "n_params")
    print_one_se(pick, metric = selection_metric)
    selection_rule_applied <- "one_se"
    pick$config_id
  } else if (!is.null(by_config)) {
    if (identical(selection_rule, "one_se")) {
      # SAY WHY THE RULE DID NOT APPLY. Falling back silently to rank 1 while
      # the header says the run uses one_se is worse than not having the rule:
      # the report would describe a selection that never happened.
      missing <- setdiff(c(paste0(selection_metric, "_se"), "n_params"),
                         names(by_config))
      message("\n  NOTE: one_se() needs ", paste(missing, collapse = " and "),
              ", which this tuning run did not record. Falling back to rank 1.",
              "\n  (n_params is recorded by tuning runs from 2026-09 onward.)")
    }
    # NOT "one_se". The rule was asked for and could not be applied, and the
    # record has to carry that distinction or a reader six months from now sees
    # a one_se selection that never ran.
    selection_rule_applied <- if (identical(selection_rule, "one_se")) {
      "rank1_after_one_se_unavailable"
    } else {
      "rank1"
    }
    dplyr::filter(by_config, rank == 1L)$config_id
  } else {
    # The per-UNIT table, which is one row per (config, fold, seed). Ranking on
    # it lets a lucky seed outrank a steady mean, so the record names it
    # separately from a rank 1 taken on the per-config means.
    selection_rule_applied <- "rank1_from_units"
    dplyr::filter(ranking, rank == 1L)$config_id
  }
} else {
  selection_rule_applied <- "manual"
  message("\nConfig(s) named by hand at the top of this script: ",
          paste(selected_config_ids, collapse = ", "),
          "\n  No selection rule was applied, and the frozen record will say so.")
}

# A record with no rule in it is not a record. This cannot fire as the code
# stands -- every branch above sets it -- which is exactly when a guard is cheap
# and the next branch someone adds is when it earns its place.
stopifnot(is.character(selection_rule_applied), length(selection_rule_applied) == 1L)

# If the winner's margin is smaller than the seed noise, SAY SO HERE, where
# the choice is being made -- rather than letting the number travel onward as
# if it were a result. Selecting anyway is legitimate; not knowing is not.
if (!is.null(by_config) && nrow(by_config) > 1L &&
    "val_ccc_sd" %in% names(by_config)) {
  top2 <- dplyr::arrange(by_config, rank)[1:2, ]
  gap  <- top2$val_ccc_mean[1] - top2$val_ccc_mean[2]
  noise <- stats::median(by_config$val_ccc_sd, na.rm = TRUE)
  if (is.finite(gap) && is.finite(noise) && gap < noise) {
    message("\n  WARNING: the 1st config's margin over the 2nd (", round(gap, 4),
            ") is SMALLER than the typical sd between seeds (",
            round(noise, 4), ").")
    message("  The two are indistinguishable at this run's number of ",
            "repetitions.")
    message("  Run more seeds, or select the simplest within one SE ",
            "(see one_se()).")
  }
}

# THE CHOICE IS LOCKED HERE, WHERE IT IS MADE.
#
# Written to the TUNING run, not to this one, because that is the run whose
# grid could later be scored on the test set. score_test_grid() refuses to run
# until this file exists, so the order "chose, then looked" is enforced rather
# than remembered -- and the record carries the timestamp and the commit, so a
# reader who was not here can check it.
#
# Re-running stage 04 with the same choice is harmless; re-running it with a
# DIFFERENT choice after the grid has been scored is refused, because that is
# the loop the lock exists to prevent.
freeze_selection(tuning_dir, selected_config_ids,
                 rule   = selection_rule_applied,
                 metric = selection_metric,
                 note   = paste("stage 04 on", format(Sys.time())))

missing_cfgs <- setdiff(selected_config_ids, tune_grid_full$config_id)
if (length(missing_cfgs) > 0) {
  stop("config_ids not found in the tune_grid: ",
       paste(missing_cfgs, collapse = ", "))
}

selected_cfgs <- dplyr::filter(tune_grid_full, config_id %in% selected_config_ids)

message("\n── Configs selected for the final model ──────────────────────")
for (cid in selected_config_ids) {
  # slice(1): with repetitions the ranking holds several rows per config, and
  # their ARCHITECTURE fields are identical (it is the same config) -- so the
  # first will do. The METRICS come from the per-config table instead, where
  # they arrive with their spread attached.
  r <- dplyr::slice(dplyr::filter(ranking, config_id == cid), 1)
  b <- if (!is.null(by_config)) {
    dplyr::filter(by_config, config_id == cid)
  } else {
    NULL
  }
  metric_txt <- if (!is.null(b) && nrow(b) == 1L) {
    sprintf("val_CCC %.4f +/- %.4f (n=%d)", b$val_ccc_mean, b$val_ccc_sd,
            b$n_units)
  } else {
    sprintf("val_CCC %.4f", r$val_ccc)
  }
  message("  ", cid,
          " | window ", r$window_sizes,
          " | ", r$conv_channels,
          " | embed ", r$embedding_dim,
          " | ", r$gate_type,
          " | ", metric_txt)
}

# -- Load the patches and build the fold cache -------------------------------
# Loaded only now, because only now is it known which windows the selected
# configs use -- the patch store keeps one file per window, so loading
# everything would pay RAM for windows no config is going to touch.

windows_needed <- sort(unique(unlist(selected_cfgs$window_sizes)))
message("\nWindows needed: ", paste(windows_needed, collapse = ", "))

# Same call as stages 03 and 03b, and the same lock: the final model must be
# fitted on the store the tuning was done on, or the config that won means
# nothing here.
data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = windows_needed
)

store      <- data$store
points     <- data$points
type_table <- data$type_table
n_channels <- store$n_channels

# WHY THIS STAGE KEEPS ITS OWN SEED LOOP.
#
# dsm_train() would express it -- one refit fold, the selected configs, ten
# seeds -- but it writes a run directory laid out for TUNING, and stage 05
# reads a layout laid out for a FITTED MODEL: the weights and the scaling
# together under <run>/<config_id>/. Moving both at once, before either has
# run, is the move that costs this project its restarts. The loop below is what
# produces that layout, and it calls the same train_one_cnn() dsm_train() would.

# The final fit reuses the TUNING plan: same test set, validation carved by the
# same criterion. Selecting spatially and then stopping on a random validation
# set would change the question between the two stages.
tuning_plan <- readRDS(file.path(tuning_dir, "fold_plan.rds"))
refit       <- refit_split(tuning_plan, store$meta, validation_frac = 0.15)
index       <- refit$folds[[1]]

message("\n-- Final-fit split (from the tuning plan) --")
print(refit)

fold         <- build_fold_cache(store, points, type_table, index, windows_needed)
points_valid <- fold_points_valid(store, index)
tensor_cache <- fold$cache

# Same lesson as stage 03: every seed shares ONE cache. The scaling belongs to
# the fold, not to the seed, so rebuilding it per seed would be pure waste.
store$windows <- NULL
invisible(gc())

message("Channels: ", n_channels,
        " | Train: ", length(index$train),
        " | Val: ",   length(index$validation),
        " | Test: ",  length(index$test))

# -- Train one config with every seed ----------------------------------------

train_config_all_seeds <- function(cfg, config_id) {

  cfg_out_dir <- file.path(output_dir, config_id)
  create_output_dirs(file.path(cfg_out_dir,
                               c("models", "history", "predictions",
                                 "metrics", "gates")))

  # THE SCALING TRAVELS WITH THE WEIGHTS.
  #
  # It is estimated from the training rows of THIS fit, so it is part of the
  # fitted model, not a property of the dataset. Stage 05 reads it from here.
  #
  # It used to read a global table written by stage 01 instead -- a second copy
  # of the same fact, free to drift. After patches became raw and scaling
  # became per-fold, the two stopped agreeing: the map was being built with one
  # set of constants while the network had been trained with another, silently.
  safe_write_csv2(fold$scaling, file.path(cfg_out_dir, "predictor_scaling.csv"))

  seed_rows  <- vector("list", length(seeds))
  seed_preds <- vector("list", length(seeds))

  for (s_idx in seq_along(seeds)) {
    seed_val <- seeds[s_idx]
    message("\n── [", config_id, "] seed ", s_idx, "/", length(seeds),
            ": ", seed_val, " ──")

    set.seed(seed_val)
    torch::torch_manual_seed(seed_val)

    loaders <- .make_loaders_from_cache(tensor_cache, cfg)

    result <- tryCatch(
      do.call(
        train_one_cnn,
        c(list(cfg = cfg, n_channels = n_channels, loaders = loaders,
               points_valid = points_valid, transform = expm1, device = device,
               model_name = paste0(config_id, "_seed", seed_val)),
          training_args)
      ),
      error = function(e) {
        message("  ERROR [", config_id, "] seed ", seed_val, ": ", e$message)
        NULL
      }
    )
    if (is.null(result)) next

    sl <- sprintf("seed%04d", seed_val)
    safe_torch_save(result$best_state,  file.path(cfg_out_dir, "models",      paste0(sl, "_best.pt")))
    safe_write_csv2(result$history,     file.path(cfg_out_dir, "history",     paste0(sl, "_history.csv")))
    safe_write_csv2(result$pred_all,    file.path(cfg_out_dir, "predictions", paste0(sl, "_pred_all.csv")))
    safe_write_csv2(result$perf_all,    file.path(cfg_out_dir, "metrics",     paste0(sl, "_perf.csv")))
    safe_write_csv2(result$perf_quantile, file.path(cfg_out_dir, "metrics",   paste0(sl, "_perf_quantile.csv")))
    if (!is.null(result$gate)) {
      safe_write_csv2(result$gate$summary,    file.path(cfg_out_dir, "gates", paste0(sl, "_gate_summary.csv")))
      safe_write_csv2(result$gate$by_profile, file.path(cfg_out_dir, "gates", paste0(sl, "_gate_profiles.csv")))
    }

    # Kept for the conformal calibration below. Only the held-out roles: the
    # training rows would make the residuals smaller by exactly the amount the
    # model overfits, and the resulting intervals would be narrow and wrong in
    # the direction nobody checks.
    seed_preds[[s_idx]] <- result$pred_all %>%
      dplyr::filter(.data$dataset_role %in% c("validation", "test")) %>%
      dplyr::select(sample_id, dataset_role, obs, pred) %>%
      dplyr::mutate(seed = seed_val)

    seed_rows[[s_idx]] <- dplyr::filter(result$perf_all, dataset_role == "test") %>%
      dplyr::mutate(config_id = config_id, seed = seed_val,
                    best_epoch = result$best_epoch,
                    runtime_min = result$runtime_min)

    message("  [", config_id, "] seed ", seed_val,
            " | best_ep ", result$best_epoch,
            " | CCC ",  round(seed_rows[[s_idx]]$ccc,  4),
            " | MAE ",  round(seed_rows[[s_idx]]$mae,  3),
            " | RMSE ", round(seed_rows[[s_idx]]$rmse, 3))
    gc()
  }

  # ══════════════════════════════════════════════════════════════════════════
  # CALIBRATED UNCERTAINTY
  #
  # The ensemble gives a median and a spread. The spread is NOT an interval --
  # it measures how much the answer moves when the initialisation moves, which
  # is a property of the optimiser and not of the soil. Used as a map of
  # uncertainty it understates by about an order of magnitude here.
  #
  # Conformal fixes that with one pass: take the absolute residuals on data the
  # model never trained on, take their (1 - alpha) quantile, and every
  # prediction gets +/- that. The guarantee needs no distributional assumption.
  #
  # WHICH SET CALIBRATES AND WHICH SET CHECKS.
  #
  #   validation -> calibration. Held out of training, separated spatially with
  #                 a buffer, and already computed.
  #   test       -> the check. Neither trained nor calibrated on, so the PICP
  #                 measured there is an honest answer to "does the interval
  #                 cover what it promises", not arithmetic.
  #
  # The spread finally earns a job as the DIFFICULTY score of the normalised
  # variant: intervals then widen where the seeds disagree, and the guarantee
  # survives because the score is calibrated rather than trusted.
  # ══════════════════════════════════════════════════════════════════════════
  preds <- dplyr::bind_rows(purrr::compact(seed_preds))
  if (nrow(preds) > 0L) {
    ens <- preds %>%
      dplyr::group_by(.data$sample_id, .data$dataset_role) %>%
      dplyr::summarise(obs = dplyr::first(.data$obs),
                       pred = stats::median(.data$pred),
                       spread = stats::sd(.data$pred),
                       n_seeds = dplyr::n(), .groups = "drop")

    # WHICH RESIDUALS CALIBRATE, AND WHY NOT THESE ONES.
    #
    # The obvious choice is this run's own validation rows -- held out of
    # training, already computed. It was the first choice here, and it gave 83.6%
    # coverage against a nominal 90%.
    #
    # The refit validation is ONE fold of k = 7: 449 points from one spatial
    # region, and the easiest of the three sets by a wide margin (MAE 14.6
    # against 17.0 across the cross-validation and 18.2 on test). Calibrating on
    # one block and measuring coverage on another is exchangeability failure by
    # construction.
    #
    # The tuning run's cross-validated residuals cover every block instead --
    # each point predicted once, as validation, somewhere. Recalibrating that way
    # moved coverage to 87.8%. They are used when available, and this run's own
    # validation is the fallback for a tuning run that wrote no predictions.
    cv_cal <- cv_residuals(tuning_dir, config_id)
    cal_source <- "cross-validated residuals (all folds)"
    cal_rows <- if (!is.null(cv_cal) && nrow(cv_cal) >= nrow(ens) / 4) {
      dplyr::mutate(cv_cal, spread = NA_real_)
    } else {
      cal_source <- "this run's validation split (one fold)"
      dplyr::filter(ens, .data$dataset_role == "validation")
    }
    chk_rows <- dplyr::filter(ens, .data$dataset_role == "test")
    message("\nConformal calibration set: ", cal_source, " -- ",
            nrow(cal_rows), " point(s)")

    if (nrow(cal_rows) >= 9L && nrow(chk_rows) > 0L) {
      for (a in conformal_alpha) {
        cal <- conformal_calibrate(cal_rows$obs, cal_rows$pred, alpha = a)
        iv  <- conformal_interval(cal, chk_rows$pred, lower_limit = 0)
        message("\n-- [", config_id, "] conformal, ",
                round(100 * (1 - a)), "% --")
        print(cal)
        print(picp_report(chk_rows$obs, iv$lower, iv$upper, alpha = a))

        # THE OTHER CALIBRATION, ALONGSIDE, when both exist.
        #
        # Printing only the better one would hide the finding. The gap between
        # these two lines IS the result: how much a calibration set drawn from
        # one spatial block understates the interval a map needs.
        if (!is.null(cv_cal) && nrow(cv_cal) > 0L) {
          own <- dplyr::filter(ens, .data$dataset_role == "validation")
          if (nrow(own) >= 9L) {
            cal_o <- conformal_calibrate(own$obs, own$pred, alpha = a)
            iv_o  <- conformal_interval(cal_o, chk_rows$pred, lower_limit = 0)
            message("   for comparison, calibrated on this run's validation ",
                    "only (", nrow(own), " points, one fold):")
            print(picp_report(chk_rows$obs, iv_o$lower, iv_o$upper, alpha = a))
          }
        }

        # ── WHY THE NORMALISED INTERVAL IS NOT PRODUCED HERE ────────────────
        #
        # Locally adaptive intervals need a per-point difficulty score, and the
        # obvious one is the ensemble spread. It is available on the prediction
        # side (10 seeds of the final model) and NOT on the calibration side,
        # because the calibration set is now the cross-validated residuals, and
        # those came from 3 seeds of models trained on ~53% of the points.
        #
        # Calibrating the ratio residual/spread on one kind of spread and
        # applying it to another is not a weaker guarantee, it is no guarantee:
        # the CV models disagree with each other more than the final ensemble
        # does, so the ratio is systematically wrong and the interval would be
        # confidently the wrong width. Producing that number and labelling it
        # "90%" is exactly what this file exists to avoid.
        #
        # The principled difficulty score is the dissimilarity index from
        # R/aoa.R: it is computed the same way for a calibration point and for
        # every prediction pixel, from the same scaling, and it measures the
        # thing that actually makes a point hard -- distance from what the model
        # was trained on. That is the next step here, and it needs stage 07's
        # machinery over the calibration points.
        cal_n <- NULL
        has_spread <- "spread" %in% names(cal_rows) &&
          all(is.finite(cal_rows$spread)) && all(cal_rows$spread > 0)
        if (has_spread) {
          cal_n <- conformal_calibrate(cal_rows$obs, cal_rows$pred, alpha = a,
                                       difficulty = cal_rows$spread)
          iv_n  <- conformal_interval(cal_n, chk_rows$pred,
                                      difficulty = chk_rows$spread,
                                      lower_limit = 0)
          message("   normalised by the seed spread:")
          print(picp_report(chk_rows$obs, iv_n$lower, iv_n$upper, alpha = a))
        } else {
          message("   normalised interval: NOT produced. The calibration set ",
                  "carries no comparable\n     difficulty score -- see the note in this script. Constant width only.")
        }

        # Saved so stage 05 can put bounds on the map without recomputing --
        # and so the map and this report can never describe different intervals.
        safe_save_rds(
          list(alpha = a, constant = cal, normalised = cal_n,
               n_calibration = nrow(cal_rows), config_id = config_id,
               calibrated_on = "validation", checked_on = "test"),
          file.path(cfg_out_dir,
                    sprintf("conformal_%02d.rds", round(100 * (1 - a)))),
          compress = FALSE)
      }
    } else {
      message("\n-- [", config_id, "] conformal skipped: ", nrow(cal_rows),
              " calibration point(s), ", nrow(chk_rows), " to check on.")
    }

    # -- NEITHER OF THE TWO BLOCKS BELOW IS A CONFORMAL ARTEFACT ---------------
    #
    # They used to sit inside the "if (enough calibration rows AND a test set)"
    # above, which is the wrong gate for both, and this was caught before the
    # stage was re-run rather than after:
    #
    #   ensemble_predictions.csv is the ensemble table. It is what the later
    #   stages and every plot read, and a run that produced predictions must
    #   write it whether or not an interval could be calibrated.
    #
    #   the smearing factor is calibrated on the TUNING run's out-of-fold
    #   residuals -- a different run, a different set of points -- so this run's
    #   calibration count says nothing at all about whether it can be computed.
    #
    # The gate mattered because evaluate_test = FALSE is now a supported mode:
    # chk_rows is then empty, the condition is FALSE, and stage 05 would have
    # found no smearing.rds, written the median map alone, and printed a polite
    # note about it. A mean surface that goes missing while announcing itself as
    # a deliberate choice is worse than one that errors.
    safe_write_csv2(ens, file.path(cfg_out_dir, "ensemble_predictions.csv"))

    # ══════════════════════════════════════════════════════════════════════
    # THE BACK-TRANSFORM: a median surface and a mean surface
    #
    # expm1() of a conditional median is a median. The target is right-skewed,
    # so this model under-predicted the frozen test set by 24.4% -- more than
    # half its MAE -- and every metric except the new signed bias was blind to
    # it. Anyone summing the map for a total carbon stock got a quarter less
    # than the data say.
    #
    # Duan's smearing estimator corrects it with one scalar, calibrated on the
    # same out-of-fold residuals the conformal interval uses. NOTHING IS
    # REPLACED: the median surface is the right answer to "what is the typical
    # stock here" and the mean surface is the only one that may be summed.
    # Both are written and both are labelled.
    # ══════════════════════════════════════════════════════════════════════
    sm <- smearing_from_run(tuning_dir, config_id)
    if (!is.null(sm)) {
      print(sm)
      safe_save_rds(sm, file.path(cfg_out_dir, "smearing.rds"),
                    compress = FALSE)

      # MEASURED ON THE TEST SET, which the factor was not calibrated on.
      # The trade-off is reported rather than buried: correcting toward the
      # mean must improve RMSE and worsen MAE, and a run where it improved
      # both would mean something other than a median-to-mean move happened.
      tst <- dplyr::filter(preds, .data$dataset_role == "test") %>%
        dplyr::group_by(.data$sample_id) %>%
        dplyr::summarise(obs = dplyr::first(.data$obs),
                         pred = stats::median(.data$pred), .groups = "drop")
      if (nrow(tst) > 0L) {
        # expm1 is monotone, so the median commutes with it: log1p of the
        # ensemble median IS the ensemble median in log space. No re-reading.
        corrected <- smear(log1p(tst$pred), sm, lower_limit = 0)
        m_naive <- calc_metrics(tst$obs, tst$pred)
        m_smear <- calc_metrics(tst$obs, corrected)
        message("\n-- [", config_id, "] back-transform, on the test set --")
        print(dplyr::bind_rows(
          dplyr::mutate(m_naive, surface = "median (expm1)", .before = 1),
          dplyr::mutate(m_smear, surface = "mean (smeared)", .before = 1)))
        message(sprintf(
          "   total stock: observed %.0f | median surface %.0f (%+.1f%%) | mean surface %.0f (%+.1f%%)",
          sum(tst$obs), sum(tst$pred),
          100 * (sum(tst$pred) / sum(tst$obs) - 1),
          sum(corrected), 100 * (sum(corrected) / sum(tst$obs) - 1)))
        message("   The median surface answers \"typical stock here\"; only the")
        message("   mean surface may be summed for a total.")
      }
    } else {
      message("\nSmearing: the tuning run wrote no transform-space residuals, ",
              "so no mean surface.")
    }
  }

  dplyr::bind_rows(purrr::compact(seed_rows))
}

# -- Run every config --------------------------------------------------------

all_seed_results <- tibble::tibble()
for (i in seq_len(nrow(selected_cfgs))) {
  cfg <- selected_cfgs[i, ]
  res <- train_config_all_seeds(cfg, cfg$config_id)
  all_seed_results <- dplyr::bind_rows(all_seed_results, res)
}

if (nrow(all_seed_results) == 0) stop("No seed finished successfully.")

# EVERY REQUESTED SEED, OR STOP. A seed that failed was caught by the tryCatch
# above, printed, and dropped from all_seed_results -- and the summary then
# listed the REQUESTED seeds while the ensemble on disk held fewer. Stage 05
# would have built the map from the survivors and nothing downstream could
# tell. The failure is named here, where the run is still in front of someone.
# Per config, not pooled: with two configs a seed lost by one and kept by the
# other would survive a pooled unique().
seeds_by_cfg <- split(all_seed_results$seed, all_seed_results$config_id)
for (cid in selected_cfgs$config_id) {
  seeds_lost <- setdiff(seeds, seeds_by_cfg[[cid]])
  if (length(seeds_lost) > 0L) {
    stop("Config ", cid, ": seed(s) ", paste(seeds_lost, collapse = ", "),
         " did not finish (see the error printed above). The final model must ",
         "carry every seed it claims; fix the cause and re-run.", call. = FALSE)
  }
}
seeds_fitted <- sort(unique(all_seed_results$seed))

# -- Per config: mean +/- sd between seeds -----------------------------------

config_summary <- all_seed_results %>%
  dplyr::group_by(config_id) %>%
  dplyr::summarise(
    n_seeds   = dplyr::n(),
    ccc_mean  = mean(ccc),  ccc_sd  = sd(ccc),
    r2_mean   = mean(r2),   r2_sd   = sd(r2),
    mae_mean  = mean(mae),  mae_sd  = sd(mae),
    nse_mean  = mean(nse),  nse_sd  = sd(nse),
    rmse_mean = mean(rmse), rmse_sd = sd(rmse),
    rpd_mean  = mean(rpd),  rpd_sd  = sd(rpd),
    mqi_mean  = mean(mqi),  mqi_sd  = sd(mqi),
    best_epoch_mean   = mean(best_epoch),
    runtime_min_total = sum(runtime_min),
    .groups = "drop"
  ) %>%
  dplyr::arrange(dplyr::desc(ccc_mean))

safe_write_csv2(all_seed_results, file.path(output_dir, "comparison", "all_seed_results_test.csv"))
safe_write_csv2(config_summary,   file.path(output_dir, "comparison", "config_summary_test.csv"))

safe_save_rds(
  # selected_config_ids IS THE ORDER OF THE CHOICE. selected_cfgs comes from
  # dplyr::filter(tune_grid_full, ...), and filter keeps the GRID's order -- so
  # its first row is not necessarily the config that was chosen first. Stage 05
  # resolves config_id = "auto" from this, and with two configs the difference
  # is which model goes on the map.
  list(selected_cfgs = selected_cfgs, selected_config_ids = selected_config_ids,
       selection_rule = selection_rule_applied, seeds = seeds,
       seeds_fitted = seeds_fitted,
       all_seed_results = all_seed_results, config_summary = config_summary,
       run_id = run_id, tuning_run_id = tuning_run_id),
  file.path(output_dir, "comparison", "final_run_summary.rds"),
  compress = FALSE
)

# -- Paired comparison, seed by seed -----------------------------------------
# The same seed means the same initial RNG state, so the difference WITHIN a
# seed isolates the effect of the architecture from the effect of the draw.
# Reported as the mean difference, and whether it is consistent across seeds --
# a mean difference smaller than the spread between seeds is a tie.

# EXACTLY TWO, AND THE OTHER COUNTS SAY SO OUT LOUD.
#
# The block below names c1 and c2 and subtracts one from the other, so it is a
# two-config comparison by construction and cannot simply be relaxed. What it
# used to do with three configs was nothing at all, with no else branch and no
# message: paired_by_seed.csv would just not exist, and whoever went looking for
# it would find out then. This project has been bitten four times by a check
# that skipped itself quietly; that is why the counts are reported here.
if (length(selected_config_ids) > 2) {
  message("\n-- Paired comparison SKIPPED: ", length(selected_config_ids),
          " configs selected --")
  message("   paired_by_seed.csv compares exactly two configs, seed by seed. ",
          "With more than\n   two, the pair to compare is a choice this script ",
          "will not make for you.")
  message("   Everything else in this run is unaffected -- per-config results ",
          "are in\n   comparison/config_summary_test.csv.")
}

if (length(selected_config_ids) == 2) {
  paired <- all_seed_results %>%
    dplyr::select(config_id, seed, ccc, mae, rmse, mqi) %>%
    tidyr::pivot_wider(names_from = config_id, values_from = c(ccc, mae, rmse, mqi))

  c1 <- selected_config_ids[1]; c2 <- selected_config_ids[2]
  paired <- paired %>%
    dplyr::mutate(
      d_ccc  = .data[[paste0("ccc_",  c1)]] - .data[[paste0("ccc_",  c2)]],
      d_mae  = .data[[paste0("mae_",  c1)]] - .data[[paste0("mae_",  c2)]],
      d_rmse = .data[[paste0("rmse_", c1)]] - .data[[paste0("rmse_", c2)]],
      d_mqi  = .data[[paste0("mqi_",  c1)]] - .data[[paste0("mqi_",  c2)]]
    )

  safe_write_csv2(paired, file.path(output_dir, "comparison", "paired_by_seed.csv"))

  message("\n-- Paired difference (", c1, " - ", c2, "), mean over seeds --")
  message(sprintf("  ΔCCC : %+.4f", mean(paired$d_ccc,  na.rm = TRUE)))
  message(sprintf("  ΔMAE : %+.3f", mean(paired$d_mae,  na.rm = TRUE)))
  message(sprintf("  ΔRMSE: %+.3f", mean(paired$d_rmse, na.rm = TRUE)))
  message(sprintf("  ΔMQI : %+.4f", mean(paired$d_mqi,  na.rm = TRUE)))
  message("  (dCCC > 0 favours ", c1,
          "; |dCCC| smaller than the spread between seeds is a tie)")
}

# -- Final report ------------------------------------------------------------

message("\n-- Per config (mean +/- sd over ", length(seeds), " seeds) --")
for (i in seq_len(nrow(config_summary))) {
  s <- config_summary[i, ]
  message("\n  ", s$config_id, " (n=", s$n_seeds, "):")
  message(sprintf("    CCC  : %.4f ± %.4f", s$ccc_mean,  s$ccc_sd))
  message(sprintf("    MAE  : %.3f ± %.3f", s$mae_mean,  s$mae_sd))
  message(sprintf("    RMSE : %.3f ± %.3f", s$rmse_mean, s$rmse_sd))
  message(sprintf("    R²   : %.4f ± %.4f", s$r2_mean,   s$r2_sd))
  message(sprintf("    NSE  : %.4f ± %.4f", s$nse_mean,  s$nse_sd))
  message(sprintf("    RPD  : %.3f ± %.3f", s$rpd_mean,  s$rpd_sd))
  message(sprintf("    MQI  : %.4f ± %.4f", s$mqi_mean,  s$mqi_sd))
}

message("\nResults saved in: ", output_dir)
