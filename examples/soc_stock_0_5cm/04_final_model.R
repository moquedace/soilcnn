project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

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

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "patches.R"))
source(file.path(project_root, "R", "preprocess.R"))
source(file.path(project_root, "R", "dataset.R"))
source(file.path(project_root, "R", "resample.R"))
source(file.path(project_root, "R", "metrics.R"))
source(file.path(project_root, "R", "cnn_architecture.R"))
source(file.path(project_root, "R", "tune_grid.R"))
source(file.path(project_root, "R", "train_cnn.R"))

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

# Seeds. Each one is an independent training run from scratch, and the spread
# between them estimates how STABLE the training is -- not how good the model
# is. A publishable result should have a low spread (ideally under ~5% of the
# mean CCC); a large one means the number being reported is partly the draw.
#
# This is the same quantity seed_noise_floor() measures during tuning, at a
# larger sample: ten seeds here against three there.
seeds <- c(7, 28, 42L, 94, 123L, 333, 456L, 666, 789L, 2025L)

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
  run_dirs <- list.dirs(output_tuning_dir, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^soc_", run_dirs)]
  if (length(run_dirs) == 0) stop("No tuning run found in: ", output_tuning_dir)
  tuning_run_id <- sort(run_dirs, decreasing = TRUE)[1]
  message("tuning_run_id resolved to: ", tuning_run_id)
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

if (is.null(selected_config_ids)) {
  selected_config_ids <- if (!is.null(by_config)) {
    dplyr::filter(by_config, rank == 1L)$config_id
  } else {
    dplyr::filter(ranking, rank == 1L)$config_id
  }
}

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

store      <- load_patch_store(patch_dir, windows_needed)
n_channels <- store$n_channels

points <- readr::read_csv2(file.path(data_dir, "full_modeling_dataset_raw.csv"),
                           show_col_types = FALSE)
type_table <- readr::read_csv2(file.path(metadata_dir, "predictor_type_table.csv"),
                               show_col_types = FALSE)
points <- align_points_to_meta(points, store$meta)

# The final fit reuses the TUNING plan: same test set, validation carved by the
# same criterion. Selecting spatially and then stopping on a random validation
# set would change the question between the two stages.
tuning_plan <- readRDS(file.path(tuning_dir, "fold_plan.rds"))
refit       <- refit_split(tuning_plan, store$meta, validation_frac = 0.15)
index       <- refit$folds[[1]]

message("
-- Final-fit split (from the tuning plan) --")
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

  seed_rows <- vector("list", length(seeds))

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
  list(selected_cfgs = selected_cfgs, seeds = seeds,
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

message("\nResultados salvos em: ", output_dir)
