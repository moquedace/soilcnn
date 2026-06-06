source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "coro",
  "dplyr",
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
source(file.path(project_root, "R", "metrics.R"))
source(file.path(project_root, "R", "cnn_architecture.R"))
source(file.path(project_root, "R", "tune_grid.R"))
source(file.path(project_root, "R", "train_cnn.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

set.seed(42)
torch::torch_manual_seed(42)

# ── Torch device ──────────────────────────────────────────────────────────────

device <- setup_torch_device(n_threads = 16, use_cuda = TRUE)

# ── Training hyperparameters ──────────────────────────────────────────────────
# These are fixed across all configs in the grid.
# Only architecture and optimiser params vary per config (see tune_grid.R).

training_args <- list(
  n_epochs            = 500L,
  patience            = 60L,
  es_min_delta        = 0.0005,
  warmup_start_lr     = 1e-5,
  lr_plateau_factor   = 0.5,
  lr_plateau_patience = 15L,
  lr_plateau_min_delta = 0.0005,
  min_lr              = 1e-6,
  gradient_clip       = 1.0,
  print_every         = 10L
)

# ── Paths ─────────────────────────────────────────────────────────────────────

patch_file  <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label,
                          "patches_all_splits.rds")

output_tuning_dir <- file.path(project_root, "outputs", "tuning",
                                "soc_stock_modeling", target_label)

# ── Load patches ──────────────────────────────────────────────────────────────

if (!file.exists(patch_file)) {
  stop("Patch file not found. Run 02_extract_patches.R first.\n", patch_file)
}

message("Loading patches...")
patches <- readRDS(patch_file)

n_channels <- dim(patches$train$x_3x3_array)[2]
message("Channels: ", n_channels)
message("Train profiles:      ", nrow(patches$train$meta))
message("Validation profiles: ", nrow(patches$validation$meta))
message("Test profiles:       ", nrow(patches$test$meta))

# points_valid: metadata passed to predict_loader() and gate analysis.
# Each tibble must have: profile_id, sample_id, target_native, target_transform.
points_valid <- list(
  train      = patches$train$meta,
  validation = patches$validation$meta,
  test       = patches$test$meta
)

# ── Tuning grid ───────────────────────────────────────────────────────────────
# See R/tune_grid.R and docs/tuning_guide.md for full parameter descriptions.
#
# make_tune_grid(): random sample from the full parameter space (like caret's
#   tuneLength). Good for exploration.
#
# make_manual_tune_grid(): explicit grid over selected parameters. Good for
#   follow-up experiments after identifying promising regions.
#
# fixed = list(...): parameters to hold constant across all configs.
#   Useful when you already know good values for some parameters.

tune_grid <- make_tune_grid(
  tune_length = 10L,
  seed        = 42L,
  fixed = list(
    loss_fn    = "smooth_l1",   # Huber loss — robust to outlier SOC values
    batch_size = 512L           # fixed by GPU memory; change if OOM errors occur
  )
)

# Preview the grid before running (check it makes sense)
message("\n── Tune grid (", nrow(tune_grid), " configs) ──────────────────────")
print(
  dplyr::mutate(
    tune_grid,
    window_sizes  = purrr::map_chr(window_sizes,  paste, collapse = "x"),
    conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")
  ),
  width = Inf
)

# ── Run tuning ────────────────────────────────────────────────────────────────
# For each config in tune_grid:
#   1. Builds a dual_branch_cnn with that config's architecture
#   2. Trains with early stopping on validation SmoothL1 loss
#   3. Evaluates on train / validation / test (all metrics)
#   4. Saves weights, history, predictions, metrics, gate analysis
#   5. Appends to comparison table (ranked by test CCC then MAE)
#
# transform = expm1: back-transform from log1p space to native ton/ha.
#   Applied to predictions before computing CCC, MAE, etc.
#   The model trains in log1p space; metrics are always in native units.

run_id <- paste0("soc_0_5cm_", format(Sys.time(), "%Y%m%d_%H%M%S"))

results <- do.call(
  run_cnn_tuning,
  c(
    list(
      tune_grid    = tune_grid,
      n_channels   = n_channels,
      patches      = patches,
      points_valid = points_valid,
      transform    = expm1,
      output_dir   = output_tuning_dir,
      device       = device,
      run_id       = run_id
    ),
    training_args
  )
)

# ── Results summary ───────────────────────────────────────────────────────────

message("\n── Tuning complete ────────────────────────────────────────────────")
message("Run ID: ", run_id)
message("Results saved to: ", file.path(output_tuning_dir, run_id))

if (nrow(results$comparison) > 0) {
  message("\nTop 5 configs by test CCC:")
  print(
    dplyr::select(
      results$comparison,
      rank, config_id, window_sizes, conv_channels,
      embedding_dim, gate_type, use_residual, use_se_block,
      base_lr, best_epoch, runtime_min,
      test_ccc, test_mae, test_r2, test_nse, test_rmse
    ),
    n = 5, width = Inf
  )
}
