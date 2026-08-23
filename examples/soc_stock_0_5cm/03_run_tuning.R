source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
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

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Training hyperparameters ──────────────────────────────────────────────────
# These are fixed across all configs in the grid.
# Only architecture and optimiser params vary per config (see tune_grid.R).

training_args <- list(
  n_epochs            = 500L,
  patience            = 60L,
  es_min_delta        = 0.0005,
  warmup_start_lr     = 1e-5,
  lr_plateau_factor   = 0.5,
  lr_plateau_patience = 25L,   # raised from 15: the exploratory run cut LR too
                               # early, before models could settle on the plateau
  lr_plateau_min_delta = 0.0005,
  min_lr              = 1e-6,
  gradient_clip       = 1.0,
  print_every         = 10L,
  augment             = TRUE   # D4 rotation/flip augmentation (regulariser)
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

# Channel count from the first available patch array (robust to window choice —
# don't assume a 3x3 array exists).
.first_array_key <- grep("_array$", names(patches$train), value = TRUE)[1]
n_channels <- dim(patches$train[[.first_array_key]])[2]
message("Channels: ", n_channels, " (from ", .first_array_key, ")")
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
# RESOLUTION RESET. The earlier exploratory run (where 7×7 dominated, CCC ~0.61)
# was at 20 km, where 7×7 covers ~140 km of context. THIS run is at 250 m, where
# the same window spans only ~1.75 km — so that finding does NOT transfer. We
# therefore make WINDOW and LEARNING RATE the two primary search axes and let the
# data tell us which spatial scale matters at 250 m, instead of pre-committing:
#   • window_sizes: all six 3 / 9 / 15 options (single + dual) → ~0.75–3.75 km.
#   • base_lr: a spread (1e-4 … 1e-3); don't assume 20 km's 1e-4 still wins.
# Secondary knobs (depth, SE, gate, embedding, dropout, weight_decay) vary lightly
# around a lean baseline. D4 augmentation is on.
#
# `fixed` fixes single values AND restricts multi-value pools (see R/tune_grid.R);
# duplicate draws are dropped automatically. tune_length = 24 gives every window a
# few LR/architecture samples — raise it for denser coverage, lower for a quick
# first pass. make_manual_tune_grid() is the alternative for a full factorial.

tune_grid <- make_tune_grid(
  tune_length = 24L,
  seed        = 42L,
  fixed = list(
    loss_fn       = "smooth_l1",                # robust to outlier SOC values
    batch_size    = 256L,                       # 15×15 patches are larger than 7×7;
                                                # 256 is safe — raise to 512 if GPU allows
    # PRIMARY axis #1 — spatial scale at 250 m (single branch and dual branch)
    window_sizes  = list(c(3L), c(9L), c(15L),
                         c(3L, 9L), c(3L, 15L), c(9L, 15L)),
    # PRIMARY axis #2 — learning rate
    base_lr       = c(1e-4, 3e-4, 1e-3),
    # Secondary knobs — lean baseline
    conv_channels = list(c(64L, 128L),
                         c(64L, 128L, 128L)),
    use_residual  = TRUE,                        # always on for ≥2 blocks
    use_se_block  = c(TRUE, FALSE),
    gate_type     = c("vector_featurewise", "no_gate_concat"),
    embedding_dim = c(256L, 384L),
    # flatten keeps full spatial detail (params grow with window²); gap pools to
    # C and keeps large windows light. Let tuning decide which wins at 250 m —
    # especially relevant for the 15×15 branch (~11 M params under flatten).
    embed_pool    = c("flatten", "gap"),
    dropout       = c(0.1, 0.2),                 # mild–moderate regularisation
    weight_decay  = c(0.0, 1e-4)
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
#   5. Appends to comparison table (ranked by VALIDATION CCC then validation MAE;
#      test metrics are recorded for diagnostics only, never used for selection)
#
# transform = expm1: back-transform from log1p space to native ton/ha.
#   Applied to predictions before computing CCC, MAE, etc.
#   The model trains in log1p space; metrics are always in native units.

# Para RETOMAR um run interrompido (crash, queda de luz, etc.): preencha
# resume_run_id com o run_id exato da pasta em outputs/tuning/ que parou no
# meio (ex: "soc_0_5cm_20260715_093000") e rode o script de novo. run_cnn_tuning
# detecta sozinho quais config_id já têm checkpoint (models/{id}_best.pt) e
# pula direto para os que faltam -- NÃO retreina do zero. Deixe NULL para
# sempre começar um run novo (comportamento padrão, gera timestamp novo).
resume_run_id <- NULL

run_id <- if (is.null(resume_run_id)) {
  paste0("soc_0_5cm_", format(Sys.time(), "%Y%m%d_%H%M%S"))
} else {
  resume_run_id
}

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
      run_id       = run_id,
      resume       = TRUE
    ),
    training_args
  )
)

# ── Results summary ───────────────────────────────────────────────────────────

message("\n── Tuning complete ────────────────────────────────────────────────")
message("Run ID: ", run_id)
message("Results saved to: ", file.path(output_tuning_dir, run_id))

if (nrow(results$comparison) > 0) {
  message("\nTop 5 configs by VALIDATION CCC (the selection metric; ",
          "test_* shown for diagnostics only):")
  print(
    dplyr::select(
      results$comparison,
      rank, config_id, window_sizes, conv_channels,
      embedding_dim, gate_type, use_se_block, dropout, base_lr,
      best_epoch, runtime_min,
      val_ccc, val_mae, val_rmse, val_mqi,
      test_ccc, test_mae
    ),
    n = 5, width = Inf
  )
}

