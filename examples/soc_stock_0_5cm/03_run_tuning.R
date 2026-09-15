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
  "DescTools",
  "terra"       # only to read the raster resolution for the leakage report
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
source(file.path(project_root, "R", "diagnostics.R"))
source(file.path(project_root, "R", "resample.R"))
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

patch_dir   <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
data_dir    <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)

output_tuning_dir <- file.path(project_root, "outputs", "tuning",
                                "soc_stock_modeling", target_label)

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
  tune_length = 3,
  seed        = 666,
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
print_wide(
  dplyr::mutate(
    tune_grid,
    window_sizes  = purrr::map_chr(window_sizes,  paste, collapse = "x"),
    conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")
  )
)

# ── Load only the windows this grid actually needs ────────────────────────────
# The patch store keeps one file per window, so a grid that never uses 15x15
# does not pay ~6 GB of RAM for it. That is why the grid is built first.

windows_needed <- sort(unique(unlist(tune_grid$window_sizes)))
message("\nWindows required by this grid: ",
        paste(windows_needed, collapse = ", "))

store <- load_patch_store(patch_dir, windows_needed)
n_channels <- store$n_channels

# Point values feed the SCALING only -- the patches themselves are already
# extracted. 02 drops points whose window was not fully valid, so the dataset
# written by 01 has more rows than the store: align on sample_id rather than
# trusting row order.
points <- readr::read_csv2(file.path(data_dir, "full_modeling_dataset_raw.csv"),
                           show_col_types = FALSE)
type_table <- readr::read_csv2(file.path(metadata_dir, "predictor_type_table.csv"),
                               show_col_types = FALSE)
points <- align_points_to_meta(points, store$meta)

# -- Raster resolution, and the units the coordinates are in ------------------
#
# Read from the RASTER ITSELF, never written by hand: block_size and buffer
# are given in the SAME units as x/y, and here those are DEGREES (lon/lat --
# WOSIS is global), not metres. A buffer written as "3750" with metres in
# mind would be 3750 degrees and the plan would abort; a block_size wrong in
# the other direction would produce a split that only LOOKS spatial, aborting
# nothing.
r_ref     <- terra::rast(readr::read_csv2(
  file.path(metadata_dir, "raster_table_used.csv"),
  show_col_types = FALSE)$raster_file[1])
cell_size <- terra::res(r_ref)[1]
rm(r_ref)

# -- THE STORE LOCK ------------------------------------------------------------
#
# Everything from here on is expensive, and every expensive thing assumes the
# store on disk was built under the configuration this script is running. That
# assumption has been wrong before, and it never announced itself: a store
# extracted with one predictor set, read by a script expecting another, trains
# and converges and produces a map -- of the wrong variable.
#
# So it is checked, here, in milliseconds, against what 02 recorded. It costs
# one file read and refuses in seconds what would otherwise waste hours.
target_config <- readr::read_csv2(file.path(metadata_dir, "target_config.csv"),
                                  show_col_types = FALSE)

check_store_spec(
  store      = store,
  predictors = type_table$predictor,
  windows    = windows_needed,
  target_col = target_config$target_col[1],
  cell_size  = cell_size
)
message("Store spec: OK (predictors, windows, target and resolution match).")

message(sprintf("\nRaster resolution: %.8f per pixel (in x/y units)",
                cell_size))
message(sprintf("Buffer floor (window %d x resolution): %.6f",
                max(windows_needed), max(windows_needed) * cell_size))

# ── RESAMPLING PLAN ───────────────────────────────────────────────────────────
#
# ONE cut point, ONE criterion. The plan carves the test set AND the folds, and
# it carves both the same way. Change this line and nothing below changes.
#
#   holdout(meta, validation_frac = , test_frac = )      no folds
#   random_folds(meta, k = , test_frac = )               ignores geography
#   spatial_folds(meta, k = , test_frac = , block_size = , buffer = )
#   region_folds(meta, group = , k = , test_frac = )
#
# WHY THE TEST SET IS CARVED HERE AND NOT IN STAGE 01. It used to be a
# stratified random draw written into the point table, which then travelled
# inside the 16 GB patch store -- so changing it cost a re-extraction. And it
# was random even when the validation was spatial, which put two numbers in
# one table that cannot be compared: measured on the full data, the random
# test came out 0.042 CCC EASIER than the spatial validation, while carrying
# the name that suggests it is the stricter of the two.
#
# THE TEST SET IS FROZEN ON FIRST USE. A test set redrawn on every run is not
# a test set. The first run writes data_split.csv; every run after reads it.
# Delete that file to draw a new one -- deliberately, not by accident.
#
# THE BUFFER. Two patches of width w at resolution res share a pixel when their
# centres are within w-1 cells in BOTH axes -- a SQUARE condition, which is why
# the buffer metric is chebyshev and `max(window) * cell_size` is exact. Under
# a circular buffer the diagonal escapes: on the previous run that left 0.5% to
# 1% of validation points still sharing patch pixels with training, while the
# report showed a clean 0% for "same raster cell".

test_frac <- 0.15

split_file <- file.path(metadata_dir, "data_split.csv")
frozen_test <- if (file.exists(split_file)) {
  ds <- readr::read_csv2(split_file, show_col_types = FALSE)
  message("Frozen test set read from data_split.csv: ",
          sum(ds$role == "test"), " points")
  ds$sample_id[ds$role == "test"]
} else {
  NULL
}

k_folds <- 3L

# BLOCK_SIZE IS MEASURED HERE, NOT WRITTEN HERE.
#
# It used to be the literal 2, with a table of measurements from the FULL point
# set beside it saying 1,279 blocks and a largest block of 4.4%. That number
# was right for that point set and wrong for this one, in a way nothing would
# have reported: block-subsampling keeps WHOLE blocks, so a 10% draw has a
# tenth of the blocks at the SAME size -- and the block that held 4.4% of the
# full data holds 34% of the subsample. With k = 3 that single block would
# have decided a fold, and the fold would have been scored on whatever one
# landscape it happens to be.
#
# Bigger blocks separate better, so suggest_block_size() takes the LARGEST size
# whose worst block still fits inside the balance constraint. The table it
# measured is printed, because the choice should be readable, not trusted.
block_choice <- suggest_block_size(store$meta, k = k_folds, max_share = 0.10)
print_block_choice(block_choice)

plan <- spatial_folds(
  store$meta,
  k          = k_folds,
  test_frac  = test_frac,
  test_ids   = frozen_test,
  block_size = as.numeric(block_choice),
  buffer     = max(windows_needed) * cell_size    # 15 px, exact under chebyshev
)

if (!file.exists(split_file)) {
  test_pos <- plan$folds[[1]]$test
  safe_write_csv2(
    tibble::tibble(
      sample_id = store$meta$sample_id,
      role      = ifelse(seq_len(nrow(store$meta)) %in% test_pos,
                         "test", "modelling")
    ),
    split_file
  )
  message("Test set frozen to: ", split_file)
}

message("\n-- Resampling plan --")
print(plan)

# How much this plan still leaks, in the SAME metric the 99 reports for the
# whole dataset -- so a number here is comparable with a number there.
leak <- fold_leakage_report(plan, store$meta, cell_size = cell_size,
                            windows = windows_needed)
# IDENTICAL PATCHES are a defect under any plan: two points in the same pixel
# feed the network the same input, so one can be scored on exactly what the
# other trained on.
#
# SHARED PIXELS between neighbours are NOT a defect. Under a random plan they
# are the condition being measured -- "how well does this predict at new points
# drawn from the same spatial distribution" -- so reporting them as leakage
# would misstate the question the plan was chosen to answer. They are printed
# only when the plan claims to be spatial, where they describe how well the
# separation held.
message("\n-- Identical patches, train vs validation, per fold --")
print_wide(dplyr::filter(leak, matters), n = Inf)

if (grepl("spatial|region", plan$method)) {
  message("\n-- Shared pixels between neighbours (context, not a defect) --")
  print_wide(dplyr::filter(leak, !matters, window == max(windows_needed)),
             n = Inf)
}

# The scaling is fitted on the training rows of EACH fold, inside
# run_cnn_resample() -- which is the whole reason the patches are stored RAW. A
# fold whose scaling came from another fold's training set has already seen
# data it should not have.
#
# Note: it is fitted over the points that actually TRAIN (after the window QC),
# whereas stage 01's predictor_scaling.csv used every training point, including
# the ~0.8% later dropped by the full-window rule. The difference is small, but
# this is the honest version -- and the only one that generalises to a fold.

message("Channels: ", n_channels, " | Points: ", nrow(store$meta))

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

# To RESUME an interrupted run (crash, power cut): fill
# resume_run_id with the exact run_id of the directory under outputs/tuning/
# that stopped half way (e.g. "soc_0_5cm_20260715_093000") and run the script
# again. The runner works out by itself which units already have a checkpoint
# (models/{id}_best.pt) and skips straight to the ones that are missing -- it
# does NOT retrain from scratch. Leave NULL to always start a new run (the
# default; generates a fresh timestamp).
resume_run_id <- NULL

run_id <- if (is.null(resume_run_id)) {
  paste0("soc_0_5cm_", format(Sys.time(), "%Y%m%d_%H%M%S"))
} else {
  resume_run_id
}

# HOW MANY SEEDS PER CONFIG
#
# Three, not one. With one seed each, "config A beat config B" is a claim
# with no error bar: if retraining the SAME config under a different seed
# moves CCC more than the distance between A and B, the ranking is a ranking
# of luck. Three repetitions are the minimum that estimates that floor, and
# the run prints it at the end.
#
# Cost: 3x the grid's time. Raising it later is RESUMABLE -- the repetitions
# already on disk are recognised and only the new ones train, so it is fine
# to start at 1, see the grid stand up, and go to 3 without losing anything.
n_seeds <- 3L

results <- do.call(
  run_cnn_resample,
  c(
    list(
      tune_grid  = tune_grid,
      store      = store,
      points     = points,
      type_table = type_table,
      plan       = plan,
      transform  = expm1,
      output_dir = output_tuning_dir,
      device     = device,
      run_id     = run_id,
      n_seeds    = n_seeds,
      resume     = TRUE
    ),
    training_args
  )
)

# ── Results summary ───────────────────────────────────────────────────────────

message("\n── Tuning complete ────────────────────────────────────────────────")
message("Run ID: ", run_id)
message("Results saved to: ", file.path(output_tuning_dir, run_id))

# The decision is read from the PER-CONFIG table (mean +/- sd over the
# repetitions), not from the per-unit one: a lucky seed of a mediocre config
# outranks the steady mean of a good one whenever rows are what get ranked.
if (nrow(results$by_config) > 0) {
  message("\nTop 5 configs by mean VALIDATION CCC (the selection metric; ",
          "test_* is diagnostic only):")
  print_wide(
    dplyr::select(
      results$by_config,
      rank, config_id, n_units, n_folds, n_seeds,
      dplyr::starts_with("val_ccc"), dplyr::starts_with("val_mae"),
      dplyr::starts_with("test_ccc"), n_failed
    ),
    n = 5
  )

  message("\nNoise floor -- only the seed changes:")
  print_noise_floor(seed_noise_floor(results$comparison))
}

if (nrow(results$comparison) > 0) {
  message("\nAudit trail (every unit, as it was measured): ",
          file.path(results$run_dir, "comparison", "comparison_ranked.csv"))
}

