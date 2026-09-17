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

# One source() instead of ten, in an order that is not guessable. See
# R/load_all.R -- when this becomes a package, that file disappears and
# library() takes its place.
source(file.path(project_root, "R", "load_all.R"))

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
  # THREE, ON PURPOSE, UNTIL THE PIPELINE CLOSES END TO END.
  #
  # Three configs x 3 folds x 3 seeds = 27 units, about 50 minutes. That is not
  # enough to say anything about the CNN as a family -- the best of three draws
  # in a space of a couple of thousand combinations supports no claim, including
  # the negative one 03b reported -- and it is not meant to.
  #
  # What it is for is the OTHER kind of failure. Stages 04, 05 and 07 have never
  # run against this API, and 04 now calls one_se(), freeze_selection() and the
  # conformal calibration for the first time; 05 writes interval bands for the
  # first time; 07 has never executed at all. A defect in any of them is found
  # in 50 minutes at this size and after a lost night at tune_length = 24.
  #
  # Raise this to 24 once 03 -> 03b -> 04 -> 05 -> 07 has run clean once. The
  # answer about the family comes from that run, not from this one.
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

# ONE CALL FOR THE WHOLE PREAMBLE.
#
# dsm_load() opens the store, reads the points and the predictor types, aligns
# the points to the store, reads the raster resolution from the raster itself,
# and REFUSES if the store was built under a different predictor set, window
# set, target or resolution.
#
# Those were six separate steps here, and three of this project's lost runs
# came from exactly that stretch: a scaling read from the wrong file, a
# data_dir one directory off, a resolution written by hand. They are now one
# call that cannot be half-done.
data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = windows_needed,
  target_col   = readr::read_csv2(file.path(metadata_dir, "target_config.csv"),
                                  show_col_types = FALSE)$target_col[1]
)

# Kept as plain names because the rest of the script reads better for it, and
# because the leakage report and the snapshot below are script-level work that
# the framework does not own.
store      <- data$store
points     <- data$points
type_table <- data$type_table
cell_size  <- data$cell_size
n_channels <- store$n_channels

message(sprintf("Buffer floor (window %d x resolution): %.6f",
                max(windows_needed), max(windows_needed) * cell_size))

# ── RESAMPLING PLAN ───────────────────────────────────────────────────────────
#
# ONE cut point, ONE criterion. The plan carves the test set AND the folds, and
# it carves both the same way. Change this line and nothing below changes.
#
#   holdout_cv(validation_frac = , test_frac = )   no folds
#   random_cv(k = , test_frac = )                  ignores geography
#   spatial_cv(k = , block_size = , buffer = )     blocks of ground, buffered
#   region_cv(group = , k = , test_frac = )        leave-one-region-out
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

# THE SPEC SAYS WHAT KIND OF SPLIT; THE PLAN IS WHAT THAT BECOMES ON THESE
# POINTS. Keeping the two apart is what lets "auto" mean "measure it when you
# see the data" instead of "guess now".
#
# Swap this one line and nothing below changes:
#   random_cv(k = 10)                  ignores geography, on purpose
#   holdout_cv(validation_frac = 0.2)  a single split
#   region_cv(group = points$biome)    leave-one-region-out
#
# block_size = "auto" takes the LARGEST block whose worst case still fits the
# balance constraint, measured on THESE points, and prints the table it
# measured. It used to be the literal 2, with measurements beside it from the
# FULL point set -- 1,279 blocks, largest holding 4.4%. On a 10% draw the same
# 2 degrees gives a largest block holding 34%, because block-subsampling keeps
# WHOLE blocks: a tenth of the data has a tenth of the blocks at the same
# width. With k = 3 that one block would have decided a fold.
#
# buffer = "auto" is max(window) x cell_size -- the exact SQUARE separation
# distance. Half of it, or a circular radius, leaves the diagonal sharing
# pixels while the leakage report shows a clean zero.
cv <- spatial_cv(
  k          = k_folds,
  block_size = "auto",
  buffer     = "auto",
  test_frac  = test_frac,
  max_share  = 0.10
)

plan <- resolve_resampling(cv, data, test_ids = frozen_test,
                           windows = windows_needed)

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

# dsm_train() dispatches on the model's declared input: "patches" goes to
# run_cnn_resample(), "table" to run_table_resample(). The engine is the same
# one this script called by hand -- what changes is that the plumbing is in one
# tested place instead of in every script that wants to fit something.
#
# `plan` rather than `cv`: the plan was resolved above so the leakage report
# could be read BEFORE committing CPU to it. Passing the spec directly works
# too, and resolves it here.
results <- do.call(
  dsm_train,
  c(
    list(
      data       = data,
      model      = "cnn",
      resampling = plan,
      tune_grid  = tune_grid,
      transform  = expm1,
      output_dir = output_tuning_dir,
      device     = device,
      run_id     = run_id,
      n_seeds    = n_seeds,
      resume     = TRUE,
      verbose    = FALSE      # the summary below is this script's own
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
  # test_* is NOT shown, and the columns are NA anyway (evaluate_test = FALSE).
  #
  # It used to be printed "for diagnosis". There is no such thing: a test score
  # on screen next to the selection metric IS selection on the test set, done
  # by the person reading rather than by an argmax, and it inflates the final
  # number by the same amount. The test is scored once, in stage 04, on the
  # config chosen without it.
  message("\nTop 5 configs by mean VALIDATION CCC (the selection metric):")
  print_wide(
    dplyr::select(
      results$by_config,
      rank, config_id, n_units, n_folds, n_seeds,
      dplyr::starts_with("val_ccc"), dplyr::starts_with("val_mae"),
      dplyr::any_of("n_params"), n_failed
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

