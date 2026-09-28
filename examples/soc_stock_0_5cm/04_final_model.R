# ══════════════════════════════════════════════════════════════════════════════
# Stage 04 -- the final model: the configuration the tuning run supports,
# refitted under ten seeds, with what the map needs from them
#
# ONE CALL NOW. This script was 850 lines that did, for the SOC data, what
# dsm_final() does for any: choose the configuration (one_se by default),
# refit it on everything but the test set under every seed, and turn the seeds
# into the ensemble median, a calibrated interval and the smearing factor of
# the mean surface. dsm_final() writes the same files in the same places, so
# stage 05 reads a run of this script exactly as it read one of the old.
# What is left here is what belongs to this dataset: its paths, its seeds, its
# schedule.
#
# WHAT CHANGED WITH IT, and why it matters to anyone comparing runs:
#
#   * The seeds train SIDE BY SIDE, each in its own R process with
#     threads_per_unit threads (5), instead of one after another in this
#     session with 30. T2 measured side by side 1.52x faster, and a seed's
#     numbers depend on its seed and its thread count and on nothing else
#     (T1, T2). So a new run does not reproduce the deployed model's seeds
#     (final_20260918_150311, fitted by the old script with 30 threads): it
#     differs from it the way another draw of seeds would.
#   * Stage 05 maps the NEWEST finished final run. Running this makes the new
#     fit the one the next map is built from.
#   * A run that stops (a crash, a power cut) is resumed with resume_run_id
#     below, and only with the settings it started with.
#   * The interval normalised by the seed spread is gone. The old script
#     never produced it once its calibration came from the cross-validated
#     residuals -- their spread is not the final ensemble's -- and the
#     locally adaptive interval is dsm_predict()'s (level and DI).
#
# The reasoning of each step -- the selection rule, which residuals calibrate
# the interval, the smearing -- is in R/final.R, beside the code that does it.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/04_final_model.R")
# ══════════════════════════════════════════════════════════════════════════════

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/cnn_architecture.R.
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
    if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c(
  "torch",
  "coro",
  "dplyr",
  "readr",
  "tibble",
  "purrr",
  "callr",      # dsm_final() trains every seed in its own R process
  "ps"          # and reads the RAM it may use
)

install_load_pkg(pkg)

rm(list = setdiff(ls(), "project_root"))  # keep the root found above
gc()

options(width = 200)

setwd(project_root)

# The framework is a package: pkgload::load_all() loads it from this source
# tree as it stands. library(soilcnn) loads an installed copy. The workers
# load whichever this session loaded.
pkgload::load_all(project_root)

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

# Which tuning run to use. "latest" takes the newest FINISHED one (by the time
# its ranking was written) -- what a fit right after stage 03 wants -- or name
# a run_id. What it resolves to is printed below, beside the tuning run the
# final model stage 05 maps now was selected from.
tuning_run_id <- "latest"

# Which configs to train as the final model. Each is trained with EVERY seed.
#
# NULL = apply the selection rule to the per-config ranking -- the right choice
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
#
# Naming ids is recorded as what it is: the frozen selection says "manual",
# not the rule below.
selected_config_ids <- NULL

# How to choose when selected_config_ids is NULL.
#
# THE BEST MEAN IS OFTEN THE LUCKIEST DRAW. When several configs sit inside one
# standard error of the winner, the ranking has not separated them; taking rank
# 1 anyway carries the largest model in the tie, because complexity and luck
# correlate. one_se takes the SIMPLEST config within one standard error of the
# best -- caret's oneSE, Breiman's 1-SE rule -- and it is the default because
# this project's own noise floor showed the ranking does not separate: 0.038
# CCC between seeds against gaps of 0.008 between families. When nothing ties
# it returns the winner, so it costs nothing when the ranking IS decisive.
#
#   "one_se" -- the simplest config within one standard error of the best
#   "rank1"  -- the best mean, whatever its spread
selection_rule   <- "one_se"
selection_metric <- "val_ccc"

# Nominal coverages for the conformal intervals. 0.1 -> 90%, 0.05 -> 95%.
# Each needs at least ceiling(1/alpha) - 1 calibration points to be certifiable
# (9 and 19 respectively).
conformal_alpha <- c(0.1, 0.05)

# Seeds. Each one is an independent training run from scratch, and the spread
# between them estimates how STABLE the training is -- not how good the model
# is. A publishable result should have a low spread (ideally under ~5% of the
# mean CCC); a large one means the number being reported is partly the draw.
#
# This is the same quantity seed_noise_floor() measures during tuning, at a
# larger sample: ten seeds here against three there.
seeds <- c(7, 28, 42L, 94, 123L, 333, 456L, 666, 789L, 2025L)

# THREADS PER SEED ARE PART OF THE RESULT, like the seed. T1 measured that the
# thread count changes a unit's numbers (0.067 val_ccc between counts in 12
# epochs) and T2 that nothing else does: not the neighbours, not how many
# seeds fit on the machine. 5 was measured best here (T2: three units of 5
# side by side, 1.52x one unit of 15). The run records it; to reproduce a run,
# keep it.
threads_per_unit <- 5L

# Cores for the whole fit. NULL is the physical cores minus one; how many seeds
# train at once is n_cores %/% threads_per_unit, capped by the RAM. That
# changes only the time, never the numbers.
n_cores <- NULL

# The share of the non-test points the refit stops on, carved by the tuning
# plan's own criterion (refit_split()): same test set, validation cut the way
# the selection's folds were.
validation_frac <- 0.15

# To RESUME an interrupted fit: name its run directory here (e.g.
# "final_20260928_101500", under outputs/final_model/soc_stock_modeling/
# soc_stock_0_5cm/) and run the script again. Every seed with its checkpoint
# and its record is kept; only the rest train. dsm_final() refuses a resume
# that would compute the rest another way -- other threads per seed, another
# schedule, another tuning run -- and refuses a directory it did not start,
# so a model fitted by the old stage 04 is never trained into. NULL starts a
# new run.
resume_run_id <- NULL

# ── Overrides, for driving this stage from outside ────────────────────────────
#
# THE rm(list = ls()) AT THE TOP IS WHY THESE ARE ENVIRONMENT VARIABLES.
#
# Every other example script can be driven by setting a variable before the
# source(). This one cannot: the rm() at the top erases the workspace,
# deliberately, so that a stale object from a previous run can never leak into
# a final model. Sys.setenv() survives that erasure; a workspace object does
# not.
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
#   soc_final_run_id          the run to resume, as resume_run_id above
#
# env_chr()/env_csv() live in R/utils.R -- one reader, shared with 03 and 05a.

tuning_run_id       <- env_chr("soc_final_tuning_run_id", tuning_run_id)
selected_config_ids <- env_csv("soc_final_config_ids", selected_config_ids)
seeds               <- env_csv("soc_final_seeds", seeds, as_int = TRUE)
resume_run_id       <- env_chr("soc_final_run_id", resume_run_id)

if (nzchar(Sys.getenv("soc_final_config_ids")) ||
    nzchar(Sys.getenv("soc_final_seeds")) ||
    nzchar(Sys.getenv("soc_final_tuning_run_id")) ||
    nzchar(Sys.getenv("soc_final_run_id"))) {
  message("\n-- Driven by environment overrides --")
  message("  tuning run : ", tuning_run_id)
  message("  configs    : ",
          if (is.null(selected_config_ids)) "(by the selection rule)"
          else paste(selected_config_ids, collapse = ", "))
  message("  seeds      : ", paste(seeds, collapse = ", "),
          "  (", length(seeds), ")")
  message("  run        : ", if (is.null(resume_run_id)) "(a new one)" else resume_run_id)
  message("  Unset these variables to return to the values written in this file.")
}

# ── Final training schedule ───────────────────────────────────────────────────
# More epochs and more patience than during tuning: the configs are known now,
# so there is no reason to hurry convergence. Tuning trades a little accuracy
# per config for covering the grid; this stage does not. These are also
# dsm_final()'s defaults; they are written out because they are part of what
# this stage fits. The clamp is not here: the refit takes the tuning run's.

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
# Where stage 05 looks for final runs. dsm_final()'s own default would be
# beside the tuning runs, so it is given.
final_model_dir   <- file.path(project_root, "outputs", "final_model",
                                "soc_stock_modeling", target_label)

# Resolve "latest" to the newest finished tuning run
if (identical(tuning_run_id, "latest")) {
  tuning_run_id <- latest_run_dir(
    output_tuning_dir, prefix = "soc_",
    require_file = file.path("comparison", "comparison_ranked.csv"),
    label = "tuning_run_id")
}
tuning_dir <- file.path(output_tuning_dir, tuning_run_id)

# THE FINAL MODEL STAGE 05 MAPS NOW, beside the run this fit comes from. The
# fit takes its place, and "latest" need not be the run it was selected from:
# on 2026-09-28 it resolved to soc_0_5cm_design_spatial, while the deployed
# model came from soc_0_5cm_20260916_232318. Said here, at the start of an
# hours-long fit, rather than found out from the next map.
mapped_run <- latest_run_dir(final_model_dir, prefix = "final_",
                             require_file = file.path("comparison", "final_run_summary.rds"),
                             label = "final model stage 05 maps now", on_none = "null")
if (!is.null(mapped_run)) {
  mapped_from <- readRDS(file.path(final_model_dir, mapped_run, "comparison",
                                   "final_run_summary.rds"))$tuning_run_id
  message(if (identical(mapped_from, tuning_run_id)) {
    paste0("  It was selected from this same tuning run. This fit refits it under ",
           "new seeds and threads, and takes its place in stage 05.")
  } else {
    paste0("  It was selected from ", mapped_from, "; this fit comes from ",
           tuning_run_id, ", and takes its place in stage 05.")
  })
}

# ── The data ──────────────────────────────────────────────────────────────────
#
# Same call as stages 03 and 03b, and the same lock: the final model must be
# fitted on the store the tuning was done on, or the config that won means
# nothing here.
#
# Every window the store holds (3, 9 and 15 here; ~0.85 GB). This session
# reads only the store's table from it -- the refit split, the scaling -- and
# each worker loads the windows its configuration needs. Loading fewer here
# would mean knowing the configuration before dsm_final() chooses it.
data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  target_col   = readr::read_csv2(file.path(metadata_dir, "target_config.csv"),
                                  show_col_types = FALSE)$target_col[1]
)

# ── The final model ───────────────────────────────────────────────────────────
#
# The back-transform is the STORE'S (transform = NULL): dsm_load() read
# "log1p" from its manifest, and every metric is computed after its inverse,
# expm1 -- in native t/ha. The old script typed expm1 by hand.
final <- dsm_final(
  tuning           = tuning_dir,
  data             = data,
  config           = if (is.null(selected_config_ids)) "auto" else selected_config_ids,
  rule             = selection_rule,
  metric           = selection_metric,
  seeds            = seeds,
  validation_frac  = validation_frac,
  training         = training_args,
  n_cores          = n_cores,
  threads_per_unit = threads_per_unit,
  conformal_alpha  = conformal_alpha,
  output_dir       = final_model_dir,
  run_id           = resume_run_id
)

# ── Per config: mean +/- sd between seeds, on the test set ────────────────────

message("\n-- Per config (mean +/- sd over ", length(final$seeds), " seeds, test set) --")
print_wide(final$config_summary, n = Inf)

message("\nResults saved in: ", final$run_dir)
message("The declaration of the model (how it was chosen, every hyperparameter): ",
        final$report_file)
