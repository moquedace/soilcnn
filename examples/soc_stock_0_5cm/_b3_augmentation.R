# ══════════════════════════════════════════════════════════════════════════════
# B3 -- D4 augmentation ON vs OFF, on real data, with everything else identical
#
# WHAT IS UNDER TEST.
#
# `augment` is a training-time axis: it changes what the network SEES, epoch by
# epoch, rather than what the network IS. Every other axis this project has
# varied on real data (window, learning rate, depth, gate, pooling) is a
# property of the architecture and can be studied with a cheap model; this one
# cannot. A random forest has no batches to rotate, so tier A could not answer
# it and this script costs CNN time.
#
# WHY IT IS WORTH THE CPU.
#
# The project already believes augmentation helps. docs/design_decisions.md
# section 8 says so, and the evidence it cites is
#
#     "Round-1 to round-2 improvement in CCC (0.569 -> 0.585-0.605)"
#
# which is a comparison between two ROUNDS, not between two arms. Round 2 also
# changed the windows (3/5/7 -> 3/9/15), the resolution (20 km -> 250 m), the
# grid and the training schedule. Augmentation is one of at least five things
# that moved, so that number supports no claim about augmentation at all. It is
# a belief carried in a document, in a project whose three costliest defects
# were beliefs that had never been measured.
#
# THE DESIGN, AND WHY IT IS THE ONLY ONE THAT ANSWERS THE QUESTION.
#
#   * TWO RUNS OF THE SAME GRID. `augment` is NOT a tune_grid dimension -- it
#     reaches train_one_cnn() through the `...` of run_cnn_tuning(), the way
#     03_run_tuning.R:59 sets it in `training_args`. So the on/off contrast
#     cannot be a second row in one grid; it has to be two runs that differ in
#     one argument and in nothing else.
#   * ONE FOLD PLAN, BUILT ONCE, PASSED TO BOTH. Read from the stage 03 run and
#     handed to both arms unchanged, and check_plan_unchanged() is called
#     against each arm's directory before and after. Two arms on two plans are
#     two experiments, and their difference is the difference between the
#     plans.
#   * ONE CONFIG, THE ONE THE PROJECT DEPLOYS. Read from the frozen selection
#     of the stage 03 run, so the answer is about the model that actually goes
#     in the map rather than about an architecture nobody uses.
#   * PAIRED ON (config, fold, seed). Same base_seed on both arms, so every
#     unit has a twin that saw the same training rows from the same
#     initialisation, and paired_family_test() subtracts BOTH nuisances out --
#     the fold's difficulty and the seed's luck -- before asking whether
#     anything is left. In run soc_0_5cm_20260916_232318 the second is the one
#     that matters for this config: its fold means sit within 0.019 CCC while
#     the seed sd within a fold is 0.026-0.038. The note above the paired test
#     below carries the numbers and says how to read pt$se against
#     pt$se_unpaired when the folds are this flat.
#
# COST. 2 arms x 1 config x 3 folds x 3 seeds = 18 CNN units. The same config
# took 1.6-2.0 min per unit in run soc_0_5cm_20260916_232318, so expect roughly
# 35-50 minutes, plus about a minute per fold to build each scaled cache (six
# of those, three per arm). The exact estimate is derived from that run's own
# recorded runtimes and printed below before any training starts.
#
# WHAT THE ANSWER IS ALLOWED TO BE.
#
# "No detectable difference" is a result here, not a failure. D4 augmentation
# is a strong prior -- it asserts that a patch carries no preferred orientation
# -- and a network trained on 1,950 patches with dropout, weight decay and
# early stopping may already be regularised past the point where that prior
# adds anything. Publishing that is worth more than inventing a winner, so the
# verdict below distinguishes FOUR outcomes, not two: an effect that matters,
# an effect too small to matter, a measured absence, and an experiment that
# could not resolve the question. The last two are not the same claim and the
# script never prints one for the other.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_b3_augmentation.R")
# ══════════════════════════════════════════════════════════════════════════════

rm(list = ls())
gc()

# PACKAGES BY library(), NOT BY install_load_pkg() AS IN 03.
#
# 03 sources its installer over the network before rm(list = ls()) wipes it.
# A check script should not need the network to start, and this one runs
# against a store and a tuning run that are already on disk. The set is 03's,
# minus what only 03 uses.
#
# dplyr is a PREREQUISITE, not a convenience: nothing under R/ calls library(),
# while R/resample.R uses %>% inside summarise_resamples(), which is on every
# dsm_train() return path. Without it attached the first arm dies after it has
# trained.
suppressPackageStartupMessages({
  library(torch)
  library(coro)
  library(dplyr)
  library(readr)
  library(tibble)
  library(purrr)
  library(terra)   # only so dsm_load() can read the cell size off the raster
})

options(width = 200)

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
setwd(project_root)

# One source() instead of ten, in an order that is not guessable. See
# R/load_all.R.
source(file.path(project_root, "R", "load_all.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"

# Seeded here only for anything that happens OUTSIDE the runner. Every unit is
# re-seeded by run_cnn_tuning() with base_seed + repetition - 1, which is what
# makes the two arms start from identical weights.
set.seed(42)
torch::torch_manual_seed(42)

# WHICH TUNING RUN THIS TEST HANGS OFF.
#
# PINNED BY NAME, never resolved as "latest". The two arms are comparable with
# each other, and with the stage 03 numbers, only if all of them are defined
# against ONE run: the config, the grid row and the fold plan below all come
# from it. "latest" would silently re-point this test at the next 03 run
# somebody starts, and the verdict would still print -- a verdict carries no
# record of which folds produced it. Point this line at a newer 03 run on
# purpose and the checks below will tell you whether the arms already on disk
# still belong to it.
#
# NOT because "latest" would land on a baselines run. It would not, and it is
# worth writing down so nobody re-derives the wrong reason: baselines runs keep
# their fold_plan.rds one level DOWN, one per family (mlp_centre/, rf_centre/,
# rf_context/), so 03b_run_baselines.R:92, which scans for a TOP-LEVEL
# fold_plan.rds, never sees them -- and sort(decreasing = TRUE) at 03b:98 and
# 04_final_model.R:116 puts "soc_" ahead of "baselines_" regardless, on top of
# 04's own ^soc_ filter at :114.
tuning_run_id <- "soc_0_5cm_20260916_232318"

# WHICH CONFIG IS THE SUBJECT.
#
# NULL reads the frozen selection of that run (comparison/selection.rds, written
# by freeze_selection() when stage 04 locked its choice). That is the config the
# pipeline deploys, which is the one whose training recipe is worth auditing.
#
# Name a config_id here to override -- but note it is a LABEL, not an identity:
# the same "cfg_003" is a different architecture in every tuning run, so an id
# copied from another run names something else entirely.
#
# There is deliberately NO fallback to rank 1 when the selection is missing. A
# fallback would quietly change what this script measured while the header kept
# claiming the deployed config, which is the kind of silent substitution this
# project has already paid for twice.
config_id <- NULL

# HOW MANY REPETITIONS.
#
# Three, and not one, because the whole verdict rests on a noise floor and
# seed_noise_floor() can only estimate one from more than one seed within the
# same (config, fold). With n_seeds = 1 the script would have to print "not
# estimable" and then pass judgement anyway -- the standing rule applied
# without the number it needs.
n_seeds <- 3L

# The seeds themselves. Both arms get the same base, which is what makes a unit
# in one arm the twin of a unit in the other. Passed explicitly rather than
# left to dsm_train()'s default: a changed default would silently unpair the
# two arms and nothing in the output would say so.
base_seed <- 42L

# The decision metric. val_ccc is the metric this project selects on, so it is
# the metric the verdict is read from. val_mae is reported beside it as
# corroboration only -- see the note at that comparison for why a second metric
# is not a second chance to declare a winner.
decision_metric <- "val_ccc"
support_metric  <- "val_mae"

# ── Torch device ──────────────────────────────────────────────────────────────

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Training hyperparameters ──────────────────────────────────────────────────
#
# VERBATIM FROM 03_run_tuning.R, MINUS `augment`. An axis test varies one axis;
# everything else has to be the value the project actually trains with, or the
# answer is about some other training regime. If 03's list changes, change this
# one with it -- and say so in docs/project_log.md, because a B3 result is only
# about the schedule it was measured under.
#
# `augment` is deliberately ABSENT here and supplied per arm. Leaving it in
# would make c(training_args, list(augment = ...)) pass the argument twice, and
# do.call() would abort with "formal argument matched by multiple actual
# arguments" -- loud, but after the data load. The check below is louder and
# earlier.
training_args <- list(
  n_epochs             = 500L,
  patience             = 60L,
  es_min_delta         = 0.0005,
  warmup_start_lr      = 1e-5,
  lr_plateau_factor    = 0.5,
  lr_plateau_patience  = 25L,
  lr_plateau_min_delta = 0.0005,
  min_lr               = 1e-6,
  gradient_clip        = 1.0,
  print_every          = 25L   # 03 uses 10; this run has nothing to watch
                               # epoch by epoch and the log is read afterwards
)

if ("augment" %in% names(training_args)) {
  stop("training_args must NOT carry `augment`: this script supplies it per ",
       "arm, and passing it twice is the one way the two arms could end up ",
       "with the same value while the log claims otherwise.", call. = FALSE)
}

# EVERY NAME IN training_args MUST BE A FORMAL OF train_one_cnn().
#
# The list travels dsm_train(...) -> run_cnn_resample(...) -> run_cnn_tuning(...)
# -> train_one_cnn(), which has no `...` of its own. A misspelt name therefore
# dies with "unused argument" -- but only when the first unit starts training,
# minutes in and after the store is loaded. Checked here, in the first second.
unknown_args <- setdiff(names(training_args), names(formals(train_one_cnn)))
if (length(unknown_args) > 0L) {
  stop("training_args names that train_one_cnn() does not accept: ",
       paste(unknown_args, collapse = ", "),
       "\n  They would travel through `...` and abort the first unit.",
       call. = FALSE)
}

# THE ASSUMPTION THIS WHOLE SCRIPT RESTS ON, ASSERTED RATHER THAN BELIEVED.
#
# If `augment` ever becomes a tune_grid dimension, passing it through `...`
# stops being the way to set it -- the grid column would win, both arms would
# train under whatever the grid says, and the paired test would faithfully
# report no difference between two identical experiments.
if (!"augment" %in% names(formals(train_one_cnn))) {
  stop("train_one_cnn() has no `augment` argument any more. This script sets ",
       "augmentation through it; find where the axis moved to before ",
       "trusting anything below.", call. = FALSE)
}

# ── Paths ─────────────────────────────────────────────────────────────────────

patch_dir    <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
data_dir     <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)

output_tuning_dir <- file.path(project_root, "outputs", "tuning",
                               "soc_stock_modeling", target_label)

tuning_dir <- file.path(output_tuning_dir, tuning_run_id)

# THE TWO ARMS LIVE UNDER ONE PARENT, WITH FIXED NAMES.
#
# Fixed, because a timestamped run_id defeats resume: run_cnn_tuning() looks for
# comparison_all.rds in the SAME run_dir, so a session that dies at unit 14 of
# 18 would otherwise cost the whole arm instead of the unit in flight. An hour
# of CNN time is exactly the thing worth resuming.
#
# Nested, because 04 resolves "latest" over directories matching ^soc_ and 03b
# over directories holding a top-level fold_plan.rds. b3_augmentation/ holds
# neither -- the arms one level down do -- so these runs are invisible to both
# by construction rather than by the lexical luck of "b" sorting before "s".
b3_parent  <- "b3_augmentation"
b3_dir     <- file.path(output_tuning_dir, b3_parent)
arm_run_id <- function(arm) file.path(b3_parent, arm)
arm_dir    <- function(arm) file.path(output_tuning_dir, arm_run_id(arm))

create_output_dirs(b3_dir)

# -- Preconditions, each with what to do about it -----------------------------
#
# All of them stop. Not one of them messages and continues: a B3 that skips its
# own subject and still prints a verdict is the failure mode this project has
# been bitten by three times.

if (!dir.exists(patch_dir)) {
  stop("Patch store not found: ", patch_dir,
       "\n  Run 02_extract_patches.R first.", call. = FALSE)
}
if (!dir.exists(tuning_dir)) {
  stop("Tuning run not found: ", tuning_dir,
       "\n  Set tuning_run_id to a run under ", output_tuning_dir, call. = FALSE)
}

grid_file <- file.path(tuning_dir, "tune_grid.rds")
plan_file <- file.path(tuning_dir, "fold_plan.rds")
sel_file  <- file.path(tuning_dir, "comparison", "selection.rds")
cmp_file  <- file.path(tuning_dir, "comparison", "comparison_all.rds")

for (f in c(grid_file, plan_file)) {
  if (!file.exists(f)) {
    stop("Missing ", f, "\n  This run cannot supply the grid and the folds ",
         "B3 has to reuse. Point tuning_run_id at a complete 03 run.",
         call. = FALSE)
  }
}

# ── The config under test ─────────────────────────────────────────────────────

tune_grid_full <- readRDS(grid_file)

if (is.null(config_id)) {
  if (!file.exists(sel_file)) {
    stop("No frozen selection in ", tuning_dir, ".\n",
         "  B3 tests the config the pipeline deploys, and nothing on disk ",
         "says which that is.\n",
         "  Run 04_final_model.R against this tuning run (it calls ",
         "freeze_selection()),\n  or set config_id explicitly at the top of ",
         "this script.", call. = FALSE)
  }
  selection <- readRDS(sel_file)
  config_id <- selection$config_id[1]
  message("Config under test, from the frozen selection: ", config_id,
          "  (", selection$rule, " on ", selection$metric, ", frozen ",
          format(selection$frozen_at), ")")
} else {
  selection <- NULL
  message("Config under test, named in this script: ", config_id)
}

tune_grid <- tune_grid_full[tune_grid_full$config_id == config_id, , drop = FALSE]
if (nrow(tune_grid) != 1L) {
  stop("Expected exactly one row for config '", config_id, "' in ", grid_file,
       ", found ", nrow(tune_grid), ".\n  The grid of that run holds: ",
       paste(tune_grid_full$config_id, collapse = ", "), call. = FALSE)
}

# ONE CONFIG, AND THAT IS A CHOICE WITH A COST -- STATED, NOT HIDDEN.
#
# The budget in docs/test_plan.md is 2 x 9 units. Nine units per arm buys one
# of three shapes, and only one of them answers the question:
#
#   1 config x 3 folds x 3 seeds   9 paired units, noise floor over 3 (cfg,fold)
#   3 configs x 3 folds x 1 seed   3 paired units, NO noise floor (1 seed)
#   3 configs x 1 fold  x 3 seeds  3 paired units, and one fold is not a
#                                  resampling estimate of anything
#
# paired_family_test() represents a family by ONE config -- the one someone
# would deploy -- so the second and third shapes give it three pairs, its bare
# minimum, while the second also removes the seed noise floor the verdict
# depends on. The first is taken.
#
# WHAT IS GIVEN UP: the answer is about ONE architecture. A lean single-branch
# network may need the orientation prior less, or more, than a dual-branch one
# with several times the parameters -- the printed row below says which one this
# run is about. Widening it to the whole grid is a one-line change (drop the
# subset above) at 3x the cost, and it is resumable: the units already on disk
# are recognised and only the new ones train.
#
# AND ONE MORE BIAS, WORTH NAMING: this config was SELECTED under augment =
# TRUE, in a grid where every config was. That tilts the comparison slightly
# towards the ON arm, because a config that happens to need augmentation is
# more likely to have won. The tilt cannot be removed without re-tuning both
# ways, which is the tune_length = 24 science run's problem, not this one's.
message("\n-- The config under test --")
print_wide(
  dplyr::mutate(
    tune_grid,
    window_sizes  = purrr::map_chr(window_sizes,  paste, collapse = "x"),
    conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")
  )
)

if ("augment" %in% names(tune_grid)) {
  stop("This tune_grid has an `augment` column, so augmentation is now a grid ",
       "dimension.\n  The two arms below would both take the grid's value and ",
       "the comparison would be\n  between two identical experiments. Rewrite ",
       "this test as one grid with two rows.", call. = FALSE)
}

# ── Load the store, for the windows this config needs ─────────────────────────

windows_needed <- sort(unique(unlist(tune_grid$window_sizes)))
message("\nWindows required by this config: ",
        paste(windows_needed, collapse = ", "))

# Same call, same lock as stages 03, 03b and 04: the store has to be the one
# the tuning ran on, or the config that won means nothing here.
data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = windows_needed,
  target_col   = safe_read_csv2(
    file.path(metadata_dir, "target_config.csv"))$target_col[1]
)

store      <- data$store
cell_size  <- data$cell_size
n_channels <- store$n_channels

message("Channels: ", n_channels, " | Points: ", nrow(store$meta))

# ── The fold plan: read, never rebuilt ────────────────────────────────────────
#
# READ FROM THE 03 RUN, the way 04_final_model.R reads it. Rebuilding it from
# spatial_cv(k = 3, "auto", "auto") would almost certainly reproduce it --
# suggest_block_size() is deterministic on these points and the test set is
# frozen in data_split.csv -- but "almost certainly" is the wrong standard for
# the object both arms are defined against. Reading it makes the two arms share
# a plan by construction, and makes them share it with 03, 03b and 04 as well,
# so a B3 number can be read beside a stage 03 number.
plan <- readRDS(plan_file)
if (!inherits(plan, "fold_plan")) {
  stop(plan_file, " is not a fold_plan.", call. = FALSE)
}

message("\n-- Resampling plan (read from ", tuning_run_id, ") --")
print(plan)

# THE BUFFER MUST STILL COVER THIS CONFIG'S WINDOW.
#
# Two patches of width w at resolution res share a pixel when their centres are
# within w-1 cells in BOTH axes, so the exact separation is w * res under the
# Chebyshev metric. The plan was buffered for the LARGEST window of its own
# grid; that is >= this config's window whenever the config came from that grid,
# and it is checked rather than assumed because config_id can be overridden
# above and a future plan may be read from elsewhere. A buffer that is too small
# leaves validation patches overlapping training patches while the leakage
# report prints a clean zero -- this project's own defect, twice.
buffer_needed <- max(windows_needed) * cell_size
if (is.null(plan$params$buffer) || !is.finite(plan$params$buffer)) {
  stop("The plan in ", plan_file, " records no buffer. A window of ",
       max(windows_needed), " needs at least ", format(buffer_needed),
       " in x/y units.", call. = FALSE)
}
if (plan$params$buffer + 1e-12 < buffer_needed) {
  stop(sprintf(
    paste0("The plan's buffer (%.8f) is smaller than this config's window ",
           "requires (%d x %.8f = %.8f).\n  Validation patches would share ",
           "pixels with training patches and nothing downstream would say so."),
    plan$params$buffer, max(windows_needed), cell_size, buffer_needed),
    call. = FALSE)
}
message(sprintf("Buffer: %.8f in x/y units, needed %.8f (window %d x cell %.8f)",
                plan$params$buffer, buffer_needed, max(windows_needed), cell_size))

# ── What this run is about to cost ────────────────────────────────────────────
#
# DERIVED FROM WHAT THE 03 RUN WROTE, not from a number typed here. A literal
# would go stale the first time the config or the machine changes, and a stale
# estimate is worse than none: it is the number someone plans a morning around.
n_folds     <- length(plan$folds)
n_per_arm   <- nrow(tune_grid) * n_folds * n_seeds
n_units_all <- 2L * n_per_arm

prior_minutes <- NA_real_
if (file.exists(cmp_file)) {
  prior <- readRDS(cmp_file)
  prior <- prior[prior$config_id == config_id & prior$status == "success", ,
                 drop = FALSE]
  if (nrow(prior) > 0L) prior_minutes <- mean(prior$runtime_min, na.rm = TRUE)
}

message("\n", strrep("=", 78))
message(sprintf("B3: %d unit(s) -- 2 arm(s) x %d config x %d fold(s) x %d seed(s)",
                n_units_all, nrow(tune_grid), n_folds, n_seeds))
if (is.finite(prior_minutes)) {
  message(sprintf(
    "Expected: ~%.0f min of training (%s averaged %.2f min/unit in %s), plus %d",
    n_units_all * prior_minutes, config_id, prior_minutes, tuning_run_id,
    2L * n_folds))
  message("          fold cache build(s). Resumable: rerun this script and it ",
          "continues.")
} else {
  message("Expected: unknown -- ", tuning_run_id, " recorded no runtime for ",
          config_id, ".")
}
message(strrep("=", 78))

# ── The probe: proof that the flag reached the training loop ──────────────────
#
# THE SILENT FAILURE THIS EXISTS TO CATCH.
#
# `augment` is not written anywhere. It is not a grid column, so it is not in
# tune_grid.csv; it travels through three `...` and lands in a local variable
# inside train_one_cnn(). If it failed to arrive -- a renamed formal, a typo in
# a name that partial-matches something else, an argument swallowed by a runner
# that grew a formal of its own -- BOTH arms would train identically, the paired
# test would report a difference of 0.000 with a tight interval, and this script
# would print "no detectable difference" in the most convincing possible terms.
# That is not a hypothetical class of defect here; it is the class that produced
# rf_grid()'s four identical forests and the buffer that protected one set.
#
# Comparing the two arms' numbers cannot rule it out. Two runs of the same
# config under the same seed are not bit-identical on a GPU -- cuDNN picks
# algorithms non-deterministically -- so "the arms differ" is satisfied by noise
# and proves nothing.
#
# So the call itself is counted. augment_d4_batch() is looked up by name from
# train_one_cnn(), whose closure is the global environment (load_all.R sources
# into it), so a global wrapper IS the function the training loop calls. The ON
# arm must call it; the OFF arm must never call it. That is a fact about this
# session, not an inference from the results.
#
# The wrapper delegates to the original rather than reimplementing it, so the
# augmentation being measured is the framework's, unmodified. It is restored at
# the end; a script that stops midway leaves a wrapper that is semantically the
# original, and any pipeline script restores it anyway (rm(list = ls()) followed
# by load_all.R re-sources R/utils.R).
.b3_augment_d4_batch_real <- augment_d4_batch
.b3_augment_calls <- 0L
augment_d4_batch <- function(tensor_list) {
  .b3_augment_calls <<- .b3_augment_calls + 1L
  .b3_augment_d4_batch_real(tensor_list)
}

# Checkpoints are written only after a unit finishes, so counting them before
# and after an arm says how many units THIS session actually trained -- which is
# what makes the probe readable on a resumed run.
.n_checkpoints <- function(run_dir) {
  length(list.files(file.path(run_dir, "models"), pattern = "_best[.]pt$"))
}

# THE PROBE RECORD OUTLIVES THE SESSION.
#
# On a resumed run every unit is skipped, the counter stays at zero for BOTH
# arms, and a check that reads only this session would either fire falsely on
# the ON arm or quietly excuse itself. Neither is acceptable, so each session
# appends what it observed and the assertion is made over every session that
# ever trained a unit here.
#
# RDS is the authoritative copy and the CSV is the readable one, the same
# arrangement write_comparison() uses -- read_csv2() would return `recorded_at`
# as a POSIXct and the append would abort on "can't combine", which is precisely
# the type drift that cost this project a finished unit once already.
.probe_record <- function(run_dir, arm, augment_flag, n_before, n_after, calls) {
  csv_path <- file.path(run_dir, "augment_probe.csv")
  rds_path <- file.path(run_dir, "augment_probe.rds")
  old <- if (file.exists(rds_path)) readRDS(rds_path) else NULL
  row <- tibble::tibble(
    arm           = arm,
    augment       = augment_flag,
    units_before  = as.integer(n_before),
    units_after   = as.integer(n_after),
    units_trained = as.integer(n_after - n_before),
    augment_calls = as.integer(calls),
    recorded_at   = format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  )
  out <- dplyr::bind_rows(old, row)
  safe_save_rds(out, rds_path, compress = FALSE)
  safe_write_csv2(out, csv_path)
  out
}

# ── One arm ───────────────────────────────────────────────────────────────────

run_arm <- function(arm, augment_flag) {
  run_id  <- arm_run_id(arm)
  run_dir <- arm_dir(arm)

  message("\n", strrep("=", 78))
  message("ARM: ", arm, "   (augment = ", augment_flag, ")")
  message(strrep("=", 78))

  # REFUSE BEFORE SPENDING THE CPU, NOT AFTER.
  #
  # run_cnn_resample() makes this same call itself, but only once it is already
  # inside the run and has loaded the store. Here it costs a millisecond and it
  # answers the question this whole script is built on: are the units already in
  # this directory units of THIS experiment? If the plan moved -- a new 03 run,
  # a changed buffer -- check_plan_unchanged() stops, and it must: resuming
  # would rank units fitted on different training sets against each other, and
  # the arms would no longer be arms of one experiment.
  check_plan_unchanged(plan, run_dir, resume = TRUE)

  n_before <- .n_checkpoints(run_dir)
  .b3_augment_calls <<- 0L

  fit <- do.call(
    dsm_train,
    c(
      list(
        data          = data,
        model         = "cnn",
        resampling    = plan,          # the plan itself, identical for both arms
        tune_grid     = tune_grid,
        transform     = expm1,         # metrics in native ton/ha, never in log1p
        output_dir    = output_tuning_dir,
        device        = device,
        run_id        = run_id,
        base_seed     = base_seed,
        n_seeds       = n_seeds,
        # RESUME IS NOT OPTIONAL WITH MORE THAN ONE FOLD. run_cnn_tuning() is
        # called once per fold and rebuilds `comparison` from the run
        # directory's own record each time; with resume = FALSE it starts from
        # an empty table, so the last fold's write leaves only the last fold's
        # rows. The returned comparison would be a third of this arm.
        resume        = TRUE,
        # THE TEST SET IS NOT READ. B3 asks a question about training, and the
        # answer is a validation-set comparison. Scoring the test here would
        # spend the frozen set on a methods experiment -- and 04 has already
        # spent it once, on the config chosen without it.
        evaluate_test = FALSE,
        verbose       = FALSE          # the report below is this script's own
      ),
      training_args,
      list(augment = augment_flag)
    )
  )

  probe <- .probe_record(run_dir, arm, augment_flag, n_before,
                         .n_checkpoints(run_dir), .b3_augment_calls)

  # ...and again, now that the runner has written its own fold_plan.rds. Before
  # the run this compared the plan with a directory that may have been empty;
  # after it, it compares the plan with what the run recorded. The two calls
  # answer different questions and both are cheap.
  check_plan_unchanged(plan, run_dir, resume = TRUE)

  message(sprintf("\n  %s: %d unit(s) trained this session | augment_d4_batch() called %d time(s)",
                  arm, probe$units_trained[nrow(probe)],
                  probe$augment_calls[nrow(probe)]))

  list(arm = arm, augment = augment_flag, fit = fit, run_dir = run_dir,
       probe = probe)
}

# ── One arm, checked the moment it finishes ───────────────────────────────────
#
# CALLED AFTER EACH ARM, NOT AFTER BOTH. A failed unit in the first arm makes
# the whole comparison impossible, and finding that out after the second arm has
# also run costs twenty minutes for nothing.
#
# THE COUNTS COME FROM THE PLAN AND THE GRID, never from this file. A literal 9
# would pass by arithmetic coincidence the day somebody changes n_seeds and the
# fold count together, and would fail loudly for a change that is perfectly
# legitimate.
check_arm <- function(fit, nm) {
  d <- fit$comparison
  d <- d[d$config_id == config_id, , drop = FALSE]

  if (nrow(d) != n_per_arm) {
    stop("Arm ", nm, " holds ", nrow(d), " unit(s) for ", config_id,
         ", expected ", n_per_arm, " (", nrow(tune_grid), " config x ",
         n_folds, " fold x ", n_seeds, " seed).\n  If the config or the seed ",
         "count changed since this directory was written, delete ",
         arm_dir(nm), " and start it again.", call. = FALSE)
  }
  if (any(d$status != "success")) {
    bad <- d[d$status != "success", , drop = FALSE]
    stop("Arm ", nm, " has ", nrow(bad), " failed unit(s): ",
         paste(bad$unit_id, collapse = ", "), "\n  First error: ",
         bad$error_message[1],
         "\n  A paired comparison over a partial arm compares two different ",
         "experiments.", call. = FALSE)
  }
  if (!setequal(d$seed, base_seed + seq_len(n_seeds) - 1L)) {
    stop("Arm ", nm, " ran under seeds ",
         paste(sort(unique(d$seed)), collapse = ", "), ", expected ",
         paste(base_seed + seq_len(n_seeds) - 1L, collapse = ", "),
         ".\n  The two arms are paired by (fold, seed); different seeds means ",
         "there is nothing to pair.", call. = FALSE)
  }
  if (!setequal(d$fold, seq_len(n_folds))) {
    stop("Arm ", nm, " covers fold(s) ",
         paste(sort(unique(d$fold)), collapse = ", "), ", expected 1 to ",
         n_folds, ".", call. = FALSE)
  }
  message("  ", nm, ": ", nrow(d), " successful unit(s) over ", n_folds,
          " fold(s) and ", n_seeds, " seed(s).")
  d
}

# ── Run both arms ─────────────────────────────────────────────────────────────
#
# ON FIRST, on purpose. It is the arm that repeats what run soc_0_5cm_20260916_232318
# already did for this config, so if anything under R/ has moved since then it
# shows up in the reproduction table below -- before the OFF arm spends its
# twenty minutes.

arm_on <- run_arm("augment_on", TRUE)
cmp_on <- check_arm(arm_on$fit, "augment_on")

# CONTEXT, NOT A CHECK -- and the difference matters.
#
# The ON arm retrains the same config, on the same folds, under the same seeds
# as the 03 run, so the two should land on nearly the same numbers. They are NOT
# asserted equal: R/train_cnn.R and R/metrics.R both changed after that run
# (2026-09-17), so a small shift is a fact about the code's history rather than
# a fault in this experiment. What B3 needs is that its OWN two arms share the
# code, and they do, by running in one session.
#
# A LARGE shift would still be worth knowing before reading anything below,
# which is why it is printed here and not buried in the report.
repro <- NULL
if (file.exists(cmp_file)) {
  prior <- readRDS(cmp_file)
  prior <- prior[prior$config_id == config_id & prior$status == "success", ,
                 drop = FALSE]
  repro <- dplyr::inner_join(
    dplyr::select(cmp_on, unit_id, fold, seed, b3_val_ccc = val_ccc),
    dplyr::select(prior, unit_id, run03_val_ccc = val_ccc),
    by = "unit_id"
  ) %>%
    dplyr::mutate(difference = b3_val_ccc - run03_val_ccc) %>%
    dplyr::arrange(fold, seed)

  if (nrow(repro) > 0L) {
    message("\n-- The ON arm against the same units in ", tuning_run_id,
            " (context, not a check) --")
    print_wide(repro, n = Inf)
    message(sprintf(
      "   largest |difference| = %.4f over %d matched unit(s). R/train_cnn.R and",
      max(abs(repro$difference), na.rm = TRUE), nrow(repro)))
    message("   R/metrics.R changed after that run, so a small shift is expected;")
    message("   a large one means something under R/ moved that nobody recorded.")
  }
}

arm_off <- run_arm("augment_off", FALSE)
cmp_off <- check_arm(arm_off$fit, "augment_off")

# ── Did the two arms actually differ in the one thing they were meant to? ─────
#
# Everything below reads two comparison tables. These checks are what stands
# between "the two arms differ only in augmentation" and "the two arms are two
# tables". Each one stops, because a verdict computed on top of a broken
# premise is worse than no verdict: it is a wrong answer wearing the clothes of
# a measured one.

probe_all <- dplyr::bind_rows(arm_on$probe, arm_off$probe)
message("\n-- The augmentation probe, every session recorded --")
print_wide(probe_all, n = Inf)

# The OFF arm must never have called it, in any session.
off_calls <- probe_all$augment_calls[probe_all$arm == "augment_off"]
if (any(off_calls > 0L)) {
  stop("The OFF arm called augment_d4_batch() ", sum(off_calls), " time(s).\n",
       "  augment = FALSE did not reach the training loop, so the arms are not ",
       "what they claim.\n  Delete ", arm_dir("augment_off"),
       " and find where the argument is being overridden.", call. = FALSE)
}

# ...and the ON arm must have, in at least one session that trained something.
on_rows <- probe_all[probe_all$arm == "augment_on", , drop = FALSE]
if (!any(on_rows$augment_calls > 0L)) {
  stop("No session ever recorded a call to augment_d4_batch() in the ON arm.\n",
       "  Either every unit was resumed from a run that predates this probe, ",
       "or augment = TRUE\n  never reached train_one_cnn(). Delete ",
       arm_dir("augment_on"), " and re-run so the\n  flag is proven rather ",
       "than assumed -- an unproven ON arm makes the whole comparison ",
       "unreadable.", call. = FALSE)
}
if (any(on_rows$units_trained > 0L & on_rows$augment_calls == 0L)) {
  stop("A session trained units in the ON arm without calling ",
       "augment_d4_batch() once.\n  augment = TRUE did not reach the training ",
       "loop for those units.", call. = FALSE)
}

# Each arm was checked on its own as it finished. What is left is the property
# no single arm can have: that the two describe the same experiment.
unit_key <- function(d) paste(d$fold, d$seed, sep = "_")
if (!setequal(unit_key(cmp_on), unit_key(cmp_off))) {
  stop("The two arms do not cover the same (fold, seed) units.\n  on : ",
       paste(sort(unit_key(cmp_on)), collapse = " "), "\n  off: ",
       paste(sort(unit_key(cmp_off)), collapse = " "), call. = FALSE)
}

# The hyperparameters must be the same on both sides too -- the grid is one
# object passed twice, so this can only fail if a cached unit was fitted under
# an older grid and .resumable_units() let it through under the same name.
shared_pars <- setdiff(intersect(names(cmp_on), names(tune_grid)), "config_id")
for (p in shared_pars) {
  if (!identical(unique(as.character(cmp_on[[p]])),
                 unique(as.character(cmp_off[[p]])))) {
    stop("The two arms disagree on hyperparameter '", p, "': on = ",
         paste(unique(cmp_on[[p]]),  collapse = ", "), " | off = ",
         paste(unique(cmp_off[[p]]), collapse = ", "),
         "\n  They are not two arms of one experiment.", call. = FALSE)
  }
}

message("\nBoth arms: ", n_per_arm, " unit(s) each, same ", n_folds,
        " fold(s), same seeds (", paste(base_seed + seq_len(n_seeds) - 1L,
                                        collapse = ", "),
        "), same hyperparameters, augmentation the only difference.")

# ── What each arm scored ──────────────────────────────────────────────────────

by_arm <- dplyr::bind_rows(
  dplyr::mutate(summarise_resamples(cmp_on),  arm = "augment_on"),
  dplyr::mutate(summarise_resamples(cmp_off), arm = "augment_off")
)

message("\n-- Per arm (mean +/- sd over ", n_per_arm, " unit(s)) --")
# all_of() rather than bare names, because `n_folds` and `n_seeds` are also
# variables in this script. dplyr looks at the data first, so the columns win
# today -- but if summarise_resamples() ever renames them, bare names would
# quietly fall through to the environment and select columns 3 and 3 instead of
# failing. all_of() cannot do that: it errors on a name the table lacks.
print_wide(
  dplyr::select(by_arm,
                dplyr::all_of(c("arm", "config_id", "n_units", "n_folds",
                                "n_seeds")),
                dplyr::starts_with("val_ccc"), dplyr::starts_with("val_mae"),
                dplyr::any_of("n_params"), dplyr::all_of("n_failed")),
  n = Inf
)

# ── The seed noise floor, per arm ─────────────────────────────────────────────
#
# How much val_ccc moves when NOTHING changes but the random draw. Measured
# within (config, fold), so the folds' own difficulty is not in it.
#
# THE HARSHER OF THE TWO ARMS IS THE YARDSTICK. The question the verdict has to
# answer is whether turning augmentation on would visibly change a result
# somebody reports, and a result reported from the noisier arm is the one that
# has to clear it.
nf_on  <- seed_noise_floor(cmp_on,  metric = decision_metric)
nf_off <- seed_noise_floor(cmp_off, metric = decision_metric)

message("\n-- Noise floor, ON arm --")
print_noise_floor(nf_on)
message("\n-- Noise floor, OFF arm --")
print_noise_floor(nf_off)

if (!is.finite(nf_on$median_sd) || !is.finite(nf_off$median_sd)) {
  stop("The seed noise floor could not be estimated on both arms ",
       "(on: ", nf_on$n_comparable, " combination(s), off: ",
       nf_off$n_comparable, ").\n  Every verdict below is stated relative to ",
       "it, so there is nothing honest to print without it.\n  It needs more ",
       "than one seed within the same (config, fold): raise n_seeds.",
       call. = FALSE)
}
noise_floor <- max(nf_on$median_sd, nf_off$median_sd)

# ── The paired comparison ─────────────────────────────────────────────────────
#
# PAIRED, on (fold, seed). In run soc_0_5cm_20260916_232318 this config's fold
# means sit within 0.019 CCC of each other (0.4671 / 0.4657 / 0.4851, and no
# pair of folds differs by more) while the seed sd WITHIN a fold is
# 0.026-0.038, with ranges of 0.047 / 0.074 / 0.074. So the dominant nuisance
# HERE is the initialisation draw, not the fold.
#
# Pairing removes both, which is why it is the right design either way: the two
# arms share base_seed, so a pair is two networks that started from the same
# weights on the same training rows, and the difference between them is what
# augmentation did to them.
#
# READ pt$se AGAINST pt$se_unpaired WITH THAT IN MIND. Both are printed below.
# Pairing buys a large SE reduction only when the nuisance it removes is large,
# and in this run the fold spread is nearly flat -- so the two standard errors
# coming out close together is the folds being flat, NOT the pairing failing.
# The seed draw it is actually subtracting does not show up in that ratio,
# because an unpaired SE over 9 units carries the seed spread too.
# tests/test_resample.R:828-835 shows the other regime, where the folds are
# what swamps the effect: 0.00 SE paired against 0.12 unpaired on a constant
# 0.02 shift under a 0.30 fold spread.
#
# config_a and config_b are passed explicitly. They would default to the best
# config per family, which is the same config here -- but a default that
# chooses by metric would let the val_mae comparison below silently compare a
# different pair than the val_ccc one does, and a report whose two lines
# describe two different models is unreadable.
pt <- paired_family_test(
  cmp_on, cmp_off, metric = decision_metric,
  config_a = config_id, config_b = config_id,
  label_a = "augment_on", label_b = "augment_off"
)
print(pt)

# ── The verdict ───────────────────────────────────────────────────────────────
#
# TWO QUESTIONS, ASKED SEPARATELY, BECAUSE THEY ARE NOT THE SAME QUESTION.
#
#   1. Did the experiment RESOLVE a difference?  -> the paired interval
#   2. Is that difference LARGE ENOUGH TO MATTER? -> the seed noise floor
#
# The interval answers the first and only the first: it is built from the
# standard error of the paired differences, which is the right yardstick for
# "is there an effect". The noise floor is NOT that yardstick -- it is the
# spread of one arm across seeds -- and using it as one is the mistake the
# first 03b run made (see the note above paired_family_test()). It answers the
# second question instead: an effect smaller than the distance between two
# retrainings of the SAME model will not change any decision made from this
# pipeline, because the next retraining moves the number by more.
#
# Four cells, and the two "no difference" ones are different claims:
#
#   resolved + above the floor : augmentation changes the result
#   resolved + below the floor : real, and too small to act on
#   not resolved, interval inside the floor : measured absence -- the experiment
#       could have seen an effect the size of the noise and there is none
#   not resolved, interval wider than the floor : the experiment cannot tell,
#       and says how many pairs it would take
ci_half     <- (pt$ci[2] - pt$ci[1]) / 2
resolved    <- !(pt$ci[1] <= 0 && pt$ci[2] >= 0)
above_floor <- abs(pt$diff) >= noise_floor
# Read the sign the way the framework reads it, so that pointing
# decision_metric at an error metric cannot silently name the losing arm:
# for MAE and friends a NEGATIVE difference is the ON arm winning.
direction   <- ifelse(xor(pt$diff > 0, !pt$higher_is_better),
                      "augment_on", "augment_off")

verdict_code <- if (resolved && above_floor) {
  "augmentation_changes_the_result"
} else if (resolved) {
  "effect_smaller_than_the_seed_noise"
} else if (ci_half <= noise_floor) {
  "no_detectable_difference"
} else {
  "unresolved_this_experiment_cannot_tell"
}

message("\n", strrep("=", 78))
message("B3 VERDICT -- D4 augmentation, ", decision_metric, ", config ",
        config_id)
message(strrep("=", 78))
cat(sprintf("  augment_on   mean %s = %.4f\n", decision_metric, pt$mean_a))
cat(sprintf("  augment_off  mean %s = %.4f\n", decision_metric, pt$mean_b))
cat(sprintf("  paired difference (on - off) = %+.4f, 95%% CI [%+.4f, %+.4f], %d pairs\n",
            pt$diff, pt$ci[1], pt$ci[2], pt$n_pairs))
cat(sprintf("  seed noise floor (harsher arm) = %.4f\n", noise_floor))
cat("\n")

if (identical(verdict_code, "augmentation_changes_the_result")) {
  cat(sprintf("  -> AUGMENTATION CHANGES THE RESULT. %s is ahead by %.4f to %.4f\n",
              direction, min(abs(pt$ci)), max(abs(pt$ci))))
  cat(sprintf("     %s, and the gap (%.4f) is larger than the %.4f that\n",
              decision_metric, abs(pt$diff), noise_floor))
  cat("     retraining the same model under another seed moves it. This is an\n")
  cat("     effect a reader would see. Record it in docs/design_decisions.md\n")
  cat("     section 8, which currently rests on a between-round comparison.\n")

} else if (identical(verdict_code, "effect_smaller_than_the_seed_noise")) {
  cat(sprintf("  -> A REAL BUT SMALL EFFECT. The pairing resolves a difference (%s\n",
              direction))
  cat(sprintf("     ahead), but %.4f is SMALLER than the %.4f this model moves\n",
              abs(pt$diff), noise_floor))
  cat("     between seeds. By this project's standing rule that is not evidence\n")
  cat("     anyone can act on: the next retraining moves the number by more than\n")
  cat("     the treatment does. Report the interval, not a recommendation.\n")

} else if (identical(verdict_code, "no_detectable_difference")) {
  cat("  -> NO DETECTABLE DIFFERENCE, AND THAT IS A RESULT.\n")
  cat("     The interval contains zero and lies entirely inside the seed\n")
  cat(sprintf("     noise floor (half-width %.4f <= %.4f), so this experiment had\n",
              ci_half, noise_floor))
  cat("     the resolution to see an effect the size of the noise and there is\n")
  cat("     none. D4 augmentation neither helps nor hurts this config under this\n")
  cat("     schedule -- the orientation prior is already paid for by dropout,\n")
  cat("     weight decay and early stopping, or the network never used it.\n")
  cat("     docs/design_decisions.md section 8 claims the opposite on the\n")
  cat("     strength of a between-round comparison in which four other things\n")
  cat("     also changed. This measurement supersedes it.\n")

} else {
  n_needed <- ceiling(pt$n_pairs * (ci_half / noise_floor)^2)
  cat("  -> UNRESOLVED. This experiment cannot tell.\n")
  cat("     The interval contains zero but is wider than the noise floor\n")
  cat(sprintf("     (half-width %.4f > %.4f), so an effect worth acting on is\n",
              ci_half, noise_floor))
  cat("     still compatible with what was measured. This is NOT 'no difference'\n")
  cat("     and must not be reported as one.\n")
  cat(sprintf("     At this spread it would take about %d paired unit(s) per arm\n",
              n_needed))
  cat(sprintf("     (this run had %d) to shrink the interval to the noise floor:\n",
              pt$n_pairs))
  cat(sprintf("     raise n_seeds to %d, or widen the plan to more folds. Both are\n",
              ceiling(n_needed / n_folds)))
  cat("     resumable -- the units on disk are recognised and only the new ones\n")
  cat("     train.\n")
}
message(strrep("=", 78))

# ── Corroboration on the error metric ─────────────────────────────────────────
#
# NOT A SECOND CHANCE TO FIND A WINNER. Two metrics are two opportunities to
# declare an effect, and picking whichever one separates is how a null result
# becomes a finding. The verdict above is read from val_ccc alone, because that
# is the metric this project selects on; val_mae is here to say whether the two
# metrics agree, which is a fact about the measurement rather than about
# augmentation. Lower is better for MAE, so agreement means the SIGN flips.
pt_mae <- paired_family_test(
  cmp_on, cmp_off, metric = support_metric,
  config_a = config_id, config_b = config_id,
  label_a = "augment_on", label_b = "augment_off"
)
print(pt_mae)

# Agreement means the signs are OPPOSITE: the arm with the higher CCC should be
# the arm with the lower MAE.
agree <- isTRUE(sign(pt$diff) == -sign(pt_mae$diff))
message(sprintf("  %s and %s %s about which arm is ahead.",
                decision_metric, support_metric,
                if (agree) "AGREE" else "DISAGREE"))
if (!agree) {
  message("  A disagreement is not a tie-break to be spent: it says the two ",
          "arms differ in")
  message("  the SHAPE of their errors, not in their size. Look at the ",
          "quantile tables in")
  message("  each arm's metrics/ directory before writing anything about it.")
}

# ── What leaves this script ───────────────────────────────────────────────────
#
# The tables, so the verdict above can be re-read by someone who was not here.
# PT-BR locale via safe_write_csv2(): ';' separator, ',' decimal, UTF-8 BOM.

units_out <- dplyr::bind_rows(
  dplyr::mutate(cmp_on,  arm = "augment_on",  augment = TRUE),
  dplyr::mutate(cmp_off, arm = "augment_off", augment = FALSE)
) %>%
  dplyr::relocate(arm, augment) %>%
  dplyr::arrange(arm, fold, seed)

paired_out <- tibble::tibble(
  metric          = c(pt$metric, pt_mae$metric),
  role            = c("decision", "corroboration"),
  config_id       = config_id,
  tuning_run_id   = tuning_run_id,
  n_pairs         = c(pt$n_pairs, pt_mae$n_pairs),
  n_unpaired      = c(pt$dropped, pt_mae$dropped),
  mean_augment_on = c(pt$mean_a, pt_mae$mean_a),
  mean_augment_off = c(pt$mean_b, pt_mae$mean_b),
  difference      = c(pt$diff, pt_mae$diff),
  se              = c(pt$se, pt_mae$se),
  se_unpaired     = c(pt$se_unpaired, pt_mae$se_unpaired),
  ci_low          = c(pt$ci[1], pt_mae$ci[1]),
  ci_high         = c(pt$ci[2], pt_mae$ci[2]),
  t               = c(pt$t, pt_mae$t),
  p               = c(pt$p, pt_mae$p),
  seed_noise_floor = c(noise_floor, NA_real_),
  verdict         = c(verdict_code, NA_character_)
)

noise_out <- dplyr::bind_rows(
  dplyr::mutate(nf_on$by_config,  arm = "augment_on"),
  dplyr::mutate(nf_off$by_config, arm = "augment_off")
) %>%
  dplyr::relocate(arm)

safe_write_csv2(units_out,  file.path(b3_dir, "b3_units.csv"))
safe_write_csv2(paired_out, file.path(b3_dir, "b3_paired.csv"))
safe_write_csv2(noise_out,  file.path(b3_dir, "b3_noise_floor.csv"))
safe_write_csv2(probe_all,  file.path(b3_dir, "b3_augment_probe.csv"))
if (!is.null(repro) && nrow(repro) > 0L) {
  safe_write_csv2(repro, file.path(b3_dir, "b3_reproduction_vs_03.csv"))
}

message("\nWritten to ", b3_dir, ":")
message("  b3_units.csv          every unit of both arms, as it was measured")
message("  b3_paired.csv         the paired comparison and the verdict")
message("  b3_noise_floor.csv    the seed spread, per (config, fold), per arm")
message("  b3_augment_probe.csv  proof that the flag reached the training loop")
if (!is.null(repro) && nrow(repro) > 0L) {
  message("  b3_reproduction_vs_03.csv  the ON arm against ", tuning_run_id)
}

# The framework's own function goes back, so nothing sourced after this in the
# same session keeps counting. A pipeline script would restore it anyway --
# rm(list = ls()) then load_all.R -- but leaving a wrapper behind for the next
# reader to discover is not a courtesy.
augment_d4_batch <- .b3_augment_d4_batch_real
rm(.b3_augment_d4_batch_real, .b3_augment_calls)

message("\nB3 complete: ", verdict_code)
