# ══════════════════════════════════════════════════════════════════════════════
# B1 -- kNNDM folds on the real points, and what they actually bought
#
# WHAT THIS IS.
#
# docs/test_plan.md, tier B, row B1: `knndm_cv()` on the real points. Everything
# this project knows about kNNDM so far is either a property of the code
# (tests/test_knndm.R, on 320 synthetic points) or a measurement of the COST
# (_measure_knndm.R). The two things those settled are not repeated here:
#
#   * the coordinates are global lon/lat, so every distance must be computed in
#     an equal-area projection -- a degree of longitude is 111 km at the equator
#     and 0 at the pole, and a "distance" computed over that is not a distance;
#   * projected, the nearest-neighbour search is a kd-tree and the cost is not a
#     consideration at this n. Unprojected it is O(n^2) in time AND memory.
#
# _measure_knndm.R closed with the sentence this script exists to answer:
# "What this does NOT settle: whether kNNDM folds are BETTER than the block
# folds in use. That is an empirical question."
#
# THE MEASUREMENT, AND WHY IT IS A DISTANCE DISTRIBUTION AND NOT A SCORE.
#
# The tempting comparison is to train on both plans and see which reports the
# better CCC. That comparison is meaningless in the direction people read it:
# the plan that leaks scores HIGHER, so "kNNDM scored lower" would be evidence
# in its favour and "kNNDM scored higher" evidence against, and neither could be
# distinguished from noise at three folds. A score cannot rank two definitions
# of what the score means.
#
# kNNDM's own claim is geometric and can be checked geometrically. It exists to
# make the fold geometry resemble what PREDICTION faces, by minimising the
# Wasserstein distance W between two empirical distributions:
#
#   Gij  for every validation point, the distance to its nearest TRAINING point
#        in its own fold -- what the model is scored on;
#   Gj   for every prediction pixel, the distance to its nearest TRAINING point
#        -- what the model will actually be asked to do.
#
# If those two distributions coincide, the cross-validated number describes the
# map. If validation points sit far closer to training points than prediction
# pixels do, the number is optimistic; far further, and it is pessimistic. Both
# happen, and the sign is not guessable in advance -- which is the whole reason
# to measure rather than to argue about block sizes.
#
# So the script computes Gij for BOTH plans against the SAME Gj, prints the
# distributions behind the summary, and recomputes W itself instead of trusting
# the single number CAST reports. W is the summary; the summary is what hides a
# wrong answer.
#
# WHAT THIS SCRIPT DOES NOT DO: IT TRAINS NOTHING.
#
# Fold construction and geometry only. The `rf` run on these folds is a separate,
# later step -- see the clearly marked block at the end of this file, which also
# says why it is separate. The plan is written to disk precisely so that step
# does not have to rebuild it and risk building a different one.
#
# ── RUNTIME, ESTIMATED FROM THE CODE RATHER THAN FROM THE BUDGET ──────────────
#
# docs/test_plan.md budgets ~1 h for B1. That budget is for B1 INCLUDING the rf
# run that follows; this script is the cheap half of it. Reading the code says:
#
#   packages + load_all.R + dsm_load (one window)        ~1 min
#   the 20 km footprint: 181 rasters x 797 x 2004 cells  ~1-4 min   <- dominates
#   prediction_sample() regular draw                     seconds
#   CAST::knndm, 3,137 points, ~5,000 predpoints, k = 3  ~1-3 min
#   the nearest-neighbour distances (FNN, kd-tree)       seconds
#   fold_leakage_report() at windows 3/9/15              ~1 min
#
#   -> about 5 minutes, 10 if CRAN has to install CAST first.
#
# The dominant cost is NOT kNNDM. It is reading 181 rasters to find out where
# the map has values at all, and that is worth saying out loud because the
# instinct is to budget for the clustering.
#
# ── BEFORE YOU RUN IT ─────────────────────────────────────────────────────────
#
# The prediction grid comes from soc_predict_raster_dir, the same variable 05
# reads, and there is no default (the reason is at the check itself). Set it in
# the console FIRST -- it survives the rm(list = ls()) below, a workspace object
# does not:
#
#   Sys.setenv(soc_predict_raster_dir =
#     "D:/usuario_armazenamento/cassio/R/predictors_resolution_20000m")
#   source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_b1_knndm_folds.R")
#
# _b4_shard_merge_check.R leaves that variable set deliberately, so after a B4
# run it is already there.
# ══════════════════════════════════════════════════════════════════════════════

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

# torch is here for ONE reason and it is not training. R/load_all.R sources
# R/cnn_architecture.R, which calls torch::nn_module() at the top level, so the
# framework cannot be loaded without torch being loadable. Nothing below builds
# a network or touches a device.
#
# CAST is listed so a machine without it installs it now rather than eight
# minutes in, after the raster stack has been read. docs/project_log.md records
# that CAST was NOT installed when _measure_knndm.R was written, so this is the
# likely first run on this machine.
pkg <- c(
  "torch",
  "dplyr",
  "readr",
  "tibble",
  "purrr",       # create_output_dirs() walks with it
  "sf",          # the projection
  "terra",       # the prediction grid
  "FNN",         # kd-tree nearest neighbours -- the same package _measure_knndm used
  "CAST"         # the reference kNNDM implementation; never re-derived here
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# One source() instead of ten, in an order that is not guessable. See
# R/load_all.R.
source(file.path(project_root, "R", "load_all.R"))

# ── Settings ──────────────────────────────────────────────────────────────────

target_label <- "soc_stock_0_5cm"

# The seed kNNDM samples with. Fixed and stated: a fold plan that cannot be
# rebuilt identically is a fold plan the later rf run cannot be compared against.
knndm_seed <- 42L

# How many prediction points the distance distribution is estimated from. kNNDM
# compares DISTRIBUTIONS, and a few thousand regularly spaced pixels estimate
# one well; this is prediction_sample()'s own default and CAST's advice.
# More would be slower for no gain in a distribution this smooth.
target_predpoints <- 5000L

# The smallest realised sample this script will accept. Not a tuning knob: below
# a couple of thousand the tail of Gj -- the part that decides W -- is estimated
# from a few dozen points, and W becomes a number about the draw.
min_predpoints <- 2000L

# THE PROJECTION, DEFINED ONCE AND PASSED TO BOTH SIDES.
#
# knndm_folds() defaults to this same Mollweide string, so passing it explicitly
# changes nothing today. It is passed anyway because THIS script also projects
# the points itself, to compute the distance distributions, and two independent
# defaults that happen to agree are two defaults free to stop agreeing. The plan
# records what it used and the check below compares the two.
#
# Equal-area rather than conformal: what is being compared is distance across
# the whole domain, and an equal-area projection keeps that comparison honest at
# the scale the folds are cut at.
moll <- "+proj=moll +lon_0=0 +datum=WGS84 +units=m"

# ── Paths ─────────────────────────────────────────────────────────────────────

patch_dir    <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
data_dir     <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
tuning_base  <- file.path(project_root, "outputs", "tuning",
                          "soc_stock_modeling", target_label)

# NESTED TWO DEEP, ON PURPOSE. 03b resolves "latest" over directories DIRECTLY
# under tuning_base that hold a top-level fold_plan.rds, and 04 over names
# matching ^soc_. This script writes a fold_plan.rds; put it one level up and
# the next 03b run would silently score the baselines on kNNDM folds. Under
# capability_sweep/<id> it is invisible to both by construction rather than by
# the lexical luck of "b" sorting before "s".
# Created after the preconditions below, not here: a run that stops because an
# environment variable is unset should not leave an empty report directory
# behind for the next reader to wonder about.
report_dir <- file.path(tuning_base, "capability_sweep", "b1_knndm")

message("\n", strrep("=", 78))
message("B1 -- kNNDM folds on the real points (fold geometry only, no training)")
message(strrep("=", 78))
message("Estimated runtime: about 5 minutes, 10 if CAST has to be installed.")
message("The dominant cost is reading every predictor raster over the")
message("prediction grid to find where the map has values -- not the")
message("clustering. Nothing is trained here.")
message(strrep("=", 78), "\n")

# ── The two preconditions, checked before anything expensive ──────────────────
#
# ORDER IS THE POINT. Both of these are instant and both are fatal, and the work
# after them costs minutes. Checked afterwards, a missing environment variable
# would be reported after the patch store had been read -- a minute spent to
# learn something that was knowable at once. This project has paid that bill
# often enough to make the ordering deliberate.

# 1. CAST. install_load_pkg() above installs it if it is absent, but require()
#    only WARNS when an install fails, and a warning scrolls past. knndm_folds()
#    would then produce its own good message several minutes from here.
if (!requireNamespace("CAST", quietly = TRUE)) {
  stop("The CAST package is not available, and kNNDM is not re-implemented ",
       "here on purpose -- a published CV method re-coded locally is a method ",
       "that quietly differs from the one being cited.\n",
       "  install.packages(\"CAST\")", call. = FALSE)
}

# 2. WHERE THE MAP WILL BE PREDICTED, read the way 05_predict_spatial.R:115-118
#    reads it. The rm(list = ls()) above erases any variable set before the
#    source(), so an environment variable is the only way to pass a parameter
#    into a source()d run: Sys.setenv() survives that rm(), a workspace object
#    does not.
predict_raster_dir <- NULL
if (nzchar(Sys.getenv("soc_predict_raster_dir"))) {
  predict_raster_dir <- Sys.getenv("soc_predict_raster_dir")
}

# UNSET IS REFUSED RATHER THAN DEFAULTED, and this is the one place where that
# is a judgement call worth stating. Unset means "the 250 m grid the patches
# were cut from" -- a legitimate prediction area, and the only one whose CV
# numbers are comparable with the validation metrics. It is refused anyway
# because the footprint below has to read every predictor over the whole grid,
# and at 250 m that is 181 rasters of ~11 GB each. That is a morning, not a
# check. If the 250 m footprint is ever wanted, this is the line to change, and
# it should be changed deliberately.
if (is.null(predict_raster_dir)) {
  stop("soc_predict_raster_dir is not set, so this script has no prediction ",
       "grid to match the folds to.\n",
       "  Sys.setenv(soc_predict_raster_dir = ",
       "\"D:/usuario_armazenamento/cassio/R/predictors_resolution_20000m\")\n",
       "  and source this file again. The 20 km grid is the one the ",
       "pipeline-closing prediction was drawn on.\n",
       "  Defaulting to the 250 m grid would read 181 rasters of ~11 GB each ",
       "to build one mask.", call. = FALSE)
}
if (!dir.exists(predict_raster_dir)) {
  stop("predict_raster_dir does not exist: ", predict_raster_dir, call. = FALSE)
}
message("Prediction rasters: ", predict_raster_dir, "\n")

create_output_dirs(report_dir)

# ── The check ledger ──────────────────────────────────────────────────────────
#
# WHY A LEDGER AND NOT A RUN OF stopifnot(). A script that stops at the first
# failure reports one fact; this one is supposed to answer "is the plan sane AND
# what did it buy", and the second half is still worth reading when the first
# half has a problem. So every check is recorded with the NUMBER it measured,
# and the ones whose failure makes everything after them meaningless stop
# immediately afterwards, by name.
#
# `measured` is not decoration. A check that records only TRUE/FALSE cannot be
# re-read in six months against a run that has moved; the number can.

.b1_checks <- list()

check_that <- function(id, what, ok, measured) {
  ok <- isTRUE(ok)
  .b1_checks[[id]] <<- tibble::tibble(
    id = id, check = what, ok = ok, measured = as.character(measured)[1])
  message(sprintf("  [%s] %-8s %-44s %s", if (ok) "PASS" else "FAIL", id,
                  what, substr(as.character(measured)[1], 1, 110)))
  invisible(ok)
}

# EVERY CHECK THIS SCRIPT PROMISES, WRITTEN OUT.
#
# The verdict is not "nothing in the ledger is FALSE" -- an empty ledger
# satisfies that, and a ledger missing the one expensive check satisfies it too.
# This project has already been bitten by exactly that shape: tests/helper.R
# carries the same guard for the same reason, and _b4_shard_merge_check.R exists
# because a check turned itself off and still printed PASS. So the verdict
# requires every id below to be PRESENT and TRUE.
required_checks <- c(
  "b1_01",  # the block plan belongs to this store
  "b1_02",  # the frozen test set is the one on disk
  "b1_03",  # the predpoint grid is the grid 05 predicted on
  "b1_04",  # that grid's footprint covers what 05 actually wrote
  "b1_05",  # enough predpoints survived the NA mask
  "b1_06",  # check_fold_plan() accepts the kNNDM plan
  "b1_07",  # every fold has both training and validation points
  "b1_08",  # the folds partition the pool: every point validated exactly once
  "b1_09",  # no point holds two roles in one fold
  "b1_10",  # the test set appears in no fold's training or validation
  "b1_11",  # the plan projected distances the way this script does
  "b1_12",  # CAST's W is reproducible from the distributions behind it
  "b1_13"   # kNNDM matches prediction better than the blocks do
)

# ── 1. The store, and the block plan already in use ───────────────────────────
#
# ONE WINDOW IS LOADED, NOT THREE. dsm_load() is used rather than reading
# patch_meta.csv directly because it is the call that cannot be half-done: it
# aligns the points to the store by sample_id, reads the resolution from the
# raster itself, and refuses if the store was built under a different predictor
# set, target or resolution. Those are the checks that stop this script from
# measuring geometry on a point table that is not the one the folds index into.
#
# The window is the smallest the store has because nothing here reads a tensor:
# fold geometry needs x, y and sample_id. Loading 3/9/15 would cost 0.85 GB to
# hold patches nothing touches. (The window sizes the LEAKAGE report needs are
# read from the 03 grid separately, below -- that report works from coordinates
# and a cell size, not from pixels.)
store_manifest <- readRDS(file.path(patch_dir, "patch_manifest.rds"))
windows_available <- as.integer(trimws(
  strsplit(store_manifest$windows_extracted[1], ",")[[1]]))

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = min(windows_available),
  target_col   = safe_read_csv2(file.path(metadata_dir,
                                          "target_config.csv"))$target_col[1]
)

meta      <- data$store$meta
cell_size <- data$cell_size
n_points  <- nrow(meta)

if (is.null(cell_size) || !is.finite(cell_size)) {
  stop("dsm_load() could not resolve the raster resolution, and the leakage ",
       "report below is expressed in raster cells. Check that the first ",
       "raster_file in raster_table_used.csv still exists.", call. = FALSE)
}

# THE BLOCK PLAN IS READ, NOT REBUILT.
#
# The comparison asked for is against "the block folds already in use" -- the
# plan stage 03 actually trained on, buffer applied and all. Rebuilding it here
# with spatial_cv() would reproduce it only as long as block_size = "auto" and
# the buffer rule stay put, and a comparison against a plan that is nearly the
# one in use is a comparison of two unknowns.
#
# Resolved the way 03b resolves it (a directory holding a top-level
# fold_plan.rds) AND the way 04 resolves it (a name matching ^soc_). Both rules
# together, because 03b's alone also matches baselines_* -- which sorts after
# the CNN runs and would silently win "latest".
run_dirs <- list.dirs(tuning_base, recursive = FALSE, full.names = FALSE)
run_dirs <- run_dirs[grepl("^soc_", run_dirs) &
                     file.exists(file.path(tuning_base, run_dirs,
                                           "fold_plan.rds"))]
if (length(run_dirs) == 0L) {
  stop("No tuning run with a top-level fold_plan.rds under:\n  ", tuning_base,
       "\nB1 compares kNNDM against the block folds already in use, so there ",
       "has to be a run to compare with. Run 03_run_tuning.R first.",
       call. = FALSE)
}
cnn_run_id  <- sort(run_dirs, decreasing = TRUE)[1]
cnn_run_dir <- file.path(tuning_base, cnn_run_id)
message("\nBlock folds read from tuning run: ", cnn_run_id)

plan_block <- readRDS(file.path(cnn_run_dir, "fold_plan.rds"))
stopifnot(inherits(plan_block, "fold_plan"))

# The fold indices are ROW POSITIONS into the store's meta. A plan built on a
# different store indexes the wrong points and every number below would be
# about a point set that is not this one -- and it would not error, which is the
# failure mode this project keeps meeting.
check_that(
  "b1_01", "block plan fits this store",
  plan_block$n_rows == n_points &&
    max(unlist(lapply(plan_block$folds, function(f)
      c(f$train, f$validation, f$test)))) <= n_points,
  sprintf("plan says %d rows, store holds %d, method '%s'",
          plan_block$n_rows, n_points, plan_block$method))
if (!.b1_checks[["b1_01"]]$ok) {
  stop("The block plan was built on a different point set than the store holds. ",
       "Re-run 03_run_tuning.R against this store, or point cnn_run_dir at the ",
       "run that matches it.", call. = FALSE)
}

k_folds <- plan_block$n_folds

# THE TEST SET IS TAKEN FROM THE PLAN, NOT FROM THE COUNT.
#
# Two plans held out on different data are two experiments, so kNNDM is given
# the same frozen ids by test_ids. They are derived from the plan the run used
# and then cross-checked against data_split.csv -- the file 03 froze them into.
# Checking the SIZE instead would be worthless: a fresh 15% draw is
# ceiling(0.15 * 3728) = 560 and the frozen set is 591, so a size check passes
# on a completely different set of points by arithmetic coincidence.
test_pos <- plan_block$folds[[1]]$test
if (length(test_pos) == 0L) {
  stop("The block plan holds out no test set, so there is nothing to freeze ",
       "and the two plans would be built on different data.\n  Re-run ",
       "03_run_tuning.R with test_frac > 0, or drop the frozen test set from ",
       "this comparison deliberately.", call. = FALSE)
}
pool_pos    <- setdiff(seq_len(n_points), test_pos)
frozen_test <- meta$sample_id[test_pos]

split_file <- file.path(metadata_dir, "data_split.csv")
if (!file.exists(split_file)) {
  stop("data_split.csv not found: ", split_file,
       "\nIt is the frozen test set. Without it there is nothing to prove the ",
       "plan's held-out points are the ones the project froze.", call. = FALSE)
}
ds <- safe_read_csv2(split_file)
split_test <- ds$sample_id[ds$role == "test"]
check_that(
  "b1_02", "frozen test set is the one on disk",
  setequal(as.character(frozen_test), as.character(split_test)) &&
    length(frozen_test) == length(split_test),
  sprintf("%d ids in the plan, %d in data_split.csv, %d shared",
          length(frozen_test), length(split_test),
          length(intersect(as.character(frozen_test),
                           as.character(split_test)))))
if (!.b1_checks[["b1_02"]]$ok) {
  stop("The block plan's test set and data_split.csv disagree. One of them is ",
       "stale; delete the tuning run or re-freeze the split deliberately ",
       "before comparing anything against it.", call. = FALSE)
}

message(sprintf("\nPoints: %d  |  pool %d  |  frozen test %d  |  k = %d",
                n_points, length(pool_pos), length(test_pos), k_folds))

# ── 2. Where the map will actually be predicted ───────────────────────────────
#
# kNNDM without a prediction area is an expensive random split, so this section
# is the method, not preparation for it. The directory itself was resolved and
# checked at the top, before anything expensive ran.
#
# THE REMAPPING, DONE THE WAY 05 DOES IT.
#
# 05_predict_spatial.R:321-337 matches predictors by FILE NAME into the override
# directory, keeps the TRAINING channel order rather than the alphabetical order
# a directory listing returns, and refuses if one is missing. The same pattern
# is used here. The order does not matter for a footprint -- every channel has
# to be present regardless of which is first -- but reproducing the refusal does:
# a directory missing a predictor is a directory that describes a different
# prediction area than the one 05 drew.
raster_table <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
predictor_cols <- raster_table$predictor
n_channels     <- length(predictor_cols)

avail    <- list.files(predict_raster_dir, pattern = "\\.(tif|tiff)$",
                       full.names = TRUE, ignore.case = TRUE)
remapped <- file.path(predict_raster_dir, basename(raster_table$raster_file))
absent   <- raster_table$predictor[!remapped %in% avail]
if (length(absent) > 0L) {
  stop("predict_raster_dir is missing ", length(absent), " of the ", n_channels,
       " predictors the model needs, among them: ",
       paste(utils::head(absent, 8), collapse = ", "),
       "\nThe prediction footprint is where ALL channels have a value, so a ",
       "missing one cannot be worked around here any more than it can in 05.",
       call. = FALSE)
}

rast_stack <- terra::rast(remapped)
names(rast_stack) <- predictor_cols

# The points are lon/lat and knndm_folds() is told crs = 4326. The predpoints
# come back in the RASTER's CRS, so if that is not lon/lat the two sets are
# projected from different starting points and every distance below is wrong --
# silently, because both would still be numbers in metres.
if (!isTRUE(terra::is.lonlat(rast_stack))) {
  stop("The prediction rasters are not in lon/lat, but the point coordinates ",
       "are (crs = 4326). Projecting the two from different assumptions ",
       "produces distances that look fine and are not comparable.\n  CRS: ",
       terra::crs(rast_stack, describe = TRUE)$name, call. = FALSE)
}

r_nrow <- as.integer(terra::nrow(rast_stack))
r_ncol <- as.integer(terra::ncol(rast_stack))
n_cell <- as.numeric(terra::ncell(rast_stack))
message(sprintf("\nPrediction grid: %d rows x %d cols x %d layers at %.8f/pixel",
                r_nrow, r_ncol, n_channels, terra::res(rast_stack)[1]))
message(sprintf("Training grid  : %.8f/pixel", cell_size))
# Both are printed because the two resolutions differ by 80x here, and that is
# NOT a problem for this script the way it is for 05: the folds are cut on the
# POINTS, in degrees, and the prediction grid enters only as a set of locations
# to measure distances to. A coarser grid gives a sparser estimate of Gj, not a
# different Gj.

# WHAT PROVES THIS IS THE GRID THE MAP WAS DRAWN ON.
#
# Nothing so far does. The directory could hold any raster set; b4 guards the
# same risk with stopifnot(exp_ncol < 10000L), which catches the 250 m grid and
# nothing else. 05 wrote down the grid it predicted, so that record is what this
# is compared against -- the only source that cannot be satisfied by a plausible
# wrong directory.
#
# Resolved exactly as 05, 05a and _b4_shard_merge_check.R resolve it, so all of
# them agree on which run is meant.
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)
final_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
final_dirs <- final_dirs[grepl("^final_", final_dirs)]
if (length(final_dirs) == 0L) {
  stop("No final model run under: ", final_model_base,
       "\nThe recorded prediction grid is read from the 05 run of that model, ",
       "and it is what proves the predpoints below come from the grid the map ",
       "was drawn on. Run 04 and 05 first.", call. = FALSE)
}
final_run_id <- sort(final_dirs, decreasing = TRUE)[1]
config_id <- readRDS(file.path(final_model_base, final_run_id, "comparison",
                               "final_run_summary.rds"))$selected_cfgs$config_id[1]
pred_cfg_file <- file.path(project_root, "outputs", "spatial_prediction",
                           "soc_stock_modeling", target_label, config_id,
                           "log", "prediction_config.csv")
if (!file.exists(pred_cfg_file)) {
  stop("05 has left no record of the grid it predicted: ", pred_cfg_file,
       "\nWithout it nothing here can show that soc_predict_raster_dir points ",
       "at the grid the map was drawn on, and 'the folds match prediction' is ",
       "the only claim this script makes.\n",
       "  Run 05_predict_spatial.R with soc_predict_raster_dir set to this ",
       "same directory, then source this file again.", call. = FALSE)
}
pred_cfg <- safe_read_csv2(pred_cfg_file)
# The unsuffixed file belongs to the unpartitioned run: 05 appends a shard
# suffix whenever it is partitioned. Asserted rather than assumed, because a
# tile's r_nrow/r_ncol are the whole grid's but its n_valid is not.
if (!isTRUE(pred_cfg$n_row_shards[1] == 1L && pred_cfg$n_col_shards[1] == 1L)) {
  stop("prediction_config.csv records a partitioned run (", pred_cfg$n_row_shards[1],
       " x ", pred_cfg$n_col_shards[1], "). The unsuffixed file should be the ",
       "1x1 run; this one is not, so its n_valid describes a tile.",
       call. = FALSE)
}

check_that(
  "b1_03", "predpoint grid is the grid 05 predicted",
  pred_cfg$r_nrow[1] == r_nrow && pred_cfg$r_ncol[1] == r_ncol,
  sprintf("this stack %d x %d, 05 recorded %d x %d (config %s)",
          r_nrow, r_ncol, pred_cfg$r_nrow[1], pred_cfg$r_ncol[1], config_id))
if (!.b1_checks[["b1_03"]]$ok) {
  stop("soc_predict_raster_dir points at a different grid than the one 05 ",
       "predicted on. Point it at that grid, or re-run 05 on this one.",
       call. = FALSE)
}

# ── 3. The footprint, and the predpoints drawn from it ────────────────────────
#
# WHERE THE MAP HAS VALUES IS NOT WHERE THE RASTERS COVER.
#
# 05 predicts a pixel only when every channel is FINITE at its centre
# (05_predict_spatial.R:652, `quick_ok`), and then only when a full patch can be
# built around it. The first condition is reproduced here; the second is not,
# because it depends on the config's window and on build_patches_multi()'s own
# validity rule, and a second implementation of that rule is a second chance to
# differ from it.
#
# `is.na` here against 05's `is.finite`: the two differ only on an infinity
# stored in a GeoTIFF, which nothing in this predictor set does and which would
# have broken the scaling in stage 02 long before reaching here. The difference
# is named rather than hidden because it is the reason the check below is a
# bound rather than an equality -- one of two reasons, the other being the rim.
#
# So this footprint is a SUPERSET of what 05 wrote, by the thin rim where a
# patch does not fit -- which is why the check below is a bound and not an
# equality. An equality check here would fail a correct run, and the natural
# repair (loosening it to a tolerance) would be a check that no longer decides
# anything.
#
# The alternative considered and discarded: sample 05's own valid_patch_mask
# raster, which is exactly the footprint and costs one file read. Discarded
# because it makes B1 depend on the OUTPUT of a 05 run rather than on the
# predictor directory, and the point of honouring soc_predict_raster_dir is that
# the prediction area is defined by the rasters, not by what was done with them.
message("\nBuilding the prediction footprint (all ", n_channels,
        " channels present)... this is the slow step.")
t0 <- Sys.time()
n_missing_channels <- sum(is.na(rast_stack))
footprint <- terra::ifel(n_missing_channels == 0, 1, NA)
n_footprint <- as.numeric(terra::global(footprint, "sum", na.rm = TRUE)[1, 1])
# An empty footprint would propagate as NaN into the sample size and surface as
# a confusing error inside prediction_sample(). It means the rasters do not
# overlap each other, which is worth its own sentence.
if (!is.finite(n_footprint) || n_footprint <= 0) {
  stop("No cell in ", predict_raster_dir, " carries all ", n_channels,
       " channels, so there is no prediction area to match the folds to.\n",
       "  Check that these rasters share one grid -- 05 refuses the same case ",
       "with a compareGeom() report.", call. = FALSE)
}
message(sprintf("  %.1f min | %s of %s cells carry every channel (%.1f%%)",
                as.numeric(difftime(Sys.time(), t0, units = "mins")),
                format(n_footprint, big.mark = ","),
                format(n_cell, big.mark = ","), 100 * n_footprint / n_cell))

n_valid_recorded <- as.numeric(pred_cfg$n_valid[1])
check_that(
  "b1_04", "footprint covers what 05 wrote",
  n_footprint >= n_valid_recorded && n_footprint <= n_cell,
  sprintf("footprint %s >= 05's %s valid pixels (the %s difference is the rim where a patch does not fit)",
          format(n_footprint, big.mark = ","),
          format(n_valid_recorded, big.mark = ","),
          format(n_footprint - n_valid_recorded, big.mark = ",")))

# OVERSAMPLED ON PURPOSE. terra::spatSample(method = "regular", na.rm = TRUE)
# lays a lattice over the whole EXTENT and then drops the NA cells, so a request
# for 5,000 over a footprint covering a fifth of the grid returns about a
# thousand. Asking for target / fraction, with a margin, is what makes the
# realised sample the size that was reasoned about. The count that comes back is
# reported rather than assumed, and refused if it is too small.
oversample <- 1.15
n_request  <- as.integer(ceiling(target_predpoints * (n_cell / n_footprint) *
                                 oversample))
message(sprintf("Drawing a regular sample: asking for %s to land ~%s inside the footprint",
                format(n_request, big.mark = ","),
                format(target_predpoints, big.mark = ",")))
predpoints <- prediction_sample(footprint, size = n_request)
n_drawn <- nrow(predpoints)

# THE DRAW IS VERIFIED AGAINST THE FOOTPRINT, NOT TRUSTED TO IT.
#
# prediction_sample() passes na.rm = TRUE to terra::spatSample(), but it also
# passes values = FALSE -- and whether na.rm is honoured when no values are
# returned has changed between terra versions. If it is not, this sample is a
# lattice over the whole extent, three quarters of it ocean, and Gj becomes the
# distance from open water to the nearest soil profile. That number would look
# entirely plausible: larger than the CV distances, in metres, with a sensible
# spread. Nothing downstream could catch it.
#
# So every drawn point is looked up in the footprint raster. Points outside it
# are dropped HERE, with the count printed -- an explicit repair rather than a
# silent one, and it costs one extract over a few thousand cells.
cell_value <- terra::extract(footprint,
                             as.matrix(predpoints[, c("x", "y")]))
# The LAST column, because extract() returns an ID column for some input shapes
# and not for others; the layer is always last. Written to survive a bare vector
# too, so a terra release that simplifies the return value does not turn this
# guard into the error it was meant to prevent.
#
# BRACED, and not for looks. R closes `x <- if (cond) expr` at the end of the
# line, so a bare `else` starting the next one is a syntax error at top level --
# it is only legal inside an open delimiter. This exact line failed
# test_sources_parse.R with "'else' inesperado", which is the check that exists
# for it.
cell_value <- if (is.null(dim(cell_value))) {
  cell_value
} else {
  as.data.frame(cell_value)[[ncol(cell_value)]]
}
n_outside  <- sum(is.na(cell_value))
if (n_outside > 0L) {
  message(sprintf(
    "  %s of %s drawn points fell outside the footprint and were dropped",
    format(n_outside, big.mark = ","), format(n_drawn, big.mark = ",")))
  predpoints <- predpoints[!is.na(cell_value), , drop = FALSE]
}

# THINNED, NOT REDRAWN, IF TERRA OVERSHOOTS. Some terra versions compensate for
# the NA cells internally and return close to `size`; that would hand CAST four
# times the predpoints asked for. Thinning a regular sample regularly keeps the
# spread and is deterministic, where a random subsample would put a second seed
# between this script and its own result.
if (nrow(predpoints) > target_predpoints) {
  keep <- unique(round(seq(1, nrow(predpoints), length.out = target_predpoints)))
  predpoints <- predpoints[keep, , drop = FALSE]
}
check_that(
  "b1_05", "predpoints are inside the footprint",
  nrow(predpoints) >= min_predpoints && n_outside < n_drawn,
  sprintf("%d drawn, %d outside the footprint, %d kept (asked %d, floor %d)",
          n_drawn, n_outside, nrow(predpoints), n_request, min_predpoints))
if (!.b1_checks[["b1_05"]]$ok) {
  stop("Only ", nrow(predpoints), " prediction points are usable. ",
       "Raise `oversample` above, or check that the footprint is not almost ",
       "empty -- W is estimated from the tail of this distribution.",
       call. = FALSE)
}

# ── 4. The kNNDM plan ─────────────────────────────────────────────────────────
#
# THROUGH THE FRONT END, NOT THROUGH knndm_folds() DIRECTLY.
#
# B1 is `knndm_cv()` on the real points, and the spec -> plan path is where this
# framework can be wrong on its own: resolve_resampling() switches on
# spec$kind, and a switch() on a non-character EXPR selects by POSITION -- which
# is how a spatial_cv() request once came back as a holdout with no error
# (R/api.R:124-139). tests/test_knndm.R checks that routing on 320 synthetic
# points; this checks it on the real ones.
message("\nRunning CAST::knndm on ", length(pool_pos), " pooled points against ",
        nrow(predpoints), " prediction points...")
t0 <- Sys.time()
cv_knndm <- knndm_cv(k = k_folds, predpoints = predpoints,
                     hold_out_test = FALSE, crs = 4326, project_to = moll,
                     seed = knndm_seed)
plan_knndm <- resolve_resampling(cv_knndm, data, test_ids = frozen_test,
                                 verbose = TRUE)
knndm_minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
message(sprintf("  kNNDM built the plan in %.2f min", knndm_minutes))

message("\n-- The kNNDM plan --")
print(plan_knndm)

# ── 5. Is the plan sane? ──────────────────────────────────────────────────────
#
# check_fold_plan() is the framework's own proof and is used rather than
# re-implemented: a second implementation here would share whatever
# misunderstanding the first has. What is added around it is the properties it
# deliberately does NOT assert -- completeness of the pool, and the test set's
# separation across ALL folds rather than within each.

sizes_knndm <- check_fold_plan(plan_knndm, meta = meta)
sizes_block <- check_fold_plan(plan_block, meta = meta)

# Printed side by side because the SHAPE of the two plans is the first thing a
# reader wants and the last thing a summary statistic shows. The block plan's
# training sets are smaller than the pool minus one fold -- that gap is the
# buffer, and seeing it here stops it being read as a defect further down.
message("\n-- Fold sizes --")
dplyr::bind_rows(
  sizes_knndm %>% dplyr::mutate(plan = "knndm", .before = 1),
  sizes_block %>% dplyr::mutate(plan = plan_block$method, .before = 1)
) %>%
  print_wide(n = Inf)

check_that(
  "b1_06", "check_fold_plan accepts the plan",
  is.data.frame(sizes_knndm) && nrow(sizes_knndm) == k_folds,
  sprintf("%d fold(s) described, train %s, validation %s",
          nrow(sizes_knndm), paste(sizes_knndm$n_train, collapse = "/"),
          paste(sizes_knndm$n_validation, collapse = "/")))

check_that(
  "b1_07", "every fold has train and validation",
  all(sizes_knndm$n_train > 0L) && all(sizes_knndm$n_validation > 0L),
  sprintf("smallest fold: %d train, %d validation",
          min(sizes_knndm$n_train), min(sizes_knndm$n_validation)))

# EVERY POOLED POINT VALIDATED EXACTLY ONCE. check_fold_plan() refuses
# DUPLICATES but says nothing about completeness, and the difference is not
# academic here: the block plan's buffer drops validation points as well as
# training points (apply_buffer() protects validation against test), so its
# folds cover fewer than the pool. kNNDM has no buffer -- by design,
# R/knndm.R:39-50 -- so its folds must cover the pool exactly. If they do not,
# something dropped points silently. The block plan's own shortfall is reported
# in the same line rather than asserted, because for that plan it is the buffer
# working, not a fault.
val_all_knndm <- sort(unlist(lapply(plan_knndm$folds, function(f) f$validation)))
val_all_block <- sort(unlist(lapply(plan_block$folds,  function(f) f$validation)))
check_that(
  "b1_08", "folds partition the pool",
  identical(as.integer(val_all_knndm), as.integer(sort(pool_pos))),
  sprintf("kNNDM validates %d of %d pooled points, %d duplicated | the block plan validates %d (its buffer drops %d)",
          length(unique(val_all_knndm)), length(pool_pos),
          sum(duplicated(val_all_knndm)), length(unique(val_all_block)),
          length(pool_pos) - length(unique(val_all_block))))

check_that(
  "b1_09", "no point holds two roles in a fold",
  all(vapply(plan_knndm$folds, function(f) {
    length(intersect(f$train, f$validation)) == 0L &&
      length(intersect(f$test, c(f$train, f$validation))) == 0L
  }, logical(1))),
  sprintf("checked %d fold(s) for train/validation/test overlap", k_folds))

# ACROSS the folds, not within one. Within one fold is check_fold_plan()'s job;
# a test point that trains in fold 2 and is held out in fold 1 would pass that
# and still be a read of the test set.
test_leak <- sum(unlist(lapply(plan_knndm$folds, function(f)
  length(intersect(test_pos, c(f$train, f$validation))))))
check_that(
  "b1_10", "test set is out of every fold",
  test_leak == 0L &&
    all(vapply(plan_knndm$folds, function(f)
      setequal(f$test, test_pos), logical(1))),
  sprintf("%d test point(s) found in any training or validation set; %d held out in every fold",
          test_leak, length(test_pos)))

check_that(
  "b1_11", "plan projected the way this script does",
  identical(plan_knndm$params$projection, moll),
  sprintf("plan used '%s'", plan_knndm$params$projection))

# ── 6. What kNNDM bought: the distributions behind W ──────────────────────────
#
# The coordinates are projected ONCE, here, and every distance below comes from
# that one matrix. Projecting inside each comparison would be three chances to
# pass a different CRS and get three sets of metres that are not the same metre.
xy_proj <- project_xy(meta$x, meta$y, crs = 4326, to = moll)
pp_proj <- project_xy(predpoints$x, predpoints$y, crs = 4326, to = moll)

# Nearest-neighbour distance from each query point to the nearest reference
# point, planar, in metres. FNN's kd-tree is the same one _measure_knndm.R
# timed on these coordinates; k = 1 because the query set is never the
# reference set here, so a point is never its own neighbour.
nnd <- function(query_xy, reference_xy) {
  FNN::get.knnx(data = reference_xy, query = query_xy, k = 1L)$nn.dist[, 1]
}

# Gj -- THE REFERENCE DISTRIBUTION, AND IT IS PLAN-INDEPENDENT.
#
# From every prediction pixel to the nearest point of the POOL. The pool and not
# a fold's training set, because the map is drawn by the model stage 04 refits
# on ALL the non-test data: that is the sample a prediction pixel will really be
# near or far from. Using a fold's training set here would compare each plan
# against a different target and make the two W values incomparable, which is
# precisely the mistake this comparison exists to avoid.
d_pred <- nnd(pp_proj, xy_proj[pool_pos, , drop = FALSE])

# Gij -- one value per validation point, pooled over folds, per plan. The
# training set used is the fold's ACTUAL training set: post-buffer for the block
# plan, because that is what the model was fitted on. Comparing kNNDM against an
# unbuffered version of the block plan would flatter kNNDM by measuring a plan
# nobody ran.
cv_distances <- function(plan) {
  unlist(lapply(plan$folds, function(f)
    nnd(xy_proj[f$validation, , drop = FALSE],
        xy_proj[f$train, , drop = FALSE])), use.names = FALSE)
}
d_cv_knndm <- cv_distances(plan_knndm)
d_cv_block <- cv_distances(plan_block)

# W1 BETWEEN TWO EMPIRICAL DISTRIBUTIONS, RECOMPUTED HERE.
#
# W1 = integral of |F - G| over distance = integral of |F^-1 - G^-1| over
# probability. The second form is used because the two samples have different
# sizes (3,137 validation points against ~5,000 prediction points) and a common
# probability grid needs no binning decision, where a common distance grid would
# need one and the answer would depend on it.
#
# THIS IS RECOMPUTED RATHER THAN READ FROM THE PLAN ON PURPOSE. CAST reports W
# for its own plan and reports nothing for the block plan, so the block number
# has to come from somewhere; computing BOTH the same way is what makes them
# comparable. The check below then asks whether the number computed here agrees
# with CAST's -- the summary and the distributions behind it have to be the same
# thing, and if they are not, one of them is measuring something else.
w1 <- function(a, b, n_grid = 2000L) {
  p <- (seq_len(n_grid) - 0.5) / n_grid
  mean(abs(stats::quantile(a, p, names = FALSE, type = 7) -
           stats::quantile(b, p, names = FALSE, type = 7)))
}

w_knndm <- w1(d_cv_knndm, d_pred)
w_block <- w1(d_cv_block, d_pred)
w_cast  <- as.numeric(plan_knndm$params$W)

ratio_cast <- if (is.finite(w_cast) && w_cast > 0) w_knndm / w_cast else NA_real_
check_that(
  "b1_12", "CAST's W is reproducible from the data",
  is.finite(w_cast) && is.finite(ratio_cast) &&
    ratio_cast > 0.2 && ratio_cast < 5,
  sprintf("CAST reports W = %.0f m, recomputed from the two distributions W1 = %.0f m (ratio %.2f)",
          w_cast, w_knndm, ratio_cast))

# THE QUESTION B1 EXISTS TO ANSWER.
#
# A FAILURE HERE IS NOT A BUG. It means that on THIS point set, against THIS
# prediction area, the block folds already imitate prediction as well as kNNDM
# does -- which is a real and publishable answer, and exactly what R/knndm.R
# predicts can happen: "when the samples are well spread over the prediction
# area, kNNDM converges by itself to ordinary random k-fold". The verdict below
# says so rather than implying the code is broken.
check_that(
  "b1_13", "kNNDM matches prediction better than blocks",
  w_knndm < w_block,
  sprintf("W1 kNNDM %.1f km vs blocks %.1f km (%.0f%% of the block value)",
          w_knndm / 1000, w_block / 1000,
          100 * w_knndm / max(w_block, .Machine$double.eps)))

# -- the distributions themselves ---------------------------------------------
#
# W is one number and this project's standing lesson is that one number hides a
# wrong answer. These are the quantiles it summarises: read the median against
# the median to see whether validation sits closer to training than prediction
# does, and the tail against the tail to see whether the folds ever reproduce
# the far end of the map at all.
distance_summary <- function(label, d) {
  q <- stats::quantile(d, probs = c(0, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90,
                                    0.95, 1), names = FALSE, type = 7)
  tibble::tibble(
    source      = label,
    n           = length(d),
    # Below one metre is "the same place": the training grid is 250 m and these
    # points are stored to seven significant digits of a degree. Counted rather
    # than tested for exact zero, because an equality on a kd-tree distance is
    # a promise about floating point that nothing here needs to make.
    n_under_1m  = sum(d < 1),
    min_km      = q[1] / 1000,
    q05_km      = q[2] / 1000,
    q10_km      = q[3] / 1000,
    q25_km      = q[4] / 1000,
    median_km   = q[5] / 1000,
    q75_km      = q[6] / 1000,
    q90_km      = q[7] / 1000,
    q95_km      = q[8] / 1000,
    max_km      = q[9] / 1000,
    mean_km     = mean(d) / 1000)
}

distances <- dplyr::bind_rows(
  distance_summary("prediction -> nearest pool point (Gj, the target)", d_pred),
  distance_summary("validation -> nearest train, kNNDM folds", d_cv_knndm),
  distance_summary("validation -> nearest train, block folds",  d_cv_block))

message("\n-- Nearest-neighbour distance distributions, km --")
print_wide(distances, n = Inf)

# CO-LOCATED POINTS, WHICH kNNDM DOES NOT ADDRESS AND DOES NOT CLAIM TO.
#
# R/knndm.R:39-50 states this outright: kNNDM has no buffer because prediction
# really does happen next to training points, and what a buffer was ALSO
# catching -- two profiles in the same raster cell, identical covariates and
# different targets -- is a defect under any plan and is handled elsewhere.
# Measured here because on this point set it is not a footnote: the number below
# is how many of the 3,728 rows share their exact coordinate with another row.
coord_key <- paste(meta$x, meta$y, sep = "_")
n_colocated <- sum(coord_key %in% coord_key[duplicated(coord_key)])
message(sprintf(
  "\nCo-located rows in the store: %s of %s share an exact coordinate with another row.",
  format(n_colocated, big.mark = ","), format(n_points, big.mark = ",")))
message(sprintf(
  "  validation points with a training point closer than 1 m: kNNDM %s, blocks %s",
  format(sum(d_cv_knndm < 1), big.mark = ","),
  format(sum(d_cv_block < 1), big.mark = ",")))
message("  A validation point at distance 0 is scored on an identical patch. ",
        "That is not\n  a kNNDM defect -- it is the trade the method makes ",
        "deliberately -- but it is the\n  number to weigh before adopting the ",
        "plan.")

# The same leakage criterion 03 reports, so a number here is comparable with a
# number there rather than being a second, incompatible notion of leakage. The
# windows come from the grid the run searched, not from a literal list here.
tune_grid_file <- file.path(cnn_run_dir, "tune_grid.rds")
if (!file.exists(tune_grid_file)) {
  stop("tune_grid.rds not found in ", cnn_run_dir, ". The leakage report is ",
       "expressed per window, and the windows have to come from the grid the ",
       "run actually searched -- a literal list here would go stale the first ",
       "time the grid changes.", call. = FALSE)
}
windows_grid <- sort(unique(as.integer(unlist(
  readRDS(tune_grid_file)$window_sizes))))

leak_knndm <- fold_leakage_report(plan_knndm, meta, cell_size = cell_size,
                                  windows = windows_grid)
message("\n-- Identical patches, train vs validation, per fold (kNNDM) --")
leak_knndm %>%
  dplyr::filter(matters) %>%
  print_wide(n = Inf)
message("\n-- Shared pixels at the widest window (context, not a defect) --")
leak_knndm %>%
  dplyr::filter(!matters, window == max(windows_grid)) %>%
  print_wide(n = Inf)

# ── 7. The report ─────────────────────────────────────────────────────────────

checks <- dplyr::bind_rows(.b1_checks)

geometry <- tibble::tibble(
  quantity = c("w1_knndm_km", "w1_block_km", "w_cast_metres",
               "n_predpoints_drawn", "n_predpoints_outside_footprint",
               "n_predpoints_used", "n_pool", "n_test", "k_folds",
               "n_footprint_cells", "n_valid_recorded_by_05",
               "n_colocated_rows", "cv_under_1m_knndm", "cv_under_1m_block",
               "median_gj_km", "median_cv_knndm_km", "median_cv_block_km",
               "knndm_minutes"),
  value    = c(w_knndm / 1000, w_block / 1000, w_cast,
               n_drawn, n_outside,
               nrow(predpoints), length(pool_pos), length(test_pos), k_folds,
               n_footprint, n_valid_recorded,
               n_colocated, sum(d_cv_knndm < 1), sum(d_cv_block < 1),
               stats::median(d_pred) / 1000,
               stats::median(d_cv_knndm) / 1000,
               stats::median(d_cv_block) / 1000,
               knndm_minutes))

safe_write_csv2(checks,    file.path(report_dir, "b1_checks.csv"))
safe_write_csv2(distances, file.path(report_dir, "b1_distances.csv"))
safe_write_csv2(geometry,  file.path(report_dir, "b1_geometry.csv"))

# THE PLAN IS SAVED SO THE rf RUN DOES NOT REBUILD IT.
#
# CAST::knndm is seeded and deterministic, but "should reproduce" and "is the
# same plan" are different claims, and the second is free here. The assignment
# goes out as a CSV beside it in the same shape 03 writes fold_assignment.csv,
# so it can be read without R.
safe_save_rds(plan_knndm, file.path(report_dir, "fold_plan.rds"))
safe_write_csv2(plan_knndm$assignment,
                file.path(report_dir, "fold_assignment.csv"))
safe_write_csv2(predpoints, file.path(report_dir, "predpoints.csv"))

message("\nWritten to: ", report_dir)

# ── 8. Verdict ────────────────────────────────────────────────────────────────
#
# THE VERDICT IS NOT "NOTHING FAILED".
#
# An empty ledger has nothing failing in it, and so does a ledger whose
# expensive check never ran. Both have happened in this project -- tests/helper.R
# carries the same guard and says why. So every promised check must be PRESENT
# and TRUE.
missing_checks <- setdiff(required_checks, checks$id)
failed_checks  <- checks$id[!checks$ok]

message("\n-- Checks --")
print_wide(checks, n = Inf)

verdict <- length(missing_checks) == 0L && length(failed_checks) == 0L

message("\n", strrep("=", 78))
message(sprintf("B1: %s | %d of %d checks present and true | W1 kNNDM %.1f km vs blocks %.1f km",
                if (verdict) "PASS" else "FAIL",
                sum(checks$ok), length(required_checks),
                w_knndm / 1000, w_block / 1000))
message(strrep("=", 78))

if (length(missing_checks) > 0L) {
  message("\nCHECKS THAT NEVER RAN: ", paste(missing_checks, collapse = ", "))
  message("  A check that did not run is not a check that passed. Something ",
          "above returned\n  early, or an id was renamed in one place and not ",
          "the other.")
}
if (length(failed_checks) > 0L) {
  message("\nFAILED: ", paste(failed_checks, collapse = ", "))
  message("  b1_13 is the one whose failure is a FINDING rather than a fault: ",
          "it means the\n  block folds already imitate prediction as well as ",
          "kNNDM does on these points,\n  which is what R/knndm.R predicts ",
          "happens when the samples are well spread over\n  the prediction ",
          "area. Every other id failing means the plan is malformed.")
}

# HOW TO READ THE ANSWER, WHICHEVER WAY IT WENT.
message("\n-- What this measured --")
message(sprintf(
  "  A prediction pixel is typically %.1f km from the nearest training point.",
  stats::median(d_pred) / 1000))
message(sprintf(
  "  A kNNDM validation point is %.1f km from its nearest training point;",
  stats::median(d_cv_knndm) / 1000))
message(sprintf(
  "  a block validation point is %.1f km.",
  stats::median(d_cv_block) / 1000))
message("  The plan whose median sits closer to the first line is scoring the ",
        "model on\n  the job it will actually be given. W1 is that comparison ",
        "over the whole\n  distribution rather than at the median alone.")

# ══════════════════════════════════════════════════════════════════════════════
# THE rf RUN ON THESE FOLDS IS A SEPARATE, LATER STEP -- NOT DONE HERE
#
# docs/test_plan.md, B1: "minutes to build the plan; a full `rf` run on it
# after". This script is the first half and trains nothing, on purpose:
#
#   * the two halves answer different questions. This one asks whether the plan
#     is well formed and whether its geometry matches prediction, which is a
#     property of the points and is TRUE OR FALSE. The rf run asks what the
#     model scores under it, which is a number with a spread around it, and
#     mixing the two produces a report where a fold-construction bug and a
#     disappointing CCC look alike;
#   * a script that trains cannot be re-run freely, and this one should be --
#     changing the predpoint sample or k is how the geometry gets understood;
#   * the plan above is on disk, so the rf run costs no kNNDM time at all.
#
# When it is run, it belongs in _capability_sweep.R beside the other capability
# rows, or in a script of its own. The call is the same one 03b makes, with the
# plan read from disk instead of a spec -- resolve_resampling() returns a
# fold_plan unchanged (R/api.R:240), which is what makes that substitution exact:
#
#   plan_knndm <- readRDS(file.path(
#     "D:/usuario_armazenamento/cassio/R/deep_learning_caret/outputs/tuning",
#     "soc_stock_modeling/soc_stock_0_5cm/capability_sweep/b1_knndm",
#     "fold_plan.rds"))
#   fit <- dsm_train(data, model = "rf", resampling = plan_knndm, ...)
#
# ONE THING THAT MUST CHANGE WHEN IT IS RUN: the `data` this script loads holds
# ONE window, because fold geometry needs coordinates and nothing else. An rf
# run with features = "window_mean" reads every window the store was loaded
# with, so it needs its own dsm_load() with the windows the 03 grid used. Left
# as it is, the forest would quietly be fitted on a third of the context and the
# comparison with 03b's rf would be a comparison of two different models.
#
# AND THE NUMBER IT RETURNS IS NOT COMPARABLE WITH 03's. Different folds mean a
# different definition of the score, which is the entire point of building them
# -- so the comparison worth making is rf-on-kNNDM against rf-on-blocks, both
# run here, and never rf-on-kNNDM against the CNN numbers already in the run
# directory.
# ══════════════════════════════════════════════════════════════════════════════
