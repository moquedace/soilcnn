# Smoke test: the front end, on a real store, training a real model
#
# test_api.R checks the specs and the "auto" arguments against a hand-built
# dsm_data. Nothing there opens a store or fits anything, so two things stayed
# uncovered -- and they are the two the pipeline is about to depend on:
#
#   dsm_load()   reads a store from DISK, aligns points to it, reads a
#                resolution, and applies the lock
#   dsm_train()  dispatches to run_cnn_resample() for patch models and to
#                run_table_resample() for tabular ones
#
# A wrapper is exactly the kind of code that looks obviously right and passes
# the wrong argument. This trains a two-epoch CNN on 64 synthetic points to
# find out.
#
# Verified:
#   1. dsm_load() opens a store, aligns the points, and reports what it has
#   2. the lock fires through the front door -- a wrong predictor set is
#      refused by dsm_load(), not minutes later
#   3. dsm_train() on a patch model produces what run_cnn_resample() produces:
#      the same comparison shape, a fold plan on disk, checkpoints
#   4. dsm_train() on a tabular model goes down the other path and produces the
#      SAME comparison shape -- which is what makes families comparable
#   5. the plan dsm_train() used is the plan it was given
#   6. a run is resumable through the front door
#
# Run: source("D:/.../tests/test_api_run.R")   (trains; ~1 min on CPU)

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)
  library(readr)
})

# -- project root: works under source() in the console AND under Rscript ------

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
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
.load_framework(root)

ok <- c()

# =============================================================================
# A store on disk, with the spec fields stage 02 records
# =============================================================================

set.seed(20260915)
n_pts <- 64L; n_ch <- 3L; win <- 3L
preds <- paste0("p", seq_len(n_ch))
CELL  <- 0.00224579811173295

# Four clusters, so a spatial plan has blocks to work with.
site <- rep(1:4, each = n_pts / 4L)
x <- (site %% 2L) * 2 + stats::runif(n_pts, 0, 0.05)
y <- (site %/% 3L) * 2 + stats::runif(n_pts, 0, 0.05)

arr <- array(stats::rnorm(n_pts * n_ch * win * win),
             dim = c(n_pts, n_ch, win, win))
centre <- (win + 1L) %/% 2L
y_true <- 2 * arr[, 1, centre, centre] + stats::rnorm(n_pts, sd = 0.5)
y_true <- y_true - min(y_true) + 1

meta <- tibble::tibble(
  profile_id       = seq_len(n_pts),
  sample_id        = seq_len(n_pts),
  x = x, y = y,
  target_native    = y_true,
  target_transform = log1p(y_true)
)

# Cleaned on ENTRY, not only on exit: tempdir() survives between source()s in
# one session, and a run that resumes its own previous output is not a test.
store_dir <- file.path(tempdir(), "dlc_api_store")
out_root  <- file.path(tempdir(), "dlc_api_out")
unlink(store_dir, recursive = TRUE)
unlink(out_root,  recursive = TRUE)
dir.create(store_dir, recursive = TRUE, showWarnings = FALSE)

invisible(save_patch_window(arr, store_dir, win))
readr::write_csv2(meta, file.path(store_dir, "patch_meta.csv"))
# The manifest carries the spec, exactly as stage 02 writes it -- otherwise the
# lock has nothing to check and this test would pass on a store that could not
# be locked at all.
saveRDS(tibble::tibble(
  scaling_applied      = FALSE,
  predictor_cols_final = paste(preds, collapse = ";"),
  n_channels           = n_ch,
  windows_extracted    = as.character(win),
  n_points_valid       = n_pts,
  target_col           = "soc_stock",
  target_transform     = "log1p",
  cell_size            = CELL
), file.path(store_dir, "patch_manifest.rds"))

points <- meta
for (j in seq_len(n_ch)) points[[preds[j]]] <- arr[, j, centre, centre]
type_table <- tibble::tibble(predictor = preds, is_dummy = FALSE,
                             is_percentage = FALSE)

# =============================================================================
# 1-2. dsm_load()
# =============================================================================

data <- suppressMessages(dsm_load(
  patch_dir = store_dir, points = points, type_table = type_table,
  target_col = "soc_stock", verbose = FALSE))

ok["dsm_load_returns_dsm_data"] <- inherits(data, "dsm_data")
ok["dsm_load_opened_the_store"]  <- data$store$n_channels == n_ch
ok["dsm_load_aligned_the_points"] <- nrow(data$points) == nrow(data$store$meta)
ok["dsm_load_kept_the_target_name"] <- identical(data$target_col, "soc_stock")
# cell_size was not passed and no raster table exists, so it must come from the
# manifest -- the weaker source, used only because it is better than none.
ok["dsm_load_recovers_cell_size_from_the_manifest"] <-
  isTRUE(all.equal(data$cell_size, CELL))

# The lock, through the front door. A store built for one predictor set, read
# by a script expecting another, trains and converges and maps the wrong thing.
ok["dsm_load_refuses_a_wrong_predictor_set"] <- inherits(
  tryCatch(dsm_load(store_dir, points,
                    tibble::tibble(predictor = c(preds, "extra")),
                    verbose = FALSE), error = function(e) e), "error")

ok["dsm_load_refuses_a_wrong_target"] <- inherits(
  tryCatch(dsm_load(store_dir, points, type_table,
                    target_col = "something_else", verbose = FALSE),
           error = function(e) e), "error")

# =============================================================================
# 3. dsm_train() on the patch model
# =============================================================================

grid <- make_manual_tune_grid(
  window_sizes  = list(c(win)),
  conv_channels = list(c(4L)),
  embedding_dim = 8L,
  base_lr       = 0.01,
  batch_size    = 16L,
  dropout       = 0.0,
  gate_type     = "no_gate_concat",
  use_residual  = FALSE,
  use_se_block  = FALSE
)

plan <- suppressMessages(resolve_resampling(
  spatial_cv(k = 2L, block_size = 1, buffer = "auto", test_frac = 0.2),
  data, verbose = FALSE))

ok["auto_buffer_used_the_stores_resolution"] <-
  isTRUE(all.equal(plan$params$buffer, win * CELL))

fit <- suppressMessages(dsm_train(
  data, model = "cnn", resampling = plan, tune_grid = grid,
  n_seeds = 1L, transform = expm1,
  output_dir = out_root, run_id = "api_cnn",
  device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
  n_epochs = 2L, patience = 2L, print_every = 100L, augment = FALSE,
  verbose = FALSE))

ok["dsm_train_returns_a_dsm_fit"] <- inherits(fit, "dsm_fit")
ok["dsm_train_records_the_model"]  <- identical(fit$model, "cnn")
ok["dsm_train_returns_the_data"]   <- inherits(fit$data, "dsm_data")

# The comparison table is the contract every family shares.
cmp <- fit$comparison
ok["cnn_comparison_has_the_core_columns"] <- all(
  c("unit_id", "config_id", "fold", "seed", "status", "val_ccc") %in% names(cmp))
ok["cnn_trained_every_unit"] <- nrow(cmp) == nrow(grid) * plan$n_folds
ok["cnn_units_succeeded"] <- all(cmp$status == "success")
ok["cnn_unit_id_shape"] <- all(grepl("_f\\d+_s\\d+$", cmp$unit_id))

run_dir <- file.path(out_root, "api_cnn")
ok["cnn_wrote_the_fold_plan"] <- file.exists(file.path(run_dir, "fold_plan.rds"))
ok["cnn_wrote_checkpoints"] <-
  length(list.files(file.path(run_dir, "models"), pattern = "_best\\.pt$")) ==
  nrow(cmp)

# 5. the plan it used is the plan it was given
ok["plan_on_disk_is_the_plan_given"] <- identical(
  readRDS(file.path(run_dir, "fold_plan.rds"))$folds, plan$folds)

# =============================================================================
# 3b. Nothing but a budget: the grid, the inverse and the threads come from
#     the store, the plan and the machine
#
# This store holds window 3 only. Before the default grid took the store's
# windows it drew 3/9/15 -- the SOC example's -- and this call stopped with
# "This grid needs window(s) 9, 15". Its folds train on 16 and 32 points, so
# every batch size of the old grid (128/256/512) would have taken NO gradient
# step. And with no transform given, the inverse must be the store's: the
# manifest says log1p, so each prediction must be expm1 of what the network
# output. clamp is opened for that check -- clamped at zero, a network whose
# outputs are all negative would give 0 under either inverse and prove
# nothing.
# =============================================================================

# Two threads first, so that finding one afterwards means n_cores set it --
# the runs above already left the session at one.
invisible(suppressMessages(set_torch_threads(2L)))
fit_def <- suppressMessages(dsm_train(
  data, model = "cnn", resampling = plan, tune_length = 2L,
  n_seeds = 1L, n_cores = 1L,
  output_dir = out_root, run_id = "api_cnn_default",
  n_epochs = 2L, patience = 2L, print_every = 100L, augment = FALSE,
  clamp = c(-Inf, Inf), verbose = FALSE))

cmp_def <- fit_def$comparison
ok["a_default_grid_trains_on_a_store_of_one_window"] <-
  nrow(cmp_def) == 2L * plan$n_folds && all(cmp_def$status == "success")
ok["the_default_grid_asked_only_for_the_stores_window"] <-
  all(as.character(cmp_def$window_sizes) == "3")
n_train_min <- min(vapply(plan$folds, function(f) length(f$train), integer(1)))
ok["the_default_grid_batches_fit_the_smallest_fold"] <-
  all(floor(n_train_min / cmp_def$batch_size) >= 4)
ok["n_cores_set_torchs_threads"] <- torch::torch_get_num_threads() == 1L
ok["dsm_train_keeps_the_clamp_it_was_given"] <- identical(fit_def$clamp, c(-Inf, Inf)) &&
  identical(readRDS(file.path(out_root, "api_cnn_default", "clamp.rds")), c(-Inf, Inf))

pred_def <- safe_read_csv2(file.path(out_root, "api_cnn_default", "predictions",
                                     paste0(cmp_def$unit_id[1], "_pred_all.csv")))
ok["with_no_transform_given_the_stores_inverse_is_applied"] <-
  nrow(pred_def) > 0L &&
  isTRUE(all.equal(pred_def$pred, expm1(pred_def$pred_transform))) &&
  !isTRUE(all.equal(pred_def$pred, pred_def$pred_transform))

# A batch no fold can fill is refused before the first unit, and named.
big <- grid
big$batch_size <- 64L
msg_big <- tryCatch({
  suppressMessages(dsm_train(
    data, model = "cnn", resampling = plan, tune_grid = big, n_seeds = 1L,
    output_dir = out_root, run_id = "api_cnn_big_batch",
    device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
    n_epochs = 2L, verbose = FALSE))
  ""
}, error = function(e) conditionMessage(e))
ok["a_batch_no_fold_can_fill_is_refused_before_training"] <-
  grepl("no gradient step", msg_big) && grepl(grid$config_id[1], msg_big) &&
  !dir.exists(file.path(out_root, "api_cnn_big_batch", "models"))

# =============================================================================
# 4. dsm_train() on a tabular model -- the OTHER path, the SAME table
# =============================================================================

if (requireNamespace("randomForest", quietly = TRUE) ||
    requireNamespace("ranger", quietly = TRUE)) {

  rf_grid_small <- rf_grid(1L, seed = 3L)
  rf_grid_small$n_trees <- 20L

  fit_rf <- suppressWarnings(suppressMessages(dsm_train(
    data, model = "rf", resampling = plan, tune_grid = rf_grid_small,
    features = c("centre", "window_mean"),
    n_seeds = 1L, transform = expm1,
    output_dir = out_root, run_id = "api_rf", verbose = FALSE)))

  ok["rf_went_down_the_table_path"] <- identical(fit_rf$model, "rf")
  ok["rf_trained_every_unit"] <- nrow(fit_rf$comparison) == plan$n_folds

  # THE POINT OF THE WHOLE REGISTRY: two families, one table shape, so
  # summarise_resamples() / seed_noise_floor() / one_se() work across both.
  shared <- c("unit_id", "config_id", "fold", "seed", "status",
              "val_ccc", "val_mae", "test_ccc")
  ok["both_families_share_the_comparison_shape"] <-
    all(shared %in% names(cmp)) && all(shared %in% names(fit_rf$comparison))

  ok["summarise_works_on_the_table_family"] <-
    nrow(summarise_resamples(fit_rf$comparison)) == nrow(rf_grid_small)
}

# =============================================================================
# 6. Resume through the front door
# =============================================================================

fit2 <- suppressMessages(dsm_train(
  data, model = "cnn", resampling = plan, tune_grid = grid,
  n_seeds = 1L, transform = expm1,
  output_dir = out_root, run_id = "api_cnn",
  device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
  n_epochs = 2L, patience = 2L, print_every = 100L, augment = FALSE,
  resume = TRUE, verbose = FALSE))

# Nothing retrained, so the numbers are byte-identical to the first run.
ok["resume_changed_nothing"] <- isTRUE(all.equal(
  fit$comparison$val_ccc, fit2$comparison$val_ccc))
ok["resume_kept_the_unit_count"] <- nrow(fit2$comparison) == nrow(cmp)

unlink(store_dir, recursive = TRUE)
unlink(out_root,  recursive = TRUE)

cat(sprintf("  front end            : dsm_load -> dsm_train on %d points, %d unit(s)\n",
            n_pts, nrow(cmp)))
cat(sprintf("  auto buffer          : %.8f  (%d px x %.8f)\n",
            plan$params$buffer, win, CELL))

.report(ok, "test_api_run")
