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
  "randomForest"   # ranger is used instead when installed -- far faster here
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# One source() instead of ten, in a dependency order that is not guessable.
source(file.path(project_root, "R", "load_all.R"))

# ══════════════════════════════════════════════════════════════════════════════
# WHAT THIS SCRIPT IS FOR
#
# Until a baseline exists, "the CNN reached CCC 0.62" is a number with no scale
# on it. This script puts three models beside it, under THE SAME FOLDS:
#
#   rf_centre        Random Forest on the centre pixel.
#                    The classic digital soil mapping baseline. If the CNN does
#                    not beat this, nothing else in the project matters.
#
#   rf_context       Random Forest on the centre pixel PLUS the per-channel
#                    window means. The same neighbourhood the convolution sees,
#                    with the spatial ARRANGEMENT thrown away.
#
#   mlp_centre       A fully connected network on the centre pixel. Same
#                    optimiser, same loss, same early stopping as the CNN --
#                    so what separates them is the convolution and nothing
#                    else.
#
# THE GAP BETWEEN rf_context AND THE CNN IS WHAT THE CONVOLUTION IS WORTH.
#
# If they match, the convolution is doing averaging and its structure buys
# nothing. That is a falsifiable claim, it costs minutes to test, and knowing
# it is worth more than another point of CCC.
#
# READ THE GAP AGAINST THE NOISE FLOOR, NOT ON ITS OWN. A difference smaller
# than what re-seeding the same model produces is not a difference; the run
# prints both.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

set.seed(42)
torch::torch_manual_seed(42)

device <- setup_torch_device(n_threads = 30, use_cuda = TRUE)

# ── Paths ─────────────────────────────────────────────────────────────────────

# data/processed, NOT outputs/data. Stage 01 writes the point table there and
# every other script reads it there; this one said outputs/data and would have
# stopped on a file that exists, one directory away.
data_dir     <- file.path(project_root, "data", "processed",
                          "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
patch_dir    <- file.path(project_root, "outputs", "patches",
                          "soc_stock_modeling", target_label)
tuning_base  <- file.path(project_root, "outputs", "tuning",
                          "soc_stock_modeling", target_label)

# THE FOLD PLAN COMES FROM THE CNN RUN. It is not rebuilt here.
#
# Rebuilding it "the same way" makes the comparison depend on two call sites
# staying in step: one edited buffer, one changed k, and the two models are
# measured on different splits while every table still says they were not.
# Reading the plan the CNN actually used makes them identical by CONSTRUCTION.
#
# "latest" takes the most recent run; name a run_id to pin one.
cnn_run_id <- "latest"

if (identical(cnn_run_id, "latest")) {
  runs <- list.dirs(tuning_base, recursive = FALSE, full.names = FALSE)
  runs <- runs[file.exists(file.path(tuning_base, runs, "fold_plan.rds"))]
  if (length(runs) == 0L) {
    stop("No tuning run with a fold_plan.rds under: ", tuning_base,
         "\nRun 03_run_tuning.R first -- the baselines are measured on ITS ",
         "folds, not on folds invented here.")
  }
  cnn_run_id <- sort(runs, decreasing = TRUE)[1]
  message("cnn_run_id resolved to: ", cnn_run_id)
}

cnn_run_dir <- file.path(tuning_base, cnn_run_id)
plan <- readRDS(file.path(cnn_run_dir, "fold_plan.rds"))

# ── Store, points, types ──────────────────────────────────────────────────────
#
# The windows the CNN's grid used, so the window means summarise exactly the
# neighbourhoods the convolution saw.
cnn_grid       <- readRDS(file.path(cnn_run_dir, "tune_grid.rds"))
windows_needed <- sort(unique(unlist(cnn_grid$window_sizes)))
message("Windows from the CNN grid: ", paste(windows_needed, collapse = ", "))

# The same call stage 03 makes, including the same lock. A baseline measured
# against a store the run cannot serve is worse than no baseline: it is a wrong
# number with a comparison attached.
data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = windows_needed,
  target_col   = readr::read_csv2(file.path(metadata_dir, "target_config.csv"),
                                  show_col_types = FALSE)$target_col[1]
)

# ── Repetitions ───────────────────────────────────────────────────────────────
#
# The same number of seeds as the CNN run, so the two noise floors are
# estimated from the same amount of evidence. A baseline with one seed and a
# CNN with three gives the CNN an error bar and the baseline a point, and the
# comparison then reads as if only one of them were uncertain.
n_seeds   <- 3L
base_seed <- 42L

run_id <- paste0("baselines_", format(Sys.time(), "%Y%m%d_%H%M%S"))

# ══════════════════════════════════════════════════════════════════════════════
# THE THREE RUNS
# ══════════════════════════════════════════════════════════════════════════════

results <- list()

# -- 1. RF on the centre pixel ------------------------------------------------
message("\n", strrep("#", 78))
message("# rf_centre -- the classic DSM baseline")
message(strrep("#", 78))

results$rf_centre <- dsm_train(
  data       = data,
  model      = "rf",
  resampling = plan,
  features   = "centre",
  windows    = windows_needed,
  transform  = expm1,                 # inverse of the log1p the target carries
  output_dir = tuning_base,
  run_id     = file.path(run_id, "rf_centre"),
  base_seed  = base_seed, n_seeds = n_seeds,
  tune_length = 4L
)

# -- 2. RF on centre + window means -------------------------------------------
message("\n", strrep("#", 78))
message("# rf_context -- the neighbourhood, WITHOUT its arrangement")
message(strrep("#", 78))

results$rf_context <- dsm_train(
  data       = data,
  model      = "rf",
  resampling = plan,
  features   = c("centre", "window_mean"),
  windows    = windows_needed,
  transform  = expm1,
  output_dir = tuning_base,
  run_id     = file.path(run_id, "rf_context"),
  base_seed  = base_seed, n_seeds = n_seeds,
  tune_length = 4L
)

# -- 3. MLP on the centre pixel -----------------------------------------------
message("\n", strrep("#", 78))
message("# mlp_centre -- is it the architecture, or just the covariates?")
message(strrep("#", 78))

results$mlp_centre <- dsm_train(
  data       = data,
  model      = "mlp",
  resampling = plan,
  features   = "centre",
  windows    = windows_needed,
  transform  = expm1,
  output_dir = tuning_base,
  run_id     = file.path(run_id, "mlp_centre"),
  base_seed  = base_seed, n_seeds = n_seeds,
  tune_length = 6L,
  device     = device
)

# ══════════════════════════════════════════════════════════════════════════════
# THE COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

# The BEST config of each family, so every family is represented by the choice
# someone would actually make from it -- not by its average config, which is an
# average over a grid nobody would deploy.
best_of <- function(res, label) {
  if (nrow(res$by_config) == 0L) return(tibble::tibble())
  res$by_config %>%
    dplyr::slice(1) %>%
    dplyr::transmute(
      family    = label,
      config_id = .data$config_id,
      n_units   = .data$n_units,
      val_ccc   = .data$val_ccc_mean,
      val_ccc_sd = .data$val_ccc_sd,
      val_mae   = .data$val_mae_mean
    )
}

cnn_comparison <- file.path(cnn_run_dir, "comparison", "comparison_all.rds")
cnn_best <- if (file.exists(cnn_comparison)) {
  best_of(list(by_config = summarise_resamples(readRDS(cnn_comparison))), "cnn")
} else {
  message("\nNOTE: no CNN comparison table found -- the baselines stand alone.")
  tibble::tibble()
}

board <- dplyr::bind_rows(
  best_of(results$rf_centre,  "rf_centre"),
  best_of(results$rf_context, "rf_context"),
  best_of(results$mlp_centre, "mlp_centre"),
  cnn_best
) %>% dplyr::arrange(dplyr::desc(val_ccc))

message("\n", strrep("=", 78))
message("BEST OF EACH FAMILY, on identical folds")
message(strrep("=", 78))
print(board)

# THE ONE NUMBER THIS SCRIPT EXISTS FOR.
if (nrow(cnn_best) > 0L && "rf_context" %in% board$family) {
  gap <- cnn_best$val_ccc[1] -
         board$val_ccc[board$family == "rf_context"][1]
  nf  <- seed_noise_floor(readRDS(cnn_comparison))
  message(sprintf("\nCNN - rf_context = %+.4f CCC", gap))
  # median_sd, not max_range: the TYPICAL spread between seeds of one
  # (config, fold), which is the scale a difference between families has to
  # clear before it means anything. The max would set the bar at the unluckiest
  # config in the grid and declare every real difference invisible.
  if (!is.null(nf) && is.finite(nf$median_sd)) {
    message(sprintf("Seed noise floor (CNN)   = %.4f  (over %d config x fold)",
                    nf$median_sd, nf$n_comparable))
    if (abs(gap) < nf$median_sd) {
      message("\n-> THE GAP IS SMALLER THAN THE NOISE. On this evidence the ",
              "convolution's\n   spatial structure buys nothing that ",
              "per-channel window means do not.")
    } else if (gap > 0) {
      message("\n-> The convolution is worth more than the noise: the ",
              "ARRANGEMENT of the\n   neighbourhood carries signal, not just ",
              "its average.")
    } else {
      message("\n-> The context RF is ahead of the CNN by more than the ",
              "noise. Before\n   concluding anything about convolutions, ",
              "check the CNN's training\n   (early stopping, learning rate) ",
              "-- this is usually a fitting problem.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# THE SAME QUESTION, PAIRED
#
# The block above compares two means against the seed noise floor. That is a
# useful sanity bound, but it is not the right test, and it under-uses the
# design: every family ran on the SAME folds with the SAME seeds, so the units
# come in matched pairs. Most of the spread between units is fold difficulty,
# which both families feel together -- subtracting it out is free power.
#
# What comes out is a difference with a confidence interval. That is the honest
# form of the answer here, because the expected finding is "no difference", and
# a bare "not significant" over 9 pairs cannot distinguish "there is nothing"
# from "we could not see it". The interval says how large an effect the data
# still permit.
# ══════════════════════════════════════════════════════════════════════════════

message("
", strrep("=", 78))
message("PAIRED TESTS -- same folds, same seeds, difference per unit")
message(strrep("=", 78))

cnn_cmp <- if (file.exists(cnn_comparison)) readRDS(cnn_comparison) else NULL

# ONE CONFIG PER FAMILY, CHOSEN ONCE.
#
# paired_family_test() will otherwise pick the best config for whichever metric
# it is given, and the first run of this block did exactly that: the CCC
# comparison used rf_001 and the MAE comparison used rf_002. Both choices are
# defensible on their own, but together they mean "rf_context" names a
# different model on each line of the same report -- and the reader has no way
# to see it.
#
# A family is represented by the configuration someone would DEPLOY, and that
# decision is made once, on the selection metric, before any comparison. The
# board above already made it; this reuses it rather than re-deciding.
board_cfg <- stats::setNames(board$config_id, board$family)
cfg_of <- function(fam) if (fam %in% names(board_cfg)) unname(board_cfg[fam]) else NULL

pairs_to_test <- list(
  # The headline: structure vs the same neighbourhood with structure removed.
  list(a = cnn_cmp,                  b = results$rf_context,
       la = "cnn",        lb = "rf_context"),
  # Is it the architecture, or just the covariates? Same recipe, no convolution.
  list(a = cnn_cmp,                  b = results$mlp_centre,
       la = "cnn",        lb = "mlp_centre"),
  # And does the neighbourhood help a forest at all? If not, the window means
  # carry nothing and the first test was never going to show anything either.
  list(a = results$rf_context,       b = results$rf_centre,
       la = "rf_context", lb = "rf_centre")
)

for (metric in c("val_ccc", "val_mae")) {
  for (p in pairs_to_test) {
    if (is.null(p$a) || is.null(p$b)) next
    res <- tryCatch(
      paired_family_test(p$a, p$b, metric = metric,
                         config_a = cfg_of(p$la), config_b = cfg_of(p$lb),
                         label_a = p$la, label_b = p$lb),
      error = function(e) {
        message("  ", p$la, " vs ", p$lb, " (", metric, "): ",
                conditionMessage(e))
        NULL
      })
    if (!is.null(res)) print(res)
  }
}

# BOTH METRICS, DELIBERATELY. The first run ranked the families one way on CCC
# and the OPPOSITE way on MAE -- rf_context had the best MAE and the CNN the
# worst, while the CCC order was reversed. That is not a contradiction: CCC
# rewards spread agreement, MAE rewards typical closeness, and a model that
# stretches its predictions to match the observed variance buys CCC with MAE.
# Reporting one of them alone would have hidden that entirely.

board_path <- file.path(tuning_base, run_id, "family_board.csv")
safe_write_csv2(board, board_path)
message("\nBoard: ", board_path)

# ══════════════════════════════════════════════════════════════════════════════
# ADDING A FOURTH FAMILY: BORROWING IT FROM caret
#
# The three families above are hand-written because we want exact control over
# them -- ranger's native API is faster than going through caret, and the MLP
# has to use the CNN's own training recipe or it answers nothing.
#
# Everything ELSE should be borrowed. caret already carries, for ~230 methods,
# the parameter names, a grid generator that knows sensible ranges, the fit and
# predict closures, and which package to load. Re-deriving that per family is
# the work caret already did.
#
# What caret does NOT get to do is resample: the fold plan stays ours, and
# caret is called with trainControl(method = "none") and a one-row grid. See
# the header of R/caret_adapter.R.
#
# To see what is on offer:
#
#   caret_available("boost|forest|svm|glmnet")
#
# To add one -- three lines, and it behaves like every other family:
#
#   register_model(caret_spec("xgbTree"), overwrite = TRUE)
#   results$xgb <- dsm_train(
#     data       = data,
#     model      = "xgbTree",
#     resampling = plan,
#     features   = c("centre", "window_mean"),
#     windows    = windows_needed,
#     transform  = expm1,
#     output_dir = tuning_base,
#     run_id     = file.path(run_id, "xgb_context"),
#     base_seed  = base_seed, n_seeds = n_seeds,
#     tune_length = 6L
#   )
#
# tune_grid is left NULL on purpose: caret's generator needs the REAL training
# data (mtry is a fraction of ncol(x); glmnet's lambda path is computed from
# the values), so the runner builds the grid from fold 1's table rather than
# against a synthetic matrix of the right width.
#
# The model's own package must be installed -- caret Suggests them rather than
# depending on them. xgbTree needs xgboost, ranger needs ranger, and so on.
# ══════════════════════════════════════════════════════════════════════════════
