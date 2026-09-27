# ══════════════════════════════════════════════════════════════════════════════
# U2 -- the deployed config's cross-validated residuals under kNNDM folds
#
# THE QUESTION.
#
# The map's interval is calibrated on cross-validated residuals, and which
# folds produced them decides which job the interval is honest for (see the
# note above cv_residuals() in R/conformal.R). B1 measured the two designs on
# these points, as the median distance from a point being scored to the
# nearest training point:
#
#   what the map actually does   824 km
#   kNNDM folds                  837 km
#   block folds                   16 km
#
# Block-fold residuals describe predicting NEAR other profiles; most of the map
# is predicted FAR from any. An interval calibrated on them is honest for the
# first job and too narrow for the second -- U1 already shows it 2-3 points
# short on the test set, which is itself drawn near the profiles. Residuals
# from kNNDM folds are the ones exchangeable with the map's pixels.
#
# WHAT IT DOES.
#
# The deployed configuration -- its exact hyperparameters, read from the final
# run -- trained under the kNNDM fold plan the C1 design run used (same frozen
# test set), 3 folds x 3 seeds, stage 03's tuning schedule. Nothing else: this
# is not a tuning run, it is the one config the map uses, cross-validated the
# other way.
#
# BEFORE TRAINING it looks for an IDENTICAL configuration in the C1 kNNDM run.
# A config_id is a label within one run -- the C1 run's cfg_003 is another
# architecture -- so the match is by every hyperparameter
# (cv_residuals_for_config()). If one is there with predictions, nothing is
# trained.
#
# Resumable: an interrupted run continues where it stopped.
#
# COST: 9 units, about 30 min at 15 threads.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_u2_knndm_residuals.R")
# ══════════════════════════════════════════════════════════════════════════════

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
source(file.path(project_root, "utils", "install_load_pkg.R"))
install_load_pkg(c("torch", "coro", "dplyr", "readr", "tibble", "purrr", "DescTools", "terra"))
source(file.path(project_root, "R", "load_all.R"))
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
data_dir     <- base("data", "processed")
patch_dir    <- base("outputs", "patches")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
c1_knndm_dir <- file.path(tuning_base, "soc_0_5cm_design_knndm")
u2_run_id    <- "soc_0_5cm_knndm_cfg003"
u2_dir       <- file.path(tuning_base, "capability_sweep", "u2_knndm")   # this script's own

# Stage 03's tuning schedule -- the residuals being compared were produced by it.
training_args <- list(
  n_epochs = 500L, patience = 60L, es_min_delta = 0.0005, warmup_start_lr = 1e-5,
  lr_plateau_factor = 0.5, lr_plateau_patience = 25L, lr_plateau_min_delta = 0.0005,
  min_lr = 1e-6, gradient_clip = 1.0, print_every = 10L, augment = TRUE)

final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
summ      <- readRDS(file.path(final_base, final_run_id, "comparison", "final_run_summary.rds"))
config_id <- selected_config_id(summ, final_run_id)
cfg_row   <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]

message("\n", strrep("=", 78))
message("U2 -- ", config_id, " of ", final_run_id, ", cross-validated under kNNDM folds")
message(strrep("=", 78))

required <- sprintf("u2_%02d", 1:5)
L <- check_ledger("U2")

plan <- readRDS(file.path(c1_knndm_dir, "fold_plan.rds"))
meta <- safe_read_csv2(file.path(patch_dir, "patch_meta.csv"))
split_tbl  <- safe_read_csv2(file.path(metadata_dir, "data_split.csv"))
frozen_test <- split_tbl$sample_id[split_tbl$role == "test"]
ledger_check(L, "u2_01", "the plan is the C1 design run's kNNDM plan",
             grepl("knndm", plan$method, ignore.case = TRUE),
             sprintf("%s | %d fold(s)", plan$method, plan$n_folds))
ledger_check(L, "u2_02", "its test set is the frozen one every design is scored on",
             setequal(meta$sample_id[plan$folds[[1]]$test], frozen_test),
             sprintf("%d test point(s)", length(frozen_test)))
ledger_check(L, "u2_03", "the grid row is the deployed configuration",
             nrow(cfg_row) == 1L && identical(cfg_row$config_id, config_id),
             paste0(config_id, ": ", .config_signature(cfg_row)))

# An identical configuration may already have been trained under these folds.
src <- NULL
if (dir.exists(c1_knndm_dir)) {
  if (!is.null(cv_residuals_for_config(c1_knndm_dir, cfg_row, required = FALSE))) {
    src <- c1_knndm_dir
    message("\nThe C1 kNNDM run already holds this configuration -- nothing to train.")
  }
}

if (is.null(src)) {
  data <- dsm_load(
    patch_dir    = patch_dir,
    points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
    type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
    raster_table = file.path(metadata_dir, "raster_table_used.csv"),
    windows      = sort(unique(unlist(cfg_row$window_sizes))),
    target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
    verbose      = FALSE)
  fit <- do.call(dsm_train, c(list(
    data = data, model = "cnn", resampling = plan, tune_grid = cfg_row,
    n_seeds = 3L, base_seed = 42L, output_dir = tuning_base, run_id = u2_run_id,
    resume = TRUE, verbose = FALSE), training_args))
  src <- fit$run_dir
  ledger_check(L, "u2_04", "every unit trained", {
    cmp <- fit$comparison
    list(ok = nrow(cmp) == 3L * plan$n_folds && all(cmp$status == "success"),
         measured = sprintf("%d of %d unit(s) succeeded", sum(cmp$status == "success"),
                            3L * plan$n_folds))
  })
} else {
  ledger_check(L, "u2_04", "every unit trained", TRUE, "not needed: the C1 run holds the config")
}

cv <- cv_residuals_for_config(src, cfg_row, required = FALSE)
ledger_check(L, "u2_05", "every non-test point has a kNNDM residual",
             !is.null(cv) && nrow(cv) == nrow(meta) - length(frozen_test),
             sprintf("%d residual(s) for %d non-test point(s) | from %s",
                     if (is.null(cv)) 0L else nrow(cv), nrow(meta) - length(frozen_test),
                     basename(src)))

create_output_dirs(u2_dir)
v <- ledger_verdict(L, required, file.path(u2_dir, "u2_checks.csv"))
if (v$pass) {
  message("\nThe kNNDM residuals are in ", src,
          "\nRun _u1_conformal_level_di.R again: it compares the two calibration sources.")
}
