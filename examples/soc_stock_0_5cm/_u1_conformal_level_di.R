# ══════════════════════════════════════════════════════════════════════════════
# U1 -- the 90% interval: constant, following the level, following the level
# and the dissimilarity -- measured on the test set
#
# THE QUESTION.
#
# The map's interval today is constant: +/- 39.6 t/ha everywhere. Measured on
# 2026-09-26 it covered 94% of the lowest fifth of predictions and 77% of the
# highest -- right on average, wrong in particular. A scale fitted on the
# predicted level evened that out (89/92/88/83/88). The decision taken then was
# to fit the scale on the level AND the dissimilarity index (R/aoa.R), because
# how far a pixel is from the training data is the other thing that should
# predict how wrong the model is there -- and the one a map pixel and a
# calibration point can be measured on the same way.
#
# WHAT IS MEASURED.
#
# Three 90% intervals, all calibrated on the tuning run's cross-validated
# residuals for the selected config (every point predicted once as validation;
# the median over the seeds), and all checked on the TEST set -- 591 points
# neither the tuning nor the refit ever used:
#
#   constant   +/- q, as the map carries now
#   level      +/- q x (a + b x level)
#   level+DI   +/- q x (a + b x level + c x DI)
#
# The scaled ones are fitted on one half of the calibration points and q is
# taken on the other (R/conformal.R explains why that keeps the guarantee).
# A calibration point's DI is its CROSS-VALIDATED one -- the distance to the
# nearest point outside its fold, which is what the model that predicted it
# faced; a test point's is against the whole reference, as a map pixel's is.
#
# Coverage is reported overall, by fifth of the predicted level, by fifth of
# the DI, and inside / outside the area of applicability. The width is
# reported beside it: an interval that covers by being enormous has not been
# improved.
#
# AND WHICH RESIDUALS. When _u2_knndm_residuals.R has run, the same config's
# residuals under kNNDM folds -- validation at the distances the map predicts
# at -- calibrate a second pair of intervals beside the block ones. Each
# source is measured with its OWN DI reference and AOA threshold, built from
# its own fold plan: a calibration residual's DI is the distance its own fold
# model faced.
#
# WHAT IT DECIDES: which interval dsm_predict() writes as the 90% band, and
# from which residuals. Not a pass/fail -- the checks below only say the
# measurement is sound.
#
# COST: no rasters, no network. Under a minute.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_u1_conformal_level_di.R")
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
install_load_pkg(c("dplyr", "readr", "tibble", "purrr"))
pkgload::load_all(project_root)
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
data_dir     <- base("data", "processed")
patch_dir    <- base("outputs", "patches")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
u1_dir       <- file.path(tuning_base, "capability_sweep", "u1_conformal")
alpha        <- 0.1

final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir <- file.path(final_base, final_run_id)
summ      <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
config_id <- selected_config_id(summ, final_run_id)
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)

message("\n", strrep("=", 78))
message("U1 -- constant, level, level+DI: the 90% interval on the test set")
message(strrep("=", 78))
message("  final run  : ", final_dir, "  (", config_id, ")")
message("  tuning run : ", tuning_dir)
message(strrep("=", 78), "\n")

# ── The reference: what the model saw, in its own space ──────────────────────
rtable     <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
predictors <- as.character(rtable$predictor)
qc_table   <- safe_read_csv2(file.path(metadata_dir, "qc_table.csv"))
scaling    <- safe_read_csv2(file.path(final_dir, config_id, "predictor_scaling.csv"))
scaling    <- scaling[match(predictors, scaling$predictor), , drop = FALSE]
plan       <- readRDS(file.path(tuning_dir, "fold_plan.rds"))
meta       <- safe_read_csv2(file.path(patch_dir, "patch_meta.csv"))
points     <- align_points_to_meta(
  safe_read_csv2(file.path(data_dir, "full_modeling_dataset_raw.csv")), meta)
aref <- aoa_reference(points, predictors, qc_table, scaling, plan)
message(sprintf("Reference: %d point(s) x %d channel(s) | AOA threshold (DI) %.4f",
                aref$ref$n, aref$ref$p, as.numeric(aref$threshold)))

# ── The calibration set: cross-validated residuals, with their CV DI ─────────
cv  <- cv_residuals(tuning_dir, config_id)
cal <- dplyr::inner_join(cv, dplyr::select(aref$cv, sample_id, di = cv_di), by = "sample_id")

# ── The test set: the final ensemble's median, with each point's DI ──────────
ens <- safe_read_csv2(file.path(final_dir, config_id, "ensemble_predictions.csv"))
tst <- dplyr::filter(ens, dataset_role == "test")
tst$di <- aoa_di(aref, as.matrix(points[match(tst$sample_id, points$sample_id), predictors]))

required <- sprintf("u1_%02d", 1:4)
L <- check_ledger("U1")
ledger_check(L, "u1_01", "the calibration set is the selected config's cross-validated residuals",
             !is.null(cv) && nrow(cv) >= 500L,
             sprintf("%d point(s), median over %s seed(s)", if (is.null(cv)) 0L else nrow(cv),
                     if (is.null(cv)) "?" else paste(unique(cv$n_seeds), collapse = "/")))
ledger_check(L, "u1_02", "every calibration point has its cross-validated DI",
             !is.null(cv) && nrow(cal) == nrow(cv) && all(is.finite(cal$di)),
             sprintf("%d of %d", nrow(cal), if (is.null(cv)) 0L else nrow(cv)))
ledger_check(L, "u1_03", "every test point has a prediction and a DI",
             nrow(tst) > 0L && all(is.finite(tst$pred)) && all(is.finite(tst$di)),
             sprintf("%d test point(s)", nrow(tst)))

cal_const <- conformal_calibrate(cal$obs, cal$pred, alpha = alpha)
cal_level <- conformal_scaled_calibrate(cal$obs, cal$pred, data.frame(level = cal$pred),
                                        alpha = alpha)
cal_ldi   <- conformal_scaled_calibrate(cal$obs, cal$pred,
                                        data.frame(level = cal$pred, di = cal$di), alpha = alpha)
ledger_check(L, "u1_04", "the three intervals calibrated to a finite q",
             all(is.finite(c(cal_const$q, cal_level$q, cal_ldi$q))),
             sprintf("q: constant %.3f | level %.3f | level+DI %.3f",
                     cal_const$q, cal_level$q, cal_ldi$q))

cov_tst_level <- data.frame(level = tst$pred)
cov_tst_ldi   <- data.frame(level = tst$pred, di = tst$di)
iv <- list(
  constant   = conformal_interval(cal_const, tst$pred, lower_limit = 0),
  level      = conformal_scaled_interval(cal_level, tst$pred, cov_tst_level, lower_limit = 0),
  `level+DI` = conformal_scaled_interval(cal_ldi, tst$pred, cov_tst_ldi, lower_limit = 0))
aoa_of <- list(constant = aref$threshold, level = aref$threshold, `level+DI` = aref$threshold)

# ── The kNNDM source, when _u2 has produced it ───────────────────────────────
cfg_row <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
knndm_src <- NULL
for (cand in file.path(tuning_base, c("soc_0_5cm_knndm_cfg003", "soc_0_5cm_design_knndm"))) {
  if (dir.exists(cand) &&
      !is.null(suppressMessages(cv_residuals_for_config(cand, cfg_row, required = FALSE)))) {
    knndm_src <- cand
    break
  }
}
if (is.null(knndm_src)) {
  message("\nkNNDM source: none on disk serves ", config_id,
          " -- run _u2_knndm_residuals.R to compare the two sources.")
} else {
  aref_k <- aoa_reference(points, predictors, qc_table, scaling,
                          readRDS(file.path(knndm_src, "fold_plan.rds")))
  cal_k  <- dplyr::inner_join(cv_residuals_for_config(knndm_src, cfg_row),
                              dplyr::select(aref_k$cv, sample_id, di = cv_di), by = "sample_id")
  di_k   <- aoa_di(aref_k, as.matrix(points[match(tst$sample_id, points$sample_id), predictors]))
  cov_k  <- data.frame(level = tst$pred, di = di_k)
  kc_const <- conformal_calibrate(cal_k$obs, cal_k$pred, alpha = alpha)
  kc_ldi   <- conformal_scaled_calibrate(cal_k$obs, cal_k$pred,
                                         data.frame(level = cal_k$pred, di = cal_k$di),
                                         alpha = alpha)
  message(sprintf("\nkNNDM source: %s | %d residual(s) | AOA threshold (DI) %.4f",
                  basename(knndm_src), nrow(cal_k), as.numeric(aref_k$threshold)))
  message(sprintf("  kNNDM CV DI: median %.3f | q90 %.3f  (block: median %.3f | q90 %.3f)",
                  stats::median(cal_k$di), stats::quantile(cal_k$di, 0.9),
                  stats::median(cal$di), stats::quantile(cal$di, 0.9)))
  print(kc_ldi)
  iv[["constant, kNNDM"]] <- conformal_interval(kc_const, tst$pred, lower_limit = 0)
  iv[["level+DI, kNNDM"]] <- conformal_scaled_interval(kc_ldi, tst$pred, cov_k, lower_limit = 0)
  aoa_of[["constant, kNNDM"]] <- aref_k$threshold
  aoa_of[["level+DI, kNNDM"]] <- aref_k$threshold
}

fifth <- function(v) {
  br <- unique(stats::quantile(v, 0:5 / 5, na.rm = TRUE))
  cut(v, br, include.lowest = TRUE, labels = FALSE)
}
by_level <- fifth(tst$pred)
by_di    <- fifth(tst$di)
rows <- list()
for (nm in names(iv)) {
  in_aoa <- inside_aoa(tst$di, aoa_of[[nm]])
  hit <- tst$obs >= iv[[nm]]$lower & tst$obs <= iv[[nm]]$upper
  rows[[nm]] <- tibble::tibble(
    interval = nm,
    coverage = mean(hit),
    mean_width = mean(iv[[nm]]$width),
    by_level_fifth = paste(sprintf("%.0f", 100 * tapply(hit, by_level, mean)), collapse = "/"),
    by_di_fifth    = paste(sprintf("%.0f", 100 * tapply(hit, by_di, mean)), collapse = "/"),
    level_spread = diff(range(tapply(hit, by_level, mean))),
    di_spread    = diff(range(tapply(hit, by_di, mean))),
    inside_aoa   = mean(hit[in_aoa]),
    outside_aoa  = if (any(!in_aoa)) mean(hit[!in_aoa]) else NA_real_)
}
tab <- dplyr::bind_rows(rows)

v <- ledger_verdict(L, required, file.path(u1_dir, "u1_checks.csv"))

message("\n-- The fitted scales --")
print(cal_level)
print(cal_ldi)

message(sprintf("\n-- DI: calibration (cross-validated) vs test --\n  calibration: median %.3f | q90 %.3f\n  test       : median %.3f | q90 %.3f\n  test points outside the AOA: %d of %d",
                stats::median(cal$di), stats::quantile(cal$di, 0.9),
                stats::median(tst$di), stats::quantile(tst$di, 0.9),
                sum(!inside_aoa(tst$di, aref$threshold)), nrow(tst)))

message("\n-- The 90% interval on the test set (coverage in %, fifths low -> high) --")
print_wide(dplyr::mutate(tab, coverage = round(100 * coverage, 1),
                         mean_width = round(mean_width, 1),
                         level_spread = round(100 * level_spread, 1),
                         di_spread = round(100 * di_spread, 1),
                         inside_aoa = round(100 * inside_aoa, 1),
                         outside_aoa = round(100 * outside_aoa, 1)), n = Inf)

create_output_dirs(u1_dir)
safe_write_csv2(tab, file.path(u1_dir, "u1_coverage.csv"))
safe_write_csv2(tibble::tibble(
  interval = c("level", "level+DI"),
  intercept = c(cal_level$coef[[1]], cal_ldi$coef[[1]]),
  level = c(cal_level$coef[["level"]], cal_ldi$coef[["level"]]),
  di = c(NA_real_, cal_ldi$coef[["di"]]),
  floor = c(cal_level$floor, cal_ldi$floor),
  r2_fit = c(cal_level$r2_fit, cal_ldi$r2_fit),
  q = c(cal_level$q, cal_ldi$q)), file.path(u1_dir, "u1_scales.csv"))
message("\nTables: ", u1_dir)
