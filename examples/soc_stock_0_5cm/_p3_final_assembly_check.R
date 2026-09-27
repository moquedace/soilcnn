# ══════════════════════════════════════════════════════════════════════════════
# P3 -- dsm_final() turns N seeds into what stage 04 turned them into
#
# WHAT THIS PROVES.
#
# Stage 04's seed loop and everything after it moved into R/final.R. Two
# halves, proved separately:
#
#   the training   each seed trained by train_one_cnn() with the same
#                  arguments; T2 showed a seed's numbers depend on its seed
#                  and its thread count only, and tests/test_final.R that two
#                  workers and one give identical seeds
#   the assembly   the N seeds turned into the ensemble median, the conformal
#                  interval, the smearing factor and the summary tables --
#                  THIS script
#
# The assembly is what the map is built from: a median that moved, a
# conformal quantile that changed, a smearing factor off in the third digit,
# and every map dsm_predict() draws would carry it. So it is run here over
# the per-seed files the deployed stage-04 run left on disk, into a separate
# directory, and every number is compared with what stage 04 wrote.
#
# WHY NOT RETRAIN AND COMPARE. dsm_final() trains each seed with 5 threads in
# a subprocess; stage 04 trained with 30 in the session. T1 showed the thread
# count changes a seed's numbers -- so a refit could only ever be compared
# statistically, and that comparison would say nothing about whether the
# assembly is right. Over the SAME files, it must be exact.
#
# WHAT MAY DIFFER, AND ONLY THAT:
#   * values that pass through a CSV twice (the per-seed predictions and
#     metrics, which stage 04 held in memory) may differ in the last binary
#     digit -- readr's reader is not correctly rounded (P1, 2026-09-26) -- so
#     they are compared to 1e-12, relative
#   * runtime_min: stage 04 kept it in memory and wrote no record of it
#   * conformal_##.rds gains a calibration_source field
#
# COST: reading ~40 CSVs, about a minute.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_p3_final_assembly_check.R")
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
install_load_pkg(c("dplyr", "readr", "tibble", "purrr", "DescTools"))
source(file.path(project_root, "R", "load_all.R"))
options(width = 200)

target_label <- "soc_stock_0_5cm"
final_base   <- file.path(project_root, "outputs", "final_model", "soc_stock_modeling", target_label)
tuning_base  <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)
p3_dir       <- file.path(tuning_base, "capability_sweep", "p3_final_assembly")

# The latest stage-04 run, as stage 05 resolves it.
final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir <- file.path(final_base, final_run_id)
summ      <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)
seeds     <- summ$seeds
ids       <- summ$selected_config_ids
alpha     <- c(0.1, 0.05)                     # stage 04's conformal_alpha

message("\n", strrep("=", 78))
message("P3 -- dsm_final()'s assembly over stage 04's own per-seed files")
message(strrep("=", 78))
message("  stage-04 run : ", final_dir)
message("  tuning run   : ", tuning_dir)
message("  config(s)    : ", paste(ids, collapse = ", "), " | seeds: ", paste(seeds, collapse = ", "))
message("  output       : ", p3_dir)
message(strrep("=", 78), "\n")

required <- sprintf("p3_%02d", 1:7)
L <- check_ledger("P3")

tol <- 1e-12
num_same <- function(a, b) {
  isTRUE(all.equal(as.numeric(a), as.numeric(b), tolerance = tol)) &&
    identical(is.na(a), is.na(b))
}

ledger_check(L, "p3_01", "the stage-04 run holds every seed's files", {
  need <- unlist(lapply(ids, function(cid) file.path(final_dir, cid, c(
    sprintf("predictions/seed%04d_pred_all.csv", seeds),
    sprintf("metrics/seed%04d_perf.csv", seeds),
    sprintf("history/seed%04d_history.csv", seeds)))))
  list(ok = all(file.exists(need)) && dir.exists(tuning_dir),
       measured = sprintf("%d of %d file(s) | tuning run on disk: %s",
                          sum(file.exists(need)), length(need), dir.exists(tuning_dir)))
})

unlink(p3_dir, recursive = TRUE)   # this script's own output, rebuilt every run
asm <- list()
for (cid in ids) {
  asm[[cid]] <- .final_assemble_config(
    cfg_dir = file.path(final_dir, cid), cfg_out_dir = file.path(p3_dir, cid),
    config_id = cid, seeds = seeds, tuning_dir = tuning_dir,
    conformal_alpha = alpha, transform_name = "log1p", verbose = FALSE)
}

ledger_check(L, "p3_02", "ensemble_predictions.csv: the same median, spread and seed count", {
  res <- vapply(ids, function(cid) {
    a <- safe_read_csv2(file.path(final_dir, cid, "ensemble_predictions.csv"))
    b <- safe_read_csv2(file.path(p3_dir, cid, "ensemble_predictions.csv"))
    identical(dim(a), dim(b)) && identical(a$sample_id, b$sample_id) &&
      identical(a$dataset_role, b$dataset_role) && num_same(a$obs, b$obs) &&
      num_same(a$pred, b$pred) && num_same(a$spread, b$spread) &&
      identical(as.integer(a$n_seeds), as.integer(b$n_seeds))
  }, logical(1))
  list(ok = all(res), measured = paste(sprintf("%s: %s", ids, res), collapse = " | "))
})

for (k in seq_along(alpha)) {
  lvl <- round(100 * (1 - alpha[k]))
  ledger_check(L, sprintf("p3_%02d", 2L + k),
               sprintf("conformal_%02d.rds: the same quantile, from the same points", lvl), {
    res <- vapply(ids, function(cid) {
      a <- readRDS(file.path(final_dir, cid, sprintf("conformal_%02d.rds", lvl)))$constant
      b <- readRDS(file.path(p3_dir, cid, sprintf("conformal_%02d.rds", lvl)))$constant
      identical(a$q, b$q) && identical(a$n, b$n) && identical(a$k, b$k)
    }, logical(1))
    q <- vapply(ids, function(cid)
      readRDS(file.path(p3_dir, cid, sprintf("conformal_%02d.rds", lvl)))$constant$q, numeric(1))
    list(ok = all(res), measured = paste(sprintf("%s: q %.4f, identical %s", ids, q, res),
                                         collapse = " | "))
  })
}

ledger_check(L, "p3_05", "smearing.rds: the same factor", {
  res <- vapply(ids, function(cid) {
    a <- readRDS(file.path(final_dir, cid, "smearing.rds"))
    b <- readRDS(file.path(p3_dir, cid, "smearing.rds"))
    identical(a, b)
  }, logical(1))
  s <- vapply(ids, function(cid) readRDS(file.path(p3_dir, cid, "smearing.rds"))$s, numeric(1))
  list(ok = all(res), measured = paste(sprintf("%s: S %.6f, identical %s", ids, s, res),
                                       collapse = " | "))
})

seed_rows <- dplyr::bind_rows(lapply(asm, `[[`, "seed_rows"))
old_rows  <- safe_read_csv2(file.path(final_dir, "comparison", "all_seed_results_test.csv"))
metric_cols <- c("n", "ccc", "r2", "mae", "nse", "rmse", "rpd", "mqi", "bias", "bias_pct")

ledger_check(L, "p3_06", "all_seed_results_test.csv: every seed's test metrics and best epoch", {
  same_keys <- identical(as.character(old_rows$config_id), as.character(seed_rows$config_id)) &&
    num_same(old_rows$seed, seed_rows$seed)
  diff_cols <- metric_cols[!vapply(metric_cols, function(m) num_same(old_rows[[m]], seed_rows[[m]]),
                                   logical(1))]
  same_epoch <- num_same(old_rows$best_epoch, seed_rows$best_epoch)
  list(ok = same_keys && length(diff_cols) == 0L && same_epoch,
       measured = sprintf("%d seed row(s) | keys %s | metrics %s | best_epoch %s", nrow(seed_rows),
                          same_keys, if (length(diff_cols)) paste("differ:", paste(diff_cols, collapse = ", ")) else "all within 1e-12",
                          same_epoch))
})

ledger_check(L, "p3_07", "config_summary_test.csv: the same mean and sd between seeds", {
  a <- safe_read_csv2(file.path(final_dir, "comparison", "config_summary_test.csv"))
  b <- .final_config_summary(seed_rows)
  cols <- setdiff(intersect(names(a), names(b)), c("config_id", "runtime_min_total"))
  bad  <- cols[!vapply(cols, function(m) num_same(a[[m]], b[[m]]), logical(1))]
  list(ok = identical(a$config_id, b$config_id) && length(bad) == 0L,
       measured = if (length(bad)) paste("differ:", paste(bad, collapse = ", ")) else
         sprintf("%d column(s) within 1e-12", length(cols)))
})

v <- ledger_verdict(L, required, file.path(p3_dir, "p3_checks.csv"))
if (v$pass) {
  message("\ndsm_final() computes, from the same seeds, what stage 04 computed.")
  message("The copy at ", p3_dir, " can be deleted.")
} else {
  message("\nThe assembly differs from stage 04's -- the table above says where. ",
          "Do not switch 04 to dsm_final() until that is understood.")
}
