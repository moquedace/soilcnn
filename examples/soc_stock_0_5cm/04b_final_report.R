# ══════════════════════════════════════════════════════════════════════════════
# 04b -- the declaration of the deployed final model: which CNN, exactly
#
# The final model on disk was fitted by stage 04 before dsm_final() existed,
# so it has its ten seeds, its interval and its smearing factor, but nothing
# that says, parameter by parameter, what the selected CNN IS and which of
# those values the search actually chose. dsm_final() writes that declaration
# as it fits; for a model fitted before it, dsm_report_final() writes it from
# the files, without retraining -- retraining would also change the seeds'
# numbers (T1: the thread count does).
#
# The per-seed numbers are re-assembled the way P3 proved dsm_final() does,
# 7 of 7 identical to what stage 04 wrote (2026-09-27).
#
# WHAT IT CHANGES: nothing that is there. It ADDS two files to the run
# directory -- final_report.md and selected_hyperparameters.csv -- and prints
# the declaration.
#
# COST: reading the seeds' CSVs, about a minute.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/04b_final_report.R")
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

# Which final run: "latest", as stage 05 resolves it, or a run id.
final_run_id <- "latest"
if (identical(final_run_id, "latest")) {
  final_run_id <- latest_run_dir(final_base, prefix = "final_",
                                 require_file = file.path("comparison", "final_run_summary.rds"),
                                 label = "final_run_id")
}
run_dir <- file.path(final_base, final_run_id)
summ    <- readRDS(file.path(run_dir, "comparison", "final_run_summary.rds"))

rep <- dsm_report_final(run_dir,
                        tuning_dir = file.path(tuning_base, summ$tuning_run_id),
                        conformal_alpha = c(0.1, 0.05))   # stage 04's

message("\nDeclaration written to:\n  ", rep$report_file,
        "\n  ", file.path(run_dir, "selected_hyperparameters.csv"))
