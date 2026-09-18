# ── Load the framework ────────────────────────────────────────────────────────
#
#   source("<project_root>/R/load_all.R")
#
# and that is the whole preamble. Every pipeline script used to carry ten
# source() lines in a specific order, which is ten chances to omit one and get
# "could not find function" minutes into a run -- and the order is not
# guessable: R/api.R uses helpers from R/resample.R, which uses helpers from
# R/utils.R.
#
# This is not a package yet. When it becomes one this file disappears and
# library(deeplearningcaret) takes its place; until then the order lives in ONE
# place instead of in every script that wants to use the framework.
#
# Optional files are sourced only if present, and files that need an absent
# package are allowed to fail loudly here rather than halfway through a run:
# a missing dependency is cheaper to learn about now.

.dlc_root <- (function() {
  # Works whether this file is source()d by path, from the project root, or
  # from a script inside examples/.
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(dirname(normalizePath(f[1], mustWork = FALSE))))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(dirname(normalizePath(of, mustWork = FALSE))))
    }
  }
  cand <- c(cand, getwd(), dirname(getwd()))
  for (d in cand) {
    if (file.exists(file.path(d, "R", "cnn_architecture.R"))) return(d)
  }
  stop("Could not locate the project root from here. source() this file with ",
       "its full path, or setwd() to the project root first.", call. = FALSE)
})()

# THE ORDER IS THE DEPENDENCY ORDER, and it is written out rather than sorted:
# an alphabetical list would put api.R first, which cannot work.
.dlc_files <- c(
  "utils.R",            # paths, safe IO, device
  "patches.R",          # patch geometry, shared by extraction and prediction
  "preprocess.R",       # QC (fold-independent) vs scaling (fold-dependent)
  "metrics.R",          # ccc() and the rest
  "dataset.R",          # the patch store, the fold cache, the table view
  "resample.R",         # fold plans, buffers, noise floor, one_se
  "knndm.R",            # folds matched to where the map will be predicted
  "diagnostics.R",      # checks about THIS run on real data
  "aoa.R",              # dissimilarity index / area of applicability
  "conformal.R",        # calibrated intervals + PICP
  "smearing.R",         # the mean surface beside the median surface
  "cnn_architecture.R", # the model
  "tune_grid.R",        # the search space
  "train_cnn.R",        # the CNN runner
  "test_optimism.R",    # freeze_selection / score_test_grid (after 03)
  "occlusion.R",        # does the network use the neighbourhood at all?
  "model_registry.R",   # model_spec / register_model
  "baselines.R",        # rf, mlp, cnn -- registered
  "train_table.R",      # the tabular runner
  "caret_adapter.R",    # optional: borrow caret's model library
  "api.R"               # dsm_load / spatial_cv / dsm_train
)

for (.f in .dlc_files) {
  .p <- file.path(.dlc_root, "R", .f)
  if (file.exists(.p)) source(.p) else {
    warning("Framework file missing, skipped: ", .f, call. = FALSE)
  }
}
rm(.f, .p)

message("deep_learning_caret loaded from ", .dlc_root,
        "  (", length(.dlc_files), " modules)")
