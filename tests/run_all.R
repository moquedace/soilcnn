# Run every test and summarise at the end.
#
#   source("tests/run_all.R")
#
# Each test runs in its own environment, so they cannot leak variables into
# one another, and a failure in one does not stop the rest — you get the full
# picture in a single pass instead of fixing them one at a time.

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
  stop("Project root not found. setwd() to the package root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

# Fast: ~3 seconds for all of them. Run after every edit, without thinking.
test_files <- c(
  # FIRST, always: a file that cannot be parsed makes every test in it fail for
  # a reason that has nothing to do with what they test. One bad escape in
  # R/diagnostics.R once stopped the 99 before its first check.
  "test_sources_parse.R",    # every .R file under R/, tests/ and tools/
  "test_package_metadata.R", # NAMESPACE = the tags, DESCRIPTION = the pkg:: calls, the worker guards
  "test_patch_geometry.R",   # geometry of the patches, both extraction paths
  "test_transform_loss.R",   # early-stopping loss vs the real torch loss
  "test_validation.R",       # clamp contract + option-set validation
  "test_preprocess.R",       # QC vs scaling split, and order equivalence
  "test_patch_store_io.R",   # store round-trip + the torch_save 2^31 guard
  "test_metrics_reporting.R",# the three functions that lie quietly
  "test_resample.R",         # fold plans: partition, leakage, seed
  "test_knndm.R",            # folds matched to the prediction area
  "test_store_spec.R",       # the store lock: what it REFUSES
  "test_model_registry.R",   # registry contract + the tabular fold view
  "test_aoa.R",              # dissimilarity index + area of applicability
  "test_api.R",              # the front end: specs, "auto", dispatch
  "test_train_defaults.R",   # what dsm_train() reads from the store: windows, batches, inverse, cores
  "test_selection_order.R",  # the test set is scored only after the choice
  "test_occlusion.R",        # does the model use the neighbourhood?
  "test_importance.R",       # which variables a model relies on, against known models
  "test_conformal.R",        # the interval covers what it promises
  "test_smearing.R",         # the mean surface beside the median surface
  "test_block_bootstrap.R",  # intervals from whole blocks; equal-area blocks; Moran's I by distance
  "test_architecture.R",     # embed_pool wiring
  "test_fcn.R",              # the fully convolutional path gives what the network gives
  "test_augmentation.R",     # D4 symmetries
  "test_run_dirs.R",         # "latest" by time, the pt-br csv round trip, resume identity
  "test_checks.R",           # the check ledger cannot pass by doing nothing
  "test_fold_cache.R"        # scaling from the fold's own training rows; store_complete; patch centres
)

# Slow: these train real torch models (~3 min). They are the only ones that
# prove the modules are WIRED to each other, so the default is to run them.
#
# Set FALSE while iterating minute by minute -- but run them before any
# expensive script, which is where they pay for themselves: it is how the
# bind_rows type guess and the zero noise floor were both found.
run_slow <- TRUE

slow_files <- c(
  "test_resample_run.R",     # end-to-end wiring: store -> folds -> tables
  "test_api_run.R",          # the same, through dsm_load() and dsm_train()
  "test_train_side_by_side.R", # dsm_train()'s units side by side: 2 workers == 1; the record; resume
  "test_final.R",            # dsm_final(): N seeds side by side == one by one; the report
  "test_predict.R",          # dsm_predict(): every band at every pixel, by hand; 2 workers == 1
  "test_prepare.R",          # points + raster folder -> store; 2 cores == 1
  "test_package_install.R",  # build, install to a temp library, library(); workers open the same code
  "test_examples.R"          # every example of man/ runs, \donttest{} too; every failure at once
)

if (run_slow) test_files <- c(test_files, slow_files)

rule   <- strrep("-", 72)
status <- character(0)
t0     <- Sys.time()

# THE PACKAGE IS LOADED ONCE FOR THE SUITE -- after the parse test, which needs
# nothing and must come first -- and every test is told so by a mark in the
# environment it runs in (see .load_framework() in helper.R). A package that
# does not load fails every test after it for the same reason, so the suite
# stops there and says why, instead of printing it twenty-eight times.
source(file.path(root, "tests", "helper.R"))
loaded <- FALSE

for (tf in test_files) {
  cat("\n", rule, "\n", tf, "\n", sep = "")
  if (!loaded && !identical(tf, "test_sources_parse.R")) {
    load_error <- tryCatch({ .load_framework(root); NULL },
                           error = function(e) conditionMessage(e))
    if (!is.null(load_error)) {
      cat("  the package did not load: ", load_error, "\n", sep = "")
      status[setdiff(test_files, names(status))] <- "NOT RUN"
      break
    }
    loaded <- TRUE
  }
  test_env <- new.env()
  test_env$.suite_loaded <- TRUE
  status[tf] <- tryCatch(
    {
      source(file.path(root, "tests", tf), local = test_env)
      "PASS"
    },
    error = function(e) {
      cat("  ", conditionMessage(e), "\n", sep = "")
      "FAIL"
    }
  )
}

el <- Sys.time() - t0

cat("\n", rule, "\n", sep = "")
for (tf in names(status)) {
  cat(sprintf("  [%s] %s\n", status[tf], tf))
}
cat(sprintf("\n  %d/%d passed in %.1f %s\n",
            sum(status == "PASS"), length(status),
            as.numeric(el), units(el)))

if (!run_slow) {
  cat("  (slow tests SKIPPED -- run_slow <- FALSE at the top of this file)\n")
}

if (any(status != "PASS")) {
  cat("  -> rerun a failing file on its own for the full output\n")
}
