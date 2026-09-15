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
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

# Fast: ~3 seconds for all of them. Run after every edit, without thinking.
test_files <- c(
  "test_patch_geometry.R",   # geometry of the patches, both extraction paths
  "test_transform_loss.R",   # early-stopping loss vs the real torch loss
  "test_validation.R",       # clamp contract + option-set validation
  "test_preprocess.R",       # QC vs scaling split, and order equivalence
  "test_patch_store_io.R",   # store round-trip + the torch_save 2^31 guard
  "test_metrics_reporting.R",# as tres funcoes que mentem em silencio
  "test_resample.R",         # fold plans: partition, leakage, seed
  "test_architecture.R",     # embed_pool wiring
  "test_augmentation.R"      # D4 symmetries
)

# Slow: these train real torch models (~3 min). They are the only ones that
# prove the modules are WIRED to each other, so the default is to run them.
#
# Set FALSE while iterating minute by minute -- but run them before any
# expensive script, which is where they pay for themselves: it is how the
# bind_rows type guess and the zero noise floor were both found.
run_slow <- TRUE

slow_files <- c(
  "test_resample_run.R"      # end-to-end wiring: store -> folds -> tables
)

if (run_slow) test_files <- c(test_files, slow_files)

rule   <- strrep("-", 72)
status <- character(0)
t0     <- Sys.time()

for (tf in test_files) {
  cat("\n", rule, "\n", tf, "\n", sep = "")
  status[tf] <- tryCatch(
    {
      source(file.path(root, "tests", tf), local = new.env())
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
  cat("  (slow tests SKIPPED -- run_slow <- FALSE at the top of this file)
")
}

if (any(status != "PASS")) {
  cat("  -> rerun a failing file on its own for the full output\n")
}
