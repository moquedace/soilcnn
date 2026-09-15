# Unit test: the store lock
#
# check_store_spec() exists because of a failure mode with no symptom. A patch
# store carries no visible mark of the predictor set, the target or the
# resolution it was built under. Read it with a script that expects something
# else and nothing errors: the network trains, converges, and produces a map of
# the wrong thing.
#
# The lock is worth only as much as its ability to REFUSE. A check that passes
# everything is indistinguishable from no check at all, and that is precisely
# the shape a broken one takes -- so what is tested here is mostly the refusals.
#
# Verified:
#   1. a matching configuration passes, silently
#   2. each of the four mismatches is refused on its own
#   3. the PREDICTOR ORDER is refused even when the SET is identical
#   4. every mismatch is reported, not just the first
#   5. an old store (no spec in the manifest) is tolerated, not falsely failed
#   6. store_spec() reports what the store HOLDS, not the subset loaded
#
# Run: source("D:/.../tests/test_store_spec.R")     (CPU, no torch needed)

suppressMessages({
  library(tibble)
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
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "dataset.R"))

ok <- c()

# =============================================================================
# A fake store: the lock reads the manifest and two fields, nothing else.
#
# No patches, no torch, no disk. Building a real store to test a metadata
# comparison would make this test slow enough not to be run, which is the only
# way a fast check actually fails.
# =============================================================================

PREDS <- c("elev", "slope", "clay", "ndvi")

fake_store <- function(preds = PREDS, windows = "3, 9, 15",
                       target = "soc_stock_0_5cm", cell = 0.00208333,
                       spec = TRUE) {
  m <- tibble::tibble(
    predictor_cols_final = paste(preds, collapse = ";"),
    windows_extracted    = windows,
    n_channels           = length(preds)
  )
  if (spec) {
    m$target_col       <- target
    m$target_transform <- "log1p"
    m$cell_size        <- cell
  }
  list(manifest = m, predictors = preds, n_channels = length(preds),
       # deliberately NOT the full set: this is the subset a session loaded,
       # and store_spec() must ignore it.
       window_sizes = 3L)
}

st <- fake_store()

# -- helper: did the check refuse, and did it say why? ------------------------
refused <- function(..., pattern) {
  msg <- tryCatch({ check_store_spec(...); NA_character_ },
                  error = function(e) conditionMessage(e))
  !is.na(msg) && grepl(pattern, msg)
}

# =============================================================================
# 1. The matching case passes, and returns nothing to print
# =============================================================================

res <- tryCatch(
  check_store_spec(st, predictors = PREDS, windows = c(3L, 9L),
                   target_col = "soc_stock_0_5cm", cell_size = 0.00208333),
  error = function(e) e)
ok["match_does_not_stop"]     <- !inherits(res, "error")
ok["match_returns_no_issues"] <- !inherits(res, "error") && length(res) == 0L

# Nothing asked for is nothing checked: a caller that only cares about windows
# must not be failed on a target it never mentioned.
ok["nulls_are_skipped"] <- !inherits(
  tryCatch(check_store_spec(st), error = function(e) e), "error")

# =============================================================================
# 2. Each mismatch, on its own
# =============================================================================

ok["refuses_missing_predictor"] <- refused(
  st, predictors = c(PREDS, "twi"), pattern = "PREDICTORS differ")
ok["names_the_missing_predictor"] <- refused(
  st, predictors = c(PREDS, "twi"), pattern = "twi")

ok["refuses_extra_predictor"] <- refused(
  st, predictors = PREDS[1:3], pattern = "PREDICTORS differ")

ok["refuses_absent_window"] <- refused(
  st, windows = c(3L, 21L), pattern = "WINDOWS 21")

ok["refuses_other_target"] <- refused(
  st, target_col = "soc_stock_0_30cm", pattern = "TARGET differs")

ok["refuses_other_resolution"] <- refused(
  st, cell_size = 0.1666667, pattern = "RESOLUTION differs")

# Floating point must not manufacture a failure: the same resolution read twice
# from the same file differs in the last bits and means the same thing.
ok["tolerates_float_noise"] <- !inherits(
  tryCatch(check_store_spec(st, cell_size = 0.00208333 + 1e-13),
           error = function(e) e), "error")

# =============================================================================
# 3. ORDER: the set is identical, and it is still wrong
#
# The order is the contract tying channel i to band i. Scramble it and the
# network is fed one predictor while the map is built from another -- no error,
# no warning, and every metric still looks reasonable.
# =============================================================================

ok["refuses_reordered_predictors"] <- refused(
  st, predictors = PREDS[c(2, 1, 3, 4)], pattern = "PREDICTOR ORDER")

ok["order_is_not_reported_as_a_set_difference"] <- !refused(
  st, predictors = PREDS[c(2, 1, 3, 4)], pattern = "PREDICTORS differ")

# =============================================================================
# 4. Every mismatch at once
#
# Finding one, fixing it, re-running and finding the next is how a five-minute
# correction becomes an afternoon.
# =============================================================================

bad <- check_store_spec(st, predictors = c(PREDS, "twi"), windows = c(3L, 21L),
                        target_col = "other", cell_size = 0.5, strict = FALSE)
ok["non_strict_returns_all_four"] <- length(bad) == 4L
ok["non_strict_does_not_stop"]    <- is.character(bad)

# =============================================================================
# 5. A store written before the spec existed
#
# It cannot answer, and "cannot answer" is not "answered wrongly". Failing it
# would force a re-extraction to gain information the store already implies.
# =============================================================================

old <- fake_store(spec = FALSE)
ok["old_store_tolerated"] <- !inherits(
  tryCatch(check_store_spec(old, target_col = "anything", cell_size = 99),
           error = function(e) e), "error")
# ...but what it can still answer, it answers.
ok["old_store_still_checks_predictors"] <- refused(
  old, predictors = "only_one", pattern = "PREDICTORS differ")

ok["old_store_spec_is_NA_not_error"] <-
  is.na(store_spec(old)$target_col) && is.na(store_spec(old)$cell_size)

# =============================================================================
# 6. store_spec() reports the STORE, not the session
# =============================================================================

sp <- store_spec(st)
ok["spec_windows_come_from_manifest"] <- identical(sp$windows, c(3L, 9L, 15L))
ok["spec_predictors_in_store_order"]  <- identical(sp$predictors, PREDS)
ok["spec_cell_size_is_numeric"]       <- is.numeric(sp$cell_size) &&
                                         abs(sp$cell_size - 0.00208333) < 1e-12

cat("  lock checked             : 4 mismatches, each refused and named\n")

.report(ok, "test_store_spec")
