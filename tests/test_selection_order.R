# Unit test: the test set is scored only after the choice is locked
#
# WHAT THIS PROTECTS.
#
# Tuning does not score the test set (`evaluate_test = FALSE`), because the best
# of 216 noisy test scores is systematically higher than any one of them, and a
# test column beside the selection metric is selection on the test set carried
# out by whoever reads the table.
#
# score_test_grid() exists so that the grid CAN be scored on the test set
# afterwards, to measure selection optimism -- a number worth publishing. What
# makes that safe is the ORDER, and an order that is merely documented is an
# order that will be got wrong once and never noticed. So it is enforced, and
# the enforcement is what this file tests:
#
#   1. scoring refuses outright while no selection is frozen
#   2. freezing is idempotent for the same config -- reruns are harmless
#   3. freezing a DIFFERENT config over a frozen one is refused, because that
#      is the score-dislike-refreeze-rescore loop the file exists to stop
#   4. the record carries when it was frozen and against which commit, so a
#      third party can check the order without having been there
#   5. the escape hatch exists, and marks the result as unsupported
#
# No model is trained here: the refusals all happen before any torch call, and
# they are the part that carries the guarantee.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_selection_order.R")

suppressMessages({
  library(tibble)
  library(dplyr)
})

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
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "test_optimism.R"))

ok <- logical(0)

run_dir <- file.path(tempdir(), "test_selection_order")
unlink(run_dir, recursive = TRUE)
create_output_dirs(file.path(run_dir, c("comparison", "models")))

fake_data <- list(store = list(), points = tibble(), type_table = tibble())

# ── 1. no frozen selection: refuse, and say what to do ───────────────────────

err <- tryCatch(score_test_grid(run_dir, fake_data), error = function(e) e)
ok["unfrozen_run_is_refused"] <- inherits(err, "error")
# The message has to name the fix. A refusal that leaves the reader guessing
# gets worked around, and the work-around is the thing being prevented.
ok["refusal_names_freeze_selection"] <-
  grepl("freeze_selection", conditionMessage(err), fixed = TRUE)

# ── 2-4. freezing ────────────────────────────────────────────────────────────

rec <- suppressMessages(freeze_selection(run_dir, "cfg_002", rule = "one_se",
                                         metric = "val_ccc"))
ok["freeze_writes_the_record"] <-
  file.exists(file.path(run_dir, "comparison", "selection.rds"))
ok["freeze_records_the_config"] <- identical(rec$config_id, "cfg_002")
ok["freeze_records_the_rule"]   <- identical(rec$rule, "one_se")
ok["freeze_records_the_time"]   <- inherits(rec$frozen_at, "POSIXct")

# IDEMPOTENT for the same config: re-running stage 04 must not be an error, and
# must not move the timestamp -- a timestamp that refreshes proves nothing.
again <- suppressMessages(freeze_selection(run_dir, "cfg_002"))
ok["refreezing_the_same_config_is_quiet"] <- identical(again$config_id, "cfg_002")
ok["refreezing_does_not_move_the_clock"]  <-
  identical(again$frozen_at, rec$frozen_at)

# ...and a DIFFERENT config is refused. This is the loop the file prevents:
# score the grid, dislike the answer, re-freeze on the config that won on test.
err2 <- tryCatch(freeze_selection(run_dir, "cfg_007"), error = function(e) e)
ok["refreezing_another_config_is_refused"] <- inherits(err2, "error")
ok["that_refusal_names_the_first_choice"] <-
  grepl("cfg_002", conditionMessage(err2), fixed = TRUE)

# Two configs at once is legitimate -- one per family, all chosen on validation
# before anything was scored. What is refused is CHANGING a locked choice.
run_dir2 <- file.path(tempdir(), "test_selection_order_2")
unlink(run_dir2, recursive = TRUE)
create_output_dirs(file.path(run_dir2, "comparison"))
rec2 <- suppressMessages(freeze_selection(run_dir2, c("cfg_002", "rf_001")))
ok["freeze_accepts_one_config_per_family"] <- length(rec2$config_id) == 2L
ok["order_of_a_multi_freeze_does_not_matter"] <- {
  r <- suppressMessages(freeze_selection(run_dir2, c("rf_001", "cfg_002")))
  length(r$config_id) == 2L
}

# ── 5. the escape hatch is available and self-reporting ──────────────────────
#
# It still fails here, but LATER and for a different reason (no tune_grid.rds),
# which is exactly the point: allow_unfrozen removes the ordering guard and
# nothing else.
err3 <- tryCatch(score_test_grid(run_dir, fake_data, allow_unfrozen = TRUE),
                 error = function(e) e)
ok["escape_hatch_passes_the_ordering_guard"] <-
  inherits(err3, "error") &&
  !grepl("freeze_selection", conditionMessage(err3), fixed = TRUE)

# With a selection frozen, the next refusal must also be about the run's
# contents rather than about ordering -- the guard must not fire twice.
err4 <- tryCatch(score_test_grid(run_dir, fake_data), error = function(e) e)
ok["frozen_run_gets_past_the_guard"] <-
  inherits(err4, "error") &&
  !grepl("frozen selection", conditionMessage(err4), fixed = TRUE)

# A run with no data at all must say WHICH file is missing, not fail obscurely
# three frames deep.
ok["missing_grid_is_named"] <- grepl("tune_grid.rds", conditionMessage(err4),
                                     fixed = TRUE)

cat("  selection record         : config, rule, metric, time",
    if (!is.na(rec$git_commit)) paste0(", commit ", rec$git_commit) else "",
    "\n", sep = "")
cat("  enforced order           : freeze_selection() -> score_test_grid()\n")

unlink(c(run_dir, run_dir2), recursive = TRUE)
.report(ok, "test_selection_order")
