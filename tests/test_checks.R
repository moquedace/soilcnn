# Unit test: the check ledger cannot pass by doing nothing
#
# WHY THIS FILE EXISTS.
#
# Every capability check script records PASS/FAIL rows and ends with a verdict.
# The verdict's one job is to refuse the two ways a script can look green while
# having checked nothing: an empty ledger, and a ledger missing the check that
# mattered because the script returned early. This project has hit both.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_checks.R")

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
source(file.path(root, "R", "checks.R"))

ok <- logical(0)
quiet <- function(expr) suppressMessages(expr)

# ── 1. AN EMPTY LEDGER FAILS, AND SAYS WHICH CHECKS NEVER RAN ────────────────
L <- check_ledger("t")
v <- quiet(ledger_verdict(L, required = c("t_01", "t_02")))
ok["empty_ledger_fails"] <- isFALSE(v$pass)
ok["empty_ledger_names_the_missing"] <- setequal(v$missing, c("t_01", "t_02"))

# ── 2. ALL PROMISED, ALL TRUE -> PASS; ONE MISSING -> FAIL EVEN IF THE REST PASS
L <- check_ledger("t")
quiet(ledger_check(L, "t_01", "one", TRUE, "x"))
quiet(ledger_check(L, "t_02", "two", 1 == 1, "y"))
ok["complete_and_true_passes"] <- isTRUE(quiet(ledger_verdict(L, c("t_01", "t_02")))$pass)
ok["a_missing_promised_check_fails_the_verdict"] <-
  isFALSE(quiet(ledger_verdict(L, c("t_01", "t_02", "t_03")))$pass)

# ── 3. A FALSE CHECK FAILS AND IS NAMED ──────────────────────────────────────
L <- check_ledger("t")
quiet(ledger_check(L, "t_01", "one", FALSE, "saw 3, wanted 4"))
v <- quiet(ledger_verdict(L, "t_01"))
ok["a_false_check_fails"] <- isFALSE(v$pass) && identical(v$failed, "t_01")
ok["the_measurement_is_kept"] <- identical(v$table$measured, "saw 3, wanted 4")

# ── 4. A CHECK THAT ERRORS IS A FINDING, NOT A CRASH ─────────────────────────
L <- check_ledger("t")
row <- quiet(ledger_check(L, "t_01", "reads a column that is not there",
                          stop("object 'nope' not found")))
ok["an_erroring_check_is_recorded_as_failed"] <- isFALSE(row$ok)
ok["the_error_text_is_the_measurement"] <- grepl("check errored", row$measured)
ok["the_script_survived_it"] <- TRUE

# ── 5. A VECTOR-VALUED CHECK DOES NOT KILL THE SCRIPT EITHER ─────────────────
L <- check_ledger("t")
row <- quiet(ledger_check(L, "t_01", "forgot an all()", c(TRUE, FALSE, TRUE)))
ok["a_vector_check_is_flagged_not_fatal"] <- grepl("returned 3 values", row$measured)

# ── 6. A CHECK MAY CARRY ITS OWN MEASUREMENT ─────────────────────────────────
L <- check_ledger("t")
row <- quiet(ledger_check(L, "t_01", "list form", list(ok = TRUE, measured = "n = 591")))
ok["list_form_carries_ok_and_measured"] <- isTRUE(row$ok) && identical(row$measured, "n = 591")

# ── 7. NA IS NOT A PASS ──────────────────────────────────────────────────────
L <- check_ledger("t")
quiet(ledger_check(L, "t_01", "na", NA))
ok["na_is_not_a_pass"] <- isFALSE(quiet(ledger_verdict(L, "t_01"))$pass)

# ── 8. UNPROMISED CHECKS ARE REPORTED, DUPLICATE IDS ARE MARKED ──────────────
L <- check_ledger("t")
quiet(ledger_check(L, "t_01", "a", TRUE))
quiet(ledger_check(L, "t_09", "not in the list", TRUE))
v <- quiet(ledger_verdict(L, "t_01"))
ok["unpromised_checks_are_listed"] <- identical(v$extra, "t_09")
quiet(ledger_check(L, "t_01", "a again", TRUE))
ok["a_duplicate_id_is_marked"] <- grepl("DUPLICATE", L$rows[["t_01"]]$measured)

# ── 9. THE CSV IS WRITTEN WHEN ASKED ─────────────────────────────────────────
tmp <- file.path(tempdir(), "test_checks", "ledger.csv")
quiet(ledger_verdict(L, c("t_01", "t_09"), report_csv = tmp))
ok["the_ledger_csv_is_written"] <- file.exists(tmp) &&
  nrow(safe_read_csv2(tmp)) == 2L
unlink(dirname(tmp), recursive = TRUE, force = TRUE)

cat("  empty ledger             : FAIL, and names every promised id\n")
cat("  erroring / vector checks : recorded as findings, script survives\n")

.report(ok, "test_checks")
