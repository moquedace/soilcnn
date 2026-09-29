# ── A ledger of checks, and a verdict that cannot be an empty ledger ──────────
#
# WHY THIS EXISTS.
#
# Every capability check of the SOC project (_b1, _b2, _b4, _b6, _c1) grew
# its own copy of the same three things: a function that records one check as
# PASS or FAIL with the number it measured, a list of the check ids the script
# PROMISES to run, and a verdict that requires every promised id to be present
# AND true.
# Four copies, already different: one wraps the check in tryCatch, three do not;
# one guards against a check that returns a vector, three would die on it.
#
# THE VERDICT IS NOT "NOTHING FAILED". An empty ledger has nothing failing in it,
# and so does a ledger whose expensive check never ran because the script
# returned early. This project has been bitten by that shape six times, in
# tests/helper.R (.report() counted a zero-length accumulator as a pass), in a
# snapshot guard that switched the pixel comparison off and printed PASS, and in
# a two-config gate that skipped three configs in silence. So the verdict takes
# the promised ids and refuses to pass while any is missing.
#
# THE CHECK EXPRESSION RUNS INSIDE tryCatch. A check written against a shape
# the framework does not return is itself a finding -- it means the script and
# the framework disagree about a file -- and it must be recorded as one, not
# crash the script two minutes before the verdict that would have named it.
#
# WHAT IT DOES NOT DO: it does not stop(). A check script decides what to do
# with a FAIL; a stage that calls this decides differently. The verdict returns
# the truth and prints it in a banner nobody can miss; stopping is the caller's
# one line.

#' Open a ledger.
#' @param label  the script's name, printed on the verdict banner.
#' @return an object of class "check_ledger"; pass it to ledger_check() and
#'   ledger_verdict(). It is an environment, so recording mutates it in place
#'   and no `<<-` is needed at the call sites.
#' @noRd
check_ledger <- function(label) {
  L <- new.env(parent = emptyenv())
  L$label <- label
  L$rows  <- list()
  class(L) <- "check_ledger"
  L
}

#' Record one check.
#'
#' @param L        the ledger.
#' @param id       short id, unique within the script ("b2_08"). It is the key
#'   the verdict matches against the promised list, so a typo here is a check
#'   that "never ran" -- which the verdict reports rather than forgives.
#' @param what     one line saying what was checked.
#' @param ok       a logical(1); or an expression returning one; or a list with
#'   $ok and $measured. Evaluated lazily inside tryCatch, so an error becomes a
#'   recorded failure with the error text as `measured`.
#' @param measured what was seen, printed beside the flag. Ignored when `ok`
#'   returned a list carrying its own.
#' @return the recorded row, invisibly.
#' @noRd
ledger_check <- function(L, id, what, ok, measured = NULL) {
  stopifnot(inherits(L, "check_ledger"), is.character(id), length(id) == 1L)
  r <- tryCatch(force(ok), error = function(e) {
    list(ok = NA, measured = paste("check errored:", conditionMessage(e)))
  })
  if (!is.list(r)) r <- list(ok = r, measured = measured)
  if (is.null(r$measured)) r$measured <- measured
  # as.logical()[1] rather than isTRUE(): a check that returns a VECTOR is a
  # check that forgot an all(), and `if` on a vector is an error in R >= 4.2 --
  # which would kill the script instead of recording the mistake.
  flag <- suppressWarnings(as.logical(r$ok)[1])
  if (length(r$ok) > 1L) {
    r$measured <- paste0("[check returned ", length(r$ok), " values; first used] ",
                         as.character(r$measured)[1])
  }
  passed <- !is.na(flag) && isTRUE(flag)
  row <- tibble::tibble(
    id = id, check = what, ok = passed,
    measured = if (is.null(r$measured)) NA_character_ else as.character(r$measured)[1])
  if (!is.null(L$rows[[id]])) {
    row$measured <- paste0("[DUPLICATE id -- earlier row overwritten] ", row$measured)
  }
  L$rows[[id]] <- row
  message(sprintf("  [%s] %-8s %-48s %s",
                  if (passed) "PASS" else if (is.na(flag)) "ERR " else "FAIL",
                  id, what, substr(row$measured, 1, 100)))
  invisible(row)
}

#' The verdict: every promised id present, and every one of them true.
#'
#' @param L          the ledger.
#' @param required   the ids the script promised. Written out at the top of the
#'   script, next to what each one checks, so a reader sees the contract before
#'   the code.
#' @param report_csv optional path; the ledger is written there (PT-BR csv).
#' @return invisibly, a list: pass (logical), table, missing, failed.
#' @noRd
ledger_verdict <- function(L, required, report_csv = NULL) {
  stopifnot(inherits(L, "check_ledger"), is.character(required), length(required) > 0L)
  tbl <- if (length(L$rows) == 0L) {
    tibble::tibble(id = character(0), check = character(0), ok = logical(0),
                   measured = character(0))
  } else {
    dplyr::bind_rows(L$rows)
  }
  missing <- setdiff(required, tbl$id)
  failed  <- tbl$id[!tbl$ok]
  extra   <- setdiff(tbl$id, required)
  pass    <- length(missing) == 0L && length(failed) == 0L

  if (!is.null(report_csv)) safe_write_csv2(tbl, report_csv)

  message("\n-- Checks --")
  if (nrow(tbl) > 0L) print_wide(tbl, n = Inf) else message("  (none ran)")
  message("\n", strrep("=", 78))
  message(sprintf("%s: %s | %d of %d promised checks present and true",
                  L$label, if (pass) "PASS" else "FAIL",
                  sum(tbl$ok[tbl$id %in% required]), length(required)))
  message(strrep("=", 78))
  if (length(missing) > 0L) {
    message("\nCHECKS THAT NEVER RAN: ", paste(missing, collapse = ", "))
    message("  A check that did not run is not a check that passed. Something ",
            "above returned\n  early, or an id was renamed in one place and not ",
            "the other.")
  }
  if (length(failed) > 0L) {
    message("\nFAILED: ", paste(failed, collapse = ", "))
  }
  if (length(extra) > 0L) {
    message("\nRan but not promised (add them to `required`, or they can vanish ",
            "unnoticed): ", paste(extra, collapse = ", "))
  }
  invisible(list(pass = pass, table = tbl, missing = missing, failed = failed,
                 extra = extra))
}

#' Print a `check_ledger`
#'
#' @param x   A `check_ledger`, the record a check script keeps.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.check_ledger <- function(x, ...) {
  cat("<check_ledger> ", x$label, ": ", length(x$rows), " check(s) recorded\n", sep = "")
  invisible(x)
}
