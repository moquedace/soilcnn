# Shared test helpers
#
# Sourced by every test after it has located the project root. Keeps the
# reporting identical across tests and, crucially, never calls quit() in an
# interactive session -- a failing test must not close the console.

# ── reporting ─────────────────────────────────────────────────────────────────

#' Print PASS/FAIL for a named logical vector and end the test accordingly.
#'
#' In an interactive session a failure raises a condition (visible, does not
#' kill the session). Under Rscript it exits non-zero so CI can see it.
#'
#' @param results Named logical vector, TRUE = assertion held.
#' @param label   Test name printed on the summary line.
#' @param detail  Optional character vector of extra lines shown on failure.
.report <- function(results, label, detail = character(0)) {
  failed <- names(results)[!results]

  if (length(failed) == 0L) {
    cat(sprintf("  [PASS] all %d assertions\n", length(results)))
    cat(label, ": ALL PASS\n", sep = "")
    return(invisible(TRUE))
  }

  for (nm in failed) cat(sprintf("  [FAIL] %s\n", nm))
  if (length(detail)) cat(paste(detail, collapse = "\n"), "\n", sep = "")

  msg <- sprintf("%s: %d/%d FAILED", label, length(failed), length(results))
  if (interactive()) {
    stop(msg, call. = FALSE)
  } else {
    cat(msg, "\n", sep = "")
    quit(status = 1L)
  }
}
