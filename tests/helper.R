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
  # AN EMPTY ACCUMULATOR IS A FAILURE, NOT A PASS.
  #
  # This printed "[PASS] all 0 assertions" followed by "ALL PASS" for a
  # zero-length vector, which is the worst output a test harness can produce:
  # a file whose assertions never ran reports success, and the run_all summary
  # counts it among the passes.
  #
  # It is not hypothetical. The accumulator was called `ok` in 15 files and
  # `results` in 6, and a block written against the wrong name in this project
  # stopped with "objeto results nao encontrado" -- which was luck. Had the name
  # existed but been empty, or had the block been skipped by a FALSE guard, the
  # file would have passed silently. The names are now unified, and this makes
  # the remaining shape of that mistake loud.
  #
  # A test file with nothing to assert has no reason to call .report().
  if (length(results) == 0L) {
    stop(label, ": .report() received 0 assertions. Either the accumulator was ",
         "never filled, or a name is wrong. A test that asserts nothing must ",
         "not report a pass.", call. = FALSE)
  }
  if (!is.logical(results)) {
    stop(label, ": .report() needs a logical vector, got ", class(results)[1],
         ". A non-logical accumulator makes !results silently wrong.",
         call. = FALSE)
  }
  if (anyNA(results)) {
    # NA is neither pass nor fail, and !NA is NA, so an NA assertion would
    # vanish from names(results)[!results] and be counted as a pass.
    na_names <- names(results)[is.na(results)]
    stop(label, ": ", length(na_names), " assertion(s) returned NA, which ",
         "would have been counted as passing: ",
         paste(na_names, collapse = ", "), call. = FALSE)
  }

  # AND AN UNNAMED ONE IS THE SAME BUG WEARING A DIFFERENT HAT.
  #
  # `failed` below is names(results)[!results]. On a vector with no names that
  # is NULL[...] -- which is NULL, length 0 -- so a file whose assertions all
  # returned FALSE would report ALL PASS. c(TRUE, FALSE) passes; naming them
  # is what makes the failure reachable, and nothing until now required it.
  unnamed <- is.null(names(results)) | !nzchar(names(results))
  if (any(unnamed)) {
    stop(label, ": ", sum(unnamed), " of ", length(results), " assertion(s) ",
         "have no name. An unnamed FALSE cannot be reported and would pass ",
         "silently -- give every assertion a name.", call. = FALSE)
  }

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

# ── loading the framework ─────────────────────────────────────────────────────

#' Load the package from the source tree, once, and refuse a shadowed one.
#'
#' Every test loads the framework the way a session working on it does --
#' pkgload::load_all() -- and none source()s files of R/ one by one any more.
#' A source()d file lands in the global environment, where it answers every
#' call a test makes directly, while the framework, calling within its own
#' namespace, runs the package: the test would check one loading of the code
#' and the framework would run another, each with its own model registry.
#'
#' tests/run_all.R loads once for the suite and marks the environment it runs
#' each test in; a test run on its own loads for itself, so it always checks
#' the tree as it stands.
#'
#' A global environment that still holds functions the package defines -- what
#' a session that source()d R/load_all.R, before this was a package, leaves
#' behind -- is refused: those copies would answer the test instead of the
#' package. The package's own .onAttach() already refuses the copies source()d
#' from its R/; this also catches a function of the same name from anywhere
#' else. One that IS the package's function (the `%>%` utils.R used to bind
#' there is magrittr's own) changes nothing and passes.
#'
#' @param root The project root.
#' @return The package's namespace, invisibly.
.load_framework <- function(root) {
  suite <- isTRUE(get0(".suite_loaded", envir = parent.frame(), inherits = FALSE))
  if (!suite) {
    if (!requireNamespace("pkgload", quietly = TRUE)) {
      stop("The tests load the framework with pkgload::load_all(): ",
           "install.packages(\"pkgload\").", call. = FALSE)
    }
    pkgload::load_all(root, quiet = TRUE)
  }
  pkg <- read.dcf(file.path(root, "DESCRIPTION"), fields = "Package")[1, 1]
  ns  <- asNamespace(pkg)
  both <- intersect(ls(globalenv(), all.names = TRUE), ls(ns, all.names = TRUE))
  shadow <- Filter(function(n) {
    f <- get(n, envir = globalenv())
    is.function(f) && !identical(f, get(n, envir = ns))
  }, both)
  if (length(shadow)) {
    stop("The global environment holds ", length(shadow), " function(s) the ",
         "package also defines (", paste(utils::head(shadow, 4), collapse = ", "),
         if (length(shadow) > 4L) ", ..." else "", ") -- left by an earlier ",
         "source(). They would answer this test's calls instead of the package.",
         "\n  Remove them -- rm(list = ls(all.names = TRUE)) clears the whole ",
         "workspace -- and run it again. RStudio's Restart R keeps the global ",
         "environment, so a restart alone does not clear them.", call. = FALSE)
  }
  invisible(ns)
}
