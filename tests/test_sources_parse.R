# Unit test: every R file in the project PARSES
#
# This is the cheapest test here and the one with the worst failure mode when
# it is missing.
#
# A syntax error in R/ is not found when it is written. It is found by whoever
# next source()s the file -- and in this project that is a pipeline script,
# often minutes into a run, or the 99 right before an expensive stage. The
# error then names a line in a file the person was not editing and had no
# reason to suspect.
#
# It happened: an invalid escape ("\." instead of "\\.") reached R/diagnostics.R
# and the 99 refused to start, one line into a check that had nothing to do
# with it. R does not compile the file lazily -- a single bad escape makes the
# WHOLE file unparseable, so one character took down every function in it.
#
# parse() answers exactly this question and touches nothing: no package is
# loaded, no code runs, nothing is evaluated. It costs milliseconds per file,
# so it runs FIRST in run_all.R -- there is no point testing behaviour in a
# file that cannot be read.
#
# Verified:
#   1. every .R file under R/, tests/ and tools/ parses
#   2. the report names the file AND the line, so the fix is immediate
#
# Run: source("D:/.../tests/test_sources_parse.R")     (no packages needed)

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

# Every directory of the repository that holds R code someone will source:
# the package, its tests and the check runner. Until 2026-09-28 the SOC
# project's scripts lived here too, under examples/, and were parsed with
# them; since then the repository is the package alone.
dirs <- c("R", "tests", "tools")

files <- unlist(lapply(dirs, function(d) {
  p <- file.path(root, d)
  if (!dir.exists(p)) return(character(0))
  list.files(p, pattern = "\\.R$", full.names = TRUE)
}), use.names = FALSE)

# This file is being read right now; parsing it again is harmless but pointless.
files <- files[basename(files) != "test_sources_parse.R"]

if (length(files) == 0L) {
  stop("No .R files found under ", paste(dirs, collapse = ", "),
       " -- the test found nothing to check, which is not the same as ",
       "everything being fine.", call. = FALSE)
}

ok      <- stats::setNames(logical(length(files)), basename(files))
details <- character(0)

for (i in seq_along(files)) {
  # keep.source = FALSE: the srcref machinery is the slow part and nothing here
  # reads it. What is wanted is the yes/no and, on no, the message.
  res <- tryCatch({
    parse(files[i], keep.source = FALSE)
    TRUE
  }, error = function(e) conditionMessage(e))

  if (isTRUE(res)) {
    ok[i] <- TRUE
  } else {
    ok[i] <- FALSE
    details <- c(details, sprintf("  %s\n    %s",
                                  sub(root, "", files[i], fixed = TRUE),
                                  gsub("\n", "\n    ", res)))
  }
}

cat(sprintf("  parsed                   : %d file(s) under %s\n",
            length(files), paste(dirs, collapse = ", ")))

.report(ok, "test_sources_parse", details)
