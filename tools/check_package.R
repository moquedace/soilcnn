# R CMD check on the package, from a staged copy of its own files.
#
#   source("<package root>/tools/check_package.R")
#
# WHAT IT DOES. Copies DESCRIPTION, NAMESPACE, LICENSE, .Rbuildignore, R/,
# man/, vignettes/ and inst/ into a staging folder -- R CMD build lists every file
# under a directory before it applies .Rbuildignore, and outputs/ made that
# minutes here -- builds the tarball with its vignette, and runs
# `R CMD check --no-manual` on it. Every NOTE, WARNING and ERROR is printed
# with its detail; the full log goes to outputs/package_check/.
#
# WHAT IT DOES NOT. No manual (that needs LaTeX), and no tests: tests/ is not
# in the tarball, and tests/run_all.R is where the tests run.
#
# AS CRAN WILL. options(soilcnn.check_as_cran = TRUE) before source() adds
# --as-cran: CRAN's own checks (its incoming checks ask the internet) and the
# examples inside \donttest{} -- the ones that train on example_run(), several
# minutes more. Without it those are skipped: the default is the quick check
# run after an edit, and a CRAN check is a step of its own.
#
# A SUGGESTED PACKAGE THAT IS NOT INSTALLED -- ranger, here -- would stop the
# check before it starts. _R_CHECK_FORCE_SUGGESTS_=false runs it as a user
# without that package would meet it: the code that needs it asks for it by
# requireNamespace(), and says so.
#
# COST: a few minutes. The check installs the package, loads torch, reads
# every function for problems, runs the examples and rebuilds the vignette.

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
  stop("Project root not found. source() this file by its full path.", call. = FALSE)
})()

for (p in c("callr", "pkgbuild")) {
  if (!requireNamespace(p, quietly = TRUE)) {
    stop("check_package.R needs ", p, ": install.packages(\"", p, "\").", call. = FALSE)
  }
}

pkg  <- unname(read.dcf(file.path(root, "DESCRIPTION"), fields = "Package")[1, 1])
work <- file.path(tempdir(), paste0(pkg, "_check"))
unlink(work, recursive = TRUE)
stage <- file.path(work, "stage", pkg)
dir.create(stage, recursive = TRUE)

parts <- c("DESCRIPTION", "NAMESPACE", "LICENSE", ".Rbuildignore", "R", "man", "vignettes",
           "inst")
parts <- parts[file.exists(file.path(root, parts))]
copied <- vapply(parts, function(p) file.copy(file.path(root, p), stage, recursive = TRUE),
                 logical(1))
if (!all(copied)) {
  stop("Could not stage: ", paste(parts[!copied], collapse = ", "), call. = FALSE)
}

message("Building ", pkg, " (with its vignette) ...")
tgz <- pkgbuild::build(stage, dest_path = work, manual = FALSE, quiet = TRUE)

check_args <- c(if (isTRUE(getOption("soilcnn.check_as_cran"))) "--as-cran", "--no-manual")
message("R CMD check ", paste(check_args, collapse = " "), " ", basename(tgz), " ...")
# TWO THREADS, AS CRAN ALLOWS. torch sizes its pool to the machine unless told
# otherwise, so an example that trains would take every core here -- beside
# whatever else runs -- and a check that passes on 32 cores says nothing of
# one on CRAN's two.
res <- callr::rcmd("check", c(check_args, basename(tgz)), wd = work,
                   env = c(callr::rcmd_safe_env(), "_R_CHECK_FORCE_SUGGESTS_" = "false",
                           OMP_NUM_THREADS = "2", MKL_NUM_THREADS = "2"),
                   fail_on_status = FALSE, show = FALSE)

log_file <- file.path(work, paste0(pkg, ".Rcheck"), "00check.log")
if (!file.exists(log_file)) {
  stop("R CMD check wrote no log (status ", res$status, "):\n",
       paste(utils::tail(strsplit(paste(res$stdout, res$stderr), "\n")[[1]], 30),
             collapse = "\n"), call. = FALSE)
}
chk <- readLines(log_file, warn = FALSE)

# The log in one place that outlives tempdir(): a new file, nothing overwritten.
keep_dir <- file.path(root, "outputs", "package_check")
dir.create(keep_dir, recursive = TRUE, showWarnings = FALSE)
kept <- file.path(keep_dir, sprintf("00check_%s.log", format(Sys.time(), "%Y%m%d_%H%M%S")))
file.copy(log_file, kept)

# Every check that is not OK, with the lines that explain it.
starts  <- grep("^\\* ", chk)
blocks  <- split(chk, findInterval(seq_along(chk), starts))
flagged <- Filter(function(b) grepl("(NOTE|WARNING|ERROR)\\s*$", b[1]), blocks)

cat("\n", strrep("=", 78), "\n", sep = "")
cat("R CMD check -- ", pkg, " -- ", length(flagged), " check(s) not OK\n", sep = "")
cat(strrep("=", 78), "\n", sep = "")
for (b in flagged) cat(b, "", sep = "\n")
cat(grep("^Status:", chk, value = TRUE), "\n", sep = "")
cat("Full log: ", kept, "\n", sep = "")
