# R CMD check on the package, from a staged copy of its own files.
#
#   source("<package root>/tools/check_package.R")
#
# WHAT IT DOES. Copies DESCRIPTION, NAMESPACE, LICENSE, .Rbuildignore, R/,
# man/, vignettes/, inst/ and tests/ into a staging folder -- R CMD build lists
# every file under a directory before it applies .Rbuildignore, and outputs/
# made that minutes here -- builds the tarball with its vignette, and runs
# `R CMD check` on it. Every NOTE, WARNING and ERROR is printed with its
# detail; the full log goes to outputs/package_check/. Of tests/, the tarball
# holds the testthat subset only, which the check runs; tests/run_all.R is
# where the full suite runs.
#
# THE MANUAL. Left out (--no-manual) unless options(soilcnn.check_manual =
# TRUE): CRAN's check builds the PDF manual, and here that needs LaTeX --
# TinyTeX since 2026-10-04, found where it installs when this session's PATH
# predates it. makeindex, which TinyTeX lacks and R's index needs, is
# installed first. Then the manual is built on its own (R CMD Rd2pdf, a
# minute): each build stops at the LaTeX package it misses, tinytex reads the
# error and installs it, and the build runs again -- before the check spends
# its minutes. A failure that is not a missing package stops here, with what
# R and LaTeX said and their logs kept in outputs/package_check/.
#
# AS CRAN WILL. options(soilcnn.check_as_cran = TRUE) before source() adds
# --as-cran: CRAN's own checks (its incoming checks ask the internet) and the
# examples inside \donttest{} -- the ones that train on example_run(), several
# minutes more. Without it those are skipped: the default is the quick check
# run after an edit, and a CRAN check is a step of its own.
#
# ON A MACHINE LIKE CRAN'S. options(soilcnn.check_without_libtorch = TRUE)
# runs the check with TORCH_HOME at an empty folder, where torch finds no
# libtorch: CRAN installs the torch package and never downloads its backend.
# Every example and test that needs it must then skip, and the rest pass --
# which a check here, where libtorch is installed, cannot show otherwise. That
# torch really sees none is asked first, and the check stops if it does.
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
           "inst", "tests")
parts <- parts[file.exists(file.path(root, parts))]
copied <- vapply(parts, function(p) file.copy(file.path(root, p), stage, recursive = TRUE),
                 logical(1))
if (!all(copied)) {
  stop("Could not stage: ", paste(parts[!copied], collapse = ", "), call. = FALSE)
}

message("Building ", pkg, " (with its vignette) ...")
tgz <- pkgbuild::build(stage, dest_path = work, manual = FALSE, quiet = TRUE)

check_manual <- isTRUE(getOption("soilcnn.check_manual"))
check_args <- c(if (isTRUE(getOption("soilcnn.check_as_cran"))) "--as-cran",
                if (!check_manual) "--no-manual")
# TWO THREADS, AS CRAN ALLOWS. torch sizes its pool to the machine unless told
# otherwise, so an example that trains would take every core here -- beside
# whatever else runs -- and a check that passes on 32 cores says nothing of
# one on CRAN's two.
check_env <- c(callr::rcmd_safe_env(), "_R_CHECK_FORCE_SUGGESTS_" = "false",
               OMP_NUM_THREADS = "2", MKL_NUM_THREADS = "2")
if (isTRUE(getOption("soilcnn.check_without_libtorch"))) {
  no_backend <- file.path(work, "torch_home_empty")
  dir.create(no_backend)
  # TORCH_INSTALL = 0: a torch that found no backend must not fetch one.
  check_env <- c(check_env, TORCH_HOME = no_backend, TORCH_INSTALL = "0")
  sees <- callr::r(function() torch::torch_is_installed(), env = check_env)
  if (!identical(sees, FALSE)) {
    stop("With TORCH_HOME at an empty folder, torch::torch_is_installed() still says ",
         format(sees), ": this check would not be the one CRAN's machines run.", call. = FALSE)
  }
  message("Checked as on a machine without libtorch: torch::torch_is_installed() is FALSE there.")
}

if (check_manual) {
  # pdflatex where the check runs. A TinyTeX installed after this session
  # started is on the PATH of the next session only, so it is looked for where
  # TinyTeX installs: tinytex's own answer, then its default folders.
  latex_bin <- function() {
    on_path <- Sys.which("pdflatex")
    if (nzchar(on_path)) return(dirname(on_path))
    roots <- c(if (requireNamespace("tinytex", quietly = TRUE)) tinytex::tinytex_root(error = FALSE),
               file.path(Sys.getenv("APPDATA"), "TinyTeX"), path.expand("~/.TinyTeX"),
               path.expand("~/Library/TinyTeX"))
    for (r in roots[nzchar(roots)]) {
      for (d in list.dirs(file.path(r, "bin"), recursive = FALSE)) {
        if (any(file.exists(file.path(d, c("pdflatex.exe", "pdflatex"))))) return(d)
      }
    }
    ""
  }
  bin <- latex_bin()
  if (!nzchar(bin)) {
    stop("The PDF manual needs LaTeX, and there is none: no pdflatex on the PATH and no ",
         "TinyTeX. tinytex::install_tinytex() installs one.", call. = FALSE)
  }
  path_before <- Sys.getenv("PATH")
  with_latex  <- paste(normalizePath(bin), path_before, sep = .Platform$path.sep)
  check_env   <- c(check_env, PATH = with_latex)
  Sys.setenv(PATH = with_latex)      # for tinytex's tlmgr below; put back after

  # MAKEINDEX TOO. R builds the manual's index with it, after LaTeX's first
  # pass; TinyTeX does not bring it, and without it the build stops with no
  # LaTeX package to name -- the first run here, 2026-10-04, after 79 pages.
  has_makeindex <- function() any(file.exists(file.path(bin, c("makeindex.exe", "makeindex"))))
  if (!has_makeindex() && requireNamespace("tinytex", quietly = TRUE)) {
    message("makeindex is missing beside pdflatex: installing it with tinytex.")
    tinytex::tlmgr_install("makeindex")
  }
  if (!has_makeindex()) {
    Sys.setenv(PATH = path_before)
    stop("The PDF manual needs makeindex beside pdflatex (", bin, "), and it is not ",
         "there. With TinyTeX: tinytex::tlmgr_install(\"makeindex\").", call. = FALSE)
  }

  manual <- file.path(work, paste0(pkg, "-manual.pdf"))
  tried  <- character(0)
  repeat {
    # --no-clean: the build folder stays, and with it LaTeX's own log.
    m <- callr::rcmd("Rd2pdf", c("--batch", "--no-preview", "--force", "--no-clean",
                                 paste0("--output=", manual), stage),
                     wd = work, env = check_env, fail_on_status = FALSE, show = FALSE)
    if (m$status == 0L && file.exists(manual)) break
    out <- strsplit(paste(m$stdout, m$stderr, sep = "\n"), "\n")[[1]]
    wanted <- if (requireNamespace("tinytex", quietly = TRUE)) {
      setdiff(tinytex::parse_packages(text = out), tried)
    } else character(0)
    if (length(wanted) == 0L) {
      Sys.setenv(PATH = path_before)
      # WHAT FAILED, NOT HOW THE OUTPUT ENDS. The transcript's tail is overfull
      # boxes, and the first version printed that tail and hid the error. Now
      # R's own words (its stderr), LaTeX's "! " lines with their context, and
      # everything, with LaTeX's log, where it outlives tempdir().
      keep_dir <- file.path(root, "outputs", "package_check")
      dir.create(keep_dir, recursive = TRUE, showWarnings = FALSE)
      stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
      kept_out <- file.path(keep_dir, paste0("manual_", stamp, ".out"))
      writeLines(out, kept_out)
      tex_log <- list.files(work, pattern = "^Rd2\\.log$", recursive = TRUE, all.files = TRUE,
                            full.names = TRUE)
      kept_log <- file.path(keep_dir, paste0("manual_", stamp, "_Rd2.log"))
      if (length(tex_log)) file.copy(tex_log[1], kept_log)
      bang  <- grep("^! ", out)
      shown <- out[sort(unique(c(bang, bang + 1L, bang + 2L)))]
      said  <- strsplit(m$stderr, "\n")[[1]]
      stop("The PDF manual does not build, and not for a LaTeX package that could be installed.",
           if (length(said)) paste0("\nR said:\n", paste(utils::tail(said, 15), collapse = "\n")),
           if (length(bang)) paste0("\nLaTeX's errors:\n",
                                    paste(utils::head(shown[!is.na(shown)], 30), collapse = "\n")),
           "\n  All of the output: ", kept_out,
           if (length(tex_log)) paste0("\n  LaTeX's log: ", kept_log) else "", call. = FALSE)
    }
    message("The manual misses the LaTeX package(s) ", paste(wanted, collapse = ", "),
            ": installing them with tinytex, then building it again.")
    tinytex::tlmgr_install(wanted)
    tried <- c(tried, wanted)
  }
  Sys.setenv(PATH = path_before)
  message("PDF manual built on its own (", round(file.size(manual) / 1e3), " KB) with ",
          bin, if (length(tried)) paste0("; LaTeX packages installed for it: ",
                                         paste(tried, collapse = ", ")) else "", ".")
}

message("R CMD check ", paste(check_args, collapse = " "), " ", basename(tgz), " ...")
res <- callr::rcmd("check", c(check_args, basename(tgz)), wd = work, env = check_env,
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
