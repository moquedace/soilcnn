# Unit test: every example of the package runs
#
# WHY THIS FILE EXISTS.
#
# R CMD check runs the examples one after another in one session and stops at
# the first that fails, so one broken example hides every one after it and
# each fix costs a whole check. Here each runs on its own -- from the man/
# pages roxygen wrote, \donttest{} included, as R CMD check --as-cran runs
# them -- and every failure is reported at once, with the time each took.
#
# The examples that need a fitted model share example_run(): the first one to
# run trains it (77 s here, on one core) and the rest reuse it. Its
# workers load this source tree, as the session does.
#
# What each example printed goes to <tempdir>/soilcnn_examples/<topic>.out,
# its messages and warnings to <topic>.msg, and the plots to examples.pdf.
#
# Verified:
#   1. every exported function's page has an example
#   2. each example runs without an error
#
# Run AFTER roxygen2::roxygenise(): the examples are read from man/.
# Run: source("<package root>/tests/test_examples.R")

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
ns <- .load_framework(root)

ok <- logical(0)

ns_lines <- readLines(file.path(root, "NAMESPACE"), warn = FALSE)
exported <- sub("^export\\((.*)\\)$", "\\1", grep("^export\\(", ns_lines, value = TRUE))
rd_files <- sort(list.files(file.path(root, "man"), pattern = "\\.Rd$", full.names = TRUE))

out_dir <- file.path(tempdir(), "soilcnn_examples")
unlink(out_dir, recursive = TRUE)
dir.create(out_dir, recursive = TRUE)

# ── the code of each example, as R CMD check extracts it ─────────────────────
# Rd2ex() writes nothing for a page without an example: a print method's.
code <- vapply(rd_files, function(rd) {
  f <- file.path(out_dir, sub("\\.Rd$", ".R", basename(rd)))
  tools::Rd2ex(rd, out = f, commentDontrun = TRUE, commentDonttest = FALSE)
  if (file.exists(f) && file.size(f) > 0) f else NA_character_
}, character(1))
names(code) <- sub("\\.Rd$", "", basename(rd_files))
code <- code[!is.na(code)]

no_example <- setdiff(exported, names(code))
ok["every_export_has_an_example_page"] <- length(no_example) == 0L

# ── each one, on its own ──────────────────────────────────────────────────────
# In an environment of its own over the global one, where R CMD check runs
# them; every top-level value printed, as there. The plots go to one PDF, and
# torch runs on two threads, as CRAN allows -- the session's count after.
threads_before <- torch::torch_get_num_threads()
torch::torch_set_num_threads(2L)
grDevices::pdf(file.path(out_dir, "examples.pdf"))
pdf_dev <- grDevices::dev.cur()
run_one <- function(topic) {
  msg_file <- file.path(out_dir, paste0(topic, ".msg"))
  note <- function(kind, text) cat(kind, ": ", text, "\n", sep = "", file = msg_file, append = TRUE)
  t0 <- Sys.time()
  err <- tryCatch({
    withCallingHandlers(
      utils::capture.output(
        source(code[[topic]], local = new.env(parent = globalenv()), echo = TRUE,
               max.deparse.length = Inf),
        file = file.path(out_dir, paste0(topic, ".out"))),
      message = function(m) { note("message", conditionMessage(m)); invokeRestart("muffleMessage") },
      warning = function(w) { note("warning", conditionMessage(w)); invokeRestart("muffleWarning") })
    NA_character_
  }, error = function(e) conditionMessage(e))
  secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  %-28s %7.1f s  %s\n", topic, secs, if (is.na(err)) "ok" else "FAILED"))
  data.frame(topic = topic, seconds = secs, error = err, stringsAsFactors = FALSE)
}
cat("  each example, by page name, as R CMD check runs them:\n")
res <- do.call(rbind, lapply(names(code), run_one))
if (pdf_dev %in% grDevices::dev.list()) grDevices::dev.off(pdf_dev)
torch::torch_set_num_threads(threads_before)

for (i in seq_len(nrow(res))) ok[paste0("example_runs_", res$topic[i])] <- is.na(res$error[i])

detail <- character(0)
if (length(no_example)) {
  detail <- c(detail, paste0("  without an example page: ", paste(no_example, collapse = ", ")))
}
failed <- res[!is.na(res$error), , drop = FALSE]
if (nrow(failed)) {
  detail <- c(detail, "  what failed:",
              sprintf("    %s: %s", failed$topic, failed$error),
              paste0("  the output of each: ", out_dir))
}
cat(sprintf("\n  %d example(s) in %.1f min; the slowest: %s\n", nrow(res), sum(res$seconds) / 60,
            paste(sprintf("%s %.0f s", utils::head(res$topic[order(-res$seconds)], 5),
                          utils::head(sort(res$seconds, decreasing = TRUE), 5)), collapse = ", ")))

.report(ok, "test_examples", detail)
