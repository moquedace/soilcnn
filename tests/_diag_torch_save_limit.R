# Where exactly does torch_save() break?
#
# The scale diagnostic showed: 1.41 GB writes, 2.17 GB kills the session.
# 2^31 bytes = 2.147 GB falls exactly inside that window, which points at a
# 32-bit integer overflow in the serializer.
#
# Counter-evidence that has to be explained: patches_w15.pt is 5.98 GB, was
# written by torch_save and READ BACK perfectly (right shape, zero non-finite
# cells out of 1.49 billion). If the limit were 2^31, that file would not
# exist.
#
# This test isolates torch_save ALONE: the tensors come from torch_empty(),
# with no R array anywhere near, so nothing else competes for memory or muddies
# the reading of the result. It sweeps sizes around 2^31 and, at the end, tries
# a LARGE tensor (in the w15 range) to test the counter-evidence.
#
# Log on disk, flushed every line -- the answer survives the crash.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/_diag_torch_save_limit.R")

suppressMessages(library(torch))

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/cnn_architecture.R.
project_root <- (function() {
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
  for (d in cand) for (up in c(".", "..", "../..", "../../..")) {
    r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
    if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
log_file <- file.path(project_root, "outputs", "diag_torch_save_limit.log")
dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)

log_linha <- function(...) {
  linha <- paste0(format(Sys.time(), "%H:%M:%S"), "  ", paste0(..., collapse = ""))
  cat(linha, "\n", sep = "", file = log_file, append = TRUE)
  message(linha)
}

cat("", file = log_file)
log_linha("torch ", as.character(utils::packageVersion("torch")))
log_linha("2^31 bytes = ", format(2^31, big.mark = ","),
          "  (", sprintf("%.3f GB", 2^31 / 1e9), ")")
log_linha("2^32 bytes = ", format(2^32, big.mark = ","),
          "  (", sprintf("%.3f GB", 2^32 / 1e9), ")")

tmp <- file.path(tempdir(), "diag_limit")
dir.create(tmp, showWarnings = FALSE, recursive = TRUE)
f <- file.path(tmp, "t.pt")

# ── one size at a time, tensor created directly by torch ─────────────────────
# torch_empty goes through no R array at all: it isolates the serializer.

testar <- function(n_elem, rotulo) {
  bytes <- n_elem * 4
  log_linha("--- ", rotulo, ": ", format(n_elem, big.mark = ","), " elements = ",
            format(bytes, big.mark = ","), " bytes (",
            sprintf("%.3f GB", bytes / 1e9), ")",
            if (bytes > 2^31) "   ABOVE 2^31" else "   below 2^31")

  x <- torch_empty(n_elem, dtype = torch_float())
  log_linha("    tensor created, writing...")

  if (file.exists(f)) file.remove(f)
  torch_save(x, f)

  got <- file.size(f)
  log_linha("    WROTE -- ", format(got, big.mark = ","), " bytes",
            if (abs(got - bytes) < 5000) "  [size ok]" else
              sprintf("  [EXPECTED %s]", format(bytes, big.mark = ",")))

  rm(x); invisible(gc(verbose = FALSE))
  file.remove(f)
  invisible(TRUE)
}

# Sweep around 2^31. Every step is logged BEFORE it is tried, so the last line
# of the log identifies the size that killed the session.
n_2_31 <- floor(2^31 / 4)   # elements that come to exactly 2^31 bytes

testar(floor(n_2_31 * 0.90), "90% of 2^31")
testar(floor(n_2_31 * 0.99), "99% of 2^31")
testar(n_2_31 - 1000L,       "just BELOW 2^31")
testar(n_2_31 + 1000L,       "just ABOVE 2^31")
testar(floor(n_2_31 * 1.10), "110% of 2^31")

# The counter-evidence: w15 has 1,494,485,325 elements and it wrote. If we get
# here, 2^31 is not the limit and w15 stops being a contradiction.
log_linha("=== the whole 2^31 sweep passed ===")
testar(1494485325L, "w15 size (5.98 GB)")

log_linha("=== EVERYTHING WROTE -- torch_save has no size limit here ===")
log_linha("If this line appears, the defect depends on something else: a live ",
          "R array alongside it, the heap state, or the 4D shape.")

unlink(tmp, recursive = TRUE)
message("\nLog: ", log_file)
