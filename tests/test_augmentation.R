# Unit test: D4 per-sample augmentation (augment_d4_batch in R/utils.R)
#
# Verifies the augmentation is label-preserving and geometrically consistent:
#   1. shape preserved
#   2. centre cell unchanged (the centre-point target must survive every symmetry)
#   3. per-(sample,channel) value multiset preserved (only positions permute)
#   4. BOTH branches receive the SAME symmetry per sample (dual-branch consistency)
#   5. symmetries are drawn PER SAMPLE (not one per batch)
#
# Run: source("tests/test_augmentation.R")   (no GPU needed; runs on CPU)

suppressMessages(library(torch))

# -- project root: works under source() in the console AND under Rscript ------
# commandArgs("--file=") is empty when the file is source()d, so fall back to
# the frame that source() sets up, then to getwd(). Anchored on a file that
# only exists at the project root, so a wrong guess fails loudly here instead
# of silently sourcing nothing.

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
source(file.path(root, "R", "utils.R"))

set.seed(1); torch_manual_seed(1)

N <- 16L; C <- 2L; w1 <- 3L; w2 <- 5L
b1 <- torch_tensor(array(rnorm(N * C * w1 * w1), dim = c(N, C, w1, w1)), dtype = torch_float())
b2 <- torch_tensor(array(rnorm(N * C * w2 * w2), dim = c(N, C, w2, w2)), dtype = torch_float())

out <- augment_d4_batch(list(b1, b2))
a1 <- as.array(b1); a2 <- as.array(b2)
ao1 <- as.array(out[[1]]); ao2 <- as.array(out[[2]])

ok_shape <- identical(dim(ao1), dim(a1)) && identical(dim(ao2), dim(a2))

c1 <- (w1 + 1L) / 2L; c2 <- (w2 + 1L) / 2L
ok_center <- all(abs(ao1[, , c1, c1] - a1[, , c1, c1]) < 1e-5) &&
             all(abs(ao2[, , c2, c2] - a2[, , c2, c2]) < 1e-5)

ok_multiset <- TRUE
for (i in 1:N) for (ch in 1:C) {
  if (!isTRUE(all.equal(sort(as.numeric(a1[i, ch, , ])),
                        sort(as.numeric(ao1[i, ch, , ])), tolerance = 1e-5)))
    ok_multiset <- FALSE
}

match_k <- integer(N); ok_cross <- TRUE
for (i in 1:N) {
  found <- NA_integer_
  for (k in 1:8) {
    cand <- as.array(apply_d4(b1[i, , , , drop = FALSE], k))
    if (max(abs(cand - ao1[i, , , , drop = FALSE])) < 1e-5) { found <- k; break }
  }
  match_k[i] <- found
  if (is.na(found)) { ok_cross <- FALSE; next }
  cand2 <- as.array(apply_d4(b2[i, , , , drop = FALSE], found))
  if (max(abs(cand2 - ao2[i, , , , drop = FALSE])) >= 1e-5) ok_cross <- FALSE
}
n_distinct_k <- length(unique(match_k[!is.na(match_k)]))

ok <- c(
  shape_preserved     = ok_shape,
  center_preserved    = ok_center,
  value_multiset_kept = ok_multiset,
  cross_branch_same_k = ok_cross,
  per_sample_sampling = n_distinct_k > 1
)
cat(sprintf("  (%d distinct symmetries across %d samples)\n", n_distinct_k, N))
.report(ok, "test_augmentation")
