# Unit test: writing a window and reading it back gives exactly the same data
#
# This test exists because of a concrete loss. The previous version of the
# store wrote float32 tensors with torch_save() to save disk. torch_save() in
# R torch 0.17.0 breaks above 2^31 bytes -- and it does not break honestly:
# in one case it produced a file of the RIGHT SIZE whose tail was 4.3 GB of
# zeros. It passed a "no non-finite value" check, because zero is finite. It
# cost ~10 h of extraction and two dead sessions.
#
# The lesson became a rule: no write check may settle for the size of the
# file. It has to read back and compare CONTENT.
#
# Checks:
#   1. the round-trip is exact for every window size
#   2. the size on disk matches what patch_window_bytes() predicts
#   3. load_patch_window() returns float32 with the right shape
#   4. the shape assertions catch a file from another set of points
#   5. safe_torch_save() REFUSES a tensor above 2^31 instead of corrupting it
#   6. a truncated file is detected
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_patch_store_io.R")

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)    # mutate(), in the alignment block at the end
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
.load_framework(root)

set.seed(3)
ok <- logical(0)

store <- file.path(tempdir(), "test_store")
unlink(store, recursive = TRUE)
dir.create(store, recursive = TRUE)

n <- 40L; ch <- 7L

# ── 1-3: round-trip per window ────────────────────────────────────────────────

for (w in c(3L, 9L, 15L)) {
  arr <- array(rnorm(n * ch * w * w), dim = c(n, ch, w, w))

  res <- save_patch_window(arr, store, w)
  ok[sprintf("w%02d_size_ok", w)] <- res$ok

  back <- load_patch_window(store, w, expect_points = n, expect_channels = ch)

  # CONTENT, not size. float32 loses the double's precision, so the comparison
  # is against the same precision round-trip -- and it has to match exactly.
  esperado <- as.array(torch_tensor(arr, dtype = torch_float()))
  ok[sprintf("w%02d_roundtrip_exact", w)] <-
    identical(dim(as.array(back)), dim(esperado)) &&
    max(abs(as.array(back) - esperado)) == 0

  ok[sprintf("w%02d_dtype_float32", w)] <-
    identical(as.character(back$dtype), "Float")

  rm(back); gc(verbose = FALSE)
}

# ── 4: a shape from another set of points is refused ─────────────────────────

ok["wrong_point_refused"] <- inherits(
  try(load_patch_window(store, 3L, expect_points = n + 1L), silent = TRUE),
  "try-error")
ok["wrong_channel_refused"] <- inherits(
  try(load_patch_window(store, 3L, expect_channels = ch + 1L), silent = TRUE),
  "try-error")

# ── 5: safe_torch_save refuses above 2^31 instead of corrupting ──────────────
# The limit is measured, not trusted: 2,147,479,648 bytes wrote and
# 2,147,487,648 killed the session. The guard has to fire BEFORE it tries.

too_big <- torch_empty(floor(2^31 / 4) + 1000L, dtype = torch_float())
ok["safe_torch_save_refuses_above_2_31"] <- inherits(
  try(safe_torch_save(too_big, file.path(store, "must_not_exist.pt")),
      silent = TRUE), "try-error")
ok["safe_torch_save_created_no_file"] <-
  !file.exists(file.path(store, "must_not_exist.pt"))
rm(too_big); gc(verbose = FALSE)

# and a small tensor still passes as usual
pequeno <- torch_empty(1000L, dtype = torch_float())
ok["safe_torch_save_accepts_small"] <- !inherits(
  try(safe_torch_save(pequeno, file.path(store, "ok.pt")), silent = TRUE),
  "try-error")
rm(pequeno); gc(verbose = FALSE)

# ── 6: a truncated file is detected ──────────────────────────────────────────
# The failure mode that went unnoticed was a file of the right size with
# garbage inside. Here the reverse -- the wrong size -- must be caught on read.

f3 <- patch_window_path(store, 3L)
con <- file(f3, "r+b"); truncate(con, 5000L); close(con)
ok["truncated_file_detected"] <- inherits(
  try(load_patch_window(store, 3L), silent = TRUE), "try-error")

unlink(store, recursive = TRUE)

# =============================================================================
# align_points_to_meta()
#
# Stage 02 drops points whose window was not fully valid, so the point table
# always has MORE rows than the store, in a different order. Every fold index
# in this framework is a position in the STORE, so if the alignment is wrong
# the model trains on one point's covariates and is scored against another's
# target -- silently, with no error and a plausible CCC.
#
# It had no test at all. The defect that revealed that: the final check was
# stopifnot(identical(out$sample_id, meta$sample_id)), and identical() compares
# storage TYPE. Both sides come from read_csv2() in this pipeline, so both are
# double and it passed; a point table built in R carries INTEGER ids, and the
# check then failed on 1L vs 1 with a message that named nothing.
# =============================================================================

pm <- tibble::tibble(sample_id = c(4, 1, 7),      # store order, and a subset
                     x = c(40, 10, 70), y = 0)
pt <- tibble::tibble(sample_id = 1:8,             # INTEGER, and more rows
                     v = (1:8) * 10L)

al <- align_points_to_meta(pt, pm)
ok["align_reorders_to_the_store"] <-
  identical(as.numeric(al$sample_id), c(4, 1, 7))
ok["align_carries_the_values"] <- identical(al$v, c(40L, 10L, 70L))
ok["align_drops_what_the_store_dropped"] <- nrow(al) == nrow(pm)

# THE TYPE MUST NOT MATTER. An id is a label; integer 4 and double 4 name the
# same observation, and a framework that refuses one of them breaks for anyone
# who builds their point table in R instead of reading it from a CSV.
ok["align_ignores_integer_vs_double"] <- !inherits(
  try(align_points_to_meta(
        dplyr::mutate(pt, sample_id = as.numeric(sample_id)),
        dplyr::mutate(pm, sample_id = as.integer(sample_id))),
      silent = TRUE), "try-error")

# ...and character ids are a legitimate choice, so they must work too.
ok["align_accepts_character_ids"] <- {
  a <- align_points_to_meta(
    dplyr::mutate(pt, sample_id = as.character(sample_id)),
    dplyr::mutate(pm, sample_id = as.character(sample_id)))
  identical(a$sample_id, c("4", "1", "7"))
}

# A store point with no row in the point table is the real failure, and it must
# say so rather than silently shortening the table.
ok["align_refuses_a_missing_point"] <- inherits(
  try(align_points_to_meta(pt[1:3, ], pm), silent = TRUE), "try-error")

cat(sprintf("  synthetic store     : %d points x %d channels, windows 3/9/15\n", n, ch))
cat(sprintf("  torch_save limit    : %s bytes (2^31)\n",
            format(2^31, big.mark = ",")))
.report(ok, "test_patch_store_io")
