# Unit test: patch geometry is identical on both extraction paths
#
# Uses a SYNTHETIC raster whose every cell is identifiable by construction:
#
#   value[row, col, band] = row * 1e6 + col * 1e3 + band
#
# so the correct content of any patch is known analytically. That lets this
# test catch, in one shot, the four failures that produce a plausible but
# wrong map:
#   • transposed patch (H and W swapped)
#   • row/col swapped in the linear index
#   • off-by-one in the window offsets
#   • channel misalignment
#
# Verifies:
#   1. training-side assembly (band by band)   == analytic truth
#   2. prediction-side assembly (all bands)    == analytic truth
#   3. the two paths agree with each other, exactly
#   4. the full-window validity rule flags exactly the affected centres
#   5. the edge rule matches "every index lands inside the strip"
#
# Parametrised over window size, channel count and centre position, so a
# regression in any one combination fails loudly.
#
# Run: source("tests/test_patch_geometry.R")   (no torch, no GPU, no real data)

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
source(file.path(root, "R", "patches.R"))

# ── synthetic raster ──────────────────────────────────────────────────────────

n_rows <- 40L
n_cols <- 50L

make_cells <- function(n_rows, n_cols, n_ch) {
  cells <- seq_len(n_rows * n_cols)
  row   <- (cells - 1L) %/% n_cols + 1L
  col   <- (cells - 1L) %%  n_cols + 1L
  outer(row * 1e6 + col * 1e3, seq_len(n_ch), "+")
}

# what the patch MUST contain, derived from the construction rule alone
truth_patch <- function(centre_row, centre_col, window_size, n_ch) {
  half <- (window_size - 1L) %/% 2L
  out  <- array(NA_real_, dim = c(length(centre_row), n_ch,
                                  window_size, window_size))
  for (i in seq_along(centre_row)) {
    for (ch in seq_len(n_ch)) {
      for (h in seq_len(window_size)) {
        for (w in seq_len(window_size)) {
          out[i, ch, h, w] <- (centre_row[i] + h - 1L - half) * 1e6 +
                              (centre_col[i] + w - 1L - half) * 1e3 + ch
        }
      }
    }
  }
  out
}

# training side builds [n, C, w, w] one band at a time — stack to compare
assemble_band_by_band <- function(cell_values, cell_index, window_size, n_ch) {
  n   <- nrow(cell_index)
  out <- array(NA_real_, dim = c(n, n_ch, window_size, window_size))
  for (ch in seq_len(n_ch)) {
    out[, ch, , ] <- patch_band_assemble(cell_values[, ch], cell_index,
                                         window_size)$array
  }
  out
}

# prediction side: gather once, then reshape
assemble_all_bands <- function(cell_values, cell_index, window_size, n_ch,
                               keep = NULL) {
  g <- patch_gather(cell_values, cell_index, n_ch)
  if (is.null(keep)) keep <- seq_len(nrow(cell_index))
  patch_finish(g$values, keep, n_ch, window_size)
}

# ── 1-3: both paths against the truth, and against each other ─────────────────

windows  <- c(3L, 5L, 9L, 15L)
channels <- c(1L, 3L, 187L)
results  <- logical(0)
detail   <- character(0)

for (w in windows) {
  half <- (w - 1L) %/% 2L
  # interior, top-left limit, bottom-right limit
  centre_row <- c(20L, half + 1L, n_rows - half)
  centre_col <- c(25L, half + 1L, n_cols - half)

  cell_index <- patch_cell_index(centre_row, centre_col, n_cols, w)

  for (n_ch in channels) {
    cells <- make_cells(n_rows, n_cols, n_ch)
    truth <- truth_patch(centre_row, centre_col, w, n_ch)

    a_train <- assemble_band_by_band(cells, cell_index, w, n_ch)
    a_pred  <- assemble_all_bands(cells, cell_index, w, n_ch)

    ok_train <- identical(dim(a_train), dim(truth)) &&
                max(abs(a_train - truth)) == 0
    ok_pred  <- identical(dim(a_pred), dim(truth)) &&
                max(abs(a_pred - truth)) == 0
    ok_same  <- max(abs(a_train - a_pred)) == 0

    key <- sprintf("w%02d_c%03d", w, n_ch)
    results[paste0("train_vs_truth_", key)] <- ok_train
    results[paste0("pred_vs_truth_",  key)] <- ok_pred
    results[paste0("paths_agree_",    key)] <- ok_same

    if (!ok_train || !ok_pred) {
      bad <- which(a_train != truth)[1]
      detail <- c(detail, sprintf(
        "  %s: first mismatch train=%g pred=%g truth=%g",
        key, a_train[bad], a_pred[bad], truth[bad]
      ))
    }
  }
}

# ── 4: full-window validity flags exactly the affected centres ────────────────

w          <- 9L
half       <- (w - 1L) %/% 2L
centre_row <- c(20L, 30L)
centre_col <- c(25L, 40L)
cell_index <- patch_cell_index(centre_row, centre_col, n_cols, w)
cells      <- make_cells(n_rows, n_cols, 4L)

# poison one cell inside centre 1's window only, in one channel only
poison_idx <- (20L + half - 1L) * n_cols + 25L
cells_na   <- cells
cells_na[poison_idx, 3L] <- NA_real_

valid_all  <- patch_gather(cells_na, cell_index, 4L)$valid
valid_band <- patch_band_assemble(cells_na[, 3L], cell_index, w)$valid
valid_ok   <- identical(valid_all,  c(FALSE, TRUE)) &&
              identical(valid_band, c(FALSE, TRUE)) &&
              all(patch_gather(cells, cell_index, 4L)$valid)

results["validity_flags_affected_centre_only"] <- valid_ok

# an untouched channel must still see centre 1 as valid
results["validity_is_per_channel_before_intersect"] <-
  all(patch_band_assemble(cells_na[, 1L], cell_index, w)$valid)

# keep= must drop exactly the invalid centre and renumber cleanly
kept <- assemble_all_bands(cells_na, cell_index, w, 4L, keep = which(valid_all))
results["keep_drops_invalid_centre"] <-
  identical(dim(kept), c(1L, 4L, w, w)) &&
  max(abs(kept[1, , , ] - truth_patch(30L, 40L, w, 4L)[1, , , ])) == 0

# ── 5: edge rule matches "every index inside the strip" ───────────────────────

edge_ok <- TRUE
for (w in windows) {
  half <- (w - 1L) %/% 2L
  grid <- expand.grid(r = seq_len(n_rows), c = seq_len(n_cols))
  in_b <- patch_centre_in_bounds(grid$r, grid$c, n_rows, n_cols, w)

  idx    <- patch_cell_index(grid$r[in_b], grid$c[in_b], n_cols, w)
  inside <- all(idx >= 1L) && all(idx <= n_rows * n_cols)
  # and the rule must not be needlessly strict: one step out must fail
  just_out <- patch_centre_in_bounds(half, half + 1L, n_rows, n_cols, w)
  edge_ok  <- edge_ok && inside && !just_out
}
results["edge_rule_keeps_every_index_in_strip"] <- edge_ok

# odd-window contract
results["even_window_rejected"] <- inherits(
  try(patch_cell_index(5L, 5L, n_cols, 4L), silent = TRUE), "try-error"
)

# ── report ────────────────────────────────────────────────────────────────────

cat(sprintf("  synthetic raster    : %d x %d cells, value = row*1e6 + col*1e3 + band\n",
            n_rows, n_cols))
cat(sprintf("  combinations tested : %d windows x %d channel counts x 3 positions\n",
            length(windows), length(channels)))

.report(results, "test_patch_geometry", detail)
