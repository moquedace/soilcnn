# Unit test: scaling a patch is the same whether it happens before or after
# extraction -- the equivalence the whole step-3 refactor rests on.
#
# Today script 02 applies the z-score while reading each raster band, so the
# stored patches are already scaled and are therefore tied to one split. The
# refactor stores patches RAW and scales at tensor-build time, which is what
# makes k-fold cross-validation cost one broadcast instead of k re-extractions.
#
# That is only legitimate if the two orders give the same numbers. This test
# proves it on a synthetic raster, so the refactor can be verified without the
# 17 GB of real patches (which no longer exist anyway).
#
# Verifies:
#   1. scale-then-extract == extract-then-scale, for every predictor type
#   2. the same holds through the tensor path (float32), within float32 error
#   3. QC is fold-independent: it gives the same answer on any subset
#   4. fit_scaling() on a subset reproduces a hand-computed mean/sd
#   5. different folds really do produce different scalings (the whole point)
#   6. channel-count mismatch is caught instead of silently broadcasting
#   7. QC order: the NA floor wins over the clamp
#
# Run: source("D:/.../tests/test_preprocess.R")    (CPU, no GPU needed)

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)
})

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
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "patches.R"))
source(file.path(root, "R", "preprocess.R"))

set.seed(11)
results <- logical(0)

# ── a synthetic raster with one channel of each predictor type ────────────────

n_rows <- 30L
n_cols <- 40L
n_cell <- n_rows * n_cols

predictors <- c("elevation", "landsat_surface_temperature_celsius",
                "pnv_shrubs", "soil_class_fao_gleysols")
type_table <- tibble(
  predictor     = predictors,
  is_dummy      = c(FALSE, FALSE, FALSE, TRUE),
  is_percentage = c(FALSE, FALSE, TRUE,  FALSE)
)

cells <- cbind(
  rnorm(n_cell, 800, 250),                    # continuous
  c(rnorm(n_cell - 5, 18, 9), rep(-999, 5)),  # continuous + sensor nodata
  pmin(pmax(rnorm(n_cell, 40, 45), -8), 112), # percentage, overshooting 0/100
  rbinom(n_cell, 1L, 0.3)                     # dummy
)
colnames(cells) <- predictors

# na_below is passed explicitly: make_qc_table() no longer ships a default
# rule about surface temperature, because a physical floor is a property of
# the caller's predictors, not of the framework. This test asserts that
# passing it works -- and, by having failed when the default was removed,
# proved it was really being used.
qc_table <- make_qc_table(
  predictors,
  na_below    = c("surface_temperature_celsius$" = -100),
  clamp_range = "pnv_shrubs"
)

# and with NO rules nothing is removed -- the framework stays neutral
qc_none <- make_qc_table(predictors)
results["no_rules_means_no_qc"] <-
  identical(qc_band_values(cells[, 2L], qc_none[2L, ]), cells[, 2L])

# QC applied band by band, exactly as extraction will do it
cells_qc <- cells
for (i in seq_along(predictors)) {
  cells_qc[, i] <- qc_band_values(cells[, i], qc_table[i, ])
}

# points table used to estimate the scaling (the profile locations)
pt_rows  <- sample(n_cell, 120L)
points   <- as.data.frame(cells_qc[pt_rows, , drop = FALSE])
fold_a   <- 1:80
fold_b   <- 41:120

scaling_a <- fit_scaling(points, type_table, fold_a)
scaling_b <- fit_scaling(points, type_table, fold_b)

# ── 1: scale-then-extract == extract-then-scale (double precision) ────────────

w          <- 5L
centre_row <- c(10L, 20L, 15L)
centre_col <- c(12L, 30L, 25L)
cell_index <- patch_cell_index(centre_row, centre_col, n_cols, w)
n_ch       <- length(predictors)

# OLD order: scale each band first, then assemble
cells_scaled <- cells_qc
for (i in seq_len(n_ch)) {
  cells_scaled[, i] <- (cells_qc[, i] - scaling_a$center[i]) / scaling_a$scale[i]
}
old <- patch_gather(cells_scaled, cell_index, n_ch)
old_arr <- patch_finish(old$values, seq_len(nrow(cell_index)), n_ch, w)

# NEW order: assemble raw, then scale the array
raw <- patch_gather(cells_qc, cell_index, n_ch)
new_arr <- scale_patches_array(
  patch_finish(raw$values, seq_len(nrow(cell_index)), n_ch, w), scaling_a
)

results["array_path_orders_agree"] <- max(abs(old_arr - new_arr)) < 1e-12

# ── 2: same through the tensor path, within float32 error ────────────────────

new_t <- scale_patches(
  torch_tensor(patch_finish(raw$values, seq_len(nrow(cell_index)), n_ch, w),
               dtype = torch_float()),
  scaling_a
)
rel <- max(abs(as.array(new_t) - old_arr)) / max(abs(old_arr))
results["tensor_path_matches_within_f32"] <- rel < 1e-6

# ── 3: QC is fold-independent ────────────────────────────────────────────────
# Running QC on a subset must give the same values as running it on everything
# and then subsetting -- otherwise it could not be baked in at extraction.

sub <- sample(n_cell, 200L)
qc_sub <- cells[sub, 2L]
qc_sub <- qc_band_values(qc_sub, qc_table[2L, ])
results["qc_independent_of_subset"] <-
  identical(qc_sub, cells_qc[sub, 2L])

# ── 4: fit_scaling reproduces a hand-computed mean/sd ────────────────────────

results["fit_scaling_matches_manual"] <-
  isTRUE(all.equal(scaling_a$center[1], mean(points[[1]][fold_a], na.rm = TRUE))) &&
  isTRUE(all.equal(scaling_a$scale[1],  sd(points[[1]][fold_a],  na.rm = TRUE)))

# fixed constants must NOT be estimated
results["percentage_scale_is_fixed_100"] <- scaling_a$scale[3] == 100 &&
                                            scaling_a$center[3] == 0
results["dummy_scale_is_identity"] <- scaling_a$scale[4] == 1 &&
                                      scaling_a$center[4] == 0

# ── 5: different folds give different scalings ───────────────────────────────
# If this ever passed trivially, the refactor would be pointless.

results["folds_give_different_scaling"] <-
  scaling_a$center[1] != scaling_b$center[1] &&
  scaling_a$scale[1]  != scaling_b$scale[1]

# and the fixed ones must NOT move between folds
results["fixed_scalings_stable_across_folds"] <-
  identical(scaling_a$center[3:4], scaling_b$center[3:4]) &&
  identical(scaling_a$scale[3:4],  scaling_b$scale[3:4])

# ── 6: channel mismatch caught ───────────────────────────────────────────────

results["channel_mismatch_errors"] <- inherits(
  try(scale_patches_array(new_arr, scaling_a[1:2, ]), silent = TRUE), "try-error"
)

# ── 7: QC order — NA floor beats the clamp ───────────────────────────────────
# A predictor with both rules must send an out-of-range value to NA, not clamp
# it back into the valid band and hide a broken reading.

rule_both <- tibble(predictor = "x", na_below = 0,
                    clamp_lower = 0, clamp_upper = 100)
results["na_floor_wins_over_clamp"] <-
  is.na(qc_band_values(c(-5, 50, 150), rule_both)[1]) &&
  qc_band_values(c(-5, 50, 150), rule_both)[3] == 100

# sensor nodata really was removed, and nothing else was
results["qc_removed_only_the_nodata"] <-
  sum(is.na(cells_qc[, 2L])) == 5L && !anyNA(cells_qc[, c(1L, 3L, 4L)])

# percentage overshoot was clamped, not discarded
results["percentage_overshoot_clamped"] <-
  min(cells_qc[, 3L]) == 0 && max(cells_qc[, 3L]) == 100 &&
  !anyNA(cells_qc[, 3L])

# ── report ────────────────────────────────────────────────────────────────────

cat(sprintf("  synthetic raster    : %d x %d cells, %d channels (cont/temp/pct/dummy)\n",
            n_rows, n_cols, n_ch))
cat(sprintf("  fold A vs B mu      : %.2f vs %.2f  (channel 1)\n",
            scaling_a$center[1], scaling_b$center[1]))
cat(sprintf("  tensor path rel err : %.3e  (float32)\n", rel))
.report(results, "test_preprocess")
