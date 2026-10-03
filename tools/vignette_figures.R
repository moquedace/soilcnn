# ══════════════════════════════════════════════════════════════════════════════
# The vignette's figures, drawn once from real data and kept as PNG files in
# vignettes/figures/ (the vignette shows them; it does not compute them).
#
# None of them needs a trained model, so none of them waits for the SOC 0-30
# cm full run (2026-10-02, Cassio: "pelo menos as figuras"):
#
#   workflow.png   the chain, one call per step
#   patches.png    what the network reads: the windows around one profile
#   designs.png    the five validation designs on the same profiles, by fold
#
# The data are the trial's: the full store's profiles over Latin America and
# the Caribbean, 3,000 of them drawn so that the folds can be seen, and the
# cut 250 m rasters. The maps of soil carbon come after the full run.
#
# Run: source("D:/usuario_armazenamento/cassio/projects/soilcnn/tools/vignette_figures.R")
# ══════════════════════════════════════════════════════════════════════════════

root  <- "D:/usuario_armazenamento/cassio/projects/soilcnn"
trial <- "D:/usuario_armazenamento/cassio/projects/soc_stock_0_30cm_lac/outputs/full"
rdir  <- "D:/usuario_armazenamento/cassio/data/predictors_resolution_250m_latam"
fig   <- file.path(root, "vignettes", "figures")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)
suppressMessages(library(soilcnn))
read_csv2_quiet <- function(f) suppressMessages(readr::read_csv2(f, show_col_types = FALSE))

# ── 1. workflow.png ──────────────────────────────────────────────────────────
steps <- list(
  c("dsm_prepare()", "points + rasters", "-> a patch store"),
  c("dsm_load()", "the store, checked", ""),
  c("spatial_cv()", "or another design:", "who trains, who scores"),
  c("dsm_train()", "the tuning,", "side by side"),
  c("dsm_final()", "N seeds, intervals,", "smearing"),
  c("dsm_predict()", "the map, its intervals", "and its AOA"))
grDevices::png(file.path(fig, "workflow.png"), width = 2000, height = 420, res = 200)
graphics::par(mar = c(0, 0, 0, 0))
graphics::plot.new()
graphics::plot.window(xlim = c(0, 6), ylim = c(0, 1))
for (i in seq_along(steps)) {
  x0 <- i - 1 + 0.06
  x1 <- i - 0.06
  graphics::rect(x0, 0.18, x1, 0.82, col = if (i == 3) "#fde7c4" else "#e3eef8",
                 border = "#4a6f8f", lwd = 1.5)
  graphics::text((x0 + x1) / 2, 0.64, steps[[i]][1], font = 2, cex = 0.85, family = "mono")
  graphics::text((x0 + x1) / 2, 0.42, steps[[i]][2], cex = 0.75)
  graphics::text((x0 + x1) / 2, 0.28, steps[[i]][3], cex = 0.75)
  if (i < length(steps)) {
    graphics::arrows(x1 + 0.005, 0.5, x1 + 0.115, 0.5, length = 0.07, lwd = 1.5, col = "#4a6f8f")
  }
}
invisible(grDevices::dev.off())

# ── 2. patches.png ───────────────────────────────────────────────────────────
#
# One profile in the south-east of Brazil, the closest to Piracicaba, and the
# three windows the network reads around it, over two of its 174 channels.
meta <- read_csv2_quiet(file.path(trial, "patches", "patch_meta.csv"))
here <- which.min((meta$x - (-47.65))^2 + (meta$y - (-22.72))^2)
px <- meta$x[here]
py <- meta$y[here]
show <- c(landsat_2020_2025_ndvi = "NDVI (Landsat, 2020-2025)", clay = "Clay content")
grDevices::png(file.path(fig, "patches.png"), width = 2000, height = 1000, res = 200)
graphics::par(mfrow = c(1, 2), mar = c(1, 1, 2.5, 4))
for (nm in names(show)) {
  r <- terra::rast(file.path(rdir, paste0(nm, ".tif")))
  cell <- terra::res(r)[1]
  # The cell the profile falls in, and 12 cells around it.
  cx <- terra::xFromCol(r, terra::colFromX(r, px))
  cy <- terra::yFromRow(r, terra::rowFromY(r, py))
  box <- terra::ext(cx - 12.5 * cell, cx + 12.5 * cell, cy - 12.5 * cell, cy + 12.5 * cell)
  terra::plot(terra::crop(r, box), main = show[[nm]], axes = FALSE,
              col = grDevices::hcl.colors(50, if (nm == "clay") "Oranges" else "Greens", rev = TRUE))
  for (w in c(3, 9, 15)) {
    h <- w / 2 * cell
    graphics::rect(cx - h, cy - h, cx + h, cy + h, border = "black",
                   lwd = c(`3` = 2.5, `9` = 2, `15` = 1.5)[[as.character(w)]],
                   lty = c(`3` = 1, `9` = 2, `15` = 3)[[as.character(w)]])
  }
  graphics::points(cx, cy, pch = 3, cex = 0.8)
}
graphics::legend("bottomleft", inset = c(0, -0.02), xpd = NA, bty = "n", cex = 0.8,
                 lty = c(1, 2, 3), lwd = c(2.5, 2, 1.5),
                 legend = c("3 x 3 (0.75 km)", "9 x 9 (2.25 km)", "15 x 15 (3.75 km)"))
invisible(grDevices::dev.off())

# ── 3. designs.png ───────────────────────────────────────────────────────────
#
# 3,000 of the 25,887 profiles, and the five designs over them with one test
# set for all, as in the trial: the spatial design carves it, in whole blocks,
# and the other four take it as given.
eco <- read_csv2_quiet(file.path(trial, "ecoregions.csv"))
pp  <- read_csv2_quiet(file.path(trial, "predpoints.csv"))
set.seed(20261002)
sub <- meta[sort(sample(nrow(meta), 3000)), , drop = FALSE]
grp <- as.character(eco$eco_id)[match(sub$sample_id, eco$sample_id)]
sp  <- spatial_folds(sub, k = 10, test_frac = 0.15, block_size = 1,
                     buffer = 15 * 0.0022457981, seed = 42)
test_ids <- sub$sample_id[sp$folds[[1]]$test]
plans <- list(
  "Spatial blocks (1 degree, buffered)" = sp,
  "kNNDM (matched to the map)"          = knndm_folds(sub, k = 10, predpoints = pp,
                                                      test_ids = test_ids, seed = 42),
  "Random"                              = random_folds(sub, k = 10, test_ids = test_ids, seed = 42),
  "Holdout (one split)"                 = holdout(sub, validation_frac = 0.25,
                                                  test_ids = test_ids, seed = 42),
  "WWF ecoregions"                      = region_folds(sub, group = grp, k = 10,
                                                       test_ids = test_ids, seed = 42))

# Each profile's role: the fold it validates in, the test set, training only
# (a single split), or left out (a buffer dropped it).
role_of <- function(plan) {
  r <- rep("out", nrow(sub))
  for (j in seq_along(plan$folds)) {
    r[plan$folds[[j]]$train[r[plan$folds[[j]]$train] == "out"]] <- "train"
  }
  for (j in seq_along(plan$folds)) r[plan$folds[[j]]$validation] <- as.character(j)
  r[plan$folds[[1]]$test] <- "test"
  r
}
fold_col <- stats::setNames(grDevices::hcl.colors(10, "Dark 3"), as.character(1:10))
cols <- c(fold_col, train = "grey75", test = "black", out = "grey88")

land <- terra::rast(file.path(rdir, "bio1.tif"))
land <- terra::spatSample(land, 4e5, method = "regular", as.raster = TRUE)
land <- terra::ifel(is.na(land), NA, 1)

grDevices::png(file.path(fig, "designs.png"), width = 2100, height = 1500, res = 200)
graphics::par(mfrow = c(2, 3), mar = c(0.5, 0.5, 2, 0.5))
for (nm in names(plans)) {
  role <- role_of(plans[[nm]])
  terra::plot(land, col = "grey95", legend = FALSE, axes = FALSE, main = nm, cex.main = 0.95)
  o <- order(role == "test")                  # the test set on top
  graphics::points(sub$x[o], sub$y[o], pch = 16, cex = 0.35, col = cols[role[o]])
}
graphics::plot.new()
graphics::legend("center", bty = "n", pch = 16, pt.cex = 1.4, cex = 0.95,
                 col = c(fold_col[c(1, 4, 7, 10)], "grey75", "black", "grey88"),
                 legend = c("validation fold 1", "fold 4", "fold 7", "fold 10 (of 10)",
                            "training only (holdout)", "the common test set",
                            "left out by the buffer"))
invisible(grDevices::dev.off())

message("Figures: ", fig, "\n  ", paste(list.files(fig), collapse = ", "))
