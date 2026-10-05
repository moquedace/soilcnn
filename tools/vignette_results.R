# The vignette's result figures, from a run of the SOC 0-30 cm trial:
#
#   trial_selection   the spatial design's configurations and the one-SE rule
#   trial_designs     the five designs on the one test set, and the paired
#                     difference in error by whole blocks
#   trial_maps        the median, the width of the 90% interval and the AOA
#   trial_importance  permutation against SHAP by theme, and SHAP maps
#   trial_intervals   the three interval calibrations on the one test set:
#                     coverage against the nominal 90%, and mean width
#
# Drawn from the SMOKE run while the real one runs -- for the layout only: its
# models never learned, every figure says so across its face, and its record
# in tools/figure_inputs/ keeps it out of any commit (test_package_metadata).
# The same code draws the real figures: results_from = "sample10" (or "full").
#
#   results_from <- "smoke"   # before source(), or leave it to the default
#   figures <- "intervals"    # some of them only; rm(figures) for all five
#   maps_from <- "maps/check_tile"   # the maps figure rehearsed on 05a's tile
#   results_dir <- "<a run folder>"  # a run outside outputs/, e.g. an archived one
#   source("D:/usuario_armazenamento/cassio/projects/soilcnn/tools/vignette_results.R")
#
# Reads the run's files directly, not 00_settings.R, which a running stage of
# the trial re-reads. The importance is 07_importance.R's, read from
# <run>/importance/: it stops until 07 has run.

if (!exists("results_from")) results_from <- "smoke"
# Which figures: all five, or some -- figures <- "intervals" draws that one,
# from a run that has not mapped the region or measured the importance yet.
if (!exists("figures")) figures <- c("selection", "designs", "maps", "importance", "intervals")
# Where the spatial design's map is in the run -- "maps", 05b's region, by
# default. maps_from <- "maps/check_tile" rehearses the maps figure on 05a's
# tile before 05b's days end: drawn into <run>/vignette/, never into the
# vignette's figures, and with no record.
if (!exists("maps_from")) maps_from <- "maps"
rehearsal <- !identical(maps_from, "maps")
if (!exists("root")) root <- "D:/usuario_armazenamento/cassio/projects/soilcnn"
trial <- "D:/usuario_armazenamento/cassio/projects/soc_stock_0_30cm_lac"
# results_dir, and not run_dir: a run_dir left in the session by anything else
# would draw the figures from another run, and say nothing.
run   <- if (exists("results_dir")) results_dir else file.path(trial, "outputs", results_from)
stopifnot(dir.exists(run))
base::source(file.path(root, "tools", "vignette_style.R"))
if (requireNamespace("soilcnn", quietly = TRUE)) {
  suppressMessages(library(soilcnn))
} else {
  # Plot cached results without loading the model-training runtime.
  base::source(file.path(root, "R", "metrics.R"))
  base::source(file.path(root, "R", "block_bootstrap.R"))
}
csv <- function(p) utils::read.csv2(p, stringsAsFactors = FALSE)
designs <- c(spatial = "Spatial blocks", knndm = "kNNDM", random = "Random folds",
             holdout = "Holdout", region = "Ecoregions")
kept <- file.path(run, "vignette")
dir.create(kept, showWarnings = FALSE)

# ── 1. The selection: the spatial design's configurations ─────────────────────
if ("selection" %in% figures) {
bc  <- csv(file.path(run, "tuning", "spatial", "comparison", "comparison_by_config.csv"))
sel <- readRDS(file.path(run, "tuning", "spatial", "comparison", "selection.rds"))$config_id[1]
best <- which.max(bc$val_ccc_mean)
threshold <- bc$val_ccc_mean[best] - bc$val_ccc_se[best]
chosen <- match(sel, bc$config_id)
start("trial_selection", 1800, 1150)
heading("MODEL SELECTION / THE APPLICATION",
        "The configurations the spatial design tried",
        sprintf("Mean validation CCC +/- 1 SE over %d folds x %d seeds; the one-SE rule's threshold and choice.",
                max(bc$n_folds), max(bc$n_seeds)))
x0 <- .095; x1 <- .70; y0 <- .22; y1 <- .735
xl <- range(log10(bc$n_params)) + c(-.1, .1)
yl <- range(c(bc$val_ccc_mean - bc$val_ccc_se, bc$val_ccc_mean + bc$val_ccc_se), na.rm = TRUE)
yl <- yl + c(-.05, .05) * diff(yl)
xp <- function(x) x0 + (log10(x) - xl[1]) / diff(xl) * (x1 - x0)
yp <- function(y) y0 + (y - yl[1]) / diff(yl) * (y1 - y0)
rect(x0, yp(max(threshold, yl[1])), x1, y1, col = "#EDF3E9", border = NA)
segments(x0, y0, x1, y0, col = C["muted"], lwd = .8)
for (y in pretty(yl, 5)) if (y >= yl[1] && y <= yl[2]) {
  segments(x0, yp(y), x1, yp(y), col = "#E1E7E5", lwd = .7)
  txt(x0 - .012, yp(y), format(y, scientific = FALSE, drop0trailing = TRUE), .78, C["muted"], adj = 1)
}
# Ticks at 1, 2 and 5 of every decade, so a narrow range still has several.
for (x in as.vector(outer(c(1, 2, 5), 10^(3:9)))) if (log10(x) >= xl[1] && log10(x) <= xl[2]) {
  segments(xp(x), y0, xp(x), y0 - .008, col = C["muted"])
  txt(xp(x), y0 - .029, format(x / 1e6, scientific = FALSE, drop0trailing = TRUE), .78, C["muted"], adj = .5)
}
txt((x0 + x1) / 2, y0 - .078, "Trainable parameters (millions, log scale)", .98, adj = .5)
text(.025, (y0 + y1) / 2, "Validation CCC", srt = 90, cex = .98, col = C["ink"])
segments(x0, yp(threshold), x1, yp(threshold), col = C["olive"], lty = 2, lwd = 1.4)
for (i in seq_len(nrow(bc))) {
  x <- xp(bc$n_params[i]); y <- yp(bc$val_ccc_mean[i])
  se <- bc$val_ccc_se[i] / diff(yl) * (y1 - y0)
  co <- if (i == chosen) C["earth"] else if (i == best) C["blue"] else C["muted"]
  if (is.finite(se)) segments(x, y - se, x, y + se, col = co, lwd = 1.2)
  points(x, y, pch = if (i == chosen) 23 else 21, bg = if (i %in% c(best, chosen)) co else "white",
         col = co, cex = if (i %in% c(best, chosen)) 1.3 else .9, lwd = 1.4)
}
txt(.755, .70, paste("Best mean /", bc$config_id[best]), 1.03, C["blue"], TRUE)
txt(.755, .655, paste0("CCC ", fmt_value(bc$val_ccc_mean[best]), ", SE ", fmt_value(bc$val_ccc_se[best])), .88, C["muted"])
txt(.755, .57, "One-SE threshold", 1.03, C["olive"], TRUE)
txt(.755, .525, fmt_value(threshold), .88, C["muted"])
txt(.755, .44, paste("Selected /", sel), 1.03, C["earth"], TRUE)
txt(.755, .395, paste0("CCC ", fmt_value(bc$val_ccc_mean[chosen]), ", ", fmt_value(bc$n_params[chosen] / 1e6), "M parameters"), .88, C["muted"])
mark_source("trial_selection", results_from, run)
finish()
}

# ── 2. The five designs on the one test set ───────────────────────────────────
if ("designs" %in% figures) {
meta <- csv(file.path(run, "patches", "patch_meta.csv"))
test <- lapply(names(designs), function(d) {
  fr <- file.path(run, "final_model", d)
  cfg <- readRDS(file.path(fr, "comparison", "final_run_summary.rds"))$selected_config_ids[1]
  e <- csv(file.path(fr, cfg, "ensemble_predictions.csv"))
  e <- e[e$dataset_role == "test", c("sample_id", "obs", "pred")]
  e[order(e$sample_id), ]
})
names(test) <- names(designs)
stopifnot(all(vapply(test, function(e) identical(e$sample_id, test$spatial$sample_id), logical(1))))
at  <- match(test$spatial$sample_id, meta$sample_id)
obs <- test$spatial$obs
blk <- equal_area_blocks(meta$x[at], meta$y[at], size_km = 100)
paired <- do.call(rbind, lapply(setdiff(names(designs), "spatial"), function(d) {
  b <- block_bootstrap(obs, test[[d]]$pred, blk, against = test$spatial$pred, metric = "mae",
                       weights = "profile", n_boot = 2000L, seed = 20261004L)
  data.frame(design = d, diff = b$difference, lo = b$ci_low, hi = b$ci_high)
}))
start("trial_designs", 1800, 1450)
heading("VALIDATION DESIGNS / THE APPLICATION",
        "Five designs, one test set",
        sprintf("Each design's final model on the same %s test profiles; the error against the spatial design's, by 100 km blocks.",
                format(length(obs), big.mark = ",")))
lim <- range(c(obs, unlist(lapply(test, `[[`, "pred"))), na.rm = TRUE)
for (i in seq_along(designs)) {
  d <- names(designs)[i]; j <- (i - 1) %% 3; row <- (i - 1) %/% 3
  bx0 <- .07 + j * .31; by0 <- .52 - row * .40; s <- .25
  rect(bx0, by0, bx0 + s, by0 + s, col = C["pale"], border = NA)
  segments(bx0, by0, bx0 + s, by0 + s, col = C["line"], lwd = 1)
  px <- bx0 + (test[[d]]$obs - lim[1]) / diff(lim) * s
  py <- by0 + (test[[d]]$pred - lim[1]) / diff(lim) * s
  points(px, py, pch = 16, cex = .3, col = grDevices::adjustcolor(C["blue"], .35))
  m <- calc_metrics(test[[d]]$obs, test[[d]]$pred)
  txt(bx0, by0 + s + .03, designs[[d]], .95, bold = TRUE)
  txt(bx0, by0 - .03, sprintf("CCC %.2f | MAE %.1f | bias %+.1f", m$ccc, m$mae, m$bias), .68, C["muted"])
}
txt(.07, .03, "Observed (x) against predicted (y), t/ha, on one scale; the line is 1:1.", .7, C["muted"])
# The sixth panel: the paired difference in absolute error, against spatial.
bx0 <- .69; by0 <- .12; s <- .25
txt(bx0, by0 + s + .03, "MAE minus spatial's", .95, bold = TRUE)
xr <- range(c(0, paired$lo, paired$hi)); xr <- xr + c(-.08, .08) * diff(xr)
xq <- function(v) bx0 + (v - xr[1]) / diff(xr) * s
segments(xq(0), by0, xq(0), by0 + s, col = C["muted"], lty = 2)
for (k in seq_len(nrow(paired))) {
  y <- by0 + s - k * s / (nrow(paired) + 1)
  segments(xq(paired$lo[k]), y, xq(paired$hi[k]), y, col = C["earth"], lwd = 2)
  points(xq(paired$diff[k]), y, pch = 16, col = C["earth"])
  txt(bx0 - .008, y, designs[[paired$design[k]]], .62, C["muted"], adj = 1)
}
txt(bx0, by0 - .03, "95% interval from whole 100 km blocks", .62, C["muted"])
mark_source("trial_designs", results_from, run)
finish()
}

# ── 3. The maps: median, interval width, applicability ────────────────────────
if ("maps" %in% figures) {
md <- file.path(run, maps_from, "spatial")
r_med <- terra::rast(file.path(md, "ensemble_median.vrt"))
# A band by what it is, from the map's bands.csv, as 06_compare.R finds them:
# the file names carry the calibration source (aoa_block, ..._block), and a
# map from before 2026-10-05 has no width column -- its method WAS the width,
# and its intervals were all cv.
bands <- csv(file.path(md, "bands.csv"))
if (!"width" %in% names(bands)) {
  bands$width  <- bands$method
  bands$method <- ifelse(bands$kind == "interval", "cv", NA)
}
band_vrt <- function(kind, method = NA, width = NA, side = NA) {
  hit <- bands$kind == kind & (is.na(method) | bands$method %in% method) &
    (is.na(width) | bands$width %in% width) & (is.na(side) | bands$side %in% side) &
    (kind != "interval" | bands$label %in% "pi90")
  if (sum(hit) != 1L) stop("The map has ", sum(hit), " band(s) of kind '", kind, "' here: bands.csv says which.")
  file.path(md, paste0(bands$band[hit], ".vrt"))
}
# The width the text reads: the interval that follows the level and the DI
# (a constant one is flat, but where the floor at 0 cuts it), calibrated on
# the calibration set where the map carries it -- split, the one with a
# guarantee -- and on the folds otherwise.
shown <- if (any(bands$method %in% "split")) "split" else "cv"
r_wid <- terra::rast(band_vrt("interval", shown, "level_di", "upper")) -
  terra::rast(band_vrt("interval", shown, "level_di", "lower"))
r_aoa <- terra::rast(band_vrt("aoa"))
pal_med <- grDevices::hcl.colors(80, "YlGnBu", rev = TRUE)
pal_wid <- grDevices::hcl.colors(80, "YlOrBr", rev = TRUE)
start("trial_maps", 2400, 1150, dir = if (rehearsal) kept else fig)
heading("THE MAP / THE APPLICATION", "What the map says, and where it may be believed",
        sprintf("The spatial design's final model: the ensemble median, the width of its 90%% %s interval, which follows the level and the DI, and the area of applicability.",
                if (shown == "split") "split conformal" else "cross-validated"))
z1 <- raster_panel(r_med, .02, .14, .33, .80, pal_med)
z2 <- raster_panel(r_wid, .345, .14, .655, .80, pal_wid)
raster_panel(r_aoa, .67, .14, .98, .80, c("#DADFDA", C[["olive"]]), classes = c(0, 1),
             method = "near")
colour_bar(.08, .07, .19, pal_med, z1, "SOC stock, 0-30 cm (t/ha)")
colour_bar(.405, .07, .19, pal_wid, z2, "90% interval width (t/ha)")
rect(.73, .07, .745, .084, col = "#DADFDA", border = NA); txt(.75, .077, "outside", .66, C["muted"])
rect(.82, .07, .835, .084, col = C["olive"], border = NA); txt(.84, .077, "inside the AOA", .66, C["muted"])
if (!rehearsal) mark_source("trial_maps", results_from, run)
finish()
if (rehearsal) message("The maps figure, rehearsed on ", maps_from, ": ", file.path(kept, "trial_maps.png"))
}

# ── 4. What the model learned, by theme ───────────────────────────────────────
if ("importance" %in% figures) {
# The importance 07_importance.R measured -- the spatial design's permutation
# and SHAP on the test set, and its SHAP over the map -- read, not measured
# again: the figure shows the numbers the text reads (07's compare_*.csv,
# through tools/vignette_numbers.R), and the run's hours are spent once.
imp_dir <- file.path(run, "importance")
imp_files <- file.path(imp_dir, c("permutation_spatial.rds", "shap_spatial.rds",
                                  file.path("shap_map_spatial", "shap.tif")))
if (!all(file.exists(imp_files))) {
  stop("No importance of the spatial design in ", imp_dir, " yet (missing: ",
       paste(basename(imp_files[!file.exists(imp_files)]), collapse = ", "),
       "): run 07_importance.R first.", call. = FALSE)
}
imp <- list(perm = readRDS(imp_files[1])$table, shap = readRDS(imp_files[2])$table,
            shap_files = imp_files[3])
both <- merge(imp$perm[, c("variable", "importance")], imp$shap[, c("variable", "importance")],
              by = "variable", suffixes = c("_perm", "_shap"))
both <- utils::head(both[order(-both$importance_shap), ], 10)
shap_maps <- terra::rast(imp$shap_files)
top <- utils::head(both$variable[both$variable %in% names(shap_maps)], 3)
start("trial_importance", 2200, 1800)
heading("INTERPRETATION / THE APPLICATION", "Importance and its spatial expression",
        "Aligned rankings in original units; signed SHAP maps below share one symmetric colour scale.")
for (panel in 1:2) {
  x0 <- if (panel == 1) .20 else .64; x1 <- x0 + .30
  vals <- if (panel == 1) both$importance_perm else both$importance_shap
  rr <- range(c(0, vals), na.rm = TRUE); if (diff(rr) == 0) rr <- c(-.001, .001)
  rr <- rr + c(-.04, .08) * diff(rr)
  xp <- function(v) x0 + (v - rr[1]) / diff(rr) * (x1 - x0)
  txt(x0, .79, if (panel == 1) "A / Permutation: change in CCC" else "B / Mean |SHAP| (log1p units)", .92, bold = TRUE)
  for (v in pretty(rr, 4)) if (v >= rr[1] && v <= rr[2]) {
    segments(xp(v), .49, xp(v), .76, col = C["line"])
    txt(xp(v), .47, fmt_value(v), .66, C["muted"], adj = .5)
  }
  segments(xp(0), .49, xp(0), .76, col = C["muted"], lty = 2)
  for (k in seq_len(nrow(both))) {
    y <- .747 - (k - 1) * .026
    if (panel == 1) txt(x0 - .014, y, gsub("_", " ", both$variable[k]), .72, adj = 1)
    segments(xp(0), y, xp(vals[k]), y, col = C["line"], lwd = 2)
    points(xp(vals[k]), y, pch = 16, cex = .95, col = if (panel == 1) C["earth"] else C["blue"])
  }
}
pal_shap <- grDevices::colorRampPalette(c("#1F5673", "#6E9DB5", "#F2EEE6", "#C99372", "#8C4A2F"))(81)
zl <- max(vapply(top, function(n) max(abs(stats::quantile(terra::values(shap_maps[[n]]), c(.02, .98), na.rm = TRUE))), numeric(1)))
if (!is.finite(zl) || zl == 0) zl <- 1e-9
for (k in seq_along(top)) {
  bx0 <- .035 + (k - 1) * .32
  raster_panel(shap_maps[[top[k]]], bx0, .105, bx0 + .30, .42, pal_shap, zlim = c(-zl, zl), smooth = 2L)
  txt(bx0 + .15, .435, gsub("_", " ", top[k]), .92, bold = TRUE, adj = .5)
}
colour_bar(.34, .055, .32, pal_shap, c(-zl, zl), "Signed SHAP (log1p units)")
txt(.04, .014, "Common limits: largest theme-specific 2-98% range; values beyond limits clipped. Display: 2 x 2 means. Blue lowers; earth raises.", .64, C["muted"])
mark_source("trial_importance", results_from, run)
finish()
}

# ── 5. The intervals on the one test set ──────────────────────────────────────
if ("intervals" %in% figures) {
#
# Do the 90% intervals keep their promise? Each design's three calibrations --
# cv, split, CV+ -- at both widths, by point, as dsm_final() checked them on
# the test set (04's intervals_summary.csv): the coverage, left, against the
# 90% promised, and the mean width, right, which is what the coverage costs.
# The rest -- by block, inside and outside the AOA, by level and ecoregion --
# is in the comparison's intervals.csv, and the text reads it.
iv <- csv(file.path(run, "final_model", "intervals_summary.csv"))
iv <- iv[iv$level == "pi90" & iv$weighting == "point", ]
meth <- c(cv = "cv residuals", split = "split (calibration set)", cv_plus = "CV+ (fold models)")
start("trial_intervals", 1800, 1300)
heading("UNCERTAINTY / THE APPLICATION", "Do the 90% intervals keep their promise?",
        sprintf("Coverage and mean width on the same %s test profiles. Open: constant width; filled: following the level and the DI.",
                format(max(iv$n_test, na.rm = TRUE), big.mark = ",")))
ax0 <- .27; ax1 <- .60; bx0 <- .68; bx1 <- .95
y_top <- .75; y_bot <- .13
txt((ax0 + ax1) / 2, .81, "A / Observed coverage", .94, bold = TRUE, adj = .5)
txt((bx0 + bx1) / 2, .81, "B / Mean interval width", .94, bold = TRUE, adj = .5)
n_rows <- length(designs) * length(meth)
gap <- .6                                   # between designs, in rows
step <- (y_top - y_bot) / (n_rows - 1 + gap * (length(designs) - 1))
row_y <- function(i, j) y_top - ((i - 1) * length(meth) + (j - 1) + (i - 1) * gap) * step
cov_lim <- c(min(.6, floor(min(iv$coverage, na.rm = TRUE) * 20) / 20), 1)
wid_lim <- c(0, max(iv$mean_width[is.finite(iv$mean_width)], na.rm = TRUE) * 1.08)
xc <- function(v) ax0 + (v - cov_lim[1]) / diff(cov_lim) * (ax1 - ax0)
xw <- function(v) bx0 + (v - wid_lim[1]) / diff(wid_lim) * (bx1 - bx0)
for (v in seq(cov_lim[1], 1, by = .1)) {
  segments(xc(v), y_bot - .02, xc(v), y_top + .02, col = C["line"], lwd = .8)
  txt(xc(v), y_bot - .045, sprintf("%.0f%%", 100 * v), .66, C["muted"], adj = .5)
}
segments(xc(.9), y_bot - .02, xc(.9), y_top + .03, col = C["olive"], lty = 2, lwd = 1.5)
# Beside the line, under panel A's title: centred above it, the two met.
txt(xc(.9) + .006, y_top + .03, "nominal 90%", .62, C["olive"], TRUE, adj = 0)
for (v in pretty(wid_lim, 4)) if (v >= wid_lim[1] && v <= wid_lim[2]) {
  segments(xw(v), y_bot - .02, xw(v), y_top + .02, col = C["line"], lwd = .8)
  txt(xw(v), y_bot - .045, format(v, big.mark = ","), .66, C["muted"], adj = .5)
}
txt((ax0 + ax1) / 2, y_bot - .085, "Coverage of the test profiles", .8, adj = .5)
txt((bx0 + bx1) / 2, y_bot - .085, "Mean width (t/ha)", .8, adj = .5)
for (i in seq_along(designs)) {
  d <- names(designs)[i]
  if (i > 1) {
    sy <- (row_y(i - 1, 3) + row_y(i, 1)) / 2
    segments(.035, sy, .95, sy, col = C["line"], lwd = .8)
  }
  txt(.035, row_y(i, 2) + .009, designs[[d]], .8, bold = TRUE)
  nn <- unique(iv$n_test[iv$design == d])
  txt(.035, row_y(i, 2) - .016, paste0("n = ", paste(nn, collapse = "/")), .6, C["muted"])
  for (j in seq_along(meth)) {
    y <- row_y(i, j)
    txt(ax0 - .012, y, meth[[j]], .58, C["muted"], adj = 1)
    # The holdout has no CV+, and an empty row would read as a missing value.
    if (!any(iv$design == d & iv$method == names(meth)[j])) {
      txt(ax0 + .008, y, "none: one fold, and CV+ needs two", .58, C["muted"])
      next
    }
    for (w in c("constant", "level_di")) {
      r <- iv[iv$design == d & iv$method == names(meth)[j] & iv$width == w, , drop = FALSE]
      if (nrow(r) != 1L) next
      filled <- identical(w, "level_di")
      co <- if (filled) C["earth"] else C["blue"]
      if (is.finite(r$coverage)) {
        points(xc(r$coverage), y, pch = 21, bg = if (filled) co else "white", col = co, cex = 1.1, lwd = 1.3)
      }
      if (is.finite(r$mean_width)) {
        points(xw(r$mean_width), y, pch = 21, bg = if (filled) co else "white", col = co, cex = 1.1, lwd = 1.3)
      }
    }
  }
}
# The key, stacked under the design names: side by side it ran into the
# coverage axis's title (the smoke's figure, 2026-10-05).
points(.04, .07, pch = 21, bg = "white", col = C["blue"], cex = 1.1, lwd = 1.3)
txt(.052, .07, "constant width", .62, C["muted"])
points(.04, .035, pch = 21, bg = C["earth"], col = C["earth"], cex = 1.1, lwd = 1.3)
txt(.052, .035, "width by the level and the DI", .62, C["muted"])
txt(.68, .014, "Descriptive estimates; no profile-independent confidence intervals.", .59, C["muted"])
mark_source("trial_intervals", results_from, run)
finish()
}

# A rehearsed maps figure said where it went; it is not one of these.
into_vignette <- if (rehearsal) setdiff(figures, "maps") else figures
if (length(into_vignette)) {
  message("Result figures (", paste(into_vignette, collapse = ", "), ") drawn from '", results_from,
          "' into ", fig)
}
