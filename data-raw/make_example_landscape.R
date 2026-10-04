# ── The example landscape: a small synthetic data set the package ships ──────
#
# WHAT IT IS FOR. CRAN asks that the exported functions have examples that
# run, and examples need data; so do a fast test suite inside the package and
# a vignette that computes. The SOC application's data cannot go in a package
# (26,000 profiles, 174 rasters of a continent), and a toy with no structure
# would demonstrate nothing. This builds a landscape small enough to ship and
# made so that each part of the package has something to find:
#
#   80 x 80 cells of 0.0025 degrees (about 250 m), in coordinates of south-
#   eastern Brazil -- the place is borrowed, the landscape is made up;
#   eight predictors, one file each, as dsm_prepare() reads them: elevation,
#   temperature, precipitation, ndvi, clay_pct (a percentage), and geology as
#   three 0/1 dummies (a one-hot set for importance_groups() to find);
#   160 soil profiles, 90 of them in six survey clusters (two in hollows) and
#   70 scattered, so a test set is clustered as real ones are
#   (block_bootstrap());
#   a soil organic carbon stock (t/ha, 0-30 cm) drawn from the predictors,
#   right-skewed as stocks are, so log1p and the smearing factor matter.
#
# THE SIGNAL ONLY A NEIGHBOURHOOD SHOWS. The stock rises in hollows: it
# follows the cell's topographic position, its elevation less the mean of its
# 7 x 7 neighbourhood. A model of the centre pixel sees elevation and not the
# hollow; a network reading a 7 x 7 window can find it. The other terms are
# the usual ones: vegetation (ndvi), clay, temperature, and the geology.
#
#   log(stock) = 2.6 + 1.5 ndvi + 0.012 clay_pct - 0.03 (temperature - 20)
#                - 0.005 position + 0.25 basalt - 0.15 sandstone
#                + a smooth field (sd 0.15) + noise (sd 0.25)
#
# Run once, with soilcnn's working directory as the working directory or the
# path below; it writes inst/extdata/landscape/ and data-raw/example_landscape.png.
# The seed fixes every draw, so a second run writes the same files.
#
# Run: source("D:/usuario_armazenamento/cassio/projects/soilcnn/data-raw/make_example_landscape.R")

root <- "D:/usuario_armazenamento/cassio/projects/soilcnn"
out  <- file.path(root, "inst", "extdata", "landscape")
dir.create(file.path(out, "rasters"), recursive = TRUE, showWarnings = FALSE)
set.seed(20261004)

n    <- 80L
cell <- 0.0025
xmin <- -49.70; ymax <- -20.00
row_i <- matrix(rep(seq_len(n), times = n), n, n)    # 1 at the north
col_j <- matrix(rep(seq_len(n), each = n), n, n)     # 1 at the west

# Gaussian bumps: hills (positive), hollows (negative), smooth fields.
bumps <- function(k, amp, width) {
  ci <- stats::runif(k, 1, n); cj <- stats::runif(k, 1, n)
  a  <- stats::runif(k, amp[1], amp[2]); w <- stats::runif(k, width[1], width[2])
  Reduce(`+`, lapply(seq_len(k), function(b) {
    a[b] * exp(-((row_i - ci[b])^2 + (col_j - cj[b])^2) / (2 * w[b]^2))
  }))
}
smooth_noise <- function(sd) {
  f <- bumps(12, c(-1, 1), c(5, 12))
  sd * (f - mean(f)) / stats::sd(as.vector(f))
}

# Hollows narrow enough to show in a 7 x 7 window: 2 to 4 cells wide and 60
# to 140 m deep. Wider ones, tried first, left the position within +/- 15 m,
# a signal the noise drowned. Two survey clusters are put in them below.
hollow_at <- cbind(stats::runif(5, 12, n - 11), stats::runif(5, 12, n - 11))
hollow_depth <- stats::runif(5, 60, 140)
hollow_width <- stats::runif(5, 2, 4)
hollows <- Reduce(`+`, lapply(seq_len(5), function(b) {
  hollow_depth[b] * exp(-((row_i - hollow_at[b, 1])^2 + (col_j - hollow_at[b, 2])^2) /
                          (2 * hollow_width[b]^2))
}))
elevation <- 380 + 3.5 * (col_j - n / 2) + bumps(6, c(80, 260), c(5, 13)) - hollows
temperature <- 26 - 0.0065 * elevation + smooth_noise(0.4)
precipitation <- 1150 + 5 * (row_i - n / 2) + 0.35 * (elevation - 400) + smooth_noise(60)
ndvi <- 0.2 + 0.7 * stats::plogis(-0.4 + 0.004 * (precipitation - 1150) + smooth_noise(0.9))
seeds <- cbind(stats::runif(5, 1, n), stats::runif(5, 1, n))
nearest <- apply(cbind(as.vector(row_i), as.vector(col_j)), 1, function(p) {
  which.min((seeds[, 1] - p[1])^2 + (seeds[, 2] - p[2])^2)
})
geology <- matrix(c("granite", "basalt", "sandstone", "basalt", "granite")[nearest], n, n)
# pmax() and pmin() keep the attributes of their FIRST argument: the matrix
# goes first, or the result is a vector without its dimensions.
clay_pct <- pmin(pmax(24 + 12 * (geology == "basalt") - 8 * (geology == "sandstone") +
                        smooth_noise(7), 5), 70)

to_raster <- function(m) {
  terra::rast(nrows = n, ncols = n, xmin = xmin, xmax = xmin + n * cell, ymin = ymax - n * cell,
              ymax = ymax, crs = "EPSG:4326", vals = as.vector(t(m)))
}
layers <- list(elevation = elevation, temperature = temperature, precipitation = precipitation,
               ndvi = ndvi, clay_pct = clay_pct,
               geology_basalt = (geology == "basalt") * 1, geology_granite = (geology == "granite") * 1,
               geology_sandstone = (geology == "sandstone") * 1)
rasters <- lapply(layers, to_raster)
for (nm in names(rasters)) {
  is_dummy <- startsWith(nm, "geology_")
  terra::writeRaster(rasters[[nm]], file.path(out, "rasters", paste0(nm, ".tif")), overwrite = TRUE,
                     datatype = if (is_dummy) "INT1U" else "FLT4S", gdal = "COMPRESS=DEFLATE")
}

# The topographic position: elevation less its 7 x 7 mean, the cell included.
position <- as.matrix(rasters$elevation - terra::focal(rasters$elevation, w = 7, fun = "mean",
                                                        na.rm = TRUE), wide = TRUE)

# THE PROFILES: six survey clusters of 15, each within about a kilometre --
# two of them in hollows, as a survey of wetlands would be -- and 70
# scattered; none closer to the edge than five cells, so a 7 x 7 window
# around each lies whole inside the rasters.
cluster_at <- rbind(hollow_at[1:2, ],
                    cbind(stats::runif(4, 10, n - 9), stats::runif(4, 10, n - 9)))
clustered <- do.call(rbind, lapply(seq_len(6), function(k) {
  cbind(cluster_at[k, 1] + stats::rnorm(15, 0, 1.5), cluster_at[k, 2] + stats::rnorm(15, 0, 1.5))
}))
scattered <- cbind(stats::runif(70, 6, n - 5), stats::runif(70, 6, n - 5))
rc <- pmin(pmax(rbind(clustered, scattered), 6), n - 5)     # fractional row, column
ri <- ceiling(rc[, 1]); cj <- ceiling(rc[, 2])
at <- cbind(ri, cj)
spatial_field <- smooth_noise(0.15)
log_stock <- 2.6 + 1.5 * ndvi[at] + 0.012 * clay_pct[at] - 0.03 * (temperature[at] - 20) -
  0.005 * position[at] + 0.25 * (geology[at] == "basalt") - 0.15 * (geology[at] == "sandstone") +
  spatial_field[at] + stats::rnorm(nrow(at), 0, 0.25)
# Each profile lies inside the cell its stock was drawn from, away from the
# cell's edges: the store reads that cell's values at it.
profiles <- data.frame(
  profile_id = sprintf("p%03d", seq_len(nrow(at))),
  x = round(xmin + (cj - 1 + stats::runif(nrow(at), 0.1, 0.9)) * cell, 6),
  y = round(ymax - (ri - 1 + stats::runif(nrow(at), 0.1, 0.9)) * cell, 6),
  survey = c(rep(sprintf("survey_%d", seq_len(6)), each = 15), rep("scattered", 70)),
  soc_stock = round(exp(log_stock), 2))
utils::write.csv(profiles, file.path(out, "profiles.csv"), row.names = FALSE)

# A LOOK AT IT, for whoever made it: every layer, the profiles on the ndvi.
grDevices::png(file.path(root, "data-raw", "example_landscape.png"), width = 1600, height = 900,
               res = 110)
op <- graphics::par(mfrow = c(3, 4), mar = c(2, 2, 2.5, 4))
for (nm in names(rasters)) {
  terra::plot(rasters[[nm]], main = nm, axes = FALSE)
  if (identical(nm, "ndvi")) graphics::points(profiles$x, profiles$y, pch = 16, cex = 0.5)
}
terra::plot(to_raster(position), main = "position (7 x 7, not shipped)", axes = FALSE)
graphics::hist(profiles$soc_stock, breaks = 25, main = "soc_stock (t/ha)", xlab = "")
graphics::par(op)
grDevices::dev.off()

cat(sprintf("Example landscape: %d x %d cells, %d rasters, %d profiles\n", n, n, length(rasters),
            nrow(profiles)))
cat(sprintf("  soc_stock: median %.1f t/ha, mean %.1f, range %.1f to %.1f\n",
            stats::median(profiles$soc_stock), mean(profiles$soc_stock), min(profiles$soc_stock),
            max(profiles$soc_stock)))
cat(sprintf("  files: %.0f KB in %s\n",
            sum(file.size(list.files(out, recursive = TRUE, full.names = TRUE))) / 1024, out))
cat("  look: ", file.path(root, "data-raw", "example_landscape.png"), "\n", sep = "")
