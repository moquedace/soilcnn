# The vignette's validation geometry (designs.png): fold 1 of each of the five
# designs, from a run's own fold plans -- training, validation, the
# calibration set and the test set, on the map. The geometry only, no model
# result: a test run's plans are as true as any (the rule that keeps smoke
# figures out of a commit is for results). Drawn by tools/vignette_figures.R
# with the rest, or alone:
#
#   plans_from <- "sample10"    # the run whose plans are drawn
#   source("D:/usuario_armazenamento/cassio/projects/soilcnn/tools/vignette_designs.R")
#
# Stops, naming what is missing, until the run has all five plans: 03_tune.R
# writes each when that design starts.

if (!exists("root")) root <- "D:/usuario_armazenamento/cassio/projects/soilcnn"
if (!exists("plans_from")) plans_from <- "sample10"
if (!exists("rdir")) rdir <- "D:/usuario_armazenamento/cassio/data/predictors_resolution_250m_latam"
outputs <- "D:/usuario_armazenamento/cassio/projects/soc_stock_0_30cm_lac/outputs"
# Here, not in the global environment: tools/vignette_figures.R sources this
# with helpers of its own.
base::source(file.path(root, "tools", "vignette_style.R"), local = environment())
csv <- function(p) utils::read.csv2(p, stringsAsFactors = FALSE, check.names = FALSE)

methods <- c("spatial", "knndm", "random", "holdout", "region")
plan_files <- file.path(outputs, plans_from, "tuning", methods, "fold_plan.rds")
missing <- methods[!file.exists(plan_files)]
if (length(missing)) {
  stop("The '", plans_from, "' run has no fold plan yet for: ", paste(missing, collapse = ", "),
       ". 03_tune.R writes each when that design starts.", call. = FALSE)
}
sm    <- csv(file.path(outputs, plans_from, "patches", "patch_meta.csv"))
plans <- lapply(plan_files, readRDS)
if (any(vapply(plans, function(p) p$n_rows != nrow(sm), logical(1)))) {
  stop("The fold plans and the run's patch_meta.csv do not hold the same rows.", call. = FALSE)
}
shared <- function(ids) all(vapply(ids, function(x) setequal(x, ids[[1]]), logical(1)))
if (!shared(lapply(plans, function(p) sm$sample_id[p$folds[[1]]$test]))) {
  stop("The five designs do not share one test set.", call. = FALSE)
}
if (!shared(lapply(plans, function(p) sm$sample_id[p$calibration]))) {
  stop("The five designs do not share one calibration set.", call. = FALSE)
}

# The land behind the points: a regular overview of one predictor's valid cells.
land <- as.data.frame(terra::spatSample(terra::rast(file.path(rdir, "bio1.tif")), 200000,
                                        method = "regular", as.raster = TRUE), xy = TRUE)
names(land) <- c("x", "y", "z")
valid <- is.finite(land$z) & land$z > -9990
land_xy <- project_lac(land$x[valid], land$y[valid])

# Each role's mark; drawn in this order, so the held-out sets lie on top.
role_marks <- list(
  train       = list(col = C[["train"]], pch = 16, cex = .38),
  out         = list(col = C[["muted"]], pch = 1,  cex = .65),
  validation  = list(col = C[["blue"]],  pch = 16, cex = .65),
  calibration = list(col = C[["olive"]], pch = 18, cex = .8),
  test        = list(col = C[["earth"]], pch = 17, cex = .65))
map <- function(x0, y0, x1, y1, points_data, roles) {
  xy <- project_lac(points_data$x, points_data$y)
  xr <- range(land_xy[, 1]); yr <- range(land_xy[, 2])
  # Equal x/y scales in the device's inches: normalised coordinates alone do
  # not keep the aspect on a rectangular PNG.
  inches <- par("pin")
  scale <- min((x1 - x0) * inches[1] / diff(xr), (y1 - y0) * inches[2] / diff(yr)) * .98
  sx <- scale / inches[1]; sy <- scale / inches[2]
  ox <- (x0 + x1) / 2 - mean(xr) * sx; oy <- (y0 + y1) / 2 - mean(yr) * sy
  points(ox + land_xy[, 1] * sx, oy + land_xy[, 2] * sy, pch = 15, cex = .31, col = "#E9EEEC")
  for (role in names(role_marks)) {
    q <- which(roles == role); m <- role_marks[[role]]
    points(ox + xy[q, 1] * sx, oy + xy[q, 2] * sy, pch = m$pch, cex = m$cex, col = m$col)
  }
}

run_text <- c(sample10 = "the application run on a tenth of the profiles",
              smoke = "a test run on 1% of the profiles")
from_text <- if (plans_from %in% names(run_text)) run_text[[plans_from]] else paste0("the '", plans_from, "' run")
start("designs", 1800, 1800)
heading("VALIDATION GEOMETRY", "Same observations, different validation questions",
        paste0("Fold 1 of each design, from ", from_text, ". The geometry only; no model results."))
titles <- c("Spatial blocks", "kNNDM", "Random folds", "Holdout", "Ecoregions")
for (i in seq_along(methods)) {
  j <- (i - 1) %% 2; row <- (i - 1) %/% 2; x <- .045 + j * .49; y <- .535 - row * .235
  txt(x, y + .255, paste0(letters[i], "   ", titles[i]), 1.07, bold = TRUE)
  p <- plans[[i]]; f <- p$folds[[1]]
  role <- rep("out", nrow(sm))
  role[f$train] <- "train"; role[f$validation] <- "validation"
  role[p$calibration] <- "calibration"; role[f$test] <- "test"
  map(x + .04, y, x + .275, y + .235, sm, role)
  txt(x + .302, y + .176, paste0(length(f$train), " train"), .74, C["muted"])
  txt(x + .302, y + .142, paste0(length(f$validation), " validate"), .74, C["blue"])
  txt(x + .302, y + .108, paste0(length(p$calibration), " calibrate"), .74, C["olive"])
  txt(x + .302, y + .074, paste0(length(f$test), " test"), .74, C["earth"])
}
legend(.55, .265, legend = c("Training", "Validation", "Shared calibration set", "Shared test set",
                             "Not used in this fold"),
       col = c(C["train"], C["blue"], C["olive"], C["earth"], C["muted"]),
       pch = c(16, 16, 18, 17, 1), bty = "n", cex = .9, y.intersp = 1.6)
txt(.535, .09, paste0(format(nrow(sm), big.mark = ","), " profiles / fold 1\n",
                      "One test set and one calibration set for all five designs"), .79, C["muted"])
sp <- plans[[1]]$params
txt(.04, .03, sprintf(paste("Spatial plan: %s-degree blocks; buffer %s degrees. Maps: spherical",
                            "Lambert azimuthal equal-area, 75 W / 15 S; equal x/y scale."),
                      format(signif(sp$block_size, 3)), format(signif(sp$buffer, 3))), .72, C["muted"])
finish()
writeLines(c(paste("plans:", plans_from), paste("run:", file.path(outputs, plans_from)),
             paste("drawn:", format(Sys.time()))), file.path(records, "designs.source"))
message("designs.png drawn from the '", plans_from, "' run's fold plans, into ", fig)
