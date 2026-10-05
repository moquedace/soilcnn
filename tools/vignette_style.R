# The vignette's figure style, for the scripts that draw from a run's results.
# The same palette, font, canvas and heading as tools/vignette_figures.R --
# copied, not sourced, because that script draws its own figures as it is
# read, from the full store, which a results figure does not need.

if (!exists("root")) root <- "D:/usuario_armazenamento/cassio/projects/soilcnn"
fig <- file.path(root, "vignettes", "figures")
records <- file.path(root, "tools", "figure_inputs")
dir.create(fig, recursive = TRUE, showWarnings = FALSE)
dir.create(records, recursive = TRUE, showWarnings = FALSE)
C <- c(ink = "#253840", muted = "#63767D", blue = "#246589", olive = "#60734F",
       earth = "#AB6645", pale = "#F3F6F5", line = "#DCE4E3", train = "#A9B6BA")
if (.Platform$OS.type == "windows") windowsFonts(editorial = windowsFont("Segoe UI"))
font <- if (.Platform$OS.type == "windows") "editorial" else "sans"
# `dir`: the vignette's figures unless a rehearsal draws elsewhere.
start <- function(name, w = 1800, h = 1100, dir = fig) {
  png(file.path(dir, paste0(name, ".png")), w, h, res = 180,
      type = if (.Platform$OS.type == "windows") "windows" else "cairo")
  par(mar = rep(0, 4), family = font, fg = C["ink"], col = C["ink"], xpd = NA)
  plot.new(); plot.window(c(0, 1), c(0, 1), xaxs = "i", yaxs = "i")
}
txt <- function(x, y, s, size = 1, col = C["ink"], bold = FALSE, adj = 0)
  text(x, y, s, cex = size, col = col, font = if (bold) 2 else 1, adj = adj)
heading <- function(k, title, sub) {
  txt(.04, .953, k, .76, C["blue"], TRUE); txt(.04, .902, title, 1.65, bold = TRUE)
  txt(.04, .856, sub, .86, C["muted"])
}
finish <- function() invisible(dev.off())

# A figure drawn from a run that is not the result -- the smoke, whose models
# never learned -- says so across its face, and its record says so to
# tests/test_package_metadata.R, which refuses to let it into a commit.
mark_source <- function(name, source, run_dir) {
  if (identical(source, "smoke")) {
    text(.5, .45, "SMOKE RUN -- LAYOUT ONLY", srt = 25, cex = 4.2, font = 2,
         col = grDevices::adjustcolor(C["earth"], .09))
  }
  writeLines(c(paste("source:", source), paste("run:", run_dir),
               paste("drawn:", format(Sys.time()))),
             file.path(records, paste0(name, ".source")))
}

# Spherical Lambert azimuthal equal-area, centred on Latin America (75 W, 15 S),
# as in the other figures.
crs_lac <- "+proj=laea +lat_0=-15 +lon_0=-75 +R=6371008.8 +units=m +no_defs"
project_lac <- function(lon, lat) {
  rad <- pi / 180; lambda <- (lon + 75) * rad; phi <- lat * rad; phi0 <- -15 * rad
  k <- sqrt(2 / (1 + sin(phi0) * sin(phi) + cos(phi0) * cos(phi) * cos(lambda)))
  r <- 6371008.8
  cbind(x = r * k * cos(phi) * sin(lambda),
        y = r * k * (cos(phi0) * sin(phi) - sin(phi0) * cos(phi) * cos(lambda)))
}

# A raster drawn into the canvas box (x0, y0, x1, y1) at equal x/y scale:
# sampled to about `cells` cells, projected, coloured, and set as an image.
raster_panel <- function(r, x0, y0, x1, y1, palette, zlim = NULL, classes = NULL,
                         cells = 250000, method = "bilinear", smooth = 1L) {
  s <- terra::spatSample(r, cells, method = "regular", as.raster = TRUE)
  # smooth > 1: the mean of smooth x smooth cells, for a surface read from
  # scattered points (SHAP every 200 cells), which otherwise shows as noise.
  if (smooth > 1L) s <- terra::aggregate(s, smooth, fun = "mean", na.rm = TRUE)
  # Trimmed to where there is data, so the land fills the box it is given.
  p <- terra::trim(terra::project(s, crs_lac, method = method))
  m <- terra::as.matrix(p, wide = TRUE)
  if (is.null(classes)) {
    zlim <- zlim %||% stats::quantile(m, c(.02, .98), na.rm = TRUE, names = FALSE)
    k <- 1 + floor((pmin(pmax(m, zlim[1]), zlim[2]) - zlim[1]) / diff(zlim) * (length(palette) - 1))
    col <- palette[k]
  } else {
    col <- palette[match(m, classes)]
  }
  col[is.na(m)] <- NA
  img <- matrix(col, nrow(m), ncol(m))
  e <- terra::ext(p); w <- e$xmax - e$xmin; h <- e$ymax - e$ymin
  inches <- par("pin")
  sc <- min((x1 - x0) * inches[1] / w, (y1 - y0) * inches[2] / h)
  bw <- w * sc / inches[1]; bh <- h * sc / inches[2]
  cx <- (x0 + x1) / 2; cy <- (y0 + y1) / 2
  rasterImage(as.raster(img), cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2,
              interpolate = FALSE)
  invisible(zlim)
}
colour_bar <- function(x0, y, w, palette, zlim, label, digits = 0) {
  n <- length(palette)
  rect(x0 + (seq_len(n) - 1) * w / n, y, x0 + seq_len(n) * w / n, y + .014,
       col = palette, border = NA)
  txt(x0, y - .02, format(round(zlim[1], digits), nsmall = digits), .62, C["muted"])
  txt(x0 + w, y - .02, format(round(zlim[2], digits), nsmall = digits), .62, C["muted"], adj = 1)
  txt(x0 + w / 2, y + .034, label, .7, C["muted"], adj = .5)
}
