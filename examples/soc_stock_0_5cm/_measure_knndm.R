# ══════════════════════════════════════════════════════════════════════════════
# Is kNNDM affordable on this dataset?
#
# NOTHING IS PROMISED BEFORE IT IS MEASURED. kNNDM (Linnenbrink et al., 2024,
# in CAST) builds folds whose nearest-neighbour distance distribution matches
# the one prediction will actually face -- a better-founded spatial CV than
# fixed blocks, which pick a block size by eye.
#
# The catch is geometric, and it is specific to THIS dataset: the points are
# global lon/lat. Nearest-neighbour search in a projected plane is O(n log n)
# with a kd-tree; on the sphere, without a projection, it degrades to full
# pairwise geodesic distances -- O(n^2) both in time and in memory.
#
#   n = 31,000 -> 9.6e8 pairs -> ~7.7 GB for ONE double matrix
#   n =  3,100 ->  9.6e6      -> ~77 MB
#
# So the question is not "is kNNDM good" (it is) but "does it have to be
# spherical here". This script measures both paths and says which.
#
# It trains nothing and writes nothing but a small report.
# ══════════════════════════════════════════════════════════════════════════════

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
source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("dplyr", "readr", "tibble", "sf", "FNN")
install_load_pkg(pkg)

rm(list = setdiff(ls(), "project_root"))  # keep the root found above
gc()

options(width = 200)

setwd(project_root)

pkgload::load_all(project_root)

target_label <- "soc_stock_0_5cm"

patch_dir <- file.path(project_root, "outputs", "patches",
                       "soc_stock_modeling", target_label)
meta_file <- file.path(patch_dir, "patch_meta.csv")

if (!file.exists(meta_file)) {
  stop("patch_meta.csv not found: ", meta_file,
       "\nRun 01_prepare_dataset.R first (it builds the patch store) -- this measures the cost on the ",
       "REAL coordinates, not on a simulation of them.")
}

meta <- readr::read_csv2(meta_file, show_col_types = FALSE)
stopifnot(all(c("x", "y") %in% names(meta)))
n_total <- nrow(meta)

message("Points: ", n_total)
message("Extent: x [", round(min(meta$x), 3), ", ", round(max(meta$x), 3),
        "]  y [", round(min(meta$y), 3), ", ", round(max(meta$y), 3), "]")

is_lonlat <- max(abs(meta$x)) <= 180 && max(abs(meta$y)) <= 90
message("Coordinates look like: ", if (is_lonlat) "LON/LAT (degrees)" else
        "a projected CRS")

# ── 1. The projection question ────────────────────────────────────────────────
#
# An EQUAL-AREA projection is the right one for this: kNNDM compares distance
# DISTRIBUTIONS, so what must be preserved is relative distance over the whole
# extent, not angles. Mollweide is the standard global equal-area choice and is
# what CAST's own global examples use.
#
# Projecting is not a workaround for the cost -- it is the correct thing to do
# anyway. A degree of longitude is 111 km at the equator and 0 km at the pole,
# so a "distance" in degrees over a global point set is not a distance.

proj_time <- NA_real_
if (is_lonlat) {
  t0 <- Sys.time()
  pts <- sf::st_as_sf(meta, coords = c("x", "y"), crs = 4326)
  pts_proj <- sf::st_transform(pts, "+proj=moll +lon_0=0 +datum=WGS84 +units=m")
  xy <- sf::st_coordinates(pts_proj)
  proj_time <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  message(sprintf("\nProjection to Mollweide: %.2f s for %d points",
                  proj_time, n_total))
} else {
  xy <- as.matrix(meta[, c("x", "y")])
}

# ── 2. Planar nearest-neighbour cost, at increasing n ─────────────────────────
#
# This is the operation kNNDM repeats: nearest-neighbour distances within the
# sample, and from prediction locations to the sample. Measured at several
# sizes so the growth is OBSERVED rather than assumed -- a kd-tree is
# O(n log n) in theory and this says what it is here.

sizes <- unique(pmin(n_total, c(1000L, 2500L, 5000L, 10000L, 20000L, n_total)))
sizes <- sizes[sizes >= 500L]

bench <- lapply(sizes, function(n) {
  idx <- sample.int(n_total, n)
  s   <- xy[idx, , drop = FALSE]
  t0  <- Sys.time()
  # k = 2: the first neighbour of a point is itself.
  nn <- FNN::get.knn(s, k = 2L)
  el  <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  tibble::tibble(n = n, seconds = el,
                 median_nn_km = stats::median(nn$nn.dist[, 1]) / 1000)
})
bench <- dplyr::bind_rows(bench)

message("\n-- Planar nearest-neighbour cost (kd-tree) --")
print(bench)

# How it actually grows, between the smallest and largest measured.
if (nrow(bench) > 1L) {
  growth <- log(bench$seconds[nrow(bench)] / bench$seconds[1]) /
            log(bench$n[nrow(bench)] / bench$n[1])
  message(sprintf("\nObserved growth: time ~ n^%.2f  (kd-tree ~n^1, ",
                  growth),
          "pairwise ~n^2)")
}

# ── 3. What the spherical path would cost ─────────────────────────────────────
#
# Not run -- estimated. Running it is the thing this script exists to avoid.

mem_gb <- (n_total^2 * 8) / 1e9
message(sprintf("\nA full pairwise geodesic matrix at n = %d would need ",
                n_total),
        sprintf("%.1f GB\nfor one double matrix (before any copy).", mem_gb))

# ── 4. Is CAST here at all? ───────────────────────────────────────────────────

has_cast <- requireNamespace("CAST", quietly = TRUE)
message("\nCAST installed: ", has_cast)

if (has_cast) {
  # knndm on the PROJECTED points, on a subsample, to confirm it runs and to
  # time it honestly. A subsample first: a function that has never run here
  # should not be met at full size.
  n_try <- min(5000L, n_total)
  idx   <- sample.int(n_total, n_try)
  tp    <- sf::st_as_sf(as.data.frame(xy[idx, , drop = FALSE]),
                        coords = c("X", "Y"),
                        crs = "+proj=moll +lon_0=0 +datum=WGS84 +units=m")
  t0 <- Sys.time()
  res <- try(CAST::knndm(tpoints = tp, k = 3L), silent = TRUE)
  el <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  if (inherits(res, "try-error")) {
    message("knndm failed on ", n_try, " points:\n", as.character(res))
  } else {
    message(sprintf("knndm on %d projected points: %.1f s", n_try, el))
    message(sprintf("  -> extrapolated to %d at n^2: %.1f min",
                    n_total, el * (n_total / n_try)^2 / 60))
    message(sprintf("  -> extrapolated to %d at n log n: %.1f min",
                    n_total,
                    el * (n_total * log(n_total)) /
                         (n_try * log(n_try)) / 60))
    print(table(res$clusters))
  }
} else {
  message("  install.packages(\"CAST\")   # to run the last part")
}

# ── 5. The verdict ────────────────────────────────────────────────────────────

message("\n", strrep("=", 78))
message("VERDICT")
message(strrep("=", 78))
if (is_lonlat) {
  message(
    "The coordinates are lon/lat, so kNNDM must NOT be run on them directly:\n",
    "a degree of longitude is 111 km at the equator and 0 at the pole, and a\n",
    "distance computed over that is not a distance. Project to an equal-area\n",
    "CRS first (Mollweide above). That is correct regardless of cost -- and it\n",
    "is also what turns the O(n^2) spherical path into an O(n log n) planar\n",
    "one, which the timings above quantify.")
} else {
  message("The coordinates are already projected; the planar timings apply.")
}
message(
  "\nWhat this does NOT settle: whether kNNDM folds are BETTER than the block\n",
  "folds in use. That is an empirical question and costs a full run to answer,\n",
  "which is why it is not being spent before the pipeline is stable.")
