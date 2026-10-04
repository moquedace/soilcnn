# ── The example landscape the package ships ──────────────────────────────────
#
# A synthetic data set small enough to install with the package, made by
# data-raw/make_example_landscape.R (in the repository, not in the package):
# what the examples, the fast tests and a computing vignette run on. The
# script's header says what was put in it and why; in short, every part of the
# package has something to find in it -- a one-hot set, a percentage, clustered
# profiles, a right-skewed target, and a signal only a neighbourhood shows.

#' A small synthetic landscape, for the examples and the tests.
#'
#' 80 x 80 cells of 0.0025 degrees (about 250 m) in coordinates of
#' south-eastern Brazil -- the place is borrowed, the landscape is made up --
#' with eight predictors and 160 soil profiles, 90 of them in six survey
#' clusters. The soil organic carbon stock (t/ha, 0-30 cm) was drawn from
#' vegetation, clay, temperature, geology and the topographic position of the
#' cell in its 7 x 7 neighbourhood: a network that reads the neighbourhood has
#' something to find that the centre pixel does not show.
#'
#' @return A list: `raster_dir`, the folder of the eight GeoTIFFs (elevation,
#'   temperature, precipitation, ndvi, clay_pct, and geology as three 0/1
#'   dummies); `profiles`, a data frame of `profile_id`, `x`, `y`, `survey` and
#'   `soc_stock`; and `truth`, the formula the stock was drawn from.
#' @examples
#' ex <- example_landscape()
#' head(ex$profiles)
#' list.files(ex$raster_dir)
#' @export
example_landscape <- function() {
  dir <- system.file("extdata", "landscape", package = "soilcnn")
  if (!nzchar(dir) || !file.exists(file.path(dir, "profiles.csv"))) {
    stop("The example landscape is not in this copy of soilcnn (inst/extdata/landscape).",
         call. = FALSE)
  }
  list(raster_dir = file.path(dir, "rasters"),
       profiles = utils::read.csv(file.path(dir, "profiles.csv"), stringsAsFactors = FALSE),
       truth = paste("log(soc_stock) = 2.6 + 1.5 ndvi + 0.012 clay_pct - 0.03 (temperature - 20)",
                     "- 0.005 position + 0.25 basalt - 0.15 sandstone + a smooth field (sd 0.15)",
                     "+ noise (sd 0.25), where position is the elevation less its 7 x 7 mean"))
}
