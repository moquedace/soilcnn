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

pkg <- c(
  "sf",
  "terra",
  "dplyr",
  "readr",
  "tibble",
  "janitor",
  "purrr",
  "ggplot2",
  "ps"          # measures free RAM, so dsm_prepare() can size its workers
)

install_load_pkg(pkg)

rm(list = setdiff(ls(), "project_root"))  # keep the root found above
gc()

options(width = 200)

setwd(project_root)
pkgload::load_all(project_root)

# ══════════════════════════════════════════════════════════════════════════════
# 01 -- the SOC points and the 181 predictor rasters become a patch store
#
# WHAT THIS SCRIPT IS NOW.
#
# Stages 01 and 02 used to be two scripts, 1,300 lines between them: 01 read
# the GPKG, extracted the centre values, ran the QC and classified the
# predictors; 02 read what 01 wrote and cut the patches. Both are now one call
# to dsm_prepare() (R/prepare.R), and what is left here is what is specific to
# THIS dataset: where the points and rasters are, which channels are dropped,
# which are percentages, the temperature sentinel, the windows.
#
# _p1_prepare_check.R proved, on 2026-09-26, that dsm_prepare() with these
# settings builds the identical store: the three windows bit for bit, every
# table value for value, 19 of 19 checks. (The one textual difference it
# found: the old two-step path carried target values through a CSV and back,
# and readr's reader moved 13% of them by one unit in the last place. The
# function keeps them in memory, and they now equal the GPKG exactly.)
#
# Stage 02 is folded in here; 02_extract_patches.R now only says so.
# ══════════════════════════════════════════════════════════════════════════════

# ── Run profile ───────────────────────────────────────────────────────────────
#
# "full" uses every profile. "dev" keeps a fraction of them, in whole spatial
# blocks, so the whole pipeline runs end to end in minutes instead of a day.
# A dev number is comparable only with another dev number: what must survive a
# subsample is the SHAPE of a result -- see docs/reference_performance.md.
# The profile is written into the store and 99 announces it, so a dev result
# can never be mistaken for a real one later.
run_profile <- "dev"           # "dev" or "full"

dev_subsample <- list(
  frac       = 0.10,           # of POINTS, not of blocks
  block_size = 2,              # degrees -- same grid the spatial folds use
  seed       = 20260914L
)

# ── Settings ──────────────────────────────────────────────────────────────────

target_col   <- "soc_stock_ton_ha_0_5cm"
target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

soc_gpkg_file <- file.path(project_root, "data", "raw",
                           "wosis_profile_soc_stock_spline_clean_preSpline.gpkg")

# Where the USER's rasters are -- the one path no self-location can find.
predictor_raster_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"

# Channels dropped by hand.
#
# These six are CONSTANT (all zero) at every one of the ~37k profiles, so they
# carry no information for training -- but they are NOT constant over the map:
# they switch on over glaciers, small islands, soil-map no-data and deep ocean
# floor. Their input is always 0 during training, so their gradient is always
# 0, so with weight_decay = 0 their weights stay at random initialisation for
# ever -- and each one then applies an arbitrary, seed-dependent bias to
# exactly the pixels outside anything the network ever saw. Dropping them costs
# nothing and removes that. dsm_prepare()'s channel-risk report names any
# other constant channel; decide about those deliberately.
manual_predictor_drop <- c(
  "geology_ice_and_glaciers",
  "soil_class_fao_islands",
  "soil_class_fao_no_data",
  "terrestrial_habitat_deep_ocean_floor",
  "pnv_moss_and_lichen",
  "pnv_open_forest_deciduous_needleleaf"
)

# Temperature QC: a value at or below this is the rasters' nodata sentinel.
temperature_min_valid_celsius <- -100

# Percentage channels (0-100), as regular expressions over the cleaned names.
# They are divided by 100 instead of z-scored, and clamped into [0, 100]: they
# are interpolated surfaces that overshoot slightly near sharp transitions,
# and turning that overshoot into NA once emptied most of a test tile.
percentage_predictor_patterns <- c(
  "^pnv_",
  "^clay_",
  "^peatland_extent$"
)

# Patch sizes in pixels. A window's ground extent is window x resolution, so
# re-pick them whenever the resolution changes. At 250 m:
#   3x3   ~ 0.75 km  (immediate neighbourhood, land cover, micro-relief)
#   9x9   ~ 2.25 km  (hillslope, drainage, soil-landscape position)
#   15x15 ~ 3.75 km  (local landscape, catchment context, local climate)
window_sizes <- c(3L, 9L, 15L)

# Raster rows per chunk: a performance knob only, the store does not depend
# on it. n_cores NULL = physical cores minus one, fewer if RAM cannot hold
# them (dsm_prepare() prints its plan before it starts).
chunk_nrows <- 1000L
n_cores     <- NULL

# ── Paths ─────────────────────────────────────────────────────────────────────
#
# The three places this pipeline has always used, so every later stage finds
# its inputs where it always did.

output_data_dir     <- file.path(project_root, "data",    "processed", "soc_stock_modeling", target_label)
output_metadata_dir <- file.path(project_root, "outputs", "metadata",  "soc_stock_modeling", target_label)
output_patch_dir    <- file.path(project_root, "outputs", "patches",   "soc_stock_modeling", target_label)
output_figure_dir   <- file.path(project_root, "outputs", "figures",   "soc_stock_modeling", target_label)

if (!file.exists(soc_gpkg_file)) stop("Target GPKG not found: ", soc_gpkg_file, call. = FALSE)

# ── The points ────────────────────────────────────────────────────────────────

soc_sf <- sf::st_read(soc_gpkg_file, quiet = TRUE) %>%
  janitor::clean_names() %>%
  dplyr::mutate(profile_id = as.character(profile_id))

# ── The store ─────────────────────────────────────────────────────────────────
#
# overwrite = TRUE because running this script means rebuilding: the store in
# output_patch_dir is removed before the new one is written. Interrupted, the
# store is gone and this script is the way back.

store <- dsm_prepare(
  points            = soc_sf,
  target            = target_col,
  raster_dir        = predictor_raster_dir,
  windows           = window_sizes,
  store_dir         = output_patch_dir,
  metadata_dir      = output_metadata_dir,
  points_file       = file.path(output_data_dir, "full_modeling_dataset_raw.csv"),
  profile_id        = "profile_id",
  percentage        = percentage_predictor_patterns,
  dummy             = "auto",
  drop              = manual_predictor_drop,
  na_below          = c("surface_temperature_celsius$" = temperature_min_valid_celsius),
  percentage_limits = c(0, 100),
  transform         = "log1p",
  target_min        = 0,       # a stock of 0 t/ha is a missing value here, not a measurement
  subsample         = if (identical(run_profile, "dev")) dev_subsample else NULL,
  target_label      = target_label,
  target_unit       = target_unit,
  chunk_nrows       = chunk_nrows,
  n_cores           = n_cores,
  overwrite         = TRUE)

# ── Figures ───────────────────────────────────────────────────────────────────
#
# The target itself, in both spaces. This stage decides no roles, so there is
# no split to draw -- that is a fold plan's job, in stage 03.

create_output_dirs(output_figure_dir)
points_tbl <- safe_read_csv2(store$points_file)

p_dens_native <- ggplot2::ggplot(points_tbl, ggplot2::aes(x = target_native)) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(x = paste0("SOC stock, ", target_unit), y = "Density",
                title = paste(target_label, "- native distribution")) +
  ggplot2::theme_bw()

p_dens_transform <- ggplot2::ggplot(points_tbl, ggplot2::aes(x = target_transform)) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(x = paste0(store$recipe$transform, "(SOC stock)"), y = "Density",
                title = paste(target_label, "- training-space distribution")) +
  ggplot2::theme_bw()

for (p in list(p_dens_native, p_dens_transform)) print(p)

ggplot2::ggsave(file.path(output_figure_dir, "target_density_native.png"),
                p_dens_native, width = 7, height = 5, dpi = 300)
ggplot2::ggsave(file.path(output_figure_dir, "target_density_log1p.png"),
                p_dens_transform, width = 7, height = 5, dpi = 300)

message("\nNext: 99_check_pipeline.R, then 03_run_tuning.R.")
