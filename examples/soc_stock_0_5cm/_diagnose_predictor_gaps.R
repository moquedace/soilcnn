project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("terra", "dplyr", "readr", "tibble", "purrr", "stringr", "ggplot2")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)
source(file.path(project_root, "R", "utils.R"))

# ══════════════════════════════════════════════════════════════════════════════
# Diagnostic: why does the final map have gaps (speckle) scattered worldwide?
#
# Hypothesis under test: the pipeline requires ALL 187 predictors to be finite
# over the WHOLE window (up to 15x15 = 225 cells) to predict a pixel.
# If any predictor carries sparse, scattered NA (common in global raster
# products -- mosaic seams, sensor dropouts, data gaps), a single NA pixel in
# that predictor invalidates a neighbourhood of up to 15x15 around it.
# That amplifies sparse NA (barely visible looking at 1 predictor alone) into
# much larger "holes" that are visible in the final map.
#
# This script:
#   1. Samples regular points over LAND (uses a reference layer with complete
#      coverage on land -- e.g. elevation -- as a land/water mask, since
#      running the NA-check on all 187 full predictors, at native resolution,
#      is expensive).
#   2. For each of the 187 predictors, computes the NA fraction OVER LAND.
#   3. Ranks the predictors by NA fraction -- points at the culprit(s).
#   4. Quantifies the window amplification effect: compares
#        - the "naive" valid fraction (centre pixel only, window ignored)
#        - the pipeline's real valid fraction (read from the valid_mask.tif
#          already generated)
#   5. Saves a ranked CSV + a bar chart of the worst predictors.
#
# Usage: adjust `n_samples` and `land_reference_predictor` below if needed.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"
config_id    <- "cfg_022"

# How many points to sample (regular grid over the whole raster). 1-2 million
# is fast (seconds to a few minutes) and statistically representative.
n_samples <- 2000000L

# Predictor used as the "this is land" mask -- it needs complete coverage over
# all dry land (no known gaps). The DEM is the safest choice (global DEM
# products are typically void-filled).
land_reference_pattern <- "^ensemble_digital_terrain_model"

metadata_dir <- file.path(project_root, "outputs", "metadata",
                          "soc_stock_modeling", target_label)
raster_table_file <- file.path(metadata_dir, "raster_table_used.csv")

output_dir <- file.path(project_root, "outputs", "spatial_prediction",
                        "soc_stock_modeling", target_label, config_id)
diag_dir <- file.path(output_dir, "diagnostics")
create_output_dirs(diag_dir)

# ── Load the predictor table ──────────────────────────────────────────────────

raster_table <- readr::read_csv2(raster_table_file, show_col_types = FALSE)
predictor_cols <- raster_table$predictor
n_predictors <- length(predictor_cols)

message("Predictors: ", n_predictors)

missing_files <- raster_table$raster_file[!file.exists(raster_table$raster_file)]
if (length(missing_files) > 0) {
  print(missing_files)
  stop("Some predictor rasters no longer exist.")
}

rast_stack <- terra::rast(raster_table$raster_file)
names(rast_stack) <- predictor_cols

r_nrow <- terra::nrow(rast_stack)
r_ncol <- terra::ncol(rast_stack)
message("Grid: ", r_nrow, " x ", r_ncol, " x ", n_predictors, " bands")

# ── Identify the reference predictor (land mask) ──────────────────────────────

land_ref_idx <- grep(land_reference_pattern, predictor_cols, ignore.case = TRUE)

if (length(land_ref_idx) == 0) {
  message("\nWARNING: no predictor matched the pattern '", land_reference_pattern,
          "'. Available predictors (first 30):")
  print(head(predictor_cols, 30))
  stop("Set 'land_reference_pattern' to a predictor with complete coverage over land.")
}

land_ref_name <- predictor_cols[land_ref_idx[1]]
message("Reference predictor (land mask): ", land_ref_name)

# ── Regular sampling over the WHOLE raster (ocean included, filtered later) ───

message("\nSampling ", format(n_samples, big.mark = ","), " regular points...")
t0 <- Sys.time()

samp <- terra::spatSample(rast_stack, size = n_samples, method = "regular",
                          na.rm = FALSE, values = TRUE, xy = FALSE)

message("Sampling finished in ",
        round(as.numeric(Sys.time() - t0, units = "secs"), 1), "s | ",
        nrow(samp), " points sampled.")

# ── Land mask: points where the reference predictor is finite ─────────────────

is_land <- is.finite(samp[[land_ref_name]])
n_land  <- sum(is_land)
message(sprintf("Points classified as land: %s / %s (%.1f%%)",
                format(n_land, big.mark = ","), format(nrow(samp), big.mark = ","),
                100 * n_land / nrow(samp)))

samp_land <- samp[is_land, , drop = FALSE]

# ── NA fraction per predictor, OVER LAND ──────────────────────────────────────

na_frac <- purrr::map_dbl(predictor_cols, function(p) {
  mean(!is.finite(samp_land[[p]]))
})

na_summary <- tibble::tibble(
  predictor = predictor_cols,
  na_fraction_over_land = na_frac,
  na_pct_over_land = round(100 * na_frac, 4)
) %>%
  dplyr::arrange(dplyr::desc(na_fraction_over_land))

message("\n── Top 20 predictors by NA fraction over land ──────────────────────")
print_wide(head(na_summary, 20), n = 20)

n_offenders <- sum(na_summary$na_fraction_over_land > 0)
message(sprintf("\n%d of %d predictors have at least 1 NA over land in the sample.",
                n_offenders, n_predictors))

# ── Window amplification: naive (1 pixel) vs pipeline (whole window) ──────────
# "Naive": the fraction of land points where ALL 187 predictors are finite
#          IN THAT SINGLE PIXEL (neighbourhood ignored).
# That is the theoretical coverage floor IF the window amplified nothing.
# Compare with the pipeline's real valid_fraction (raster_summary / valid_mask).

all_finite_center <- rowSums(!sapply(predictor_cols, function(p) {
  is.finite(samp_land[[p]])
})) == 0
naive_valid_fraction <- mean(all_finite_center)

message(sprintf(
  "\nNaive coverage (1 pixel, no window): %.2f%% of the sampled land",
  100 * naive_valid_fraction))
message("(compare this number with the pipeline's real valid_fraction, saved in")
message(" outputs/spatial_prediction/.../cfg_022/log/prediction_config*.csv")
message(" or in the merge log: if the final value is MUCH smaller than this,")
message(" window amplification is the dominant cause of the gaps.)")

# ── Save the results ──────────────────────────────────────────────────────────

safe_write_csv2(na_summary, file.path(diag_dir, "predictor_na_fraction_over_land.csv"))

diag_summary <- tibble::tibble(
  n_samples_total = nrow(samp),
  n_samples_land = n_land,
  land_reference_predictor = land_ref_name,
  n_predictors = n_predictors,
  n_predictors_with_any_na_over_land = n_offenders,
  naive_valid_fraction_single_pixel = naive_valid_fraction,
  computed_at = as.character(Sys.time())
)
safe_write_csv2(diag_summary, file.path(diag_dir, "diagnostic_summary.csv"))

# ── Chart of the worst offenders ──────────────────────────────────────────────

top_n <- 25
plot_data <- na_summary %>%
  dplyr::slice_head(n = top_n) %>%
  dplyr::filter(na_fraction_over_land > 0) %>%
  dplyr::mutate(predictor = factor(predictor, levels = rev(predictor)))

if (nrow(plot_data) > 0) {
  p <- ggplot2::ggplot(plot_data, ggplot2::aes(x = na_pct_over_land, y = predictor)) +
    ggplot2::geom_col() +
    ggplot2::labs(
      title = paste0(target_label, " -- predictors with the most NA over land"),
      subtitle = paste0("Sample: ", format(n_land, big.mark = ","), " land points"),
      x = "% NA over land (sampled)", y = NULL
    ) +
    ggplot2::theme_bw()

  print(p)
  ggplot2::ggsave(
    filename = file.path(diag_dir, "predictor_na_fraction_top25.png"),
    plot = p, width = 9, height = 7, dpi = 300
  )
}

message("\n── Diagnostic complete ───────────────────────────────────────────")
message("Results saved in: ", diag_dir)
message("  predictor_na_fraction_over_land.csv -- full ranking of the 187 predictors")
message("  diagnostic_summary.csv -- summary + naive coverage")
message("  predictor_na_fraction_top25.png -- chart of the worst offenders")
