source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c(
  "sf",
  "terra",
  "dplyr",
  "tidyr",
  "readr",
  "tibble",
  "janitor",
  "purrr",
  "ggplot2"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))

set.seed(123)

# ── Settings ──────────────────────────────────────────────────────────────────

target_col   <- "soc_stock_ton_ha_0_5cm"
target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"

train_fraction      <- 0.70
validation_fraction <- 0.15
test_fraction       <- 0.15
split_n_bins        <- 10

soc_gpkg_file <- file.path(
  project_root,
  "data", "processed", "soc_stock_modeling", "full_data",
  "wosis_profile_soc_stock_spline_clean_preSpline.gpkg"
)

predictor_raster_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"

# Predictors to drop manually (leave empty to use all)
manual_predictor_drop <- character(0)

# Temperature QC: values below this threshold are set to NA
temperature_min_valid_celsius <- -100

# Regex patterns that identify percentage predictors (bounded 0–100)
# These are scaled to [0, 1] by dividing by 100 (instead of z-score)
percentage_predictor_patterns <- c(
  "^pnv_",
  "^clay_",
  "^peatland_extent$"
)

# Predictors that look like dummies (only 0/1 values) but should be
# treated as continuous/percentage — override the auto-detection
force_as_percentage <- janitor::make_clean_names(c(
  "pnv_moss_and_lichen",
  "pnv_open_forest_deciduous_needleleaf"
))

stopifnot(
  abs(train_fraction + validation_fraction + test_fraction - 1) < 1e-8
)

# ── Paths ─────────────────────────────────────────────────────────────────────

output_data_dir     <- file.path(project_root, "data",    "processed", "soc_stock_modeling", target_label)
output_metadata_dir <- file.path(project_root, "outputs", "metadata",  "soc_stock_modeling", target_label)
output_figure_dir   <- file.path(project_root, "outputs", "figures",   "soc_stock_modeling", target_label)

create_output_dirs(c(output_data_dir, output_metadata_dir, output_figure_dir))

# ── Validation ────────────────────────────────────────────────────────────────

if (!file.exists(soc_gpkg_file)) {
  stop("Target GPKG not found: ", soc_gpkg_file)
}

if (!dir.exists(predictor_raster_dir)) {
  stop("Predictor raster directory not found: ", predictor_raster_dir)
}

# ── Read target GPKG ──────────────────────────────────────────────────────────

soc_sf <- sf::st_read(soc_gpkg_file, quiet = TRUE) %>%
  janitor::clean_names() %>%
  dplyr::mutate(profile_id = as.character(profile_id))

if (!target_col %in% names(soc_sf)) {
  stop("Target column '", target_col, "' not found in GPKG.")
}

message("Profiles read: ", nrow(soc_sf))

# ── List raster predictors ────────────────────────────────────────────────────
# Ignores any subdirectory (e.g. 'nused') — only .tif files at the root level

raster_files_all <- list.files(
  predictor_raster_dir,
  pattern   = "\\.tif$",
  full.names = TRUE,
  recursive  = FALSE
)

if (length(raster_files_all) == 0) {
  stop("No .tif files found in: ", predictor_raster_dir)
}

raster_table_all <- tibble::tibble(
  raster_file     = raster_files_all,
  raster_name_raw = tools::file_path_sans_ext(basename(raster_files_all)),
  predictor       = janitor::make_clean_names(raster_name_raw)
)

dup_predictors <- dplyr::count(raster_table_all, predictor) %>%
  dplyr::filter(n > 1)

if (nrow(dup_predictors) > 0) {
  print(dup_predictors)
  stop("Duplicated predictor names after clean_names(). Rename the raster files.")
}

manual_predictor_drop <- janitor::make_clean_names(manual_predictor_drop)
drop_present <- intersect(manual_predictor_drop, raster_table_all$predictor)
drop_missing <- setdiff(manual_predictor_drop, raster_table_all$predictor)

if (length(drop_present) > 0) {
  message("Dropping manually flagged predictors: ", paste(drop_present, collapse = ", "))
}
if (length(drop_missing) > 0) {
  message("Manual drop entries not matched in rasters: ", paste(drop_missing, collapse = ", "))
}

raster_table_use <- raster_table_all %>%
  dplyr::filter(!predictor %in% drop_present) %>%
  dplyr::arrange(predictor)

predictor_cols_final <- raster_table_use$predictor

message("Predictor rasters to use: ", nrow(raster_table_use))

# ── Load raster stack ─────────────────────────────────────────────────────────

predictor_rasters <- terra::rast(raster_table_use$raster_file)
names(predictor_rasters) <- raster_table_use$predictor

if (!is.na(terra::crs(predictor_rasters)) == FALSE) {
  stop("Predictor rasters have no valid CRS.")
}

# ── Extract predictor values at profile locations ─────────────────────────────

soc_projected <- sf::st_transform(
  soc_sf,
  crs = sf::st_crs(terra::crs(predictor_rasters, proj = TRUE))
)

coords <- sf::st_coordinates(soc_projected)

soc_points <- soc_projected %>%
  dplyr::mutate(
    x = as.numeric(coords[, 1]),
    y = as.numeric(coords[, 2])
  ) %>%
  dplyr::select(profile_id, x, y, dplyr::all_of(target_col))

predictor_values <- terra::extract(
  predictor_rasters,
  terra::vect(soc_points),
  ID = FALSE
) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric))

dataset_extracted <- soc_points %>%
  sf::st_drop_geometry() %>%
  dplyr::bind_cols(predictor_values) %>%
  dplyr::mutate(
    profile_id     = as.character(profile_id),
    target_native  = as.numeric(.data[[target_col]]),
    target_log1p   = log1p(target_native)
  )

# ── Predictor QC ──────────────────────────────────────────────────────────────

# Identify predictors that match percentage patterns
percentage_predictor_cols <- predictor_cols_final[
  purrr::map_lgl(predictor_cols_final, function(x) {
    any(grepl(paste(percentage_predictor_patterns, collapse = "|"), x))
  })
]

# Temperature: values below threshold are physically invalid → NA
temperature_cols <- grep(
  "surface_temperature_celsius$", predictor_cols_final, value = TRUE
)

dataset_qc <- dataset_extracted

if (length(temperature_cols) > 0) {
  dataset_qc <- dataset_qc %>%
    dplyr::mutate(dplyr::across(
      dplyr::all_of(temperature_cols),
      ~ dplyr::if_else(!is.na(.x) & is.finite(.x) & .x <= temperature_min_valid_celsius,
                       NA_real_, as.numeric(.x))
    ))
}

# Percentage predictors: clamp out-of-range values into [0, 100].
# These predictors are continuous/interpolated probability-like surfaces
# (e.g. PNV classes, clay mineralogy) that legitimately overshoot slightly
# below 0 or above 100 near sharp spatial transitions (Gibbs-like ringing
# from whatever smoothing produced them) — not sensor error, not missing
# data. Treating that overshoot as NA (old behaviour) discarded genuinely
# valid near-boundary signal and, downstream, blew up into large windowed
# gaps once the CNN's full-window validity rule amplified each discarded
# pixel into its surrounding patch footprint. Clamping preserves the value
# (effectively ~0% or ~100%) instead of manufacturing missingness that
# was never really there. Genuine NA/Inf are left untouched.
if (length(percentage_predictor_cols) > 0) {
  dataset_qc <- dataset_qc %>%
    dplyr::mutate(dplyr::across(
      dplyr::all_of(percentage_predictor_cols),
      ~ dplyr::if_else(!is.na(.x) & is.finite(.x),
                       pmin(pmax(as.numeric(.x), 0), 100), as.numeric(.x))
    ))
}

# Flag rows with any problem in target or predictors
qc_flags <- dataset_qc %>%
  dplyr::mutate(
    problem_target = is.na(profile_id) | is.na(x) | is.na(y) |
      !is.finite(x) | !is.finite(y) |
      is.na(target_native) | !is.finite(target_native) | target_native <= 0 |
      is.na(target_log1p)  | !is.finite(target_log1p),
    problem_predictor = !dplyr::if_all(
      dplyr::all_of(predictor_cols_final),
      ~ !is.na(.x) & is.finite(.x)
    )
  )

qc_summary <- qc_flags %>%
  dplyr::summarise(
    n_rows_extracted   = dplyr::n(),
    n_target_problem   = sum(problem_target,   na.rm = TRUE),
    n_predictor_problem = sum(problem_predictor, na.rm = TRUE),
    n_any_problem      = sum(problem_target | problem_predictor, na.rm = TRUE),
    pct_any_problem    = round(100 * n_any_problem / n_rows_extracted, 2),
    n_rows_after_qc    = n_rows_extracted - n_any_problem
  )

message("\n── QC summary ──────────────────────────────")
print(qc_summary)

dataset_model_raw <- qc_flags %>%
  dplyr::filter(!problem_target, !problem_predictor) %>%
  dplyr::select(-problem_target, -problem_predictor) %>%
  dplyr::select(
    profile_id, x, y,
    dplyr::all_of(target_col), target_native, target_log1p,
    dplyr::all_of(predictor_cols_final)
  ) %>%
  dplyr::distinct(profile_id, .keep_all = TRUE)

if (nrow(dataset_model_raw) == 0) {
  stop("No rows remained after QC.")
}

message("Rows after QC: ", nrow(dataset_model_raw))

# ── Predictor type classification ─────────────────────────────────────────────
# dummy      : only values 0 and 1 → no scaling needed
# percentage : bounded [0, 100] → scale to [0, 1] by dividing by 100
# continuous : z-score from train set statistics

predictor_type_table <- purrr::map_dfr(predictor_cols_final, function(nm) {
  x      <- dataset_model_raw[[nm]]
  x_use  <- x[!is.na(x) & is.finite(x)]
  uniq   <- sort(unique(x_use))
  tibble::tibble(
    predictor     = nm,
    n_unique      = length(uniq),
    min_value     = min(x_use, na.rm = TRUE),
    max_value     = max(x_use, na.rm = TRUE),
    is_dummy      = length(uniq) <= 2 && all(uniq %in% c(0, 1)),
    is_percentage = nm %in% percentage_predictor_cols
  )
})

# Override: some predictors look like dummies by value range but should
# be treated as percentages (e.g., rare PNV classes with mostly 0/1 but
# the variable represents a continuous probability)
predictor_type_table <- predictor_type_table %>%
  dplyr::mutate(
    is_dummy = dplyr::if_else(predictor %in% force_as_percentage, FALSE, is_dummy),
    is_percentage = dplyr::if_else(predictor %in% force_as_percentage, TRUE, is_percentage)
  )

predictor_cols_dummy      <- dplyr::filter(predictor_type_table, is_dummy)$predictor
predictor_cols_percentage <- dplyr::filter(predictor_type_table, is_percentage)$predictor
predictor_cols_continuous <- dplyr::filter(predictor_type_table, !is_dummy, !is_percentage)$predictor

message(
  "\nPredictor types — dummy: ", length(predictor_cols_dummy),
  " | percentage: ", length(predictor_cols_percentage),
  " | continuous: ", length(predictor_cols_continuous)
)

# ── Stratified train/validation/test split ────────────────────────────────────
# Bins by target quantile → samples within each bin are randomly assigned
# to splits in the requested proportions. This ensures that the distribution
# of the target variable is similar across splits (important for skewed SOC).

dataset_model_split <- dataset_model_raw %>%
  dplyr::mutate(
    sample_id = dplyr::row_number(),
    split_bin = dplyr::ntile(target_native, split_n_bins)
  ) %>%
  dplyr::group_by(split_bin) %>%
  dplyr::mutate(
    split_order        = sample(dplyr::n()),
    n_bin              = dplyr::n(),
    n_train_bin        = floor(n_bin * train_fraction),
    n_validation_bin   = floor(n_bin * validation_fraction),
    dataset_role = dplyr::case_when(
      split_order <= n_train_bin                         ~ "train",
      split_order <= n_train_bin + n_validation_bin      ~ "validation",
      TRUE                                               ~ "test"
    )
  ) %>%
  dplyr::ungroup() %>%
  dplyr::select(-split_order, -n_bin, -n_train_bin, -n_validation_bin)

train_raw      <- dplyr::filter(dataset_model_split, dataset_role == "train")
validation_raw <- dplyr::filter(dataset_model_split, dataset_role == "validation")
test_raw       <- dplyr::filter(dataset_model_split, dataset_role == "test")

# ── Scaling ───────────────────────────────────────────────────────────────────
# All scaling parameters are derived from the TRAIN set only and applied
# identically to validation and test to prevent data leakage.

predictor_scaling <- predictor_type_table %>%
  dplyr::mutate(
    scaling_method = dplyr::case_when(
      predictor %in% predictor_cols_dummy      ~ "none_dummy_0_1",
      predictor %in% predictor_cols_percentage ~ "percentage_0_100_to_0_1",
      TRUE                                     ~ "zscore_train"
    ),
    train_mean = purrr::map_dbl(predictor, function(nm) {
      if (nm %in% predictor_cols_dummy)      return(0)
      if (nm %in% predictor_cols_percentage) return(0)
      mean(train_raw[[nm]], na.rm = TRUE)
    }),
    train_sd = purrr::map_dbl(predictor, function(nm) {
      if (nm %in% predictor_cols_dummy)      return(1)
      if (nm %in% predictor_cols_percentage) return(100)
      sd(train_raw[[nm]], na.rm = TRUE)
    })
  )

# Drop predictors with degenerate scaling (zero or NA sd)
bad_scaling <- dplyr::filter(
  predictor_scaling,
  is.na(train_mean) | !is.finite(train_mean) |
    is.na(train_sd) | !is.finite(train_sd) | train_sd <= 0
)

if (nrow(bad_scaling) > 0) {
  message("\nDropping predictors with degenerate scaling (zero variance or NA):")
  print(bad_scaling$predictor)

  predictor_cols_final  <- setdiff(predictor_cols_final, bad_scaling$predictor)
  predictor_type_table  <- dplyr::filter(predictor_type_table,  predictor %in% predictor_cols_final)
  predictor_scaling     <- dplyr::filter(predictor_scaling,     predictor %in% predictor_cols_final)
  predictor_cols_dummy      <- dplyr::filter(predictor_type_table, is_dummy)$predictor
  predictor_cols_percentage <- dplyr::filter(predictor_type_table, is_percentage)$predictor
  predictor_cols_continuous <- dplyr::filter(predictor_type_table, !is_dummy, !is_percentage)$predictor
}

apply_scaling <- function(df, scaling_table, cols) {
  for (i in seq_len(nrow(scaling_table))) {
    nm  <- scaling_table$predictor[i]
    if (!nm %in% cols) next
    df[[nm]] <- (df[[nm]] - scaling_table$train_mean[i]) / scaling_table$train_sd[i]
  }
  df
}

train_scaled      <- apply_scaling(train_raw,      predictor_scaling, predictor_cols_final)
validation_scaled <- apply_scaling(validation_raw, predictor_scaling, predictor_cols_final)
test_scaled       <- apply_scaling(test_raw,       predictor_scaling, predictor_cols_final)

# ── Checks ────────────────────────────────────────────────────────────────────

dataset_check <- dataset_model_split %>%
  dplyr::summarise(
    target_col              = target_col,
    n_rows                  = dplyr::n(),
    n_profiles              = dplyr::n_distinct(profile_id),
    min_target              = min(target_native, na.rm = TRUE),
    q01_target              = as.numeric(stats::quantile(target_native, 0.01, na.rm = TRUE)),
    median_target           = median(target_native, na.rm = TRUE),
    mean_target             = mean(target_native, na.rm = TRUE),
    q99_target              = as.numeric(stats::quantile(target_native, 0.99, na.rm = TRUE)),
    max_target              = max(target_native, na.rm = TRUE),
    median_target_log1p     = median(target_log1p, na.rm = TRUE),
    n_predictors            = length(predictor_cols_final),
    n_dummy_predictors      = length(predictor_cols_dummy),
    n_percentage_predictors = length(predictor_cols_percentage),
    n_continuous_predictors = length(predictor_cols_continuous)
  )

split_check <- dataset_model_split %>%
  dplyr::group_by(dataset_role) %>%
  dplyr::summarise(
    n             = dplyr::n(),
    n_profiles    = dplyr::n_distinct(profile_id),
    min_target    = min(target_native, na.rm = TRUE),
    median_target = median(target_native, na.rm = TRUE),
    mean_target   = mean(target_native, na.rm = TRUE),
    max_target    = max(target_native, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(factor(dataset_role, levels = c("train", "validation", "test")))

split_bin_check <- dataset_model_split %>%
  dplyr::count(split_bin, dataset_role) %>%
  tidyr::pivot_wider(names_from = dataset_role, values_from = n, values_fill = 0L)

scaled_extreme_check <- purrr::map_dfr(predictor_cols_final, function(nm) {
  tibble::tibble(
    predictor          = nm,
    scaling_method     = predictor_scaling$scaling_method[match(nm, predictor_scaling$predictor)],
    max_abs_train      = max(abs(train_scaled[[nm]]), na.rm = TRUE),
    max_abs_validation = max(abs(validation_scaled[[nm]]), na.rm = TRUE),
    max_abs_test       = max(abs(test_scaled[[nm]]), na.rm = TRUE),
    train_mean_raw     = mean(train_raw[[nm]], na.rm = TRUE),
    train_sd_raw       = sd(train_raw[[nm]], na.rm = TRUE)
  )
}) %>%
  dplyr::arrange(dplyr::desc(max_abs_train))

message("\n── Dataset summary ──────────────────────────")
print(dataset_check, width = Inf)
message("\n── Split summary ────────────────────────────")
print(split_check, width = Inf)
message("\n── Split × bin check ────────────────────────")
print(split_bin_check, n = Inf, width = Inf)
message("\n── Top scaled extremes ──────────────────────")
print(dplyr::slice_head(scaled_extreme_check, n = 20), width = Inf)

# ── Export ────────────────────────────────────────────────────────────────────

export_cols <- c(
  "profile_id", "sample_id", "dataset_role", "split_bin", "x", "y",
  target_col, "target_native", "target_log1p",
  predictor_cols_final
)

safe_write_csv2(
  dplyr::select(dataset_model_split, dplyr::all_of(export_cols)),
  file.path(output_data_dir, "full_modeling_dataset_raw.csv")
)

for (role in c("train", "validation", "test")) {
  df <- dplyr::filter(dataset_model_split, dataset_role == role)
  df_sc <- switch(role,
    train      = train_scaled,
    validation = validation_scaled,
    test       = test_scaled
  )
  safe_write_csv2(
    dplyr::select(df,    dplyr::all_of(export_cols)),
    file.path(output_data_dir, paste0(role, "_raw.csv"))
  )
  safe_write_csv2(
    dplyr::select(df_sc, dplyr::all_of(export_cols)),
    file.path(output_data_dir, paste0(role, "_scaled.csv"))
  )
}

# Metadata
safe_write_csv2(qc_summary,            file.path(output_metadata_dir, "qc_summary.csv"))
safe_write_csv2(predictor_type_table,  file.path(output_metadata_dir, "predictor_type_table.csv"))
safe_write_csv2(predictor_scaling,     file.path(output_metadata_dir, "predictor_scaling.csv"))
safe_write_csv2(raster_table_all,      file.path(output_metadata_dir, "raster_table_all.csv"))
safe_write_csv2(raster_table_use,      file.path(output_metadata_dir, "raster_table_used.csv"))
safe_write_csv2(dataset_check,         file.path(output_metadata_dir, "dataset_check.csv"))
safe_write_csv2(split_check,           file.path(output_metadata_dir, "split_check.csv"))
safe_write_csv2(split_bin_check,       file.path(output_metadata_dir, "split_bin_check.csv"))
safe_write_csv2(scaled_extreme_check,  file.path(output_metadata_dir, "scaled_extreme_check.csv"))

safe_write_csv2(
  dataset_model_split %>%
    dplyr::select(profile_id, sample_id, dataset_role, split_bin, x, y,
                  target_native, target_log1p),
  file.path(output_metadata_dir, "split_metadata.csv")
)

safe_write_csv2(
  tibble::tibble(
    target_label                = target_label,
    target_col                  = target_col,
    target_unit                 = target_unit,
    train_fraction              = train_fraction,
    validation_fraction         = validation_fraction,
    test_fraction               = test_fraction,
    split_n_bins                = split_n_bins,
    predictor_raster_dir        = predictor_raster_dir,
    manual_predictor_drop       = paste(drop_present, collapse = ";"),
    temperature_min_valid_celsius = temperature_min_valid_celsius,
    percentage_predictor_patterns = paste(percentage_predictor_patterns, collapse = ";"),
    n_predictors_final          = length(predictor_cols_final),
    n_dummy                     = length(predictor_cols_dummy),
    n_percentage                = length(predictor_cols_percentage),
    n_continuous                = length(predictor_cols_continuous),
    n_rows_after_qc             = nrow(dataset_model_raw),
    soc_gpkg_file               = soc_gpkg_file
  ),
  file.path(output_metadata_dir, "target_config.csv")
)

# ── Figures ───────────────────────────────────────────────────────────────────

p_count <- ggplot2::ggplot(split_check, ggplot2::aes(x = dataset_role, y = n)) +
  ggplot2::geom_col() +
  ggplot2::labs(x = "Dataset role", y = "Number of profiles",
                title = paste(target_label, "– split sizes")) +
  ggplot2::theme_bw()

p_dens_native <- ggplot2::ggplot(
  dataset_model_split,
  ggplot2::aes(x = target_native, linetype = dataset_role)
) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(x = paste0("SOC stock, ", target_unit), y = "Density",
                linetype = "Split",
                title = paste(target_label, "– native distribution by split")) +
  ggplot2::theme_bw()

p_dens_log1p <- ggplot2::ggplot(
  dataset_model_split,
  ggplot2::aes(x = target_log1p, linetype = dataset_role)
) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(x = paste0("log1p(SOC stock, ", target_unit, ")"), y = "Density",
                linetype = "Split",
                title = paste(target_label, "– log1p distribution by split")) +
  ggplot2::theme_bw()

p_boxplot <- ggplot2::ggplot(
  dataset_model_split,
  ggplot2::aes(x = dataset_role, y = target_native)
) +
  ggplot2::geom_boxplot(outlier.alpha = 0.2) +
  ggplot2::labs(x = "Split", y = paste0("SOC stock, ", target_unit),
                title = paste(target_label, "– boxplot by split")) +
  ggplot2::theme_bw()

p_scaled_extreme <- scaled_extreme_check %>%
  dplyr::slice_head(n = 30) %>%
  dplyr::mutate(predictor = factor(predictor, levels = rev(predictor))) %>%
  ggplot2::ggplot(ggplot2::aes(x = max_abs_train, y = predictor)) +
  ggplot2::geom_col() +
  ggplot2::labs(x = "Max |scaled value| in train", y = NULL,
                title = paste(target_label, "– top scaled predictor extremes")) +
  ggplot2::theme_bw()

for (p in list(p_count, p_dens_native, p_dens_log1p, p_boxplot, p_scaled_extreme)) {
  print(p)
}

ggplot2::ggsave(file.path(output_figure_dir, "split_count.png"),
                p_count, width = 7, height = 5, dpi = 300)
ggplot2::ggsave(file.path(output_figure_dir, "target_native_density_by_split.png"),
                p_dens_native, width = 8, height = 5, dpi = 300)
ggplot2::ggsave(file.path(output_figure_dir, "target_log1p_density_by_split.png"),
                p_dens_log1p, width = 8, height = 5, dpi = 300)
ggplot2::ggsave(file.path(output_figure_dir, "target_boxplot_by_split.png"),
                p_boxplot, width = 8, height = 5, dpi = 300)
ggplot2::ggsave(file.path(output_figure_dir, "scaled_extreme_predictors.png"),
                p_scaled_extreme, width = 9, height = 7, dpi = 300)

message("\nDataset saved to:  ", output_data_dir)
message("Metadata saved to: ", output_metadata_dir)
message("Figures saved to:  ", output_figure_dir)
