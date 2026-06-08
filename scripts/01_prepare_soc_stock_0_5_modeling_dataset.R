library(sf)
library(terra)
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(janitor)
library(purrr)
library(ggplot2)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

set.seed(123)

target_label <- "soc_stock_0_5cm"
target_col <- "soc_stock_ton_ha_0_5cm"
target_unit <- "ton_ha"

target_file <- "./outputs/soc_stock_layers_qc/wosis_profile_soc_stock_spline_clean_p999_zero_spike.gpkg"
predictor_dir <- "../predictors_resolution_20000m"

train_fraction <- 0.70
validation_fraction <- 0.15
test_fraction <- 0.15
split_n_bins <- 10

temperature_min_valid_celsius <- -100

percentage_predictor_patterns <- c(
  "^pnv_",
  "^clay_",
  "^peatland_extent$"
)

force_percentage_predictors <- c(
  "pnv_moss_and_lichen",
  "pnv_open_forest_deciduous_needleleaf"
)

manual_predictor_drop <- character(0)

output_data_dir <- file.path(
  "data",
  "processed",
  "soc_stock_modeling",
  target_label
)

output_metadata_dir <- file.path(
  "outputs",
  "metadata",
  "soc_stock_modeling",
  target_label
)

output_figure_dir <- file.path(
  "outputs",
  "figures",
  "soc_stock_modeling",
  target_label
)

dir.create(output_data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_metadata_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_figure_dir, recursive = TRUE, showWarnings = FALSE)

save_plot <- function(plot_object, output_file, width = 8, height = 5) {
  print(plot_object)
  ggplot2::ggsave(
    filename = output_file,
    plot = plot_object,
    width = width,
    height = height,
    dpi = 300
  )
}

check_required_files <- function(target_file, predictor_dir) {
  if (!file.exists(target_file)) {
    stop("Target file was not found: ", target_file)
  }
  
  if (!dir.exists(predictor_dir)) {
    stop("Predictor directory was not found: ", predictor_dir)
  }
}

build_raster_inventory <- function(predictor_dir, manual_predictor_drop) {
  raster_files <- list.files(
    predictor_dir,
    pattern = "\\.tif$",
    full.names = TRUE,
    recursive = FALSE
  )
  
  if (length(raster_files) == 0) {
    stop("No .tif files were found in predictor_dir: ", predictor_dir)
  }
  
  raster_table <- tibble::tibble(
    raster_file = raster_files,
    raster_name_raw = tools::file_path_sans_ext(basename(raster_files)),
    predictor = janitor::make_clean_names(raster_name_raw)
  ) %>%
    dplyr::arrange(predictor)
  
  duplicate_predictors <- raster_table %>%
    dplyr::count(predictor) %>%
    dplyr::filter(n > 1)
  
  if (nrow(duplicate_predictors) > 0) {
    print(duplicate_predictors)
    stop("Duplicated predictor names after janitor::make_clean_names().")
  }
  
  manual_drop_clean <- janitor::make_clean_names(manual_predictor_drop)
  manual_drop_present <- intersect(manual_drop_clean, raster_table$predictor)
  manual_drop_missing <- setdiff(manual_drop_clean, raster_table$predictor)
  
  if (length(manual_drop_present) > 0) {
    message("Dropping manual predictors: ", paste(manual_drop_present, collapse = ", "))
  }
  
  if (length(manual_drop_missing) > 0) {
    message("Manual drop predictors not found: ", paste(manual_drop_missing, collapse = ", "))
  }
  
  raster_table_used <- raster_table %>%
    dplyr::filter(!predictor %in% manual_drop_present)
  
  if (nrow(raster_table_used) == 0) {
    stop("No predictor raster remained after manual drop.")
  }
  
  list(
    all = raster_table,
    used = raster_table_used,
    manual_drop_present = manual_drop_present,
    manual_drop_missing = manual_drop_missing
  )
}

read_target_points <- function(target_file, target_col) {
  target_sf <- sf::st_read(target_file, quiet = TRUE) %>%
    janitor::clean_names()
  
  if (!"profile_id" %in% names(target_sf)) {
    stop("The target file must contain profile_id.")
  }
  
  if (!target_col %in% names(target_sf)) {
    stop("Target column was not found: ", target_col)
  }
  
  target_sf %>%
    dplyr::mutate(profile_id = as.character(profile_id))
}

extract_predictor_values <- function(target_sf, raster_table_used) {
  message("Loading predictor raster stack...")
  
  predictor_stack <- terra::rast(raster_table_used$raster_file)
  names(predictor_stack) <- raster_table_used$predictor
  
  raster_crs <- terra::crs(predictor_stack, proj = TRUE)
  
  if (is.na(raster_crs) || raster_crs == "") {
    stop("Predictor rasters do not have a valid CRS.")
  }
  
  geometry_ok <- purrr::map_lgl(
    seq_len(terra::nlyr(predictor_stack)),
    function(layer_id) {
      terra::compareGeom(
        predictor_stack[[1]],
        predictor_stack[[layer_id]],
        stopOnError = FALSE
      )
    }
  )
  
  if (!all(geometry_ok)) {
    bad_geometry <- tibble::tibble(
      predictor = names(predictor_stack),
      geometry_ok = geometry_ok
    ) %>%
      dplyr::filter(!geometry_ok)
    
    print(bad_geometry)
    stop("Some predictor rasters do not share the same geometry.")
  }
  
  target_projected <- sf::st_transform(
    target_sf,
    crs = sf::st_crs(raster_crs)
  )
  
  point_coordinates <- sf::st_coordinates(target_projected)
  
  point_table <- target_projected %>%
    dplyr::mutate(
      x = as.numeric(point_coordinates[, 1]),
      y = as.numeric(point_coordinates[, 2])
    )
  
  message("Extracting predictor values for ", nrow(point_table), " profiles...")
  
  predictor_values <- terra::extract(
    predictor_stack,
    terra::vect(point_table),
    ID = FALSE
  ) %>%
    tibble::as_tibble() %>%
    dplyr::mutate(
      dplyr::across(
        dplyr::everything(),
        as.numeric
      )
    )
  
  point_table %>%
    sf::st_drop_geometry() %>%
    dplyr::bind_cols(predictor_values)
}

classify_predictors <- function(
    dataset,
    predictor_cols,
    percentage_predictor_patterns,
    force_percentage_predictors
) {
  percentage_regex <- paste(percentage_predictor_patterns, collapse = "|")
  force_percentage_clean <- janitor::make_clean_names(force_percentage_predictors)
  
  purrr::map_dfr(
    predictor_cols,
    function(predictor_name) {
      x <- dataset[[predictor_name]]
      x <- x[!is.na(x) & is.finite(x)]
      unique_values <- sort(unique(x))
      
      is_dummy <- length(unique_values) <= 2 &&
        all(unique_values %in% c(0, 1))
      
      is_percentage <- grepl(percentage_regex, predictor_name)
      
      if (predictor_name %in% force_percentage_clean) {
        is_dummy <- FALSE
        is_percentage <- TRUE
      }
      
      tibble::tibble(
        predictor = predictor_name,
        n_unique = length(unique_values),
        min_value = ifelse(length(x) > 0, min(x), NA_real_),
        max_value = ifelse(length(x) > 0, max(x), NA_real_),
        is_dummy = is_dummy,
        is_percentage = is_percentage,
        predictor_type = dplyr::case_when(
          is_dummy ~ "dummy",
          is_percentage ~ "percentage",
          TRUE ~ "continuous"
        )
      )
    }
  )
}

apply_predictor_qc <- function(
    dataset,
    predictor_cols,
    percentage_cols,
    temperature_min_valid_celsius
) {
  output <- dataset
  
  temperature_cols <- grep(
    "surface_temperature_celsius$",
    predictor_cols,
    value = TRUE
  )
  
  if (length(temperature_cols) > 0) {
    message("Applying temperature QC to ", length(temperature_cols), " predictor(s).")
    
    output <- output %>%
      dplyr::mutate(
        dplyr::across(
          dplyr::all_of(temperature_cols),
          ~ dplyr::if_else(
            !is.na(.x) & is.finite(.x) & .x <= temperature_min_valid_celsius,
            NA_real_,
            as.numeric(.x)
          )
        )
      )
  }
  
  if (length(percentage_cols) > 0) {
    message("Applying percentage QC to ", length(percentage_cols), " predictor(s).")
    
    output <- output %>%
      dplyr::mutate(
        dplyr::across(
          dplyr::all_of(percentage_cols),
          ~ dplyr::if_else(
            !is.na(.x) & is.finite(.x) & (.x < 0 | .x > 100),
            NA_real_,
            as.numeric(.x)
          )
        )
      )
  }
  
  output
}

make_stratified_split <- function(dataset, target_col, split_n_bins, train_fraction, validation_fraction) {
  dataset %>%
    dplyr::mutate(
      sample_id = dplyr::row_number(),
      split_bin = dplyr::ntile(.data[[target_col]], split_n_bins)
    ) %>%
    dplyr::group_by(split_bin) %>%
    dplyr::mutate(
      split_order = sample(dplyr::n()),
      n_bin = dplyr::n(),
      n_train_bin = floor(n_bin * train_fraction),
      n_validation_bin = floor(n_bin * validation_fraction),
      dataset_role = dplyr::case_when(
        split_order <= n_train_bin ~ "train",
        split_order <= n_train_bin + n_validation_bin ~ "validation",
        TRUE ~ "test"
      )
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(
      -split_order,
      -n_bin,
      -n_train_bin,
      -n_validation_bin
    )
}

build_scaling_table <- function(dataset_split, predictor_type_table) {
  train_data <- dataset_split %>%
    dplyr::filter(dataset_role == "train")
  
  predictor_type_table %>%
    dplyr::mutate(
      scaling_method = dplyr::case_when(
        predictor_type == "dummy" ~ "none_dummy_0_1",
        predictor_type == "percentage" ~ "percentage_0_100_to_0_1",
        TRUE ~ "zscore_train"
      ),
      train_center = purrr::map2_dbl(
        predictor,
        scaling_method,
        function(predictor_name, method) {
          if (method == "zscore_train") {
            mean(train_data[[predictor_name]], na.rm = TRUE)
          } else {
            0
          }
        }
      ),
      train_scale = purrr::map2_dbl(
        predictor,
        scaling_method,
        function(predictor_name, method) {
          if (method == "zscore_train") {
            sd(train_data[[predictor_name]], na.rm = TRUE)
          } else if (method == "percentage_0_100_to_0_1") {
            100
          } else {
            1
          }
        }
      )
    )
}

apply_scaling_table <- function(dataset, scaling_table) {
  output <- dataset
  
  for (i in seq_len(nrow(scaling_table))) {
    predictor_name <- scaling_table$predictor[i]
    
    output[[predictor_name]] <- (
      output[[predictor_name]] - scaling_table$train_center[i]
    ) / scaling_table$train_scale[i]
  }
  
  output
}

make_quantile_group <- function(x) {
  q <- stats::quantile(
    x,
    probs = c(0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
    na.rm = TRUE,
    names = FALSE
  )
  
  dplyr::case_when(
    x <= q[1] ~ "Q00_Q25",
    x <= q[2] ~ "Q25_Q50",
    x <= q[3] ~ "Q50_Q75",
    x <= q[4] ~ "Q75_Q90",
    x <= q[5] ~ "Q90_Q95",
    x <= q[6] ~ "Q95_Q99",
    TRUE ~ "Q99_Q100"
  )
}

message("Starting dataset preparation for ", target_label)

stopifnot(abs(train_fraction + validation_fraction + test_fraction - 1) < 1e-8)

check_required_files(
  target_file = target_file,
  predictor_dir = predictor_dir
)

raster_inventory <- build_raster_inventory(
  predictor_dir = predictor_dir,
  manual_predictor_drop = manual_predictor_drop
)

predictor_cols_raw <- raster_inventory$used$predictor

message("Predictor rasters found: ", nrow(raster_inventory$all))
message("Predictor rasters used before QC: ", length(predictor_cols_raw))

target_sf <- read_target_points(
  target_file = target_file,
  target_col = target_col
)

dataset_extracted <- extract_predictor_values(
  target_sf = target_sf,
  raster_table_used = raster_inventory$used
) %>%
  dplyr::mutate(
    profile_id = as.character(profile_id),
    target_native = as.numeric(.data[[target_col]]),
    target_log1p = log1p(target_native)
  )

predictor_type_initial <- classify_predictors(
  dataset = dataset_extracted,
  predictor_cols = predictor_cols_raw,
  percentage_predictor_patterns = percentage_predictor_patterns,
  force_percentage_predictors = force_percentage_predictors
)

percentage_cols_initial <- predictor_type_initial %>%
  dplyr::filter(predictor_type == "percentage") %>%
  dplyr::pull(predictor)

dataset_after_qc <- apply_predictor_qc(
  dataset = dataset_extracted,
  predictor_cols = predictor_cols_raw,
  percentage_cols = percentage_cols_initial,
  temperature_min_valid_celsius = temperature_min_valid_celsius
)

row_qc <- dataset_after_qc %>%
  dplyr::mutate(
    has_target_problem = is.na(profile_id) |
      is.na(x) |
      is.na(y) |
      !is.finite(x) |
      !is.finite(y) |
      is.na(target_native) |
      !is.finite(target_native) |
      target_native <= 0 |
      is.na(target_log1p) |
      !is.finite(target_log1p),
    has_predictor_problem = !dplyr::if_all(
      dplyr::all_of(predictor_cols_raw),
      ~ !is.na(.x) & is.finite(.x)
    )
  )

predictor_qc_summary <- row_qc %>%
  dplyr::summarise(
    n_rows_extracted = dplyr::n(),
    n_target_problem = sum(has_target_problem, na.rm = TRUE),
    n_predictor_problem = sum(has_predictor_problem, na.rm = TRUE),
    n_any_problem = sum(has_target_problem | has_predictor_problem, na.rm = TRUE),
    pct_any_problem = 100 * n_any_problem / n_rows_extracted,
    n_rows_after_qc = n_rows_extracted - n_any_problem
  )

print(predictor_qc_summary)

dataset_model_raw <- row_qc %>%
  dplyr::filter(
    !has_target_problem,
    !has_predictor_problem
  ) %>%
  dplyr::select(
    -has_target_problem,
    -has_predictor_problem
  ) %>%
  dplyr::distinct(
    profile_id,
    .keep_all = TRUE
  )

if (nrow(dataset_model_raw) == 0) {
  stop("No rows remained after target and predictor QC.")
}

predictor_type_table <- classify_predictors(
  dataset = dataset_model_raw,
  predictor_cols = predictor_cols_raw,
  percentage_predictor_patterns = percentage_predictor_patterns,
  force_percentage_predictors = force_percentage_predictors
)

predictor_cols_final <- predictor_type_table$predictor

dataset_model_split <- make_stratified_split(
  dataset = dataset_model_raw,
  target_col = "target_native",
  split_n_bins = split_n_bins,
  train_fraction = train_fraction,
  validation_fraction = validation_fraction
)

predictor_scaling <- build_scaling_table(
  dataset_split = dataset_model_split,
  predictor_type_table = predictor_type_table
)

bad_scaling <- predictor_scaling %>%
  dplyr::filter(
    is.na(train_center) |
      is.na(train_scale) |
      !is.finite(train_center) |
      !is.finite(train_scale) |
      train_scale <= 0
  )

if (nrow(bad_scaling) > 0) {
  message("Dropping predictors with invalid scaling: ", paste(bad_scaling$predictor, collapse = ", "))
  
  predictor_cols_final <- setdiff(predictor_cols_final, bad_scaling$predictor)
  
  predictor_type_table <- predictor_type_table %>%
    dplyr::filter(predictor %in% predictor_cols_final)
  
  raster_inventory$used <- raster_inventory$used %>%
    dplyr::filter(predictor %in% predictor_cols_final)
  
  predictor_scaling <- build_scaling_table(
    dataset_split = dataset_model_split,
    predictor_type_table = predictor_type_table
  )
}

dataset_model_split <- dataset_model_split %>%
  dplyr::select(
    profile_id,
    sample_id,
    dataset_role,
    split_bin,
    x,
    y,
    dplyr::all_of(target_col),
    target_native,
    target_log1p,
    dplyr::all_of(predictor_cols_final)
  )

dataset_model_scaled <- apply_scaling_table(
  dataset = dataset_model_split,
  scaling_table = predictor_scaling
)

train_raw <- dataset_model_split %>%
  dplyr::filter(dataset_role == "train")

validation_raw <- dataset_model_split %>%
  dplyr::filter(dataset_role == "validation")

test_raw <- dataset_model_split %>%
  dplyr::filter(dataset_role == "test")

train_scaled <- dataset_model_scaled %>%
  dplyr::filter(dataset_role == "train")

validation_scaled <- dataset_model_scaled %>%
  dplyr::filter(dataset_role == "validation")

test_scaled <- dataset_model_scaled %>%
  dplyr::filter(dataset_role == "test")

dataset_check <- dataset_model_split %>%
  dplyr::summarise(
    target_label = target_label,
    target_col = target_col,
    target_unit = target_unit,
    n_rows = dplyr::n(),
    n_profiles = dplyr::n_distinct(profile_id),
    n_predictors = length(predictor_cols_final),
    n_dummy_predictors = sum(predictor_type_table$predictor_type == "dummy"),
    n_percentage_predictors = sum(predictor_type_table$predictor_type == "percentage"),
    n_continuous_predictors = sum(predictor_type_table$predictor_type == "continuous"),
    min_target = min(target_native, na.rm = TRUE),
    q01_target = as.numeric(stats::quantile(target_native, 0.01, na.rm = TRUE)),
    q05_target = as.numeric(stats::quantile(target_native, 0.05, na.rm = TRUE)),
    median_target = median(target_native, na.rm = TRUE),
    mean_target = mean(target_native, na.rm = TRUE),
    q95_target = as.numeric(stats::quantile(target_native, 0.95, na.rm = TRUE)),
    q99_target = as.numeric(stats::quantile(target_native, 0.99, na.rm = TRUE)),
    max_target = max(target_native, na.rm = TRUE)
  )

split_check <- dataset_model_split %>%
  dplyr::group_by(dataset_role) %>%
  dplyr::summarise(
    n = dplyr::n(),
    n_profiles = dplyr::n_distinct(profile_id),
    min_target = min(target_native, na.rm = TRUE),
    q05_target = as.numeric(stats::quantile(target_native, 0.05, na.rm = TRUE)),
    median_target = median(target_native, na.rm = TRUE),
    mean_target = mean(target_native, na.rm = TRUE),
    q95_target = as.numeric(stats::quantile(target_native, 0.95, na.rm = TRUE)),
    q99_target = as.numeric(stats::quantile(target_native, 0.99, na.rm = TRUE)),
    max_target = max(target_native, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  dplyr::arrange(factor(dataset_role, levels = c("train", "validation", "test")))

split_bin_check <- dataset_model_split %>%
  dplyr::count(split_bin, dataset_role) %>%
  tidyr::pivot_wider(
    names_from = dataset_role,
    values_from = n,
    values_fill = 0
  ) %>%
  dplyr::arrange(split_bin)

tail_group_check <- dataset_model_split %>%
  dplyr::mutate(
    obs_quantile_group = make_quantile_group(target_native)
  ) %>%
  dplyr::count(obs_quantile_group, dataset_role) %>%
  tidyr::pivot_wider(
    names_from = dataset_role,
    values_from = n,
    values_fill = 0
  ) %>%
  dplyr::arrange(obs_quantile_group)

scaled_check <- dataset_model_scaled %>%
  dplyr::group_by(dataset_role) %>%
  dplyr::summarise(
    n = dplyr::n(),
    max_abs_scaled_value = max(abs(as.matrix(dplyr::select(dplyr::cur_data(), dplyr::all_of(predictor_cols_final)))), na.rm = TRUE),
    .groups = "drop"
  )

scaled_extreme_check <- purrr::map_dfr(
  predictor_cols_final,
  function(predictor_name) {
    tibble::tibble(
      predictor = predictor_name,
      scaling_method = predictor_scaling$scaling_method[
        match(predictor_name, predictor_scaling$predictor)
      ],
      max_abs_train = max(abs(train_scaled[[predictor_name]]), na.rm = TRUE),
      max_abs_validation = max(abs(validation_scaled[[predictor_name]]), na.rm = TRUE),
      max_abs_test = max(abs(test_scaled[[predictor_name]]), na.rm = TRUE),
      train_min_raw = min(train_raw[[predictor_name]], na.rm = TRUE),
      train_median_raw = median(train_raw[[predictor_name]], na.rm = TRUE),
      train_max_raw = max(train_raw[[predictor_name]], na.rm = TRUE)
    )
  }
) %>%
  dplyr::arrange(dplyr::desc(max_abs_train))

print(dataset_check)
print(split_check)
print(split_bin_check, n = Inf)
print(tail_group_check, n = Inf)
print(scaled_check)
print(scaled_extreme_check %>% dplyr::slice_head(n = 30), n = 30)

selected_predictors <- tibble::tibble(
  predictor_order = seq_along(predictor_cols_final),
  predictor = predictor_cols_final,
  predictor_type = predictor_type_table$predictor_type[
    match(predictor_cols_final, predictor_type_table$predictor)
  ],
  scaling_method = predictor_scaling$scaling_method[
    match(predictor_cols_final, predictor_scaling$predictor)
  ]
)

raw_cols_export <- c(
  "profile_id",
  "sample_id",
  "dataset_role",
  "split_bin",
  "x",
  "y",
  target_col,
  "target_native",
  "target_log1p",
  predictor_cols_final
)

readr::write_csv2(dataset_model_split, file.path(output_data_dir, "full_modeling_dataset_raw.csv"))
readr::write_csv2(dataset_model_scaled, file.path(output_data_dir, "full_modeling_dataset_scaled.csv"))
readr::write_csv2(train_raw %>% dplyr::select(dplyr::all_of(raw_cols_export)), file.path(output_data_dir, "train_raw.csv"))
readr::write_csv2(validation_raw %>% dplyr::select(dplyr::all_of(raw_cols_export)), file.path(output_data_dir, "validation_raw.csv"))
readr::write_csv2(test_raw %>% dplyr::select(dplyr::all_of(raw_cols_export)), file.path(output_data_dir, "test_raw.csv"))
readr::write_csv2(train_scaled %>% dplyr::select(dplyr::all_of(raw_cols_export)), file.path(output_data_dir, "train_scaled.csv"))
readr::write_csv2(validation_scaled %>% dplyr::select(dplyr::all_of(raw_cols_export)), file.path(output_data_dir, "validation_scaled.csv"))
readr::write_csv2(test_scaled %>% dplyr::select(dplyr::all_of(raw_cols_export)), file.path(output_data_dir, "test_scaled.csv"))

readr::write_csv2(predictor_qc_summary, file.path(output_metadata_dir, "predictor_qc_summary.csv"))
readr::write_csv2(predictor_type_table, file.path(output_metadata_dir, "predictor_type_table.csv"))
readr::write_csv2(predictor_scaling, file.path(output_metadata_dir, "predictor_scaling.csv"))
readr::write_csv2(raster_inventory$all, file.path(output_metadata_dir, "predictor_raster_table_all.csv"))
readr::write_csv2(raster_inventory$used, file.path(output_metadata_dir, "predictor_raster_table_used.csv"))
readr::write_csv2(selected_predictors, file.path(output_metadata_dir, "selected_predictors.csv"))
readr::write_csv2(dataset_check, file.path(output_metadata_dir, "dataset_check.csv"))
readr::write_csv2(split_check, file.path(output_metadata_dir, "split_check.csv"))
readr::write_csv2(split_bin_check, file.path(output_metadata_dir, "split_bin_check.csv"))
readr::write_csv2(tail_group_check, file.path(output_metadata_dir, "tail_group_check.csv"))
readr::write_csv2(scaled_check, file.path(output_metadata_dir, "scaled_check.csv"))
readr::write_csv2(scaled_extreme_check, file.path(output_metadata_dir, "scaled_extreme_check.csv"))

target_config <- tibble::tibble(
  target_label = target_label,
  target_col = target_col,
  target_unit = target_unit,
  target_file = target_file,
  predictor_dir = predictor_dir,
  n_rasters_found = nrow(raster_inventory$all),
  n_predictors_used = length(predictor_cols_final),
  train_fraction = train_fraction,
  validation_fraction = validation_fraction,
  test_fraction = test_fraction,
  split_n_bins = split_n_bins,
  seed = 123,
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  percentage_predictor_patterns = paste(percentage_predictor_patterns, collapse = ";"),
  force_percentage_predictors = paste(force_percentage_predictors, collapse = ";"),
  manual_predictor_drop = paste(raster_inventory$manual_drop_present, collapse = ";")
)

readr::write_csv2(target_config, file.path(output_metadata_dir, "target_config.csv"))

run_manifest <- list(
  target_config = target_config,
  predictor_qc_summary = predictor_qc_summary,
  predictor_type_table = predictor_type_table,
  predictor_scaling = predictor_scaling,
  selected_predictors = selected_predictors,
  dataset_check = dataset_check,
  split_check = split_check,
  split_bin_check = split_bin_check,
  tail_group_check = tail_group_check,
  scaled_check = scaled_check
)

saveRDS(
  run_manifest,
  file.path(output_metadata_dir, "prepare_modeling_dataset_manifest.rds"),
  compress = FALSE
)

split_levels <- c("train", "validation", "test")

plot_split_count <- split_check %>%
  dplyr::mutate(dataset_role = factor(dataset_role, levels = split_levels)) %>%
  ggplot2::ggplot(
    ggplot2::aes(x = dataset_role, y = n)
  ) +
  ggplot2::geom_col(width = 0.55) +
  ggplot2::geom_text(
    ggplot2::aes(label = n),
    vjust = -0.3,
    size = 4
  ) +
  ggplot2::labs(
    x = "Dataset role",
    y = "Number of profiles",
    title = paste0(target_label, " split size")
  ) +
  ggplot2::theme_bw()

save_plot(
  plot_split_count,
  file.path(output_figure_dir, "split_count.png"),
  width = 7,
  height = 5
)

plot_target_density <- dataset_model_split %>%
  dplyr::mutate(dataset_role = factor(dataset_role, levels = split_levels)) %>%
  ggplot2::ggplot(
    ggplot2::aes(x = target_native, colour = dataset_role)
  ) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(
    x = "SOC stock, ton/ha",
    y = "Density",
    colour = "Dataset role",
    title = paste0(target_label, " target distribution by split")
  ) +
  ggplot2::theme_bw()

save_plot(
  plot_target_density,
  file.path(output_figure_dir, "target_native_density_by_split.png"),
  width = 8,
  height = 5
)

plot_target_log1p_density <- dataset_model_split %>%
  dplyr::mutate(dataset_role = factor(dataset_role, levels = split_levels)) %>%
  ggplot2::ggplot(
    ggplot2::aes(x = target_log1p, colour = dataset_role)
  ) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(
    x = "log1p(SOC stock, ton/ha)",
    y = "Density",
    colour = "Dataset role",
    title = paste0(target_label, " log1p target distribution by split")
  ) +
  ggplot2::theme_bw()

save_plot(
  plot_target_log1p_density,
  file.path(output_figure_dir, "target_log1p_density_by_split.png"),
  width = 8,
  height = 5
)

plot_scaled_extreme <- scaled_extreme_check %>%
  dplyr::slice_head(n = 30) %>%
  dplyr::arrange(max_abs_train) %>%
  dplyr::mutate(
    predictor = factor(predictor, levels = predictor)
  ) %>%
  ggplot2::ggplot(
    ggplot2::aes(x = max_abs_train, y = predictor)
  ) +
  ggplot2::geom_col(width = 0.7) +
  ggplot2::labs(
    x = "Maximum absolute scaled value in train",
    y = NULL,
    title = paste0(target_label, " largest scaled predictor values")
  ) +
  ggplot2::theme_bw()

save_plot(
  plot_scaled_extreme,
  file.path(output_figure_dir, "scaled_extreme_predictors_train.png"),
  width = 9,
  height = 7
)

message("Dataset preparation finished.")
message("Data saved to: ", output_data_dir)
message("Metadata saved to: ", output_metadata_dir)
message("Figures saved to: ", output_figure_dir)
