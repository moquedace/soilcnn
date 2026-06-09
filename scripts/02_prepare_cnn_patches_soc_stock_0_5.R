source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "readr",
  "dplyr",
  "tidyr",
  "tibble",
  "purrr",
  "janitor",
  "terra",
  "ggplot2",
  "future",
  "future.apply",
  "parallelly"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

set.seed(123)

target_label <- "soc_stock_0_5cm"
predictor_dir <- "../predictors_resolution_20000m"

input_data_dir <- file.path(
  "data",
  "processed",
  "soc_stock_modeling",
  target_label
)

input_metadata_dir <- file.path(
  "outputs",
  "metadata",
  "soc_stock_modeling",
  target_label
)

output_patch_dir <- file.path(
  "outputs",
  "cnn_patches",
  "soc_stock_modeling",
  target_label
)

output_metadata_dir <- file.path(
  "outputs",
  "metadata",
  "soc_stock_modeling",
  target_label,
  "cnn_patches"
)

output_figure_dir <- file.path(
  "outputs",
  "figures",
  "soc_stock_modeling",
  target_label,
  "cnn_patches"
)

dir.create(output_patch_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_metadata_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(output_figure_dir, recursive = TRUE, showWarnings = FALSE)

train_file <- file.path(input_data_dir, "train_raw.csv")
validation_file <- file.path(input_data_dir, "validation_raw.csv")
test_file <- file.path(input_data_dir, "test_raw.csv")

selected_predictors_file <- file.path(input_metadata_dir, "selected_predictors.csv")
predictor_scaling_file <- file.path(input_metadata_dir, "predictor_scaling.csv")
raster_table_file <- file.path(input_metadata_dir, "predictor_raster_table_used.csv")
target_config_file <- file.path(input_metadata_dir, "target_config.csv")

patch_size_large <- 5
patch_size_small <- 3
patch_radius_large <- 2

chunk_size_points <- 1000
n_workers_patch <- min(12, parallelly::availableCores())
future_globals_max_size_gb <- 30

temperature_min_valid_celsius <- -100

options(
  future.globals.maxSize = future_globals_max_size_gb * 1024^3
)

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

check_input_files <- function(input_files) {
  missing_files <- input_files[!file.exists(input_files)]
  
  if (length(missing_files) > 0) {
    print(missing_files)
    stop("Some required input files were not found.")
  }
}

read_point_data <- function(input_file, dataset_name) {
  required_cols <- c(
    "profile_id",
    "sample_id",
    "dataset_role",
    "split_bin",
    "x",
    "y",
    "target_native",
    "target_log1p"
  )
  
  point_data <- readr::read_csv2(
    input_file,
    show_col_types = FALSE
  ) %>%
    janitor::clean_names()
  
  missing_cols <- setdiff(required_cols, names(point_data))
  
  if (length(missing_cols) > 0) {
    print(missing_cols)
    stop("Missing required columns in ", dataset_name, ".")
  }
  
  point_data %>%
    dplyr::mutate(
      profile_id = as.character(profile_id),
      dataset_role = as.character(dataset_role),
      sample_id = as.integer(sample_id),
      split_bin = as.integer(split_bin),
      x = as.numeric(x),
      y = as.numeric(y),
      target_native = as.numeric(target_native),
      target_log1p = as.numeric(target_log1p)
    )
}

read_patch_metadata <- function(
    selected_predictors_file,
    predictor_scaling_file,
    raster_table_file
) {
  selected_predictors <- readr::read_csv2(
    selected_predictors_file,
    show_col_types = FALSE
  ) %>%
    janitor::clean_names()
  
  predictor_scaling <- readr::read_csv2(
    predictor_scaling_file,
    show_col_types = FALSE
  ) %>%
    janitor::clean_names()
  
  raster_table <- readr::read_csv2(
    raster_table_file,
    show_col_types = FALSE
  ) %>%
    janitor::clean_names()
  
  required_selected_cols <- c(
    "predictor_order",
    "predictor",
    "predictor_type",
    "scaling_method"
  )
  
  required_scaling_cols <- c(
    "predictor",
    "scaling_method",
    "train_center",
    "train_scale"
  )
  
  required_raster_cols <- c(
    "raster_file",
    "predictor"
  )
  
  missing_selected <- setdiff(required_selected_cols, names(selected_predictors))
  missing_scaling <- setdiff(required_scaling_cols, names(predictor_scaling))
  missing_raster <- setdiff(required_raster_cols, names(raster_table))
  
  if (length(missing_selected) > 0) {
    print(missing_selected)
    stop("selected_predictors.csv is missing required columns.")
  }
  
  if (length(missing_scaling) > 0) {
    print(missing_scaling)
    stop("predictor_scaling.csv is missing required columns.")
  }
  
  if (length(missing_raster) > 0) {
    print(missing_raster)
    stop("predictor_raster_table_used.csv is missing required columns.")
  }
  
  selected_predictors <- selected_predictors %>%
    dplyr::arrange(predictor_order)
  
  predictor_cols <- selected_predictors$predictor
  
  predictor_scaling <- predictor_scaling %>%
    dplyr::filter(predictor %in% predictor_cols) %>%
    dplyr::arrange(match(predictor, predictor_cols))
  
  raster_table <- raster_table %>%
    dplyr::filter(predictor %in% predictor_cols) %>%
    dplyr::arrange(match(predictor, predictor_cols))
  
  missing_scaling_predictors <- setdiff(predictor_cols, predictor_scaling$predictor)
  missing_raster_predictors <- setdiff(predictor_cols, raster_table$predictor)
  
  if (length(missing_scaling_predictors) > 0) {
    print(missing_scaling_predictors)
    stop("Some selected predictors are missing scaling parameters.")
  }
  
  if (length(missing_raster_predictors) > 0) {
    print(missing_raster_predictors)
    stop("Some selected predictors are missing raster paths.")
  }
  
  missing_raster_files <- raster_table$raster_file[!file.exists(raster_table$raster_file)]
  
  if (length(missing_raster_files) > 0) {
    print(missing_raster_files)
    stop("Some raster files were not found.")
  }
  
  if (!identical(selected_predictors$predictor, predictor_scaling$predictor)) {
    stop("Predictor order mismatch between selected_predictors and predictor_scaling.")
  }
  
  if (!identical(selected_predictors$predictor, raster_table$predictor)) {
    stop("Predictor order mismatch between selected_predictors and raster_table.")
  }
  
  if (any(is.na(predictor_scaling$train_center)) ||
      any(is.na(predictor_scaling$train_scale)) ||
      any(!is.finite(predictor_scaling$train_center)) ||
      any(!is.finite(predictor_scaling$train_scale)) ||
      any(predictor_scaling$train_scale <= 0)) {
    stop("Invalid scaling parameters were found.")
  }
  
  list(
    selected_predictors = selected_predictors,
    predictor_scaling = predictor_scaling,
    raster_table = raster_table,
    predictor_cols = predictor_cols
  )
}

build_raster_stack <- function(raster_table, predictor_cols) {
  raster_stack <- terra::rast(raster_table$raster_file)
  names(raster_stack) <- predictor_cols
  
  if (!identical(names(raster_stack), predictor_cols)) {
    stop("Raster stack names are not aligned with predictor order.")
  }
  
  geometry_ok <- purrr::map_lgl(
    seq_len(terra::nlyr(raster_stack)),
    function(layer_id) {
      terra::compareGeom(
        raster_stack[[1]],
        raster_stack[[layer_id]],
        stopOnError = FALSE
      )
    }
  )
  
  if (!all(geometry_ok)) {
    bad_geometry <- tibble::tibble(
      predictor = names(raster_stack),
      geometry_ok = geometry_ok
    ) %>%
      dplyr::filter(!geometry_ok)
    
    print(bad_geometry)
    stop("Some predictor rasters do not share the same geometry.")
  }
  
  raster_stack
}

prepare_point_index <- function(point_data, raster_template, patch_radius) {
  xy <- point_data %>%
    dplyr::select(x, y) %>%
    as.matrix()
  
  center_cell <- terra::cellFromXY(raster_template, xy)
  center_row_col <- terra::rowColFromCell(raster_template, center_cell)
  
  point_data %>%
    dplyr::mutate(
      original_row_id = dplyr::row_number(),
      center_cell = center_cell,
      center_row = center_row_col[, 1],
      center_col = center_row_col[, 2]
    ) %>%
    dplyr::filter(
      !is.na(center_cell),
      !is.na(center_row),
      !is.na(center_col),
      center_row - patch_radius >= 1,
      center_row + patch_radius <= terra::nrow(raster_template),
      center_col - patch_radius >= 1,
      center_col + patch_radius <= terra::ncol(raster_template)
    )
}

make_offset_grid <- function(patch_radius) {
  expand.grid(
    row_offset = seq(-patch_radius, patch_radius),
    col_offset = seq(-patch_radius, patch_radius)
  )
}

scale_patch_matrix <- function(
    patch_matrix,
    predictor_cols,
    scaling_method,
    train_center,
    train_scale,
    temperature_min_valid_celsius
) {
  patch_matrix_scaled <- patch_matrix
  
  for (band_id in seq_along(predictor_cols)) {
    predictor_name <- predictor_cols[band_id]
    method <- scaling_method[band_id]
    x <- as.numeric(patch_matrix_scaled[, band_id])
    
    if (grepl("surface_temperature_celsius$", predictor_name)) {
      x[
        !is.na(x) &
          is.finite(x) &
          x <= temperature_min_valid_celsius
      ] <- NA_real_
    }
    
    if (method == "none_dummy_0_1") {
      x <- x
    } else if (method == "percentage_0_100_to_0_1") {
      x[
        !is.na(x) &
          is.finite(x) &
          (x < 0 | x > 100)
      ] <- NA_real_
      
      x <- x / 100
    } else if (method == "zscore_train") {
      x <- (x - train_center[band_id]) / train_scale[band_id]
    } else {
      stop("Unsupported scaling method: ", method)
    }
    
    patch_matrix_scaled[, band_id] <- x
  }
  
  patch_matrix_scaled
}

extract_chunk_patch_arrays <- function(
    point_chunk,
    raster_files,
    predictor_cols,
    scaling_method,
    train_center,
    train_scale,
    dummy_band_id,
    patch_radius_large,
    patch_size_large,
    patch_size_small,
    temperature_min_valid_celsius
) {
  raster_stack <- terra::rast(raster_files)
  names(raster_stack) <- predictor_cols
  
  n_raster_cols <- terra::ncol(raster_stack)
  n_bands <- length(predictor_cols)
  
  offset_grid <- make_offset_grid(patch_radius_large)
  n_patch_cells <- nrow(offset_grid)
  
  cell_matrix <- sapply(
    seq_len(nrow(point_chunk)),
    function(i) {
      rows_use <- point_chunk$center_row[i] + offset_grid$row_offset
      cols_use <- point_chunk$center_col[i] + offset_grid$col_offset
      (rows_use - 1) * n_raster_cols + cols_use
    }
  )
  
  if (is.vector(cell_matrix)) {
    cell_matrix <- matrix(
      cell_matrix,
      ncol = nrow(point_chunk)
    )
  }
  
  patch_values <- terra::extract(
    x = raster_stack,
    y = as.vector(cell_matrix)
  )
  
  if ("ID" %in% names(patch_values)) {
    patch_values <- patch_values %>%
      dplyr::select(-ID)
  }
  
  if (!all(predictor_cols %in% names(patch_values))) {
    missing_cols <- setdiff(predictor_cols, names(patch_values))
    print(missing_cols)
    stop("Some predictors are missing from extracted patch values.")
  }
  
  patch_matrix <- patch_values[, predictor_cols, drop = FALSE] %>%
    as.matrix()
  
  storage.mode(patch_matrix) <- "double"
  
  patch_matrix_scaled <- scale_patch_matrix(
    patch_matrix = patch_matrix,
    predictor_cols = predictor_cols,
    scaling_method = scaling_method,
    train_center = train_center,
    train_scale = train_scale,
    temperature_min_valid_celsius = temperature_min_valid_celsius
  )
  
  row_is_finite <- apply(
    patch_matrix_scaled,
    1,
    function(x) {
      all(is.finite(x))
    }
  )
  
  patch_is_finite <- colSums(
    matrix(
      row_is_finite,
      nrow = n_patch_cells,
      ncol = nrow(point_chunk)
    )
  ) == n_patch_cells
  
  patch_dummy_ok <- rep(TRUE, nrow(point_chunk))
  
  if (length(dummy_band_id) > 0) {
    dummy_values <- patch_matrix_scaled[, dummy_band_id, drop = FALSE]
    
    dummy_row_ok <- apply(
      dummy_values,
      1,
      function(x) {
        all(
          abs(x - round(x)) < 1e-6 &
            round(x) %in% c(0, 1)
        )
      }
    )
    
    patch_dummy_ok <- colSums(
      matrix(
        dummy_row_ok,
        nrow = n_patch_cells,
        ncol = nrow(point_chunk)
      )
    ) == n_patch_cells
  }
  
  patch_is_valid <- patch_is_finite & patch_dummy_ok
  valid_position <- which(patch_is_valid)
  n_valid <- length(valid_position)
  
  x_5x5_array <- array(
    data = NA_real_,
    dim = c(n_valid, n_bands, patch_size_large, patch_size_large)
  )
  
  x_3x3_array <- array(
    data = NA_real_,
    dim = c(n_valid, n_bands, patch_size_small, patch_size_small)
  )
  
  if (n_valid > 0) {
    small_start <- patch_radius_large
    small_end <- patch_radius_large + patch_size_small - 1
    
    for (j in seq_along(valid_position)) {
      point_position <- valid_position[j]
      row_id <- ((point_position - 1) * n_patch_cells + 1):(point_position * n_patch_cells)
      
      patch_one <- patch_matrix_scaled[row_id, , drop = FALSE]
      
      patch_large_one <- array(
        data = NA_real_,
        dim = c(n_bands, patch_size_large, patch_size_large)
      )
      
      for (band_id in seq_len(n_bands)) {
        patch_large_one[band_id, , ] <- matrix(
          patch_one[, band_id],
          nrow = patch_size_large,
          ncol = patch_size_large,
          byrow = FALSE
        )
      }
      
      x_5x5_array[j, , , ] <- patch_large_one
      x_3x3_array[j, , , ] <- patch_large_one[
        ,
        small_start:small_end,
        small_start:small_end,
        drop = FALSE
      ]
    }
  }
  
  points_valid <- point_chunk %>%
    dplyr::slice(valid_position)
  
  list(
    x_5x5_array = x_5x5_array,
    x_3x3_array = x_3x3_array,
    points_valid = points_valid,
    check = tibble::tibble(
      n_requested = nrow(point_chunk),
      n_valid = n_valid,
      n_removed = nrow(point_chunk) - n_valid,
      n_invalid_finite = sum(!patch_is_finite),
      n_invalid_dummy = sum(patch_is_finite & !patch_dummy_ok)
    )
  )
}

bind_array_first_dim <- function(array_list) {
  array_list <- array_list[
    vapply(
      array_list,
      function(x) {
        dim(x)[1] > 0
      },
      logical(1)
    )
  ]
  
  if (length(array_list) == 0) {
    stop("No non-empty arrays to bind.")
  }
  
  output_dim <- dim(array_list[[1]])
  output_dim[1] <- sum(
    vapply(
      array_list,
      function(x) {
        dim(x)[1]
      },
      numeric(1)
    )
  )
  
  output_array <- array(
    data = NA_real_,
    dim = output_dim
  )
  
  start_id <- 1
  
  for (i in seq_along(array_list)) {
    n_i <- dim(array_list[[i]])[1]
    end_id <- start_id + n_i - 1
    
    output_array[start_id:end_id, , , ] <- array_list[[i]]
    start_id <- end_id + 1
  }
  
  output_array
}

extract_dataset_patches <- function(
    point_data,
    dataset_name,
    raster_template,
    raster_files,
    predictor_cols,
    selected_predictors,
    predictor_scaling,
    patch_radius_large,
    patch_size_large,
    patch_size_small,
    chunk_size_points,
    n_workers_patch,
    temperature_min_valid_celsius,
    output_file
) {
  message(dataset_name, " | preparing point index")
  
  point_index <- prepare_point_index(
    point_data = point_data,
    raster_template = raster_template,
    patch_radius = patch_radius_large
  )
  
  if (nrow(point_index) == 0) {
    stop(dataset_name, " has no valid internal points for patch extraction.")
  }
  
  chunk_table <- point_index %>%
    dplyr::mutate(
      chunk_id = ceiling(dplyr::row_number() / chunk_size_points)
    )
  
  chunk_list <- split(chunk_table, chunk_table$chunk_id)
  
  scaling_method <- predictor_scaling$scaling_method
  train_center <- predictor_scaling$train_center
  train_scale <- predictor_scaling$train_scale
  dummy_band_id <- which(selected_predictors$predictor_type == "dummy")
  
  message(dataset_name, " | requested points: ", nrow(point_data))
  message(dataset_name, " | internal points: ", nrow(point_index))
  message(dataset_name, " | chunks: ", length(chunk_list))
  message(dataset_name, " | workers: ", n_workers_patch)
  
  time_start <- proc.time()[["elapsed"]]
  old_plan <- future::plan()
  
  future::plan(
    future::multisession,
    workers = n_workers_patch
  )
  
  on.exit(
    future::plan(old_plan),
    add = TRUE
  )
  
  chunk_result_list <- future.apply::future_lapply(
    chunk_list,
    FUN = extract_chunk_patch_arrays,
    raster_files = raster_files,
    predictor_cols = predictor_cols,
    scaling_method = scaling_method,
    train_center = train_center,
    train_scale = train_scale,
    dummy_band_id = dummy_band_id,
    patch_radius_large = patch_radius_large,
    patch_size_large = patch_size_large,
    patch_size_small = patch_size_small,
    temperature_min_valid_celsius = temperature_min_valid_celsius,
    future.seed = TRUE,
    future.packages = c(
      "terra",
      "dplyr",
      "tibble"
    )
  )
  
  future::plan(future::sequential)
  
  time_end <- proc.time()[["elapsed"]]
  
  message(dataset_name, " | binding arrays")
  
  x_5x5_array <- bind_array_first_dim(
    lapply(chunk_result_list, function(x) x$x_5x5_array)
  )
  
  x_3x3_array <- bind_array_first_dim(
    lapply(chunk_result_list, function(x) x$x_3x3_array)
  )
  
  points_valid <- dplyr::bind_rows(
    lapply(chunk_result_list, function(x) x$points_valid)
  )
  
  chunk_check <- dplyr::bind_rows(
    lapply(chunk_result_list, function(x) x$check)
  )
  
  y_native <- points_valid$target_native
  y_log1p <- points_valid$target_log1p
  
  runtime <- paste(
    round(time_end - time_start, 2),
    units(time_end - time_start)
  )
  
  check <- tibble::tibble(
    dataset = dataset_name,
    n_requested_points = nrow(point_data),
    n_internal_points = nrow(point_index),
    n_valid_patches = nrow(points_valid),
    n_removed_outside_border = nrow(point_data) - nrow(point_index),
    n_removed_invalid_patch = nrow(point_index) - nrow(points_valid),
    n_removed_total = nrow(point_data) - nrow(points_valid),
    removed_fraction_total = (nrow(point_data) - nrow(points_valid)) / nrow(point_data),
    n_predictors = length(predictor_cols),
    patch_5x5_dim = paste(dim(x_5x5_array), collapse = " x "),
    patch_3x3_dim = paste(dim(x_3x3_array), collapse = " x "),
    n_na_5x5 = sum(is.na(x_5x5_array)),
    n_na_3x3 = sum(is.na(x_3x3_array)),
    n_infinite_5x5 = sum(!is.finite(x_5x5_array)),
    n_infinite_3x3 = sum(!is.finite(x_3x3_array)),
    n_target_native = length(y_native),
    n_target_log1p = length(y_log1p),
    n_na_target_native = sum(is.na(y_native)),
    n_na_target_log1p = sum(is.na(y_log1p)),
    min_target_native = min(y_native, na.rm = TRUE),
    median_target_native = median(y_native, na.rm = TRUE),
    max_target_native = max(y_native, na.rm = TRUE),
    n_invalid_finite = sum(chunk_check$n_invalid_finite),
    n_invalid_dummy = sum(chunk_check$n_invalid_dummy),
    n_workers_patch = n_workers_patch,
    chunk_size_points = chunk_size_points,
    runtime = runtime
  )
  
  print(check)
  
  stopifnot(check$n_na_5x5 == 0)
  stopifnot(check$n_na_3x3 == 0)
  stopifnot(check$n_infinite_5x5 == 0)
  stopifnot(check$n_infinite_3x3 == 0)
  stopifnot(check$n_target_native == check$n_valid_patches)
  stopifnot(check$n_target_log1p == check$n_valid_patches)
  stopifnot(check$n_na_target_native == 0)
  stopifnot(check$n_na_target_log1p == 0)
  stopifnot(check$n_invalid_dummy == 0)
  
  patch_object <- list(
    x_5x5_array = x_5x5_array,
    x_3x3_array = x_3x3_array,
    y_native = y_native,
    y_log1p = y_log1p,
    points_valid = points_valid,
    check = check,
    chunk_check = chunk_check
  )
  
  saveRDS(
    patch_object,
    output_file,
    compress = FALSE
  )
  
  list(
    output_file = output_file,
    check = check,
    chunk_check = chunk_check
  )
}

input_files <- c(
  train_file,
  validation_file,
  test_file,
  selected_predictors_file,
  predictor_scaling_file,
  raster_table_file,
  target_config_file
)

check_input_files(input_files)

if (!dir.exists(predictor_dir)) {
  stop("Predictor directory was not found: ", predictor_dir)
}

message("Reading split datasets...")

train_data <- read_point_data(train_file, "train")
validation_data <- read_point_data(validation_file, "validation")
test_data <- read_point_data(test_file, "test")

message("Reading predictor metadata...")

patch_metadata <- read_patch_metadata(
  selected_predictors_file = selected_predictors_file,
  predictor_scaling_file = predictor_scaling_file,
  raster_table_file = raster_table_file
)

selected_predictors <- patch_metadata$selected_predictors
predictor_scaling <- patch_metadata$predictor_scaling
raster_table <- patch_metadata$raster_table
predictor_cols <- patch_metadata$predictor_cols
raster_files <- raster_table$raster_file

message("Building raster stack template...")

raster_stack_template <- build_raster_stack(
  raster_table = raster_table,
  predictor_cols = predictor_cols
)

raster_template <- raster_stack_template[[1]]

message("Predictors used for patches: ", length(predictor_cols))
message("Dummy predictors: ", sum(selected_predictors$predictor_type == "dummy"))
message("Percentage predictors: ", sum(selected_predictors$predictor_type == "percentage"))
message("Continuous predictors: ", sum(selected_predictors$predictor_type == "continuous"))

time_start_all     <- proc.time()[["elapsed"]]
time_start_all_str <- as.character(Sys.time())   # timestamp legível para o log

train_extraction <- extract_dataset_patches(
  point_data = train_data,
  dataset_name = "train",
  raster_template = raster_template,
  raster_files = raster_files,
  predictor_cols = predictor_cols,
  selected_predictors = selected_predictors,
  predictor_scaling = predictor_scaling,
  patch_radius_large = patch_radius_large,
  patch_size_large = patch_size_large,
  patch_size_small = patch_size_small,
  chunk_size_points = chunk_size_points,
  n_workers_patch = n_workers_patch,
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  output_file = file.path(output_patch_dir, "train_patch_data.rds")
)

gc()

validation_extraction <- extract_dataset_patches(
  point_data = validation_data,
  dataset_name = "validation",
  raster_template = raster_template,
  raster_files = raster_files,
  predictor_cols = predictor_cols,
  selected_predictors = selected_predictors,
  predictor_scaling = predictor_scaling,
  patch_radius_large = patch_radius_large,
  patch_size_large = patch_size_large,
  patch_size_small = patch_size_small,
  chunk_size_points = chunk_size_points,
  n_workers_patch = n_workers_patch,
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  output_file = file.path(output_patch_dir, "validation_patch_data.rds")
)

gc()

test_extraction <- extract_dataset_patches(
  point_data = test_data,
  dataset_name = "test",
  raster_template = raster_template,
  raster_files = raster_files,
  predictor_cols = predictor_cols,
  selected_predictors = selected_predictors,
  predictor_scaling = predictor_scaling,
  patch_radius_large = patch_radius_large,
  patch_size_large = patch_size_large,
  patch_size_small = patch_size_small,
  chunk_size_points = chunk_size_points,
  n_workers_patch = n_workers_patch,
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  output_file = file.path(output_patch_dir, "test_patch_data.rds")
)

gc()

time_end_all     <- proc.time()[["elapsed"]]
time_end_all_str <- as.character(Sys.time())   # timestamp legível para o log

patch_runtime <- paste(
  round((time_end_all - time_start_all) / 60, 2),
  "min"
)

patch_check <- dplyr::bind_rows(
  train_extraction$check,
  validation_extraction$check,
  test_extraction$check
)

patch_chunk_check <- dplyr::bind_rows(
  train_extraction$chunk_check %>% dplyr::mutate(dataset = "train"),
  validation_extraction$chunk_check %>% dplyr::mutate(dataset = "validation"),
  test_extraction$chunk_check %>% dplyr::mutate(dataset = "test")
)

patch_runtime_check <- tibble::tibble(
  time_start = time_start_all_str,
  time_end   = time_end_all_str,
  runtime = patch_runtime,
  n_workers_patch = n_workers_patch,
  chunk_size_points = chunk_size_points,
  future_globals_max_size_gb = future_globals_max_size_gb
)

patch_config <- tibble::tibble(
  target_label = target_label,
  predictor_dir = predictor_dir,
  n_predictors = length(predictor_cols),
  patch_size_large = patch_size_large,
  patch_size_small = patch_size_small,
  patch_radius_large = patch_radius_large,
  tensor_layout = "samples x channels x height x width",
  patch_storage_format = "RDS arrays",
  n_workers_patch = n_workers_patch,
  chunk_size_points = chunk_size_points,
  future_globals_max_size_gb = future_globals_max_size_gb,
  temperature_min_valid_celsius = temperature_min_valid_celsius,
  train_file = train_file,
  validation_file = validation_file,
  test_file = test_file,
  train_patch_file = train_extraction$output_file,
  validation_patch_file = validation_extraction$output_file,
  test_patch_file = test_extraction$output_file
)

patch_manifest <- list(
  patch_config = patch_config,
  selected_predictors = selected_predictors,
  predictor_scaling = predictor_scaling,
  raster_table = raster_table,
  patch_check = patch_check,
  patch_chunk_check = patch_chunk_check,
  patch_runtime_check = patch_runtime_check,
  train_patch_file = train_extraction$output_file,
  validation_patch_file = validation_extraction$output_file,
  test_patch_file = test_extraction$output_file
)

saveRDS(
  patch_manifest,
  file.path(output_patch_dir, "patch_manifest.rds"),
  compress = FALSE
)

readr::write_csv2(patch_check, file.path(output_metadata_dir, "patch_extraction_check.csv"))
readr::write_csv2(patch_chunk_check, file.path(output_metadata_dir, "patch_chunk_check.csv"))
readr::write_csv2(patch_runtime_check, file.path(output_metadata_dir, "patch_runtime_check.csv"))
readr::write_csv2(patch_config, file.path(output_metadata_dir, "patch_config.csv"))

readr::write_csv2(
  selected_predictors,
  file.path(output_metadata_dir, "predictor_order.csv")
)

readr::write_csv2(
  predictor_scaling,
  file.path(output_metadata_dir, "predictor_scaling.csv")
)

plot_patch_validity <- patch_check %>%
  dplyr::select(
    dataset,
    n_valid_patches,
    n_removed_total
  ) %>%
  tidyr::pivot_longer(
    cols = c(n_valid_patches, n_removed_total),
    names_to = "patch_status",
    values_to = "n"
  ) %>%
  ggplot2::ggplot(
    ggplot2::aes(
      x = dataset,
      y = n,
      fill = patch_status
    )
  ) +
  ggplot2::geom_col(width = 0.7) +
  ggplot2::labs(
    x = "Dataset",
    y = "Number of points",
    fill = "Patch status",
    title = paste0(target_label, " patch extraction validity")
  ) +
  ggplot2::theme_bw()

save_plot(
  plot_patch_validity,
  file.path(output_figure_dir, "patch_extraction_validity.png"),
  width = 8,
  height = 5
)

plot_removed_fraction <- patch_check %>%
  ggplot2::ggplot(
    ggplot2::aes(
      x = dataset,
      y = 100 * removed_fraction_total
    )
  ) +
  ggplot2::geom_col(width = 0.6) +
  ggplot2::labs(
    x = "Dataset",
    y = "Removed patches, %",
    title = paste0(target_label, " removed patches during extraction")
  ) +
  ggplot2::theme_bw()

save_plot(
  plot_removed_fraction,
  file.path(output_figure_dir, "patch_removed_fraction.png"),
  width = 8,
  height = 5
)

message("Patch extraction finished.")
message("Patch files saved to: ", output_patch_dir)
message("Metadata saved to: ", output_metadata_dir)
message("Figures saved to: ", output_figure_dir)
