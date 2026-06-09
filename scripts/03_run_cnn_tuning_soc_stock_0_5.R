source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/install_load_pkg.R"
)

pkg <- c(
  "torch",
  "coro",
  "dplyr",
  "tidyr",
  "readr",
  "tibble",
  "purrr",
  "ggplot2",
  "DescTools"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

target_label <- "soc_stock_0_5cm"
target_unit <- "ton_ha"
target_version <- "log1p"

tune_length <- 100
tune_grid <- NULL
stop_on_error <- FALSE
use_cuda_if_available <- TRUE
print_every <- 5

tuning_run_id <- paste0(
  "cnn_tuning_",
  target_label,
  "_",
  format(Sys.time(), "%Y%m%d_%H%M%S")
)

input_patch_dir <- file.path(
  "outputs",
  "cnn_patches",
  "soc_stock_modeling",
  target_label
)

input_patch_metadata_dir <- file.path(
  "outputs",
  "metadata",
  "soc_stock_modeling",
  target_label,
  "cnn_patches"
)

output_root_dir <- file.path(
  "outputs",
  "tuning",
  "soc_stock_modeling",
  target_label,
  tuning_run_id
)

output_config_root_dir <- file.path(output_root_dir, "configs")
output_comparison_dir <- file.path(output_root_dir, "comparison")
output_comparison_table_dir <- file.path(output_comparison_dir, "tables")
output_comparison_figure_dir <- file.path(output_comparison_dir, "figures")
output_decision_dir <- file.path(output_comparison_dir, "decision")

purrr::walk(
  c(
    output_root_dir,
    output_config_root_dir,
    output_comparison_dir,
    output_comparison_table_dir,
    output_comparison_figure_dir,
    output_decision_dir
  ),
  function(path_value) {
    dir.create(path_value, recursive = TRUE, showWarnings = FALSE)
  }
)

train_patch_file <- file.path(input_patch_dir, "train_patch_data.rds")
validation_patch_file <- file.path(input_patch_dir, "validation_patch_data.rds")
test_patch_file <- file.path(input_patch_dir, "test_patch_data.rds")
patch_manifest_file <- file.path(input_patch_dir, "patch_manifest.rds")
predictor_order_file <- file.path(input_patch_metadata_dir, "predictor_order.csv")

required_input_files <- c(
  train_patch_file,
  validation_patch_file,
  test_patch_file,
  patch_manifest_file,
  predictor_order_file
)

missing_input_files <- required_input_files[!file.exists(required_input_files)]

if (length(missing_input_files) > 0) {
  print(missing_input_files)
  stop("Some required patch files were not found.")
}

safe_write_csv2 <- function(data, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  readr::write_csv2(data, path)
  invisible(path)
}

safe_save_rds <- function(object, path, compress = FALSE) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  saveRDS(object = object, file = path, compress = compress)
  invisible(path)
}

safe_torch_save <- function(object, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  torch::torch_save(object, path)
  invisible(path)
}

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

default_config <- function() {
  tibble::tibble(
    config_id = "cfg_001",
    gate_type = "vector_featurewise",
    fusion_type = "fused_plus_abs_difference",
    branch3_mode = "conv3_padding1_then_conv1x1",
    branch5_mode = "conv3_stack_padding1",
    use_se_block = TRUE,
    se_reduction = 16L,
    activation = "silu",
    conv_channels_1 = 64L,
    conv_channels_2 = 128L,
    conv_channels_3 = 128L,
    embedding_dim = 384L,
    branch_spatial_dropout = 0.03,
    branch_embedding_dropout = 0.00,
    gate_dropout = 0.10,
    head_dropout_1 = 0.20,
    head_dropout_2 = 0.10,
    loss_function = "smooth_l1",
    smoothl1_beta = NA_real_,
    asymmetric_alpha = NA_real_,
    use_sample_weights = FALSE,
    weight_scheme = NA_character_,
    checkpoint_metric = "validation_loss",
    optimizer_type = "adam",
    batch_size_train = 512L,
    batch_size_eval = 2048L,
    n_epochs = 700L,
    patience = 90L,
    early_stopping_min_delta = 0.0005,
    base_learning_rate = 0.001,
    warmup_start_lr = 0.00001,
    warmup_epochs = 5L,
    scheduler_type = "reduce_lr_on_plateau",
    lr_plateau_factor = 0.5,
    lr_plateau_patience = 20L,
    lr_plateau_min_delta = 0.0005,
    min_learning_rate = 0.000001,
    weight_decay = 0.0001,
    gradient_clip_norm = 1.0,
    seed = 456L,
    torch_threads = 16L
  )
}

bind_configs <- function(config_list) {
  dplyr::bind_rows(config_list) %>%
    dplyr::mutate(
      config_id = sprintf("cfg_%03d", dplyr::row_number())
    )
}

build_tune_grid_from_length <- function(tune_length = 1) {
  base <- default_config()
  
  configs <- list(base)
  
  if (tune_length >= 3) {
    configs <- c(
      configs,
      list(
        base %>% dplyr::mutate(embedding_dim = 256L),
        base %>% dplyr::mutate(embedding_dim = 512L)
      )
    )
  }
  
  if (tune_length >= 5) {
    configs <- c(
      configs,
      list(
        base %>% dplyr::mutate(head_dropout_1 = 0.30),
        base %>% dplyr::mutate(branch_spatial_dropout = 0.08)
      )
    )
  }
  
  if (tune_length >= 10) {
    configs <- c(
      configs,
      list(
        base %>% dplyr::mutate(gate_type = "scalar_per_sample"),
        base %>% dplyr::mutate(gate_type = "no_gate_concat", gate_dropout = 0),
        base %>% dplyr::mutate(base_learning_rate = 0.0005),
        base %>% dplyr::mutate(loss_function = "smooth_l1_beta", smoothl1_beta = 0.75),
        base %>% dplyr::mutate(checkpoint_metric = "composite_mae_ccc")
      )
    )
  }
  
  if (tune_length > 10) {
    extra <- tidyr::crossing(
      gate_type = c("vector_featurewise", "scalar_per_sample", "no_gate_concat"),
      embedding_dim = c(256L, 384L, 512L),
      base_learning_rate = c(0.0005, 0.001),
      seed = c(123L, 456L)
    ) %>%
      dplyr::mutate(row_id = dplyr::row_number()) %>%
      dplyr::filter(row_id <= tune_length) %>%
      dplyr::select(-row_id)
    
    extra_configs <- purrr::pmap(
      extra,
      function(gate_type, embedding_dim, base_learning_rate, seed) {
        base %>%
          dplyr::mutate(
            gate_type = gate_type,
            embedding_dim = embedding_dim,
            base_learning_rate = base_learning_rate,
            seed = seed,
            gate_dropout = dplyr::if_else(gate_type == "no_gate_concat", 0, gate_dropout)
          )
      }
    )
    
    configs <- c(configs, extra_configs)
  }
  
  bind_configs(configs) %>%
    dplyr::slice_head(n = tune_length)
}

validate_tune_grid <- function(tune_grid) {
  required_cols <- names(default_config())
  missing_cols <- setdiff(required_cols, names(tune_grid))
  
  if (length(missing_cols) > 0) {
    print(missing_cols)
    stop("tune_grid is missing required columns.")
  }
  
  allowed_gate <- c("vector_featurewise", "scalar_per_sample", "no_gate_concat")
  allowed_fusion <- c("fused_plus_abs_difference", "concat_features_and_abs_difference")
  allowed_activation <- c("silu", "relu", "gelu")
  allowed_loss <- c("smooth_l1", "smooth_l1_beta", "asymmetric_mse", "mse")
  allowed_checkpoint <- c("validation_loss", "composite_mae_ccc")
  allowed_optimizer <- c("adam", "adamw")
  allowed_scheduler <- c("reduce_lr_on_plateau", "none")
  
  bad <- tune_grid %>%
    dplyr::filter(
      !gate_type %in% allowed_gate |
        !fusion_type %in% allowed_fusion |
        !activation %in% allowed_activation |
        !loss_function %in% allowed_loss |
        !checkpoint_metric %in% allowed_checkpoint |
        !optimizer_type %in% allowed_optimizer |
        !scheduler_type %in% allowed_scheduler
    )
  
  if (nrow(bad) > 0) {
    print(bad)
    stop("Unsupported tuning values were found.")
  }
  
  bad_ranges <- tune_grid %>%
    dplyr::filter(
      conv_channels_1 < 16 | conv_channels_1 > 128 |
        conv_channels_2 < 32 | conv_channels_2 > 256 |
        conv_channels_3 < 32 | conv_channels_3 > 256 |
        embedding_dim < 64 | embedding_dim > 512 |
        branch_spatial_dropout < 0 | branch_spatial_dropout > 0.20 |
        branch_embedding_dropout < 0 | branch_embedding_dropout > 0.30 |
        gate_dropout < 0 | gate_dropout > 0.30 |
        head_dropout_1 < 0 | head_dropout_1 > 0.50 |
        head_dropout_2 < 0 | head_dropout_2 > 0.40 |
        base_learning_rate < 0.0001 | base_learning_rate > 0.003 |
        weight_decay < 0 | weight_decay > 0.005 |
        gradient_clip_norm <= 0 | gradient_clip_norm > 5
    )
  
  if (nrow(bad_ranges) > 0) {
    print(bad_ranges)
    stop("Some tuning values are outside accepted ranges.")
  }
  
  bad_loss <- tune_grid %>%
    dplyr::filter(
      loss_function == "smooth_l1_beta" & (is.na(smoothl1_beta) | smoothl1_beta <= 0) |
        loss_function == "asymmetric_mse" & (is.na(asymmetric_alpha) | asymmetric_alpha <= 0 | asymmetric_alpha >= 1)
    )
  
  if (nrow(bad_loss) > 0) {
    print(bad_loss)
    stop("Some loss configurations are incomplete.")
  }
  
  tune_grid %>%
    dplyr::mutate(
      gate_dropout = dplyr::if_else(gate_type == "no_gate_concat", 0, gate_dropout),
      config_id = as.character(config_id)
    )
}

if (!is.null(tune_grid)) {
  message("Using user-provided tune_grid.")
  if (!is.null(tune_length)) {
    message("Both tune_grid and tune_length were supplied. tune_grid has priority.")
  }
} else {
  if (is.null(tune_length)) {
    tune_length <- 1
  }
  message("Building tune_grid from tune_length = ", tune_length)
  tune_grid <- build_tune_grid_from_length(tune_length)
}

tune_grid <- validate_tune_grid(tune_grid)

safe_write_csv2(
  tune_grid,
  file.path(output_comparison_table_dir, "tune_grid.csv")
)

message("Total configurations: ", nrow(tune_grid))
message("Max training cost: ", sum(tune_grid$n_epochs), " epochs across all configs.")
message("Patch input directory: ", input_patch_dir)
message("Output directory: ", output_root_dir)
print(tune_grid, width = Inf)

torch_threads_were_set <- FALSE

set_torch_runtime <- function(torch_threads, seed) {
  set.seed(seed)
  torch::torch_manual_seed(seed)
  
  if (!torch_threads_were_set) {
    if ("torch_set_num_threads" %in% getNamespaceExports("torch")) {
      try(torch::torch_set_num_threads(torch_threads), silent = TRUE)
    }
    
    if ("torch_set_num_interop_threads" %in% getNamespaceExports("torch")) {
      try(torch::torch_set_num_interop_threads(torch_threads), silent = TRUE)
    }
    
    torch_threads_were_set <<- TRUE
  }
}

device <- if (use_cuda_if_available && torch::cuda_is_available()) {
  torch::torch_device("cuda")
} else {
  torch::torch_device("cpu")
}

message("Using device: ", device$type)

read_patch_data <- function() {
  train_patch <- readRDS(train_patch_file)
  validation_patch <- readRDS(validation_patch_file)
  test_patch <- readRDS(test_patch_file)
  patch_manifest <- readRDS(patch_manifest_file)
  predictor_order <- readr::read_csv2(predictor_order_file, show_col_types = FALSE)
  
  n_predictors <- dim(train_patch$x_3x3_array)[2]
  
  list(
    train_patch = train_patch,
    validation_patch = validation_patch,
    test_patch = test_patch,
    patch_manifest = patch_manifest,
    predictor_order = predictor_order,
    n_predictors = n_predictors
  )
}

make_tensor_bundle <- function(patch_data) {
  train_patch <- patch_data$train_patch
  validation_patch <- patch_data$validation_patch
  test_patch <- patch_data$test_patch
  
  list(
    x_train_3x3 = torch::torch_tensor(train_patch$x_3x3_array, dtype = torch::torch_float()),
    x_train_5x5 = torch::torch_tensor(train_patch$x_5x5_array, dtype = torch::torch_float()),
    y_train = torch::torch_tensor(as.numeric(train_patch$y_log1p), dtype = torch::torch_float())$view(c(-1, 1)),
    x_validation_3x3 = torch::torch_tensor(validation_patch$x_3x3_array, dtype = torch::torch_float()),
    x_validation_5x5 = torch::torch_tensor(validation_patch$x_5x5_array, dtype = torch::torch_float()),
    y_validation = torch::torch_tensor(as.numeric(validation_patch$y_log1p), dtype = torch::torch_float())$view(c(-1, 1)),
    x_test_3x3 = torch::torch_tensor(test_patch$x_3x3_array, dtype = torch::torch_float()),
    x_test_5x5 = torch::torch_tensor(test_patch$x_5x5_array, dtype = torch::torch_float()),
    y_test = torch::torch_tensor(as.numeric(test_patch$y_log1p), dtype = torch::torch_float())$view(c(-1, 1)),
    train_points = train_patch$points_valid,
    validation_points = validation_patch$points_valid,
    test_points = test_patch$points_valid
  )
}

patch_data <- read_patch_data()
tensor_bundle <- make_tensor_bundle(patch_data)

rm(patch_data)
gc()

make_loaders <- function(tensor_bundle, batch_size_train, batch_size_eval) {
  train_dataset <- torch::tensor_dataset(
    tensor_bundle$x_train_3x3,
    tensor_bundle$x_train_5x5,
    tensor_bundle$y_train
  )
  
  validation_dataset <- torch::tensor_dataset(
    tensor_bundle$x_validation_3x3,
    tensor_bundle$x_validation_5x5,
    tensor_bundle$y_validation
  )
  
  test_dataset <- torch::tensor_dataset(
    tensor_bundle$x_test_3x3,
    tensor_bundle$x_test_5x5,
    tensor_bundle$y_test
  )
  
  list(
    train_loader = torch::dataloader(train_dataset, batch_size = batch_size_train, shuffle = TRUE),
    train_eval_loader = torch::dataloader(train_dataset, batch_size = batch_size_eval, shuffle = FALSE),
    validation_loader = torch::dataloader(validation_dataset, batch_size = batch_size_eval, shuffle = FALSE),
    test_loader = torch::dataloader(test_dataset, batch_size = batch_size_eval, shuffle = FALSE)
  )
}

make_activation <- function(activation) {
  if (activation == "silu" && "nn_silu" %in% getNamespaceExports("torch")) {
    torch::nn_silu()
  } else if (activation == "gelu" && "nn_gelu" %in% getNamespaceExports("torch")) {
    torch::nn_gelu()
  } else {
    torch::nn_relu()
  }
}

identity_module <- torch::nn_module(
  initialize = function() {},
  forward = function(x) {
    x
  }
)

se_block <- torch::nn_module(
  initialize = function(channels, reduction = 16, activation = "silu") {
    hidden_channels <- max(4L, as.integer(channels %/% reduction))
    self$fc1 <- torch::nn_linear(channels, hidden_channels)
    self$activation <- make_activation(activation)
    self$fc2 <- torch::nn_linear(hidden_channels, channels)
  },
  forward = function(x) {
    s <- torch::nnf_adaptive_avg_pool2d(x, output_size = c(1, 1))
    s <- s$view(c(s$size(1), s$size(2)))
    s <- self$fc1(s)
    s <- self$activation(s)
    s <- self$fc2(s)
    s <- torch::torch_sigmoid(s)
    s <- s$view(c(s$size(1), s$size(2), 1, 1))
    x * s
  }
)

cnn_tuning_model <- torch::nn_module(
  initialize = function(n_channels, config) {
    activation <- config$activation
    conv_channels_1 <- as.integer(config$conv_channels_1)
    conv_channels_2 <- as.integer(config$conv_channels_2)
    conv_channels_3 <- as.integer(config$conv_channels_3)
    embedding_dim <- as.integer(config$embedding_dim)
    se_reduction <- as.integer(config$se_reduction)
    
    branch3_padding_first <- ifelse(config$branch3_mode == "conv3_padding0_then_conv1x1", 0L, 1L)
    branch3_flatten_size <- ifelse(branch3_padding_first == 0L, conv_channels_3 * 1 * 1, conv_channels_3 * 3 * 3)
    branch5_flatten_size <- conv_channels_3 * 5 * 5
    
    se3 <- if (isTRUE(config$use_se_block)) {
      se_block(conv_channels_3, se_reduction, activation)
    } else {
      identity_module()
    }
    
    se5 <- if (isTRUE(config$use_se_block)) {
      se_block(conv_channels_3, se_reduction, activation)
    } else {
      identity_module()
    }
    
    self$gate_type <- config$gate_type
    self$embedding_dim <- embedding_dim
    
    self$branch_3x3 <- torch::nn_sequential(
      torch::nn_conv2d(n_channels, conv_channels_1, kernel_size = 3, padding = branch3_padding_first),
      torch::nn_batch_norm2d(conv_channels_1),
      make_activation(activation),
      torch::nn_conv2d(conv_channels_1, conv_channels_2, kernel_size = 1, padding = 0),
      torch::nn_batch_norm2d(conv_channels_2),
      make_activation(activation),
      torch::nn_conv2d(conv_channels_2, conv_channels_3, kernel_size = 1, padding = 0),
      torch::nn_batch_norm2d(conv_channels_3),
      make_activation(activation),
      se3,
      torch::nn_dropout2d(p = config$branch_spatial_dropout),
      torch::nn_flatten(),
      torch::nn_linear(branch3_flatten_size, embedding_dim),
      torch::nn_batch_norm1d(embedding_dim),
      make_activation(activation),
      torch::nn_dropout(p = config$branch_embedding_dropout)
    )
    
    self$branch_5x5 <- torch::nn_sequential(
      torch::nn_conv2d(n_channels, conv_channels_1, kernel_size = 3, padding = 1),
      torch::nn_batch_norm2d(conv_channels_1),
      make_activation(activation),
      torch::nn_conv2d(conv_channels_1, conv_channels_2, kernel_size = 3, padding = 1),
      torch::nn_batch_norm2d(conv_channels_2),
      make_activation(activation),
      torch::nn_conv2d(conv_channels_2, conv_channels_3, kernel_size = 3, padding = 1),
      torch::nn_batch_norm2d(conv_channels_3),
      make_activation(activation),
      se5,
      torch::nn_dropout2d(p = config$branch_spatial_dropout),
      torch::nn_flatten(),
      torch::nn_linear(branch5_flatten_size, embedding_dim),
      torch::nn_batch_norm1d(embedding_dim),
      make_activation(activation),
      torch::nn_dropout(p = config$branch_embedding_dropout)
    )
    
    if (self$gate_type == "vector_featurewise") {
      self$gate_net <- torch::nn_sequential(
        torch::nn_linear(embedding_dim * 4, embedding_dim),
        torch::nn_batch_norm1d(embedding_dim),
        make_activation(activation),
        torch::nn_dropout(p = config$gate_dropout),
        torch::nn_linear(embedding_dim, embedding_dim),
        torch::nn_sigmoid()
      )
      head_input_dim <- embedding_dim * 2
    } else if (self$gate_type == "scalar_per_sample") {
      self$gate_net <- torch::nn_sequential(
        torch::nn_linear(embedding_dim * 4, embedding_dim),
        torch::nn_batch_norm1d(embedding_dim),
        make_activation(activation),
        torch::nn_dropout(p = config$gate_dropout),
        torch::nn_linear(embedding_dim, 1),
        torch::nn_sigmoid()
      )
      head_input_dim <- embedding_dim * 2
    } else {
      self$gate_net <- NULL
      head_input_dim <- embedding_dim * 3
    }
    
    self$head <- torch::nn_sequential(
      torch::nn_linear(head_input_dim, embedding_dim),
      torch::nn_batch_norm1d(embedding_dim),
      make_activation(activation),
      torch::nn_dropout(p = config$head_dropout_1),
      torch::nn_linear(embedding_dim, 128),
      torch::nn_batch_norm1d(128),
      make_activation(activation),
      torch::nn_dropout(p = config$head_dropout_2),
      torch::nn_linear(128, 64),
      make_activation(activation),
      torch::nn_linear(64, 1)
    )
  },
  forward_with_gate = function(x_3x3, x_5x5) {
    features_3x3 <- self$branch_3x3(x_3x3)
    features_5x5 <- self$branch_5x5(x_5x5)
    abs_difference <- torch::torch_abs(features_3x3 - features_5x5)
    product_interaction <- features_3x3 * features_5x5
    
    if (self$gate_type == "no_gate_concat") {
      gate <- torch::torch_full(
        size = c(features_3x3$size(1), 1),
        fill_value = NA_real_,
        device = features_3x3$device
      )
      head_input <- torch::torch_cat(
        list(features_3x3, features_5x5, abs_difference),
        dim = 2
      )
    } else {
      gate_input <- torch::torch_cat(
        list(features_3x3, features_5x5, abs_difference, product_interaction),
        dim = 2
      )
      gate <- self$gate_net(gate_input)
      fused_features <- gate * features_3x3 + (1 - gate) * features_5x5
      head_input <- torch::torch_cat(list(fused_features, abs_difference), dim = 2)
    }
    
    pred <- self$head(head_input)
    
    list(
      pred = pred,
      gate = gate,
      features_3x3 = features_3x3,
      features_5x5 = features_5x5,
      abs_difference = abs_difference
    )
  },
  forward = function(x_3x3, x_5x5) {
    self$forward_with_gate(x_3x3, x_5x5)$pred
  }
)

make_loss_function <- function(config) {
  function(pred, target) {
    if (config$loss_function == "mse") {
      torch::nnf_mse_loss(pred, target)
    } else if (config$loss_function == "smooth_l1_beta") {
      beta <- config$smoothl1_beta
      diff_abs <- torch::torch_abs(pred - target)
      loss <- torch::torch_where(
        diff_abs < beta,
        0.5 * diff_abs^2 / beta,
        diff_abs - 0.5 * beta
      )
      loss$mean()
    } else if (config$loss_function == "asymmetric_mse") {
      alpha <- config$asymmetric_alpha
      error <- pred - target
      weight <- torch::torch_where(error < 0, alpha, 1 - alpha)
      (weight * error^2)$mean()
    } else {
      torch::nnf_smooth_l1_loss(pred, target)
    }
  }
}

make_optimizer <- function(model, config) {
  if (config$optimizer_type == "adamw" && "optim_adamw" %in% getNamespaceExports("torch")) {
    torch::optim_adamw(
      params = model$parameters,
      lr = config$warmup_start_lr,
      weight_decay = config$weight_decay
    )
  } else {
    torch::optim_adam(
      params = model$parameters,
      lr = config$warmup_start_lr,
      weight_decay = config$weight_decay
    )
  }
}

set_optimizer_lr <- function(optimizer, lr) {
  for (i in seq_along(optimizer$param_groups)) {
    optimizer$param_groups[[i]]$lr <- lr
  }
}

get_optimizer_lr <- function(optimizer) {
  optimizer$param_groups[[1]]$lr
}

calc_metrics <- function(obs, pred) {
  obs <- as.numeric(obs)
  pred <- as.numeric(pred)
  
  keep <- !is.na(obs) & !is.na(pred) & is.finite(obs) & is.finite(pred)
  obs <- obs[keep]
  pred <- pred[keep]
  
  if (length(obs) < 2) {
    return(tibble::tibble(n = length(obs), ccc = NA_real_, r2 = NA_real_, mae = NA_real_, nse = NA_real_, rmse = NA_real_, mqi = NA_real_))
  }
  
  ccc <- try(DescTools::CCC(obs, pred, conf.level = 0.95)$rho.c$est, silent = TRUE)
  ccc <- ifelse(inherits(ccc, "try-error"), NA_real_, as.numeric(ccc)[1])
  
  r2 <- try(cor(pred, obs, use = "pairwise.complete.obs"), silent = TRUE)
  r2 <- ifelse(inherits(r2, "try-error"), NA_real_, as.numeric(r2)^2)
  
  mae <- mean(abs(pred - obs), na.rm = TRUE)
  nse <- 1 - (sum((obs - pred)^2, na.rm = TRUE) / sum((obs - mean(obs, na.rm = TRUE))^2, na.rm = TRUE))
  rmse <- sqrt(mean((pred - obs)^2, na.rm = TRUE))
  mqi <- ifelse(is.na(ccc) | is.na(nse) | is.na(mae) | mean(obs, na.rm = TRUE) == 0 | mae == 0, NA_real_, (ccc * nse) / (mae / mean(obs, na.rm = TRUE)))
  
  tibble::tibble(n = length(obs), ccc = ccc, r2 = r2, mae = mae, nse = nse, rmse = rmse, mqi = mqi)
}

make_performance_table <- function(pred_obs_data) {
  pred_obs_data %>%
    dplyr::group_by(config_id, dataset_role) %>%
    dplyr::group_modify(~ calc_metrics(obs = .x$obs, pred = .x$pred)) %>%
    dplyr::ungroup()
}

make_quantile_performance <- function(pred_obs_data) {
  pred_obs_data %>%
    dplyr::mutate(
      obs_quantile_group = dplyr::case_when(
        obs <= stats::quantile(obs, 0.25, na.rm = TRUE) ~ "Q00_Q25",
        obs <= stats::quantile(obs, 0.50, na.rm = TRUE) ~ "Q25_Q50",
        obs <= stats::quantile(obs, 0.75, na.rm = TRUE) ~ "Q50_Q75",
        obs <= stats::quantile(obs, 0.90, na.rm = TRUE) ~ "Q75_Q90",
        obs <= stats::quantile(obs, 0.95, na.rm = TRUE) ~ "Q90_Q95",
        obs <= stats::quantile(obs, 0.99, na.rm = TRUE) ~ "Q95_Q99",
        TRUE ~ "Q99_Q100"
      )
    ) %>%
    dplyr::group_by(config_id, dataset_role, obs_quantile_group) %>%
    dplyr::group_modify(~ calc_metrics(obs = .x$obs, pred = .x$pred)) %>%
    dplyr::ungroup()
}

evaluate_loss <- function(model, data_loader, loss_fn, device) {
  model$eval()
  total_loss <- 0
  total_n <- 0
  
  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      x_3x3 <- batch[[1]]$to(device = device)
      x_5x5 <- batch[[2]]$to(device = device)
      y <- batch[[3]]$to(device = device)
      pred <- model(x_3x3, x_5x5)
      loss <- loss_fn(pred, y)
      batch_n <- y$size(1)
      total_loss <- total_loss + as.numeric(loss$item()) * batch_n
      total_n <- total_n + batch_n
    })
  })
  
  total_loss / total_n
}

predict_loader <- function(model, data_loader, points_valid, dataset_role, config_id, device) {
  model$eval()
  pred_log1p_all <- c()
  
  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      pred_batch <- model(
        batch[[1]]$to(device = device),
        batch[[2]]$to(device = device)
      )
      pred_log1p_all <- c(pred_log1p_all, as.numeric(pred_batch$to(device = "cpu")))
    })
  })
  
  pred_native <- pmax(expm1(pred_log1p_all), 0)
  obs_native <- as.numeric(points_valid$target_native)
  
  if (length(pred_native) != nrow(points_valid)) {
    stop("Prediction length does not match metadata rows for ", dataset_role)
  }
  
  tibble::tibble(
    profile_id = points_valid$profile_id,
    sample_id = points_valid$sample_id,
    config_id = config_id,
    dataset_role = dataset_role,
    target_version = target_version,
    obs = obs_native,
    pred = pred_native,
    obs_log1p = as.numeric(points_valid$target_log1p),
    pred_log1p = pred_log1p_all,
    residual = pred_native - obs_native,
    abs_error = abs(pred_native - obs_native)
  )
}

extract_gate_values <- function(model, data_loader, points_valid, dataset_role, config_id, gate_type, device) {
  if (gate_type == "no_gate_concat") {
    return(
      list(
        gate_summary = tibble::tibble(config_id = config_id, dataset_role = dataset_role, gate_type = gate_type),
        gate_by_profile = tibble::tibble()
      )
    )
  }
  
  model$eval()
  mean_gate_profile_all <- c()
  
  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      out <- model$forward_with_gate(
        batch[[1]]$to(device = device),
        batch[[2]]$to(device = device)
      )
      gate <- as.array(out$gate$to(device = "cpu"))
      if (length(dim(gate)) == 1) {
        gate <- matrix(gate, nrow = 1)
      }
      mean_gate_profile_all <- c(mean_gate_profile_all, rowMeans(gate))
    })
  })
  
  gate_by_profile <- tibble::tibble(
    profile_id = points_valid$profile_id,
    sample_id = points_valid$sample_id,
    config_id = config_id,
    dataset_role = dataset_role,
    target_native = as.numeric(points_valid$target_native),
    mean_gate_profile = mean_gate_profile_all
  )
  
  gate_summary <- gate_by_profile %>%
    dplyr::summarise(
      config_id = dplyr::first(.data$config_id),
      dataset_role = dplyr::first(.data$dataset_role),
      gate_type = gate_type[1],
      n = dplyr::n(),
      mean_gate = mean(mean_gate_profile, na.rm = TRUE),
      sd_gate = sd(mean_gate_profile, na.rm = TRUE),
      min_gate = min(mean_gate_profile, na.rm = TRUE),
      q25_gate = as.numeric(stats::quantile(mean_gate_profile, 0.25, na.rm = TRUE)),
      median_gate = median(mean_gate_profile, na.rm = TRUE),
      q75_gate = as.numeric(stats::quantile(mean_gate_profile, 0.75, na.rm = TRUE)),
      max_gate = max(mean_gate_profile, na.rm = TRUE)
    )
  
  list(
    gate_summary = gate_summary,
    gate_by_profile = gate_by_profile
  )
}

train_one_config <- function(config, tensor_bundle, device) {
  config_id <- config$config_id
  message("Starting config: ", config_id)
  
  set_torch_runtime(
    torch_threads = config$torch_threads,
    seed = config$seed
  )
  
  config_dir <- file.path(output_config_root_dir, config_id)
  model_dir <- file.path(config_dir, "model")
  performance_dir <- file.path(config_dir, "performance")
  prediction_dir <- file.path(config_dir, "pred_obs")
  figure_dir <- file.path(config_dir, "figures")
  log_dir <- file.path(config_dir, "training_log")
  gate_dir <- file.path(config_dir, "gate")
  
  purrr::walk(
    c(config_dir, model_dir, performance_dir, prediction_dir, figure_dir, log_dir, gate_dir),
    function(path_value) {
      dir.create(path_value, recursive = TRUE, showWarnings = FALSE)
    }
  )
  
  safe_write_csv2(config, file.path(config_dir, "model_config.csv"))
  
  loaders <- make_loaders(
    tensor_bundle = tensor_bundle,
    batch_size_train = config$batch_size_train,
    batch_size_eval = config$batch_size_eval
  )
  
  n_channels <- tensor_bundle$x_train_3x3$shape[[2]]
  
  model <- cnn_tuning_model(
    n_channels = n_channels,
    config = config
  )
  model$to(device = device)
  
  loss_fn <- make_loss_function(config)
  optimizer <- make_optimizer(model, config)
  
  best_metric <- Inf
  best_epoch <- NA_integer_
  best_state <- NULL
  epochs_without_improvement <- 0L
  current_lr <- config$warmup_start_lr
  plateau_best <- Inf
  plateau_bad_epochs <- 0L
  
  training_history <- list()
  time_start <- Sys.time()

  for (epoch in seq_len(config$n_epochs)) {
    model$train()
    
    if (epoch <= config$warmup_epochs) {
      current_lr <- config$warmup_start_lr +
        (config$base_learning_rate - config$warmup_start_lr) * (epoch / config$warmup_epochs)
      set_optimizer_lr(optimizer, current_lr)
    }
    
    train_loss_sum <- 0
    train_n <- 0
    
    coro::loop(for (batch in loaders$train_loader) {
      optimizer$zero_grad()
      
      x_3x3 <- batch[[1]]$to(device = device)
      x_5x5 <- batch[[2]]$to(device = device)
      y <- batch[[3]]$to(device = device)
      
      pred <- model(x_3x3, x_5x5)
      loss <- loss_fn(pred, y)
      
      loss$backward()
      if (
        config$gradient_clip_norm > 0 &&
        "nn_utils_clip_grad_norm_" %in% getNamespaceExports("torch")
      ) {
        torch::nn_utils_clip_grad_norm_(
          parameters = model$parameters,
          max_norm = config$gradient_clip_norm
        )
      }
      optimizer$step()
      
      batch_n <- y$size(1)
      train_loss_sum <- train_loss_sum + as.numeric(loss$item()) * batch_n
      train_n <- train_n + batch_n
    })
    
    train_loss <- train_loss_sum / train_n
    validation_loss <- evaluate_loss(model, loaders$validation_loader, loss_fn, device)
    
    pred_validation_epoch <- predict_loader(
      model = model,
      data_loader = loaders$validation_loader,
      points_valid = tensor_bundle$validation_points,
      dataset_role = "validation",
      config_id = config_id,
      device = device
    )
    
    validation_metrics_epoch <- calc_metrics(
      obs = pred_validation_epoch$obs,
      pred = pred_validation_epoch$pred
    )
    
    validation_monitor <- validation_loss
    
    if (config$checkpoint_metric == "composite_mae_ccc") {
      validation_monitor <- validation_metrics_epoch$mae / mean(pred_validation_epoch$obs, na.rm = TRUE) -
        validation_metrics_epoch$ccc
    }
    
    if (config$scheduler_type == "reduce_lr_on_plateau" && epoch > config$warmup_epochs) {
      if (validation_loss < plateau_best - config$lr_plateau_min_delta) {
        plateau_best <- validation_loss
        plateau_bad_epochs <- 0L
      } else {
        plateau_bad_epochs <- plateau_bad_epochs + 1L
      }
      
      if (plateau_bad_epochs >= config$lr_plateau_patience) {
        current_lr <- max(get_optimizer_lr(optimizer) * config$lr_plateau_factor, config$min_learning_rate)
        set_optimizer_lr(optimizer, current_lr)
        plateau_bad_epochs <- 0L
      }
    }
    
    improved <- validation_monitor < best_metric - config$early_stopping_min_delta
    
    if (improved) {
      best_metric <- validation_monitor
      best_epoch <- epoch
      best_state <- model$state_dict()
      epochs_without_improvement <- 0L
    } else {
      epochs_without_improvement <- epochs_without_improvement + 1L
    }
    
    training_history[[epoch]] <- tibble::tibble(
      config_id = config_id,
      epoch = epoch,
      train_loss = train_loss,
      validation_loss = validation_loss,
      validation_mae = validation_metrics_epoch$mae,
      validation_ccc = validation_metrics_epoch$ccc,
      validation_r2 = validation_metrics_epoch$r2,
      validation_nse = validation_metrics_epoch$nse,
      validation_rmse = validation_metrics_epoch$rmse,
      validation_mqi = validation_metrics_epoch$mqi,
      monitor_metric = validation_monitor,
      learning_rate = get_optimizer_lr(optimizer),
      best_epoch = best_epoch,
      best_metric = best_metric,
      epochs_without_improvement = epochs_without_improvement
    )
    
    if (epoch %% print_every == 0 || epoch == 1 || improved) {
      message(
        config_id,
        " | epoch ",
        epoch,
        " | train_loss = ",
        round(train_loss, 5),
        " | validation_loss = ",
        round(validation_loss, 5),
        " | validation_MAE = ",
        round(validation_metrics_epoch$mae, 4),
        " | validation_CCC = ",
        round(validation_metrics_epoch$ccc, 4),
        " | validation_RMSE = ",
        round(validation_metrics_epoch$rmse, 4),
        " | monitor = ",
        round(validation_monitor, 5),
        " | best_epoch = ",
        best_epoch
      )
    }
    
    if (epochs_without_improvement >= config$patience) {
      message(config_id, " | early stopping at epoch ", epoch)
      break
    }
  }
  
  training_history <- dplyr::bind_rows(training_history)
  
  if (!is.null(best_state)) {
    model$load_state_dict(best_state)
  }
  
  pred_train <- predict_loader(model, loaders$train_eval_loader, tensor_bundle$train_points, "train", config_id, device)
  pred_validation <- predict_loader(model, loaders$validation_loader, tensor_bundle$validation_points, "validation", config_id, device)
  pred_test <- predict_loader(model, loaders$test_loader, tensor_bundle$test_points, "test", config_id, device)
  
  pred_all <- dplyr::bind_rows(pred_train, pred_validation, pred_test)
  performance_all <- make_performance_table(pred_all)
  quantile_performance_all <- make_quantile_performance(pred_all)
  
  gate_test <- extract_gate_values(
    model = model,
    data_loader = loaders$test_loader,
    points_valid = tensor_bundle$test_points,
    dataset_role = "test",
    config_id = config_id,
    gate_type = config$gate_type,
    device = device
  )
  
  runtime <- sprintf("%.2f %s", Sys.time() - time_start, units(Sys.time() - time_start))
  stopped_epoch <- max(training_history$epoch, na.rm = TRUE)
  
  run_summary <- list(
    config = config,
    best_epoch = best_epoch,
    stopped_epoch = stopped_epoch,
    best_metric = best_metric,
    runtime = runtime,
    performance_all = performance_all,
    quantile_performance_all = quantile_performance_all,
    gate_summary_test = gate_test$gate_summary
  )
  
  safe_torch_save(best_state, file.path(model_dir, "best_state.pt"))
  safe_torch_save(model$state_dict(), file.path(model_dir, "state.pt"))
  safe_write_csv2(training_history, file.path(log_dir, "training_history.csv"))
  safe_write_csv2(pred_all, file.path(prediction_dir, "pred_obs_all.csv"))
  safe_write_csv2(pred_test, file.path(prediction_dir, "pred_obs_test.csv"))
  safe_write_csv2(performance_all, file.path(performance_dir, "performance_all.csv"))
  safe_write_csv2(quantile_performance_all, file.path(performance_dir, "quantile_performance_all.csv"))
  safe_write_csv2(gate_test$gate_summary, file.path(gate_dir, "gate_summary_test.csv"))
  if (nrow(gate_test$gate_by_profile) > 0) {
    safe_write_csv2(gate_test$gate_by_profile, file.path(gate_dir, "gate_by_profile_test.csv"))
  }
  safe_save_rds(run_summary, file.path(config_dir, "run_summary.rds"), compress = FALSE)
  
  loss_plot <- training_history %>%
    tidyr::pivot_longer(
      cols = c(train_loss, validation_loss),
      names_to = "loss_type",
      values_to = "loss"
    ) %>%
    ggplot2::ggplot(ggplot2::aes(x = epoch, y = loss, colour = loss_type)) +
    ggplot2::geom_line(linewidth = 0.8) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = "Epoch", y = "Loss", colour = "Loss", title = paste0(config_id, " loss by epoch"))
  
  save_plot(loss_plot, file.path(figure_dir, "loss_by_epoch.png"), width = 8, height = 5)
  
  test_pred_plot <- pred_test %>%
    ggplot2::ggplot(ggplot2::aes(x = obs, y = pred)) +
    ggplot2::geom_abline(linewidth = 0.8) +
    ggplot2::geom_point(alpha = 0.25, size = 0.8) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = "Observed SOC stock, ton/ha", y = "Predicted SOC stock, ton/ha", title = paste0(config_id, " test observed vs predicted"))
  
  save_plot(test_pred_plot, file.path(figure_dir, "test_pred_obs.png"), width = 6, height = 5)
  
  residual_plot <- pred_test %>%
    ggplot2::ggplot(ggplot2::aes(x = obs, y = residual)) +
    ggplot2::geom_hline(yintercept = 0, linewidth = 0.8) +
    ggplot2::geom_point(alpha = 0.25, size = 0.8) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = "Observed SOC stock, ton/ha", y = "Residual", title = paste0(config_id, " test residuals"))
  
  save_plot(residual_plot, file.path(figure_dir, "test_residuals.png"), width = 6, height = 5)
  
  if (nrow(gate_test$gate_by_profile) > 0) {
    gate_plot <- gate_test$gate_by_profile %>%
      ggplot2::ggplot(ggplot2::aes(x = target_native, y = mean_gate_profile)) +
      ggplot2::geom_point(alpha = 0.25, size = 0.8) +
      ggplot2::geom_smooth(se = FALSE, linewidth = 0.8) +
      ggplot2::theme_bw() +
      ggplot2::labs(x = "Observed SOC stock, ton/ha", y = "Mean gate", title = paste0(config_id, " gate behavior"))
    
    save_plot(gate_plot, file.path(figure_dir, "mean_gate_by_soc_stock.png"), width = 7, height = 5)
  }
  
  tibble::tibble(
    config_id = config_id,
    status = "success",
    best_epoch = best_epoch,
    stopped_epoch = stopped_epoch,
    best_metric = best_metric,
    runtime = runtime
  )
}

run_results <- list()
error_log <- list()

for (i in seq_len(nrow(tune_grid))) {
  config <- tune_grid[i, ]
  
  result <- try(
    train_one_config(
      config = config,
      tensor_bundle = tensor_bundle,
      device = device
    ),
    silent = TRUE
  )
  
  if (inherits(result, "try-error")) {
    message("Config failed: ", config$config_id)
    message(as.character(result))
    
    error_log[[config$config_id]] <- tibble::tibble(
      config_id = config$config_id,
      error_message = as.character(result)
    )
    
    if (stop_on_error) {
      stop(result)
    }
  } else {
    run_results[[config$config_id]] <- result
  }
  
  gc()
}

run_status <- dplyr::bind_rows(run_results)
error_log_tbl <- dplyr::bind_rows(error_log)

safe_write_csv2(run_status, file.path(output_comparison_table_dir, "run_status.csv"))

if (nrow(error_log_tbl) > 0) {
  safe_write_csv2(error_log_tbl, file.path(output_comparison_table_dir, "error_log.csv"))
}

performance_all_configs <- purrr::map_dfr(
  run_status$config_id,
  function(config_id) {
    readr::read_csv2(
      file.path(output_config_root_dir, config_id, "performance", "performance_all.csv"),
      show_col_types = FALSE
    )
  }
)

quantile_all_configs <- purrr::map_dfr(
  run_status$config_id,
  function(config_id) {
    readr::read_csv2(
      file.path(output_config_root_dir, config_id, "performance", "quantile_performance_all.csv"),
      show_col_types = FALSE
    )
  }
)

validation_tail_mae <- quantile_all_configs %>%
  dplyr::filter(dataset_role == "validation", obs_quantile_group %in% c("Q95_Q99", "Q99_Q100")) %>%
  dplyr::select(config_id, obs_quantile_group, mae) %>%
  tidyr::pivot_wider(
    names_from = obs_quantile_group,
    values_from = mae,
    names_prefix = "mae_"
  )

validation_ranking <- performance_all_configs %>%
  dplyr::filter(dataset_role == "validation") %>%
  dplyr::left_join(validation_tail_mae, by = "config_id") %>%
  dplyr::mutate(
    global_balance_score = rank(mae, ties.method = "min") +
      rank(-ccc, ties.method = "min") +
      rank(-r2, ties.method = "min") +
      rank(-nse, ties.method = "min") +
      rank(rmse, ties.method = "min") +
      rank(-mqi, ties.method = "min"),
    tail_balance_score = rank(mae_Q95_Q99, ties.method = "min") +
      rank(mae_Q99_Q100, ties.method = "min"),
    final_selection_score = global_balance_score + tail_balance_score
  ) %>%
  dplyr::left_join(tune_grid, by = "config_id") %>%
  dplyr::arrange(final_selection_score, global_balance_score, tail_balance_score, mae, dplyr::desc(ccc))

best_config <- validation_ranking %>%
  dplyr::slice(1)

test_tail_mae <- quantile_all_configs %>%
  dplyr::filter(dataset_role == "test", obs_quantile_group %in% c("Q95_Q99", "Q99_Q100")) %>%
  dplyr::select(config_id, obs_quantile_group, mae) %>%
  tidyr::pivot_wider(
    names_from = obs_quantile_group,
    values_from = mae,
    names_prefix = "mae_"
  )

test_ranking <- performance_all_configs %>%
  dplyr::filter(dataset_role == "test") %>%
  dplyr::left_join(test_tail_mae, by = "config_id") %>%
  dplyr::left_join(tune_grid, by = "config_id") %>%
  dplyr::arrange(mae, dplyr::desc(ccc))

safe_write_csv2(performance_all_configs, file.path(output_comparison_table_dir, "performance_all_configs.csv"))
safe_write_csv2(quantile_all_configs, file.path(output_comparison_table_dir, "quantile_performance_all_configs.csv"))
safe_write_csv2(validation_ranking, file.path(output_decision_dir, "validation_ranking.csv"))
safe_write_csv2(test_ranking, file.path(output_decision_dir, "test_ranking.csv"))
safe_write_csv2(best_config, file.path(output_decision_dir, "best_config.csv"))

ranking_plot <- validation_ranking %>%
  dplyr::mutate(config_id = factor(config_id, levels = rev(config_id))) %>%
  ggplot2::ggplot(ggplot2::aes(x = final_selection_score, y = config_id)) +
  ggplot2::geom_col(width = 0.7) +
  ggplot2::theme_bw() +
  ggplot2::labs(x = "Final selection score, lower is better", y = "Config", title = "CNN tuning ranking")

save_plot(
  ranking_plot,
  file.path(output_comparison_figure_dir, "validation_ranking.png"),
  width = 8,
  height = 5
)

saveRDS(
  object = list(
    tuning_run_id = tuning_run_id,
    tune_grid = tune_grid,
    run_status = run_status,
    error_log = error_log_tbl,
    validation_ranking = validation_ranking,
    test_ranking = test_ranking,
    best_config = best_config
  ),
  file = file.path(output_root_dir, "tuning_run_summary.rds"),
  compress = FALSE
)

message("CNN tuning finished.")
message("Output saved to: ", output_root_dir)
message("Best config: ", best_config$config_id[1])
