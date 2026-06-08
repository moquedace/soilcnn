# Training engine — single config + full tuning loop
#
# Main entry points
# -----------------
# train_one_cnn()  – train a single CNN configuration, return results
# run_cnn_tuning() – iterate over a tune_grid, rank configs, save everything

# ── Prediction helper ─────────────────────────────────────────────────────────

#' Run inference over a DataLoader and return a pred/obs tibble.
#'
#' @param model        Trained dual_branch_cnn.
#' @param data_loader  DataLoader whose batches are (x1 [, x2], y).
#' @param points_valid Tibble with at least: profile_id, sample_id,
#'   target_native, target_transform (transformed scale used during training).
#' @param dataset_role Character label: "train", "validation", or "test".
#' @param transform    Inverse transform function applied to predictions.
#'   Default: identity (no transform). For log1p training use expm1.
#' @param device       torch_device.
predict_loader <- function(model, data_loader, points_valid, dataset_role,
                           transform = identity, device) {
  model$eval()
  pred_raw <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      inputs <- lapply(batch[-length(batch)], function(t) t$to(device = device))
      out    <- do.call(model, inputs)
      pred_raw <- c(pred_raw, as.numeric(out$to(device = "cpu")))
    })
  })
  pred_native <- pmax(transform(pred_raw), 0)
  obs_native  <- as.numeric(points_valid$target_native)
  if (length(pred_native) != nrow(points_valid)) {
    stop("Prediction length != metadata rows for split: ", dataset_role)
  }
  tibble::tibble(
    profile_id       = points_valid$profile_id,
    sample_id        = points_valid$sample_id,
    dataset_role     = dataset_role,
    obs              = obs_native,
    pred             = pred_native,
    obs_transform    = as.numeric(points_valid$target_transform),
    pred_transform   = pred_raw,
    residual         = pred_native - obs_native,
    abs_error        = abs(pred_native - obs_native),
    residual_transform = pred_raw - as.numeric(points_valid$target_transform),
    abs_error_transform = abs(pred_raw - as.numeric(points_valid$target_transform))
  )
}

# ── Loss in transform-space (CPU, from collected predictions) ─────────────────

#' Compute the training loss directly from already-collected predictions.
#'
#' Reproduces the torch loss functions in transform (e.g. log1p) space so the
#' per-epoch validation loss can be derived from the single forward pass that
#' predict_loader() already performs — avoiding a second GPU pass over the
#' validation set every epoch. Numerically identical to averaging the torch
#' loss over the loader (mean reduction, SmoothL1 beta = 1.0).
#'
#' @param pred_t Predicted values in transform space (model raw output).
#' @param obs_t  Observed values in transform space.
#' @param loss_fn_name One of "smooth_l1", "mse", "mae".
transform_space_loss <- function(pred_t, obs_t, loss_fn_name) {
  d <- as.numeric(pred_t) - as.numeric(obs_t)
  switch(loss_fn_name,
    smooth_l1 = mean(ifelse(abs(d) < 1, 0.5 * d^2, abs(d) - 0.5)),  # beta = 1.0
    mse       = mean(d^2),
    mae       = mean(abs(d)),
    stop("Unknown loss_fn: ", loss_fn_name)
  )
}

# ── Gate analysis helper ──────────────────────────────────────────────────────

#' Extract gate values and branch norms for interpretability.
extract_gate_analysis <- function(model, data_loader, points_valid,
                                  dataset_role, device) {
  if (model$n_branches < 2L || model$gate_type == "no_gate_concat") {
    return(NULL)
  }
  model$eval()
  gate_rows   <- NULL
  mean_gate   <- numeric(0)
  norm1_all   <- numeric(0)
  norm2_all   <- numeric(0)
  cos_sim_all <- numeric(0)

  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      inputs <- lapply(batch[-length(batch)], function(t) t$to(device = device))
      out    <- do.call(model$forward_with_internals, inputs)
      gate   <- as.array(out$gate$to(device = "cpu"))
      f1     <- as.array(out$f1$to(device = "cpu"))
      f2     <- as.array(out$f2$to(device = "cpu"))
      if (is.null(dim(gate))) gate <- matrix(gate, nrow = 1L)
      if (is.null(dim(f1)))   f1   <- matrix(f1,   nrow = 1L)
      if (is.null(dim(f2)))   f2   <- matrix(f2,   nrow = 1L)
      n1   <- sqrt(rowSums(f1^2))
      n2   <- sqrt(rowSums(f2^2))
      coss <- rowSums(f1 * f2) / (n1 * n2 + 1e-8)
      gate_rows   <- if (is.null(gate_rows)) gate else rbind(gate_rows, gate)
      mean_gate   <- c(mean_gate, rowMeans(gate))
      norm1_all   <- c(norm1_all, n1)
      norm2_all   <- c(norm2_all, n2)
      cos_sim_all <- c(cos_sim_all, coss)
    })
  })

  gate_by_profile <- points_valid %>%
    dplyr::select(profile_id, sample_id, target_native) %>%
    dplyr::mutate(
      dataset_role      = dataset_role,
      mean_gate         = mean_gate,
      norm_branch1      = norm1_all,
      norm_branch2      = norm2_all,
      norm_ratio_1_2    = norm1_all / (norm2_all + 1e-8),
      cosine_similarity = cos_sim_all
    )

  gv <- as.numeric(gate_rows)
  gate_summary <- tibble::tibble(
    dataset_role            = dataset_role,
    n                       = nrow(gate_by_profile),
    n_gate_dims             = ncol(gate_rows),
    mean_gate               = mean(gv, na.rm = TRUE),
    median_gate             = median(gv, na.rm = TRUE),
    q05_gate                = as.numeric(stats::quantile(gv, 0.05, na.rm = TRUE)),
    q95_gate                = as.numeric(stats::quantile(gv, 0.95, na.rm = TRUE)),
    mean_norm_branch1       = mean(norm1_all, na.rm = TRUE),
    mean_norm_branch2       = mean(norm2_all, na.rm = TRUE),
    median_norm_ratio       = median(norm1_all / (norm2_all + 1e-8), na.rm = TRUE),
    mean_cosine_similarity  = mean(cos_sim_all, na.rm = TRUE)
  )

  list(summary = gate_summary, by_profile = gate_by_profile)
}

# ── Single-config training ────────────────────────────────────────────────────

#' Train one CNN configuration from a tune_grid row.
#'
#' @param cfg            One-row tibble from make_tune_grid().
#' @param n_channels     Number of predictor channels.
#' @param loaders        Named list: train, train_eval, validation, test.
#' @param points_valid   Named list: train, validation, test – metadata tibbles.
#' @param transform      Inverse of the target transformation (e.g., expm1).
#' @param device         torch_device.
#' @param n_epochs       Maximum number of training epochs.
#' @param patience       Early stopping patience (epochs without improvement).
#' @param es_min_delta   Minimum improvement to reset early stopping counter.
#' @param warmup_start_lr  Initial LR at epoch 1 (before warmup).
#' @param lr_plateau_factor  LR reduction factor on plateau.
#' @param lr_plateau_patience  Epochs to wait before reducing LR.
#' @param lr_plateau_min_delta  Min delta for plateau detection.
#' @param min_lr         Minimum LR (floor for plateau reduction).
#' @param gradient_clip  Max gradient norm (0 = disabled).
#' @param print_every    Print progress every N epochs.
#' @param model_name     Label written into the pred/obs output.
#' @param augment        Apply D4 (rotation/flip) augmentation during training.
#'   Patches are rotation/mirror invariant for a centre-point target, so this
#'   is a label-preserving regulariser. Applied to training batches only.
#'
#' @return A list with: history, pred_all, perf_all, perf_quantile,
#'   gate, best_epoch, runtime, config.
train_one_cnn <- function(
  cfg,
  n_channels,
  loaders,
  points_valid,
  transform          = identity,
  device,
  n_epochs           = 700L,
  patience           = 90L,
  es_min_delta       = 0.0005,
  warmup_start_lr    = 1e-5,
  lr_plateau_factor  = 0.5,
  lr_plateau_patience = 20L,
  lr_plateau_min_delta = 0.0005,
  min_lr             = 1e-6,
  gradient_clip      = 1.0,
  print_every        = 5L,
  model_name         = "cnn",
  augment            = TRUE
) {
  base_lr       <- cfg$base_lr
  batch_size    <- cfg$batch_size
  warmup_epochs <- cfg$warmup_epochs

  model <- build_cnn_from_config(cfg, n_channels)$to(device = device)

  loss_fn <- switch(cfg$loss_fn,
    smooth_l1 = torch::nn_smooth_l1_loss(),
    mse       = torch::nn_mse_loss(),
    mae       = torch::nn_l1_loss(),
    stop("Unknown loss_fn: ", cfg$loss_fn)
  )

  optimizer   <- torch::optim_adam(model$parameters, lr = warmup_start_lr,
                                   weight_decay = cfg$weight_decay)
  current_lr  <- warmup_start_lr
  best_metric <- Inf
  best_val_loss <- Inf
  best_epoch  <- NA_integer_
  no_improve  <- 0L
  plateau_wait <- 0L
  plateau_best <- Inf
  best_state  <- NULL

  history <- tibble::tibble(
    epoch = integer(), train_loss = double(), validation_loss = double(),
    validation_ccc = double(), validation_r2 = double(),
    validation_mae = double(), validation_nse = double(),
    validation_rmse = double(), validation_mqi = double(),
    monitor_metric = double(), best_epoch = integer(),
    current_lr = double(), no_improve = integer()
  )

  t0 <- Sys.time()

  for (epoch in seq_len(n_epochs)) {
    # --- LR warmup ---
    if (epoch <= warmup_epochs) {
      current_lr <- warmup_start_lr +
        (base_lr - warmup_start_lr) * epoch / warmup_epochs
      set_optimizer_lr(optimizer, current_lr)
    }

    # --- Training pass ---
    model$train()
    tr_loss_sum <- 0; tr_n <- 0L
    coro::loop(for (batch in loaders$train) {
      inputs <- lapply(batch[-length(batch)], function(t) t$to(device = device))
      y      <- batch[[length(batch)]]$to(device = device)
      if (augment) inputs <- augment_d4_batch(inputs)
      optimizer$zero_grad()
      pred   <- do.call(model, inputs)
      loss   <- loss_fn(pred, y)
      loss$backward()
      if (gradient_clip > 0 &&
          "nn_utils_clip_grad_norm_" %in% getNamespaceExports("torch")) {
        torch::nn_utils_clip_grad_norm_(model$parameters, max_norm = gradient_clip)
      }
      optimizer$step()
      bn <- as.integer(inputs[[1]]$shape[[1]])
      tr_loss_sum <- tr_loss_sum + as.numeric(loss$item()) * bn
      tr_n <- tr_n + bn
    })
    tr_loss <- tr_loss_sum / tr_n

    # --- Validation: one forward pass, then loss + metrics derived from it ---
    # predict_loader() already iterates the whole validation loader; the loss is
    # computed in transform space from those predictions, so there is no second
    # GPU pass (compute_loader_loss is not called per epoch).
    pred_val <- predict_loader(model, loaders$validation,
                               points_valid$validation, "validation",
                               transform, device)
    val_loss <- transform_space_loss(pred_val$pred_transform,
                                     pred_val$obs_transform, cfg$loss_fn)
    perf_val <- make_performance_table(
      dplyr::mutate(pred_val, model = model_name, target_version = cfg$loss_fn)
    )

    monitor <- val_loss   # early stopping based on SmoothL1 validation loss

    if (val_loss < best_val_loss) best_val_loss <- val_loss

    if (monitor < best_metric - es_min_delta) {
      best_metric <- monitor
      best_epoch  <- epoch
      no_improve  <- 0L
      best_state  <- clone_state_dict(model$state_dict())
    } else {
      no_improve <- no_improve + 1L
    }

    # --- LR plateau ---
    if (epoch > warmup_epochs) {
      if (monitor < plateau_best - lr_plateau_min_delta) {
        plateau_best <- monitor
        plateau_wait <- 0L
      } else {
        plateau_wait <- plateau_wait + 1L
      }
      if (plateau_wait >= lr_plateau_patience && current_lr > min_lr) {
        current_lr <- max(current_lr * lr_plateau_factor, min_lr)
        set_optimizer_lr(optimizer, current_lr)
        plateau_wait <- 0L
        message("  LR reduced to ", signif(current_lr, 4), " at epoch ", epoch)
      }
    }

    history <- dplyr::bind_rows(history, tibble::tibble(
      epoch          = epoch,
      train_loss     = tr_loss,
      validation_loss = val_loss,
      validation_ccc  = perf_val$ccc[1],
      validation_r2   = perf_val$r2[1],
      validation_mae  = perf_val$mae[1],
      validation_nse  = perf_val$nse[1],
      validation_rmse = perf_val$rmse[1],
      validation_mqi  = perf_val$mqi[1],
      monitor_metric  = monitor,
      best_epoch      = best_epoch,
      current_lr      = current_lr,
      no_improve      = no_improve
    ))

    if (epoch %% print_every == 0L || epoch == 1L) {
      message(sprintf(
        "  epoch %d | lr %.2e | tr %.5f | val %.5f | MAE %.3f | CCC %.3f | best %d",
        epoch, current_lr, tr_loss, val_loss,
        perf_val$mae[1], perf_val$ccc[1], best_epoch
      ))
    }

    if (no_improve >= patience) {
      message("  Early stopping at epoch ", epoch,
              " (best: ", best_epoch, ")")
      break
    }
    gc()
  }

  runtime <- difftime(Sys.time(), t0, units = "mins")
  if (is.null(best_state)) stop("No best_state saved — training may have failed.")
  model$load_state_dict(best_state)

  # --- Final evaluation on all splits ---
  pred_train <- predict_loader(model, loaders$train_eval,
                               points_valid$train, "train", transform, device)
  pred_val2  <- predict_loader(model, loaders$validation,
                               points_valid$validation, "validation", transform, device)
  pred_test  <- predict_loader(model, loaders$test,
                               points_valid$test, "test", transform, device)

  pred_all <- dplyr::mutate(
    dplyr::bind_rows(pred_train, pred_val2, pred_test),
    model = model_name, target_version = cfg$loss_fn
  )

  perf_all      <- make_performance_table(pred_all)
  perf_quantile <- make_quantile_performance(pred_all)

  gate <- extract_gate_analysis(model, loaders$test, points_valid$test,
                                 "test", device)

  list(
    model         = model,
    best_state    = best_state,
    history       = history,
    pred_all      = pred_all,
    perf_all      = perf_all,
    perf_quantile = perf_quantile,
    gate          = gate,
    best_epoch    = best_epoch,
    best_val_loss = best_val_loss,
    runtime_min   = as.numeric(runtime),
    config        = cfg
  )
}

# ── Full tuning loop ──────────────────────────────────────────────────────────

#' Run all configurations in a tune_grid and save results.
#'
#' Mirrors caret's train() but for the dual-branch CNN.
#' For each config the function:
#'   1. Builds and trains the model.
#'   2. Saves weights, history, predictions, metrics.
#'   3. Appends a row to the comparison table.
#'   4. Ranks configs by validation CCC (then val MAE). Test metrics are written
#'      for diagnostic reference but are NOT used for selection.
#'
#' @param tune_grid    tibble from make_tune_grid() or make_manual_tune_grid().
#' @param n_channels   Number of predictor channels.
#' @param patches      Named list: train, validation, test – each a list with
#'   x_arrays (one per window size) and y (target, transformed scale).
#' @param points_valid Named list: train, validation, test – metadata tibbles.
#' @param transform    Inverse transform for predictions (default: identity).
#' @param output_dir   Root output directory.
#' @param device       torch_device.
#' @param run_id       String label for this tuning run.
#' @param base_seed    Base RNG seed. Config i is trained after setting the seed
#'   to base_seed + i (both R and torch), so each config has a reproducible
#'   weight initialisation independent of the configs run before it.
#' @param ...          Passed to train_one_cnn() (n_epochs, patience, etc.).
run_cnn_tuning <- function(
  tune_grid,
  n_channels,
  patches,
  points_valid,
  transform   = identity,
  output_dir  = "./outputs/tuning",
  device,
  run_id      = format(Sys.time(), "%Y%m%d_%H%M%S"),
  base_seed   = 42L,
  ...
) {
  run_dir <- file.path(output_dir, run_id)
  dirs    <- file.path(run_dir, c("models", "history", "predictions",
                                   "metrics", "gates", "comparison"))
  create_output_dirs(dirs)

  # Save the grid so the run can be reproduced
  safe_save_rds(tune_grid, file.path(run_dir, "tune_grid.rds"), compress = FALSE)
  safe_write_csv2(
    dplyr::mutate(tune_grid,
      window_sizes  = purrr::map_chr(window_sizes, paste, collapse = "_"),
      conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")
    ),
    file.path(run_dir, "tune_grid.csv")
  )

  # Build the tensor cache once for every window size used anywhere in the grid,
  # then reuse it across all configs (avoids recreating large tensors per row).
  windows_needed <- sort(unique(unlist(tune_grid$window_sizes)))
  message("Building tensor cache for windows: ",
          paste(windows_needed, collapse = ", "))
  tensor_cache <- .build_tensor_cache(patches, windows_needed)

  comparison <- tibble::tibble()
  n_cfg      <- nrow(tune_grid)

  for (i in seq_len(n_cfg)) {
    cfg <- tune_grid[i, ]

    # Per-config reproducible seed (init independent of previously run configs)
    set.seed(base_seed + i)
    torch::torch_manual_seed(base_seed + i)

    message("\n── Config ", i, "/", n_cfg, ": ", cfg$config_id, " ──")
    message("  window_sizes : ", paste(cfg$window_sizes[[1]], collapse = "x"))
    message("  conv_channels: ", paste(cfg$conv_channels[[1]], collapse = ", "))
    message("  embedding_dim: ", cfg$embedding_dim,
            " | gate: ", cfg$gate_type,
            " | residual: ", cfg$use_residual,
            " | se: ", cfg$use_se_block)
    message("  base_lr: ", cfg$base_lr,
            " | batch: ", cfg$batch_size,
            " | loss: ", cfg$loss_fn)

    # Build DataLoaders for this config's window sizes from the shared cache
    loaders <- .make_loaders_from_cache(tensor_cache, cfg)

    result <- tryCatch(
      train_one_cnn(
        cfg          = cfg,
        n_channels   = n_channels,
        loaders      = loaders,
        points_valid = points_valid,
        transform    = transform,
        device       = device,
        model_name   = cfg$config_id,
        ...
      ),
      error = function(e) {
        message("  ERROR in config ", cfg$config_id, ": ", e$message)
        NULL
      }
    )
    if (is.null(result)) next

    # Save outputs
    cid <- cfg$config_id
    safe_torch_save(result$best_state, file.path(run_dir, "models",
                    paste0(cid, "_best.pt")))
    safe_write_csv2(result$history,
                    file.path(run_dir, "history", paste0(cid, "_history.csv")))
    safe_write_csv2(result$pred_all,
                    file.path(run_dir, "predictions", paste0(cid, "_pred_all.csv")))
    safe_write_csv2(result$perf_all,
                    file.path(run_dir, "metrics", paste0(cid, "_perf.csv")))
    safe_write_csv2(result$perf_quantile,
                    file.path(run_dir, "metrics", paste0(cid, "_perf_quantile.csv")))
    if (!is.null(result$gate)) {
      safe_write_csv2(result$gate$summary,
                      file.path(run_dir, "gates", paste0(cid, "_gate_summary.csv")))
      safe_write_csv2(result$gate$by_profile,
                      file.path(run_dir, "gates", paste0(cid, "_gate_profiles.csv")))
    }

    # Append to comparison table
    val_perf  <- dplyr::filter(result$perf_all, dataset_role == "validation")
    test_perf <- dplyr::filter(result$perf_all, dataset_role == "test")
    row <- dplyr::bind_cols(
      tibble::tibble(
        config_id      = cid,
        best_epoch     = result$best_epoch,
        runtime_min    = round(result$runtime_min, 2),
        best_val_loss  = round(result$best_val_loss, 6),
        window_sizes   = paste(cfg$window_sizes[[1]], collapse = "x"),
        conv_channels  = paste(cfg$conv_channels[[1]], collapse = "_"),
        status         = "success"
      ),
      dplyr::select(cfg, -config_id, -window_sizes, -conv_channels),
      dplyr::rename_with(val_perf,  ~ paste0("val_",  .x),
                         .cols = c(ccc, r2, mae, nse, rmse, rpd, mqi)),
      dplyr::rename_with(test_perf, ~ paste0("test_", .x),
                         .cols = c(ccc, r2, mae, nse, rmse, rpd, mqi))
    )
    comparison <- dplyr::bind_rows(comparison, row)
    safe_write_csv2(comparison,
                    file.path(run_dir, "comparison", "comparison_all.csv"))

    gc()
  }

  # Rank by VALIDATION metrics only — test set is read-only diagnostic
  if (nrow(comparison) > 0) {
    comparison <- comparison %>%
      dplyr::arrange(dplyr::desc(val_ccc), val_mae) %>%
      dplyr::mutate(rank = dplyr::row_number())
    safe_write_csv2(comparison,
                    file.path(run_dir, "comparison", "comparison_ranked.csv"))
    message("\n── Best config: ", comparison$config_id[1],
            " | val_CCC=", round(comparison$val_ccc[1], 3),
            " | val_MAE=", round(comparison$val_mae[1], 3),
            " | test_CCC=", round(comparison$test_ccc[1], 3), " (diagnostic only) ──")
  }

  invisible(list(comparison = comparison, run_dir = run_dir))
}

# ── DataLoader builders (internal) ────────────────────────────────────────────

#' Build float tensors once per (split, window) and per split target.
#'
#' The patch arrays are large (the 7×7 train array is ~1.6 GB). Converting them
#' to tensors once and reusing across all configs avoids recreating the same
#' tensors on every grid row. Tensors live on CPU; batches are moved to the
#' device inside the training/eval loops.
#'
#' @param patches      Named list: train, validation, test.
#' @param window_sizes Integer vector of window sizes to cache (union over grid).
#' @return Nested list: cache[[split]][["x_WxW_array"]] and cache[[split]]$y.
.build_tensor_cache <- function(patches, window_sizes) {
  splits <- c("train", "validation", "test")
  cache  <- vector("list", length(splits))
  names(cache) <- splits
  for (split in splits) {
    cache[[split]] <- list()
    for (w in window_sizes) {
      key <- paste0("x_", w, "x", w, "_array")
      cache[[split]][[key]] <- torch::torch_tensor(
        patches[[split]][[key]], dtype = torch::torch_float()
      )
    }
    cache[[split]]$y <- torch::torch_tensor(
      as.numeric(patches[[split]]$y), dtype = torch::torch_float()
    )$view(c(-1L, 1L))
  }
  cache
}

#' Build the four DataLoaders for one config from a prebuilt tensor cache.
#'
#' tensor_dataset only references the cached tensors (no copy); dataloaders are
#' cheap to (re)create per config, so only batch_size-dependent objects are
#' rebuilt here.
.make_loaders_from_cache <- function(cache, cfg) {
  ws       <- cfg$window_sizes[[1]]
  bs_train <- cfg$batch_size
  bs_eval  <- min(bs_train * 4L, 2048L)

  make_ds <- function(split) {
    arrays <- lapply(ws, function(w) cache[[split]][[paste0("x_", w, "x", w, "_array")]])
    do.call(torch::tensor_dataset, c(arrays, list(cache[[split]]$y)))
  }

  train_ds <- make_ds("train")
  val_ds   <- make_ds("validation")
  test_ds  <- make_ds("test")

  # drop_last = TRUE on the training loader: prevents a final batch of size 1,
  # which would make BatchNorm fail (variance of a single sample). Eval loaders
  # keep all samples (no BatchNorm update in eval mode).
  list(
    train      = torch::dataloader(train_ds, batch_size = bs_train, shuffle = TRUE, drop_last = TRUE),
    train_eval = torch::dataloader(train_ds, batch_size = bs_eval,  shuffle = FALSE),
    validation = torch::dataloader(val_ds,   batch_size = bs_eval,  shuffle = FALSE),
    test       = torch::dataloader(test_ds,  batch_size = bs_eval,  shuffle = FALSE)
  )
}

#' Convenience wrapper: build loaders for a single config directly from patches.
#' Used by the final-model script (single config). Caches only the windows that
#' config needs. `device` is kept for backward compatibility (tensors are CPU).
.make_loaders <- function(patches, cfg, device = NULL) {
  cache <- .build_tensor_cache(patches, cfg$window_sizes[[1]])
  .make_loaders_from_cache(cache, cfg)
}
