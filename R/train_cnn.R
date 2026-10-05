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
#' @param data_loader  DataLoader whose batches are `(x1, y)`, or `(x1, x2, y)`
#'   for two branches.
#' @param points_valid Tibble with at least: profile_id, sample_id,
#'   target_native, target_transform (transformed scale used during training).
#' @param dataset_role Character label: "train", "validation", or "test".
#' @param transform    Inverse transform function applied to predictions.
#'   Default: identity (no transform). For log1p training use expm1.
#' @param device       torch_device.
#' @param clamp        Length-2 numeric, the plausible range of the target in
#'   NATIVE units, applied after `transform`. Default c(0, Inf) suits a stock
#'   or concentration, which cannot be negative. Pass c(-Inf, Inf) for a target
#'   that legitimately goes negative (a centred variable, a log-ratio, a
#'   temperature) -- clamping one of those at zero destroys half the
#'   predictions silently, and the metrics still come out looking plausible.
#' @noRd
predict_loader <- function(model, data_loader, points_valid, dataset_role,
                           transform = identity, device, clamp = c(0, Inf)) {
  if (length(clamp) != 2L || anyNA(clamp) || clamp[1] > clamp[2]) {
    stop("clamp must be c(lower, upper) with lower <= upper, no NA.",
         call. = FALSE)
  }
  check_point_contract(points_valid, what = "points_valid")
  model$eval()
  pred_raw <- numeric(0)
  torch::with_no_grad({
    coro::loop(for (batch in data_loader) {
      inputs <- lapply(batch[-length(batch)], function(t) t$to(device = device))
      out    <- do.call(model, inputs)
      pred_raw <- c(pred_raw, as.numeric(out$to(device = "cpu")))
    })
  })
  pred_native <- transform(pred_raw)
  if (is.finite(clamp[1])) pred_native <- pmax(pred_native, clamp[1])
  if (is.finite(clamp[2])) pred_native <- pmin(pred_native, clamp[2])
  obs_native  <- as.numeric(points_valid$target_native)
  if (length(pred_native) != nrow(points_valid)) {
    stop("Role '", dataset_role, "': the loader produced ", length(pred_native),
         " prediction(s) but points_valid has ", nrow(points_valid), " row(s). ",
         "Both must be sliced with the same fold index -- see fold_points_valid().",
         call. = FALSE)
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
#' @noRd
transform_space_loss <- function(pred_t, obs_t, loss_fn_name) {
  d <- as.numeric(pred_t) - as.numeric(obs_t)
  switch(loss_fn_name,
    smooth_l1 = mean(ifelse(abs(d) < 1, 0.5 * d^2, abs(d) - 0.5)),  # beta = 1.0
    mse       = mean(d^2),
    mae       = mean(abs(d)),
    stop("loss_fn must be one of smooth_l1, mse, mae -- got '", loss_fn_name,
         "'.", call. = FALSE)
  )
}

# ── Gate analysis helper ──────────────────────────────────────────────────────

#' Extract gate values and branch norms for interpretability.
#' @noRd
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
    median_gate             = stats::median(gv, na.rm = TRUE),
    q05_gate                = as.numeric(stats::quantile(gv, 0.05, na.rm = TRUE)),
    q95_gate                = as.numeric(stats::quantile(gv, 0.95, na.rm = TRUE)),
    mean_norm_branch1       = mean(norm1_all, na.rm = TRUE),
    mean_norm_branch2       = mean(norm2_all, na.rm = TRUE),
    median_norm_ratio       = stats::median(norm1_all / (norm2_all + 1e-8), na.rm = TRUE),
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
#' @param clamp          Plausible range of the target in native units, passed
#'   to predict_loader(). See there: c(0, Inf) by default.
#' @param on_epoch       NULL, or a function(epoch) called at the end of every
#'   epoch, after its collection. It must touch neither the model nor the RNG:
#'   it exists for the final worker's memory trace (options(dsm.final.trace_mem
#'   = TRUE)), which reads memory and nothing else.
#'
#' @return A list with: history, pred_all, perf_all, perf_quantile,
#'   gate, best_epoch, runtime, config. The trained model is NOT returned --
#'   only `best_state` is, so the caller cannot accidentally keep a whole
#'   model alive across grid iterations.
#' @noRd
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
  augment            = TRUE,
  clamp              = c(0, Inf),
  on_epoch           = NULL
) {
  base_lr       <- cfg$base_lr
  batch_size    <- cfg$batch_size
  warmup_epochs <- cfg$warmup_epochs

  model <- build_cnn_from_config(cfg, n_channels)
  model <- model$to(device = device)
  gc()  # avoids an R GC / CUDA async-copy race

  loss_fn <- switch(cfg$loss_fn,
    smooth_l1 = torch::nn_smooth_l1_loss(),
    mse       = torch::nn_mse_loss(),
    mae       = torch::nn_l1_loss(),
    stop("loss_fn must be one of smooth_l1, mse, mae -- got '", cfg$loss_fn,
         "' in config ", cfg$config_id, ".", call. = FALSE)
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
      # augment_d4_batch uses R-style indexing (out[idx,,,] <-), which CUDA
      # tensors do not support -- apply on CPU, then move to the device
      inputs <- lapply(batch[-length(batch)], function(t) t)
      if (augment) inputs <- augment_d4_batch(inputs)
      inputs <- lapply(inputs, function(t) t$to(device = device))
      y      <- batch[[length(batch)]]$to(device = device)
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
                               transform, device, clamp)
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
      if (!is.null(on_epoch)) on_epoch(epoch)
      break
    }
    gc()
    if (!is.null(on_epoch)) on_epoch(epoch)
  }

  runtime <- Sys.time() - t0
  if (is.null(best_state)) {
    # Reached only when no epoch improved on Inf: every validation loss was
    # NA or non-finite. The two causes seen here are NA in target_transform on
    # this fold's validation rows and a learning rate that diverged at once.
    stop("Unit ", model_name, ": no epoch improved the validation loss, so there ",
         "is no model to keep -- every validation loss was NA or non-finite.",
         "\n  Check target_transform for NA on this fold's validation rows, or ",
         "lower base_lr (", base_lr, ").", call. = FALSE)
  }
  model$load_state_dict(best_state)

  # --- Final evaluation on all splits ---
  pred_train <- predict_loader(model, loaders$train_eval,
                               points_valid$train, "train", transform, device,
                               clamp)
  pred_val2  <- predict_loader(model, loaders$validation,
                               points_valid$validation, "validation", transform,
                               device, clamp)
  pred_test <- if (!is.null(loaders$test)) {
    predict_loader(model, loaders$test, points_valid$test, "test", transform,
                   device, clamp)
  } else {
    NULL
  }
  # The calibration set (refit_split()): predicted by the final model, never
  # trained on, never watched for stopping -- its residuals calibrate the
  # "split" interval. Only a final refit's index carries it.
  pred_cal <- if (!is.null(loaders$calibration)) {
    predict_loader(model, loaders$calibration, points_valid$calibration, "calibration",
                   transform, device, clamp)
  } else {
    NULL
  }

  pred_all <- dplyr::mutate(
    dplyr::bind_rows(pred_train, pred_val2, pred_test, pred_cal),
    model = model_name, target_version = cfg$loss_fn
  )

  perf_all      <- make_performance_table(pred_all)
  perf_quantile <- make_quantile_performance(pred_all)

  # Gate analysis on the test set when there is one, otherwise on validation:
  # it is an interpretability readout, not a metric, so it is better computed
  # on held-out data than skipped.
  gate_role <- if (!is.null(loaders$test)) "test" else "validation"
  gate <- extract_gate_analysis(model, loaders[[gate_role]],
                                points_valid[[gate_role]], gate_role, device)

  # `model` is deliberately NOT returned. Nothing consumed it, and because the
  # caller only overwrites `result` on the next iteration, returning it kept a
  # whole trained model alive through the following config's training.
  list(
    best_state    = best_state,
    history       = history,
    pred_all      = pred_all,
    perf_all      = perf_all,
    perf_quantile = perf_quantile,
    gate          = gate,
    best_epoch    = best_epoch,
    best_val_loss = best_val_loss,
    runtime_min   = as.numeric(runtime, units = "mins"),
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
#' @param cache        Scaled, split tensors from build_fold_cache()$cache:
#'   `cache[[role]][[window_key]]` plus `cache[[role]]$y`. Built by the CALLER,
#'   because the scaling is fold-dependent – the grid loop must not own
#'   that decision. When resampling arrives this argument simply becomes the
#'   current fold's cache, with no change to the loop below.
#' @param points_valid Named list of metadata tibbles, one per role, sliced
#'   with the SAME index as the tensors (see fold_points_valid()). predict_loader()
#'   matches loader row i to metadata row i, so a mismatch here silently
#'   pairs the wrong observation with the wrong prediction.
#' @param transform    Inverse transform for predictions (default: identity).
#' @param output_dir   Root output directory.
#' @param device       torch_device.
#' @param run_id       String label for this tuning run. Reuse the SAME run_id
#'   across restarts to resume — a new (timestamped) run_id always starts fresh.
#' @param base_seed    Base RNG seed. Repetition s of EVERY config is trained
#'   under the same seed, base_seed + s - 1 (both R and torch).
#'
#'   THE SEED IS SHARED ACROSS CONFIGS, ON PURPOSE. It used to be
#'   base_seed + i, one per config, which meant two configs differed both in
#'   their hyperparameters AND in the random draw that initialised them -- so
#'   part of every comparison was luck, and there was no way to tell how much.
#'   With the seed tied to the repetition instead, configs within a repetition
#'   start from the same draw, and the spread ACROSS repetitions measures the
#'   luck directly.
#'
#'   Reproducibility is unchanged: a given (config, fold, seed) trains
#'   identically whether it is reached fresh or on resume.
#' @param n_seeds      Repetitions per config, each with its own seed.
#'   Raising it later is resumable: the repetitions already on disk are
#'   recognised and only the new ones train. It is
#'   what turns "config A beat config B" into a claim with an error bar: with
#'   one seed each, a gap smaller than the seed-to-seed spread is
#'   indistinguishable from noise, and picking the winner is picking the
#'   luckiest draw.
#' @param fold         Which fold of the resampling plan this call is training.
#'   Recorded in every row and in the checkpoint name; run_cnn_resample() sets
#'   it. Left at 1 for a single holdout.
#' @param resume       If TRUE (default), a config is skipped when its model
#'   checkpoint (`models/{config_id}_best.pt`) already exists in `run_dir` —
#'   the checkpoint is only written after train_one_cnn() returns successfully,
#'   so a config that crashed mid-training (power loss, OOM, etc.) has no
#'   checkpoint and is correctly retrained, never silently treated as done.
#'   Existing rows are reloaded from `comparison/comparison_all.csv` so the
#'   final ranking still includes configs completed in earlier runs.
#'   Set FALSE to force retraining every config (e.g. after changing code that
#'   affects already-trained configs).
#' @param ...          Passed to train_one_cnn() (n_epochs, patience, etc.).
# Reads a comparison_all.csv from a run written BEFORE the RDS existed.
#
# For those only; new runs read the RDS and never come through here. No column
# type is guessed or listed by hand: it comes from `tune_grid`, which is the
# authority on the hyperparameters and is loaded right there. The remaining
# columns are few and known -- identifiers and status are text, counters are
# integers, and what is left are metrics, which are numeric.
#' @noRd
.comparison_from_csv <- function(path, tune_grid) {
  cmp <- readr::read_csv2(path, show_col_types = FALSE)

  as_chr <- c("unit_id", "config_id", "status", "error_message",
              "window_sizes", "conv_channels")
  as_int <- c("fold", "seed", "best_epoch", "rank")

  for (nm in intersect(as_chr, names(cmp))) cmp[[nm]] <- as.character(cmp[[nm]])
  for (nm in intersect(as_int, names(cmp))) cmp[[nm]] <- as.integer(cmp[[nm]])

  # Hyperparameters: the type is whatever tune_grid says it is.
  for (nm in intersect(names(tune_grid), names(cmp))) {
    if (nm %in% c(as_chr, as_int)) next
    want <- class(tune_grid[[nm]])[1]
    got  <- class(cmp[[nm]])[1]
    if (identical(want, got)) next
    cmp[[nm]] <- switch(want,
      character = as.character(cmp[[nm]]),
      integer   = as.integer(cmp[[nm]]),
      numeric   = as.numeric(cmp[[nm]]),
      logical   = as.logical(cmp[[nm]]),
      cmp[[nm]])
  }

  # What is left is metrics: numeric, and text here means the locale
  # failed to parse some value (scientific notation, for instance).
  known <- unique(c(as_chr, as_int, names(tune_grid)))
  for (nm in setdiff(names(cmp), known)) {
    if (is.character(cmp[[nm]])) {
      cmp[[nm]] <- suppressWarnings(as.numeric(sub(",", ".", cmp[[nm]],
                                                   fixed = TRUE)))
    }
  }
  cmp
}

# The CSV is the readable copy; the RDS is the AUTHORITATIVE one, and the one
# resume reads. Always written together, so they cannot disagree.
write_comparison <- function(comparison, csv_path, rds_path) {
  safe_write_csv2(comparison, csv_path)
  safe_save_rds(comparison, rds_path, compress = FALSE)
  invisible(comparison)
}

# THE GRID IS WRITTEN WITH THE RUN, AND A RESUME IS HELD TO IT. If a
# tune_grid.rds already exists for this run_id, it MUST match the grid passed
# in now (same config_ids in the same order) before we trust any checkpoint
# found under models/. Reusing a run_id with a DIFFERENT grid would silently
# pair the wrong checkpoint with the wrong config_id. Both runners write it:
# the fold loop in this session, and the units side by side
# (R/train_workers.R).
.write_tune_grid <- function(run_dir, tune_grid, resume) {
  grid_rds_path <- file.path(run_dir, "tune_grid.rds")
  if (resume && file.exists(grid_rds_path)) {
    prev_grid <- readRDS(grid_rds_path)
    if (!identical(prev_grid$config_id, tune_grid$config_id)) {
      stop("resume = TRUE, but tune_grid.rds in ", run_dir, " lists config ids ",
           "that differ from the grid passed now.\n  Use a new run_id, or pass ",
           "the grid this run was started with.", call. = FALSE)
    }
  }

  # Save the grid so the run can be reproduced
  safe_save_rds(tune_grid, grid_rds_path, compress = FALSE)
  safe_write_csv2(
    dplyr::mutate(tune_grid,
      window_sizes  = purrr::map_chr(window_sizes, paste, collapse = "_"),
      conv_channels = purrr::map_chr(conv_channels, paste, collapse = "_")
    ),
    file.path(run_dir, "tune_grid.csv")
  )
  invisible(grid_rds_path)
}

run_cnn_tuning <- function(
  tune_grid,
  n_channels,
  cache,
  points_valid,
  transform   = identity,
  output_dir,
  device,
  run_id      = format(Sys.time(), "%Y%m%d_%H%M%S"),
  base_seed   = 42L,
  n_seeds     = 1L,
  fold        = 1L,
  resume      = TRUE,
  evaluate_test = FALSE,
  ...
) {
  n_seeds <- as.integer(n_seeds)
  fold    <- as.integer(fold)
  stopifnot(n_seeds >= 1L, fold >= 1L)
  run_dir <- file.path(output_dir, run_id)
  dirs    <- file.path(run_dir, c("models", "history", "predictions",
                                   "metrics", "gates", "comparison"))
  create_output_dirs(dirs)

  .write_tune_grid(run_dir, tune_grid, resume)

  # The cache is built by the caller and reused across every config here. It
  # used to be built inside this function from raw patches, which forced the
  # scaling to be a property of the stored data; now the scaling belongs to
  # the fold and the grid loop only consumes the result.
  windows_needed <- sort(unique(unlist(tune_grid$window_sizes)))
  keys_needed    <- patch_window_key(windows_needed)
  have_keys      <- setdiff(names(cache[[1]]), "y")
  missing_keys   <- setdiff(keys_needed, have_keys)
  if (length(missing_keys) > 0L) {
    stop("The grid needs window(s) ", paste(missing_keys, collapse = ", "),
         " but the cache only holds ", paste(have_keys, collapse = ", "),
         call. = FALSE)
  }

  n_cfg <- nrow(tune_grid)

  # ── Resume: reload comparison rows already computed, skip done configs ─────
  comparison_path <- file.path(run_dir, "comparison", "comparison_all.csv")
  comparison_rds  <- file.path(run_dir, "comparison", "comparison_all.rds")
  comparison <- tibble::tibble()
  done_ids   <- character(0)
  if (resume && file.exists(comparison_path)) {
    # A RESUME READS THE RDS, NEVER THE CSV.
    #
    # The CSV is for humans; it does not preserve type. read_csv2() guesses,
    # and guessed wrong in two ways that have each killed a run here:
    #
    #   window_sizes  "3"      (a single-window config) -> read as a NUMBER
    #   weight_decay  "1e-04"  (scientific notation)    -> with a decimal comma,
    #                                                     unparseable -> TEXT
    #
    # Either way bind_rows aborts with "Can't combine <double> and
    # <character>" -- AFTER training, losing that unit's work.
    #
    # The first attempt forced a LIST of columns to character. Wrong
    # strategy: the list is never complete, and every new column is another
    # chance to repeat it. The RDS keeps the tibble as it is -- no column to
    # remember, no type to guess.
    #
    # The CSV fallback exists only for runs written before this RDS did.
    comparison <- if (file.exists(comparison_rds)) {
      readRDS(comparison_rds)
    } else {
      .comparison_from_csv(comparison_path, tune_grid)
    }

    # Older runs have no unit_id column (one row per config). Treat those rows
    # as units named by their config_id, which is exactly what they were.
    if (!"unit_id" %in% names(comparison)) {
      comparison$unit_id <- comparison$config_id
    }
    done_ids <- comparison$unit_id[comparison$status == "success"]
  }
  # Belt-and-suspenders: a config only counts as done if BOTH the comparison
  # row AND the model checkpoint exist (checkpoint is written after training
  # succeeds, comparison row after that) — protects against a partially
  # written comparison_all.csv from a crash mid-write.
  done_ids <- done_ids[file.exists(file.path(run_dir, "models",
                                             paste0(done_ids, "_best.pt")))]

  # ...and a third condition, of a different kind: the cached unit must still
  # DESCRIBE the config its name claims. The checkpoint test above proves the
  # unit finished; this one proves it finished on the hyperparameters the grid
  # now asks for. See .resumable_units() for the run where that came apart.
  done_ids <- .resumable_units(done_ids, comparison, tune_grid)

  # ── The unit of work is (config, seed), not config ──────────────────────────
  # Flattened into one table instead of nested loops so that resume, ordering
  # and reporting all see the same list of things to do. Seeds run INNERMOST:
  # a config's repetitions finish together, so an interrupted run leaves whole
  # configs measured rather than every config measured once and none twice.
  units <- expand.grid(seed_i = seq_len(n_seeds), i = seq_len(n_cfg))
  units <- units[order(units$i, units$seed_i), c("i", "seed_i")]
  n_units <- nrow(units)

  if (length(done_ids) > 0L) {
    # `done_ids` counts the finished units of the WHOLE RUN (every fold);
    # `n_units` is what THIS call will train (one fold). Mixing the two
    # produced lines like "Resume: 18/9", which means nothing. Report this
    # fold's fraction and the run's total separately.
    mine <- sum(done_ids %in% sprintf("%s_f%d_s%d",
                                      rep(tune_grid$config_id, each = n_seeds),
                                      fold, seq_len(n_seeds)))
    message(sprintf(
      "Resume: %d/%d units of this fold already trained (%d in the whole run)",
      mine, n_units, length(done_ids)))
  }

  training <- list(...)
  for (u in seq_len(n_units)) {
    i      <- units$i[u]
    seed_i <- units$seed_i[u]
    cfg    <- tune_grid[i, ]

    # The unit name is ALWAYS the same shape.
    #
    # The first version of this shortened it to the bare config_id when there
    # was one fold and one seed, to keep the filenames from before resampling
    # existed. It was a trap: changing n_seeds renamed the units, so resuming a
    # 1-seed run and asking for 3 retrained everything AND left duplicate rows
    # in the comparison (config_id AND config_id_f1_s1 for the same work).
    #
    # A uniform name makes "raise n_seeds" a RESUMABLE change: the repetitions
    # already on disk are recognised and only the new ones train. What is lost
    # is reading run directories from before resampling -- and there are none:
    # stage 03 has not run since the outputs were cleared.
    unit_id <- sprintf("%s_f%d_s%d", cfg$config_id, fold, seed_i)

    if (unit_id %in% done_ids) {
      message("\n-- ", u, "/", n_units, ": ", unit_id,
              " -- already trained, skipping --")
      next
    }

    # Seed depends on the REPETITION, never on the config: see base_seed.
    this_seed <- base_seed + seed_i - 1L
    row <- .train_unit(
      cfg = cfg, unit_id = unit_id, fold = fold, this_seed = this_seed,
      header = sprintf("%d/%d: %s  (config %d/%d, fold %d, seed %d)", u, n_units,
                       unit_id, i, n_cfg, fold, this_seed),
      cache = cache, points_valid = points_valid, n_channels = n_channels,
      transform = transform, device = device, run_dir = run_dir,
      evaluate_test = evaluate_test, training = training)

    # A config that never reaches the comparison table is indistinguishable
    # from one that was never run -- and `resume` filters on status ==
    # "success", which implied a "failed" was meant to exist. Write it.
    if (nrow(comparison) > 0L) {
      # .env$ is not optional here: the column and the local variable share a
      # name, and without the pronoun dplyr resolves BOTH to the column, the
      # filter is always FALSE, and every row survives -- duplicating the unit
      # instead of replacing it.
      comparison <- dplyr::filter(comparison, unit_id != .env$unit_id)
    }
    comparison <- dplyr::bind_rows(comparison, row)
    write_comparison(comparison, comparison_path, comparison_rds)
  }

  comparison <- .rank_comparison(comparison, run_dir, sprintf("fold %d", fold),
                                 comparison_path)
  invisible(list(comparison = comparison, run_dir = run_dir))
}

# ── One unit: trained, its files written, its row returned ────────────────────
#
# THE ONE PLACE A UNIT IS TRAINED. The fold loop above calls it in this session,
# and each worker of the units side by side calls it in a process of its own
# (R/train_workers.R): one implementation, so a unit trained either way writes
# the same files and the same row. It was the body of the fold loop until
# 2026-09-29, and is that body unchanged.
#
#   cfg        one row of the grid
#   unit_id    <config>_f<fold>_s<repetition>
#   this_seed  the repetition's seed, base_seed + s - 1, set here, before the
#              loaders: they shuffle with torch's generator
#   header     the unit's line in the log
#   cache, points_valid  the fold's tensors and their metadata, row for row
#   training   the arguments for train_one_cnn()
#
# Returns the unit's row of the comparison table: status "success", or
# "failed" with its error_message -- a failure is recorded, never fatal.
.train_unit <- function(cfg, unit_id, fold, this_seed, header, cache, points_valid,
                        n_channels, transform, device, run_dir, evaluate_test,
                        training) {
  # The unit's seed, and the session's random numbers put back when it returns
  # (.rng_state(), R/resample.R): trained in the user's session, it no longer
  # moves their next draws.
  rng <- .rng_state()
  on.exit(.rng_restore(rng), add = TRUE)
  set.seed(this_seed)
  torch::torch_manual_seed(this_seed)

  message("\n-- ", header, " --")
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
  loaders <- .make_loaders_from_cache(cache, cfg)

  result <- tryCatch(
    do.call(train_one_cnn, c(list(
      cfg          = cfg,
      n_channels   = n_channels,
      loaders      = loaders,
      points_valid = points_valid,
      transform    = transform,
      device       = device,
      model_name   = unit_id), training)),
    error = function(e) {
      message("  ERROR in config ", cfg$config_id, ": ", conditionMessage(e))
      e
    }
  )

  if (inherits(result, "error")) {
    row <- dplyr::bind_cols(
      tibble::tibble(
        unit_id       = unit_id,
        config_id     = cfg$config_id,
        fold          = fold,
        seed          = this_seed,
        best_epoch    = NA_integer_,
        runtime_min   = NA_real_,
        best_val_loss = NA_real_,
        window_sizes  = paste(cfg$window_sizes[[1]], collapse = "x"),
        conv_channels = paste(cfg$conv_channels[[1]], collapse = "_"),
        status        = "failed",
        error_message = conditionMessage(result)
      ),
      dplyr::select(cfg, -config_id, -window_sizes, -conv_channels)
    )
    rm(result, loaders); invisible(gc(verbose = FALSE))
    return(row)
  }

  # Save outputs, named by UNIT: two seeds of the same config are two
  # models, two histories and two prediction tables, never one overwriting
  # the other.
  cid <- unit_id
  safe_torch_save(result$best_state, file.path(run_dir, "models",
                  paste0(cid, "_best.pt")))
  safe_write_csv2(result$history,
                  file.path(run_dir, "history", paste0(cid, "_history.csv")))
  # NO TEST ROW LEAVES THIS RUNNER WHILE evaluate_test IS FALSE.
  #
  # Three artefacts carry dataset_role and all three used to be written whole:
  # predictions/ was filtered on 2026-09-16, metrics/_perf.csv and
  # metrics/_perf_quantile.csv were not -- so the per-unit test CCC was on
  # disk in plain text for every unit of the run whose selection was later
  # frozen. See .drop_test_rows() in R/utils.R for the full account.
  #
  # Nothing is lost: score_test_grid() recomputes the test from the
  # checkpoints, after the selection is frozen, which is the whole point.
  safe_write_csv2(.drop_test_rows(result$pred_all, evaluate_test),
                  file.path(run_dir, "predictions", paste0(cid, "_pred_all.csv")))
  safe_write_csv2(.drop_test_rows(result$perf_all, evaluate_test),
                  file.path(run_dir, "metrics", paste0(cid, "_perf.csv")))
  safe_write_csv2(.drop_test_rows(result$perf_quantile, evaluate_test),
                  file.path(run_dir, "metrics", paste0(cid, "_perf_quantile.csv")))
  if (!is.null(result$gate)) {
    safe_write_csv2(result$gate$summary,
                    file.path(run_dir, "gates", paste0(cid, "_gate_summary.csv")))
    safe_write_csv2(result$gate$by_profile,
                    file.path(run_dir, "gates", paste0(cid, "_gate_profiles.csv")))
  }

  # Append to comparison table
  val_perf  <- dplyr::filter(result$perf_all, dataset_role == "validation")
  # ── The test set is NOT scored during tuning (evaluate_test) ──────────────
  #
  # A frozen test set is frozen only while nothing reads it. Scoring it on
  # every unit puts test_ccc in the comparison table beside val_ccc, and from
  # there it takes one glance to prefer the config that "also does well on
  # test" -- which is selection on the test set, done by a human instead of
  # an argmax, and it inflates the final number by exactly as much.
  #
  # The columns still EXIST, holding NA, so the table keeps one shape whether
  # the test was scored or not and every reader downstream is unchanged.
  # Stage 04 scores the test once, on the chosen config, which is the only
  # moment the number means what it is reported to mean.
  test_perf <- dplyr::filter(result$perf_all, dataset_role == "test")
  if (!isTRUE(evaluate_test)) test_perf <- test_perf[0, , drop = FALSE]
  if (nrow(test_perf) == 0L) {
    # No test set in this plan: the columns still exist, holding NA, so the
    # table has one shape whatever the plan was.
    test_perf <- val_perf
    test_perf[] <- lapply(test_perf, function(z) z[NA_integer_])
  }
  val_metrics <- val_perf %>%
    dplyr::select(n, ccc, r2, mae, nse, rmse, rpd, mqi, bias, bias_pct) %>%
    dplyr::rename_with(~ paste0("val_", .x))
  test_metrics <- test_perf %>%
    dplyr::select(n, ccc, r2, mae, nse, rmse, rpd, mqi, bias, bias_pct) %>%
    dplyr::rename_with(~ paste0("test_", .x))
  row <- dplyr::bind_cols(
    tibble::tibble(
      unit_id        = unit_id,
      config_id      = cfg$config_id,
      fold           = fold,
      seed           = this_seed,
      best_epoch     = result$best_epoch,
      runtime_min    = round(result$runtime_min, 2),
      best_val_loss  = round(result$best_val_loss, 6),
      window_sizes   = paste(cfg$window_sizes[[1]], collapse = "x"),
      conv_channels  = paste(cfg$conv_channels[[1]], collapse = "_"),
      status         = "success",
      error_message  = NA_character_
    ),
    dplyr::select(cfg, -config_id, -window_sizes, -conv_channels),
    val_metrics,
    test_metrics
  )
  rm(result, loaders); invisible(gc(verbose = FALSE))
  row
}

# ── A table ranked, written and reported ──────────────────────────────────────
#
# The end of every fold in this session, and of the run side by side: `scope`
# says which, in the one message that names it.
.rank_comparison <- function(comparison, run_dir, scope, comparison_path) {
  # Rank by VALIDATION metrics only — test set is read-only diagnostic
  #
  # Two tables, because they answer different questions:
  #   comparison_ranked.csv     every unit, as it was measured (the audit trail)
  #   comparison_by_config.csv  one row per config, mean +/- sd (the decision)
  # Ranking UNITS would let one lucky seed of a mediocre config outrank the
  # steady mean of a good one -- which is precisely the mistake repetitions
  # exist to prevent.
  n_ok <- sum(comparison$status == "success", na.rm = TRUE)
  # A FOLD IN WHICH NOTHING TRAINED STOPS WITH THE FIRST ERROR. Failed rows
  # carry no metric columns, so the arrange() below used to die inside dplyr
  # with a message about val_ccc -- after minutes of cache building, and with
  # the real cause sitting unread in error_message.
  if (nrow(comparison) > 0L && n_ok == 0L) {
    first_err <- comparison$error_message[!is.na(comparison$error_message)][1]
    stop("Every unit of ", scope, " failed. The first error was:\n  ",
         first_err, "\n  See status / error_message in ", comparison_path,
         call. = FALSE)
  }
  if (nrow(comparison) > 0) {
    # Failed configs sort last (val_ccc is NA) and get no rank -- they are
    # listed so the run is auditable, not ranked as if they had competed.
    comparison <- comparison %>%
      dplyr::arrange(dplyr::desc(val_ccc), val_mae) %>%
      dplyr::mutate(
        rank = dplyr::if_else(status == "success",
                              cumsum(status == "success"),
                              NA_integer_)
      )
    safe_write_csv2(comparison,
                    file.path(run_dir, "comparison", "comparison_ranked.csv"))

    # Aggregated view, written whenever there is anything to aggregate.
    by_config <- summarise_resamples(comparison)
    if (nrow(by_config) > 0L) {
      safe_write_csv2(by_config,
                      file.path(run_dir, "comparison", "comparison_by_config.csv"))
      if (max(by_config$n_units) > 1L) {
        message("\n-- Per config (mean +/- sd over ", max(by_config$n_units),
                " repetition(s)) --")
        print_wide(dplyr::slice_head(
          dplyr::select(by_config, rank, config_id, n_units, n_folds, n_seeds,
                        dplyr::starts_with("val_ccc"),
                        dplyr::starts_with("val_mae"), n_failed),
          n = 5))
        print_noise_floor(seed_noise_floor(comparison))
      }
    }

    n_bad <- nrow(comparison) - n_ok
    if (n_bad > 0L) {
      message("\n", n_bad, " config(s) FAILED -- see status/error_message in ",
              "comparison_ranked.csv: ",
              paste(comparison$config_id[comparison$status != "success"],
                    collapse = ", "))
    }
  }
  if (n_ok > 0L) {
    # THE BEST UNIT OF THE FOLD, and only the metric that chose it.
    #
    # This line used to end with the test CCC and the words "(diagnostic
    # only)". With evaluate_test = FALSE it printed "test_CCC=NA (diagnostic
    # only)" -- noise where a stale claim used to be. Both are gone: the
    # phrase was wrong even when the number was real, because a test score on
    # screen beside the selection metric is not a diagnostic, it is the input
    # to a choice somebody is about to make.
    message("\n-- Best config: ", comparison$config_id[1],
            " | val_CCC=", round(comparison$val_ccc[1], 3),
            " | val_MAE=", round(comparison$val_mae[1], 3), " --")
  } else {
    message("\n-- No config completed successfully. --")
  }

  comparison
}

# ── Resampling: the same grid, once per fold ──────────────────────────────────
#
# WHY THE FOLD IS THE OUTER LOOP
# The per-fold cache is the expensive object: scaling is fitted on that fold's
# training rows and broadcast over every patch, which costs seconds of CPU and
# gigabytes of RAM. Training one config is minutes. So the loop order is fold
# outside, configs and seeds inside -- each cache is built once and amortised
# over the whole grid. The intuitive order ("for each config, for each fold")
# would rebuild every cache k times for nothing.
#
# Everything lands in ONE run directory: the fold is a column, not a folder.
# That is what lets summarise_resamples() average over folds and seeds without
# anybody having to stitch directories together afterwards.

#' Run a tuning grid across every fold of a resampling plan.
#'
#' @param tune_grid    Grid from make_tune_grid().
#' @param store        Patch store from load_patch_store().
#' @param points       Point values, aligned via align_points_to_meta().
#' @param type_table   Predictor types, in channel order.
#' @param plan         A fold_plan (holdout(), spatial_folds(), ...).
#' @param n_seeds      Repetitions per config within each fold.
#' @param release_store When TRUE (default) the raw patch tensors are dropped
#'   after the LAST fold's cache is built. They are needed until then, since
#'   each fold rescales them from raw.
#' @param evaluate_test Score the held-out test set on every unit? FALSE, and
#'   deliberately so: a frozen test set stops being frozen the moment its score
#'   sits in the tuning table next to the validation score. Stage 04 scores it
#'   once, on the config that was chosen without it. Set TRUE only to study the
#'   optimism itself -- never to choose anything.
#' @param in_session   TRUE (the default here) trains the units one at a time
#'   in this session, fold by fold; FALSE trains them side by side, each worker
#'   an R process of its own (R/train_workers.R). dsm_train() decides which.
#' @param threads_per_unit Side by side: the threads of each unit.
#' @param n_cores      Side by side: the cores of the whole run.
#' @param max_ram_gb   Side by side: the RAM the workers may use together.
#' @param ...          Passed through to run_cnn_tuning() and train_one_cnn().
#' @return list(comparison, by_config, run_dir, plan), and side by side
#'   n_workers, threads_per_unit and peak_gb too.
#' @noRd
run_cnn_resample <- function(tune_grid, store, points, type_table, plan,
                             transform  = identity,
                             output_dir,
                             device,
                             run_id     = format(Sys.time(), "%Y%m%d_%H%M%S"),
                             base_seed  = 42L,
                             n_seeds    = 1L,
                             resume     = TRUE,
                             evaluate_test = FALSE,
                             release_store = TRUE,
                             in_session = TRUE,
                             threads_per_unit = 5L,
                             n_cores = NULL,
                             max_ram_gb = NULL,
                             ...) {
  stopifnot(inherits(plan, "fold_plan"))

  # A batch larger than a fold's training set trains nothing (see
  # .make_loaders_from_cache()). Found HERE, before the first unit, instead of
  # once per unit, and named by config -- the fix is in the grid.
  n_train_min <- min(vapply(plan$folds, function(f) length(f$train), integer(1)))
  if ("batch_size" %in% names(tune_grid)) {
    too_big <- tune_grid$batch_size > n_train_min
    if (any(too_big)) {
      stop("batch_size larger than the smallest fold's ", n_train_min,
           " training point(s) in config(s) ",
           paste(tune_grid$config_id[too_big], collapse = ", "), " (",
           paste(unique(tune_grid$batch_size[too_big]), collapse = ", "),
           "). Such a unit would take no gradient step and report an untrained ",
           "network as a result. Use batch_size <= ", n_train_min, ".",
           call. = FALSE)
    }
  }

  # A broken plan costs a second to find here and the whole run to find later.
  # meta is passed so the no-split-group property is PROVEN on this data,
  # not merely intended by the constructor that built the plan.
  fold_sizes <- check_fold_plan(plan, meta = store$meta)
  message("\n-- Resampling plan --")
  print(plan)

  # -- The size of each config, ONCE -----------------------------------------
  #
  # one_se() picks the SIMPLEST config within one standard error of the best,
  # and "simplest" has to be a number in the table or the framework is inventing
  # an ordering. The tabular runner already records n_params from the fitted
  # model; the CNN runner did not, so one_se() failed on a CNN tuning run with
  # a message about a missing column -- on a run that had just cost hours.
  #
  # Computed per CONFIG, not per unit: it depends only on the architecture, and
  # a run repeats each config once per fold and per seed. Building the module to
  # count its parameters is cheap beside training it, but nine times over is
  # still eight times too many.
  #
  # It rides in the grid, so it reaches the comparison table through the same
  # bind_cols() as every other config field -- no second path to keep in sync.
  if (!"n_params" %in% names(tune_grid)) {
    tune_grid$n_params <- vapply(seq_len(nrow(tune_grid)), function(i) {
      tryCatch(as.numeric(count_model_params(tune_grid[i, ], store$n_channels)),
               error = function(e) NA_real_)
    }, numeric(1))
  }

  windows_needed <- sort(unique(unlist(tune_grid$window_sizes)))
  run_dir <- file.path(output_dir, run_id)
  create_output_dirs(file.path(run_dir, "comparison"))

  # The plan is written BEFORE any training: results whose folds cannot be
  # reconstructed are results that cannot be defended.
  # BEFORE the plan is written, or the check compares the plan with itself.
  # The cached units were fitted on the cached plan; if the split has moved, no
  # amount of matching hyperparameters makes them comparable.
  check_plan_unchanged(plan, run_dir, resume = resume)
  safe_save_rds(plan, file.path(run_dir, "fold_plan.rds"), compress = FALSE)
  safe_write_csv2(dplyr::mutate(fold_sizes, method = plan$method),
                  file.path(run_dir, "fold_sizes.csv"))
  if (!is.null(plan$assignment)) {
    safe_write_csv2(plan$assignment,
                    file.path(run_dir, "fold_assignment.csv"))
  }
  if (!is.null(plan$buffer_dropped)) {
    safe_write_csv2(plan$buffer_dropped,
                    file.path(run_dir, "fold_buffer_dropped.csv"))
  }

  # SIDE BY SIDE the units train in workers of their own, each building its own
  # fold caches (R/train_workers.R). What follows is the fold loop in this
  # session, one unit at a time.
  if (!isTRUE(in_session)) {
    res <- .train_side_by_side(
      tune_grid = tune_grid, store = store, points = points, type_table = type_table,
      plan = plan, transform = transform, run_dir = run_dir, base_seed = base_seed,
      n_seeds = n_seeds, resume = resume, evaluate_test = evaluate_test,
      training = list(...), windows = windows_needed, n_cores = n_cores,
      threads_per_unit = threads_per_unit, max_ram_gb = max_ram_gb)
    return(list(comparison = res$comparison,
                by_config = summarise_resamples(res$comparison),
                run_dir = run_dir, plan = plan, n_workers = res$n_workers,
                threads_per_unit = threads_per_unit, peak_gb = res$peak_gb))
  }

  # ONE BUFFER FOR EVERY FOLD. Each fold's roles used to be tensors of their
  # own, released before the next fold's were built -- released to mimalloc,
  # which keeps every freed block of ~250 MB or more (T6). T8 (2026-09-28), the
  # deployed configuration on the dev store's three folds: without the buffer
  # the folds after the first left 0.13 and 0.74 GB behind them, about their
  # 0.32 GB training role a fold; with it, 0.03 and 0.00 -- and the tuning
  # table the same to the last digit. The full data set's roles are ~11x
  # larger. The roles are now slices of one tensor per window, made on the
  # first fold (new_fold_buffer()); options(dsm.fold_buffer = FALSE) goes back
  # to tensors of their own, for comparison.
  buffer <- if (.use_fold_buffer()) new_fold_buffer() else NULL

  # THE FOLD LOOP'S MEMORY, when asked (options(dsm.train.trace_mem = TRUE);
  # T8 does): the process's level before the first fold and, per fold, after
  # its cache is built, after its units trained, and after it is released --
  # each after a collection, so a level is what is held -- written to
  # logs/train_mem_trace.rds at every mark. The dsm_final() worker's trace is
  # the same measurement, per unit (.final_worker()).
  trace_mem <- isTRUE(getOption("dsm.train.trace_mem", FALSE))
  trace <- list()
  t0 <- Sys.time()
  mark <- function(phase, fold = NA_integer_) {
    if (!trace_mem) return(invisible(NULL))
    g <- gc(verbose = FALSE)
    trace[[length(trace) + 1L]] <<- data.frame(
      fold = as.integer(fold), phase = phase,
      seconds = as.numeric(difftime(Sys.time(), t0, units = "secs")),
      rss_gb = .predict_rss_gb(), private_gb = .predict_private_gb(),
      peak_gb = .final_peak_gb(), r_heap_gb = sum(g[, 2]) / 1024,
      stringsAsFactors = FALSE)
    create_output_dirs(file.path(run_dir, "logs"))
    safe_save_rds(do.call(rbind, trace), file.path(run_dir, "logs", "train_mem_trace.rds"),
                  compress = FALSE)
    invisible(NULL)
  }
  mark("start")

  comparison <- tibble::tibble()
  for (j in seq_along(plan$folds)) {
    idx <- plan$folds[[j]]
    message("\n", strrep("=", 78))
    message("FOLD ", j, "/", plan$n_folds, " -- ",
            paste(sprintf("%s=%d", names(idx), lengths(idx)), collapse = " | "))
    message(strrep("=", 78))

    # Scaling is fitted on THIS fold's training rows. That is the whole reason
    # the patches are stored raw: a fold whose scaling came from another fold's
    # training set has already seen data it should not have.
    fold <- build_fold_cache(store, points, type_table, idx, windows_needed,
                             buffer = buffer)
    mark("fold_cache", j)
    safe_write_csv2(fold$scaling,
                    file.path(run_dir, sprintf("scaling_fold%02d.csv", j)))

    if (release_store && j == length(plan$folds)) {
      store$windows <- NULL
      invisible(gc(verbose = FALSE))
    }

    res <- run_cnn_tuning(
      tune_grid    = tune_grid,
      n_channels   = store$n_channels,
      cache        = fold$cache,
      points_valid = fold_points_valid(store, idx),
      transform    = transform,
      output_dir   = output_dir,
      device       = device,
      run_id       = run_id,
      base_seed    = base_seed,
      n_seeds      = n_seeds,
      fold         = j,
      resume       = resume,
      evaluate_test = evaluate_test,
      ...
    )
    comparison <- res$comparison
    mark("fold_trained", j)

    # This fold's cache goes before the next one is built: with a buffer the
    # next fold overwrites it, and without one, two folds of scaled patches in
    # memory at once is the one thing that does not fit.
    rm(fold); invisible(gc(verbose = FALSE))
    mark("fold_released", j)
  }
  rm(buffer); invisible(gc(verbose = FALSE))

  by_config <- summarise_resamples(comparison)
  list(comparison = comparison, by_config = by_config,
       run_dir = run_dir, plan = plan)
}

# ── DataLoader builders (internal) ────────────────────────────────────────────

#' Build the four DataLoaders for one config from a prebuilt fold cache.
#'
#' tensor_dataset only references the cached tensors (no copy); dataloaders are
#' cheap to (re)create per config, so only batch_size-dependent objects are
#' rebuilt here.
#'
#' Window keys come from patch_window_key(), the same helper the patch store
#' uses for its filenames – one naming rule, so a window can never be looked
#' up under a name nothing ever wrote.
#' @noRd
.make_loaders_from_cache <- function(cache, cfg) {
  ws       <- cfg$window_sizes[[1]]
  bs_train <- cfg$batch_size
  bs_eval  <- min(bs_train * 4L, 2048L)
  keys     <- patch_window_key(ws)

  make_ds <- function(role) {
    arrays <- lapply(keys, function(k) cache[[role]][[k]])
    gone   <- vapply(arrays, is.null, logical(1))
    if (any(gone)) {
      stop("Cache for role '", role, "' is missing window(s): ",
           paste(keys[gone], collapse = ", "), call. = FALSE)
    }
    do.call(torch::tensor_dataset, c(arrays, list(cache[[role]]$y)))
  }

  train_ds <- make_ds("train")
  val_ds   <- make_ds("validation")

  # A BATCH LARGER THAN THE TRAINING SET TRAINS NOTHING. The training loader
  # drops its incomplete last batch (below), so with more rows per batch than
  # the fold has, every epoch is ZERO gradient steps -- and the unit then
  # "succeeds": the validation loss of the untrained network is finite, it
  # becomes the best epoch, and the comparison table ranks a random network
  # beside trained ones. run_cnn_resample() refuses such a grid before the
  # first unit; this is the backstop for any other path here.
  n_tr <- length(train_ds)
  if (n_tr < bs_train) {
    stop("Config ", cfg$config_id, ": batch_size ", bs_train, " is larger than ",
         "this fold's ", n_tr, " training point(s). The training loader drops ",
         "the incomplete last batch (BatchNorm cannot take a batch of one), so ",
         "the unit would take no gradient step and report an untrained network ",
         "as a result. Use batch_size <= ", n_tr, ", or let dsm_train() draw the ",
         "grid: it sizes the batches to the smallest fold.", call. = FALSE)
  }
  # The test role is OPTIONAL. A plan is allowed to carve no test set -- the
  # framework does not invent one nobody asked for -- and training must work
  # under that, evaluating and reporting only what exists.
  test_ds  <- if (!is.null(cache$test)) make_ds("test") else NULL

  # drop_last = TRUE on the training loader: prevents a final batch of size 1,
  # which would make BatchNorm fail (variance of a single sample). Eval loaders
  # keep every sample and never shuffle – predict_loader() depends on that,
  # since it pairs loader row i with metadata row i.
  out <- list(
    train      = torch::dataloader(train_ds, batch_size = bs_train, shuffle = TRUE, drop_last = TRUE),
    train_eval = torch::dataloader(train_ds, batch_size = bs_eval,  shuffle = FALSE),
    validation = torch::dataloader(val_ds,   batch_size = bs_eval,  shuffle = FALSE)
  )
  if (!is.null(test_ds)) {
    out$test <- torch::dataloader(test_ds, batch_size = bs_eval, shuffle = FALSE)
  }
  # The calibration set, in a final refit's cache only: predicted, never
  # shuffled, never trained on.
  if (!is.null(cache$calibration)) {
    out$calibration <- torch::dataloader(make_ds("calibration"), batch_size = bs_eval,
                                         shuffle = FALSE)
  }
  out
}
