# Hyperparameter grid generation — caret-style tuning for CNN
#
# Usage
# -----
# grid <- make_tune_grid(tune_length = 30)         # random sample
# grid <- make_tune_grid(tune_length = 30, seed = 42)
# grid <- make_full_tune_grid()                     # full factorial (large!)
#
# Each row of the returned tibble is one complete model configuration.
# Pass it directly to run_cnn_tuning().
#
# Parameter reference — see docs/tuning_guide.md for full explanations.

# ── Parameter space definition ────────────────────────────────────────────────

.cnn_param_space <- list(

  # ── Spatial architecture ──────────────────────────────────────────────────
  # Which patch size(s) to use.
  # Length-1 vector → single branch. Length-2 → dual branch with gate.
  # Smaller window = local processes. Larger window = landscape processes.
  # Typical values for 20 km resolution rasters (20 km grid):
  #   3×3 patch covers 60 km, 5×5 covers 100 km, 7×7 covers 140 km.
  window_sizes = list(
    c(3L),
    c(5L),
    c(7L),
    c(3L, 5L),
    c(3L, 7L),
    c(5L, 7L)
  ),

  # ── Conv channel configurations ───────────────────────────────────────────
  # Each vector defines: how many feature maps per conv block.
  # Length of vector = number of conv blocks (network depth).
  # More channels = more capacity = more parameters = higher overfitting risk.
  # Start conservative (64, 128) for datasets < 30 k samples.
  conv_channels = list(
    c(32L, 64L),
    c(64L, 128L),
    c(64L, 128L, 128L),
    c(128L, 256L),
    c(128L, 256L, 256L)
  ),

  # ── Skip connections (residual) ───────────────────────────────────────────
  # ResNet-style skip connections: the input to a conv block is added to
  # its output. Critical for networks with ≥ 2 conv blocks because they
  # prevent vanishing gradients and allow the network to learn incremental
  # refinements rather than full transformations.
  # Recommendation: always TRUE when n_conv_blocks ≥ 2.
  use_residual = c(TRUE, FALSE),

  # ── Channel attention (SE block) ──────────────────────────────────────────
  # Squeeze-and-Excitation block re-weights each feature map channel using
  # a global summary. With 200+ input predictors compressed into feature
  # maps, SE helps the network focus on the most relevant predictor groups
  # for each spatial context.
  use_se_block = c(TRUE, FALSE),

  # ── SE bottleneck ratio ───────────────────────────────────────────────────
  # The SE MLP has channels / reduction neurons in its hidden layer.
  # Smaller reduction = bigger bottleneck = more parameters in SE.
  # se_reduction = 16 (default) is a good balance.
  se_reduction = c(8L, 16L, 32L),

  # ── Embedding dimension ───────────────────────────────────────────────────
  # Size of the vector produced by each branch after the linear projection.
  # The gate and head operate on this space.
  # Larger = more representational capacity, but scales parameters quadratically.
  # For 200+ channels: 128–512 is a reasonable range.
  embedding_dim = c(128L, 256L, 384L, 512L),

  # ── Fusion gate type (dual-branch only) ──────────────────────────────────
  # Controls how the two branch embeddings are combined.
  # "vector_featurewise": one gate weight per embedding dimension (most expressive).
  #   The gate is learned from [f1, f2, |f1-f2|, f1*f2], so it sees both
  #   agreement and disagreement between scales.
  # "scalar_per_sample": one weight for the whole embedding (simpler).
  # "no_gate_concat": no gating; branches are concatenated and the head learns fusion.
  gate_type = c("vector_featurewise", "scalar_per_sample", "no_gate_concat"),

  # ── Regularisation: dropout rates ─────────────────────────────────────────
  # Spatial dropout (2D): drops entire feature map channels during training.
  # More aggressive than element-wise dropout for structured data.
  branch_spatial_dropout = c(0.0, 0.02, 0.05),

  # Embedding dropout: applied after the branch linear projection.
  embed_dropout = c(0.0, 0.05, 0.10),

  # Gate dropout: applied inside the gate network.
  gate_dropout = c(0.0, 0.05, 0.10, 0.20),

  # Head dropout layers (two independent rates).
  head_dropout_1 = c(0.10, 0.20, 0.30),
  head_dropout_2 = c(0.0, 0.05, 0.10),

  # ── Optimiser and learning rate ───────────────────────────────────────────
  # Adam is used throughout (adaptive LR per parameter, moment estimates).
  # base_lr: the peak learning rate after warmup.
  # Too high → unstable / diverges. Too low → slow convergence.
  # Range 1e-4 to 3e-3 covers almost all practical CNN regression tasks.
  base_lr = c(0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.003),

  # L2 regularisation on all weights (AdamW-style).
  # Higher values shrink weights toward zero → reduces overfitting.
  # For moderately-sized CNNs on soil data: 0 to 1e-3.
  weight_decay = c(0.0, 1e-5, 1e-4, 1e-3),

  # ── Batch size ────────────────────────────────────────────────────────────
  # Number of samples per gradient update step.
  # Larger batches: more stable gradients, fewer updates per epoch.
  # Smaller batches: noisier updates (can act as regulariser).
  # With CUDA: limited by GPU memory. 512 is usually safe for 200+ channels.
  batch_size = c(128L, 256L, 512L),

  # ── Loss function ─────────────────────────────────────────────────────────
  # Training objective (note: metrics for model selection are computed
  # separately using all metrics in metrics.R).
  # "smooth_l1" (Huber loss): MAE for large residuals, MSE-like near zero.
  #   Robust to outliers in skewed distributions (e.g., high-SOC peatlands).
  #   Recommended for soil carbon data.
  # "mse": MSE loss. More sensitive to extreme values.
  # "mae": pure L1 loss. Equally robust to outliers but non-smooth.
  loss_fn = c("smooth_l1", "mse", "mae"),

  # ── LR warmup epochs ─────────────────────────────────────────────────────
  # Number of epochs to linearly ramp LR from warmup_start_lr to base_lr.
  # Prevents gradient explosions at the start when weights are random.
  # 5 epochs is sufficient for most datasets.
  warmup_epochs = c(3L, 5L, 10L)
)

# ── Grid generation ───────────────────────────────────────────────────────────

#' Generate a random hyperparameter grid (like caret's tuneLength).
#'
#' @param tune_length  Number of configurations to sample.
#' @param seed         Random seed for reproducibility.
#' @param fixed        Named list of parameters to fix (overrides sampling).
#'   Example: fixed = list(loss_fn = "smooth_l1", batch_size = 512L)
#'
#' @return A tibble with one row per configuration.
#'   window_sizes and conv_channels are stored as list-columns.
make_tune_grid <- function(tune_length = 20L, seed = NULL, fixed = list()) {
  if (!is.null(seed)) set.seed(seed)

  space <- .cnn_param_space
  for (nm in names(fixed)) space[[nm]] <- list(fixed[[nm]])

  rows <- vector("list", tune_length)
  for (i in seq_len(tune_length)) {
    row <- lapply(space, function(choices) {
      if (is.list(choices)) choices[[sample(length(choices), 1L)]]
      else                  choices[sample(length(choices), 1L)]
    })
    # Enforce consistency: single window → gate irrelevant
    if (length(row$window_sizes) == 1L) row$gate_type <- "no_gate_concat"
    row$config_id <- sprintf("cfg_%03d", i)
    rows[[i]] <- row
  }

  .rows_to_tibble(rows)
}

#' Build a manual grid from explicit lists of values per parameter.
#'
#' @param ...  Named arguments, each a vector or list of values to cross.
#'   Only the provided parameters are varied; all others use their first
#'   (default) value from the parameter space.
#'
#' Example:
#'   make_manual_tune_grid(
#'     embedding_dim = c(256L, 384L),
#'     gate_type     = c("vector_featurewise", "no_gate_concat"),
#'     base_lr       = c(0.001, 0.0005)
#'   )
make_manual_tune_grid <- function(...) {
  overrides <- list(...)
  space     <- lapply(.cnn_param_space, function(x) {
    if (is.list(x)) x[1L] else x[1L]
  })
  for (nm in names(overrides)) {
    v <- overrides[[nm]]
    space[[nm]] <- if (is.list(v)) v else as.list(v)
  }
  combos <- expand.grid(
    lapply(space, seq_along),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  rows <- vector("list", nrow(combos))
  for (i in seq_len(nrow(combos))) {
    row <- mapply(function(choices, idx) {
      if (is.list(choices)) choices[[idx]] else choices[idx]
    }, space, as.integer(combos[i, ]), SIMPLIFY = FALSE)
    if (length(row$window_sizes) == 1L) row$gate_type <- "no_gate_concat"
    row$config_id <- sprintf("cfg_%03d", i)
    rows[[i]] <- row
  }
  .rows_to_tibble(rows)
}

# ── Internal helper ───────────────────────────────────────────────────────────

.rows_to_tibble <- function(rows) {
  scalar_cols <- setdiff(names(rows[[1L]]), c("window_sizes", "conv_channels"))

  scalar_df <- do.call(rbind, lapply(rows, function(r) {
    as.data.frame(r[scalar_cols], stringsAsFactors = FALSE)
  }))

  window_list <- lapply(rows, `[[`, "window_sizes")
  conv_list   <- lapply(rows, `[[`, "conv_channels")

  tibble::tibble(
    config_id       = scalar_df$config_id,
    window_sizes    = window_list,
    conv_channels   = conv_list,
    use_residual    = as.logical(scalar_df$use_residual),
    use_se_block    = as.logical(scalar_df$use_se_block),
    se_reduction    = as.integer(scalar_df$se_reduction),
    embedding_dim   = as.integer(scalar_df$embedding_dim),
    gate_type       = scalar_df$gate_type,
    spatial_dropout = as.numeric(scalar_df$branch_spatial_dropout),
    embed_dropout   = as.numeric(scalar_df$embed_dropout),
    gate_dropout    = as.numeric(scalar_df$gate_dropout),
    head_dropout_1  = as.numeric(scalar_df$head_dropout_1),
    head_dropout_2  = as.numeric(scalar_df$head_dropout_2),
    base_lr         = as.numeric(scalar_df$base_lr),
    weight_decay    = as.numeric(scalar_df$weight_decay),
    batch_size      = as.integer(scalar_df$batch_size),
    loss_fn         = scalar_df$loss_fn,
    warmup_epochs   = as.integer(scalar_df$warmup_epochs)
  )
}
