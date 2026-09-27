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
  #
  # A window's physical extent is window_size × raster resolution — it is NOT a
  # fixed distance, it scales with resolution. A window set tuned at one
  # resolution does not transfer to another, so re-pick when you change it.
  # See docs/tuning_guide.md.
  # Applied example at 250 m:
  #   3×3   ≈ 0.75 km  (local neighbourhood)
  #   9×9   ≈ 2.25 km  (hillslope / soil–landscape position)
  #   15×15 ≈ 3.75 km  (local landscape / catchment context)
  #
  # THESE SIX ARE THE SOC EXAMPLE'S, NOT A DEFAULT. dsm_train() never draws
  # from them: it passes the windows the loaded store holds, and
  # .window_options() turns those into every single window and every pair --
  # which for a store of 3, 9 and 15 is exactly this list, in this order, so
  # a seed draws the same grid either way. They stay here for make_tune_grid()
  # called on its own, without a store to ask.
  window_sizes = list(
    c(3L),
    c(9L),
    c(15L),
    c(3L, 9L),
    c(3L, 15L),
    c(9L, 15L)
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
  # Fixed at 16: this is a 2nd-order knob whose effect on accuracy is
  # negligible relative to window size, depth and learning rate. Varying it
  # only wastes tuning budget. Override via fixed=list(se_reduction=...) if
  # you specifically want to study it.
  se_reduction = c(16L),

  # ── Embedding dimension ───────────────────────────────────────────────────
  # Size of the vector produced by each branch after the linear projection.
  # The gate and head operate on this space.
  # Larger = more representational capacity, but scales parameters quadratically.
  # For 200+ channels: 128–512 is a reasonable range.
  embedding_dim = c(128L, 256L, 384L, 512L),

  # ── Pre-embedding spatial reduction ───────────────────────────────────────
  # How each branch reduces its C×w×w feature map before the linear projection:
  #   "flatten" — keep every cell (input C·w·w). Full spatial detail, but the
  #     linear layer grows with window² — a 15×15 branch concentrates ~11 M
  #     params here. Best for small windows or when spatial detail matters.
  #   "gap"     — global average pool to C (input C, window-independent). Removes
  #     the window² blow-up, so large windows stay light and overfit less.
  # Default first value "flatten" preserves the original architecture; include
  # "gap" in the search when large windows risk overfitting (esp. fine
  # resolution). Both branches of a dual model share the choice.
  embed_pool = c("flatten", "gap"),

  # ── Convolution padding ───────────────────────────────────────────────────
  # "same"        pad with a ring of zeros, keeping the spatial size. Every
  #               output position of a small patch then depends partly on
  #               invented values -- with window 3 and 2 blocks the centre's
  #               receptive field is already 5x5, LARGER than the patch, so
  #               there is no position that does not.
  # "valid"       no padding: only measured values, and the map shrinks by 2
  #               per block. Needs window > 2 x blocks or there is nothing
  #               left, so it is not available to a 3x3 branch.
  # "valid_large" the useful one: valid on the LARGE branch, same on the small.
  #               A 15x15 branch loses 4 of 15 pixels and keeps only real data;
  #               the 3x3 branch, which cannot afford to shrink, is untouched.
  #
  # Why the large branch is where the gain is: a w x w patch re-reads each
  # pixel w^2 times across the dataset, so the 15 branch is 225x redundant and
  # the 3 branch only 9x. Trading border for honesty is nearly free on one and
  # expensive on the other.
  #
  # "same" is listed first so it remains the default behaviour; the other two
  # are in the search because the question is open and costs one parameter.
  conv_padding = c("same", "valid_large"),

  # ── Fusion gate type (dual-branch only) ──────────────────────────────────
  # Controls how the two branch embeddings are combined.
  # "vector_featurewise": one gate weight per embedding dimension (most expressive).
  #   The gate is learned from [f1, f2, |f1-f2|, f1*f2], so it sees both
  #   agreement and disagreement between scales.
  # "scalar_per_sample": one weight for the whole embedding (simpler).
  # "no_gate_concat": no gating; branches are concatenated and the head learns fusion.
  gate_type = c("vector_featurewise", "scalar_per_sample", "no_gate_concat"),

  # ── Regularisation: a single dropout knob ─────────────────────────────────
  # The model has five internal dropout sites (spatial, embedding, gate, two
  # head layers). Tuning them independently makes the search space huge and
  # non-identifiable — many combinations are practically equivalent. Instead a
  # single `dropout` strength controls overall regularisation and is mapped to
  # the five sites with fixed sensible ratios (see .expand_dropout):
  #   spatial = 0.25·d   embedding = 0   gate = 0.5·d
  #   head_1  = 1.0·d    head_2    = 0.5·d
  # This keeps the knob interpretable: 0 = none, 0.3 = strong. Power users can
  # still set the five sites explicitly via make_manual_tune_grid().
  dropout = c(0.0, 0.1, 0.2, 0.3),

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
  # Fixed at 5: an optimisation detail, not a model hyperparameter — 5 epochs
  # is sufficient across datasets and varying it wastes budget.
  warmup_epochs = c(5L)
)

# ── Dropout expansion ─────────────────────────────────────────────────────────
# Map the single `dropout` strength to the five internal dropout sites used by
# dual_branch_cnn(). Embedding dropout stays 0 (its effect is marginal right
# after BatchNorm). Returns a named list.

.expand_dropout <- function(d) {
  list(
    spatial_dropout = round(0.25 * d, 4),
    embed_dropout   = 0.0,
    gate_dropout    = round(0.50 * d, 4),
    head_dropout_1  = round(1.00 * d, 4),
    head_dropout_2  = round(0.50 * d, 4)
  )
}

# ── Grid generation ───────────────────────────────────────────────────────────

#' Generate a random hyperparameter grid (like caret's tuneLength).
#'
#' @param tune_length  Number of configurations to sample.
#' @param seed         Random seed for reproducibility.
#' @param fixed        Named list used to FIX or RESTRICT parameters:
#'   - a length-1 value fixes the parameter        (e.g. loss_fn = "smooth_l1")
#'   - a length>1 vector restricts the sampling pool (e.g. base_lr = c(1e-4, 3e-4))
#'   - for the list-valued params window_sizes / conv_channels, pass a list of
#'     options (e.g. window_sizes = list(c(7L), c(5L, 7L))) or a single vector.
#'   Example focused search:
#'     fixed = list(
#'       loss_fn      = "smooth_l1",
#'       batch_size   = 512L,
#'       base_lr      = c(1e-4, 3e-4),
#'       window_sizes = list(c(7L), c(5L, 7L))
#'     )
#' @param windows      The patch sizes the store holds (`data$store$window_sizes`).
#'   The window options become every single window and every pair of them --
#'   a dual-branch model takes two. NULL keeps the SOC example's 3/9/15 set;
#'   dsm_train() always passes the store's.
#' @param n_train      Training points in the smallest fold. Batch sizes that
#'   would give an epoch fewer than four gradient steps are left out; if none
#'   is left, the largest power of two that gives four is used. NULL keeps the
#'   full set. dsm_train() passes it from the plan.
#'
#' @return A tibble with one row per configuration.
#'   window_sizes and conv_channels are stored as list-columns.
make_tune_grid <- function(tune_length = 20L, seed = NULL, fixed = list(),
                           windows = NULL, n_train = NULL) {
  if (!is.null(seed)) set.seed(seed)

  space        <- .cnn_param_space
  list_params  <- c("window_sizes", "conv_channels")
  if (!is.null(windows)) {
    space$window_sizes <- .window_options(windows)
    # A restriction that asks for a window the store does not hold is refused
    # HERE, where the fix is one argument, and not by the loader, once the run
    # is under way.
    if (!is.null(fixed$window_sizes)) {
      have  <- sort(unique(as.integer(windows)))
      gone  <- setdiff(sort(unique(unlist(fixed$window_sizes))), have)
      if (length(gone) > 0L) {
        stop("fixed$window_sizes asks for window(s) ", paste(gone, collapse = ", "),
             ", which the store does not hold (it has ",
             paste(have, collapse = ", "), ").", call. = FALSE)
      }
    }
  }
  if (!is.null(n_train)) space$batch_size <- .batch_options(space$batch_size, n_train)
  for (nm in names(fixed)) {
    v <- fixed[[nm]]
    space[[nm]] <- if (nm %in% list_params) {
      if (is.list(v)) v else list(v)   # single vector → one option
    } else {
      v                                 # atomic vector → restricted pool
    }
  }

  # Sample with replacement but DE-DUPLICATE: two independent draws can land on
  # the same configuration, which would waste training budget on a repeat. Keep
  # drawing until tune_length unique configs are found (or the space is
  # exhausted, capped by max_tries).
  rows      <- vector("list", tune_length)
  seen      <- character(0)
  n_kept    <- 0L
  tries     <- 0L
  max_tries <- tune_length * 100L

  while (n_kept < tune_length && tries < max_tries) {
    tries <- tries + 1L
    row <- lapply(space, function(choices) {
      if (is.list(choices)) choices[[sample(length(choices), 1L)]]
      else                  choices[sample(length(choices), 1L)]
    })
    # Enforce consistency: single window → gate irrelevant
    if (length(row$window_sizes) == 1L) row$gate_type <- "no_gate_concat"
    # ...and a padding the geometry cannot honour is NOT a different config.
    #
    # dual_branch_cnn() resolves "valid_large" to "same" wherever the branch
    # cannot afford to shrink. Left unnormalised here, two rows that build the
    # IDENTICAL model survive de-duplication under different names, and the
    # grid then spends part of its budget training the same thing twice and
    # reports it as two results.
    row$conv_padding <- .normalise_conv_padding(row$conv_padding,
                                                row$window_sizes,
                                                row$conv_channels)

    # Signature over all parameters (config_id not yet assigned)
    sig <- paste(rapply(row, function(z) paste(z, collapse = "-"), how = "unlist"),
                 collapse = "|")
    if (sig %in% seen) next

    seen          <- c(seen, sig)
    n_kept        <- n_kept + 1L
    row$config_id <- sprintf("cfg_%03d", n_kept)
    rows[[n_kept]] <- row
  }

  if (n_kept < tune_length) {
    rows <- rows[seq_len(n_kept)]
    message("make_tune_grid: only ", n_kept, " unique configs available ",
            "(requested ", tune_length, ") — search space likely exhausted.")
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
    row$conv_padding <- .normalise_conv_padding(row$conv_padding,
                                                row$window_sizes,
                                                row$conv_channels)
    row$config_id <- sprintf("cfg_%03d", i)
    rows[[i]] <- row
  }
  .rows_to_tibble(rows)
}

# ── Internal helpers ──────────────────────────────────────────────────────────

# The window options a store can serve: every window alone, then every pair,
# smaller first -- dual_branch_cnn() takes at most two, a small one and a
# large one. Singles ascending, then pairs in lexicographic order, because
# that is the order the SOC list was written in: for 3, 9, 15 this returns
# that list exactly, so a grid drawn under a seed before this function
# existed is the grid drawn under the same seed now.
#
# Every pair, including two windows of nearly the same size: whether a second
# branch at a similar scale earns its parameters is what the search is for,
# not something to decide by leaving it out.
.window_options <- function(windows) {
  w <- suppressWarnings(as.integer(windows))
  if (length(w) == 0L || anyNA(w) || any(w != windows) || any(w < 1L) ||
      any(w %% 2L != 1L)) {
    stop("windows must be odd whole numbers (a patch has one centre pixel); ",
         "got ", paste(windows, collapse = ", "), ".", call. = FALSE)
  }
  w <- sort(unique(w))
  singles <- lapply(w, function(z) z)
  pairs <- if (length(w) >= 2L) {
    cmb <- utils::combn(w, 2L)
    lapply(seq_len(ncol(cmb)), function(j) cmb[, j])
  } else list()
  c(singles, pairs)
}

# Batch sizes an epoch can use. The training loader drops the incomplete last
# batch, so an epoch takes floor(n_train / batch) gradient steps -- and a
# batch larger than the fold trains nothing at all (see
# .make_loaders_from_cache()).
#
# FOUR STEPS, AND WHY FOUR. warmup, the LR plateau and early stopping all
# count in epochs; an epoch of one update turns "patience = 60" into sixty
# updates, a schedule that means something else. Four is the fewest the SOC
# grid ever used -- 512 on its ~2,100 training points -- so on that data
# nothing is left out, and a grid drawn under a seed is the grid it was.
#
# When no option gives four, the largest power of two that does, and never
# below two: BatchNorm needs two rows to have a variance.
.batch_options <- function(choices, n_train, min_steps = 4L) {
  if (!is.numeric(n_train) || length(n_train) != 1L || is.na(n_train) || n_train < 2) {
    stop("n_train must be one number >= 2 (the training points of the ",
         "smallest fold); got ", paste(n_train, collapse = ", "), ".", call. = FALSE)
  }
  keep <- choices[floor(n_train / choices) >= min_steps]
  if (length(keep) > 0L) return(keep)
  as.integer(max(2, 2^floor(log2(max(2, n_train / min_steps)))))
}

# A padding the geometry cannot honour collapses to the one it will actually
# get. Each 3x3 convolution without padding removes one pixel per side, so a
# branch needs window > 2 x blocks to survive; "valid_large" applies only to
# the largest window, so it is that one the rule is asked about.
#
# Done in the grid, not only in the model, so that the config table says what
# WILL be built. A grid row that claims a padding the model silently replaces
# is a row whose results cannot be explained by reading it.
.normalise_conv_padding <- function(conv_padding, window_sizes, conv_channels) {
  if (is.null(conv_padding) || is.na(conv_padding[1])) return("same")
  cp <- as.character(conv_padding[1])
  if (identical(cp, "same")) return(cp)
  nb <- length(conv_channels)
  w  <- if (identical(cp, "valid")) min(window_sizes) else max(window_sizes)
  if (w <= 2L * nb) "same" else cp
}

# ── Internal helper ───────────────────────────────────────────────────────────

.rows_to_tibble <- function(rows) {
  scalar_cols <- setdiff(names(rows[[1L]]), c("window_sizes", "conv_channels"))

  scalar_df <- do.call(rbind, lapply(rows, function(r) {
    as.data.frame(r[scalar_cols], stringsAsFactors = FALSE)
  }))

  window_list <- lapply(rows, `[[`, "window_sizes")
  conv_list   <- lapply(rows, `[[`, "conv_channels")

  # Expand the single dropout knob into the five internal dropout sites that
  # build_cnn_from_config() reads. Done per-row to honour each config's value.
  drop_df <- do.call(rbind, lapply(scalar_df$dropout, function(d) {
    as.data.frame(.expand_dropout(d), stringsAsFactors = FALSE)
  }))

  tibble::tibble(
    config_id       = scalar_df$config_id,
    window_sizes    = window_list,
    conv_channels   = conv_list,
    use_residual    = as.logical(scalar_df$use_residual),
    use_se_block    = as.logical(scalar_df$use_se_block),
    se_reduction    = as.integer(scalar_df$se_reduction),
    embedding_dim   = as.integer(scalar_df$embedding_dim),
    embed_pool      = as.character(scalar_df$embed_pool),
    conv_padding    = as.character(scalar_df$conv_padding),
    gate_type       = scalar_df$gate_type,
    dropout         = as.numeric(scalar_df$dropout),
    spatial_dropout = as.numeric(drop_df$spatial_dropout),
    embed_dropout   = as.numeric(drop_df$embed_dropout),
    gate_dropout    = as.numeric(drop_df$gate_dropout),
    head_dropout_1  = as.numeric(drop_df$head_dropout_1),
    head_dropout_2  = as.numeric(drop_df$head_dropout_2),
    base_lr         = as.numeric(scalar_df$base_lr),
    weight_decay    = as.numeric(scalar_df$weight_decay),
    batch_size      = as.integer(scalar_df$batch_size),
    loss_fn         = scalar_df$loss_fn,
    warmup_epochs   = as.integer(scalar_df$warmup_epochs)
  )
}
