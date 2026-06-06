# CNN architecture for spatial regression with raster patches
#
# Design rationale
# ────────────────
# Each soil profile has N raster layers (predictors) extracted as a spatial
# patch (e.g., 3×3 or 5×5 cells). The CNN learns spatial patterns within that
# patch across all predictor channels simultaneously.
#
# The dual-branch design captures two spatial scales:
#   • Small branch (e.g., 3×3): local-scale processes (micro-relief, land cover)
#   • Large branch (e.g., 5×5 or 7×7): landscape-scale processes (climate, geology)
# A learned gate fuses the two branch embeddings, letting each sample draw
# from whichever scale is more informative for that location.
#
# Building blocks:
#   conv_block          – single conv + BN + activation
#   residual_conv_block – conv_block with skip connection (ResNet-style)
#   se_block            – Squeeze-and-Excitation channel attention
#   cnn_branch          – full branch: stack of blocks → SE → flatten → embedding
#   dual_branch_cnn     – two branches + gate + regression head

source(here::here("R", "utils.R"))

# ── conv_block ────────────────────────────────────────────────────────────────
# A plain convolutional block: Conv2d → BatchNorm → Activation.
# kernel_size = 3, padding = 1 preserves spatial dimensions.

conv_block <- torch::nn_module(
  initialize = function(in_ch, out_ch) {
    self$conv <- torch::nn_conv2d(in_ch, out_ch, kernel_size = 3, padding = 1)
    self$bn   <- torch::nn_batch_norm2d(out_ch)
    self$act  <- make_activation()
  },
  forward = function(x) self$act(self$bn(self$conv(x)))
)

# ── residual_conv_block ───────────────────────────────────────────────────────
# Same as conv_block but adds a skip connection from input to output.
# When in_ch ≠ out_ch, a 1×1 conv aligns channels before adding.
# Benefit: gradients flow back through the skip path even if the main
# path saturates, making deeper networks easier to train (He et al., 2016).

residual_conv_block <- torch::nn_module(
  initialize = function(in_ch, out_ch) {
    self$conv <- torch::nn_conv2d(in_ch, out_ch, kernel_size = 3, padding = 1)
    self$bn   <- torch::nn_batch_norm2d(out_ch)
    self$act  <- make_activation()
    self$shortcut <- if (in_ch != out_ch) {
      torch::nn_sequential(
        torch::nn_conv2d(in_ch, out_ch, kernel_size = 1),
        torch::nn_batch_norm2d(out_ch)
      )
    } else {
      torch::nn_identity()
    }
  },
  forward = function(x) self$act(self$bn(self$conv(x)) + self$shortcut(x))
)

# ── se_block ──────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation (Hu et al., 2018): learns a per-channel weight
# by pooling spatial info (squeeze) and then passing through a small MLP
# (excitation). The output weights recalibrate each channel of the feature map.
#
# In soil mapping context: with 200+ predictors, SE helps the network
# learn which predictor groups (climate, vegetation, soil texture…) matter
# most for a given spatial context.
#
# reduction: bottleneck ratio. Larger = smaller MLP = fewer parameters.
#   Common values: 8, 16 (default), 32.

se_block <- torch::nn_module(
  initialize = function(channels, reduction = 16) {
    hidden <- max(4L, as.integer(channels %/% reduction))
    self$fc1 <- torch::nn_linear(channels, hidden)
    self$act  <- make_activation()
    self$fc2  <- torch::nn_linear(hidden, channels)
  },
  forward = function(x) {
    # Global average pool: N×C×H×W → N×C
    s <- torch::nnf_adaptive_avg_pool2d(x, output_size = c(1L, 1L))
    s <- s$view(c(s$size(1L), s$size(2L)))
    s <- self$fc2(self$act(self$fc1(s)))
    s <- torch::torch_sigmoid(s)$view(c(s$size(1L), s$size(2L), 1L, 1L))
    x * s
  }
)

# ── cnn_branch ────────────────────────────────────────────────────────────────
# One branch of the dual-branch CNN.
#
# Parameters
# ----------
# n_channels     : number of input predictor channels (raster layers)
# window_size    : spatial size of the input patch (e.g., 3, 5, or 7)
# conv_channels  : integer vector, one entry per conv block
#                  e.g., c(64, 128) → 2 blocks (64 filters, then 128)
# use_residual   : add skip connections to each conv block
# use_se_block   : add SE channel attention after last conv block
# se_reduction   : SE bottleneck ratio (see se_block)
# embedding_dim  : size of the output embedding vector per sample
# spatial_dropout: dropout rate applied to 2D feature maps (drops whole channels)
# embed_dropout  : dropout rate applied after the embedding linear layer

cnn_branch <- torch::nn_module(
  initialize = function(n_channels, window_size, conv_channels,
                        use_residual    = TRUE,
                        use_se_block    = TRUE,
                        se_reduction    = 16L,
                        embedding_dim   = 256L,
                        spatial_dropout = 0.03,
                        embed_dropout   = 0.0) {

    n_blocks   <- length(conv_channels)
    block_fn   <- if (use_residual) residual_conv_block else conv_block
    in_channels <- c(n_channels, conv_channels[-length(conv_channels)])

    # Build conv blocks as a module list
    blocks <- vector("list", n_blocks)
    for (i in seq_len(n_blocks)) {
      blocks[[i]] <- block_fn(in_channels[i], conv_channels[i])
    }
    self$blocks <- torch::nn_module_list(blocks)

    self$use_se_block    <- use_se_block
    self$spatial_dropout <- torch::nn_dropout2d(p = spatial_dropout)

    if (use_se_block) {
      self$se <- se_block(channels = conv_channels[n_blocks], reduction = se_reduction)
    }

    flatten_size <- conv_channels[n_blocks] * window_size * window_size

    self$flatten <- torch::nn_flatten()
    self$linear  <- torch::nn_linear(flatten_size, embedding_dim)
    self$bn_emb  <- torch::nn_batch_norm1d(embedding_dim)
    self$act_emb <- make_activation()
    self$drop_emb <- torch::nn_dropout(p = embed_dropout)
  },

  forward = function(x) {
    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](x)
    }
    if (self$use_se_block) x <- self$se(x)
    x <- self$spatial_dropout(x)
    x <- self$flatten(x)
    x <- self$drop_emb(self$act_emb(self$bn_emb(self$linear(x))))
    x
  }
)

# ── dual_branch_cnn ───────────────────────────────────────────────────────────
# Full model: two branches (or one) + fusion gate + regression head.
#
# Gate types
# ----------
# "vector_featurewise"  – gate is a vector of embedding_dim values,
#   one gate weight per embedding dimension. Computed from the concatenation
#   of [f_small, f_large, |f_small - f_large|, f_small * f_large].
#   The absolute difference and product capture how much the two scales
#   agree and interact. Most expressive option.
#
# "scalar_per_sample"   – gate is a single scalar per sample (not per feature).
#   Simpler: the whole sample gets one weight deciding how much to favour
#   the small-scale branch.
#
# "no_gate_concat"      – no gating; both embeddings are concatenated directly.
#   Equivalent to letting the head learn the fusion implicitly.
#
# Head input
# ----------
# For vector and scalar gates:  [fused_embedding, abs_difference] → 2 × embed_dim
#   The absolute difference is kept as an extra signal: it captures locations
#   where small and large scales disagree, which may itself be informative
#   (e.g., a wetland surrounded by drier landscape).
# For no_gate_concat:           [f_small, f_large] → 2 × embed_dim
# For single branch:            [f_branch] → embed_dim
#
# n_branches = 1 disables the gate entirely and uses window_sizes[1].

dual_branch_cnn <- torch::nn_module(

  initialize = function(
    n_channels,
    window_sizes    = c(3L, 5L),   # length 1 → single branch; length 2 → dual
    conv_channels   = c(64L, 128L),
    use_residual    = TRUE,
    use_se_block    = TRUE,
    se_reduction    = 16L,
    embedding_dim   = 256L,
    gate_type       = "vector_featurewise",  # ignored when n_branches = 1
    spatial_dropout = 0.03,
    embed_dropout   = 0.0,
    gate_dropout    = 0.10,
    head_dropout_1  = 0.20,
    head_dropout_2  = 0.10
  ) {

    n_branches <- length(window_sizes)
    if (!n_branches %in% c(1L, 2L)) stop("window_sizes must have length 1 or 2.")

    self$n_branches <- n_branches
    self$gate_type  <- gate_type

    branch_args <- list(
      n_channels    = n_channels,
      conv_channels = conv_channels,
      use_residual  = use_residual,
      use_se_block  = use_se_block,
      se_reduction  = se_reduction,
      embedding_dim = embedding_dim,
      spatial_dropout = spatial_dropout,
      embed_dropout = embed_dropout
    )

    self$branch1 <- do.call(cnn_branch, c(branch_args, list(window_size = window_sizes[1])))

    if (n_branches == 2L) {
      self$branch2 <- do.call(cnn_branch, c(branch_args, list(window_size = window_sizes[2])))

      gate_in_dim <- switch(gate_type,
        vector_featurewise = embedding_dim * 4L,
        scalar_per_sample  = embedding_dim * 4L,
        no_gate_concat     = NULL
      )

      if (!is.null(gate_in_dim)) {
        gate_out_dim <- if (gate_type == "vector_featurewise") embedding_dim else 1L
        self$gate_net <- torch::nn_sequential(
          torch::nn_linear(gate_in_dim, embedding_dim),
          torch::nn_batch_norm1d(embedding_dim),
          make_activation(),
          torch::nn_dropout(p = gate_dropout),
          torch::nn_linear(embedding_dim, gate_out_dim),
          torch::nn_sigmoid()
        )
      }

      head_in_dim <- embedding_dim * 2L   # fused + abs_diff (or concat)
    } else {
      head_in_dim <- embedding_dim
    }

    self$head <- torch::nn_sequential(
      torch::nn_linear(head_in_dim, embedding_dim),
      torch::nn_batch_norm1d(embedding_dim),
      make_activation(),
      torch::nn_dropout(p = head_dropout_1),
      torch::nn_linear(embedding_dim, 128L),
      torch::nn_batch_norm1d(128L),
      make_activation(),
      torch::nn_dropout(p = head_dropout_2),
      torch::nn_linear(128L, 64L),
      make_activation(),
      torch::nn_linear(64L, 1L)
    )
  },

  forward = function(...) {
    inputs <- list(...)
    f1     <- self$branch1(inputs[[1]])

    if (self$n_branches == 1L) {
      return(self$head(f1))
    }

    f2      <- self$branch2(inputs[[2]])
    abs_dif <- torch::torch_abs(f1 - f2)

    if (self$gate_type == "no_gate_concat") {
      head_in <- torch::torch_cat(list(f1, f2), dim = 2L)
    } else {
      gate_in  <- torch::torch_cat(list(f1, f2, abs_dif, f1 * f2), dim = 2L)
      gate      <- self$gate_net(gate_in)
      fused     <- gate * f1 + (1 - gate) * f2
      head_in   <- torch::torch_cat(list(fused, abs_dif), dim = 2L)
    }

    self$head(head_in)
  },

  # Extended forward: returns intermediate tensors for gate analysis.
  forward_with_internals = function(...) {
    inputs <- list(...)
    f1     <- self$branch1(inputs[[1]])

    if (self$n_branches == 1L) {
      pred <- self$head(f1)
      return(list(pred = pred, f1 = f1))
    }

    f2      <- self$branch2(inputs[[2]])
    abs_dif <- torch::torch_abs(f1 - f2)

    if (self$gate_type == "no_gate_concat") {
      gate    <- NULL
      fused   <- NULL
      head_in <- torch::torch_cat(list(f1, f2), dim = 2L)
    } else {
      gate_in <- torch::torch_cat(list(f1, f2, abs_dif, f1 * f2), dim = 2L)
      gate    <- self$gate_net(gate_in)
      fused   <- gate * f1 + (1 - gate) * f2
      head_in <- torch::torch_cat(list(fused, abs_dif), dim = 2L)
    }

    list(
      pred    = self$head(head_in),
      f1      = f1,
      f2      = f2,
      gate    = gate,
      fused   = fused,
      abs_dif = abs_dif
    )
  }
)

# ── Constructor helper ────────────────────────────────────────────────────────
# Builds a dual_branch_cnn from a single named config list (as produced by
# make_tune_grid), making it easy to loop over grid rows.

build_cnn_from_config <- function(cfg, n_channels) {
  dual_branch_cnn(
    n_channels      = n_channels,
    window_sizes    = cfg$window_sizes[[1]],   # stored as a list-column
    conv_channels   = cfg$conv_channels[[1]],
    use_residual    = cfg$use_residual,
    use_se_block    = cfg$use_se_block,
    se_reduction    = cfg$se_reduction,
    embedding_dim   = cfg$embedding_dim,
    gate_type       = cfg$gate_type,
    spatial_dropout = cfg$spatial_dropout,
    embed_dropout   = cfg$embed_dropout,
    gate_dropout    = cfg$gate_dropout,
    head_dropout_1  = cfg$head_dropout_1,
    head_dropout_2  = cfg$head_dropout_2
  )
}
