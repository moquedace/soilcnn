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

# ── valid option sets ─────────────────────────────────────────────────────────
# Checked at CONSTRUCTION, not at forward time. An unknown gate_type used to
# fall through switch() to NULL, skip building gate_net, and only blow up in
# forward() -- after the model was built and, in a grid, minutes of training
# later. An unknown embed_pool was worse: identical(embed_pool, "gap") sent
# every typo silently down the "flatten" path.

.valid_gate_types  <- c("vector_featurewise", "scalar_per_sample",
                        "no_gate_concat")
.valid_conv_paddings       <- c("same", "valid")
# The model level has a third: apply "valid" only where a branch can afford it.
.valid_conv_paddings_model <- c("same", "valid", "valid_large")
.valid_embed_pools <- c("flatten", "gap")

.check_choice <- function(value, choices, what) {
  if (length(value) != 1L || is.na(value) || !value %in% choices) {
    stop(what, " must be one of: ", paste(choices, collapse = ", "),
         " -- got ", if (length(value) == 1L) paste0("'", value, "'") else
                     paste0("length ", length(value)),
         call. = FALSE)
  }
  as.character(value)
}

# ── padding: what the border of a patch is worth ──────────────────────────────
#
# padding = 1 ("same") keeps the spatial size, by inventing a ring of zeros
# around the patch. padding = 0 ("valid") uses only real data and loses one
# pixel on each side per 3x3 conv: after b blocks a w x w patch is (w - 2b).
#
# THE CASE FOR "valid": every value the network sees is a measurement. With
# "same", a 3x3 patch through two blocks has a receptive field of 5x5 at the
# centre -- larger than the patch -- so EVERY output position depends on
# invented zeros, and the network spends capacity learning the shape of its own
# border.
#
# THE CASE AGAINST: it shrinks, and a small window cannot afford to. w - 2b < 1
# is not a model. That is why this is a PER-BRANCH decision resolved by
# dual_branch_cnn(), not a flag the whole model carries.
#
# The redundancy argument is the reason the large branch is where the gain is:
# a w x w patch re-reads each pixel w^2 times across the dataset, so the 15
# branch is 225x redundant and the 3 branch only 9x. Trading border pixels for
# honest ones is nearly free on the first and expensive on the second.

# ── conv_block ────────────────────────────────────────────────────────────────
# A plain convolutional block: Conv2d → BatchNorm → Activation.
# padding = 1 preserves the spatial size; padding = 0 shrinks it by 2.

conv_block <- torch::nn_module(
  initialize = function(in_ch, out_ch, padding = 1L) {
    self$conv <- torch::nn_conv2d(in_ch, out_ch, kernel_size = 3,
                                  padding = as.integer(padding))
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
  initialize = function(in_ch, out_ch, padding = 1L) {
    self$conv <- torch::nn_conv2d(in_ch, out_ch, kernel_size = 3,
                                  padding = as.integer(padding))
    self$bn   <- torch::nn_batch_norm2d(out_ch)
    self$act  <- make_activation()
    # With padding = 0 the main path is SMALLER than the input, so the skip has
    # to be cropped to match before the addition -- a 1x1 conv only fixes the
    # channels, never the spatial size. Done in forward(), where the shrinkage
    # is known.
    self$pad_used <- as.integer(padding)
    self$shortcut <- if (in_ch != out_ch) {
      torch::nn_sequential(
        torch::nn_conv2d(in_ch, out_ch, kernel_size = 1),
        torch::nn_batch_norm2d(out_ch)
      )
    } else {
      torch::nn_identity()
    }
  },
  forward = function(x) {
    out <- self$bn(self$conv(x))
    sc  <- self$shortcut(x)
    if (self$pad_used == 0L) {
      # Centre crop: the main path lost one pixel on each side, so the skip
      # must contribute the co-located values, not the corner ones.
      d <- (sc$size(3L) - out$size(3L)) %/% 2L
      if (d > 0L) {
        sc <- sc[, , (d + 1L):(sc$size(3L) - d), (d + 1L):(sc$size(4L) - d)]
      }
    }
    self$act(out + sc)
  }
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
                        embed_dropout   = 0.0,
                        embed_pool      = "flatten",
                        conv_padding    = "same") {

    embed_pool   <- .check_choice(embed_pool, .valid_embed_pools, "embed_pool")
    conv_padding <- .check_choice(conv_padding, .valid_conv_paddings,
                                  "conv_padding")

    n_blocks   <- length(conv_channels)
    pad        <- if (identical(conv_padding, "valid")) 0L else 1L

    # The output size after every block. With "valid" this SHRINKS, and a
    # branch asked for a size it cannot produce must say so here rather than
    # fail inside torch with a shape error that names no cause.
    out_size <- if (pad == 0L) window_size - 2L * n_blocks else window_size
    if (out_size < 1L) {
      stop("conv_padding = 'valid' with a ", window_size, "x", window_size,
           " window and ", n_blocks, " conv block(s) leaves ", out_size,
           "x", out_size, " -- there is nothing left to pool.\n",
           "Each 3x3 convolution without padding removes one pixel per side, ",
           "so a valid branch needs window > 2 x blocks. Use 'same' for this ",
           "branch, fewer blocks, or a wider window.", call. = FALSE)
    }
    self$conv_padding <- conv_padding
    self$out_size     <- out_size

    block_fn   <- if (use_residual) residual_conv_block else conv_block
    in_channels <- c(n_channels, conv_channels[-length(conv_channels)])

    # Build conv blocks as a module list
    blocks <- vector("list", n_blocks)
    for (i in seq_len(n_blocks)) {
      blocks[[i]] <- block_fn(in_channels[i], conv_channels[i], padding = pad)
    }
    self$blocks <- torch::nn_module_list(blocks)

    self$use_se_block    <- use_se_block
    self$spatial_dropout <- torch::nn_dropout2d(p = spatial_dropout)

    if (use_se_block) {
      self$se <- se_block(channels = conv_channels[n_blocks], reduction = se_reduction)
    }

    # embed_pool controls how the C×w×w feature map is reduced before the linear
    # projection to embedding_dim:
    #   "flatten" — keep every cell: input size = C · w · w. Preserves the full
    #     spatial detail of the patch, but the linear layer grows with w², so a
    #     large window concentrates most of the model's parameters here.
    #   "gap"     — global average pool to C: input size = C, independent of w.
    #     Drops the w² blow-up entirely, making large windows light and far less
    #     prone to overfitting (at the cost of within-patch spatial resolution).
    # Both branches of a dual model use the same choice. Default "flatten" keeps
    # the original behaviour; tuning can compare the two.
    self$embed_pool <- embed_pool
    # out_size, not window_size: under "valid" the feature map is smaller than
    # the patch, and sizing the linear layer from the patch is a shape error
    # raised on the first forward pass -- after the fold cache has been built.
    embed_in_size <- if (identical(embed_pool, "gap")) {
      conv_channels[n_blocks]                        # GAP → C, any out_size
    } else {
      conv_channels[n_blocks] * out_size * out_size  # flatten → C·out·out
    }

    self$flatten <- torch::nn_flatten()
    self$linear  <- torch::nn_linear(embed_in_size, embedding_dim)
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
    if (identical(self$embed_pool, "gap")) {
      x <- torch::nnf_adaptive_avg_pool2d(x, output_size = c(1L, 1L))
      x <- x$view(c(x$size(1L), x$size(2L)))   # N×C
    } else {
      x <- self$flatten(x)                      # N×(C·w·w)
    }
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
    head_dropout_2  = 0.10,
    embed_pool      = "flatten", # "flatten" (C·w·w) or "gap" (C). See cnn_branch.
    conv_padding    = "same"     # "same", "valid", or "valid_large"
  ) {

    n_branches <- length(window_sizes)
    if (!n_branches %in% c(1L, 2L)) {
      stop("window_sizes must name one window (single branch) or two (dual ",
           "branch) -- got ", n_branches, ": ", paste(window_sizes, collapse = ", "),
           ".", call. = FALSE)
    }

    embed_pool <- .check_choice(embed_pool, .valid_embed_pools, "embed_pool")
    # gate_type is ignored for a single branch, so only validate when it is
    # actually going to be used -- an old single-window config may carry any
    # placeholder there.
    if (n_branches == 2L) {
      gate_type <- .check_choice(gate_type, .valid_gate_types, "gate_type")
    }

    self$n_branches <- n_branches
    self$gate_type  <- gate_type

    # PADDING IS RESOLVED PER BRANCH, and the resolution is recorded.
    #
    # "valid_large" is the useful setting and the reason this is not a single
    # flag: a 3x3 branch through 2 blocks has nothing left under "valid", while
    # a 15x15 branch loses 4 pixels of 15 and keeps only measured values. The
    # option applies the honest padding where it fits and leaves the small
    # branch alone.
    #
    # Resolved here rather than by a silent fallback inside cnn_branch: a
    # branch that quietly ignores what it was asked for is how a grid comes to
    # contain two configs that are the same model under different names.
    conv_padding <- .check_choice(conv_padding, .valid_conv_paddings_model,
                                  "conv_padding")
    n_blocks <- length(conv_channels)
    pad_for  <- function(w) {
      switch(conv_padding,
        same        = "same",
        valid       = "valid",
        valid_large = if (w > 2L * n_blocks &&
                          (n_branches == 1L || w == max(window_sizes)))
                        "valid" else "same")
    }
    self$conv_padding      <- conv_padding
    self$conv_padding_used <- vapply(window_sizes, pad_for, character(1))

    branch_args <- list(
      n_channels    = n_channels,
      conv_channels = conv_channels,
      use_residual  = use_residual,
      use_se_block  = use_se_block,
      se_reduction  = se_reduction,
      embedding_dim = embedding_dim,
      spatial_dropout = spatial_dropout,
      embed_dropout = embed_dropout,
      embed_pool    = embed_pool
    )

    self$branch1 <- do.call(cnn_branch, c(branch_args,
      list(window_size = window_sizes[1],
           conv_padding = self$conv_padding_used[1])))

    if (n_branches == 2L) {
      self$branch2 <- do.call(cnn_branch, c(branch_args,
        list(window_size = window_sizes[2],
             conv_padding = self$conv_padding_used[2])))

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

#' How many learnable parameters a config would build.
#'
#' The natural measure of "simpler" for one_se(): between two models that
#' cannot be told apart, the one with fewer parameters is the one to keep.
#' It orders the way the domain expects -- gap below flatten, small windows
#' below large, fewer blocks below more -- without anyone having to hand-write
#' that ordering.
#'
#' Building the module to count it costs milliseconds against minutes of
#' training, and it is the only way to be right: the count depends on the
#' window, the pooling and the channel widths together.
#'
#' @param cfg        One row of a tune grid.
#' @param n_channels Number of predictor channels.
#' @noRd
count_model_params <- function(cfg, n_channels) {
  m <- build_cnn_from_config(cfg, n_channels)
  n <- sum(vapply(m$parameters, function(p) prod(dim(p)), numeric(1)))
  rm(m); invisible(gc(verbose = FALSE))
  n
}

# ── Constructor helper ────────────────────────────────────────────────────────
# Builds a dual_branch_cnn from a single named config list (as produced by
# make_tune_grid), making it easy to loop over grid rows.

build_cnn_from_config <- function(cfg, n_channels) {
  # embed_pool is a newer field; tolerate older grids/configs that lack it
  # (a model trained before this option existed used "flatten"). Use [[ ]] and
  # a names() check rather than $: on a tibble, cfg$missing_col returns NULL
  # *and* emits "Unknown or uninitialised column", which is noise on a path
  # that is deliberately optional.
  # A field that did not exist when a grid was written must default to what
  # that grid MEANT, never to what is fashionable now: "flatten" and "same" are
  # the behaviours those configs actually had.
  opt <- function(field, default) {
    v <- if (field %in% names(cfg)) cfg[[field]] else NULL
    if (is.null(v) || length(v) == 0L || is.na(v[1])) default else as.character(v[1])
  }
  embed_pool   <- opt("embed_pool",   "flatten")
  conv_padding <- opt("conv_padding", "same")

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
    head_dropout_2  = cfg$head_dropout_2,
    embed_pool      = embed_pool,
    conv_padding    = conv_padding
  )
}
