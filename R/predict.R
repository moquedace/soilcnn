# ── Prediction over a raster: the network run once per strip, not once per pixel
#
# WHY THIS EXISTS.
#
# Stage 05 predicts a pixel by cutting the w x w patch around it and passing
# the patch through the network. The pixel beside it has a patch that shares
# all but one of its columns, and the network redoes every sum of it again. On
# the deployed SOC model (one 15 x 15 branch, 181 channels, 10 seeds) the
# dev run measured 164 valid pixels per second; a global map at 250 m has
# ~2.3 billion valid pixels. That is four months, and the disk was not the
# limit -- 23 s of reading against 2,184 s of network (docs/project_log.md,
# 2026-09-27).
#
# FULLY CONVOLUTIONAL, AND EXACT WHERE IT CAN BE.
#
# A branch whose convolutions use NO padding ("valid") is exactly
# translation-equivariant: a 3 x 3 valid convolution, an eval-mode BatchNorm
# (an affine map per channel), the activation, and the residual shortcut
# (a 1 x 1 convolution, centre-cropped by the same pixel per side) all compute
# at every position what they compute inside a patch. So the blocks are run
# ONCE over a whole strip of the raster, and what is left per pixel is cheap:
#
#   "gap"      the patch's mean feature is a moving average of the strip's
#              feature map -- avg_pool2d with stride 1
#   "flatten"  the linear layer over the patch's C x o x o features is a
#              convolution with that layer's weight, reshaped to (K, C, o, o)
#   SE + gap   the SE weight is constant over the patch, so it commutes with
#              the mean: pooled = s(e) * e, e the moving average -- exact
#
# and then the embedding's BatchNorm, the gate and the head, per pixel. About
# 200k multiply-adds per pixel per seed instead of 27M: ~100x.
#
# WHERE IT IS NOT EXACT, AND WHAT HAPPENS THERE. A "same" branch pads EACH
# PATCH with zeros at its own border; over a strip those positions see real
# neighbours instead, so the numbers would differ. SE with "flatten" does not
# commute with the linear layer. Those branches are run patch by patch --
# which for the usual case, the small 3 x 3 branch of a valid_large model,
# costs little. fcn_supported() says which branch takes which path.
#
# NON-FINITE INPUT. The full-window rule already discards a pixel whose window
# holds any non-finite value. Over a strip, those values are replaced by 0
# before the convolutions -- a NaN would otherwise spread through whatever
# algorithm torch picks for the convolution -- and only the discarded pixels'
# windows ever contain one.

#' Which branches of a network can be run fully convolutionally, exactly.
#'
#' @param model A dual_branch_cnn.
#' @return A logical per branch.
fcn_supported <- function(model) {
  br <- if (model$n_branches == 1L) list(model$branch1) else list(model$branch1, model$branch2)
  vapply(br, function(b) {
    identical(b$conv_padding, "valid") &&
      (!isTRUE(b$use_se_block) || identical(b$embed_pool, "gap"))
  }, logical(1))
}

# The patches of `centres` from a strip tensor, (n, C, w, w). centres is a
# 2-column integer matrix of (row, col) in STRIP coordinates, 1-based; every
# centre's window must lie inside the strip.
.fcn_gather_patches <- function(x_strip, centres, w) {
  C <- x_strip$size(2L); H <- x_strip$size(3L); W <- x_strip$size(4L)
  h <- (w - 1L) %/% 2L
  n <- nrow(centres)
  offs <- expand.grid(dc = (-h):h, dr = (-h):h)          # row-major within a patch
  base <- (centres[, 1] - 1L) * W + centres[, 2]
  cell <- as.vector(t(outer(base, offs$dr * W + offs$dc, "+")))
  x_flat <- x_strip$view(c(C, H * W))
  x_flat[, cell]$view(c(C, n, w, w))$permute(c(2L, 1L, 3L, 4L))$contiguous()
}

# One branch's embedding for every centre, patch by patch.
.fcn_branch_patchwise <- function(branch, x_strip, centres, w, batch) {
  n <- nrow(centres)
  out <- vector("list", ceiling(n / batch))
  k <- 0L
  for (s in seq(1L, n, by = batch)) {
    e <- min(n, s + batch - 1L)
    k <- k + 1L
    out[[k]] <- branch(.fcn_gather_patches(x_strip, centres[s:e, , drop = FALSE], w))
  }
  torch::torch_cat(out, dim = 1L)
}

# One branch's embedding for every centre, fully convolutionally. The map
# position (a, b) of the pooled features is the patch whose top-left corner is
# strip pixel (a, b), i.e. the centre (a + h, b + h).
.fcn_branch_convolutional <- function(branch, x_strip, centres, w) {
  h <- (w - 1L) %/% 2L
  f <- x_strip
  for (i in seq_along(branch$blocks)) f <- branch$blocks[[i]](f)
  o <- branch$out_size
  C <- f$size(2L)
  if (identical(branch$embed_pool, "gap")) {
    m <- torch::nnf_avg_pool2d(f, kernel_size = c(o, o), stride = c(1L, 1L))
    B <- m$size(4L)
    e <- m$permute(c(1L, 3L, 4L, 2L))$reshape(c(-1L, C))
    e <- e[(centres[, 1] - h - 1L) * B + (centres[, 2] - h), ]
    if (isTRUE(branch$use_se_block)) {
      s <- torch::torch_sigmoid(branch$se$fc2(branch$se$act(branch$se$fc1(e))))
      e <- e * s
    }
    z <- branch$linear(e)
  } else {
    K <- branch$linear$weight$size(1L)
    m <- torch::nnf_conv2d(f, branch$linear$weight$view(c(K, C, o, o)),
                           bias = branch$linear$bias)
    B <- m$size(4L)
    z <- m$permute(c(1L, 3L, 4L, 2L))$reshape(c(-1L, K))
    z <- z[(centres[, 1] - h - 1L) * B + (centres[, 2] - h), ]
  }
  branch$drop_emb(branch$act_emb(branch$bn_emb(z)))
}

# The gate and the head, on embeddings -- dual_branch_cnn$forward(), after the
# branches. Kept in step with it by the test that compares the two paths.
.fcn_head <- function(model, f1, f2 = NULL) {
  if (model$n_branches == 1L) return(model$head(f1))
  abs_dif <- torch::torch_abs(f1 - f2)
  head_in <- if (identical(model$gate_type, "no_gate_concat")) {
    torch::torch_cat(list(f1, f2), dim = 2L)
  } else {
    gate  <- model$gate_net(torch::torch_cat(list(f1, f2, abs_dif, f1 * f2), dim = 2L))
    fused <- gate * f1 + (1 - gate) * f2
    torch::torch_cat(list(fused, abs_dif), dim = 2L)
  }
  model$head(head_in)
}

#' Predict the centres of a strip, in the transformed space.
#'
#' @param model   A dual_branch_cnn, in eval mode.
#' @param x_strip Tensor (1, C, H, W): the strip, QC'd and scaled, with every
#'   non-finite value replaced by 0.
#' @param centres Integer matrix (n x 2) of (row, col) in strip coordinates,
#'   1-based; every window of every branch must lie inside the strip.
#' @param engine  "fcn" runs each supported branch fully convolutionally and
#'   the rest patch by patch; "patch" runs every branch patch by patch.
#' @param batch   Centres per batch on the patch-by-patch path and the head.
#' @return A numeric vector, one prediction per centre.
fcn_predict_strip <- function(model, x_strip, centres, engine = c("fcn", "patch"),
                              batch = 4096L) {
  engine <- match.arg(engine)
  if (nrow(centres) == 0L) return(numeric(0))
  windows <- vapply(seq_len(model$n_branches), function(b) {
    br <- if (b == 1L) model$branch1 else model$branch2
    as.integer(if (identical(br$conv_padding, "valid")) br$out_size + 2L * length(br$blocks)
               else br$out_size)
  }, integer(1))
  supported <- fcn_supported(model)
  torch::with_no_grad({
    emb <- lapply(seq_len(model$n_branches), function(b) {
      br <- if (b == 1L) model$branch1 else model$branch2
      if (identical(engine, "fcn") && supported[b]) {
        .fcn_branch_convolutional(br, x_strip, centres, windows[b])
      } else {
        .fcn_branch_patchwise(br, x_strip, centres, windows[b], batch)
      }
    })
    n <- nrow(centres)
    out <- numeric(n)
    for (s in seq(1L, n, by = batch)) {
      e <- min(n, s + batch - 1L)
      f1 <- emb[[1]][s:e, ]
      f2 <- if (model$n_branches == 2L) emb[[2]][s:e, ] else NULL
      out[s:e] <- as.numeric(.fcn_head(model, f1, f2)$squeeze(2L)$to(device = "cpu"))
    }
  })
  out
}
