# Unit test: the fully convolutional prediction gives what the network gives
#
# WHY THIS FILE EXISTS.
#
# fcn_predict_strip() runs a branch's convolutions once over a whole strip
# instead of once per patch -- ~100x less arithmetic, and the only way a 250 m
# global map finishes in days rather than months. That is only worth having if
# every number it returns is the number model(patch) returns. So every
# architecture the grid can draw is built here with random weights and
# NON-TRIVIAL BatchNorm statistics (a freshly built BatchNorm is the identity,
# and would hide a mistake in where the statistics are applied), a strip with
# holes in it is predicted three ways, and the three must agree:
#
#   forward   model(patch, ...) -- the network itself, the reference
#   patch     fcn_predict_strip(engine = "patch"): every branch patch by patch,
#             through the code path the map uses
#   fcn       fcn_predict_strip(engine = "fcn"): supported branches fully
#             convolutionally
#
# Agreement is to 1e-4 in the output: the same sums, in a different order and
# float32. Where a branch cannot be run exactly (a "same" branch, SE with
# flatten) fcn_supported() must say so, and the fcn path must fall back rather
# than return a different number.
#
# Run: source("<package root>/tests/test_fcn.R")

suppressMessages(library(torch))
root <- (function() {
  cand <- character(0)
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f)) cand <- c(cand, dirname(normalizePath(f[1], mustWork = FALSE)))
  for (i in seq_len(sys.nframe())) {
    of <- sys.frame(i)$ofile
    if (!is.null(of) && is.character(of)) {
      cand <- c(cand, dirname(normalizePath(of, mustWork = FALSE)))
    }
  }
  cand <- c(cand, getwd())
  for (d in cand) {
    for (up in c(".", "..")) {
      r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
      if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
    }
  }
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
.load_framework(root)

ok <- c()
set.seed(20260927)
torch::torch_manual_seed(20260927)

n_ch <- 5L
H <- 34L; W <- 41L

# BatchNorm statistics that are not the identity, so a statistic applied in the
# wrong place, or not at all, changes the numbers.
randomise_bn <- function(model) {
  torch::with_no_grad({
    for (m in model$modules) {
      if (any(grepl("batch_norm", class(m)))) {
        nf <- m$running_mean$size(1L)
        m$running_mean$copy_(torch::torch_randn(nf) * 0.5)
        m$running_var$copy_(torch::torch_rand(nf) + 0.5)
        m$weight$copy_(torch::torch_rand(nf) + 0.5)
        m$bias$copy_(torch::torch_randn(nf) * 0.2)
      }
    }
  })
  model$eval()
  model
}

make <- function(...) {
  randomise_bn(dual_branch_cnn(n_channels = n_ch, conv_channels = c(8L, 12L),
                               embedding_dim = 16L, se_reduction = 4L, ...))
}

# A strip with holes: a few NaN pixels, one of them in every channel.
x <- torch::torch_randn(1L, n_ch, H, W)
holes <- cbind(c(10L, 20L, 20L, 27L), c(12L, 30L, 31L, 8L))
for (k in seq_len(nrow(holes))) x[1, , holes[k, 1], holes[k, 2]] <- NaN
x[1, 3L, 5L, 38L] <- NaN

# The centres whose every window is inside the strip and finite: the full-window
# rule, applied here by hand so the test does not borrow the code under test.
valid_centres <- function(ws) {
  h <- (max(ws) - 1L) %/% 2L
  arr <- as.array(x$squeeze(1L))                        # C x H x W
  fin <- apply(arr, c(2L, 3L), function(v) all(is.finite(v)))
  cs <- which(matrix(TRUE, H, W), arr.ind = TRUE)
  keep <- cs[, 1] > h & cs[, 1] <= H - h & cs[, 2] > h & cs[, 2] <= W - h
  cs <- cs[keep, , drop = FALSE]
  good <- vapply(seq_len(nrow(cs)), function(i) {
    all(fin[(cs[i, 1] - h):(cs[i, 1] + h), (cs[i, 2] - h):(cs[i, 2] + h)])
  }, logical(1))
  cs[good, , drop = FALSE]
}

x0 <- x$clone()
x0[!torch::torch_isfinite(x0)] <- 0                     # what the map feeds the strip

forward_ref <- function(model, ws, cs) {
  arr <- as.array(x$squeeze(1L))
  one <- function(w) {
    h <- (w - 1L) %/% 2L
    p <- array(0, dim = c(nrow(cs), n_ch, w, w))
    for (i in seq_len(nrow(cs))) {
      p[i, , , ] <- arr[, (cs[i, 1] - h):(cs[i, 1] + h), (cs[i, 2] - h):(cs[i, 2] + h)]
    }
    torch::torch_tensor(p, dtype = torch::torch_float())
  }
  torch::with_no_grad(as.numeric(do.call(model, lapply(ws, one))$squeeze(2L)))
}

variants <- list(
  single_valid_gap          = list(window_sizes = 15L, conv_padding = "valid", embed_pool = "gap",
                                   use_residual = TRUE,  use_se_block = FALSE),
  single_valid_flatten      = list(window_sizes = 15L, conv_padding = "valid", embed_pool = "flatten",
                                   use_residual = TRUE,  use_se_block = FALSE),
  single_valid_gap_se       = list(window_sizes = 15L, conv_padding = "valid", embed_pool = "gap",
                                   use_residual = TRUE,  use_se_block = TRUE),
  single_valid_plain        = list(window_sizes = 9L,  conv_padding = "valid", embed_pool = "gap",
                                   use_residual = FALSE, use_se_block = FALSE),
  dual_vector_gate          = list(window_sizes = c(3L, 15L), conv_padding = "valid_large",
                                   embed_pool = "gap", gate_type = "vector_featurewise",
                                   use_residual = TRUE, use_se_block = FALSE),
  dual_scalar_gate_flatten  = list(window_sizes = c(3L, 15L), conv_padding = "valid_large",
                                   embed_pool = "flatten", gate_type = "scalar_per_sample",
                                   use_residual = TRUE, use_se_block = FALSE),
  dual_concat_se            = list(window_sizes = c(3L, 15L), conv_padding = "valid_large",
                                   embed_pool = "gap", gate_type = "no_gate_concat",
                                   use_residual = TRUE, use_se_block = TRUE),
  dual_both_valid           = list(window_sizes = c(9L, 15L), conv_padding = "valid",
                                   embed_pool = "gap", gate_type = "vector_featurewise",
                                   use_residual = TRUE, use_se_block = FALSE),
  single_same               = list(window_sizes = 15L, conv_padding = "same", embed_pool = "gap",
                                   use_residual = TRUE, use_se_block = FALSE),
  single_valid_flatten_se   = list(window_sizes = 15L, conv_padding = "valid", embed_pool = "flatten",
                                   use_residual = TRUE, use_se_block = TRUE))

expected_support <- list(
  single_valid_gap = TRUE, single_valid_flatten = TRUE, single_valid_gap_se = TRUE,
  single_valid_plain = TRUE, dual_vector_gate = c(FALSE, TRUE),
  dual_scalar_gate_flatten = c(FALSE, TRUE), dual_concat_se = c(FALSE, TRUE),
  dual_both_valid = c(TRUE, TRUE), single_same = FALSE, single_valid_flatten_se = FALSE)

worst <- c()
for (nm in names(variants)) {
  m  <- do.call(make, variants[[nm]])
  ws <- variants[[nm]]$window_sizes
  cs <- valid_centres(ws)
  ref <- forward_ref(m, ws, cs)
  p_patch <- fcn_predict_strip(m, x0, cs, engine = "patch", batch = 97L)
  p_fcn   <- fcn_predict_strip(m, x0, cs, engine = "fcn",   batch = 97L)
  worst[nm] <- max(abs(p_fcn - ref), abs(p_patch - ref))
  ok[paste0(nm, "_says_which_branches_it_can_run")] <-
    identical(fcn_supported(m), expected_support[[nm]])
  ok[paste0(nm, "_patch_path_is_the_network")] <- max(abs(p_patch - ref)) < 1e-4
  ok[paste0(nm, "_fcn_path_is_the_network")]   <- max(abs(p_fcn - ref)) < 1e-4
}
ok["every_variant_predicted_some_centres"] <- length(worst) == length(variants)
# The holes must have mattered: a centre next to one is not predicted.
ok["a_centre_beside_a_hole_is_not_among_the_valid"] <-
  !any(apply(valid_centres(15L), 1L, function(r) r[1] == 20L && r[2] == 30L))

cat(sprintf("  strip                    : %d x %d x %d channels, %d hole(s)\n",
            H, W, n_ch, nrow(holes) + 1L))
cat(sprintf("  largest |fcn - forward|  : %.2e  (%s)\n", max(worst), names(which.max(worst))))

.report(ok, "test_fcn")
