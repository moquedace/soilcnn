# Unit test: dual_branch_cnn embed_pool option ("flatten" vs "gap")
#
# Verifies the global-average-pool option behaves and is wired end to end:
#   1. gap produces far fewer params than flatten for large windows
#   2. both modes forward to a valid [N, 1] finite output (single and dual branch)
#   3. make_tune_grid carries an embed_pool column and samples both values
#   4. build_cnn_from_config reads embed_pool from a config row
#   5. backward compatibility: a config WITHOUT embed_pool defaults to "flatten"
#
# Run: source("tests/test_architecture.R")   (no GPU needed; runs on CPU)

suppressMessages({
  library(torch)
  library(purrr)   # pmap_lgl, in the conv_padding block
})

# -- project root: works under source() in the console AND under Rscript ------
# commandArgs("--file=") is empty when the file is source()d, so fall back to
# the frame that source() sets up, then to getwd(). Anchored on a file that
# only exists at the project root, so a wrong guess fails loudly here instead
# of silently sourcing nothing.

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
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "cnn_architecture.R"))
source(file.path(root, "R", "tune_grid.R"))

set.seed(1); torch_manual_seed(1)
n_ch <- 187L; N <- 8L
count_params <- function(m) sum(vapply(m$parameters, function(p) prod(dim(p)), numeric(1)))

# 1 + 2: dual c(3,15) flatten vs gap — params + forward
mk <- function(pool) dual_branch_cnn(
  n_channels = n_ch, window_sizes = c(3L, 15L), conv_channels = c(64L, 128L),
  use_residual = TRUE, use_se_block = TRUE, se_reduction = 16L,
  embedding_dim = 384L, gate_type = "no_gate_concat", embed_pool = pool)
m_flat <- mk("flatten"); m_gap <- mk("gap")
p_flat <- count_params(m_flat); p_gap <- count_params(m_gap)

x3  <- torch_tensor(array(rnorm(N * n_ch * 3  * 3),  dim = c(N, n_ch, 3L,  3L)),  dtype = torch_float())
x15 <- torch_tensor(array(rnorm(N * n_ch * 15 * 15), dim = c(N, n_ch, 15L, 15L)), dtype = torch_float())
m_flat$eval(); m_gap$eval()
o_flat <- as.array(m_flat(x3, x15)); o_gap <- as.array(m_gap(x3, x15))
ok_forward <- identical(dim(o_flat), c(N, 1L)) && identical(dim(o_gap), c(N, 1L)) &&
              all(is.finite(o_flat)) && all(is.finite(o_gap))

# single 15x15 — gap should be dramatically lighter
mk1 <- function(pool) dual_branch_cnn(
  n_channels = n_ch, window_sizes = c(15L), conv_channels = c(64L, 128L),
  use_residual = TRUE, use_se_block = FALSE, embedding_dim = 384L, embed_pool = pool)
s_flat <- count_params(mk1("flatten")); s_gap <- count_params(mk1("gap"))
ok_lighter <- p_gap < p_flat && s_gap < s_flat

# 3: grid carries embed_pool and samples both values
grid <- make_tune_grid(
  tune_length = 6L, seed = 7L,
  fixed = list(window_sizes = list(c(15L)), conv_channels = list(c(64L, 128L)),
               embed_pool = c("flatten", "gap"), use_se_block = FALSE,
               embedding_dim = 256L, gate_type = "no_gate_concat"))
ok_grid <- "embed_pool" %in% names(grid) &&
           all(c("flatten", "gap") %in% grid$embed_pool)

# 4: build_cnn_from_config reads embed_pool
gap_row <- grid[grid$embed_pool == "gap", ][1, ]
ok_cfg <- identical(build_cnn_from_config(gap_row, n_ch)$branch1$embed_pool, "gap")

# 5: backward compatibility — missing column defaults to flatten
gap_row_noPool <- gap_row; gap_row_noPool$embed_pool <- NULL
ok_backcompat <- identical(build_cnn_from_config(gap_row_noPool, n_ch)$branch1$embed_pool, "flatten")

cat(sprintf("  dual c(3,15) params : flatten %s | gap %s (%.1fx lighter)\n",
            format(p_flat, big.mark = ","), format(p_gap, big.mark = ","), p_flat / p_gap))
cat(sprintf("  single 15x15 params : flatten %s | gap %s (%.1fx lighter)\n",
            format(s_flat, big.mark = ","), format(s_gap, big.mark = ","), s_flat / s_gap))

# =============================================================================
# conv_padding
#
# "valid" removes one pixel per side per 3x3 block. The failure modes are all
# silent or late: a linear layer sized from the PATCH instead of the feature
# map blows up on the first forward pass, minutes into a fold; a residual skip
# added without cropping adds the wrong pixels to the wrong place; and a
# padding the geometry cannot honour turns two grid rows into one model under
# two names.
# =============================================================================

mk <- function(ws, cc, pad, residual = TRUE) {
  dual_branch_cnn(n_channels = n_ch, window_sizes = ws, conv_channels = cc,
                  use_residual = residual, use_se_block = TRUE,
                  embedding_dim = 32L, conv_padding = pad)
}

fwd_ok <- function(m, ws) {
  xs <- lapply(ws, function(w)
    torch::torch_randn(c(4L, n_ch, w, w)))
  out <- do.call(m, xs)
  all(dim(out) == c(4L, 1L)) && as.logical(torch::torch_isfinite(out)$all()$item())
}

ok2 <- c()

# 15x15 through 2 blocks under "valid" -> 11x11, and it must still run.
m_valid <- mk(15L, c(16L, 32L), "valid")
ok2["valid_shrinks_as_arithmetic_says"] <- m_valid$branch1$out_size == 11L
ok2["valid_forwards"] <- fwd_ok(m_valid, 15L)

# The same model under "same" keeps 15x15 -- and is therefore HEAVIER, because
# the flatten layer is sized C * out^2.
m_same <- mk(15L, c(16L, 32L), "same")
np <- function(m) sum(vapply(m$parameters, function(p) prod(dim(p)), numeric(1)))
ok2["same_keeps_the_patch_size"] <- m_same$branch1$out_size == 15L
ok2["valid_is_lighter_than_same"] <- np(m_valid) < np(m_same)

# A residual branch has to CROP the skip, or the addition is a shape error.
ok2["valid_forwards_with_residual"]    <- fwd_ok(mk(15L, c(16L, 32L), "valid", TRUE), 15L)
ok2["valid_forwards_without_residual"] <- fwd_ok(mk(15L, c(16L, 32L), "valid", FALSE), 15L)

# valid_large: the 15 branch shrinks, the 3 branch is left alone.
m_vl <- mk(c(3L, 15L), c(16L, 32L), "valid_large")
ok2["valid_large_spares_the_small_branch"] <-
  identical(unname(m_vl$conv_padding_used), c("same", "valid"))
ok2["valid_large_small_branch_keeps_3"]  <- m_vl$branch1$out_size == 3L
ok2["valid_large_large_branch_shrinks"]  <- m_vl$branch2$out_size == 11L
ok2["valid_large_forwards"] <- fwd_ok(m_vl, c(3L, 15L))

# A branch that cannot afford to shrink must SAY so, not fail inside torch with
# a shape error that names no cause. 3x3 through 2 blocks leaves -1.
ok2["impossible_valid_is_refused_at_build"] <- inherits(
  tryCatch(mk(3L, c(16L, 32L), "valid"), error = function(e) e), "error")
ok2["refusal_names_the_constraint"] <- {
  m <- tryCatch(mk(3L, c(16L, 32L), "valid"),
                error = function(e) conditionMessage(e))
  is.character(m) && grepl("window > 2 x blocks", m)
}

# ...and valid_large on a geometry that cannot honour it silently becomes
# "same" IN THE MODEL -- which is exactly why the grid normalises it, so the
# config table says what will actually be built.
ok2["valid_large_falls_back_when_impossible"] <-
  identical(unname(mk(3L, c(16L, 32L), "valid_large")$conv_padding_used), "same")
ok2["grid_normalises_impossible_padding"] <-
  identical(.normalise_conv_padding("valid_large", 3L, c(16L, 32L)), "same")
ok2["grid_keeps_a_padding_that_fits"] <-
  identical(.normalise_conv_padding("valid_large", c(3L, 15L), c(16L, 32L)),
            "valid_large")

# gap is independent of out_size, so it must work under valid too.
ok2["valid_works_with_gap"] <- fwd_ok(
  dual_branch_cnn(n_channels = n_ch, window_sizes = 15L,
                  conv_channels = c(16L, 32L), embedding_dim = 32L,
                  embed_pool = "gap", conv_padding = "valid"), 15L)

# Wiring: the grid carries it, and a config row WITHOUT it still builds the
# model that row used to mean.
g2 <- make_tune_grid(tune_length = 20L, seed = 11L)
ok2["grid_carries_conv_padding"] <- "conv_padding" %in% names(g2)
ok2["grid_samples_more_than_one_padding"] <-
  length(unique(g2$conv_padding)) > 1L
ok2["grid_never_claims_an_impossible_padding"] <- all(
  purrr::pmap_lgl(list(g2$conv_padding, g2$window_sizes, g2$conv_channels),
                  function(cp, ws, cc)
                    identical(cp, .normalise_conv_padding(cp, ws, cc))))

row_nopad <- g2[1, ]; row_nopad$conv_padding <- NULL
ok2["backcompat_defaults_to_same"] <- identical(
  unname(build_cnn_from_config(row_nopad, n_ch)$conv_padding_used[1]), "same")

cat(sprintf("  valid vs same (15x15): %s vs %s params (%.2fx lighter)
",
            format(np(m_valid), big.mark = ","),
            format(np(m_same),  big.mark = ","),
            np(m_same) / np(m_valid)))

results <- c(
  forward_valid          = ok_forward,
  gap_lighter            = ok_lighter,
  grid_carries_pool      = ok_grid,
  build_cnn_reads_pool   = ok_cfg,
  backcompat_flatten     = ok_backcompat,
  ok2
)
.report(results, "test_architecture")
