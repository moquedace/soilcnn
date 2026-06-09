# Unit test: dual_branch_cnn embed_pool option ("flatten" vs "gap")
#
# Verifies the global-average-pool option behaves and is wired end to end:
#   1. gap produces far fewer params than flatten for large windows
#   2. both modes forward to a valid [N, 1] finite output (single and dual branch)
#   3. make_tune_grid carries an embed_pool column and samples both values
#   4. build_cnn_from_config reads embed_pool from a config row
#   5. backward compatibility: a config WITHOUT embed_pool defaults to "flatten"
#
# Run: Rscript tests/test_architecture.R    (no GPU needed; runs on CPU)

suppressMessages(library(torch))

args      <- commandArgs(trailingOnly = FALSE)
file_arg  <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
root      <- normalizePath(file.path(script_dir, ".."))
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

results <- c(
  forward_valid          = ok_forward,
  gap_lighter            = ok_lighter,
  grid_carries_pool      = ok_grid,
  build_cnn_reads_pool   = ok_cfg,
  backcompat_flatten     = ok_backcompat
)
for (nm in names(results)) cat(sprintf("  [%s] %s\n", if (results[nm]) "PASS" else "FAIL", nm))

if (all(results)) { cat("test_architecture: ALL PASS\n") } else {
  cat("test_architecture: FAILURES\n"); quit(status = 1L)
}
