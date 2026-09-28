# Unit test: argument validation and the clamp contract
#
# Covers the step-2 corrections, all of which share one property: they are
# meant to change no number on the existing pipeline, only to stop the
# framework from failing silently on inputs it was never told to accept.
#
# Verifies:
#   1. clamp is honoured on both ends, and c(-Inf, Inf) really lets negatives
#      through -- the old hardcoded pmax(., 0) made a negative-valued target
#      impossible to model, with plausible-looking metrics
#   2. clamp defaults to c(0, Inf), preserving the previous behaviour exactly
#   3. a malformed clamp is rejected up front
#   4. an unknown gate_type errors at CONSTRUCTION, not at forward time
#   5. an unknown embed_pool errors instead of silently becoming "flatten"
#   6. a single-branch config still ignores gate_type (backward compatible)
#   7. build_cnn_from_config() on a config with no embed_pool column defaults
#      to "flatten" WITHOUT emitting a tibble warning
#
# Run: source("D:/.../tests/test_validation.R")     (CPU, no GPU needed)

suppressMessages({
  library(torch)
  library(tibble)
})

# -- project root: works under source() in the console AND under Rscript ------

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
.load_framework(root)

set.seed(7); torch_manual_seed(7)
ok <- logical(0)

# ── a tiny but real inference setup ───────────────────────────────────────────
# Small single-branch model over 8 synthetic samples. The model is untrained,
# so its raw output is arbitrary -- the transform below is what makes each
# assertion deterministic, independent of whatever the weights happen to give.

n_s <- 8L; n_ch <- 3L; w <- 3L
device <- torch_device("cpu")

x  <- torch_tensor(array(rnorm(n_s * n_ch * w * w), dim = c(n_s, n_ch, w, w)),
                   dtype = torch_float())
y  <- torch_tensor(rnorm(n_s), dtype = torch_float())$view(c(-1L, 1L))
dl <- dataloader(tensor_dataset(x, y), batch_size = 4L, shuffle = FALSE)

model <- dual_branch_cnn(
  n_channels = n_ch, window_sizes = c(3L), conv_channels = c(8L, 16L),
  use_residual = TRUE, use_se_block = FALSE, embedding_dim = 16L,
  embed_pool = "gap"
)

pv <- tibble(
  profile_id       = as.character(seq_len(n_s)),
  sample_id        = seq_len(n_s),
  target_native    = as.numeric(as.array(y$view(-1L))),
  target_transform = as.numeric(as.array(y$view(-1L)))
)

run <- function(transform, ...) {
  predict_loader(model, dl, pv, "test", transform = transform,
                 device = device, ...)$pred
}

# ── 1-2: clamp on both ends, and the default ─────────────────────────────────

# shift far below zero: with the default lower bound everything must be 0,
# and with c(-Inf, Inf) everything must stay negative
push_down <- function(z) z - 1000
ok["clamp_default_floors_at_zero"] <- all(run(push_down) == 0)
ok["clamp_open_lets_negatives_through"] <-
  all(run(push_down, clamp = c(-Inf, Inf)) < 0)

# shift far above: an upper bound must bite, and Inf must not
push_up <- function(z) z + 1000
ok["clamp_upper_bound_applied"] <- all(run(push_up, clamp = c(-Inf, 5)) == 5)
ok["clamp_upper_inf_leaves_values"] <- all(run(push_up) > 900)

# the default must reproduce the old pmax(., 0) exactly
ok["clamp_default_equals_old_pmax"] <-
  identical(run(identity), pmax(run(identity, clamp = c(-Inf, Inf)), 0))

# ── 3: malformed clamp rejected ──────────────────────────────────────────────

bad_clamp <- function(cl) {
  inherits(try(run(identity, clamp = cl), silent = TRUE), "try-error")
}
ok["clamp_rejects_reversed"]  <- bad_clamp(c(10, 0))
ok["clamp_rejects_wrong_len"] <- bad_clamp(0)
ok["clamp_rejects_na"]        <- bad_clamp(c(NA, Inf))

# ── 4-6: option sets validated at construction ───────────────────────────────

build <- function(...) {
  try(dual_branch_cnn(n_channels = n_ch, conv_channels = c(8L, 16L),
                      embedding_dim = 16L, use_se_block = FALSE, ...),
      silent = TRUE)
}

ok["bad_gate_type_errors"] <-
  inherits(build(window_sizes = c(3L, 5L), gate_type = "vetor_featurewise"),
           "try-error")
ok["bad_embed_pool_errors"] <-
  inherits(build(window_sizes = c(3L), embed_pool = "GAP"), "try-error")
ok["good_options_still_build"] <-
  !inherits(build(window_sizes = c(3L, 5L), gate_type = "scalar_per_sample",
                  embed_pool = "gap"), "try-error")

# a single branch ignores gate_type entirely -- old configs carrying a
# placeholder there must keep working
ok["single_branch_ignores_gate_type"] <-
  !inherits(build(window_sizes = c(3L), gate_type = "whatever"), "try-error")

# ── 7: missing embed_pool column defaults, and stays quiet ───────────────────

grid <- make_tune_grid(
  tune_length = 1L, seed = 3L,
  fixed = list(window_sizes = list(c(3L)), conv_channels = list(c(8L, 16L)),
               embedding_dim = 16L, use_se_block = FALSE, embed_pool = "gap")
)
row_no_pool <- grid[1, ]
row_no_pool$embed_pool <- NULL

warned <- FALSE
m_back <- withCallingHandlers(
  build_cnn_from_config(row_no_pool, n_ch),
  warning = function(cnd) { warned <<- TRUE; invokeRestart("muffleWarning") }
)
ok["missing_embed_pool_defaults_flatten"] <-
  identical(m_back$branch1$embed_pool, "flatten")
ok["missing_embed_pool_is_silent"] <- !warned

# ── report ────────────────────────────────────────────────────────────────────

cat(sprintf("  inference setup     : %d samples, %d channels, %dx%d window\n",
            n_s, n_ch, w, w))
.report(ok, "test_validation")
