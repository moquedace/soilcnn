# Unit test: transform_space_loss() reproduces the torch loss exactly
#
# Why this one matters more than it looks. Every epoch, train_one_cnn() picks
# its early-stopping signal from transform_space_loss() — an R reimplementation
# of the torch loss, run on the predictions predict_loader() already collected,
# so the validation set is not pushed through the GPU twice. The docstring
# claims the two are "numerically identical". Nothing verified that claim.
#
# If it drifts (a torch default changes, beta stops being 1.0, reduction stops
# being "mean"), training still runs, the curves still look sane, and every
# model silently stops at the wrong epoch. That is the whole selection
# criterion of the framework resting on an unchecked assertion.
#
# Verifies, for smooth_l1 / mse / mae:
#   1. agreement across residual regimes, including the smooth_l1 kink
#   2. agreement at |d| = 1 exactly (the quadratic/linear boundary)
#   3. agreement on [N, 1] tensors, the shape the training loop actually uses
#   4. agreement on a realistic log1p-SOC scale
#   5. unknown loss names still error instead of returning silently
#
# Run: source("tests/test_transform_loss.R")   (CPU, no GPU needed)

suppressMessages(library(torch))

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
source(file.path(root, "R", "train_cnn.R"))

set.seed(42); torch_manual_seed(42)

torch_loss <- function(name) {
  switch(name,
    smooth_l1 = nn_smooth_l1_loss(),
    mse       = nn_mse_loss(),
    mae       = nn_l1_loss(),
    stop("unknown: ", name)
  )
}

# The training loop feeds float32 tensors and reads predictions back through
# as.numeric(), so compare on values that already made that round trip.
as_f32 <- function(x) as.numeric(as.array(torch_tensor(x, dtype = torch_float())))

# ── residual regimes ──────────────────────────────────────────────────────────
# smooth_l1 is quadratic below |d| = 1 and linear above it, so a test that only
# samples one side proves nothing about the other.

regimes <- list(
  quadratic_only = list(obs = rnorm(2000, 3, 1),   noise = 0.15),
  linear_only    = list(obs = rnorm(2000, 3, 1),   noise = 6.0),
  mixed          = list(obs = rnorm(2000, 3, 1),   noise = 1.0),
  log1p_soc      = list(obs = log1p(rlnorm(2000, 3.4, 0.9)), noise = 0.45)
)

ok <- logical(0)
worst   <- numeric(0)

for (rg in names(regimes)) {
  obs  <- as_f32(regimes[[rg]]$obs)
  pred <- as_f32(obs + rnorm(length(obs), 0, regimes[[rg]]$noise))

  for (fn in c("smooth_l1", "mse", "mae")) {
    r_val <- transform_space_loss(pred, obs, fn)
    t_val <- as.numeric(
      torch_loss(fn)(
        torch_tensor(pred, dtype = torch_float()),
        torch_tensor(obs,  dtype = torch_float())
      )$item()
    )
    rel <- abs(r_val - t_val) / max(abs(t_val), 1e-12)
    key <- paste0(fn, "_", rg)
    ok[key] <- rel < 1e-5
    worst[key]   <- rel
  }
}

# ── the kink: |d| = 1 exactly, plus either side of it ─────────────────────────

obs_k  <- as_f32(rep(0, 7))
pred_k <- as_f32(c(-1.5, -1, -0.5, 0, 0.5, 1, 1.5))
for (fn in c("smooth_l1", "mse", "mae")) {
  r_val <- transform_space_loss(pred_k, obs_k, fn)
  t_val <- as.numeric(
    torch_loss(fn)(
      torch_tensor(pred_k, dtype = torch_float()),
      torch_tensor(obs_k,  dtype = torch_float())
    )$item()
  )
  rel <- abs(r_val - t_val) / max(abs(t_val), 1e-12)
  ok[paste0(fn, "_at_kink")] <- rel < 1e-5
  worst[paste0(fn, "_at_kink")]   <- rel
}

# ── [N, 1] shape, as the training loop actually calls it ──────────────────────
# train_one_cnn() computes loss_fn(pred, y) on column tensors; the R side gets
# plain vectors. Shape must not change the reduction.

obs_c  <- as_f32(rnorm(500, 3, 1))
pred_c <- as_f32(obs_c + rnorm(500, 0, 1))
for (fn in c("smooth_l1", "mse", "mae")) {
  r_val <- transform_space_loss(pred_c, obs_c, fn)
  t_val <- as.numeric(
    torch_loss(fn)(
      torch_tensor(pred_c, dtype = torch_float())$view(c(-1L, 1L)),
      torch_tensor(obs_c,  dtype = torch_float())$view(c(-1L, 1L))
    )$item()
  )
  rel <- abs(r_val - t_val) / max(abs(t_val), 1e-12)
  ok[paste0(fn, "_column_shape")] <- rel < 1e-5
  worst[paste0(fn, "_column_shape")]   <- rel
}

# ── unknown loss name must error, not return silently ─────────────────────────

ok["unknown_loss_errors"] <- inherits(
  try(transform_space_loss(1:3, 1:3, "huber"), silent = TRUE), "try-error"
)
worst["unknown_loss_errors"] <- 0

# ── report ────────────────────────────────────────────────────────────────────

cat(sprintf("  torch build         : %s\n", as.character(utils::packageVersion("torch"))))
cat(sprintf("  worst relative diff : %.3e  (%s)\n",
            max(worst), names(worst)[which.max(worst)]))
cat(sprintf("  tolerance           : 1e-5 relative (float32 round trip)\n"))

.report(
  ok, "test_transform_loss",
  detail = sprintf("  %-28s rel diff %.3e", names(worst)[!ok], worst[!ok])
)
