# ══════════════════════════════════════════════════════════════════════════════
# The whole framework, in one page.
#
# This is what using it looks like once the patches exist. The numbered scripts
# in examples/soc_stock_0_5cm/ do the same things with every knob exposed and
# every decision commented -- they are the worked example. This is the tour.
#
# WHAT IS DELIBERATELY NOT HERE: no source() list, no manual alignment of
# points to the store, no raster opened to read a resolution, no buffer
# arithmetic, no block size copied from another run. Each of those was a step
# someone had to get right, and each has cost this project a run.
# ══════════════════════════════════════════════════════════════════════════════

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/cnn_architecture.R.
project_root <- (function() {
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
  for (d in cand) for (up in c(".", "..", "../..", "../../..")) {
    r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
    if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
pkgload::load_all(project_root)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)

# ── 1. Load ───────────────────────────────────────────────────────────────────
#
# One call: opens the store, reads the points and the predictor types, aligns
# them, reads the raster resolution, and REFUSES if the store was built under a
# different predictor set, window set, target or resolution.

data <- dsm_load(
  patch_dir    = base("outputs", "patches"),
  points       = file.path(base("data", "processed"), "full_modeling_dataset_raw.csv"),
  type_table   = file.path(base("outputs", "metadata"), "predictor_type_table.csv"),
  raster_table = file.path(base("outputs", "metadata"), "raster_table_used.csv"),
  windows      = c(3L, 9L, 15L)
)

# ── 2. Decide who trains and who scores ───────────────────────────────────────
#
# This one line is the most consequential in the whole pipeline, which is why
# it is one line. Swap it and nothing else changes:
#
#   spatial_cv(k = 5)                        blocks of ground, buffered
#   random_cv(k = 10)                        ignores geography, on purpose
#   holdout_cv(validation_frac = 0.2)        a single split
#   region_cv(group = data$points$biome)     leave-one-region-out
#
# "auto" means MEASURED, not guessed: the block size comes from how these
# points are actually spread, and the buffer from the window and the
# resolution. Both print what they chose.

cv <- spatial_cv(k = 5, block_size = "auto", buffer = "auto", test_frac = 0.15)

# Worth looking at before spending a night on it.
plan <- resolve_resampling(cv, data)
print(plan)

# ── 3. Fit ────────────────────────────────────────────────────────────────────
#
# tune_length is a budget, not a lattice -- the same meaning caret gives it.
# n_seeds is what turns "A beat B" into a claim with an error bar.

fit <- dsm_train(
  data,
  model       = "cnn",
  resampling  = plan,          # or `cv` directly; the plan is resolved either way
  tune_length = 30,
  n_seeds     = 3,
  transform   = expm1,         # the target was trained on log1p
  output_dir  = base("outputs", "tuning"),
  n_epochs    = 500L,
  patience    = 60L
)

fit$by_config                            # mean +/- sd, one row per config
print_noise_floor(seed_noise_floor(fit$comparison))
print_one_se(one_se(fit$by_config))      # the simplest config within 1 SE

# ── 4. The baselines that give the number a scale ─────────────────────────────
#
# THE SAME PLAN, so the comparison is between models rather than between
# experiments. The gap between rf_context and the CNN is what the convolution
# is worth; if it is smaller than the noise floor, the convolution is doing
# averaging.

rf_ctx <- dsm_train(data, model = "rf", resampling = plan,
                    features = c("centre", "window_mean"),
                    tune_length = 4, n_seeds = 3, transform = expm1,
                    output_dir = base("outputs", "tuning"))

rf_pt  <- dsm_train(data, model = "rf", resampling = plan,
                    features = "centre",
                    tune_length = 4, n_seeds = 3, transform = expm1,
                    output_dir = base("outputs", "tuning"))

# Borrowing a fourth family from caret costs one line -- caret carries the
# parameter names, the grid generator and the fit/predict for ~230 methods.
# register_model(caret_spec("xgbTree"), overwrite = TRUE)
# xgb <- dsm_train(data, model = "xgbTree", resampling = plan, ...)

# ── 5. Where the map may be believed ──────────────────────────────────────────
#
# A map has a value at every pixel, including pixels whose predictor
# combination the model never saw, and nothing in the raster tells them apart.
# The cross-validated CCC does not describe those. Meyer & Pebesma (2021): the
# area of applicability belongs beside the map, not in a footnote.

fold_cache <- build_fold_cache(data$store, data$points, data$type_table,
                              plan$folds[[1]], data$store$window_sizes)
tab <- fold_table_view(fold_cache$cache, data$store$predictors,
                       features = "centre")
ref <- di_reference(tab$train$x)
th  <- aoa_threshold(ref, rep_len(seq_len(plan$n_folds), nrow(tab$train$x)))

# At prediction time the same reference is applied chunk by chunk:
#   di <- dissimilarity_index(ref, chunk_of_scaled_centre_pixels)
#   inside <- inside_aoa(di, th)
print_aoa(dissimilarity_index(ref, tab$validation$x), th, "validation points")
