source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("dplyr", "readr", "tibble", "purrr", "terra")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "load_all.R"))

# ══════════════════════════════════════════════════════════════════════════════
# WHERE THE MAP MAY BE BELIEVED
#
# A prediction raster has a value at every pixel, including pixels whose
# predictor combination the model never saw. Nothing in the raster tells the
# two apart, and the cross-validated CCC printed beside the map does not
# describe the second kind at all: it was estimated where training data exists.
#
# Meyer & Pebesma (2021, Methods in Ecology and Evolution 12:1620-1633) make
# this operational, and their recommendation is not that an area of
# applicability is nice to have -- it is that the AOA belongs beside the map.
#
# This script writes two rasters:
#
#   dissimilarity_index.tif   how far each pixel is, in predictor space, from
#                             the training data. 0 = identical to a training
#                             point. 1 = as far as two training points are from
#                             each other on average.
#   aoa_mask.tif              1 where the cross-validated error applies, 0
#                             where it does not.
#
# WHY THIS IS A SEPARATE SCRIPT, AND NOT PART OF 05.
#
# The DI needs only the CENTRE PIXEL of each cell: no patches, no network, no
# torch. Folding it into the prediction loop would have meant touching eight
# places in the one stage that takes longest to run, to add something that can
# be computed independently for a fraction of the cost. Separate, it cannot
# break the map -- and it can be run before the prediction, after it, or not at
# all.
#
# WHAT THIS IS NOT: an uncertainty estimate. Inside the AOA means the predictor
# combination resembles the training data, not that the prediction is accurate.
# Outside it, the cross-validation error simply does not apply, and the honest
# report is "not applicable" rather than a wider interval.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

# ── Which run, and which grid ─────────────────────────────────────────────────

final_run_id <- "latest"   # or a specific "final_YYYYmmdd_HHMMSS"
config_id    <- "auto"     # or a specific config

# The SAME rasters the prediction used. NULL = the ones the patches came from;
# a directory = predict the DI on that grid instead (see 05's own note: it does
# not resample, and the window is counted in pixels).
predict_raster_dir <- NULL
if (nzchar(Sys.getenv("soc_predict_raster_dir"))) {
  predict_raster_dir <- Sys.getenv("soc_predict_raster_dir")
}

# Rows of raster read at once. The DI is an exact nearest-neighbour search, so
# the cost is n_pixels x n_training x n_channels; the chunk bounds the
# temporary matrix, not the total work.
chunk_rows <- 200L

# ── Paths ─────────────────────────────────────────────────────────────────────

base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)

metadata_dir     <- base("outputs", "metadata")
data_dir         <- base("data", "processed")
patch_dir        <- base("outputs", "patches")
final_model_base <- base("outputs", "final_model")
tuning_base      <- base("outputs", "tuning")

if (identical(final_run_id, "latest")) {
  # By time, and only a FINISHED run. The AOA threshold is derived from the
  # cross-validated training data, so it needs a run that actually completed
  # -- an unfinished final_ directory has the fold plan and none of the rest.
  final_run_id <- latest_run_dir(
    final_model_base, prefix = "final_",
    require_file = file.path("comparison", "final_run_summary.rds"),
    label = "final_run_id")
}
final_run_dir <- file.path(final_model_base, final_run_id)

if (identical(config_id, "auto")) {
  fs <- file.path(final_run_dir, "comparison", "final_run_summary.rds")
  if (!file.exists(fs)) stop("Cannot resolve config_id = 'auto': ", fs)
  config_id <- readRDS(fs)$selected_cfgs$config_id[1]
  message("config_id resolved to: ", config_id)
}

# THE SCALING TRAVELS WITH THE WEIGHTS.
#
# Distances are meaningless in unscaled space -- a channel measured in
# thousands dominates one measured in tenths, and the "nearest" training point
# becomes whichever one happens to match the loudest channel. The scaling that
# must be used is the one the MODEL was fitted with, which stage 04 writes next
# to its weights.
scaling_file <- file.path(final_run_dir, config_id, "predictor_scaling.csv")
if (!file.exists(scaling_file)) {
  stop("Model scaling not found: ", scaling_file,
       "\nStage 04 writes it beside the weights. Without it the distances ",
       "would be computed in a different space than the model was trained in.")
}
scaling  <- readr::read_csv2(scaling_file, show_col_types = FALSE)
qc_table <- readr::read_csv2(file.path(metadata_dir, "qc_table.csv"),
                             show_col_types = FALSE)
rtable   <- readr::read_csv2(file.path(metadata_dir, "raster_table_used.csv"),
                             show_col_types = FALSE)

predictor_cols <- rtable$predictor
raster_files   <- rtable$raster_file

if (!is.null(predict_raster_dir)) {
  if (!dir.exists(predict_raster_dir)) {
    stop("predict_raster_dir does not exist: ", predict_raster_dir)
  }
  remapped <- file.path(predict_raster_dir, basename(raster_files))
  absent   <- predictor_cols[!file.exists(remapped)]
  if (length(absent) > 0L) {
    stop("predict_raster_dir is missing ", length(absent), " predictor(s), ",
         "among them: ", paste(utils::head(absent, 8), collapse = ", "))
  }
  raster_files <- remapped
}

# Channel ORDER is the contract: the DI is a distance in a space whose axes are
# the channels, and a permuted axis order gives a perfectly finite, perfectly
# wrong answer.
scaling <- scaling %>%
  dplyr::filter(predictor %in% predictor_cols) %>%
  dplyr::arrange(match(predictor, predictor_cols))
stopifnot(identical(as.character(scaling$predictor), predictor_cols))
stopifnot(identical(as.character(qc_table$predictor), predictor_cols))

output_dir <- base("outputs", "spatial_prediction")
output_dir <- file.path(output_dir, config_id, "aoa")
create_output_dirs(output_dir)

# ── The reference: the training data, in the model's own space ────────────────

points <- readr::read_csv2(file.path(data_dir, "full_modeling_dataset_raw.csv"),
                           show_col_types = FALSE)

# Which rows TRAINED. The threshold is derived from cross-validated distances,
# so it needs the fold each point was in -- taken from the plan the tuning run
# used, which is the plan whose error the AOA is meant to delimit.
plan_file <- file.path(final_run_dir, "fold_plan.rds")
if (!file.exists(plan_file)) {
  # The newest tuning run that HAS a plan, by the plan's own time.
  plan_file <- file.path(
    tuning_base,
    latest_run_dir(tuning_base, prefix = "soc_", require_file = "fold_plan.rds",
                   label = "fold plan source"),
    "fold_plan.rds")
}
plan <- readRDS(plan_file)
message("Fold plan: ", plan_file)
print(plan)

# patch_meta.csv, NOT load_patch_store(): the store's tensors are gigabytes and
# nothing here needs a single patch. What is needed is which points the store
# kept, and in what order, so the fold indices line up with the point rows.
store_meta <- readr::read_csv2(file.path(patch_dir, "patch_meta.csv"),
                               show_col_types = FALSE)
points <- align_points_to_meta(points, store_meta)

# The centre pixel of each training point, QC'd and scaled exactly as the
# raster will be -- one code path, so the two cannot describe different spaces.
train_mat <- as.matrix(points[, predictor_cols, drop = FALSE])
for (i in seq_along(predictor_cols)) {
  train_mat[, i] <- qc_band_values(train_mat[, i], qc_table[i, ])
}
train_mat <- scale_patches_matrix(train_mat, scaling)

# Rows that trained OR validated -- the model saw all of them across the folds,
# and the DI asks about the data the model learned from as a whole.
used <- sort(unique(unlist(lapply(plan$folds, function(f)
  c(f$train, f$validation)))))
fold_of <- integer(nrow(points))
# The validation fold is the meaningful label: it is the fold in which that
# point was HELD OUT, which is exactly the "not in its own fold" the threshold
# is defined against.
for (j in seq_along(plan$folds)) {
  fold_of[plan$folds[[j]]$validation] <- j
}
# A point that never validated -- possible under holdout, or after a buffer
# drop -- still trained, so it gets the first fold whose training set held it.
# Written with an explicit index because `fold_of[idx][cond] <- j` assigns to a
# COPY of the subset and changes nothing, silently.
for (j in seq_along(plan$folds)) {
  idx  <- plan$folds[[j]]$train
  todo <- idx[fold_of[idx] == 0L]
  if (length(todo)) fold_of[todo] <- j
}
used <- used[fold_of[used] > 0L]
used <- used[is.finite(rowSums(train_mat[used, , drop = FALSE]))]
if (length(used) < 2L) {
  stop("Fewer than 2 usable training rows after QC. Nothing can be measured ",
       "against them.", call. = FALSE)
}

ref <- di_reference(train_mat[used, , drop = FALSE])
th  <- aoa_threshold(ref, fold_of[used])

message(sprintf(
  "\nReference: %d training rows x %d channels | mean pairwise distance %.4f",
  ref$n, ref$p, ref$avg_dist))
message(sprintf("AOA threshold (DI): %.4f", as.numeric(th)))

# ── Stream the raster ─────────────────────────────────────────────────────────

rast_stack <- terra::rast(raster_files)
names(rast_stack) <- predictor_cols
r_nrow <- terra::nrow(rast_stack); r_ncol <- terra::ncol(rast_stack)

message(sprintf("\nGrid: %d x %d = %s cells at %.8f per pixel",
                r_nrow, r_ncol, format(as.numeric(r_nrow) * r_ncol,
                                       big.mark = ","),
                terra::res(rast_stack)[1]))
message(sprintf("Cost: exact nearest-neighbour over %d training rows x %d ",
                ref$n, ref$p),
        "channels per pixel.")

# WRITTEN AS IT IS COMPUTED, never held whole.
#
# Assigning into an in-memory raster is fine at 20 km (2.3M cells) and fatal at
# 250 m (15 billion). terra's writeStart/writeValues/writeStop streams row
# blocks straight to the file, so the peak memory is one chunk whatever the
# grid is -- and the 250 m case is the one this project actually wants.
f_di   <- file.path(output_dir, "dissimilarity_index.tif")
f_mask <- file.path(output_dir, "aoa_mask.tif")

di_r   <- terra::rast(rast_stack[[1]]); names(di_r)   <- "dissimilarity_index"
mask_r <- terra::rast(rast_stack[[1]]); names(mask_r) <- "aoa"

terra::writeStart(di_r, f_di, overwrite = TRUE,
                  gdal = c("COMPRESS=DEFLATE", "PREDICTOR=3", "TILED=YES"))
terra::writeStart(mask_r, f_mask, overwrite = TRUE, datatype = "INT1U",
                  gdal = c("COMPRESS=DEFLATE", "PREDICTOR=2", "TILED=YES"))

starts <- seq(1L, r_nrow, by = chunk_rows)
t0 <- Sys.time()
n_inside <- 0; n_valid <- 0

for (ci in seq_along(starts)) {
  rs <- starts[ci]
  nr <- min(chunk_rows, r_nrow - rs + 1L)

  m <- terra::values(rast_stack, row = rs, nrows = nr)
  m <- matrix(as.numeric(m), ncol = length(predictor_cols))
  for (i in seq_along(predictor_cols)) {
    m[, i] <- qc_band_values(m[, i], qc_table[i, ])
  }
  m <- scale_patches_matrix(m, scaling)

  keep <- is.finite(rowSums(m))
  di   <- rep(NA_real_, nrow(m))
  if (any(keep)) {
    di[keep] <- dissimilarity_index(ref, m[keep, , drop = FALSE])
    n_valid  <- n_valid + sum(keep)
    n_inside <- n_inside + sum(di[keep] <= as.numeric(th))
  }

  terra::writeValues(di_r,   di,                                rs, nr)
  terra::writeValues(mask_r, as.integer(di <= as.numeric(th)),  rs, nr)

  if (ci %% 10L == 0L || ci == length(starts)) {
    el <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
    message(sprintf("  chunk %d/%d | %.1f min elapsed | ETA %.1f min",
                    ci, length(starts), el,
                    el / ci * (length(starts) - ci)))
  }
}

terra::writeStop(di_r)
terra::writeStop(mask_r)

safe_write_csv2(tibble::tibble(
  final_run_id   = final_run_id,
  config_id      = config_id,
  fold_plan      = plan_file,
  n_training     = ref$n,
  n_channels     = ref$p,
  avg_pair_dist  = ref$avg_dist,
  di_threshold   = as.numeric(th),
  k_iqr          = attr(th, "k_iqr"),
  cells_valid    = n_valid,
  cells_inside   = n_inside,
  pct_inside     = 100 * n_inside / max(1, n_valid),
  raster_dir     = if (is.null(predict_raster_dir)) "training grid"
                   else predict_raster_dir,
  computed_at    = as.character(Sys.time())
), file.path(output_dir, "aoa_summary.csv"))

message("\n", strrep("=", 78))
message(sprintf("AOA: %s of %s valid cells inside (%.1f%%)",
                format(n_inside, big.mark = ","),
                format(n_valid, big.mark = ","),
                100 * n_inside / max(1, n_valid)))
message(strrep("=", 78))
if (n_inside / max(1, n_valid) < 0.5) {
  message(
    "LESS THAN HALF THE MAP IS INSIDE THE AOA.\n",
    "The cross-validated error does not describe the rest: those pixels hold\n",
    "predictor combinations the model never met. Publish the map WITH this\n",
    "mask, not the headline metric alone.")
}
message("\nRasters: ", output_dir)
