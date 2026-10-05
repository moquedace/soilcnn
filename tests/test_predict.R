# Unit test: dsm_predict() -- the map is the network's prediction at every
# pixel, and every band is what its calibration says
#
# WHY THIS FILE EXISTS.
#
# dsm_predict() is written for a 250 m global grid: row bands read through a
# buffer that keeps its halo, workers side by side, the fully convolutional
# network over chunks cropped to their valid pixels, the DI in float32 with
# the nearest profile's distance recomputed in double. Each of those is a
# place a pixel can quietly get another pixel's number. So the whole chain --
# dsm_prepare(), dsm_train(), dsm_final(), dsm_predict() -- is run on a small
# grid, and every band at every pixel is compared with a computation that
# shares none of the map's code:
#
#   valid_mask   the full-window rule, by hand
#   ensemble_*   each seed's network on each pixel's own patches, by hand
#   di, aoa      aoa_di() on the pixel's raw values
#   intervals    conformal_*_interval() on the map's median and DI
#   smeared      smear() on the map's median
#
# and then the properties the global run leans on: one worker and two give
# identical maps, bit for bit; a part is the same numbers as that part of the
# whole (and the patch engine gives what the convolutional one gives); a
# resumed map recomputes no finished unit; a lost record is recomputed; and
# what can only be wrong is refused -- including a raster table whose channels
# are swapped, which only the probe can see.
#
# The fixture: a 40 x 56 grid at 0.1 degree, 5 bands (NA holes, a QC
# sentinel, a clamped percentage, a dummy), 81 profiles in 9 sites 11+ cells
# apart -- each site inside one 1-degree block, so the spatial folds cut
# between sites and the 0.7-degree buffer drops nobody. One dual-branch config
# (3 x 3 "same", 7 x 7 "valid": one branch per engine), 2 epochs.
#
# Run: source("<package root>/tests/test_predict.R")
# (trains and starts worker processes; ~2-4 min on CPU)

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)
  library(readr)
})

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
err <- function(expr) {
  e <- tryCatch({ suppressMessages(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(e)) "" else e
}

# ── the fixture: rasters, profiles, a store, a tuning run, a final model ─────
base <- file.path(tempdir(), "dlc_predict_test")
unlink(base, recursive = TRUE)          # on the way IN: a leftover would be resumed
rdir <- file.path(base, "rasters")
dir.create(rdir, recursive = TRUE)

set.seed(20260927)
n_r <- 40L; n_c <- 56L; cs <- 0.1; x0 <- -50; y1 <- -10
rc <- expand.grid(c = seq_len(n_c), r = seq_len(n_r))    # row-major, as terra stores
cell <- function(r, c) (r - 1L) * n_c + c
mk <- function(name, vals) {
  r <- terra::rast(nrows = n_r, ncols = n_c, xmin = x0, xmax = x0 + n_c * cs,
                   ymin = y1 - n_r * cs, ymax = y1, crs = "EPSG:4326")
  terra::values(r) <- vals
  terra::writeRaster(r, file.path(rdir, paste0(name, ".tif")), overwrite = TRUE,
                     datatype = "FLT4S")
}
v_a <- sin(rc$r / 5) + cos(rc$c / 7) + stats::rnorm(nrow(rc), sd = 0.1)
v_b <- (rc$r * rc$c) / 500 + stats::rnorm(nrow(rc), sd = 0.2)
v_b[c(cell(12L, 15L), cell(25L, 32L), cell(25L, 33L), cell(26L, 32L), cell(26L, 33L))] <- NA
v_pct <- pmin(104, pmax(-2, 50 + 55 * sin(rc$c / 9) + stats::rnorm(nrow(rc), sd = 5)))
v_dum <- as.numeric((rc$r + 2L * rc$c) %% 3L == 0L)
v_tmp <- 20 + rc$r / 10 + stats::rnorm(nrow(rc), sd = 0.3)
v_tmp[c(cell(28L, 50L), cell(3L, 30L))] <- -9999
mk("band_a", v_a); mk("band_b", v_b); mk("pct_cover", v_pct)
mk("dummy_x", v_dum); mk("temp_c", v_tmp)

site_rc <- expand.grid(sc = c(8L, 24L, 40L), sr = c(7L, 20L, 33L))
prc <- do.call(rbind, lapply(seq_len(nrow(site_rc)), function(s)
  expand.grid(c = site_rc$sc[s] + (-1L:1L), r = site_rc$sr[s] + (-1L:1L))))
pts <- data.frame(profile_id = sprintf("p%03d", seq_len(nrow(prc))),
                  x = x0 + (prc$c - 0.5) * cs, y = y1 - (prc$r - 0.5) * cs,
                  soc = exp(2 + 0.5 * v_a[cell(prc$r, prc$c)] + stats::rnorm(nrow(prc), sd = 0.3)))

st <- suppressMessages(dsm_prepare(
  points = pts, target = "soc", raster_dir = rdir, windows = c(3, 7),
  out_dir = file.path(base, "prep"), percentage = "^pct_",
  na_below = c("temp_c$" = -100), transform = "log1p", target_min = 0,
  n_cores = 1L, verbose = FALSE))
data <- suppressMessages(dsm_load(st, verbose = FALSE))

grid <- make_manual_tune_grid(
  window_sizes = list(c(3L, 7L)), conv_channels = list(c(4L, 6L)), embedding_dim = 8L,
  base_lr = 0.01, batch_size = 8L, dropout = 0.0, gate_type = "vector_featurewise",
  use_residual = TRUE, use_se_block = FALSE, conv_padding = "valid_large",
  embed_pool = "gap")
fit <- suppressMessages(dsm_train(
  data, model = "cnn",
  resampling = spatial_cv(k = 2L, block_size = 1, buffer = "auto", test_frac = 0.2),
  tune_grid = grid, n_seeds = 1L, output_dir = file.path(base, "out"), run_id = "tuning",
  device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
  n_epochs = 2L, patience = 2L, print_every = 100L, augment = FALSE, verbose = FALSE))
cid <- grid$config_id[1]
fin <- suppressMessages(dsm_final(
  fit, config = cid, seeds = 2L, validation_frac = 0.34, threads_per_unit = 1L,
  n_cores = 1L, training = list(n_epochs = 2L, patience = 2L, print_every = 100L,
                                augment = FALSE),
  output_dir = file.path(base, "out", "final_model"), run_id = "final", verbose = FALSE))

map_args <- list(fin, data, calibration = c(block = fit$run_dir), n_cores = 1L,
                 threads_per_worker = 1L, step_rows = 16L, unit_rows = 16L,
                 chunk_cols = 20L, verbose = FALSE)
mp <- function(...) suppressMessages(do.call(dsm_predict, utils::modifyList(map_args, list(...))))
rd <- function(m, b) terra::as.matrix(terra::rast(m$vrt[[b]]), wide = TRUE)

# ── 1. the map, two workers, the probe first ──────────────────────────────────
# THE BUDGET IS GIVEN. Left to itself it is 70% of the RAM free at that moment,
# and with another job on the machine (the full SOC run, 2026-10-03) it held one
# worker of the ~6 GB the estimate floors every map worker at, not two -- and
# "three units on two workers" failed for the machine's reason, not the
# code's. 13 GB holds two by the estimate; the fixture uses a fraction of it.
map2 <- mp(n_cores = 2L, run_id = "map_two", max_ram_gb = 13)
bands_all <- c("ensemble_median", "ensemble_mean", "ensemble_sd", "ensemble_mad",
               "ensemble_min", "ensemble_max", "smeared_mean_block",
               "pi90_cv_constant_lower_block", "pi90_cv_constant_upper_block",
               "pi90_cv_level_di_lower_block", "pi90_cv_level_di_upper_block", "di",
               "aoa_block", "valid_mask")
ok["every_band_is_written_with_its_mosaic"] <-
  identical(map2$bands$band, bands_all) && all(file.exists(map2$vrt))
ok["three_units_on_two_workers"] <- nrow(map2$units) == 3L && identical(map2$n_workers, 2L)
ok["one_branch_per_engine"] <- identical(map2$engine, c("patch", "fcn"))
# A worker makes its step window once and hands the same tensor to every unit
# of that shape (.predict_window()): what one call wrote, the next one sees.
e_w <- list(bufs = new.env(parent = emptyenv()))
w1 <- .predict_window(e_w, 5L, 22L, 62L)
w1[1, 1, 1] <- 7
w2 <- .predict_window(e_w, 5L, 22L, 62L)
w3 <- .predict_window(e_w, 5L, 38L, 62L)
ok["a_worker_makes_its_window_once_per_shape"] <- as.numeric(w2[1, 1, 1]) == 7 &&
  all(w3$size() == c(5, 38, 62)) && as.numeric(w3[1, 1, 1]) == 0
rm(e_w, w1, w2, w3)
ok["the_probe_reproduced_the_stored_predictions_at_the_profiles"] <-
  identical(map2$probe$status, "pass") && map2$probe$n >= 9L && map2$probe$max_rel_diff < 1e-5
snap <- stats::setNames(lapply(bands_all, function(b) rd(map2, b)), bands_all)

# ── 2. every band at every pixel, against a computation that shares no code ──
preds <- data$store$predictors
C <- length(preds)
rt <- safe_read_csv2(file.path(st$store_dir, "raster_table_used.csv"))
qc <- safe_read_csv2(file.path(st$store_dir, "qc_table.csv"))
qc <- qc[match(preds, qc$predictor), , drop = FALSE]
sc <- safe_read_csv2(file.path(fin$run_dir, cid, "predictor_scaling.csv"))
sc <- sc[match(preds, sc$predictor), , drop = FALSE]
raw <- array(NA_real_, c(n_r, n_c, C))
for (k in seq_len(C)) {
  raw[, , k] <- terra::as.matrix(terra::rast(rt$raster_file[rt$predictor == preds[k]]), wide = TRUE)
}
scl <- raw
for (k in seq_len(C)) {
  scl[, , k] <- (matrix(qc_band_values(as.vector(raw[, , k]), qc[k, , drop = FALSE]), n_r, n_c) -
                   sc$center[k]) / sc$scale[k]
}
fin_all <- apply(is.finite(scl), c(1L, 2L), all)
h <- 3L
valid_ref <- matrix(FALSE, n_r, n_c)
for (r in (h + 1L):(n_r - h)) for (c in (h + 1L):(n_c - h)) {
  valid_ref[r, c] <- all(fin_all[(r - h):(r + h), (c - h):(c + h)])
}
ok["the_valid_mask_is_the_full_window_rule"] <- all((snap$valid_mask == 1) == valid_ref)
ok["the_holes_and_the_margin_mattered"] <- sum(valid_ref) < (n_r - 2L * h) * (n_c - 2L * h) &&
  !valid_ref[12L, 15L] && !valid_ref[26L, 33L]

cells <- which(valid_ref, arr.ind = TRUE)
patch <- function(w) {
  hw <- (w - 1L) %/% 2L
  a <- array(0, dim = c(nrow(cells), C, w, w))
  for (i in seq_len(nrow(cells))) {
    rr <- cells[i, 1]; cc <- cells[i, 2]
    a[i, , , ] <- aperm(scl[(rr - hw):(rr + hw), (cc - hw):(cc + hw), , drop = FALSE], c(3L, 1L, 2L))
  }
  torch::torch_tensor(a, dtype = torch::torch_float32())
}
p3 <- patch(3L); p7 <- patch(7L)
cfg <- fin$selected[fin$selected$config_id == cid, , drop = FALSE]
pt <- sapply(fin$seeds, function(s) {
  m <- build_cnn_from_config(cfg, C)
  m$load_state_dict(torch::torch_load(file.path(fin$run_dir, cid, "models", sprintf("seed%04d_best.pt", s))))
  m$eval()
  torch::with_no_grad(as.numeric(m(p3, p7)$squeeze(2L)))
})
nat <- pmax(expm1(pt), 0)
ref_stats <- list(ensemble_median = matrixStats::rowMedians(nat), ensemble_mean = rowMeans(nat),
                  ensemble_sd = matrixStats::rowSds(nat), ensemble_mad = matrixStats::rowMads(nat),
                  ensemble_min = matrixStats::rowMins(nat), ensemble_max = matrixStats::rowMaxs(nat))
rel <- function(a, b) max(abs(a - b) / (1 + abs(b)))
worst_ens <- vapply(names(ref_stats), function(b) rel(snap[[b]][cells], ref_stats[[b]]), numeric(1))
ok["every_ensemble_band_is_the_seeds_networks_at_every_pixel"] <- all(worst_ens < 1e-4)
ok["outside_the_mask_every_value_band_is_na"] <-
  all(vapply(setdiff(bands_all, "valid_mask"), function(b) all(is.na(snap[[b]][!valid_ref])), logical(1)))

cal <- readRDS(file.path(map2$run_dir, "calibration.rds"))
src <- cal$sources$block
rawc <- t(vapply(seq_len(nrow(cells)), function(i) raw[cells[i, 1], cells[i, 2], ], numeric(C)))
di_ref <- aoa_di(src$aref, rawc)
med <- snap$ensemble_median[cells]
di  <- snap$di[cells]
ok["the_di_is_aoa_di_at_every_pixel"] <- max(abs(di - di_ref)) < 1e-5
ok["the_aoa_is_the_di_against_the_threshold"] <-
  identical(as.integer(snap$aoa_block[cells]), as.integer(di_ref <= src$threshold))
iv  <- src$intervals$pi90
ivc <- conformal_interval(iv$constant, med, lower_limit = 0)
ivl <- conformal_scaled_interval(iv$level_di, med, data.frame(level = med, di = di), lower_limit = 0)
ok["the_constant_interval_is_the_median_plus_minus_q"] <-
  rel(snap$pi90_cv_constant_lower_block[cells], ivc$lower) < 1e-5 &&
  rel(snap$pi90_cv_constant_upper_block[cells], ivc$upper) < 1e-5
ok["the_level_di_interval_is_the_fitted_scale_at_every_pixel"] <-
  rel(snap$pi90_cv_level_di_lower_block[cells], ivl$lower) < 1e-5 &&
  rel(snap$pi90_cv_level_di_upper_block[cells], ivl$upper) < 1e-5
ok["the_smeared_mean_is_duans_factor_on_the_median"] <-
  rel(snap$smeared_mean_block[cells], smear(log1p(med), src$smearing, lower_limit = 0)) < 1e-5
ok["the_record_says_what_each_band_is"] <-
  file.exists(file.path(map2$run_dir, "bands.csv")) &&
  file.exists(file.path(map2$run_dir, "calibration.csv")) &&
  file.exists(file.path(map2$run_dir, "prediction_manifest.csv")) &&
  all(nzchar(map2$bands$meaning))

# ── 3. one worker and two: identical, bit for bit ─────────────────────────────
map1 <- mp(run_id = "map_one", probe = FALSE)
ok["one_worker_and_two_give_identical_maps"] <-
  all(vapply(bands_all, function(b) identical(rd(map1, b), snap[[b]]), logical(1)))

# Every map above has one step a unit. Here the first unit has two: its second
# step's halo is the first step's last rows, moved to the top of the window,
# and the window the first unit made is the second unit's too. The same rows
# go through the network in the same strips, so the numbers are the same.
map_steps <- mp(run_id = "map_steps", unit_rows = 32L, probe = FALSE)
ok["units_of_two_steps_give_identical_maps"] <- nrow(map_steps$units) == 2L &&
  all(vapply(bands_all, function(b) identical(rd(map_steps, b), snap[[b]]), logical(1)))

# A worker whose working set is over its limit after a unit exits, and a fresh
# one takes its place. Under a limit every worker is over, each unit gets a
# process of its own -- two restarts for three units -- and the map is the
# same.
old_rc <- options(dsm.predict.recycle_gb = 0.001)
map_rc <- mp(run_id = "map_recycled", probe = FALSE)
options(old_rc)
ok["a_worker_over_its_memory_gives_way_to_a_fresh_one"] <-
  identical(as.integer(map_rc$manifest$worker_restarts), 2L) &&
  all(vapply(bands_all, function(b) identical(rd(map_rc, b), snap[[b]]), logical(1)))
# Each unit was mapped by a process of its own, carries the working set its
# process kept after it, and each of the three exits left its note beside the
# unit it followed -- the record a resumed map keeps.
ok["each_unit_names_its_process_and_each_exit_leaves_a_note"] <-
  length(unique(map_rc$units$pid)) == 3L && !anyNA(map_rc$units$pid) &&
  all(is.finite(map_rc$units$rss_gb)) && all(map_rc$units$rss_gb > 0) &&
  (!identical(.Platform$OS.type, "windows") || isTRUE(all(map_rc$units$private_gb > 0))) &&
  length(Sys.glob(file.path(map_rc$run_dir, "units", "*", "recycled.rds"))) == 3L

# ── 4. a part of the map, by the other engine ─────────────────────────────────
part <- mp(run_id = "map_part_patch", engine = "patch", probe = FALSE,
           extent = list(rows = c(9L, 30L), cols = c(6L, 44L)))
pm <- rd(part, "ensemble_median"); wm <- snap$ensemble_median[9:30, 6:44]
pd <- rd(part, "di"); wd <- snap$di[9:30, 6:44]
ok["a_part_is_that_part_of_the_whole_and_the_engines_agree"] <-
  identical(dim(pm), dim(wm)) && identical(is.na(pm), is.na(wm)) &&
  max(abs(pm - wm) / (1 + abs(wm)), na.rm = TRUE) < 1e-5 &&
  max(abs(pd - wd), na.rm = TRUE) < 1e-6 && identical(part$engine, c("patch", "patch"))

# ── 5. resume ─────────────────────────────────────────────────────────────────
u2 <- file.path(map2$run_dir, "units", "u00002", "done.rds")
before <- readRDS(u2)$finished_at
again <- mp(n_cores = 2L, run_id = "map_two", probe = FALSE)
ok["a_resumed_map_recomputes_no_finished_unit"] <-
  identical(readRDS(u2)$finished_at, before) && identical(again$n_workers, 0L) &&
  identical(rd(again, "ensemble_median"), snap$ensemble_median)
unlink(u2)
redo <- mp(n_cores = 2L, run_id = "map_two", probe = FALSE)
ok["a_unit_without_its_record_is_mapped_again_to_the_same_numbers"] <-
  identical(redo$n_workers, 1L) && file.exists(u2) &&
  all(vapply(bands_all, function(b) identical(rd(redo, b), snap[[b]]), logical(1)))

# ── 6. what can only be wrong is refused ──────────────────────────────────────
ok["a_resumed_map_with_other_settings_is_refused"] <-
  grepl("different settings", err(mp(n_cores = 2L, run_id = "map_two", probe = FALSE, alpha = 0.2)))
ok["a_calibration_source_needs_a_name"] <-
  grepl("named character vector", err(mp(run_id = "x1", calibration = fit$run_dir)))
ok["a_calibration_source_must_be_a_tuning_run"] <-
  grepl("not a finished tuning run", err(mp(run_id = "x2", calibration = c(block = base))))
ok["a_raster_table_missing_a_channel_is_refused"] <-
  grepl("lacks", err(mp(run_id = "x3", rasters = rt[-1, , drop = FALSE])))
ok["a_band_that_does_not_exist_is_refused"] <-
  grepl("Unknown band", err(mp(run_id = "x4", bands = c("ensemble_median", "median_of_everything"))))
ck <- file.path(fin$run_dir, cid, "models", "seed0043_best.pt")
file.rename(ck, paste0(ck, ".away"))
ok["a_missing_seed_checkpoint_is_refused"] <- grepl("checkpoint", err(mp(run_id = "x5")))
file.rename(paste0(ck, ".away"), ck)

# The probe is the only thing that can see this one: two continuous channels
# swapped in the raster table -- every file exists, every shape matches, and
# the map would be wrong everywhere.
rt_sw <- rt
i_a <- which(rt_sw$predictor == "band_a"); i_b <- which(rt_sw$predictor == "band_b")
rt_sw$raster_file[c(i_a, i_b)] <- rt_sw$raster_file[c(i_b, i_a)]
ok["the_probe_stops_a_map_whose_channels_are_swapped"] <-
  grepl("probe failed", err(mp(run_id = "map_swapped", rasters = rt_sw)))
ok["and_nothing_was_mapped"] <-
  !any(file.exists(file.path(fin$run_dir, "maps", "map_swapped", "units",
                             sprintf("u%05d", 1:3), "done.rds")))

# A profile the refit left out -- its buffer drops some near the test set --
# has no stored prediction. The probe must draw from the ones that have one,
# not give up: P4's first run on the SOC data met 129 such profiles and did.
pa_path <- file.path(fin$run_dir, cid, "predictions", "seed0042_pred_all.csv")
pa_orig <- readLines(pa_path)
pa <- safe_read_csv2(pa_path)
row6 <- data$store$meta$sample_id[abs(data$store$meta$y - (y1 - 5.5 * cs)) < 1e-9]
safe_write_csv2(pa[!pa$sample_id %in% row6, , drop = FALSE], pa_path)
gaps <- mp(run_id = "map_probe_gaps", bands = "ensemble_median",
           extent = list(rows = c(20L, 20L), cols = c(20L, 22L)))
gap_ids <- safe_read_csv2(file.path(gaps$run_dir, "probe.csv"))$sample_id
writeLines(pa_orig, pa_path)
ok["the_probe_draws_only_profiles_with_a_stored_prediction"] <-
  length(row6) == 9L && identical(gaps$probe$status, "pass") && gaps$probe$n >= 9L &&
  !any(gap_ids %in% row6)

# ── 7. two sources over the same profiles share one DI ────────────────────────
two <- mp(run_id = "map_sources", probe = FALSE, calibration = c(block = fit$run_dir, again = fit$run_dir),
          bands = c("ensemble_median", "di", "aoa"), extent = list(rows = c(10L, 20L), cols = c(10L, 30L)))
ok["sources_with_one_reference_share_one_di_band"] <-
  identical(two$bands$band, c("ensemble_median", "di", "aoa_block", "aoa_again")) &&
  identical(rd(two, "aoa_block"), rd(two, "aoa_again")) &&
  identical(two$calibration$di_band, c("di", "di"))

# A plan that left one profile out of every fold -- as a buffer does -- has
# another reference, so the source gets its own DI band, and says so.
other <- file.path(base, "out", "tuning_other")
dir.create(other)
invisible(file.copy(list.files(fit$run_dir, full.names = TRUE), other, recursive = TRUE))
pl <- readRDS(file.path(other, "fold_plan.rds"))
gone <- pl$folds[[1]]$validation[1]
pl$folds <- lapply(pl$folds, function(f) {
  f$train <- setdiff(f$train, gone)
  f$validation <- setdiff(f$validation, gone)
  f
})
saveRDS(pl, file.path(other, "fold_plan.rds"))
diff2 <- mp(run_id = "map_sources_diff", probe = FALSE, calibration = c(block = fit$run_dir, other = other),
            bands = c("ensemble_median", "di", "aoa"), extent = list(rows = c(10L, 20L), cols = c(10L, 30L)))
ok["sources_with_different_references_get_a_di_band_each"] <-
  identical(diff2$bands$band, c("ensemble_median", "di_block", "di_other", "aoa_block", "aoa_other")) &&
  identical(diff2$calibration$di_band, c("di_block", "di_other")) &&
  identical(rd(diff2, "di_block"), rd(two, "di"))

# ── 8. the AOA weighted by an importance ──────────────────────────────────────
#
# The weights reach the map. Every channel weighted alike is no weighting, bit
# for bit; one channel alone gives another DI; and a map resumed with other
# weights, or without the ones it started with, is refused -- its finished
# units would carry one DI and its new ones another.
chs <- data$store$predictors
sub_ext  <- list(rows = c(10L, 20L), cols = c(10L, 30L))
sub_bands <- c("ensemble_median", "di", "aoa")
w_alike <- stats::setNames(rep(2, length(chs)), chs)
w_first <- stats::setNames(c(1, rep(0, length(chs) - 1L)), chs)
plain <- mp(run_id = "map_w_none", probe = FALSE, bands = sub_bands, extent = sub_ext)
alike <- mp(run_id = "map_w_alike", probe = FALSE, bands = sub_bands, extent = sub_ext,
            aoa_weights = w_alike)
first <- mp(run_id = "map_w_first", probe = FALSE, bands = sub_bands, extent = sub_ext,
            aoa_weights = w_first)
ok["weights_alike_map_the_unweighted_di"] <- identical(rd(alike, "di"), rd(plain, "di"))
ok["one_channel_alone_maps_another_di"] <- !identical(rd(first, "di"), rd(plain, "di"))
ok["a_map_resumed_with_other_weights_is_refused"] <- grepl("different settings", err(
  mp(run_id = "map_w_first", probe = FALSE, bands = sub_bands, extent = sub_ext, aoa_weights = w_alike)))
ok["a_map_resumed_without_its_weights_is_refused"] <- grepl("aoa_weights", err(
  mp(run_id = "map_w_first", probe = FALSE, bands = sub_bands, extent = sub_ext)))

# ── 9. SHAP at points of the map ──────────────────────────────────────────────
#
# A map point has no observation, so what is checked is the reading. The probe
# cuts some of the store's own profiles again and must find the store's
# patches; at every point explained, the models' mean prediction must be the
# map's own ensemble mean -- read there by the map's other engine, from the
# same rasters; points without a whole patch (the fixture's NA and -9999 cells
# are within reach) are left out, as the map leaves them; and the values laid
# back on the grid are the points' own, cell by cell.
ext9 <- c(x0 + 0.5, x0 + 5.0, y1 - 3.5, y1 - 0.5)
pts9 <- importance_points(fin, data, extent = ext9, every = 2L)
ok["map_points_are_cell_centres_every_second_cell"] <- nrow(pts9) > 100L &&
  all(abs(((pts9$x - x0) / cs) %% 1 - 0.5) < 1e-6) &&
  isTRUE(all.equal(attr(pts9, "grid")$res, c(2 * cs, 2 * cs)))
imp9 <- suppressMessages(dsm_importance(fin, data, shap_importance(samples = 16L, background = 30L),
                                        at = pts9, chunk_points = 100L, verbose = FALSE))
ok["map_points_are_explained_after_the_probe"] <- identical(imp9$rows, "map") &&
  imp9$probe_worst < 1e-5 && nrow(imp9$units) == 2L
ok["points_without_a_whole_patch_are_left_out"] <- imp9$n_dropped > 0L &&
  nrow(imp9$points) + imp9$n_dropped == nrow(pts9)
ok["the_explained_prediction_is_the_maps"] <- {
  v <- terra::extract(terra::rast(map2$vrt[["ensemble_mean"]]),
                      as.matrix(imp9$points[, c("x", "y")]))[, 1]
  isTRUE(all.equal(as.numeric(v), imp9$points$prediction_native, tolerance = 1e-4))
}
ok["each_point_keeps_its_values_for_dependence"] <- nrow(imp9$values) == nrow(imp9$points) &&
  all(c("band_a", "temp_c") %in% colnames(imp9$values))
map9 <- importance_map(imp9, output_dir = file.path(base, "shap_map"))
ok["the_map_lays_each_point_on_its_cell"] <- all(file.exists(map9$files)) && {
  v <- terra::extract(map9$shap[["band_a"]], as.matrix(imp9$points[, c("x", "y")]))[, 1]
  isTRUE(all.equal(as.numeric(v), imp9$points$band_a, tolerance = 1e-6))
}
ok["the_map_and_its_importance_draw"] <- {
  f9 <- file.path(base, "shap_map", "fig_%02d.png")
  grDevices::png(f9, width = 1100, height = 800)
  d9 <- tryCatch({ plot(map9); plot(imp9); TRUE }, error = function(e) conditionMessage(e))
  grDevices::dev.off()
  if (!isTRUE(d9)) cat("  figure failed: ", d9, "\n", sep = "")
  isTRUE(d9) && length(list.files(dirname(f9), pattern = "^fig_.*png$")) == 2L
}
ok["the_dominant_layer_names_a_variable"] <- {
  d <- terra::values(map9$dominant)[, 1]
  any(!is.na(d)) && all(d[!is.na(d)] %in% map9$legend$value)
}
ok["a_map_read_from_other_rasters_is_refused"] <- grepl("not the store's", err(
  dsm_importance(fin, data, shap_importance(samples = 4L, background = 10L), at = pts9[1:5, ],
                 rasters = rt_sw, verbose = FALSE)))
ok["only_shap_runs_at_map_points"] <-
  grepl("only shap_importance", err(dsm_importance(fin, data, permutation_importance(), at = pts9)))
ok["seeds_pick_the_models_explained"] <- nrow(suppressMessages(dsm_importance(
  fin, data, shap_importance(samples = 4L, background = 10L), at = pts9[1:20, ], seeds = 42L,
  verbose = FALSE))$units) == 1L

# ── 10. split conformal and CV+ ───────────────────────────────────────────────
#
# A plan with a calibration set: whole sites, in no fold; the final model
# predicts it and never trains on it; dsm_final() checks the three
# calibrations on the test set. On the map, the split bands are the library's
# arithmetic on the map's median and DI, and the CV+ bands are
# cv_plus_interval() on the fold models' predictions at every pixel --
# computed here from each pixel's own patches in each fold's OWN scaling, by
# hand, where the map rescales the final model's inputs by a multiply-add.
ok["split_without_a_calibration_set_is_refused"] <-
  grepl("calibration set", err(mp(run_id = "x6", intervals = "split")))
fit_c <- suppressMessages(dsm_train(
  data, model = "cnn",
  resampling = spatial_cv(k = 2L, block_size = 1, buffer = "auto", test_frac = 0.2,
                          calibration_frac = 0.2),
  tune_grid = grid, n_seeds = 1L, output_dir = file.path(base, "out"), run_id = "tuning_cal",
  device = setup_torch_device(n_threads = 1L, use_cuda = FALSE),
  n_epochs = 2L, patience = 2L, print_every = 100L, augment = FALSE, verbose = FALSE))
cal_rows <- fit_c$plan$calibration
ok["the_calibration_set_is_whole_sites_in_no_fold"] <- length(cal_rows) >= 9L &&
  length(cal_rows) %% 9L == 0L &&
  !any(cal_rows %in% unlist(lapply(fit_c$plan$folds, function(f) c(f$train, f$validation, f$test))))
fin_c <- suppressMessages(dsm_final(
  fit_c, config = cid, seeds = 2L, validation_frac = 0.34, threads_per_unit = 1L,
  n_cores = 1L, training = list(n_epochs = 2L, patience = 2L, print_every = 100L,
                                augment = FALSE),
  output_dir = file.path(base, "out", "final_model"), run_id = "final_cal", verbose = FALSE))
ens_c  <- safe_read_csv2(file.path(fin_c$run_dir, cid, "ensemble_predictions.csv"))
spec_c <- readRDS(file.path(fin_c$run_dir, "run_spec.rds"))
ok["the_final_model_predicts_the_calibration_set_and_never_trains_on_it"] <-
  setequal(ens_c$sample_id[ens_c$dataset_role == "calibration"],
           data$store$meta$sample_id[cal_rows]) &&
  setequal(spec_c$split$calibration, cal_rows) &&
  !any(cal_rows %in% c(spec_c$split$train, spec_c$split$validation))
ivs_c <- fin_c$per_config[[cid]]$intervals
ok["dsm_final_checks_every_calibration_on_the_test_set"] <- !is.null(ivs_c) &&
  all(c("cv", "split", "cv_plus") %in% ivs_c$summary$method) &&
  file.exists(file.path(fin_c$run_dir, cid, "intervals", "coverage_test.csv")) &&
  isTRUE(ivs_c$fold_check_max_rel_diff < 1e-4) &&
  any(grepl("checked on the test set", readLines(fin_c$report_file)))

map_c <- suppressMessages(dsm_predict(
  fin_c, data, calibration = c(block = fit_c$run_dir), intervals = c("cv", "split", "cv_plus"),
  n_cores = 1L, threads_per_worker = 1L, step_rows = 16L, unit_rows = 16L, chunk_cols = 20L,
  run_id = "map_cal", verbose = FALSE))
want_c <- c(sprintf("pi90_%s_%s_%s%s", rep(c("cv", "cv_plus", "split"), each = 4L),
                    rep(rep(c("constant", "level_di"), each = 2L), 3L), c("lower", "upper"),
                    rep(c("_block", "_block", ""), each = 4L)))
ok["the_map_carries_the_three_calibrations"] <- all(want_c %in% map_c$bands$band)
ok["the_probe_checks_every_fold_model_too"] <- identical(map_c$probe$status, "pass") &&
  map_c$probe$max_rel_diff < 1e-4 &&
  all(is.finite(map_c$probe$table$max_rel_diff_fold_models))

cal_c  <- readRDS(file.path(map_c$run_dir, "calibration.rds"))
snap_c <- lapply(stats::setNames(nm = c("ensemble_median", "di", want_c)), function(b) rd(map_c, b))
med_c <- snap_c$ensemble_median[cells]
di_c  <- snap_c$di[cells]
sp <- cal_c$split$intervals$pi90
ref_sc <- conformal_interval(sp$constant, med_c, lower_limit = 0)
ref_sl <- conformal_scaled_interval(sp$level_di, med_c, data.frame(level = med_c, di = di_c),
                                    lower_limit = 0)
ok["the_split_bands_are_the_calibration_sets_interval_at_every_pixel"] <-
  rel(snap_c$pi90_split_constant_lower[cells], ref_sc$lower) < 1e-5 &&
  rel(snap_c$pi90_split_constant_upper[cells], ref_sc$upper) < 1e-5 &&
  rel(snap_c$pi90_split_level_di_lower[cells], ref_sl$lower) < 1e-5 &&
  rel(snap_c$pi90_split_level_di_upper[cells], ref_sl$upper) < 1e-5

# Every pixel's patches in fold k's own scaling, from the QC'd raw values.
qraw <- raw
for (k in seq_len(C)) {
  qraw[, , k] <- matrix(qc_band_values(as.vector(raw[, , k]), qc[k, , drop = FALSE]), n_r, n_c)
}
patch_from <- function(arr, w) {
  hw <- (w - 1L) %/% 2L
  a <- array(0, dim = c(nrow(cells), C, w, w))
  for (i in seq_len(nrow(cells))) {
    rr <- cells[i, 1]; cc <- cells[i, 2]
    a[i, , , ] <- aperm(arr[(rr - hw):(rr + hw), (cc - hw):(cc + hw), , drop = FALSE], c(3L, 1L, 2L))
  }
  torch::torch_tensor(a, dtype = torch::torch_float32())
}
cp <- cal_c$sources$block$cv_plus
F_ref <- sapply(seq_along(cp$folds), function(j) {
  k <- cp$folds[j]
  s_k <- fit_scaling(data$points, data$type_table, fit_c$plan$folds[[k]]$train)
  s_k <- s_k[match(preds, s_k$predictor), , drop = FALSE]
  arr <- qraw
  for (ch in seq_len(C)) arr[, , ch] <- (qraw[, , ch] - s_k$center[ch]) / s_k$scale[ch]
  q3 <- patch_from(arr, 3L); q7 <- patch_from(arr, 7L)
  nat_k <- sapply(cp$seeds[[j]], function(s) {
    m <- build_cnn_from_config(cfg, C)
    m$load_state_dict(torch::torch_load(file.path(
      fit_c$run_dir, "models", sprintf("%s_f%d_s%d_best.pt", cp$config_id, k, s))))
    m$eval()
    pmax(expm1(torch::with_no_grad(as.numeric(m(q3, q7)$squeeze(2L)))), 0)
  })
  if (is.matrix(nat_k)) matrixStats::rowMedians(nat_k) else nat_k
})
cvp <- cp$intervals$pi90
ref_pc <- cv_plus_interval(cvp$constant, F_ref, lower_limit = 0)
ref_pl <- cv_plus_interval(cvp$level_di, F_ref, lower_limit = 0,
                           difficulty = .conformal_scale(cp$scale$coef, cp$scale$floor,
                                                         data.frame(level = med_c, di = di_c)))
worst_cvp <- max(rel(snap_c$pi90_cv_plus_constant_lower_block[cells], ref_pc$lower),
                 rel(snap_c$pi90_cv_plus_constant_upper_block[cells], ref_pc$upper),
                 rel(snap_c$pi90_cv_plus_level_di_lower_block[cells], ref_pl$lower),
                 rel(snap_c$pi90_cv_plus_level_di_upper_block[cells], ref_pl$upper))
ok["the_cv_plus_bands_are_the_fold_models_in_their_own_scaling_at_every_pixel"] <- worst_cvp < 1e-4

cat(sprintf("  fixture                  : %d x %d grid, %d channels, %d profiles, %d valid pixel(s)\n",
            n_r, n_c, C, nrow(data$store$meta), sum(valid_ref)))
cat(sprintf("  CV+ bands against a hand computation in each fold's scaling: %.2e\n", worst_cvp))
cat(sprintf("  probe                    : %d profile(s), max relative difference %.2e\n",
            map2$probe$n, map2$probe$max_rel_diff))
cat(sprintf("  worst ensemble band      : %.2e (%s)\n", max(worst_ens), names(which.max(worst_ens))))
cat(sprintf("  worst DI                 : %.2e\n", max(abs(di - di_ref))))

unlink(base, recursive = TRUE)
.report(ok, "test_predict")
