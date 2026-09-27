# ══════════════════════════════════════════════════════════════════════════════
# P4 -- dsm_predict() makes the map stage 05 made, and the chain holds on the
# 250 m grid before the world is mapped
#
# THE QUESTION.
#
# dsm_predict() replaces stage 05 with a different machine: rows read once
# through a buffer that keeps its halo, the network run fully convolutionally,
# workers side by side, the DI and the intervals in the same pass. Before it
# maps anything that takes a day, it has to be shown to make the SAME map --
# on data where the old map exists.
#
# PART A -- THE 20 km GRID, AGAINST STAGE 05 AND STAGE 07.
#
# Stage 05 mapped the deployed model on the 20 km grid (outputs/.../cfg_003/
# raster, 358,537 valid pixels, 164 px/s). dsm_predict() maps the same model
# on the same rasters, and every pixel is compared:
#
#   p4_01  the reference is the deployed model's map, on this grid
#   p4_02  dsm_predict() finished every unit and wrote every band
#   p4_03  the valid mask is 05's, pixel for pixel
#   p4_04  the six ensemble bands are 05's to 1e-4 (relative)
#   p4_05  the smeared mean is 05's, with stage 04's own factor S
#   p4_06  the constant 90% bounds are 05's, with stage 04's own q
#   p4_07  the DI is aoa_di() at 2,000 pixels drawn at random
#   p4_08  the AOA is the DI against each source's threshold
#   p4_09  a part of the map, by the patch-by-patch engine, is that part of
#          the whole -- the fully convolutional path is exact on the REAL
#          deployed network
#   (info) the DI against stage 07's raster, when 07 used the same scaling
#   (info) the speed against stage 05's
#
# PART B -- THE 250 m GRID, WHERE THE PROFILES ARE PIXELS.
#
#   p4_10  the probe passes on the training grid: a unit around the densest
#          group of profiles, read from the 181 real rasters, through the
#          whole chain, gives every seed's stored prediction at every profile
#
# and it prints what the probe unit measured -- seconds per row read, seconds
# per valid pixel of network -- as an estimate of the global map. An estimate:
# the reads of one unit on an idle disk are not three workers' on a busy one.
#
# WHAT IT WRITES: the two maps under outputs/.../spatial_prediction/.../
# dsm_predict/ (p4_20km, p4_20km_part_patch, p4_250m_probe) and the checks in
# outputs/.../tuning/.../capability_sweep/p4_predict/. Stage 05's and 07's
# outputs are read, never written. A second run resumes the maps and only
# repeats the comparisons.
#
# COST: part A ~2-5 min (most of it starting workers and the patch engine);
# part B ~2-4 min, most of it reading ~140 full rows of 181 rasters from the
# HDD.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_p4_predict_check.R")
# ══════════════════════════════════════════════════════════════════════════════

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R and in R/load_all.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/load_all.R.
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
    if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))
install_load_pkg(c("torch", "terra", "dplyr", "readr", "tibble", "purrr", "matrixStats",
                   "ps", "callr"))
source(file.path(project_root, "R", "load_all.R"))
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
data_dir     <- base("data", "processed")
patch_dir    <- base("outputs", "patches")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
sp_dir       <- base("outputs", "spatial_prediction")
maps_dir     <- file.path(sp_dir, "dsm_predict")                        # this script's maps
p4_dir       <- file.path(tuning_base, "capability_sweep", "p4_predict") # its checks
coarse_dir   <- env_chr("soc_predict_raster_dir",
                        "D:/usuario_armazenamento/cassio/R/predictors_resolution_20000m")

final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
final_dir  <- file.path(final_base, final_run_id)
summ       <- readRDS(file.path(final_dir, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
cfg_row    <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
tuning_dir <- file.path(tuning_base, summ$tuning_run_id)
ref05_dir  <- file.path(sp_dir, config_id, "raster")
ref07_dir  <- file.path(sp_dir, config_id, "aoa")

# The calibration sources: the block CV the model was selected on, and the
# kNNDM run U2 found holding the same configuration.
calibration <- c(block = tuning_dir)
for (cand in file.path(tuning_base, c("soc_0_5cm_knndm_cfg003", "soc_0_5cm_design_knndm"))) {
  if (dir.exists(cand) &&
      !is.null(suppressMessages(cv_residuals_for_config(cand, cfg_row, required = FALSE)))) {
    calibration <- c(calibration, knndm = cand)
    break
  }
}

message("\n", strrep("=", 78))
message("P4 -- dsm_predict(): the map stage 05 made, and the probe on the 250 m grid")
message(strrep("=", 78))
message("  final run   : ", final_dir, "  (", config_id, ", ", length(summ$seeds), " seeds)")
message("  calibration : ", paste(sprintf("%s = %s", names(calibration), basename(calibration)),
                                  collapse = " | "))
message("  20 km grid  : ", coarse_dir)
message(strrep("=", 78), "\n")

data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = 3L,                                   # no patch is needed; the smallest file
  target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
  verbose      = FALSE)
rt <- safe_read_csv2(file.path(metadata_dir, "raster_table_used.csv"))
qc_path <- file.path(metadata_dir, "qc_table.csv")
rt_coarse <- rt
rt_coarse$raster_file <- file.path(coarse_dir, basename(rt$raster_file))

required <- sprintf("p4_%02d", 1:10)
L <- check_ledger("P4")
f05 <- function(suffix) file.path(ref05_dir, sprintf("%s_%s_%s.tif", target_label, config_id, suffix))
m05 <- function(suffix) terra::as.matrix(terra::rast(f05(suffix)), wide = TRUE)
mine <- function(m, b) terra::as.matrix(terra::rast(m$vrt[[b]]), wide = TRUE)
rel <- function(a, b) max(abs(a - b) / (1 + abs(b)))

# ══ PART A: the 20 km grid ════════════════════════════════════════════════════
pc05 <- safe_read_csv2(file.path(sp_dir, config_id, "log", "prediction_config.csv"))
g20  <- terra::rast(rt_coarse$raster_file[1])
ledger_check(L, "p4_01", "the reference is the deployed model's map, on this grid",
             identical(as.character(pc05$final_run_id[1]), final_run_id) &&
               pc05$n_seeds[1] == length(summ$seeds) &&
               pc05$r_nrow[1] == terra::nrow(g20) && pc05$r_ncol[1] == terra::ncol(g20) &&
               all(file.exists(f05(c("valid_mask", "ensemble_median_ton_ha")))),
             sprintf("05: %s, %d seeds, %d x %d, %s valid px in %.1f min",
                     pc05$final_run_id[1], pc05$n_seeds[1], pc05$r_nrow[1], pc05$r_ncol[1],
                     format(pc05$n_valid[1], big.mark = ","), pc05$runtime_min[1]))

map <- dsm_predict(final_dir, data, rasters = rt_coarse, qc_table = qc_path,
                   calibration = calibration, output_dir = maps_dir, run_id = "p4_20km")

want <- c("ensemble_median", "ensemble_mean", "ensemble_sd", "ensemble_mad", "ensemble_min",
          "ensemble_max", "valid_mask", "di", paste0("aoa_", names(calibration)),
          paste0("smeared_mean_", names(calibration)),
          paste0("pi90_constant_", c("lower", "upper"), "_block"),
          paste0("pi90_level_di_", c("lower", "upper"), "_block"))
ledger_check(L, "p4_02", "dsm_predict() finished every unit and wrote every band",
             all(want %in% map$bands$band) && all(file.exists(map$vrt)) &&
               identical(map$probe$status, "not_applicable"),
             sprintf("%d unit(s), %d band(s); probe %s (another grid)", nrow(map$units),
                     nrow(map$bands), map$probe$status))

v05 <- m05("valid_mask") == 1
v   <- mine(map, "valid_mask") == 1
ledger_check(L, "p4_03", "the valid mask is 05's, pixel for pixel",
             identical(dim(v05), dim(v)) && all(v05 == v),
             sprintf("%s valid in both | %d pixel(s) differ", format(sum(v & v05), big.mark = ","),
                     if (identical(dim(v05), dim(v))) sum(v05 != v) else -1L))

pairs <- c(ensemble_median = "ensemble_median_ton_ha", ensemble_mean = "ensemble_mean_ton_ha",
           ensemble_sd = "ensemble_sd_ton_ha", ensemble_mad = "ensemble_mad_ton_ha",
           ensemble_min = "ensemble_min_ton_ha", ensemble_max = "ensemble_max_ton_ha")
worst <- vapply(names(pairs), function(b) rel(mine(map, b)[v], m05(pairs[[b]])[v]), numeric(1))
ledger_check(L, "p4_04", "the six ensemble bands are 05's to 1e-4 (relative)",
             all(worst < 1e-4),
             paste(sprintf("%s %.1e", sub("ensemble_", "", names(worst)), worst), collapse = " | "))

ct <- map$calibration
sm04 <- readRDS(file.path(final_dir, config_id, "smearing.rds"))
s_block <- ct$smearing_s[ct$source == "block"][1]
w_sm <- rel(mine(map, "smeared_mean_block")[v], m05("smeared_mean_ton_ha")[v])
ledger_check(L, "p4_05", "the smeared mean is 05's, with stage 04's own factor",
             isTRUE(abs(s_block - sm04$s) < 1e-12) && w_sm < 1e-4,
             sprintf("S %.6f (04: %.6f) | worst %.1e", s_block, sm04$s, w_sm))

c04 <- readRDS(file.path(final_dir, config_id, "conformal_90.rds"))
q_block <- ct$q_constant[ct$source == "block" & ct$label == "pi90"][1]
w_lo <- rel(mine(map, "pi90_constant_lower_block")[v], m05("conformal_90_lower_ton_ha")[v])
w_up <- rel(mine(map, "pi90_constant_upper_block")[v], m05("conformal_90_upper_ton_ha")[v])
ledger_check(L, "p4_06", "the constant 90% bounds are 05's, with stage 04's own q",
             is.null(c04$normalised) && isTRUE(abs(q_block - c04$constant$q) < 1e-12) &&
               w_lo < 1e-4 && w_up < 1e-4,
             sprintf("q %.6f (04: %.6f) | worst lower %.1e, upper %.1e", q_block,
                     c04$constant$q, w_lo, w_up))

cal <- readRDS(file.path(map$run_dir, "calibration.rds"))
di_map <- mine(map, "di")
vcells <- which(v, arr.ind = TRUE)
pick <- vcells[with_local_seed(42L, sample.int(nrow(vcells), min(2000L, nrow(vcells)))), , drop = FALSE]
# In the reference's channel order, whatever order the table lists them in.
aref <- cal$sources$block$aref
stk <- terra::rast(rt_coarse$raster_file[match(aref$predictors, rt_coarse$predictor)])
ex  <- terra::extract(stk, terra::cellFromRowCol(g20, pick[, 1], pick[, 2]))
if (ncol(ex) == length(aref$predictors) + 1L) ex <- ex[, -1L, drop = FALSE]   # an ID column
raw <- as.matrix(ex)
di_ref <- aoa_di(aref, raw)
w_di <- max(abs(di_map[pick] - di_ref))
ledger_check(L, "p4_07", "the DI is aoa_di() at 2,000 pixels drawn at random",
             all(is.finite(di_ref)) && w_di < 1e-5,
             sprintf("%d pixel(s) | worst %.1e | DI median %.3f, q90 %.3f", nrow(pick), w_di,
                     stats::median(di_map[v]), stats::quantile(di_map[v], 0.9)))

aoa_ok <- vapply(names(calibration), function(s) {
  a <- mine(map, paste0("aoa_", s))[v]
  th <- cal$sources[[s]]$threshold
  bad <- a != as.integer(di_map[v] <= th)
  # A pixel within float32 rounding of the threshold may land either side.
  !any(bad & abs(di_map[v] - th) > 1e-6)
}, logical(1))
inside <- vapply(names(calibration), function(s)
  mean(mine(map, paste0("aoa_", s))[v] == 1), numeric(1))
ledger_check(L, "p4_08", "the AOA is the DI against each source's threshold",
             all(aoa_ok),
             paste(sprintf("%s: DI <= %.3f, %.1f%% of the map inside", names(calibration),
                           vapply(names(calibration), function(s) cal$sources[[s]]$threshold,
                                  numeric(1)), 100 * inside), collapse = " | "))

# The part: where the profiles are, by the patch-by-patch engine.
pc <- terra::cellFromXY(g20, cbind(data$store$meta$x, data$store$meta$y))
pr <- stats::median(terra::rowFromCell(g20, pc), na.rm = TRUE)
pcc <- stats::median(terra::colFromCell(g20, pc), na.rm = TRUE)
rows <- as.integer(pmin(pmax(c(pr - 30, pr + 30), 1), terra::nrow(g20)))
cols <- as.integer(pmin(pmax(c(pcc - 75, pcc + 75), 1), terra::ncol(g20)))
part <- dsm_predict(final_dir, data, rasters = rt_coarse, qc_table = qc_path,
                    calibration = calibration, extent = list(rows = rows, cols = cols),
                    engine = "patch", probe = FALSE, bands = c("ensemble_median", "di", "valid_mask"),
                    output_dir = maps_dir, run_id = "p4_20km_part_patch", verbose = FALSE)
pm <- mine(part, "ensemble_median"); wm <- mine(map, "ensemble_median")[rows[1]:rows[2], cols[1]:cols[2]]
pdi <- mine(part, "di"); wdi <- di_map[rows[1]:rows[2], cols[1]:cols[2]]
nv_part <- sum(is.finite(pm))
ledger_check(L, "p4_09", "a part by the patch engine is that part of the fully convolutional whole",
             identical(dim(pm), dim(wm)) && identical(is.finite(pm), is.finite(wm)) && nv_part > 0L &&
               rel(pm[is.finite(pm)], wm[is.finite(wm)]) < 1e-5 &&
               max(abs(pdi - wdi), na.rm = TRUE) < 1e-6,
             sprintf("rows %d-%d, cols %d-%d: %s valid px | median worst %.1e | DI worst %.1e",
                     rows[1], rows[2], cols[1], cols[2], format(nv_part, big.mark = ","),
                     if (nv_part > 0L) rel(pm[is.finite(pm)], wm[is.finite(wm)]) else NA_real_,
                     max(abs(pdi - wdi), na.rm = TRUE)))

# (info) stage 07's DI, when it was built on the same reference
a07 <- file.path(ref07_dir, "aoa_summary.csv")
if (file.exists(a07)) {
  s07 <- safe_read_csv2(a07)
  same_ref <- isTRUE(abs(s07$avg_pair_dist[1] - cal$sources$block$aref$ref$avg_dist) < 1e-9) &&
    isTRUE(abs(s07$di_threshold[1] - cal$sources$block$threshold) < 1e-9)
  if (same_ref) {
    d07 <- terra::as.matrix(terra::rast(file.path(ref07_dir, "dissimilarity_index.tif")), wide = TRUE)
    message(sprintf("\n(info) DI against stage 07's raster at the valid pixels: worst %.1e",
                    max(abs(di_map[v] - d07[v]), na.rm = TRUE)))
  } else {
    message("\n(info) stage 07 was built on another reference (", s07$final_run_id[1],
            "): its DI is not compared.")
  }
}

# (info) the speed
px05 <- pc05$n_valid[1] / (60 * pc05$runtime_min[1])
pxme <- sum(map$units$n_valid) / (60 * map$map_minutes)
if (is.finite(pxme)) {
  message(sprintf("(info) speed: %s valid px/s, against stage 05's %s (%.0fx) -- %d worker(s) x %d thread(s)",
                  format(round(pxme), big.mark = ","), format(round(px05), big.mark = ","),
                  pxme / px05, map$n_workers, map$work$threads))
} else {
  message("(info) speed: the map was resumed, not computed, in this call -- delete ",
          map$run_dir, " to time it.")
}

# ══ PART B: the 250 m grid, where the profiles are pixels ════════════════════
#
# A one-row map beside the first profile: the map itself is trivial, and it
# is the probe before it that does the work -- the real rasters, the whole
# chain, the stored predictions.
g250 <- terra::rast(rt$raster_file[1])
c0 <- terra::cellFromXY(g250, cbind(data$store$meta$x[1], data$store$meta$y[1]))
r0 <- as.integer(terra::rowFromCell(g250, c0)); k0 <- as.integer(terra::colFromCell(g250, c0))
b <- dsm_predict(final_dir, data, rasters = rt, qc_table = qc_path, calibration = calibration,
                 extent = list(rows = c(r0, r0), cols = c(k0, k0 + 15L)),
                 bands = c("ensemble_median", "valid_mask"), output_dir = maps_dir,
                 run_id = "p4_250m_probe", n_cores = 5L, threads_per_worker = 5L)
ledger_check(L, "p4_10", "the probe passes on the 250 m grid",
             identical(b$probe$status, "pass"),
             sprintf("%s -- %s", b$probe$status, b$probe$reason %||% ""))

pr_rec <- file.path(b$run_dir, "probe", "probe", "done.rds")
if (file.exists(pr_rec)) {
  pp <- readRDS(pr_rec)
  rows_read <- (pp$r1 - pp$r0 + 1L) + 2L * ((max(unlist(cfg_row$window_sizes)) - 1L) %/% 2L)
  s_row <- pp$seconds[["read"]] / rows_read
  s_px  <- (pp$seconds[["net"]] + pp$seconds[["di"]]) / max(1, pp$n_valid)
  n_rows_250 <- terra::nrow(g250)
  valid_250 <- as.numeric(n_rows_250) * terra::ncol(g250) * pc05$valid_fraction[1]
  message(sprintf(paste0(
    "\n(info) the probe unit: %d row(s) of 181 rasters read in %.0f s (%.2f s per full row); ",
    "%s valid px through %d seed(s) + DI in %.0f s (%.2e s per px, %d thread(s)).\n",
    "       Global 250 m, roughly: %.1f h of reading on one reader, and %.1f h of network per ",
    "worker for ~%.2g valid px -- the two overlap across workers."),
    rows_read, pp$seconds[["read"]], s_row, format(pp$n_valid, big.mark = ","),
    length(summ$seeds), pp$seconds[["net"]] + pp$seconds[["di"]], s_px, pp$threads %||% 5L,
    n_rows_250 * s_row / 3600, valid_250 * s_px / 3600, valid_250))
}

create_output_dirs(p4_dir)
verdict <- ledger_verdict(L, required, file.path(p4_dir, "p4_checks.csv"))
if (verdict$pass) {
  message("\nP4 passed: dsm_predict() makes stage 05's map, and the chain holds on the 250 m grid.",
          "\nThe maps are in ", maps_dir)
}
