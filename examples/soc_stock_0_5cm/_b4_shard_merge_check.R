# ══════════════════════════════════════════════════════════════════════════════
# B4 -- the sharded path and the mosaic, checked against a known answer
#
# WHAT IS UNDER TEST.
#
# 05_predict_spatial.R predicts ONE rectangular tile. For anything at scale,
# 05a_run_parallel.R splits the raster into a grid and runs many workers, and
# 05b_merge_spatial_parts.R mosaics the tiles back. That path has not run since
# the refactor, and 05 now writes NINE bands where it used to write seven --
# the two conformal interval bands are new.
#
# WHY THIS IS WORTH A SCRIPT OF ITS OWN.
#
# A sharded prediction that is subtly wrong looks exactly like one that is
# right: nine files appear, the extent is correct, the median is plausible. The
# failure modes are seams, a dropped tile leaving a NA rectangle, a tile from
# another shard grid merged in, or the workers silently predicting the 250 m
# grid because an environment variable did not reach them.
#
# THE ONE THING THAT MAKES THIS CHECKABLE: the unpartitioned 20 km map already
# exists. A 2x2 mosaic of the same model over the same rasters must reproduce
# it. That comparison is the test; everything else here exists to make it
# possible and to stop it being faked.
#
# Run: source(".../examples/soc_stock_0_5cm/_b4_shard_merge_check.R")
# ══════════════════════════════════════════════════════════════════════════════

suppressPackageStartupMessages({
  library(processx); library(terra); library(readr); library(dplyr)
})
options(width = 200)

project_root    <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
script_dir      <- file.path(project_root, "examples", "soc_stock_0_5cm")
rscript_bin     <- file.path(R.home("bin"), "Rscript.exe")
target_label    <- "soc_stock_0_5cm"
raster_dir_20km <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_20000m"
probe_20km      <- file.path(raster_dir_20km, "aboveground_biomass_carbon.tif")

n_row_shards <- 2L
n_col_shards <- 2L

stopifnot(file.exists(rscript_bin), file.exists(probe_20km),
          file.exists(file.path(script_dir, "05a_run_parallel.R")),
          file.exists(file.path(script_dir, "05b_merge_spatial_parts.R")))

# Resolved exactly as 05a and 05b resolve them, so all three agree on the run.
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)
run_dirs     <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
final_run_id <- sort(run_dirs[grepl("^final_", run_dirs)], decreasing = TRUE)[1]
config_id    <- readRDS(file.path(final_model_base, final_run_id, "comparison",
                                  "final_run_summary.rds"))$selected_cfgs$config_id[1]

out_dir    <- file.path(project_root, "outputs", "spatial_prediction",
                        "soc_stock_modeling", target_label, config_id)
raster_dir <- file.path(out_dir, "raster")
parts_dir  <- file.path(raster_dir, "parts_2d")
log_dir    <- file.path(out_dir, "log")
ref_dir    <- file.path(out_dir, "raster_ref_1x1_20km")
ref_stamp  <- file.path(ref_dir, "REFERENCE_IS_THE_1x1_20km_RUN.txt")

message("final_run_id : ", final_run_id, "\nconfig_id    : ", config_id)

ref_grid <- terra::rast(probe_20km)
exp_nrow <- as.integer(terra::nrow(ref_grid))
exp_ncol <- as.integer(terra::ncol(ref_grid))
message(sprintf("20 km grid   : %d rows x %d cols", exp_nrow, exp_ncol))
# 160,298 columns is the 250 m grid. If that appears, the environment variable
# did not reach here and everything downstream is measuring the wrong raster.
stopifnot(exp_ncol < 10000L)


# ── STEP 0a: the recorded statistics of the 1x1 run ──────────────────────────────
#
# 05 appends a shard suffix to these two files whenever the run is partitioned,
# so the UNSUFFIXED pair belongs to the 1x1 run and survives the 2x2 run
# untouched. They are ground truth even after the rasters are overwritten.
#
# READ BEFORE THE SNAPSHOT, because the snapshot needs to know how many bands
# the run produced, and a literal count there is a silent skip waiting to happen.
ref_cfg <- readr::read_csv2(file.path(log_dir, "prediction_config.csv"),
                            show_col_types = FALSE)
ref_sum <- readr::read_csv2(file.path(log_dir, "prediction_raster_summary.csv"),
                            show_col_types = FALSE)
stopifnot(ref_cfg$r_nrow[1] == exp_nrow, ref_cfg$r_ncol[1] == exp_ncol,
          ref_cfg$n_row_shards[1] == 1L, ref_cfg$n_col_shards[1] == 1L,
          nrow(ref_sum) >= 7L)
ref_n_valid <- as.integer(ref_cfg$n_valid[1])
message("reference 1x1: n_valid = ", format(ref_n_valid, big.mark = ","))
n_bands <- nrow(ref_sum)

# ── STEP 0b: snapshot the 1x1 map, AT MOST ONCE ───────────────────────────────
#
# 05b deletes and rewrites raster/ before merging. So the single-tile map has to
# be copied aside first -- and the copy must be guarded, because on a second run
# raster/ already holds the 2x2 mosaic. An unconditional copy would overwrite
# the reference with the thing being tested, and the pixel comparison would then
# compare the 2x2 result against itself and report a perfect match for any
# result whatsoever, including garbage.
#
# This is the defect that a re-run would have hidden, and re-running after a
# session dies is the normal case here.
if (!file.exists(ref_stamp)) {
  ref_src <- list.files(raster_dir, pattern = "[.]tif$", full.names = TRUE)
  if (length(ref_src) == n_bands) {
    dir.create(ref_dir, recursive = TRUE, showWarnings = FALSE)
    file.copy(ref_src, ref_dir, overwrite = TRUE)
    writeLines(c(paste("snapshot:", format(Sys.time())),
                 "source: the completed unpartitioned (1x1) 20 km run",
                 basename(ref_src)), ref_stamp)
    message("1x1 reference snapshotted -> ", ref_dir)
  } else {
    # A SILENT SKIP IS THE WORST OUTCOME HERE, so this stops.
    #
    # The count used to be a literal 9, and the branch a message. Add a band to
    # stage 05 and the condition turns FALSE: the pixel comparison -- the only
    # check in this script that looks at values rather than at summaries --
    # switches itself off, and the remaining checks still print PASS. The
    # script would report success while having stopped doing the thing it is
    # for.
    stop("raster/ holds ", length(ref_src), " tif(s) but the 1x1 summary names ",
         n_bands, ". Either raster/ already holds a partitioned result (delete ",
         "it and re-run the 1x1 map), or stage 05 changed its bands without ",
         "this run being redone.", call. = FALSE)
  }
} else {
  message("1x1 reference already snapshotted -- left untouched.")
}

# ── STEP 0c: remove tiles from any OTHER shard grid ───────────────────────────
#
# 05b globs every *_r#of#_c#of#.tif in parts_2d with no filter on the grid or the
# resolution, and terra::merge takes the first non-NA source it finds. A leftover
# set from a different grid becomes a silent chimera -- a mosaic assembled from
# two runs, with no seam visible and no warning.
grid_pat <- sprintf("_r[0-9]{3}of%03d_c[0-9]{3}of%03d[.]tif$",
                    n_row_shards, n_col_shards)
if (dir.exists(parts_dir)) {
  tifs  <- list.files(parts_dir, pattern = "[.]tif$", full.names = TRUE)
  stale <- tifs[!grepl(grid_pat, basename(tifs))]
  if (length(stale) > 0L) {
    message("removing ", length(stale), " tile(s) from another shard grid")
    file.remove(stale)
  }
}

# ── STEP 1: 05a at 2 x 2 ──────────────────────────────────────────────────────
#
# 05a now takes the grid from the environment and passes soc_predict_raster_dir
# to its workers explicitly rather than relying on inheritance. Before that fix
# this step needed a patched copy of 05a -- and a test that requires editing the
# script it tests is a test that runs once.
Sys.setenv(soc_predict_raster_dir = raster_dir_20km,
           soc_n_row_shards = n_row_shards,
           soc_n_col_shards = n_col_shards,
           soc_max_concurrent = 4L)

t0 <- Sys.time()
a_res <- processx::run(rscript_bin,
                       args = file.path(script_dir, "05a_run_parallel.R"),
                       stdout = "|", stderr = "|", echo = TRUE,
                       error_on_status = FALSE)
message(sprintf("05a finished in %.1f min, status %d",
                as.numeric(difftime(Sys.time(), t0, units = "mins")),
                a_res$status))
stopifnot(a_res$status == 0L)

# THE SHARD OVERRIDES ARE UNSET THE MOMENT 05a RETURNS.
#
# They live in the session, and a later real 05a run in the same session would
# silently take 2 x 2 instead of 250 x 4 -- four enormous workers instead of a
# thousand small ones, which is an out-of-memory failure an hour in with no
# indication of why. soc_predict_raster_dir is left alone deliberately: 05b and
# 05c ignore it, and the next 05 in this session is meant to stay at 20 km.
Sys.unsetenv(c("soc_n_row_shards", "soc_n_col_shards", "soc_max_concurrent"))

# ── STEP 1b: the two guards 05a does not have ─────────────────────────────────
#
# (a) every shard wrote a marker recording THIS grid at THIS resolution. 05a's
#     own resume keys on the marker's existence alone, and every resolution
#     shares one output directory -- so a 250 m marker would let a 20 km run
#     skip every shard and report success.
# (b) every shard wrote NINE bands. The conformal pair is conditional on
#     conformal_90.rds being present; without it 05 writes seven, prints one
#     line, and 05b merges seven and reports success. Nothing else counts them.
for (rs in seq_len(n_row_shards)) for (cs in seq_len(n_col_shards)) {
  f <- file.path(log_dir, sprintf("prediction_config_r%03dof%03d_c%03dof%03d.csv",
                                  rs, n_row_shards, cs, n_col_shards))
  stopifnot(file.exists(f))
  cfg <- readr::read_csv2(f, show_col_types = FALSE)
  stopifnot(cfg$r_nrow[1] == exp_nrow, cfg$r_ncol[1] == exp_ncol,
            cfg$n_row_shards[1] == n_row_shards,
            cfg$n_col_shards[1] == n_col_shards)
  suf <- sprintf("_r%03dof%03d_c%03dof%03d[.]tif$", rs, n_row_shards, cs,
                 n_col_shards)
  n_band <- length(list.files(parts_dir, pattern = suf))
  message(sprintf("  shard [r%d/c%d]: %d band(s), %d x %d", rs, cs, n_band,
                  cfg$r_nrow[1], cfg$r_ncol[1]))
  stopifnot(n_band == 9L)
}

tile_files  <- list.files(parts_dir, pattern = "[.]tif$", full.names = TRUE)
stopifnot(length(tile_files) == 9L * n_row_shards * n_col_shards,
          all(grepl(grid_pat, basename(tile_files))))
tiles_mtime <- max(file.info(tile_files)$mtime)

# ── STEP 2: the merge 05a already did ─────────────────────────────────────────
#
# 05a RUNS 05b ITSELF, at 05a_run_parallel.R:283-291, and says so in its header
# at line 29. The first version of this script ran 05b a second time -- harmless,
# because the merge is idempotent over the same tiles, but it doubled the merge
# and it verified a mosaic this script had produced rather than the one the
# PIPELINE produces. Checking your own side effect is not checking the pipeline.
#
# So the merge is not re-run. What is checked is 05a's own merge log and the
# nine files it left behind.
#
# EXPECT NINE WARNINGS in that log about the mosaic geometry differing from the
# template. 05b builds the template from raster_table_used.csv's first entry --
# the 250 m raster -- and compares with stopOnError = FALSE. It never reads the
# prediction-raster override, so at 20 km every layer warns.
#
# Those warnings are not breakage, AND THEIR ABSENCE WOULD NOT BE CORRECTNESS.
# 05b's geometry check is dead for this run, which is exactly why the checks
# below redo it against a real 20 km raster rather than trusting 05b's silence.
merge_log <- file.path(log_dir, "merge.log")
stopifnot(file.exists(merge_log))
message("05a's merge log: ", merge_log)

# THE BAND LIST COMES FROM THE REFERENCE RUN, NOT FROM THIS FILE.
#
# It used to be nine names typed out here, alongside stopifnot(nrow(ref_sum) ==
# 9L). That pairing has one failure mode and one worse one: stage 05 gains a
# band and this script stops with an arithmetic complaint that says nothing
# about the band -- or the assertion is relaxed and the new band is simply never
# checked, which is how a band arrives in a map having passed nothing.
#
# The summary 05 wrote for the 1x1 run already names every band it produced, and
# every file it wrote. Reading them is what makes this script check the run in
# front of it instead of the run it was written against.
layers <- ref_sum$layer
merged <- file.path(raster_dir, basename(ref_sum$file))
message("bands to check (from the 1x1 summary): ", paste(layers, collapse = ", "))

# THE MERGE MUST HAVE HAPPENED. Without this, the files left over from the 1x1
# run satisfy "the bands exist" while 05b did nothing at all.
stopifnot(all(file.exists(merged)),
          all(file.info(merged)$mtime >= tiles_mtime))
message("all ", length(merged), " bands merged, and newer than the tiles they came from.")

# ── STEP 3: the checks ────────────────────────────────────────────────────────

close_to <- function(a, b, rel, abs_tol) {
  isTRUE(is.finite(a) && is.finite(b) && abs(a - b) <= abs_tol + rel * abs(b))
}

res <- lapply(seq_along(merged), function(i) {
  r  <- terra::rast(merged[i])
  rw <- ref_sum[ref_sum$layer == layers[i], ]
  geo <- isTRUE(terra::compareGeom(r, ref_grid, stopOnError = FALSE)) &&
         terra::nrow(r) == exp_nrow && terra::ncol(r) == exp_ncol
  n_val <- terra::global(!is.na(r), "sum")[1, 1]
  cnt <- if (bands[i] == "valid_mask") isTRUE(n_val == terra::ncell(ref_grid))
         else isTRUE(n_val == ref_n_valid)
  gm  <- terra::global(r, c("min", "mean", "max"), na.rm = TRUE)
  # TOLERANCES ARE RELATIVE, and that is not slackness.
  #
  # The 2x2 run cannot be bit-identical to the 1x1 one: the workers get a
  # different thread count, the block boundaries fall in different places so the
  # inference batches hold different pixels, and the back-transform expm1()
  # amplifies a float32 discrepancy by exp(log1p(x)) -- about 26 at the global
  # median and 113 at the maximum. A bare absolute tolerance would fail a
  # correct run at the top of the range.
  val <- close_to(gm[1, "mean"], rw$gmean[1], 1e-4, 1e-3) &&
         close_to(gm[1, "min"],  rw$gmin[1],  1e-3, 1e-3) &&
         close_to(gm[1, "max"],  rw$gmax[1],  1e-3, 1e-3)
  tibble::tibble(band = bands[i], geometry = geo, n_valid = cnt, stats = val,
                 gmean = gm[1, "mean"], ref_gmean = rw$gmean[1])
})
res <- dplyr::bind_rows(res)
print(res, n = Inf)

# ── THE COMPARISON THIS SCRIPT EXISTS FOR ─────────────────────────────────────
#
# Every other check above can be satisfied by a mosaic that is well formed and
# wrong. This one cannot: the 2x2 result is compared to the 1x1 map pixel by
# pixel. A seam, a dropped tile, a tile from another grid, or workers on the
# wrong raster all show up here and nowhere else.
pix <- NULL
if (file.exists(ref_stamp)) {
  ref_files <- file.path(ref_dir, basename(merged))
  if (all(file.exists(ref_files))) {
    pix <- vapply(seq_along(merged), function(i) {
      d <- tryCatch(terra::global(
        abs(terra::rast(merged[i]) - terra::rast(ref_files[i])), "max",
        na.rm = TRUE)[1, 1], error = function(e) NA_real_)
      d
    }, numeric(1))
    names(pix) <- bands
    tol <- 1e-3 + 1e-4 * abs(ref_sum$gmax[match(layers, ref_sum$layer)])
    message("\nMax |2x2 - 1x1| per band, against tolerance:")
    print(tibble::tibble(band = bands, max_abs_diff = pix, tolerance = tol,
                         ok = pix <= tol), n = Inf)
  } else {
    message("\nReference snapshot incomplete -- pixel comparison skipped.")
  }
} else {
  message("\nNo reference snapshot -- pixel comparison skipped. ",
          "This run proves the wiring, not the numbers.")
}

# ── STEP 4: 05c parses the shard logs ─────────────────────────────────────────
# It auto-detects the newest worker-log directory and reads the grid off the
# filenames, so 2x2 needs no edit. Sourced into its own environment so its
# variables cannot collide with this script's.
source(file.path(script_dir, "05c_estimate_eta.R"), local = new.env())

# ── Verdict ───────────────────────────────────────────────────────────────────
ok_pix <- is.null(pix) ||
  all(pix <= 1e-3 + 1e-4 * abs(ref_sum$gmax[match(layers, ref_sum$layer)]))
verdict <- all(res$geometry) && all(res$n_valid) && all(res$stats) && ok_pix

message("\n", strrep("=", 78))
message(sprintf("B4: %s | geometry %d/9 | n_valid %d/9 | stats %d/9 | pixels %s",
                if (verdict) "PASS" else "FAIL",
                sum(res$geometry), sum(res$n_valid), sum(res$stats),
                if (is.null(pix)) "not compared" else
                  if (ok_pix) "match the 1x1 map" else "DIFFER FROM THE 1x1 MAP"))
message(strrep("=", 78))
if (!verdict) {
  message("\nA mosaic can be well formed and wrong. Read the per-band table above:")
  message("  geometry FALSE -> the workers predicted the wrong grid")
  message("  n_valid  FALSE -> a tile is missing, or a seam dropped pixels")
  message("  pixels   FALSE -> the tiles do not agree with the single-tile map")
}
