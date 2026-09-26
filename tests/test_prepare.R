# Unit test: dsm_prepare() -- a point table and a folder of rasters become a
# patch store, and the decisions travel with it
#
# WHY THIS FILE EXISTS.
#
# dsm_prepare() replaces stages 01 and 02 of the SOC example, 1,300 lines that
# only ever ran on one dataset. It has to reproduce what they did -- the QC
# rules, the type detection, the edge check, the full-window rule, the store
# layout -- and do it for any folder of rasters. Every point of the fixture
# below exists to exercise one path, and every assertion is checked against a
# value computed independently from the raster's own formula.
#
# The fixture: a 30 x 40 grid in EPSG:4326, one cell per degree, windows 3 and
# 5 (so rows 3-28 and columns 3-38 are far enough from the edge). Six rasters
# and one text file that must be ignored:
#
#   Cont A.tif                        row * 100 + col      (cleans to cont_a)
#   surface_temperature_celsius.tif   15 + 0.01 * that, with a -8888 sentinel
#                                     at (10,10) and (20,22)
#   pct_clay.tif                      -5 .. 105 across the columns -> clamped
#   dummy_forest.tif                  (row + col) %% 2
#   const_glacier.tif                 0 everywhere -> a constant "dummy"
#   drop_me.tif                       declared in `drop`
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_prepare.R")

suppressMessages({
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
      if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
    }
  }
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
suppressMessages(source(file.path(root, "R", "load_all.R")))

ok <- c()

# ── the fixture ───────────────────────────────────────────────────────────────
base <- file.path(tempdir(), "dlc_prepare_test")
unlink(base, recursive = TRUE)          # on the way IN: a leftover would be reused
rdir <- file.path(base, "rasters")
dir.create(rdir, recursive = TRUE)

n_r <- 30L; n_c <- 40L
rc  <- expand.grid(c = seq_len(n_c), r = seq_len(n_r))   # row-major, as terra stores
cell <- function(r, c) (r - 1L) * n_c + c
mk <- function(name, vals, dir = rdir, nrows = n_r, ncols = n_c,
               xmax = n_c, ymax = n_r) {
  r <- terra::rast(nrows = nrows, ncols = ncols, xmin = 0, xmax = xmax,
                   ymin = 0, ymax = ymax, crs = "EPSG:4326")
  terra::values(r) <- vals
  terra::writeRaster(r, file.path(dir, paste0(name, ".tif")), overwrite = TRUE,
                     datatype = "FLT8S")
}
v_cont <- rc$r * 100 + rc$c
v_temp <- 15 + 0.01 * v_cont
v_temp[c(cell(10L, 10L), cell(20L, 22L))] <- -8888
mk("Cont A", v_cont)
mk("surface_temperature_celsius", v_temp)
mk("pct_clay", -5 + (rc$c - 1) * (110 / 39))
mk("dummy_forest", (rc$r + rc$c) %% 2)
mk("const_glacier", rep(0, n_r * n_c))
mk("drop_me", rep(1, n_r * n_c))
writeLines("not a raster", file.path(rdir, "notes.txt"))

# Each point exists for one path. x, y are cell centres.
pts_rc <- data.frame(
  profile_id = c("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p1", "p10",
                 "p11", "p12", "p13"),
  r   = c(5, 6, 10, 20,  2, 15, 12, 14, 25, 18, 26,  8, 16),
  c   = c(5, 30, 10, 20, 15, 39, 25, 12,  8, 33, 38, 18,  1),
  soc = c(10, 20, 30, 40, 50, 60,  0, NA, 70, 5.5, 100, 12, 33))
#   p3   centre on a temperature sentinel     -> dropped by the point QC
#   p4   a sentinel inside its 5x5 window     -> kept by the QC, lost to the window rule
#   p5 / p6 / p13  within 2 cells of the edge -> kept in the table, not in the store
#   p6 / p13  pct_clay over 100 / under 0     -> clamped in the table
#   p7   target 0 with target_min = 0         -> dropped
#   p8   target NA                            -> dropped
#   9th  repeats profile p1                   -> the first p1 is kept
#   p11  its window reaches pct_clay > 100    -> clamped in the patch
pts <- data.frame(profile_id = pts_rc$profile_id, x = pts_rc$c - 0.5,
                  y = (n_r + 0.5) - pts_rc$r, soc = pts_rc$soc)

prep <- function(dir, ...) {
  suppressMessages(dsm_prepare(
    points = pts, target = "soc", raster_dir = rdir, windows = c(3, 5),
    out_dir = file.path(base, dir), percentage = "^pct_",
    na_below = c("surface_temperature_celsius$" = -100), drop = "drop_me",
    transform = "log1p", target_min = 0, verbose = FALSE, ...))
}

st <- prep("serial", n_cores = 1L)
pt <- safe_read_csv2(st$points_file)
ty <- safe_read_csv2(file.path(st$metadata_dir, "predictor_type_table.csv"))
mf <- readRDS(file.path(st$store_dir, "patch_manifest.rds"))
meta <- safe_read_csv2(file.path(st$store_dir, "patch_meta.csv"))
w3 <- readRDS(file.path(st$store_dir, "patches_w03.rds"))
w5 <- readRDS(file.path(st$store_dir, "patches_w05.rds"))

# ── 1. the rasters: names, order, what is left out ───────────────────────────
preds <- c("const_glacier", "cont_a", "dummy_forest", "pct_clay",
           "surface_temperature_celsius")
ok["names_cleaned_sorted_dropped_and_the_text_file_ignored"] <-
  identical(ty$predictor, preds) && identical(st$recipe$predictors, preds)

# ── 2. the point QC ──────────────────────────────────────────────────────────
ok["qc_drops_sentinel_centre_zero_target_na_target_and_the_duplicate"] <-
  identical(as.character(pt$profile_id),
            c("p1", "p2", "p4", "p5", "p6", "p10", "p11", "p12", "p13"))
ok["the_first_of_a_repeated_profile_is_the_one_kept"] <-
  pt$x[pt$profile_id == "p1"] == 4.5
ok["percentages_are_clamped_not_dropped"] <-
  pt$pct_clay[pt$profile_id == "p6"] == 100 && pt$pct_clay[pt$profile_id == "p13"] == 0
ok["sample_id_is_the_row_key"] <- identical(as.integer(pt$sample_id), 1:9)
ok["the_target_is_carried_in_both_spaces"] <-
  isTRUE(all.equal(pt$target_transform, log1p(pt$target_native))) &&
  isTRUE(all.equal(pt$target_native, pt$soc))
qs <- safe_read_csv2(file.path(st$metadata_dir, "qc_summary.csv"))
ok["qc_summary_counts_before_the_deduplication_as_stage_01_did"] <-
  qs$n_rows_extracted == 13 && qs$n_target_problem == 2 &&
  qs$n_predictor_problem == 1 && qs$n_rows_after_qc == 10

# ── 3. the types ─────────────────────────────────────────────────────────────
tyd <- setNames(ty$is_dummy, ty$predictor)
typ <- setNames(ty$is_percentage, ty$predictor)
ok["a_declared_percentage_is_a_percentage_and_not_a_dummy"] <-
  typ[["pct_clay"]] && !tyd[["pct_clay"]]
ok["zero_one_channels_are_detected_as_dummies"] <-
  tyd[["dummy_forest"]] && tyd[["const_glacier"]]
ok["everything_else_is_continuous"] <-
  !any(tyd[c("cont_a", "surface_temperature_celsius")]) &&
  !any(typ[c("cont_a", "surface_temperature_celsius")])
cr <- safe_read_csv2(file.path(st$metadata_dir, "channel_risk.csv"))
ok["a_channel_constant_at_the_points_is_flagged"] <-
  identical(cr$risk[cr$predictor == "const_glacier"], "constant")

# ── 4. the store: edge check, window rule, geometry ──────────────────────────
ok["store_keeps_only_points_with_a_whole_valid_window"] <-
  identical(as.character(meta$profile_id), c("p1", "p2", "p10", "p11", "p12"))
ok["arrays_line_up_with_patch_meta"] <-
  dim(w3)[1] == 5L && dim(w5)[1] == 5L && dim(w5)[2] == 5L && all(dim(w5)[3:4] == 5L)
inv <- safe_read_csv2(file.path(st$metadata_dir, "patches", "channel_invalidation.csv"))
ok["the_window_rule_names_the_channel_that_lost_the_point"] <-
  inv$n_invalidated[inv$predictor == "surface_temperature_celsius"] == 1 &&
  inv$n_sole_cause[inv$predictor == "surface_temperature_celsius"] == 1

# A patch against the raster's own formula: cont_a is row * 100 + col, and the
# third index is the row offset, the fourth the column (test_patch_geometry.R).
truth <- function(r, c, w) {
  h <- (w - 1L) %/% 2L
  outer((r + seq_len(w) - 1L - h) * 100, (c + seq_len(w) - 1L - h), "+")
}
ch_cont <- match("cont_a", preds)
ok["a_5x5_patch_is_the_raster_around_its_point"] <-
  isTRUE(all.equal(w5[1, ch_cont, , ], truth(5, 5, 5), check.attributes = FALSE))
ok["a_3x3_patch_is_the_raster_around_its_point"] <-
  isTRUE(all.equal(w3[3, ch_cont, , ], truth(18, 33, 3), check.attributes = FALSE))
ch_pct <- match("pct_clay", preds)
ok["the_patch_gets_the_same_clamp_as_the_point"] <-
  max(w5[4, ch_pct, , ]) == 100 && sum(w5[4, ch_pct, , ] == 100) == 10L

aligned <- align_points_to_meta(pt, meta)
cc <- check_patch_centres(st$store_dir, aligned, preds, window = 3L)
ok["every_patch_centre_equals_its_point_value"] <- isTRUE(cc$ok)

# ── 5. what the store records ────────────────────────────────────────────────
ok["the_manifest_has_stage_02s_fields"] <-
  identical(mf$target_transform, "log1p") && isTRUE(mf$store_complete) &&
  mf$n_points_input == 9 && mf$n_points_valid == 5 &&
  identical(mf$windows_extracted, "3, 5") && mf$cell_size == 1 &&
  isFALSE(mf$scaling_applied)
ok["the_recipe_records_the_decisions"] <-
  identical(st$recipe$transform, "log1p") &&
  identical(st$recipe$windows, c(3L, 5L)) &&
  identical(st$recipe$drop_applied, "drop_me") &&
  identical(st$recipe$run_profile, "full")
ok["the_store_is_self_contained"] <-
  all(file.exists(file.path(st$store_dir, unlist(st$recipe$files))))

# ── 6. dsm_load takes the store alone, and reads the transform back ──────────
d <- suppressMessages(dsm_load(st, verbose = FALSE))
ok["dsm_load_needs_nothing_but_the_store"] <-
  inherits(d, "dsm_data") && nrow(d$store$meta) == 5L &&
  identical(d$type_table$predictor, preds)
ok["dsm_load_reads_the_transform_the_store_was_built_under"] <-
  identical(d$transform$name, "log1p") && identical(d$transform$inverse, expm1)
ok["dsm_load_reads_the_resolution_from_the_rasters"] <- d$cell_size == 1

# ── 7. n_cores changes the time, never the result ────────────────────────────
st2 <- prep("parallel", n_cores = 2L)
ok["two_cores_give_the_identical_arrays"] <-
  identical(readRDS(file.path(st2$store_dir, "patches_w03.rds")), w3) &&
  identical(readRDS(file.path(st2$store_dir, "patches_w05.rds")), w5)
ok["two_cores_give_the_identical_point_set"] <-
  isTRUE(all.equal(safe_read_csv2(file.path(st2$store_dir, "patch_meta.csv")), meta))

# ── 7b. how the raster is READ changes the time, never the result ────────────
#
# The extraction reads, per row chunk, one window per group of nearby points.
# The fixture's points all fall in one chunk and one group at the defaults, so
# the index arithmetic of a window that starts mid-raster would go untested.
# Here every point gets its own read (no merging, reads as narrow as a patch),
# and separately the rows are cut into chunks of 3 -- both must give the
# identical arrays.
st_1pt <- prep("one_read_per_point", n_cores = 1L, read_gap = 0L, read_max_cols = 5L)
ok["one_read_per_point_gives_the_identical_arrays"] <-
  identical(readRDS(file.path(st_1pt$store_dir, "patches_w03.rds")), w3) &&
  identical(readRDS(file.path(st_1pt$store_dir, "patches_w05.rds")), w5)
ok["one_read_per_point_really_split_the_reads"] <-
  st_1pt$recipe$n_reads_per_band > st$recipe$n_reads_per_band
st_rows <- prep("short_chunks", n_cores = 1L, chunk_nrows = 3L)
ok["three_row_chunks_give_the_identical_arrays"] <-
  identical(readRDS(file.path(st_rows$store_dir, "patches_w05.rds")), w5)

# The grouping itself, on columns chosen to hit each rule. Half window 2, so a
# point at column c needs columns c-2 .. c+2.
cg <- .prep_column_groups(c(10L, 12L, 30L, 31L, 100L), h = 2L, gap = 5L, max_cols = 30L)
ok["column_groups_merge_close_windows_and_split_far_ones"] <-
  identical(cg, list(1:2, 3:4, 5L))
# 10 and 20 span columns 8..22 (15 wide); adding 30 would make 8..32 = 25 > 24.
cg2 <- .prep_column_groups(c(10L, 20L, 30L, 40L), h = 2L, gap = 100L, max_cols = 24L)
ok["column_groups_never_exceed_the_width_cap"] <-
  identical(cg2, list(1:2, 3:4))
ok["the_recipe_records_how_the_store_was_read"] <-
  is.numeric(st$recipe$gdal_cache_mb) && st$recipe$n_reads_per_band >= 1

# ── 8. an sf object takes the same path as stage 01 ──────────────────────────
if (requireNamespace("sf", quietly = TRUE)) {
  pts_sf <- sf::st_as_sf(pts, coords = c("x", "y"), crs = 4326)
  st3 <- suppressMessages(dsm_prepare(
    points = pts_sf, target = "soc", raster_dir = rdir, windows = c(3, 5),
    out_dir = file.path(base, "sf"), percentage = "^pct_",
    na_below = c("surface_temperature_celsius$" = -100), drop = "drop_me",
    transform = "log1p", target_min = 0, n_cores = 1L, verbose = FALSE))
  ok["an_sf_object_gives_the_same_store"] <-
    identical(readRDS(file.path(st3$store_dir, "patches_w05.rds")), w5)
}

# ── 9. the options ───────────────────────────────────────────────────────────
st_n <- suppressMessages(dsm_prepare(
  points = pts, target = "soc", raster_dir = rdir, windows = c(3, 5),
  out_dir = file.path(base, "native"), drop = "drop_me", transform = "none",
  n_cores = 1L, verbose = FALSE))
pn <- safe_read_csv2(st_n$points_file)
ok["transform_none_keeps_the_target_as_it_is"] <-
  isTRUE(all.equal(pn$target_transform, pn$target_native)) &&
  identical(st_n$recipe$transform, "none")
ok["without_target_min_a_zero_target_is_kept"] <- "p7" %in% as.character(pn$profile_id)

st_d <- suppressMessages(dsm_prepare(
  points = pts, target = "soc", raster_dir = rdir, windows = c(3, 5),
  out_dir = file.path(base, "dummy"), drop = "drop_me",
  dummy = "dummy_forest", n_cores = 1L, verbose = FALSE))
tyd2 <- setNames(st_d$recipe$types$is_dummy, st_d$recipe$types$predictor)
# Declared, the list REPLACES the detection -- so const_glacier is no longer a
# dummy, and a non-dummy with zero variance cannot be z-scored: it is dropped.
ok["declared_dummies_replace_the_detection"] <-
  tyd2[["dummy_forest"]] && !("const_glacier" %in% names(tyd2)) &&
  identical(st_d$recipe$dropped_no_variance, "const_glacier")

# frac just under 1, so the subsample path runs but keeps enough points for
# the edge check to leave some -- the point here is the record, not the draw.
st_s <- suppressMessages(dsm_prepare(
  points = pts, target = "soc", raster_dir = rdir, windows = c(3, 5),
  out_dir = file.path(base, "sub"), drop = "drop_me", n_cores = 1L,
  subsample = list(frac = 0.99, block_size = 10, seed = 1L), verbose = FALSE))
ok["a_subsample_is_recorded_as_one"] <- identical(st_s$recipe$run_profile, "dev")

# ── 10. what it refuses, and says why ────────────────────────────────────────
err <- function(expr) {
  e <- tryCatch({ suppressMessages(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(e)) "" else e
}
ok["an_existing_store_is_not_overwritten_silently"] <-
  grepl("already holds", err(prep("serial", n_cores = 1L)))
ok["overwrite_replaces_it"] <- {
  s <- prep("serial", n_cores = 1L, overwrite = TRUE)
  identical(readRDS(file.path(s$store_dir, "patches_w05.rds")), w5)
}
ok["windows_are_required"] <-
  grepl("required", err(dsm_prepare(pts, "soc", rdir, out_dir = file.path(base, "e1"))))
ok["an_even_window_is_refused"] <-
  grepl("odd", err(dsm_prepare(pts, "soc", rdir, windows = 4,
                               out_dir = file.path(base, "e2"))))
ok["a_declared_dummy_that_is_also_a_percentage_is_refused"] <-
  grepl("both", err(dsm_prepare(pts, "soc", rdir, windows = 3,
                                out_dir = file.path(base, "e3"),
                                percentage = "^pct_", dummy = "pct_clay")))
ok["a_reserved_target_name_is_refused"] <- {
  p2 <- pts; names(p2)[names(p2) == "soc"] <- "sample_id"
  grepl("reserved", err(dsm_prepare(p2, "sample_id", rdir, windows = 3,
                                    out_dir = file.path(base, "e4"))))
}

bad1 <- file.path(base, "misaligned"); dir.create(bad1)
mk("a", v_cont, dir = bad1)
mk("b", rep(1, 20 * 40), dir = bad1, nrows = 20L, ymax = 20)
ok["a_raster_on_another_grid_is_named"] <-
  grepl("b.tif", err(dsm_prepare(pts, "soc", bad1, windows = 3,
                                 out_dir = file.path(base, "e5"))), fixed = TRUE)

bad2 <- file.path(base, "dupnames"); dir.create(bad2)
mk("A b", v_cont, dir = bad2); mk("a_b", v_cont, dir = bad2)
ok["two_files_cleaning_to_one_name_are_refused"] <-
  grepl("a_b", err(dsm_prepare(pts, "soc", bad2, windows = 3,
                               out_dir = file.path(base, "e6"))))

bad3 <- file.path(base, "multiband"); dir.create(bad3)
r2 <- c(terra::rast(file.path(rdir, "Cont A.tif")), terra::rast(file.path(rdir, "pct_clay.tif")))
terra::writeRaster(r2, file.path(bad3, "two.tif"), overwrite = TRUE)
mk("one", v_cont, dir = bad3)
ok["a_multiband_file_is_refused"] <-
  grepl("more than one band", err(dsm_prepare(pts, "soc", bad3, windows = 3,
                                              out_dir = file.path(base, "e7"))))

ok["an_unknown_transform_is_refused"] <- grepl("Unknown", err(target_transform_spec("sqrt")))
ok["resolve_cores_defaults_to_physical_minus_one"] <-
  resolve_cores(NULL) == max(1L, .physical_cores() - 1L)
ok["resolve_cores_refuses_a_fraction_and_zero"] <-
  grepl("whole number", err(resolve_cores(2.5))) && grepl("whole number", err(resolve_cores(0)))
ok["a_store_without_a_recipe_still_needs_its_tables"] <- {
  old <- file.path(base, "old_store"); dir.create(old)
  file.copy(file.path(st$store_dir, c("patch_manifest.rds", "patch_meta.csv",
                                      "patches_w03.rds", "patches_w05.rds")), old)
  grepl("recipe", err(dsm_load(old, verbose = FALSE)))
}

unlink(base, recursive = TRUE)

cat(sprintf("  fixture                  : 13 points -> 9 after QC -> 5 in the store (windows 3, 5)\n"))
cat(sprintf("  patch vs raster formula  : exact, both windows\n"))
cat(sprintf("  n_cores = 2 vs 1         : identical arrays\n"))

.report(ok, "test_prepare")
