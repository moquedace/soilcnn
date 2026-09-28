# ══════════════════════════════════════════════════════════════════════════════
# P1 -- dsm_prepare() reproduces stages 01 and 02 on the real data
#
# WHAT THIS PROVES, AND WHY IT HAS TO BE PROVED BEFORE 01 AND 02 CHANGE.
#
# dsm_prepare() replaces 1,300 lines of example script with one function. The
# plan is for 01 and 02 to become thin callers of it -- but only once it is
# shown to build THE SAME STORE, because everything downstream (every tuning
# run, the final model, the map) was built on the one currently on disk. A
# store that differed in one channel's order or one point's patch would make
# every later comparison a comparison between two experiments.
#
# So this script runs dsm_prepare() with the settings 01 and 02 recorded --
# read back from target_config.csv and the manifest, not typed again -- into a
# SEPARATE directory, and compares every artefact against the current one.
#
# WHY A SEPARATE DIRECTORY. Stage 02 skipped any window file already on disk at
# the right size. Written over the current store, a comparison would compare
# the old patches with themselves and pass having checked nothing.
#
# WHAT IS EXPECTED TO DIFFER, AND ONLY THAT:
#   * the column target_log1p is called target_transform (the point contract's
#     name, and the same column whatever the transform); median_target_log1p
#     in dataset_check is median_target_transform accordingly
#   * target_config.csv gains fields (the transform, the dummy rule, the NA
#     floors as rules) -- compared on the fields both have
#   * the store gains files: recipe.rds and copies of the four tables it needs
#   * channel_risk.csv, in has_na only: stage 01 counted NA after the QC had
#     dropped every row with one, so has_na could never fire (p1_11)
#   * extracted_at, of course
#
# Everything else must be identical: the patches bit for bit, the tables value
# for value.
#
# COST: one extraction of the store at n_cores = physical - 1. The dev store is
# 1.7 GB; reading both copies of each window to compare them needs ~2.5 GB.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/checks/_p1_prepare_check.R")
# ══════════════════════════════════════════════════════════════════════════════

rm(list = ls())
gc()

options(width = 200)

# The project root, found from where this file is: Rscript's --file, then the
# source() frame, then the working directory, climbing to R/cnn_architecture.R.
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
source(file.path(project_root, "utils", "install_load_pkg.R"))
install_load_pkg(c("sf", "terra", "dplyr", "readr", "tibble", "janitor", "purrr"))
pkgload::load_all(project_root)

target_label <- "soc_stock_0_5cm"

# ── The current store, and where the new one goes ─────────────────────────────

old_store <- file.path(project_root, "outputs", "patches",  "soc_stock_modeling", target_label)
old_meta  <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
old_pts   <- file.path(project_root, "data", "processed", "soc_stock_modeling",
                       target_label, "full_modeling_dataset_raw.csv")
p1_dir    <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling",
                       target_label, "capability_sweep", "p1_prepare")

for (f in c(file.path(old_store, "patch_manifest.rds"),
            file.path(old_meta, "target_config.csv"), old_pts)) {
  if (!file.exists(f)) stop("The current store is incomplete, missing: ", f,
                            call. = FALSE)
}

# ── The settings 01 and 02 used, read back rather than typed again ────────────
#
# target_config.csv and the manifest recorded almost everything. What they did
# not record is copied from 01 and marked: the temperature floor's PATTERN (01
# recorded the threshold but not the regex it applied it through), the target
# rule (target_native <= 0 dropped), and the dev subsample's three numbers.
# The subsample is then CHECKED rather than trusted: its description string is
# recorded, and the new one must equal it (p1_02).

tconf  <- safe_read_csv2(file.path(old_meta, "target_config.csv"))
man    <- readRDS(file.path(old_store, "patch_manifest.rds"))

# A PROOF FOR AS LONG AS THE OLD STORE IS ON DISK. It passed on 2026-09-26,
# 19 of 19, and 01 then became a caller of dsm_prepare(). It is run again
# whenever the extraction changes -- the first change was reading the centre
# values in the same pass as the patches, which retired terra::extract().
# Once 01 has run again, the store on disk IS dsm_prepare()'s, and this script
# would compare the function with itself -- a pass that proves nothing. The
# old 01 wrote soc_gpkg_file into target_config.csv and dsm_prepare() does
# not, which is how that is told.
if (!"soc_gpkg_file" %in% names(tconf)) {
  stop("The store on disk was built by dsm_prepare(), not by the old 01 + 02.\n",
       "  P1 compared the two once, on 2026-09-26 (19/19, patches bit for bit),\n",
       "  and has nothing left to compare. See docs/project_log.md.", call. = FALSE)
}
splitc <- function(s) if (is.na(s) || !nzchar(s)) character(0) else strsplit(s, ";", fixed = TRUE)[[1]]

settings <- list(
  gpkg        = tconf$soc_gpkg_file[1],
  raster_dir  = tconf$predictor_raster_dir[1],
  target      = tconf$target_col[1],
  target_unit = tconf$target_unit[1],
  drop        = splitc(tconf$manual_predictor_drop[1]),
  percentage  = splitc(tconf$percentage_predictor_patterns[1]),
  na_below    = stats::setNames(as.numeric(tconf$temperature_min_valid_celsius[1]),
                                "surface_temperature_celsius$"),     # from 01
  windows     = as.integer(trimws(strsplit(man$windows_extracted[1], ",")[[1]])),
  chunk_nrows = as.integer(man$chunk_nrows_used[1]),
  transform   = man$target_transform[1],
  target_min  = 0,                                                   # from 01
  subsample   = if (identical(tconf$run_profile[1], "dev"))
    list(frac = 0.10, block_size = 2, seed = 20260914L) else NULL)   # from 01

message("\n", strrep("=", 78))
message("P1 -- dsm_prepare() against the store stages 01 and 02 built")
message(strrep("=", 78))
message("  run profile : ", tconf$run_profile[1], " -- ", tconf$subsample[1])
message("  rasters     : ", settings$raster_dir)
message("  windows     : ", paste(settings$windows, collapse = ", "),
        " | transform: ", settings$transform)
message("  new store   : ", p1_dir)
message(strrep("=", 78), "\n")

# Exactly as 01 read it.
soc_sf <- sf::st_read(settings$gpkg, quiet = TRUE) %>%
  janitor::clean_names() %>%
  dplyr::mutate(profile_id = as.character(profile_id))

t0 <- Sys.time()
st <- dsm_prepare(
  points = soc_sf, target = settings$target, raster_dir = settings$raster_dir,
  windows = settings$windows, out_dir = p1_dir,
  percentage = settings$percentage, dummy = "auto", drop = settings$drop,
  na_below = settings$na_below, percentage_limits = c(0, 100),
  transform = settings$transform, target_min = settings$target_min,
  subsample = settings$subsample, target_label = target_label,
  target_unit = settings$target_unit, chunk_nrows = settings$chunk_nrows,
  n_cores = NULL, overwrite = TRUE)
minutes <- as.numeric(difftime(Sys.time(), t0, units = "mins"))

# ── The comparison ────────────────────────────────────────────────────────────

required <- sprintf("p1_%02d", 1:19)
L <- check_ledger("P1")

# Two tables are the same when their VALUES are: read both the same way, and
# drop what readr attaches (spec, problems) and what a tibble carries (row
# names), then identical(). Byte identity is reported alongside -- a table can
# hold the same values in different text, and when it does that is worth
# knowing, but it is not the claim.
norm <- function(d) {
  d <- as.data.frame(d)
  attr(d, "spec") <- NULL
  attr(d, "problems") <- NULL
  rownames(d) <- NULL
  d
}
same_csv <- function(a_path, b_path, rename = NULL) {
  a <- norm(safe_read_csv2(a_path)); b <- norm(safe_read_csv2(b_path))
  if (!is.null(rename)) names(a)[names(a) == names(rename)] <- rename[[1]]
  bytes <- identical(unname(tools::md5sum(a_path)), unname(tools::md5sum(b_path)))
  same  <- identical(a, b)
  why <- if (same) "" else if (!identical(dim(a), dim(b))) {
    sprintf(" | dims %s vs %s", paste(dim(a), collapse = "x"), paste(dim(b), collapse = "x"))
  } else if (!identical(names(a), names(b))) {
    paste0(" | columns differ: ", paste(setdiff(union(names(a), names(b)),
                                               intersect(names(a), names(b))), collapse = ", "))
  } else {
    bad <- names(a)[!vapply(names(a), function(n) identical(a[[n]], b[[n]]), logical(1))]
    paste0(" | column(s) differ: ", paste(utils::head(bad, 5), collapse = ", "))
  }
  list(ok = same, measured = sprintf("%d x %d | values identical: %s | bytes identical: %s%s",
                                     nrow(b), ncol(b), same, bytes, why))
}

new_man <- readRDS(file.path(st$store_dir, "patch_manifest.rds"))

ledger_check(L, "p1_01", "the new store was written and is complete",
             isTRUE(new_man$store_complete[1]),
             sprintf("%d points | %.1f min at %d core(s) | %s read(s) per band",
                     new_man$n_points_valid[1], minutes, st$recipe$n_cores,
                     format(st$recipe$n_reads_per_band, big.mark = ",")))

ledger_check(L, "p1_02", "the same subsample (or none)", {
  a <- tconf$subsample[1]; b <- safe_read_csv2(file.path(st$metadata_dir, "target_config.csv"))$subsample[1]
  list(ok = identical(a, b), measured = b)
})

ledger_check(L, "p1_03", "the same predictors, in the same order",
             identical(man$predictor_cols_final[1], new_man$predictor_cols_final[1]),
             sprintf("%d channel(s)", new_man$n_channels[1]))

# (stage 02 already renamed target_log1p to target_transform in patch_meta.csv)
ledger_check(L, "p1_04", "patch_meta.csv: the same points, row for row",
             same_csv(file.path(old_store, "patch_meta.csv"),
                      file.path(st$store_dir, "patch_meta.csv")))

# The patches, one window at a time -- each copy is up to 1.2 GB, so both
# are released before the next window is read.
for (k in seq_along(settings$windows)) {
  w <- settings$windows[k]
  id <- sprintf("p1_%02d", 4L + k)
  ledger_check(L, id, sprintf("patches_w%02d.rds: identical, bit for bit", w), {
    a <- readRDS(patch_window_path(old_store, w))
    b <- readRDS(patch_window_path(st$store_dir, w))
    res <- list(ok = identical(a, b),
                measured = sprintf("%s | max |diff| %s", paste(dim(b), collapse = " x "),
                                   if (identical(dim(a), dim(b)))
                                     format(max(abs(a - b), na.rm = TRUE)) else "dims differ"))
    rm(a, b); invisible(gc(verbose = FALSE))
    res
  })
}
# The ids above are p1_05 .. p1_07 for three windows. If the store had a
# different number of windows the promised ids would not all appear, and the
# verdict would say which never ran -- rather than this script renumbering
# itself to fit.

# By VALUE for numbers: stage 02 wrote chunk_nrows_used as 1000 (a double,
# from `chunk_nrows <- 1000`) and dsm_prepare() writes 1000L. The same number
# in two storage types is not a difference -- align_points_to_meta() carries
# that lesson already. Everything else, identical().
ledger_check(L, "p1_08", "the manifest, field by field (extracted_at aside)", {
  fields <- setdiff(intersect(names(man), names(new_man)), "extracted_at")
  same_field <- function(f) {
    a <- man[[f]][1]; b <- new_man[[f]][1]
    if (is.numeric(a) && is.numeric(b)) identical(as.numeric(a), as.numeric(b))
    else identical(a, b)
  }
  diff <- fields[!vapply(fields, same_field, logical(1))]
  list(ok = length(diff) == 0L,
       measured = if (length(diff)) paste("differ:", paste(diff, collapse = ", ")) else
         sprintf("%d field(s) identical", length(fields)))
})

ledger_check(L, "p1_09", "predictor_type_table.csv",
             same_csv(file.path(old_meta, "predictor_type_table.csv"),
                      file.path(st$metadata_dir, "predictor_type_table.csv")))
ledger_check(L, "p1_10", "qc_table.csv",
             same_csv(file.path(old_meta, "qc_table.csv"),
                      file.path(st$metadata_dir, "qc_table.csv")))
# The one table EXPECTED to differ, and only in has_na. Stage 01 counted NA
# after the QC had dropped every row with one, so its count was 0 for every
# channel; dsm_prepare() counts over every point that has data in at least
# one channel (R/prepare.R, step 9). So: the columns that describe a channel
# identical, constant and near_constant exactly where they were, has_na only
# where stage 01 said nothing -- and the channels it now names, reported.
ledger_check(L, "p1_11", "channel_risk.csv: identical but for has_na, which 01 could not fire", {
  a  <- norm(safe_read_csv2(file.path(old_meta, "channel_risk.csv")))
  b  <- norm(safe_read_csv2(file.path(st$metadata_dir, "channel_risk.csv")))
  ra <- dplyr::coalesce(as.character(a$risk), "")
  rb <- dplyr::coalesce(as.character(b$risk), "")
  cols <- c("predictor", "type", "n_unique", "min_value", "max_value")
  same_shape <- identical(dim(a), dim(b)) && identical(names(a), names(b))
  same_cols  <- same_shape && identical(a[cols], b[cols])
  same_const <- same_shape && identical(ra == "constant", rb == "constant") &&
    identical(ra == "near_constant", rb == "near_constant")
  only_new   <- same_shape && all(ra[rb == "has_na"] == "") && !any(ra == "has_na") &&
    all(a$n_na_at_points == 0L)
  named <- if (same_shape) b$predictor[rb == "has_na"] else character(0)
  list(ok = same_cols && same_const && only_new,
       measured = sprintf("%d x %d | descriptive columns identical: %s | constant / near_constant unchanged: %s | has_na now names %d channel(s)%s",
                          nrow(b), ncol(b), same_cols, same_const, length(named),
                          if (length(named)) paste0(": ", paste(utils::head(named, 5), collapse = ", "),
                                                    if (length(named) > 5L) ", ..." else "") else ""))
})
ledger_check(L, "p1_12", "channel_invalidation.csv (the window rule's blame)",
             same_csv(file.path(old_meta, "patches", "channel_invalidation.csv"),
                      file.path(st$metadata_dir, "patches", "channel_invalidation.csv")))
ledger_check(L, "p1_13", "qc_summary.csv",
             same_csv(file.path(old_meta, "qc_summary.csv"),
                      file.path(st$metadata_dir, "qc_summary.csv")))
ledger_check(L, "p1_14", "the point table (target_log1p named target_transform)",
             same_csv(old_pts, st$points_file,
                      rename = c(target_log1p = "target_transform")))
ledger_check(L, "p1_15", "raster_table_used.csv",
             same_csv(file.path(old_meta, "raster_table_used.csv"),
                      file.path(st$metadata_dir, "raster_table_used.csv")))
ledger_check(L, "p1_16", "dataset_check.csv (median_target_log1p renamed)",
             same_csv(file.path(old_meta, "dataset_check.csv"),
                      file.path(st$metadata_dir, "dataset_check.csv"),
                      rename = c(median_target_log1p = "median_target_transform")))

ledger_check(L, "p1_17", "patch_sample.rds: the same six patches", {
  a <- readRDS(file.path(old_store, "patch_sample.rds"))
  b <- readRDS(file.path(st$store_dir, "patch_sample.rds"))
  same_w <- identical(a$windows, b$windows)
  same_m <- identical(as.character(a$meta$sample_id), as.character(b$meta$sample_id))
  list(ok = same_w && same_m && identical(a$predictors, b$predictors),
       measured = sprintf("arrays identical: %s | sample ids identical: %s", same_w, same_m))
})

# The smallest window only: these two checks are about the tables and the
# transform, and each full load would be another 1.7 GB for nothing.
w_min <- min(settings$windows)
d_new <- suppressMessages(dsm_load(st, windows = w_min, verbose = FALSE))
ledger_check(L, "p1_18", "dsm_load(store) alone, and it reads the transform back",
             inherits(d_new, "dsm_data") && identical(d_new$transform$name, settings$transform),
             sprintf("%d points | transform %s | cell %s", nrow(d_new$store$meta),
                     d_new$transform$name, format(d_new$cell_size, digits = 8)))

ledger_check(L, "p1_19", "dsm_load of the old and of the new store agree", {
  d_old <- suppressMessages(dsm_load(
    patch_dir = old_store, points = old_pts,
    type_table = file.path(old_meta, "predictor_type_table.csv"),
    raster_table = file.path(old_meta, "raster_table_used.csv"),
    target_col = settings$target, windows = w_min, verbose = FALSE))
  same_meta  <- identical(norm(d_old$store$meta)$sample_id, norm(d_new$store$meta)$sample_id)
  same_types <- identical(norm(d_old$type_table), norm(d_new$type_table))
  same_cell  <- identical(d_old$cell_size, d_new$cell_size)
  rm(d_old); invisible(gc(verbose = FALSE))
  list(ok = same_meta && same_types && same_cell,
       measured = sprintf("meta %s | types %s | cell size %s", same_meta, same_types, same_cell))
})

v <- ledger_verdict(L, required, file.path(p1_dir, "p1_checks.csv"))

if (v$pass) {
  message("\ndsm_prepare() builds the store 01 and 02 built.")
  message("The new store at ", p1_dir, " can be deleted -- it is a copy.")
} else {
  message("\nThe stores differ -- the table above says where. Do NOT run 01 until ",
          "that is understood: it calls dsm_prepare() with overwrite = TRUE, and ",
          "would replace the store this script compares against.")
}
