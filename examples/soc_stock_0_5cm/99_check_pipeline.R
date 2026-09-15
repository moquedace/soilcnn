project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("dplyr", "readr", "tibble", "purrr", "stringr", "terra")
install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

# The generic checks live in R/ (the framework); this script is only the
# example's orchestrator -- anyone using the framework on other data gets the
# same checks without copying anything from here.
# One source() instead of several, in a dependency order that is not
# guessable. See R/load_all.R.
source(file.path(project_root, "R", "load_all.R"))

# ══════════════════════════════════════════════════════════════════════════════
# 99 - pipeline quality checkpoint (check & recheck)
#
# Runs automatic checks over whatever has already executed (01, 02, ...),
# against known thresholds and against the internal consistency of the files
# themselves. The aim is to catch STRUCTURAL problems -- the kind of the PNV
# clip bug, which silently discarded ~55% of the data through two months of
# processing -- in the minute a stage finishes, not weeks later.
#
# How to grow this: each stage has its own section. After adding a new stage to
# the pipeline, add a section here following the same shape (see add_check()
# below) and run the whole script again -- it re-verifies everything that has
# run, not only the new part.
#
# Usage: source() it directly, no parameters. Runs in seconds: it reads small
# CSVs and metadata only, and NEVER loads the patch arrays, which are tens of
# GB. The manifest already carries the aggregate numbers.
# ══════════════════════════════════════════════════════════════════════════════

target_label <- "soc_stock_0_5cm"

data_dir     <- file.path(project_root, "data", "processed", "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
patch_dir    <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
patch_meta_dir <- file.path(metadata_dir, "patches")

# -- Checking infrastructure ---─────────────────────────────────────────────────

.results <- tibble::tibble(
  stage = character(), check = character(), status = character(), detail = character()
)

# The whole report leaves through ONE channel.
#
# message() writes to stderr and print()/tibble to stdout; in the RStudio
# console the two interleave and lines run together ("channel_risk.csv  [OK]
# 01 | ..."), because each channel flushes on its own schedule. This script
# alternates a line of text with a print()ed tibble throughout, so the order is
# only guaranteed if everything leaves the same way -- and print() cannot be
# sent to stderr, so it is the text that moves to stdout.
#
# .say() mimics message(): pastes its arguments and adds the line break.
.say <- function(...) cat(paste0(...), "
", sep = "")

add_check <- function(stage, check, status, detail = "") {
  .results <<- dplyr::bind_rows(
    .results,
    tibble::tibble(stage = stage, check = check, status = status, detail = detail)
  )
  icon <- switch(status, PASS = "  [OK]", WARN = "[WARN]", FAIL = "[FAIL]", "  [?]")
  .say(sprintf("%s %-6s | %-45s | %s", icon, stage, check, detail))
}

check_exists <- function(stage, label, path) {
  ok <- file.exists(path)
  add_check(stage, label, if (ok) "PASS" else "FAIL",
            if (ok) path else paste("NAO ENCONTRADO:", path))
  ok
}

check_threshold <- function(stage, label, value, warn_above, fail_above, unit = "%") {
  status <- if (value > fail_above) "FAIL" else if (value > warn_above) "WARN" else "PASS"
  add_check(stage, label, status,
            sprintf("value=%.2f%s (warn>%.1f%s, fail>%.1f%s)",
                    value, unit, warn_above, unit, fail_above, unit))
  status
}

check_equal <- function(stage, label, a, b, name_a = "a", name_b = "b") {
  ok <- isTRUE(all.equal(a, b, tolerance = 1e-6))
  add_check(stage, label, if (ok) "PASS" else "FAIL",
            sprintf("%s=%s | %s=%s", name_a, a, name_b, b))
  ok
}

.say("\n", strrep("=", 90))
.say("99 - pipeline quality checkpoint: ", target_label)
.say(strrep("=", 90), "\n")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 01 - preparing the point table
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 01: dataset preparation --\n")

f_qc      <- file.path(metadata_dir, "qc_summary.csv")
f_dscheck <- file.path(metadata_dir, "dataset_check.csv")

f_ptype   <- file.path(metadata_dir, "predictor_type_table.csv")

f_rtable  <- file.path(metadata_dir, "raster_table_used.csv")
f_pmeta   <- file.path(metadata_dir, "point_metadata.csv")   # sample_id, x, y
f_tconfig <- file.path(metadata_dir, "target_config.csv")

# The six per-split CSVs are gone: no code read the *_scaled ones, and the
# *_raw ones were a filter() of the single dataset. With per-fold scaling,
# "the scaled dataset" stopped existing as one object. In their place:
# qc_table.csv (the rules stage 02 obeys) and channel_risk.csv (the channels
# that have historically broken the MAP).
f_dataset <- file.path(data_dir, "full_modeling_dataset_raw.csv")
f_qctable <- file.path(metadata_dir, "qc_table.csv")
f_crisk   <- file.path(metadata_dir, "channel_risk.csv")

# ── Run profile ───────────────────────────────────────────────────────────────
# A development run writes the SAME files, with the SAME names, on a tenth of
# the data. Without this in plain sight, a month from now a subsample CCC is
# indistinguishable from a result.
run_profile <- "full"
if (file.exists(f_tconfig)) {
  .tc <- safe_read_csv2(f_tconfig)
  if ("run_profile" %in% names(.tc)) run_profile <- as.character(.tc$run_profile[1])
  if ("run_profile" %in% names(.tc) && !identical(.tc$run_profile[1], "full")) {
    .say(strrep("!", 90))
    .say("RUN PROFILE: ", toupper(.tc$run_profile[1]),
         "  --  THIS IS NOT A RESULT RUN")
    if ("subsample" %in% names(.tc)) .say("  ", .tc$subsample[1])
    .say("  Comparable only with another run of the same profile.",
         "  See docs/reference_performance.md")
    .say(strrep("!", 90), "
")
  }
}

# Gone with the split: split_check.csv, split_bin_check.csv,
# scaled_extreme_check.csv and predictor_scaling.csv. Stage 01 no longer
# decides who trains, and the scaling is now part of the fitted model (stage
# 04 writes it next to the weights), not a property of the dataset.
files_01 <- c(f_qc, f_dscheck, f_ptype, f_rtable, f_tconfig,
             f_dataset, f_qctable, f_crisk, f_pmeta)
all_01_exist <- all(purrr::map_lgl(files_01, ~ check_exists("01", basename(.x), .x)))

if (all_01_exist) {

  qc      <- safe_read_csv2(f_qc)
  dscheck <- safe_read_csv2(f_dscheck)
  ptype   <- safe_read_csv2(f_ptype)
  rtable  <- safe_read_csv2(f_rtable)
  pmeta   <- safe_read_csv2(f_pmeta)
  tconfig <- safe_read_csv2(f_tconfig)

  # The most important check here: the share of rows dropped for a PREDICTOR
  # problem (not a target one). A bad target -- poor spline, NA, <= 0 -- is
  # normal and expected; too many predictor problems is the signature of the
  # class of bug that cost ~55% of the data for two months (clip vs NA).
  pct_pred_problem <- 100 * qc$n_predictor_problem / qc$n_rows_extracted
  check_threshold("01", "% of rows with a PREDICTOR problem (not the target)",
                  pct_pred_problem, warn_above = 2, fail_above = 10)

  # Predictor count consistent across EVERY file that should agree.
  # predictor_scaling.csv is NOT in this list any more. Stage 01 stopped
  # writing it when the scaling became a property of the FITTED MODEL: it is
  # fitted on each fold's training rows and written next to the weights by
  # stage 04. Checking it here would be checking for a file that must not
  # exist -- and this line survived the refactor as a reference to an object
  # nothing creates, which is why the 99 crashed instead of reporting.
  n_pred_rtable  <- nrow(rtable)
  n_pred_ptype   <- nrow(ptype)
  n_pred_dscheck <- dscheck$n_predictors[1]
  n_pred_tconfig <- tconfig$n_predictors_final[1]

  check_equal("01", "n_predictors: raster_table vs predictor_type_table",
              n_pred_rtable, n_pred_ptype, "raster_table", "predictor_type")
  check_equal("01", "n_predictors: raster_table vs dataset_check",
              n_pred_rtable, n_pred_dscheck, "raster_table", "dataset_check")
  check_equal("01", "n_predictors: raster_table vs target_config",
              n_pred_rtable, n_pred_tconfig, "raster_table", "target_config")

  # dummy + percentage + continuous must sum to the predictor total
  n_type_sum <- dscheck$n_dummy_predictors[1] + dscheck$n_percentage_predictors[1] +
    dscheck$n_continuous_predictors[1]
  check_equal("01", "dummy + percentage + continuous == total predictors",
              n_type_sum, n_pred_rtable, "soma_tipos", "total")

  # The 70/15/15 proportion check is gone with the split itself. Stage 01
  # decides no roles: who trains, who scores and who is held out is carved by
  # a fold plan in stage 03, from coordinates, and fold_sizes.csv is where
  # those proportions get checked -- against the plan that will actually run.
  
  # target_native and target_log1p agree (log1p(native) == log1p) -- checked
  # indirectly through the median already computed in dataset_check.csv
  implied_log1p <- log1p(dscheck$median_target[1])
  check_equal("01", "median_target_log1p == log1p(median_target)",
              round(dscheck$median_target_log1p[1], 4), round(implied_log1p, 4),
              "salvo", "recalculado")

  # The dataset, the point table and the QC summary must agree on how many
  # rows survived. Stage 01 writes them from three different objects in three
  # different blocks, so a disagreement means one was built from a stale copy --
  # which is how a store once ended up with more patches than there were points.
  # col_select = 1 keeps the read fast even with 180+ columns.
  n_dataset <- nrow(safe_read_csv2(f_dataset, col_select = 1))
  check_equal("01", "rows: dataset vs point_metadata",
              n_dataset, nrow(pmeta), "dataset", "point_metadata")
  check_equal("01", "rows: dataset vs qc_summary (after QC)",
              n_dataset, qc$n_rows_after_qc[1], "dataset", "qc_summary")
  check_equal("01", "rows: dataset vs dataset_check",
              n_dataset, dscheck$n_rows[1], "dataset", "dataset_check")

  # qc_table holds one rule per predictor, in type_table's order -- and that
  # order is what stage 02 uses to know which rule applies to which band.
  qctable <- safe_read_csv2(f_qctable)
  add_check("01", "qc_table.csv in the same order as predictor_type_table.csv",
            if (identical(qctable$predictor, ptype$predictor)) "PASS" else "FAIL",
            sprintf("%d rules / %d predictors", nrow(qctable), nrow(ptype)))

  # GUARD: no constant channel may have survived the drop.
  # A channel constant AT THE POINTS is not constant on the MAP -- it switches
  # on over glaciers, islands, ocean -- and because its gradient is always
  # zero, its weights stay at random initialisation and apply a bias exactly
  # where the network is extrapolating. If this fails, add the channel to
  # manual_predictor_drop.
  crisk    <- safe_read_csv2(f_crisk)
  n_const  <- sum(crisk$risk == "constant", na.rm = TRUE)
  n_withna <- sum(crisk$risk == "has_na",   na.rm = TRUE)

  # A CONSTANT CHANNEL MEANS TWO DIFFERENT THINGS, and only one of them is a
  # defect.
  #
  # At full size, constant is a real statement about the data: the channel
  # carries no information at any profile while being non-zero somewhere on the
  # map, so its weights never get a gradient and stay at random init exactly
  # where the network extrapolates. That is a FAIL.
  #
  # On a 10% subsample it is usually a statement about the DRAW. A rare class --
  # glaciers, evaporites, marine intertidal -- is all-zero at 4k points because
  # the subsample missed the handful of profiles that carry it. Failing on that
  # would train people to ignore this check during exactly the runs it is
  # cheapest to run, and acting on it would change the full run's predictor set
  # from an artefact of a draw.
  .const_status <- if (n_const == 0L) "PASS"
                   else if (identical(run_profile, "full")) "FAIL" else "WARN"
  add_check("01", "no constant channel survived the drop",
            .const_status,
            sprintf("%d constant(s)%s", n_const,
                    if (.const_status == "WARN")
                      " -- dev profile: re-read this at full size before acting"
                    else ""))

  # THE CHANNEL THAT ACTUALLY STOPS A RUN IS A CONTINUOUS ONE.
  #
  # build_fold_cache() refuses to z-score a channel whose sd is zero over a
  # fold's TRAINING rows, and it refuses by stopping -- correctly, since a
  # constant channel cannot be standardised. Dummies are safe: fit_scaling()
  # gives them centre 0 and scale 1 and never divides by their spread.
  #
  # So the killer is a continuous channel with almost no spread: it survives
  # the global check here, then goes to zero inside one fold and stops stage 03
  # partway through, after the patches have been extracted. Cheap to see now,
  # expensive to meet then.
  #
  # Relative to the mean, because these channels have wildly different units: a
  # sd of 0.001 is nothing for elevation and everything for a vegetation index.
  if (file.exists(f_dataset) && exists("ptype")) {
    .cont <- ptype$predictor[!ptype$is_dummy & !ptype$is_percentage]
    .cont <- intersect(.cont, names(safe_read_csv2(f_dataset, n_max = 1)))
    if (length(.cont) > 0L) {
      .d  <- safe_read_csv2(f_dataset, col_select = dplyr::all_of(.cont))
      .sd <- vapply(.d, function(z) stats::sd(z, na.rm = TRUE), numeric(1))
      .mu <- vapply(.d, function(z) mean(abs(z), na.rm = TRUE), numeric(1))
      .cv <- .sd / pmax(.mu, .Machine$double.eps)
      .risky <- names(.cv)[!is.finite(.cv) | .cv < 1e-6]
      add_check("01", "continuous channels can be z-scored in any fold",
                if (length(.risky) == 0L) "PASS" else "FAIL",
                if (length(.risky) == 0L)
                  sprintf("%d continuous channel(s), smallest sd/mean = %.2e",
                          length(.cont), min(.cv[is.finite(.cv)]))
                else
                  paste0(length(.risky),
                         " with no usable spread -- stage 03 will STOP on the ",
                         "first fold where they go flat: ",
                         paste(utils::head(.risky, 6), collapse = ", ")))
    }
  }
  add_check("01", "channels with NA at the points",
            if (n_withna == 0L) "PASS" else "WARN",
            sprintf("%d channel(s) with NA -- see channel_risk.csv", n_withna))

  # The spatial-overlap report used to live here, comparing the fixed split
  # stage 01 wrote. There is no split here any more: it is carved by the fold
  # plan in stage 03, and that is where the leakage is measured -- per fold, on
  # the plan that will actually be used. See fold_leakage_report().
  add_check("01", "point table carries no role column",
            if (!any(c("dataset_role", "split_bin") %in% names(pmeta)))
              "PASS" else "FAIL",
            paste("columns:", paste(names(pmeta), collapse = ", ")))

} else {
  .say("Stage 01 incomplete -- skipping content checks.")
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 02 - patch extraction
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 02: patch extraction --\n")

# Stage 02 writes ONE file per window instead of a single .rds holding three
# splits, and one patch_meta.csv instead of three meta_*.csv: the split became
# an index, not a property of the stored data.
f_manifest     <- file.path(patch_meta_dir, "patch_manifest.csv")
f_pfiles       <- file.path(patch_meta_dir, "patch_files.csv")
f_blame        <- file.path(patch_meta_dir, "channel_invalidation.csv")
f_patch_meta   <- file.path(patch_dir, "patch_meta.csv")
f_manifest_rds <- file.path(patch_dir, "patch_manifest.rds")

files_02 <- c(f_manifest, f_pfiles, f_blame, f_patch_meta, f_manifest_rds)

# NOT STARTED vs FAILED: different things, and only one deserves a FAIL.
#
# The workflow here is "script 1 -> check, script 2 -> check", so this script
# runs many times with later stages still to come. If a stage that never ran
# counted as a failure, every intermediate run would end in FAIL and the signal
# would lose its value exactly when it is most useful.
#
# The rule: no files present = has not run yet (informational). SOME files
# present = ran half way, and that is a problem.
n_02_presentes <- sum(file.exists(files_02))

if (n_02_presentes == 0L) {
  .say("Stage 02 not started -- no files in: ", patch_dir)
  all_02_exist <- FALSE
} else {
  all_02_exist <- all(purrr::map_lgl(files_02, ~ check_exists("02", basename(.x), .x)))
  if (!all_02_exist) {
    .say("  WARNING: stage 02 ran PARTIALLY (", n_02_presentes, " of ",
            length(files_02), " files). An incomplete store is worse than ",
            "none at all -- run 02 again.")
  }
}

if (all_02_exist) {

  manifest <- safe_read_csv2(f_manifest)

  # THE MOST IMPORTANT CHECK IN THIS WHOLE SCRIPT: the share of profiles
  # dropped during patch extraction. Before the PNV clip fix this ran
  # consistently around 55%; after it, under 2% is expected. If it climbs back
  # above 10%, SOMETHING REGRESSED -- stop and investigate before spending days
  # of tuning and prediction on broken data.
  # It is one number now: stage 02 extracts every point in a single pass.
  check_threshold("02", "pct_removed (full window)",
                  manifest$pct_removed[1], warn_above = 2, fail_above = 10)

  # COMPANION: the number above says HOW MUCH was lost; this one says WHICH
  # channel lost it. valid_common is an AND over every channel, so on its own
  # it can never name the culprit -- and without a culprit, a channel with
  # sparse NA only shows up as a hole in the map, weeks later.
  # THE THRESHOLD GOES ON THE SOLE-CAUSE COLUMN, NOT ON pct_invalidated.
  #
  # pct_invalidated counts every point where a channel was non-finite,
  # including points where all 181 were. When a point sits on a coastline the
  # whole stack is nodata there, so EVERY channel scores that point and every
  # channel ties -- which made this check WARN at 1.01% on the dev run while
  # no channel was responsible for anything.
  #
  # pct_sole_cause is the number that describes a CHANNEL: points lost to it
  # and nothing else. That is the sparse-NA case, and it is the one that turns
  # into holes in the map.
  blame <- safe_read_csv2(f_blame)
  has_sole <- "pct_sole_cause" %in% names(blame) && nrow(blame) > 0
  worst_sole <- if (has_sole) max(blame$pct_sole_cause, na.rm = TRUE) else 0
  worst_any  <- if (nrow(blame) > 0) max(blame$pct_invalidated, na.rm = TRUE) else 0

  check_threshold("02", "worst channel, % of points it ALONE invalidated",
                  worst_sole, warn_above = 0.5, fail_above = 2)

  if (has_sole && worst_sole > 0) {
    top <- blame[which.max(blame$pct_sole_cause), ]
    add_check("02", "channel with the most sole-cause losses", "PASS",
              sprintf("%s (%s): %.3f%% alone", top$predictor[1], top$type[1],
                      top$pct_sole_cause[1]))
  } else if (worst_any > 0) {
    # Reported as information, not as a warning: nothing here is actionable,
    # and a WARN that cannot be acted on is one people learn to scroll past.
    add_check("02", "points lost where the WHOLE stack is nodata", "PASS",
              sprintf(paste0("%.2f%% of points, no channel the sole cause -- ",
                             "coastline / water / raster edge, not patchy ",
                             "coverage"), worst_any))
  }

  # Patches stored unscaled: stage 03 applies the fold's scaling. A
  # pre-scaled store is tied to ONE split and is useless for resampling.
  add_check("02", "patches stored WITHOUT scaling",
            if (isFALSE(manifest$scaling_applied[1])) "PASS" else "FAIL",
            paste("scaling_applied =", manifest$scaling_applied[1]))

  # n_channels matches what stage 01 prepared -- not a fixed number, because
  # dropping a problematic predictor is a legitimate action and must not break
  # the check. `ptype` comes from the stage 01 block, kept because that block
  # only runs when 01's files exist and stage 02 has to survive without it.
  if (exists("ptype")) {
    check_equal("02", "n_channels: manifest vs predictor_type_table",
                manifest$n_channels[1], nrow(ptype), "manifest", "01")
  } else {
    add_check("02", "n_channels: manifest vs predictor_type_table", "WARN",
              "stage 01 incomplete -- nothing to compare against")
  }

  # WINDOWS: what is checked is the SHAPE, not a frozen list.
  #
  # This used to assert the literal "3, 9, 15", which made the checker wrong
  # for anyone who changed the grid -- and the grid is meant to be changed.
  # What must hold for ANY store is structural: windows are odd (a patch has a
  # centre pixel, and an even side has none) and strictly ascending. Whether
  # THIS run's grid can be served by THIS store is a different question, and
  # check_store_spec() answers it in stage 03 where the grid exists.
  w_store <- suppressWarnings(as.integer(trimws(
    strsplit(as.character(manifest$windows_extracted[1]), ",")[[1]])))
  ok_windows <- length(w_store) > 0L && !anyNA(w_store) &&
                all(w_store %% 2L == 1L) && all(diff(w_store) > 0L)
  add_check("02", "windows_extracted: odd and ascending",
            if (ok_windows) "PASS" else "FAIL",
            paste("value:", manifest$windows_extracted[1]))

  # THE STORE SPEC -- what 02 recorded, against what 01 decided and what the
  # rasters say today.
  #
  # A store carries no visible mark of the target it was built for or the
  # resolution it was cut at. Both have silently changed between runs of this
  # project before, and neither produced an error: the wrong one trains, fits
  # and maps just as well as the right one. These two checks are the whole
  # reason the manifest records a spec at all.
  if (exists("tconfig") && "target_col" %in% names(manifest)) {
    check_equal("02", "target_col: manifest vs target_config",
                as.character(manifest$target_col[1]),
                as.character(tconfig$target_col[1]), "manifest", "01")
  } else if (!"target_col" %in% names(manifest)) {
    add_check("02", "target_col recorded in the manifest", "WARN",
              "store written before the spec was recorded -- re-extract to lock it")
  }

  # requireNamespace, because this is the only place the checker touches terra
  # and a fast structural check should not fail to run over a missing optional.
  if ("cell_size" %in% names(manifest) && file.exists(f_rtable) &&
      requireNamespace("terra", quietly = TRUE)) {
    .cs_store <- suppressWarnings(as.numeric(manifest$cell_size[1]))
    .r1 <- safe_read_csv2(f_rtable)$raster_file[1]
    if (!is.na(.cs_store) && !is.na(.r1) && file.exists(.r1)) {
      .cs_now <- terra::res(terra::rast(.r1))[1]
      add_check("02", "cell_size: manifest vs the rasters now",
                if (abs(.cs_now - .cs_store) < 1e-9) "PASS" else "FAIL",
                sprintf("store %.8f | rasters %.8f", .cs_store, .cs_now))
    }
  }

  # The manifest's n_points_valid matches the real rows of patch_meta.csv
  n_patch_meta <- nrow(safe_read_csv2(f_patch_meta,
                                        col_select = 1))
  check_equal("02", "n_points_valid: manifest vs patch_meta.csv",
              manifest$n_points_valid[1], n_patch_meta, "manifest", "patch_meta")

  # Every window file exists, was verified on write, and is not suspiciously
  # small (a crash mid-write leaves a truncated file)
  pfiles <- safe_read_csv2(f_pfiles)
  missing_pt <- pfiles$file[!file.exists(file.path(patch_dir, pfiles$file))]
  add_check("02", "every window file exists on disk",
            if (length(missing_pt) == 0L) "PASS" else "FAIL",
            if (length(missing_pt) == 0L) sprintf("%d file(s)", nrow(pfiles))
            else paste("missing:", paste(missing_pt, collapse = ", ")))
  # The column is `status` ("written" / "kept" / "size_mismatch"). It was
  # `verified` in the version that wrote tensors; the 99 kept reading the old
  # name and reported "0/3 verified" on a perfect store. A check that reads the
  # wrong column is worse than no check: it spends attention on a false alarm.
  n_ok_files <- sum(pfiles$status %in% c("written", "kept"))
  add_check("02", "every file written without a size error",
            if (n_ok_files == nrow(pfiles)) "PASS" else "WARN",
            sprintf("%d/%d (%s)", n_ok_files, nrow(pfiles),
                    paste(unique(pfiles$status), collapse = ", ")))
  add_check("02", "total patch size is plausible",
            if (sum(pfiles$gb, na.rm = TRUE) > 0.5) "PASS" else "WARN",
            sprintf("%.1f GB em %d file(s)", sum(pfiles$gb, na.rm = TRUE),
                    nrow(pfiles)))

  # -- The strongest check in this file --──────────────────────────────
  # The centre of every patch MUST equal the value the point table holds for
  # that predictor at that point -- it is the same cell of the same raster,
  # reached by two completely independent routes:
  #
  #   point table : terra::extract() over a SpatVector (script 01)
  #   patch store : cellFromXY -> row/col -> patch_cell_index (script 02)
  #
  # A CRS error, a row/column swap, an off-by-one, a channel reordering or a
  # stale raster directory each break that equality. The synthetic tests prove
  # the algebra is right; this proves it was applied to the right place of the
  # REAL raster.
  if (file.exists(f_dataset) && file.exists(f_patch_meta) &&
      file.exists(f_manifest_rds)) {

    pts_all <- safe_read_csv2(f_dataset)
    pmeta   <- safe_read_csv2(f_patch_meta)
    preds   <- strsplit(readRDS(f_manifest_rds)$predictor_cols_final[1], ";")[[1]]

    cc <- try(check_patch_centres(patch_dir,
                                  align_points_to_meta(pts_all, pmeta),
                                  preds), silent = TRUE)

    if (inherits(cc, "try-error")) {
      add_check("02", "patch centre == point-table value", "FAIL",
                paste("error:", conditionMessage(attr(cc, "condition"))))
    } else {
      add_check("02", "patch centre == point-table value",
                if (cc$ok) "PASS" else "FAIL",
                sprintf("%s cells (%s pts x %d channels, window %dx%d) | %d divergent",
                        format(cc$n_cells, big.mark = ","),
                        format(cc$n_points, big.mark = ","),
                        cc$n_channels, cc$window, cc$window, cc$n_mismatch))
      if (!cc$ok) {
        .say("\n  DIVERGENCE at the patch centres -- channels affected:")
        print_wide(dplyr::slice_head(cc$by_channel, n = 10))
        .say("  Largest absolute difference: ", signif(cc$worst, 6))
      }
    }
    rm(pts_all, pmeta); invisible(gc(verbose = FALSE))
  }

  add_check("02", "patch sample saved for visual inspection",
            if (file.exists(file.path(patch_dir, "patch_sample.rds"))) "PASS" else "WARN",
            "patch_sample.rds -- written by stage 02 at no extra cost")

} else {
  .say("Stage 02 incomplete -- skipping content checks.")
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 03 - hyperparameter search (tuning)
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 03: hyperparameter tuning --\n")

tuning_dir <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label)

if (!dir.exists(tuning_dir)) {
  .say("Stage 03 not started -- directory not found: ", tuning_dir)
} else {
  tuning_runs <- list.dirs(tuning_dir, recursive = FALSE, full.names = FALSE)
  if (length(tuning_runs) == 0) {
    .say("Stage 03 incomplete -- no run found in: ", tuning_dir)
  } else {
    # Most recent by name order (run_id is timestamped) -- the same criterion
    # stages 04/05/06 use to resolve "latest".
    tuning_run_id <- sort(tuning_runs, decreasing = TRUE)[1]
    run_dir <- file.path(tuning_dir, tuning_run_id)
    .say("Most recent run: ", tuning_run_id)

    f_grid_csv <- file.path(run_dir, "tune_grid.csv")
    f_grid_rds <- file.path(run_dir, "tune_grid.rds")
    f_cmp_all  <- file.path(run_dir, "comparison", "comparison_all.csv")
    f_cmp_rank <- file.path(run_dir, "comparison", "comparison_ranked.csv")

    files_03 <- c(f_grid_csv, f_grid_rds, f_cmp_all, f_cmp_rank)
    all_03_exist <- all(purrr::map_lgl(files_03, ~ check_exists("03", basename(.x), .x)))

    if (all_03_exist) {

      tune_grid  <- safe_read_csv2(f_grid_csv)
      comparison <- safe_read_csv2(f_cmp_rank)

      # Runs from before resampling have one row per config and no
      # unit_id/fold/seed columns. Filling them with what those rows actually
      # were leaves the rest of this block with a single path.
      if (!"unit_id" %in% names(comparison)) comparison$unit_id <- comparison$config_id
      if (!"fold"    %in% names(comparison)) comparison$fold    <- 1L
      if (!"seed"    %in% names(comparison)) comparison$seed    <- NA_integer_

      n_grid <- nrow(tune_grid)
      n_cmp  <- nrow(comparison)

      # The most important check of this stage: no config in the grid was left
      # behind (silent crash, config wrongly skipped on resume, and so on).
      missing_ids <- setdiff(tune_grid$config_id, comparison$config_id)
      # Count configs in the numerator and configs in the denominator. With
      # units, `n_cmp` is ROWS -- printing "27/3 configs" mixes the two scales
      # and reads as though configs were left over.
      n_cfg_seen <- dplyr::n_distinct(comparison$config_id)
      add_check("03", "every config in the grid has a comparison row",
                if (length(missing_ids) == 0) "PASS" else "FAIL",
                if (length(missing_ids) == 0)
                  sprintf("%d/%d configs (%d units)", n_cfg_seen, n_grid, n_cmp)
                else paste("missing:", paste(missing_ids, collapse = ", ")))

      # No extra row in the comparison that is not in the current grid -- that
      # would mean two runs mixed together (a resume with a swapped tune_grid
      # and no new run_id, for instance).
      extra_ids <- setdiff(comparison$config_id, tune_grid$config_id)
      add_check("03", "no config in the comparison outside the current grid",
                if (length(extra_ids) == 0) "PASS" else "FAIL",
                if (length(extra_ids) == 0) "" else paste("extra:", paste(extra_ids, collapse = ", ")))

      # status == success everywhere. A config that errors still writes a row
      # with status "failed", so anything else here would be unexpected CSV
      # corruption rather than an ordinary training failure.
      n_not_success <- sum(comparison$status != "success", na.rm = TRUE)
      add_check("03", "every row has status == success",
                if (n_not_success == 0) "PASS" else "FAIL",
                sprintf("%d row(s) with status != success", n_not_success))

      # Every row in the comparison needs its .pt checkpoint -- that is the
      # "training really finished" signal resume relies on (see run_cnn_tuning
      # in R/train_cnn.R). A row without one would leave a future resume unsure
      # whether that unit needs retraining.
      # By UNIT: with repetitions, two seeds of one config are two models.
      # Looking by config_id would search for a file that does not exist AND
      # skip checking the ones that do.
      ckpt_files <- file.path(run_dir, "models", paste0(comparison$unit_id, "_best.pt"))
      n_missing_ckpt <- sum(!file.exists(ckpt_files))
      add_check("03", "every unit in the comparison has a .pt checkpoint",
                if (n_missing_ckpt == 0) "PASS" else "FAIL",
                sprintf("%d checkpoint(s) missing", n_missing_ckpt))

      # best_epoch cannot be NA or <= 0: that would mean training never met
      # the early-stopping improvement criterion, which is a broken run.
      n_bad_epoch <- sum(is.na(comparison$best_epoch) | comparison$best_epoch <= 0)
      add_check("03", "best_epoch valid (non-NA, > 0) in every config",
                if (n_bad_epoch == 0) "PASS" else "FAIL",
                sprintf("%d config(s) with an invalid best_epoch", n_bad_epoch))

      # Validation metrics inside a PHYSICALLY plausible range (not NA, CCC in
      # [-1, 1], MAE/RMSE > 0). This does not judge how GOOD the model is --
      # that is a modelling decision, not a structural bug -- it only rejects
      # impossible values, which signal an error in the computation.
      n_na_metrics <- sum(is.na(comparison$val_ccc) | is.na(comparison$val_mae) |
                          is.na(comparison$val_rmse))
      add_check("03", "val_ccc/val_mae/val_rmse free of NA",
                if (n_na_metrics == 0) "PASS" else "FAIL",
                sprintf("%d config(s) with an NA metric", n_na_metrics))

      n_ccc_out_of_range <- sum(comparison$val_ccc < -1 | comparison$val_ccc > 1, na.rm = TRUE)
      add_check("03", "val_ccc within [-1, 1]",
                if (n_ccc_out_of_range == 0) "PASS" else "FAIL",
                sprintf("%d config(s) out of range", n_ccc_out_of_range))

      n_nonpos_error <- sum(comparison$val_mae <= 0 | comparison$val_rmse <= 0, na.rm = TRUE)
      add_check("03", "val_mae and val_rmse > 0",
                if (n_nonpos_error == 0) "PASS" else "FAIL",
                sprintf("%d config(s) with error <= 0", n_nonpos_error))

      # -- Resampling: the bookkeeping of the repetitions --───────────────────────
      f_plan     <- file.path(run_dir, "fold_plan.rds")
      f_byconfig <- file.path(run_dir, "comparison", "comparison_by_config.csv")
      has_plan   <- file.exists(f_plan)

      n_folds_run <- dplyr::n_distinct(comparison$fold)
      n_seeds_run <- dplyr::n_distinct(comparison$seed)

      add_check("03", "resampling plan saved with the run",
                if (has_plan) "PASS" else "WARN",
                if (has_plan) {
                  pl <- readRDS(f_plan)
                  sprintf("%s | %d fold(s)%s", pl$method, pl$n_folds,
                          if (!is.null(pl$params$buffer))
                            sprintf(" | buffer %s", format(pl$params$buffer))
                          else "")
                } else "fold_plan.rds ausente -- run anterior a reamostragem")

      # The grid has to have been trained WHOLE in every fold and every seed.
      # A hole here shows up nowhere else: that config's mean comes from fewer
      # repetitions than the others and the comparison tilts.
      n_expected <- n_grid * n_folds_run * n_seeds_run
      add_check("03", "units = configs x folds x seeds",
                if (n_cmp == n_expected) "PASS" else "WARN",
                sprintf("%d rows | %d configs x %d fold(s) x %d seed(s) = %d",
                        n_cmp, n_grid, n_folds_run, n_seeds_run, n_expected))

      # The seed MUST be the same across every config of a repetition -- that
      # is what makes two configs comparable under the same draw. If each
      # config had its own seed, part of every measured difference would be
      # luck, and nothing in the result would say so.
      if (!all(is.na(comparison$seed))) {
        seeds_per_cfg <- comparison %>%
          dplyr::group_by(config_id) %>%
          dplyr::summarise(s = paste(sort(unique(seed)), collapse = ","),
                           .groups = "drop")
        add_check("03", "same set of seeds across every config",
                  if (dplyr::n_distinct(seeds_per_cfg$s) == 1L) "PASS" else "FAIL",
                  sprintf("%d distinct set(s) | seeds: %s",
                          dplyr::n_distinct(seeds_per_cfg$s),
                          seeds_per_cfg$s[1]))
      }

      # Rank 1 of the PER-CONFIG table really is the highest mean -- not the
      # row that happened to hold the best single run.
      if (file.exists(f_byconfig)) {
        by_config <- safe_read_csv2(f_byconfig)
        top_by_rank <- by_config$config_id[by_config$rank == 1][1]
        top_by_mean <- by_config$config_id[which.max(by_config$val_ccc_mean)]
        check_equal("03", "rank 1 is the highest mean val_ccc",
                    top_by_rank, top_by_mean, "rank_1", "max_mean")

        # The question that decides whether this tuning means anything: is
        # the gap between first and second larger than the seed noise?
        if (nrow(by_config) > 1L && "val_ccc_sd" %in% names(by_config)) {
          ord <- dplyr::arrange(by_config, rank)
          gap <- ord$val_ccc_mean[1] - ord$val_ccc_mean[2]
          # The noise floor is the sd between SEEDS within one fold, from
          # seed_noise_floor(). The per-config sd mixes fold and seed, so using
          # it here under the label "between seeds" reported a different
          # quantity from the one stage 03 prints, under the same name.
          nf    <- seed_noise_floor(comparison)
          noise <- nf$median_sd
          add_check("03", "the winner stands above the seed noise",
                    if (!is.finite(noise)) "WARN"
                    else if (gap >= noise) "PASS" else "WARN",
                    if (!is.finite(noise))
                      "sd entre sementes indisponivel (1 repeticao por config)"
                    else sprintf("1o-2o = %.4f | typical sd between seeds = %.4f",
                                 gap, noise))
        }
        top_by_ccc <- top_by_mean
      } else {
        top_by_rank <- comparison$config_id[comparison$rank == 1][1]
        top_by_ccc  <- comparison$config_id[which.max(comparison$val_ccc)]
        check_equal("03", "rank 1 is the highest val_ccc",
                    top_by_rank, top_by_ccc, "rank_1", "max_ccc")
      }

      # gate_summary.csv should exist only for dual-branch configs (a window
      # with "x" in the name, e.g. "9x15") whose gate is not no_gate_concat --
      # confirming that the conditional logic in extract_gate_analysis() is not
      # writing, or failing to write, a file for the wrong kind of config.
      gated_ids <- comparison$unit_id[
        grepl("x", comparison$window_sizes) & comparison$gate_type != "no_gate_concat"
      ]
      gate_files <- file.path(run_dir, "gates", paste0(gated_ids, "_gate_summary.csv"))
      n_missing_gate <- sum(!file.exists(gate_files))
      add_check("03", "gate_summary.csv exists for every gated dual-branch config",
                if (n_missing_gate == 0) "PASS" else "WARN",
                sprintf("%d/%d faltando", n_missing_gate, length(gated_ids)))

      # The mean metric when there are repetitions; the single row's when not.
      top_rows <- dplyr::filter(comparison, config_id == top_by_ccc)
      .say(sprintf(
        "\n  Best config: %s | CCC=%.3f | MAE=%.2f | RMSE=%.2f | window=%s | gate=%s | %d repetition(s)",
        top_by_ccc,
        mean(top_rows$val_ccc,  na.rm = TRUE),
        mean(top_rows$val_mae,  na.rm = TRUE),
        mean(top_rows$val_rmse, na.rm = TRUE),
        top_rows$window_sizes[1],
        top_rows$gate_type[1],
        nrow(top_rows)))

    } else {
      .say("Stage 03 incomplete -- skipping content checks.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 04 - final model (multi-seed ensemble)
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Stage 04: final model (multi-seed ensemble) --\n")

final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (!dir.exists(final_model_base)) {
  .say("Stage 04 not started -- directory not found: ", final_model_base)
} else {
  final_runs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  final_runs <- final_runs[grepl("^final_", final_runs)]
  if (length(final_runs) == 0) {
    .say("Stage 04 incomplete -- no run found in: ", final_model_base)
  } else {
    final_run_id <- sort(final_runs, decreasing = TRUE)[1]
    run_dir <- file.path(final_model_base, final_run_id)
    .say("Most recent run: ", final_run_id)

    f_summary_rds <- file.path(run_dir, "comparison", "final_run_summary.rds")
    f_all_seeds   <- file.path(run_dir, "comparison", "all_seed_results_test.csv")
    f_cfg_summary <- file.path(run_dir, "comparison", "config_summary_test.csv")

    files_04 <- c(f_summary_rds, f_all_seeds, f_cfg_summary)
    all_04_exist <- all(purrr::map_lgl(files_04, ~ check_exists("04", basename(.x), .x)))

    if (all_04_exist) {

      summary_rds      <- readRDS(f_summary_rds)
      all_seed_results  <- safe_read_csv2(f_all_seeds)
      config_summary    <- safe_read_csv2(f_cfg_summary)

      selected_cfgs    <- summary_rds$selected_cfgs
      seeds_expected    <- summary_rds$seeds
      n_seeds_expected  <- length(seeds_expected)

      # Stage 04 records which tuning run it used. This confirms that the
      # directory still exists (not deleted or renamed since) and, when stage
      # 03 also ran in this same check, that it is exactly the most recent run
      # resolved above -- which is what stops a stale tuning_run_id left in the
      # script from silently pinning the work to an old run.
      linked_tuning_dir <- file.path(tuning_dir, summary_rds$tuning_run_id)
      add_check("04", "the tuning_run_id stage 04 refers to still exists",
                if (dir.exists(linked_tuning_dir)) "PASS" else "FAIL",
                summary_rds$tuning_run_id)
      if (exists("tuning_run_id") && all_03_exist) {
        check_equal("04", "stage 04's tuning_run_id == stage 03's most recent run",
                    summary_rds$tuning_run_id, tuning_run_id, "used_by_04", "mais_recente_03")
      }

      # When selected_config_ids was left NULL (the default, recommended in
      # stage 04's header), the chosen config must be exactly rank 1 of that
      # tuning run's validation ranking -- otherwise the final model is being
      # trained on an architecture that is not the best one stage 03 found.
      # Picking a top-N by hand is legitimate, so this is a WARN, not a FAIL.
      if (exists("tuning_run_id") && all_03_exist &&
          identical(summary_rds$tuning_run_id, tuning_run_id)) {
        rank1_id <- comparison$config_id[comparison$rank == 1L]
        add_check("04", "selected config(s) include stage 03's rank 1",
                  if (rank1_id %in% selected_cfgs$config_id) "PASS" else "WARN",
                  paste0("rank1=", rank1_id, " | selecionados=",
                        paste(selected_cfgs$config_id, collapse = ", ")))
      }

      # Each selected config needs exactly n_seeds_expected result rows --
      # no missing seed (a silent crash) and no extra one (a leftover from
      # another run with different seeds).
      seed_counts <- dplyr::count(all_seed_results, config_id, name = "n_seeds_found")
      for (cid in selected_cfgs$config_id) {
        found <- seed_counts$n_seeds_found[seed_counts$config_id == cid]
        found <- if (length(found) == 0) 0L else found
        add_check("04", paste0("n_seeds completas (", cid, ")"),
                  if (found == n_seeds_expected) "PASS" else "FAIL",
                  sprintf("%d/%d seeds", found, n_seeds_expected))
      }

      # Every expected (config, seed) has a saved .pt checkpoint -- the same
      # principle as the stage 03 check: a result with no model behind it would
      # leave the run useless for future inference while still appearing as a
      # success in the metrics table.
      ckpt_paths <- character(0)
      for (cid in selected_cfgs$config_id) {
        ckpt_paths <- c(ckpt_paths, file.path(run_dir, cid, "models",
                                              sprintf("seed%04d_best.pt", seeds_expected)))
      }
      n_missing_ckpt <- sum(!file.exists(ckpt_paths))
      add_check("04", "every expected (config, seed) has a .pt checkpoint",
                if (n_missing_ckpt == 0) "PASS" else "FAIL",
                sprintf("%d/%d checkpoint(s) missing", n_missing_ckpt, length(ckpt_paths)))

      # Test metrics free of NA and inside a physically plausible range (same
      # logic as stage 03: no judgement of how good, only impossible values).
      n_na_metrics <- sum(is.na(all_seed_results$ccc) | is.na(all_seed_results$mae) |
                          is.na(all_seed_results$rmse))
      add_check("04", "ccc/mae/rmse free of NA (every seed)",
                if (n_na_metrics == 0) "PASS" else "FAIL",
                sprintf("%d row(s) with an NA metric", n_na_metrics))

      n_ccc_out <- sum(all_seed_results$ccc < -1 | all_seed_results$ccc > 1, na.rm = TRUE)
      add_check("04", "ccc within [-1, 1] (every seed)",
                if (n_ccc_out == 0) "PASS" else "FAIL",
                sprintf("%d row(s) out of range", n_ccc_out))

      n_nonpos <- sum(all_seed_results$mae <= 0 | all_seed_results$rmse <= 0, na.rm = TRUE)
      add_check("04", "mae and rmse > 0 (every seed)",
                if (n_nonpos == 0) "PASS" else "FAIL",
                sprintf("%d row(s) with error <= 0", n_nonpos))

      # Stability across seeds: sd of CCC as a percentage of the mean. A large
      # spread (see docs/design_decisions.md, section 11) points to unstable,
      # poorly reproducible training rather than mere initialisation luck -- a
      # publishable result should have a low one.
      for (i in seq_len(nrow(config_summary))) {
        cs <- config_summary[i, ]
        pct_sd <- 100 * cs$ccc_sd / cs$ccc_mean
        check_threshold("04", paste0("CCC SD relativo (", cs$config_id, ")"),
                        pct_sd, warn_above = 10, fail_above = 20, unit = "% da media")
      }

      # config_summary matches the aggregation recomputed from
      # all_seed_results -- redundancy against CSV corruption or misalignment.
      recalc <- all_seed_results %>%
        dplyr::group_by(config_id) %>%
        dplyr::summarise(ccc_mean_recalc = mean(ccc), .groups = "drop")
      merged <- dplyr::left_join(config_summary, recalc, by = "config_id")
      for (i in seq_len(nrow(merged))) {
        check_equal("04", paste0("ccc_mean salvo == recalculado (", merged$config_id[i], ")"),
                    round(merged$ccc_mean[i], 6), round(merged$ccc_mean_recalc[i], 6),
                    "salvo", "recalculado")
      }

      # A per-seed gate_summary.csv should exist only for dual-branch configs
      # (two windows) whose gate_type is not no_gate_concat -- as in stage 03.
      for (i in seq_len(nrow(selected_cfgs))) {
        cid <- selected_cfgs$config_id[i]
        ws  <- selected_cfgs$window_sizes[[i]]
        gt  <- selected_cfgs$gate_type[i]
        if (length(ws) == 2L && gt != "no_gate_concat") {
          gate_files <- file.path(run_dir, cid, "gates",
                                  sprintf("seed%04d_gate_summary.csv", seeds_expected))
          n_missing_gate <- sum(!file.exists(gate_files))
          add_check("04", paste0("gate_summary.csv por seed existe (", cid, ")"),
                    if (n_missing_gate == 0) "PASS" else "WARN",
                    sprintf("%d/%d faltando", n_missing_gate, length(seeds_expected)))
        }
      }

      for (i in seq_len(nrow(config_summary))) {
        cs <- config_summary[i, ]
        .say(sprintf(
          "\n  Modelo final [%s]: CCC=%.4f +/- %.4f | MAE=%.3f | RMSE=%.3f | %d seeds",
          cs$config_id, cs$ccc_mean, cs$ccc_sd, cs$mae_mean, cs$rmse_mean, cs$n_seeds))
      }

    } else {
      .say("Stage 04 incomplete -- skipping content checks.")
    }
  }
}

# ══════════════════════════════════════════════════════════════════════════════
# [Placeholder for future stages]
#
# Stages 05/06 (spatial prediction): the tiles' valid_fraction has not dropped
# back towards zero -- the same logic as the stage 02 check, adapted to the
# shard/merge logs. See also 05a_test.R and 05c_estimate_eta.R, which already
# cover part of this for the 2D pipeline.
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# SNAPSHOT - what changed since the previous run?
#
# In a refactor the question asked after every execution is "did anything
# move?", and answering it meant scrolling back through old output by hand.
# "Everything identical" is the result you want to see, and it is precisely the
# one that is hardest to confirm by eye.
# ══════════════════════════════════════════════════════════════════════════════

.say("\n-- Snapshot: comparison with the previous run --\n")

snap_vals <- list()
add_snap <- function(k, v) {
  if (length(v) == 1L && !is.null(v) && !is.na(v)) snap_vals[[k]] <<- v
}

if (exists("dscheck")) {
  add_snap("01_n_linhas",     dscheck$n_rows[1])
  add_snap("01_n_predictors", dscheck$n_predictors[1])
  add_snap("01_n_dummy",      dscheck$n_dummy_predictors[1])
  add_snap("01_n_percentage", dscheck$n_percentage_predictors[1])
  add_snap("01_n_continuous", dscheck$n_continuous_predictors[1])
  add_snap("01_mediana_alvo", round(dscheck$median_target[1], 6))
}
if (exists("qc")) add_snap("01_pct_problema", qc$pct_any_problem[1])
if (exists("crisk")) {
  add_snap("01_canais_constantes", sum(crisk$risk == "constant", na.rm = TRUE))
  add_snap("01_canais_com_na",     sum(crisk$risk == "has_na",   na.rm = TRUE))
}
if (exists("manifest") && "n_points_valid" %in% names(manifest)) {
  add_snap("02_n_points_valid", manifest$n_points_valid[1])
  add_snap("02_pct_removed",    manifest$pct_removed[1])
}
if (exists("blame") && nrow(blame) > 0L) {
  # SNAPSHOTTED BY SOLE CAUSE, and only when there is one.
  #
  # Under pct_invalidated every channel ties at the same value whenever the
  # losses are nodata locations, so which.max() returns whichever channel comes
  # first alphabetically. That is not a measurement: rename a predictor and the
  # snapshot reports a change that did not happen.
  if ("pct_sole_cause" %in% names(blame) &&
      max(blame$pct_sole_cause, na.rm = TRUE) > 0) {
    add_snap("02_worst_sole_cause_channel",
             blame$predictor[which.max(blame$pct_sole_cause)])
    add_snap("02_worst_sole_cause_pct",
             max(blame$pct_sole_cause, na.rm = TRUE))
  } else {
    add_snap("02_worst_sole_cause_channel", "(none)")
    add_snap("02_worst_sole_cause_pct", 0)
  }
  add_snap("02_pct_lost_any_channel", max(blame$pct_invalidated, na.rm = TRUE))
}
if (exists("cc") && !inherits(cc, "try-error")) {
  add_snap("02_centros_divergentes", cc$n_mismatch)
}
add_snap("99_n_pass", sum(.results$status == "PASS"))
add_snap("99_n_warn", sum(.results$status == "WARN"))
add_snap("99_n_fail", sum(.results$status == "FAIL"))

snap_dir <- file.path(project_root, "outputs", "qc", "snapshots")
# The 99's own counters (99_n_pass/warn/fail) are WRITTEN into the snapshot --
# they are the run's summary and belong in the history -- but they stay out of
# the diff. They are derived from every other key: if a real metric moves, it
# shows up in the diff on its own. Comparing them creates a loop in which
# FIXING a WARN produces a WARN ("values changed"), which is exactly what
# happened on the run that repaired the two broken checks.
cmp <- compare_run_snapshot(snap_vals, snap_dir,
                            exclude = c("99_n_pass", "99_n_warn", "99_n_fail"))
print_snapshot_diff(cmp)
write_run_snapshot(snap_vals, snap_dir)

if (cmp$has_previous) {
  n_changed <- sum(cmp$diff$status != "=")
  add_check("99", "values changed since the previous run",
            if (n_changed == 0L) "PASS" else "WARN",
            sprintf("%d of %d", n_changed, nrow(cmp$diff)))
}

# -- Final summary --────────────────────────────────────────────────────────────

.say("\n", strrep("=", 90))
.say("SUMMARY")
.say(strrep("=", 90))

n_pass <- sum(.results$status == "PASS")
n_warn <- sum(.results$status == "WARN")
n_fail <- sum(.results$status == "FAIL")

.say(sprintf("\n  PASS: %d   WARN: %d   FAIL: %d   (total: %d checks)\n",
                n_pass, n_warn, n_fail, nrow(.results)))

# Only what actually failed counts as a problem -- a stage that never started
# does not register a check at all.
if (n_fail > 0) {
  .say("Checks that FAILED:")
  print_wide(dplyr::filter(.results, status == "FAIL"), n = Inf)
}
if (n_warn > 0) {
  .say("\nChecks with a WARNING:")
  print_wide(dplyr::filter(.results, status == "WARN"), n = Inf)
}

report_dir <- file.path(project_root, "outputs", "qc")
dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
report_file <- file.path(report_dir, paste0("pipeline_check_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".csv"))
readr::write_csv2(.results, report_file)
.say("\nFull report saved to: ", report_file)

if (n_fail > 0) {
  warning(n_fail, " check(s) FAILED. Review before moving on to the next stage.")
} else if (n_warn > 0) {
  .say("\nNo critical failure, but there are warnings -- review before spending CPU on the next stage.")
} else {
  .say("\nAll clear. Safe to move on to the next stage.")
}
