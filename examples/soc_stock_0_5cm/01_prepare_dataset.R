source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c(
  "sf",
  "terra",
  "dplyr",
  "tidyr",
  "readr",
  "tibble",
  "janitor",
  "purrr",
  "ggplot2"
)

install_load_pkg(pkg)

rm(list = ls())
gc()

options(width = 200)

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"
setwd(project_root)

source(file.path(project_root, "R", "utils.R"))
source(file.path(project_root, "R", "preprocess.R"))
source(file.path(project_root, "R", "resample.R"))

set.seed(123)

# ── Run profile ───────────────────────────────────────────────────────────────
#
# "full" uses every profile. "dev" keeps a fraction of them so the whole
# pipeline -- 01 through prediction -- runs end to end in minutes instead of a
# day. A pipeline you can run completely is a pipeline you can fix without
# fear, and that is the only reason this exists.
#
# WHAT A DEV NUMBER IS COMPARABLE WITH. Only another dev number. Comparing a
# 10% run against the full run measures data volume, not correctness. What
# must survive a subsample is the SHAPE of the result -- see
# docs/reference_performance.md for the four relations that hold at any volume.
#
# WHY WHOLE BLOCKS. Dropping profiles at random thins the spatial clusters:
# leakage falls, the buffer discards less, and the folds look cleaner than the
# data is. Measured on the test fixture: a random subsample of the same size
# left each point with 9.4 neighbours within 1 km where the full data has 29.0,
# while the block subsample kept 28.8. A spatial bug would hide behind that.
#
# The profile is written into the metadata and the 99 announces it, so a dev
# result can never be mistaken for a real one later.

run_profile <- "dev"           # "dev" or "full"

dev_subsample <- list(
  frac       = 0.10,           # of POINTS, not of blocks
  block_size = 2,              # degrees -- same grid the spatial folds use
  seed       = 20260914L
)

# ── Settings ──────────────────────────────────────────────────────────────────

target_col   <- "soc_stock_ton_ha_0_5cm"
target_label <- "soc_stock_0_5cm"
target_unit  <- "ton_ha"


soc_gpkg_file <- file.path(
  project_root,
  "data", "raw",
  "wosis_profile_soc_stock_spline_clean_preSpline.gpkg"
)

predictor_raster_dir <- "D:/usuario_armazenamento/cassio/R/predictors_resolution_250m"

# Predictors to drop manually (leave empty to use all).
#
# These six are CONSTANT (all zero) at every one of the ~37k profiles, so they
# carry no information for training -- but they are NOT constant over the map:
# they switch on over glaciers, small islands, soil-map no-data and deep ocean
# floor. That asymmetry is the problem. Because their input is always 0 during
# training, their gradient is always 0, so with weight_decay = 0 their weights
# stay at random initialisation for ever. Each one then applies an arbitrary,
# seed-dependent bias to exactly the pixels that lie outside anything the
# network ever saw. Dropping them costs nothing and removes that.
#
# Detected automatically by the "Channel risk" report further down -- if that
# report flags a constant channel that is NOT listed here, decide about it
# deliberately rather than letting it through.
manual_predictor_drop <- c(
  "geology_ice_and_glaciers",
  "soil_class_fao_islands",
  "soil_class_fao_no_data",
  "terrestrial_habitat_deep_ocean_floor",
  "pnv_moss_and_lichen",
  "pnv_open_forest_deciduous_needleleaf"
)

# Temperature QC: values below this threshold are set to NA
temperature_min_valid_celsius <- -100

# Regex patterns that identify percentage predictors (bounded 0–100)
# These are scaled to [0, 1] by dividing by 100 (instead of z-score)
percentage_predictor_patterns <- c(
  "^pnv_",
  "^clay_",
  "^peatland_extent$"
)

# Predictors that look like dummies (only 0/1 values) but should be treated as
# continuous/percentage — overrides the auto-detection.
#
# Was c("pnv_moss_and_lichen", "pnv_open_forest_deciduous_needleleaf"). Both
# are now dropped above as constant, and the override never did anything for
# them anyway: a channel that is 0 at every profile has no type to get wrong.
# Kept as a hook, because the auto-detection genuinely can misread a rare
# percentage class as a dummy once profiles start hitting it.
force_as_percentage <- janitor::make_clean_names(character(0))

# ── Paths ─────────────────────────────────────────────────────────────────────

output_data_dir     <- file.path(project_root, "data",    "processed", "soc_stock_modeling", target_label)
output_metadata_dir <- file.path(project_root, "outputs", "metadata",  "soc_stock_modeling", target_label)
output_figure_dir   <- file.path(project_root, "outputs", "figures",   "soc_stock_modeling", target_label)

create_output_dirs(c(output_data_dir, output_metadata_dir, output_figure_dir))

# ── Validation ────────────────────────────────────────────────────────────────

if (!file.exists(soc_gpkg_file)) {
  stop("Target GPKG not found: ", soc_gpkg_file)
}

if (!dir.exists(predictor_raster_dir)) {
  stop("Predictor raster directory not found: ", predictor_raster_dir)
}

# ── Read target GPKG ──────────────────────────────────────────────────────────

soc_sf <- sf::st_read(soc_gpkg_file, quiet = TRUE) %>%
  janitor::clean_names() %>%
  dplyr::mutate(profile_id = as.character(profile_id))

if (!target_col %in% names(soc_sf)) {
  stop("Target column '", target_col, "' not found in GPKG.")
}

message("Profiles read: ", nrow(soc_sf))

# ── Development subsample ─────────────────────────────────────────────────────
# Applied HERE, before terra::extract() touches 181 rasters: this is where the
# time is, so subsampling anywhere later would save nothing.

subsample_note <- "full dataset"

if (identical(run_profile, "dev")) {
  xy <- sf::st_coordinates(sf::st_geometry(soc_sf))
  keep <- block_subsample(xy[, 1], xy[, 2],
                          frac       = dev_subsample$frac,
                          block_size = dev_subsample$block_size,
                          seed       = dev_subsample$seed)
  subsample_note <- describe_subsample(keep)
  soc_sf <- soc_sf[keep, , drop = FALSE]

  message("\n", strrep("!", 78))
  message("RUN PROFILE: dev -- this is NOT a result run")
  message("  ", subsample_note)
  message("  Comparable only with another dev run. See ",
          "docs/reference_performance.md")
  message(strrep("!", 78), "\n")
} else {
  message("RUN PROFILE: full -- ", nrow(soc_sf), " profiles")
}

# ── List raster predictors ────────────────────────────────────────────────────
# Ignores any subdirectory (e.g. 'nused') — only .tif files at the root level

raster_files_all <- list.files(
  predictor_raster_dir,
  pattern   = "\\.tif$",
  full.names = TRUE,
  recursive  = FALSE
)

if (length(raster_files_all) == 0) {
  stop("No .tif files found in: ", predictor_raster_dir)
}

raster_table_all <- tibble::tibble(
  raster_file     = raster_files_all,
  raster_name_raw = tools::file_path_sans_ext(basename(raster_files_all)),
  predictor       = janitor::make_clean_names(raster_name_raw)
)

dup_predictors <- dplyr::count(raster_table_all, predictor) %>%
  dplyr::filter(n > 1)

if (nrow(dup_predictors) > 0) {
  print(dup_predictors)
  stop("Duplicated predictor names after clean_names(). Rename the raster files.")
}

manual_predictor_drop <- janitor::make_clean_names(manual_predictor_drop)
drop_present <- intersect(manual_predictor_drop, raster_table_all$predictor)
drop_missing <- setdiff(manual_predictor_drop, raster_table_all$predictor)

if (length(drop_present) > 0) {
  message("Dropping manually flagged predictors: ", paste(drop_present, collapse = ", "))
}
if (length(drop_missing) > 0) {
  message("Manual drop entries not matched in rasters: ", paste(drop_missing, collapse = ", "))
}

raster_table_use <- raster_table_all %>%
  dplyr::filter(!predictor %in% drop_present) %>%
  dplyr::arrange(predictor)

predictor_cols_final <- raster_table_use$predictor

message("Predictor rasters to use: ", nrow(raster_table_use))

# ── Load raster stack ─────────────────────────────────────────────────────────

predictor_rasters <- terra::rast(raster_table_use$raster_file)
names(predictor_rasters) <- raster_table_use$predictor

if (!is.na(terra::crs(predictor_rasters)) == FALSE) {
  stop("Predictor rasters have no valid CRS.")
}

# ── Extract predictor values at profile locations ─────────────────────────────

soc_projected <- sf::st_transform(
  soc_sf,
  crs = sf::st_crs(terra::crs(predictor_rasters, proj = TRUE))
)

coords <- sf::st_coordinates(soc_projected)

soc_points <- soc_projected %>%
  dplyr::mutate(
    x = as.numeric(coords[, 1]),
    y = as.numeric(coords[, 2])
  ) %>%
  dplyr::select(profile_id, x, y, dplyr::all_of(target_col))

predictor_values <- terra::extract(
  predictor_rasters,
  terra::vect(soc_points),
  ID = FALSE
) %>%
  tibble::as_tibble() %>%
  dplyr::mutate(dplyr::across(dplyr::everything(), as.numeric))

dataset_extracted <- soc_points %>%
  sf::st_drop_geometry() %>%
  dplyr::bind_cols(predictor_values) %>%
  dplyr::mutate(
    profile_id     = as.character(profile_id),
    target_native  = as.numeric(.data[[target_col]]),
    target_log1p   = log1p(target_native)
  )

# ── Predictor QC ──────────────────────────────────────────────────────────────

# Identify predictors that match percentage patterns
percentage_predictor_cols <- predictor_cols_final[
  purrr::map_lgl(predictor_cols_final, function(x) {
    any(grepl(paste(percentage_predictor_patterns, collapse = "|"), x))
  })
]

# Temperature: values below threshold are physically invalid → NA
temperature_cols <- grep(
  "surface_temperature_celsius$", predictor_cols_final, value = TRUE
)

dataset_qc <- dataset_extracted

if (length(temperature_cols) > 0) {
  dataset_qc <- dataset_qc %>%
    dplyr::mutate(dplyr::across(
      dplyr::all_of(temperature_cols),
      ~ dplyr::if_else(!is.na(.x) & is.finite(.x) & .x <= temperature_min_valid_celsius,
                       NA_real_, as.numeric(.x))
    ))
}

# Percentage predictors: clamp out-of-range values into [0, 100].
# These predictors are continuous/interpolated probability-like surfaces
# (e.g. PNV classes, clay mineralogy) that legitimately overshoot slightly
# below 0 or above 100 near sharp spatial transitions (Gibbs-like ringing
# from whatever smoothing produced them) — not sensor error, not missing
# data. Treating that overshoot as NA (old behaviour) discarded genuinely
# valid near-boundary signal and, downstream, blew up into large windowed
# gaps once the CNN's full-window validity rule amplified each discarded
# pixel into its surrounding patch footprint. Clamping preserves the value
# (effectively ~0% or ~100%) instead of manufacturing missingness that
# was never really there. Genuine NA/Inf are left untouched.
if (length(percentage_predictor_cols) > 0) {
  dataset_qc <- dataset_qc %>%
    dplyr::mutate(dplyr::across(
      dplyr::all_of(percentage_predictor_cols),
      ~ dplyr::if_else(!is.na(.x) & is.finite(.x),
                       pmin(pmax(as.numeric(.x), 0), 100), as.numeric(.x))
    ))
}

# Flag rows with any problem in target or predictors
qc_flags <- dataset_qc %>%
  dplyr::mutate(
    problem_target = is.na(profile_id) | is.na(x) | is.na(y) |
      !is.finite(x) | !is.finite(y) |
      is.na(target_native) | !is.finite(target_native) | target_native <= 0 |
      is.na(target_log1p)  | !is.finite(target_log1p),
    problem_predictor = !dplyr::if_all(
      dplyr::all_of(predictor_cols_final),
      ~ !is.na(.x) & is.finite(.x)
    )
  )

qc_summary <- qc_flags %>%
  dplyr::summarise(
    n_rows_extracted   = dplyr::n(),
    n_target_problem   = sum(problem_target,   na.rm = TRUE),
    n_predictor_problem = sum(problem_predictor, na.rm = TRUE),
    n_any_problem      = sum(problem_target | problem_predictor, na.rm = TRUE),
    pct_any_problem    = round(100 * n_any_problem / n_rows_extracted, 2),
    n_rows_after_qc    = n_rows_extracted - n_any_problem
  )

message("\n── QC summary ──────────────────────────────")
print(qc_summary)

dataset_model_raw <- qc_flags %>%
  dplyr::filter(!problem_target, !problem_predictor) %>%
  dplyr::select(-problem_target, -problem_predictor) %>%
  dplyr::select(
    profile_id, x, y,
    dplyr::all_of(target_col), target_native, target_log1p,
    dplyr::all_of(predictor_cols_final)
  ) %>%
  dplyr::distinct(profile_id, .keep_all = TRUE)

if (nrow(dataset_model_raw) == 0) {
  stop("No rows remained after QC.")
}

message("Rows after QC: ", nrow(dataset_model_raw))

# ── Predictor type classification ─────────────────────────────────────────────
# dummy      : only values 0 and 1 → no scaling needed
# percentage : bounded [0, 100] → scale to [0, 1] by dividing by 100
# continuous : z-score from train set statistics

predictor_type_table <- purrr::map_dfr(predictor_cols_final, function(nm) {
  x      <- dataset_model_raw[[nm]]
  x_use  <- x[!is.na(x) & is.finite(x)]
  uniq   <- sort(unique(x_use))
  tibble::tibble(
    predictor     = nm,
    n_unique      = length(uniq),
    min_value     = min(x_use, na.rm = TRUE),
    max_value     = max(x_use, na.rm = TRUE),
    is_dummy      = length(uniq) <= 2 && all(uniq %in% c(0, 1)),
    is_percentage = nm %in% percentage_predictor_cols
  )
})

# Override: some predictors look like dummies by value range but should
# be treated as percentages (e.g., rare PNV classes with mostly 0/1 but
# the variable represents a continuous probability)
predictor_type_table <- predictor_type_table %>%
  dplyr::mutate(
    is_dummy = dplyr::if_else(predictor %in% force_as_percentage, FALSE, is_dummy),
    is_percentage = dplyr::if_else(predictor %in% force_as_percentage, TRUE, is_percentage)
  )

predictor_cols_dummy      <- dplyr::filter(predictor_type_table, is_dummy)$predictor
predictor_cols_percentage <- dplyr::filter(predictor_type_table, is_percentage)$predictor
predictor_cols_continuous <- dplyr::filter(predictor_type_table, !is_dummy, !is_percentage)$predictor

message(
  "\nPredictor types — dummy: ", length(predictor_cols_dummy),
  " | percentage: ", length(predictor_cols_percentage),
  " | continuous: ", length(predictor_cols_continuous)
)

# ── Row key ───────────────────────────────────────────────────────────────────
#
# THIS STAGE DECIDES NO ROLES.
#
# It used to cut a stratified 70/15/15 here, and that decision then travelled
# inside the patch store -- so changing the split meant five hours of
# re-extraction. It is now made by a fold plan in stage 03, from coordinates,
# in seconds: `spatial_folds(meta, k = 3, test_frac = 0.15, ...)` carves the
# test set AND the folds by one criterion.
#
# What stays here is what is expensive to produce and independent of any
# split: the point values, the coordinates, the target, and the predictor
# TYPES. Everything downstream is then free to change its mind for free.

dataset_model_split <- dplyr::mutate(dataset_model_raw,
                                     sample_id = dplyr::row_number())

# ── Degenerate predictors ─────────────────────────────────────────────────────
#
# A predictor with zero variance over the points cannot be z-scored, and a
# channel that is constant at every profile teaches the network nothing while
# still applying an arbitrary weight to the map. Checked over ALL rows, because
# whether a channel is constant does not depend on who trains.
#
# NOTE ON WHAT IS NOT HERE ANY MORE: the scaling table. Scaling is estimated
# from the training rows OF A FOLD, so it belongs to the fitted model, not to
# the dataset -- see R/preprocess.R and the note in stage 04. Keeping a global
# copy here is what let the map be built with one set of constants while the
# model had been trained with another.

pred_sd <- vapply(predictor_cols_final,
                  function(nm) stats::sd(dataset_model_split[[nm]], na.rm = TRUE),
                  numeric(1))
bad_scaling <- names(pred_sd)[is.na(pred_sd) | !is.finite(pred_sd) |
                              (pred_sd <= 0 &
                               !(names(pred_sd) %in% predictor_cols_dummy))]

if (length(bad_scaling) > 0) {
  message("\nDropping predictors with degenerate variance over the points:")
  print(bad_scaling)
  predictor_cols_final <- setdiff(predictor_cols_final, bad_scaling)
  predictor_type_table <- dplyr::filter(predictor_type_table,
                                        predictor %in% predictor_cols_final)
  predictor_cols_dummy      <- dplyr::filter(predictor_type_table, is_dummy)$predictor
  predictor_cols_percentage <- dplyr::filter(predictor_type_table, is_percentage)$predictor
  predictor_cols_continuous <- dplyr::filter(predictor_type_table,
                                             !is_dummy, !is_percentage)$predictor
}

# ── Checks ────────────────────────────────────────────────────────────────────

dataset_check <- dataset_model_split %>%
  dplyr::summarise(
    target_col              = target_col,
    n_rows                  = dplyr::n(),
    n_profiles              = dplyr::n_distinct(profile_id),
    min_target              = min(target_native, na.rm = TRUE),
    q01_target              = as.numeric(stats::quantile(target_native, 0.01, na.rm = TRUE)),
    median_target           = median(target_native, na.rm = TRUE),
    mean_target             = mean(target_native, na.rm = TRUE),
    q99_target              = as.numeric(stats::quantile(target_native, 0.99, na.rm = TRUE)),
    max_target              = max(target_native, na.rm = TRUE),
    median_target_log1p     = median(target_log1p, na.rm = TRUE),
    n_predictors            = length(predictor_cols_final),
    n_dummy_predictors      = length(predictor_cols_dummy),
    n_percentage_predictors = length(predictor_cols_percentage),
    n_continuous_predictors = length(predictor_cols_continuous)
  )



message("\n-- Dataset summary --------------------------")
print_wide(dataset_check)

# ── Export ────────────────────────────────────────────────────────────────────

export_cols <- c(
  "profile_id", "sample_id", "x", "y",
  target_col, "target_native", "target_log1p",
  predictor_cols_final
)

safe_write_csv2(
  dplyr::select(dataset_model_split, dplyr::all_of(export_cols)),
  file.path(output_data_dir, "full_modeling_dataset_raw.csv")
)

# The per-split CSVs are gone, and so is the split itself: this stage writes
# ONE dataset and one point table. Who trains, who scores and who is held out
# is decided by a fold plan in stage 03, from coordinates, in seconds.

# ── Channel risk report ───────────────────────────────────────────────────────
# Three families of channel have shipped a broken map before, and none of them
# shows up as an error or a bad metric -- they only appear at the very end, as
# holes in the raster or as extrapolation nobody asked for. So they are named
# here, loudly, while it is still cheap to act on them.
#
#   constant       zero information at the profiles, but NOT constant over the
#                  world. `geology_ice_and_glaciers` is 0 at every profile and
#                  1 over ice: at prediction time the network meets a channel
#                  combination it never saw in training. The degenerate-sd
#                  filter below cannot catch these, because dummy channels get
#                  sd = 1 and percentage channels sd = 100 by definition, never
#                  an estimated sd.
#   near_constant  almost no variation -- same risk, smaller.
#   has_na         NA at the profiles. A channel with sparse NA is the one that
#                  emptied the 250 m map once already: the full-window rule
#                  turns each NA pixel into a hole of up to 15x15 around it.

channel_risk <- predictor_type_table %>%
  dplyr::mutate(
    n_na_at_points = purrr::map_int(predictor,
                                    ~ sum(!is.finite(dataset_model_raw[[.x]]))),
    pct_na         = round(100 * n_na_at_points / nrow(dataset_model_raw), 3),
    type           = dplyr::case_when(is_dummy ~ "dummy",
                                      is_percentage ~ "percentage",
                                      TRUE ~ "continuous"),
    risk = dplyr::case_when(
      n_unique <= 1L                  ~ "constant",
      n_unique <= 2L & !is_dummy      ~ "near_constant",
      n_na_at_points > 0L             ~ "has_na",
      TRUE                            ~ ""
    )
  ) %>%
  dplyr::select(predictor, type, n_unique, min_value, max_value,
                n_na_at_points, pct_na, risk)

flagged <- dplyr::filter(channel_risk, risk != "")

message("\n── Channel risk ─────────────────────────────")
if (nrow(flagged) == 0L) {
  message("  No channel flagged.")
} else {
  message("  ", nrow(flagged), " channel(s) flagged — these are the ones that ",
          "have historically broken the MAP, not the metrics:")
  print_wide(dplyr::arrange(flagged, risk, dplyr::desc(pct_na)), n = Inf)
  n_const <- sum(flagged$risk == "constant")
  if (n_const > 0L) {
    message("\n  WARNING: ", n_const, " constant channel(s) SURVIVED the drop ",
            "list: zero information here,")
    message("  but non-zero somewhere on the map. Their weights never get a ",
            "gradient, so they stay at")
    message("  random init and apply a seed-dependent bias exactly where the ",
            "network is extrapolating.")
    # WHETHER THIS IS ACTIONABLE DEPENDS ON THE RUN PROFILE, and acting on it
    # under the wrong one is worse than ignoring it.
    #
    # A rare class -- glaciers, evaporites, marine intertidal -- is constant at
    # 4k points simply because the subsample missed the handful of profiles
    # that carry it. Dropping it on that evidence changes the PREDICTOR SET of
    # the full run based on an artefact of a 10% draw, and the predictor set is
    # one of the three things a patch store is locked to: the dev store and the
    # full store would then be incompatible by construction, for no reason.
    #
    # The rule: a channel is a drop candidate when it is constant at FULL size.
    if (identical(run_profile, "dev")) {
      message("\n  THIS IS A DEV RUN -- do NOT act on this list yet. A rare class is constant ",
              "at ", format(nrow(dataset_model_split), big.mark = ","),
              " points")
      message("  because the subsample missed it, not because it is constant ",
              "in the data. Dropping it here")
      message("  would change the predictor set of the FULL run from an ",
              "artefact of a 10% draw.")
      message("  Re-read this list after a full run; what is still constant ",
              "at 41k points is real.")
    } else {
      message("  Add them to manual_predictor_drop at the top of this script, ",
              "or keep them deliberately.")
    }
  } else {
    message("\n  No constant channel survived the drop list.")
  }
}

safe_write_csv2(channel_risk, file.path(output_metadata_dir, "channel_risk.csv"))

# ── QC rules, exported so 02 applies EXACTLY these ────────────────────────────
# Previously the rules lived as literals inside both 01 and 02 and had to be
# kept in sync by hand. Now 01 decides and 02 obeys.

qc_table <- make_qc_table(
  predictors   = predictor_cols_final,
  na_below     = setNames(temperature_min_valid_celsius,
                          "surface_temperature_celsius$"),
  clamp_range  = predictor_cols_percentage,
  clamp_limits = c(0, 100)
)
safe_write_csv2(qc_table, file.path(output_metadata_dir, "qc_table.csv"))

message("\nQC rules: ", sum(!is.na(qc_table$na_below)), " channel(s) with an NA floor, ",
        sum(!is.na(qc_table$clamp_lower)), " clamped into [0, 100].")

# Metadata
safe_write_csv2(qc_summary,            file.path(output_metadata_dir, "qc_summary.csv"))
safe_write_csv2(predictor_type_table,  file.path(output_metadata_dir, "predictor_type_table.csv"))
safe_write_csv2(raster_table_all,      file.path(output_metadata_dir, "raster_table_all.csv"))
safe_write_csv2(raster_table_use,      file.path(output_metadata_dir, "raster_table_used.csv"))
safe_write_csv2(dataset_check,         file.path(output_metadata_dir, "dataset_check.csv"))

safe_write_csv2(
  dataset_model_split %>%
    dplyr::select(profile_id, sample_id, x, y,
                  target_native, target_log1p),
  file.path(output_metadata_dir, "point_metadata.csv")
)

safe_write_csv2(
  tibble::tibble(
    # The run profile travels with the data, so no downstream stage and no
    # future reader has to guess whether a number came from a subsample.
    run_profile                 = run_profile,
    subsample                   = subsample_note,
    target_label                = target_label,
    target_col                  = target_col,
    target_unit                 = target_unit,
    predictor_raster_dir        = predictor_raster_dir,
    manual_predictor_drop       = paste(drop_present, collapse = ";"),
    temperature_min_valid_celsius = temperature_min_valid_celsius,
    percentage_predictor_patterns = paste(percentage_predictor_patterns, collapse = ";"),
    n_predictors_final          = length(predictor_cols_final),
    n_dummy                     = length(predictor_cols_dummy),
    n_percentage                = length(predictor_cols_percentage),
    n_continuous                = length(predictor_cols_continuous),
    n_rows_after_qc             = nrow(dataset_model_raw),
    soc_gpkg_file               = soc_gpkg_file
  ),
  file.path(output_metadata_dir, "target_config.csv")
)

# ── Figures ───────────────────────────────────────────────────────────────────

# ── Figures ───────────────────────────────────────────────────────────────────
# Split-based figures are gone with the split: this stage no longer knows who
# trains. What it can still show is the target itself.

p_dens_native <- ggplot2::ggplot(dataset_model_split,
                                 ggplot2::aes(x = target_native)) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(x = paste0("SOC stock, ", target_unit), y = "Density",
                title = paste(target_label, "- native distribution")) +
  ggplot2::theme_bw()

p_dens_log1p <- ggplot2::ggplot(dataset_model_split,
                                ggplot2::aes(x = target_log1p)) +
  ggplot2::geom_density(linewidth = 0.8) +
  ggplot2::labs(x = "log1p(SOC stock)", y = "Density",
                title = paste(target_label, "- log1p distribution")) +
  ggplot2::theme_bw()

for (p in list(p_dens_native, p_dens_log1p)) print(p)

ggplot2::ggsave(file.path(output_figure_dir, "target_density_native.png"),
                p_dens_native, width = 7, height = 5, dpi = 300)
ggplot2::ggsave(file.path(output_figure_dir, "target_density_log1p.png"),
                p_dens_log1p, width = 7, height = 5, dpi = 300)

