# ── From a table of points and a folder of rasters to a patch store ───────────
#
# WHAT THIS REPLACES.
#
# Until 2026-09-26 this was two scripts of the SOC project, its stages 01
# and 02, written for one dataset: the SOC stock GPKG, its column names, its
# 181 rasters, its temperature sentinel. Everything a second user needed to
# change sat in the middle of 1,300 lines of script. dsm_prepare() is those two
# scripts with the dataset taken out: every decision that was a literal there
# is an argument here, and every argument is written into the store, so no
# later stage has to be told it again.
#
# THE RECIPE.
#
# Three kinds of decision are made here, and each must be honoured identically
# every time a value crosses from the rasters into the model:
#
#   the target transform  "log1p" or "none". Training happens in that space;
#                         every metric, every map and the smearing correction
#                         need its inverse. Stage 02 wrote it into the manifest
#                         -- and stages 04 and 05 then typed the inverse again
#                         by hand, as expm1, in two places nothing linked.
#   the predictor types   dummies (not scaled), percentages (divided by 100),
#                         continuous (z-scored from each fold's training rows).
#   the QC rules          NA floors, percentage clamps, the drop list.
#
# All of it goes to recipe.rds inside the store, beside copies of the small
# tables the store needs. A store directory is therefore self-contained: it can
# be moved, and dsm_load(store) needs nothing else.
#
# WHAT IS DECLARED AND WHAT IS DETECTED.
#
# Percentages are DECLARED, as regular expressions over the cleaned raster
# names: nothing in a channel's values tells a percentage from a continuous
# variable. Dummies are DETECTED by default -- at most two distinct values, both
# in {0, 1} -- or declared. A declaration always wins over the detection: a
# channel declared a percentage is never demoted to a dummy because the points
# happened to see only 0 and 1. (On the SOC data no channel is both, so this
# reproduces stage 01's table exactly; stage 01 needed a separate override
# list, force_as_percentage, for the case.)
#
# WHAT IS NOT HERE: SCALING. Means and sds are estimated from the TRAINING
# rows of a fold, so they belong to a fitted model and not to the data -- see
# R/preprocess.R. Patches are stored RAW. QC is the only thing applied to them,
# because QC is fold-independent: a physically impossible value is impossible
# whoever trains.

# ── the target transforms this framework can invert ───────────────────────────
#
# Two, deliberately. log1p is the one the smearing correction in R/smearing.R
# knows how to undo (Duan's estimator is for a log-trained model); any other
# transform would need its own correction on the way back, and a transform
# without one is a biased map. Adding one later means adding it here and
# nowhere else, because everything downstream asks this table.
.target_transforms <- list(
  none  = list(name = "none",  forward = identity, inverse = identity),
  log1p = list(name = "log1p", forward = log1p,    inverse = expm1)
)

#' The forward and inverse functions for a named target transform.
#'
#' @param name "none" or "log1p".
#' @return list(name, forward, inverse).
#' @examples
#' tr <- target_transform_spec("log1p")
#' tr$forward(10)
#' tr$inverse(tr$forward(10))
#' @export
target_transform_spec <- function(name) {
  if (is.null(name) || length(name) != 1L || is.na(name) ||
      !name %in% names(.target_transforms)) {
    stop("Unknown target transform '", paste(name, collapse = ", "), "'. ",
         "Known: ", paste(names(.target_transforms), collapse = ", "),
         ".\n  A transform needs its inverse for every metric and map, so an ",
         "unknown one cannot be read back.", call. = FALSE)
  }
  .target_transforms[[name]]
}

# ── dsm_prepare ───────────────────────────────────────────────────────────────

#' Build a patch store from a point table and a folder of aligned rasters.
#'
#' @param points     An sf object, a data.frame with coordinate columns, or a
#'   path to a spatial file sf can read (GPKG, shapefile, GeoJSON).
#' @param target     Name of the target column in `points`.
#' @param raster_dir Folder of aligned single-band rasters, one predictor per
#'   file. Only files at the top level matching `raster_pattern` are read.
#' @param windows    Patch sizes in pixels, odd. REQUIRED: a window's ground
#'   extent is window x resolution, so there is no default that is right at
#'   every resolution. 3/9/15 were chosen for 250 m.
#' @param out_dir    Where to write, as out_dir/patches (the store),
#'   out_dir/metadata (tables for a human) and out_dir/points.csv. Or give
#'   the three separately with store_dir, metadata_dir and points_file.
#' @param profile_id Column that identifies an observation. Rows sharing it are
#'   de-duplicated (the first is kept) and later kept in the same fold. Absent,
#'   every row is its own profile.
#' @param coords,crs For a data.frame: the coordinate columns and their CRS.
#'   NULL crs means the coordinates are already in the rasters' CRS.
#' @param percentage Regular expressions over the CLEANED raster names for
#'   channels in 0-100. Anchor them (^pnv_, ^peatland_extent$) to avoid
#'   matching more than intended; every match is reported.
#' @param dummy      "auto" to detect 0/1 channels, or the channel names.
#' @param drop       Channel names (raw or cleaned) to leave out.
#' @param na_below   Named numeric: names are regexes over cleaned names,
#'   values are thresholds. A value <= its threshold becomes NA (a nodata
#'   sentinel such as -9999, or stage 01's temperature floor of -100).
#' @param percentage_limits Percentages are clamped into this range; NULL to
#'   leave them as read. The clamp keeps the slight overshoot of an
#'   interpolated surface instead of turning it into NA.
#' @param transform  "none" or "log1p" -- the space the model trains in.
#' @param target_min Rows with target <= target_min are dropped. NULL keeps any
#'   finite target the transform accepts.
#' @param subsample  NULL, or list(frac, block_size, seed, strata): keep a
#'   fraction of the points in whole spatial blocks, for a run that finishes
#'   sooner. `strata`, optional, is the side of square strata, a whole
#'   multiple of `block_size`: each keeps its points up to one common quota,
#'   so the cut falls on the densest regions and no region is lost. Recorded,
#'   so a subsampled result is never mistaken for a full one.
#' @param n_cores    Cores for the extraction, one band per core. NULL uses the
#'   physical cores minus one, fewer if `max_ram_gb` cannot hold them. The
#'   result does not depend on it.
#' @param max_ram_gb RAM the extraction may use in total. NULL for 70% of what
#'   is available when it starts (read with the ps package; without it, no
#'   cap). Decides how many of the `n_cores` workers actually run.
#' @param read_gap,read_max_cols How the points of a row chunk are grouped
#'   into reads: windows closer than `read_gap` columns share a read, and no
#'   read is wider than `read_max_cols`. Performance knobs only -- the store
#'   and the point table are identical whatever they are, and the test suite
#'   proves it.
#' @param store_dir,metadata_dir,points_file Where each part goes when it is
#'   not under `out_dir`: the store, the tables for a human, the point table.
#' @param target_label Name of the target in reports. NULL uses `target`.
#' @param target_unit  Unit of the target, for reports. NULL leaves it
#'   unrecorded.
#' @param raster_pattern Regular expression the raster file names must match.
#' @param chunk_nrows  Raster rows read per chunk. A performance knob: the
#'   store does not depend on it.
#' @param verbose    Report progress.
#' @param overwrite  A store_dir that already holds a store is refused unless
#'   TRUE, in which case that store's files are removed first.
#' @return A `dsm_store`, which dsm_load() accepts directly.
#' @examples
#' ex <- example_landscape()
#' store <- dsm_prepare(ex$profiles, target = "soc_stock", raster_dir = ex$raster_dir,
#'                      windows = c(3, 7), out_dir = file.path(tempdir(), "landscape"),
#'                      percentage = "^clay_pct$", transform = "log1p", n_cores = 1,
#'                      overwrite = TRUE, verbose = FALSE)
#' store
#' @export
dsm_prepare <- function(points, target, raster_dir, windows,
                        out_dir           = NULL,
                        store_dir         = NULL,
                        metadata_dir      = NULL,
                        points_file       = NULL,
                        profile_id        = "profile_id",
                        coords            = c("x", "y"),
                        crs               = NULL,
                        percentage        = NULL,
                        dummy             = "auto",
                        drop              = NULL,
                        na_below          = NULL,
                        percentage_limits = c(0, 100),
                        transform         = c("none", "log1p"),
                        target_min        = NULL,
                        subsample         = NULL,
                        target_label      = NULL,
                        target_unit       = NULL,
                        raster_pattern    = "\\.tif$",
                        chunk_nrows       = 1000L,
                        n_cores           = NULL,
                        max_ram_gb        = NULL,
                        read_gap          = 256L,
                        read_max_cols     = 4096L,
                        overwrite         = FALSE,
                        verbose           = TRUE) {

  t_start <- Sys.time()
  say <- function(...) if (verbose) message(...)

  # ── 0. the arguments, all checked before anything is read ─────────────────
  if (missing(windows)) {
    stop("`windows` is required: the patch sizes in pixels, e.g. c(3, 9, 15). ",
         "A window's ground extent is window x resolution, so no default is ",
         "right at every resolution.", call. = FALSE)
  }
  windows <- sort(unique(as.integer(windows)))
  if (length(windows) == 0L || anyNA(windows) || any(windows < 1L) ||
      any(windows %% 2L != 1L)) {
    stop("`windows` must be odd whole numbers (a patch has one centre pixel); ",
         "got ", paste(windows, collapse = ", "), ".", call. = FALSE)
  }
  transform <- match.arg(transform)
  tr <- target_transform_spec(transform)
  if (!is.character(target) || length(target) != 1L) {
    stop("`target` must be the name of one column.", call. = FALSE)
  }
  if (!is.character(raster_dir) || length(raster_dir) != 1L || !dir.exists(raster_dir)) {
    stop("raster_dir does not exist: ", raster_dir, call. = FALSE)
  }
  if (!is.null(percentage_limits) &&
      (!is.numeric(percentage_limits) || length(percentage_limits) != 2L ||
       percentage_limits[1] >= percentage_limits[2])) {
    stop("percentage_limits must be NULL or c(lower, upper) with lower < upper.",
         call. = FALSE)
  }
  if (!is.null(na_below) && (!is.numeric(na_below) || is.null(names(na_below)) ||
                             any(!nzchar(names(na_below))))) {
    stop("na_below must be a NAMED numeric vector, e.g. ",
         "c(\"surface_temperature_celsius$\" = -100).", call. = FALSE)
  }
  if (!identical(dummy, "auto") && !is.character(dummy)) {
    stop("dummy must be \"auto\" or a character vector of channel names.",
         call. = FALSE)
  }
  if (!is.null(target_min) && (!is.numeric(target_min) || length(target_min) != 1L)) {
    stop("target_min must be NULL or one number.", call. = FALSE)
  }
  if (!is.null(subsample) &&
      (!is.list(subsample) || !all(c("frac", "block_size") %in% names(subsample)))) {
    stop("subsample must be NULL or list(frac = , block_size = , seed = , strata = ).",
         call. = FALSE)
  }
  chunk_nrows   <- as.integer(chunk_nrows)
  read_gap      <- as.integer(read_gap)
  read_max_cols <- as.integer(read_max_cols)
  if (is.na(chunk_nrows) || chunk_nrows < 1L) {
    stop("chunk_nrows must be a whole number >= 1.", call. = FALSE)
  }
  if (is.na(read_gap) || read_gap < 0L) {
    stop("read_gap must be a whole number >= 0.", call. = FALSE)
  }
  if (is.na(read_max_cols) || read_max_cols < max(windows)) {
    stop("read_max_cols must be at least the largest window (", max(windows),
         "), or a single patch would not fit in one read.", call. = FALSE)
  }
  if (!is.null(max_ram_gb) && (!is.numeric(max_ram_gb) || length(max_ram_gb) != 1L ||
                               max_ram_gb <= 0)) {
    stop("max_ram_gb must be NULL or one positive number.", call. = FALSE)
  }
  n_cores <- resolve_cores(n_cores, what = "the extraction")

  if (!is.null(out_dir)) {
    store_dir    <- store_dir    %||% file.path(out_dir, "patches")
    metadata_dir <- metadata_dir %||% file.path(out_dir, "metadata")
    points_file  <- points_file  %||% file.path(out_dir, "points.csv")
  }
  if (is.null(store_dir) || is.null(metadata_dir) || is.null(points_file)) {
    stop("Give out_dir, or all three of store_dir, metadata_dir and ",
         "points_file.", call. = FALSE)
  }
  target_label <- target_label %||% target
  .prep_clear_store(store_dir, overwrite, say)
  create_output_dirs(c(store_dir, metadata_dir, file.path(metadata_dir, "patches"),
                       dirname(points_file)))

  # ── 1. the points ─────────────────────────────────────────────────────────
  points_source <- if (is.character(points) && length(points) == 1L) points else
    class(points)[1]
  pts   <- .prep_read_points(points)
  is_sf <- inherits(pts, "sf")
  if (!target %in% names(pts)) {
    stop("Target column '", target, "' is not in the point table. Columns: ",
         paste(setdiff(names(pts), attr(pts, "sf_column")), collapse = ", "),
         call. = FALSE)
  }
  if (!is_sf && !all(coords %in% names(pts))) {
    stop("Coordinate column(s) ", paste(setdiff(coords, names(pts)), collapse = ", "),
         " not found. Pass `coords`, or an sf object.", call. = FALSE)
  }
  say("Points read: ", nrow(pts))

  # ── 2. an optional subsample, BEFORE any raster is touched ────────────────
  #
  # The extraction below is where the time goes, so a subsample anywhere later
  # would save nothing. Whole blocks rather than random points: a random
  # subsample thins the spatial clusters, so leakage falls, the buffer discards
  # less, and the folds look cleaner than the data is (measured on the test
  # fixture: 9.4 neighbours within 1 km per point instead of 29.0).
  run_profile    <- "full"
  subsample_note <- "full dataset"
  if (!is.null(subsample)) {
    xy0 <- if (is_sf) sf::st_coordinates(sf::st_geometry(pts)) else
      as.matrix(pts[, coords])
    keep <- block_subsample(xy0[, 1], xy0[, 2], frac = subsample$frac,
                            block_size = subsample$block_size,
                            seed = subsample$seed %||% 42L, strata = subsample$strata)
    subsample_note <- describe_subsample(keep)
    pts <- pts[keep, , drop = FALSE]
    run_profile <- "dev"
    say("\n", strrep("!", 78))
    say("SUBSAMPLE -- this is NOT a result run")
    say("  ", subsample_note)
    say("  Comparable only with another run of the same subsample.")
    say(strrep("!", 78), "\n")
  }

  # ── 3. the rasters ─────────────────────────────────────────────────────────
  rt <- .prep_raster_table(raster_dir, raster_pattern, drop, say)
  preds <- rt$use$predictor
  reserved <- c("profile_id", "sample_id", "x", "y", "target_native",
                "target_transform")
  if (target %in% reserved) {
    stop("The target column may not be called ", target, ": that name is ",
         "reserved for the point table this function writes.", call. = FALSE)
  }
  clash <- intersect(preds, c(reserved, target))
  if (length(clash) > 0L) {
    stop("Raster(s) named ", paste(clash, collapse = ", "), " collide with a ",
         "column of the point table. Rename the file(s).", call. = FALSE)
  }
  stack <- terra::rast(rt$use$raster_file)
  names(stack) <- preds
  if (!nzchar(terra::crs(stack))) {
    stop("The rasters carry no CRS, so the points cannot be placed on them.",
         call. = FALSE)
  }
  say("Predictor rasters to use: ", length(preds))

  # ── 4. place the points on the rasters ────────────────────────────────────
  if (is_sf) {
    pts <- sf::st_transform(pts, crs = sf::st_crs(terra::crs(stack, proj = TRUE)))
    xy  <- sf::st_coordinates(pts)
    tab <- sf::st_drop_geometry(pts)
  } else {
    xy <- as.matrix(pts[, coords])
    storage.mode(xy) <- "double"
    if (!is.null(crs)) {
      xy <- terra::crds(terra::project(terra::vect(xy, crs = crs), terra::crs(stack)))
    }
    tab <- as.data.frame(pts)
  }
  xy <- xy[, 1:2, drop = FALSE]          # st_coordinates() may add Z or M

  pid <- if (!is.null(profile_id) && profile_id %in% names(tab)) {
    as.character(tab[[profile_id]])
  } else {
    say("  No '", profile_id %||% "profile_id", "' column: every row is its own ",
        "profile.")
    as.character(seq_len(nrow(tab)))
  }
  tn <- as.numeric(tab[[target]])

  # ── 5. ONE PASS OVER THE RASTERS: every point's centre, every patch ───────
  #
  # Stage 01 read the centre values with terra::extract(), one file after
  # another, to run the QC; stage 02 then read the patches. On these rasters
  # -- one-row strips, LZW -- both decompress the same rows, and the first
  # pass ran in series: about half an hour of the 93 minutes P1 measured.
  #
  # Here each band is read once, in parallel: the centre of EVERY point (for
  # the QC and the types), and the patches of the points far enough from the
  # edge. The QC, the types and the variance filter then run on those
  # centres, and the arrays are cut down to the rows and channels that
  # survive. What it costs: patches for points the QC then drops -- 9% of the
  # SOC points. A centre is the same cell terra::extract() returned (the cell
  # holding the point), and the QC rules below are the same qc_band_values()
  # applied to the same read.
  pct_all <- .prep_match(percentage, preds, "percentage", say)
  qc_all  <- make_qc_table(
    predictors   = preds,
    na_below     = na_below,
    clamp_range  = if (is.null(percentage_limits)) character(0) else pct_all,
    clamp_limits = percentage_limits %||% c(0, 100))

  ex <- .prep_extract_patches(
    files         = rt$use$raster_file,
    qc_table      = qc_all,
    xy            = xy,
    windows       = windows,
    chunk_nrows   = chunk_nrows,
    n_cores       = n_cores,
    max_ram_gb    = max_ram_gb,
    read_gap      = read_gap,
    read_max_cols = read_max_cols,
    cell_bytes    = rt$max_cell_bytes,
    say           = say)

  # ── 6. QC at the points ────────────────────────────────────────────────────
  #
  # The centres come back with the QC rules already applied, so there is no
  # second set of rules to keep in step with the patches' -- stage 01 wrote
  # them twice, inline and as qc_table.csv, and they agreed because someone
  # kept them in step.
  pv <- tibble::as_tibble(stats::setNames(as.data.frame(ex$centre), preds))
  df <- tibble::tibble(profile_id = pid, x = as.numeric(xy[, 1]),
                       y = as.numeric(xy[, 2]))
  df[[target]] <- tn
  df <- dplyr::bind_cols(df, pv)
  df$target_native    <- tn
  df$target_transform <- tr$forward(tn)
  df$.row             <- seq_len(nrow(df))    # where each row's patches are

  bad_pred   <- rowSums(!is.finite(as.matrix(df[, preds]))) > 0L
  bad_target <- is.na(df$profile_id) | is.na(df$x) | is.na(df$y) |
    !is.finite(df$x) | !is.finite(df$y) |
    is.na(df$target_native) | !is.finite(df$target_native) |
    is.na(df$target_transform) | !is.finite(df$target_transform)
  if (!is.null(target_min)) {
    bad_target <- bad_target | (!is.na(df$target_native) &
                                  df$target_native <= target_min)
  }

  qc_summary <- tibble::tibble(
    n_rows_extracted    = nrow(df),
    n_target_problem    = sum(bad_target),
    n_predictor_problem = sum(bad_pred),
    n_any_problem       = sum(bad_target | bad_pred))
  qc_summary$pct_any_problem <- round(100 * qc_summary$n_any_problem /
                                        qc_summary$n_rows_extracted, 2)
  qc_summary$n_rows_after_qc <- qc_summary$n_rows_extracted -
    qc_summary$n_any_problem
  if (verbose) {
    message("\n-- QC summary --")
    print_wide(qc_summary)
  }

  raw <- df[!(bad_target | bad_pred), , drop = FALSE] %>%
    dplyr::select(profile_id, x, y, dplyr::all_of(target), target_native,
                  target_transform, dplyr::all_of(preds), .row) %>%
    dplyr::distinct(profile_id, .keep_all = TRUE)
  if (nrow(raw) == 0L) stop("No rows remained after QC.", call. = FALSE)
  say("Rows after QC: ", nrow(raw))

  # ── 7. predictor types ─────────────────────────────────────────────────────
  types <- purrr::map_dfr(preds, function(nm) {
    v    <- raw[[nm]]
    vu   <- v[!is.na(v) & is.finite(v)]
    uniq <- sort(unique(vu))
    tibble::tibble(
      predictor     = nm,
      n_unique      = length(uniq),
      min_value     = min(vu, na.rm = TRUE),
      max_value     = max(vu, na.rm = TRUE),
      is_dummy      = length(uniq) <= 2 && all(uniq %in% c(0, 1)),
      is_percentage = nm %in% pct_all
    )
  })
  auto_dummy <- identical(dummy, "auto")
  if (!auto_dummy) {
    dummy_clean <- unique(vapply(dummy, janitor::make_clean_names, character(1)))
    gone <- setdiff(dummy_clean, preds)
    if (length(gone) > 0L) {
      stop("dummy names not among the rasters: ", paste(gone, collapse = ", "),
           call. = FALSE)
    }
    both <- intersect(dummy_clean, pct_all)
    if (length(both) > 0L) {
      stop("Declared both a dummy and a percentage: ",
           paste(both, collapse = ", "), ". A channel has one type.",
           call. = FALSE)
    }
    types$is_dummy <- types$predictor %in% dummy_clean
  }
  # A DECLARATION WINS OVER THE DETECTION (see the header).
  types$is_dummy[types$is_percentage] <- FALSE

  # ── 8. the row key, and channels that cannot be scaled ────────────────────
  #
  # THIS FUNCTION DECIDES NO ROLES. Which rows train, validate and test is
  # decided by a fold plan, from coordinates, in seconds -- a split carried
  # inside the store would cost a re-extraction to change.
  split     <- dplyr::mutate(raw, sample_id = dplyr::row_number())
  preds_all <- preds

  dummy_cols <- types$predictor[types$is_dummy]
  sds <- vapply(preds, function(nm) stats::sd(split[[nm]], na.rm = TRUE),
                numeric(1))
  bad_scaling <- names(sds)[is.na(sds) | !is.finite(sds) |
                              (sds <= 0 & !(names(sds) %in% dummy_cols))]
  if (length(bad_scaling) > 0L) {
    say("\nDropping channel(s) with no variance over the points (a z-score ",
        "would divide by zero): ", paste(bad_scaling, collapse = ", "))
    preds <- setdiff(preds, bad_scaling)
    types <- dplyr::filter(types, predictor %in% preds)
  }
  n_dummy <- sum(types$is_dummy)
  n_pct   <- sum(types$is_percentage)
  n_cont  <- sum(!types$is_dummy & !types$is_percentage)
  say("\nPredictor types -- dummy: ", n_dummy, " | percentage: ", n_pct,
      " | continuous: ", n_cont)
  if (auto_dummy && n_dummy > 0L) {
    dn <- types$predictor[types$is_dummy]
    say("  detected as dummy (0/1 at the points): ",
        paste(utils::head(dn, 8L), collapse = ", "),
        if (length(dn) > 8L) sprintf(" ... and %d more (predictor_type_table.csv)",
                                     length(dn) - 8L) else "")
  }

  # ── 9. the tables ──────────────────────────────────────────────────────────
  dataset_check <- tibble::tibble(
    target_col              = target,
    n_rows                  = nrow(split),
    n_profiles              = dplyr::n_distinct(split$profile_id),
    min_target              = min(split$target_native, na.rm = TRUE),
    q01_target              = as.numeric(stats::quantile(split$target_native, 0.01, na.rm = TRUE)),
    median_target           = stats::median(split$target_native, na.rm = TRUE),
    mean_target             = mean(split$target_native, na.rm = TRUE),
    q99_target              = as.numeric(stats::quantile(split$target_native, 0.99, na.rm = TRUE)),
    max_target              = max(split$target_native, na.rm = TRUE),
    median_target_transform = stats::median(split$target_transform, na.rm = TRUE),
    n_predictors            = length(preds),
    n_dummy_predictors      = n_dummy,
    n_percentage_predictors = n_pct,
    n_continuous_predictors = n_cont)
  if (verbose) {
    message("\n-- Dataset summary --")
    print_wide(dataset_check)
  }

  point_table <- dplyr::select(split, profile_id, sample_id, x, y,
                               dplyr::all_of(target), target_native,
                               target_transform, dplyr::all_of(preds))

  # CHANNEL RISK. Three families of channel have shipped a broken map before,
  # and none shows up as an error or a bad metric -- only as holes or as
  # extrapolation, at the very end:
  #   constant       zero information at the points but NOT constant over the
  #                  map (glaciers are 0 at every profile and 1 over ice); its
  #                  weights never get a gradient and stay at random init.
  #   near_constant  the same, smaller.
  #   has_na         NA at points where OTHER channels have data.
  #
  # has_na, twice corrected. Stage 01 counted it after the QC had dropped
  # every row with an NA, so it could never fire. The first correction counted
  # every extracted row -- which would have flagged EVERY channel for every
  # point in the ocean, where the whole stack is nodata, and named nothing.
  # A point that is NA everywhere says nothing about any channel; the blame
  # report below draws the same line with its n_sole_cause. So the count is
  # over the points that have data in at least one channel.
  na_mat    <- !is.finite(as.matrix(df[, preds_all]))
  with_data <- rowSums(na_mat) < ncol(na_mat)
  n_na_part <- colSums(na_mat[with_data, , drop = FALSE])
  channel_risk <- types %>%
    dplyr::mutate(
      n_na_at_points = as.integer(n_na_part[predictor]),
      pct_na = round(100 * n_na_at_points / max(1L, sum(with_data)), 3),
      type   = dplyr::case_when(is_dummy ~ "dummy",
                                is_percentage ~ "percentage",
                                TRUE ~ "continuous"),
      risk   = dplyr::case_when(
        n_unique <= 1L             ~ "constant",
        n_unique <= 2L & !is_dummy ~ "near_constant",
        n_na_at_points > 0L        ~ "has_na",
        TRUE                       ~ "")) %>%
    dplyr::select(predictor, type, n_unique, min_value, max_value,
                  n_na_at_points, pct_na, risk)
  .prep_report_risk(channel_risk, run_profile, nrow(split), say, verbose)

  pct_final <- types$predictor[types$is_percentage]
  qc_table  <- make_qc_table(
    predictors   = preds,
    na_below     = na_below,
    clamp_range  = if (is.null(percentage_limits)) character(0) else pct_final,
    clamp_limits = percentage_limits %||% c(0, 100))

  raster_used <- rt$use[match(preds, rt$use$predictor), , drop = FALSE]

  target_config <- tibble::tibble(
    run_profile                   = run_profile,
    subsample                     = subsample_note,
    target_label                  = target_label,
    target_col                    = target,
    target_unit                   = target_unit %||% NA_character_,
    target_transform              = transform,
    target_min                    = target_min %||% NA_real_,
    predictor_raster_dir          = raster_dir,
    manual_predictor_drop         = paste(rt$drop_present, collapse = ";"),
    na_below                      = .prep_serialise_rules(na_below),
    percentage_predictor_patterns = paste(percentage, collapse = ";"),
    dummy_rule                    = if (auto_dummy) "auto" else paste(dummy, collapse = ";"),
    n_predictors_final            = length(preds),
    n_dummy                       = n_dummy,
    n_percentage                  = n_pct,
    n_continuous                  = n_cont,
    n_rows_after_qc               = nrow(raw),
    points_source                 = points_source)

  safe_write_csv2(point_table, points_file)
  safe_write_csv2(channel_risk,  file.path(metadata_dir, "channel_risk.csv"))
  safe_write_csv2(qc_table,      file.path(metadata_dir, "qc_table.csv"))
  safe_write_csv2(qc_summary,    file.path(metadata_dir, "qc_summary.csv"))
  safe_write_csv2(types,         file.path(metadata_dir, "predictor_type_table.csv"))
  safe_write_csv2(rt$all,        file.path(metadata_dir, "raster_table_all.csv"))
  safe_write_csv2(raster_used,   file.path(metadata_dir, "raster_table_used.csv"))
  safe_write_csv2(dataset_check, file.path(metadata_dir, "dataset_check.csv"))
  safe_write_csv2(dplyr::select(split, profile_id, sample_id, x, y,
                                target_native, target_transform),
                  file.path(metadata_dir, "point_metadata.csv"))
  safe_write_csv2(target_config, file.path(metadata_dir, "target_config.csv"))

  # ── 10. the rows and channels that survived ───────────────────────────────
  #
  # The pass read every point and every channel; the store keeps the rows of
  # the point table, in its order, and the channels that survived. The blame
  # matrix and the edge flags are cut here. The arrays are cut ONCE, at the
  # write, with the window rule folded into the same index -- cutting them
  # here as well would copy every window twice.
  rows_final <- split$.row
  ch_final   <- match(preds, preds_all)
  blame   <- ex$blame[rows_final, ch_final, drop = FALSE]
  edge_ok <- ex$edge_ok[rows_final]

  # ── 11. who invalidated what ───────────────────────────────────────────────
  #
  # The full-window rule turns ONE non-finite pixel into a lost patch, so one
  # sparse-NA channel can empty a map -- it happened here: a test tile covered
  # 0.72% until the percentage clamp was fixed. An AND over every channel
  # cannot say which one did it; this table can. A channel high in
  # n_sole_cause is actionable. A high n_invalidated with no sole cause is a
  # place where the whole stack is nodata (coast, water, raster edge), and
  # dropping a channel would recover nothing.
  n_points  <- nrow(point_table)
  valid     <- edge_ok & rowSums(blame) == 0L
  n_valid   <- sum(valid)
  valid_idx <- which(valid)
  blame_report <- tibble::tibble(
    predictor     = preds,
    type          = dplyr::case_when(types$is_dummy ~ "dummy",
                                     types$is_percentage ~ "percentage",
                                     TRUE ~ "continuous"),
    n_invalidated = as.integer(colSums(blame)),
    n_sole_cause  = as.integer(colSums(blame & (rowSums(blame) == 1L)))) %>%
    dplyr::mutate(pct_invalidated = round(100 * n_invalidated / n_points, 3),
                  pct_sole_cause  = round(100 * n_sole_cause  / n_points, 3)) %>%
    dplyr::arrange(dplyr::desc(n_invalidated))
  safe_write_csv2(blame_report,
                  file.path(metadata_dir, "patches", "channel_invalidation.csv"))
  .prep_report_blame(blame_report, blame, n_points, n_valid, max(windows), say,
                     verbose)
  if (n_valid == 0L) stop("No point is valid after the window rule.", call. = FALSE)

  # ── 12. persist, metadata first ────────────────────────────────────────────
  #
  # patch_meta.csv defines WHICH points survived, and every window file must
  # line up with it row for row; written first, a crash mid-write leaves a
  # store that can be understood. Then one window at a time, largest first,
  # each released before the next, so the peak is bounded by the largest
  # window rather than by all of them. saveRDS of the plain array, NOT
  # torch_save of a tensor: torch_save() in this torch build corrupts silently
  # above 2^31 bytes (see R/dataset.R).
  meta_valid <- point_table[valid_idx, c("profile_id", "sample_id", "x", "y",
                                         "target_native", "target_transform")]
  safe_write_csv2(meta_valid, file.path(store_dir, "patch_meta.csv"))

  keys <- patch_window_key(windows)
  src_idx <- rows_final[valid_idx]     # rows of the pass's arrays, in store order
  # A sample for visual inspection (99b), taken while the arrays are in memory.
  sample_idx <- with_local_seed(42L, sort(sample(n_valid, min(6L, n_valid))))
  safe_save_rds(
    list(meta       = meta_valid[sample_idx, ],
         windows    = stats::setNames(lapply(keys, function(k)
           ex$patch_list[[k]][src_idx[sample_idx], ch_final, , , drop = FALSE]), keys),
         predictors = preds),
    file.path(store_dir, "patch_sample.rds"), compress = TRUE)

  saved <- tibble::tibble()
  for (wi in order(windows, decreasing = TRUE)) {
    w   <- windows[wi]
    key <- keys[wi]
    arr <- ex$patch_list[[key]][src_idx, ch_final, , , drop = FALSE]
    res <- save_patch_window(arr, store_dir, w)
    rm(arr)
    ex$patch_list[[key]] <- NULL
    invisible(gc(verbose = FALSE))
    say(sprintf("  %-20s written %.2f GB  %s", res$file, res$gb,
                if (res$ok) "[size ok]" else
                  sprintf("[SIZE MISMATCH: expected %.2f GB]", res$exp_gb)))
    saved <- dplyr::bind_rows(saved, tibble::tibble(
      window = w, file = res$file, gb = res$gb,
      status = if (res$ok) "written" else "size_mismatch"))
  }

  # The spec this store was built under. Three things force a re-extraction --
  # the predictors, the windows and the target -- and check_store_spec() reads
  # them from here to refuse, in seconds, a configuration the store cannot
  # serve. Same columns as stage 02 wrote, so every existing reader works.
  manifest <- tibble::tibble(
    target_label         = target_label,
    store_complete       = nrow(saved) == length(windows) &&
                           all(saved$status != "size_mismatch"),
    n_channels           = length(preds),
    n_points_input       = n_points,
    n_points_valid       = n_valid,
    pct_removed          = round(100 * (n_points - n_valid) / n_points, 2),
    windows_extracted    = paste(windows, collapse = ", "),
    storage              = "rds_double_one_file_per_window",
    scaling_applied      = FALSE,
    qc_applied           = "qc_table.csv",
    chunk_nrows_used     = chunk_nrows,
    predictor_cols_final = paste(preds, collapse = ";"),
    target_col           = target,
    target_transform     = transform,
    cell_size            = ex$cell_size,
    raster_nrow          = ex$n_rows,
    raster_ncol          = ex$n_cols,
    extracted_at         = as.character(Sys.time()))
  safe_save_rds(manifest, file.path(store_dir, "patch_manifest.rds"),
                compress = FALSE)
  safe_write_csv2(manifest, file.path(metadata_dir, "patches", "patch_manifest.csv"))
  safe_write_csv2(saved,    file.path(metadata_dir, "patches", "patch_files.csv"))

  # ── 13. the recipe, and the tables the store needs, INSIDE the store ──────
  store_files <- list(points       = "points.csv",
                      type_table   = "predictor_type_table.csv",
                      qc_table     = "qc_table.csv",
                      raster_table = "raster_table_used.csv")
  safe_write_csv2(point_table, file.path(store_dir, store_files$points))
  safe_write_csv2(types,       file.path(store_dir, store_files$type_table))
  safe_write_csv2(qc_table,    file.path(store_dir, store_files$qc_table))
  safe_write_csv2(raster_used, file.path(store_dir, store_files$raster_table))

  recipe <- list(
    recipe_version    = 1L,
    created_at        = as.character(Sys.time()),
    target            = target,
    target_label      = target_label,
    target_unit       = target_unit,
    transform         = transform,
    target_min        = target_min,
    profile_id        = profile_id,
    windows           = windows,
    percentage        = percentage,
    percentage_limits = percentage_limits,
    dummy             = dummy,
    drop              = drop,
    drop_applied      = rt$drop_present,
    dropped_no_variance = bad_scaling,
    na_below          = na_below,
    subsample         = subsample,
    run_profile       = run_profile,
    predictors        = preds,
    types             = dplyr::select(types, predictor, is_dummy, is_percentage),
    raster_dir        = raster_dir,
    raster_pattern    = raster_pattern,
    raster_crs        = terra::crs(stack, proj = TRUE),
    cell_size         = ex$cell_size,
    n_points_input    = n_points,
    n_points_valid    = n_valid,
    n_cores           = ex$n_workers,
    gdal_cache_mb     = ex$gdal_cache_mb,
    n_reads_per_band  = ex$n_reads,
    files             = store_files)
  safe_save_rds(recipe, file.path(store_dir, "recipe.rds"), compress = FALSE)

  out <- structure(
    list(store_dir = store_dir, metadata_dir = metadata_dir,
         points_file = points_file, recipe = recipe, manifest = manifest),
    class = "dsm_store")
  say(sprintf("\nStore written in %.1f min.",
              as.numeric(difftime(Sys.time(), t_start, units = "mins"))))
  if (verbose) print(out)
  invisible(out)
}

#' Print a `dsm_store`
#'
#' @param x   A `dsm_store`, from [dsm_prepare()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.dsm_store <- function(x, ...) {
  r <- x$recipe
  cat("<dsm_store> ", x$store_dir, "\n", sep = "")
  cat("  points     : ", format(r$n_points_valid, big.mark = ","), " valid of ",
      format(r$n_points_input, big.mark = ","), if (identical(r$run_profile, "dev"))
        "  (SUBSAMPLE)" else "", "\n", sep = "")
  cat("  channels   : ", length(r$predictors), "  (", sum(r$types$is_dummy),
      " dummy, ", sum(r$types$is_percentage), " percentage, ",
      sum(!r$types$is_dummy & !r$types$is_percentage), " continuous)\n", sep = "")
  cat("  windows    : ", paste(r$windows, collapse = ", "), "\n", sep = "")
  cat("  target     : ", r$target, "  (trained as ", r$transform, ")\n", sep = "")
  cat("  cell size  : ", format(r$cell_size, digits = 8), "\n", sep = "")
  invisible(x)
}

# ── helpers ───────────────────────────────────────────────────────────────────

# A store_dir that already holds a store is refused unless overwrite = TRUE.
# The obvious alternative -- stage 02's "skip a window file already of the right
# size" -- is exactly wrong for a function: two stores of the same shape built
# from different points or predictors have files of identical size, and the
# skip would keep the old patches under the new manifest.
.prep_clear_store <- function(store_dir, overwrite, say) {
  if (!dir.exists(store_dir)) return(invisible(character(0)))
  own <- c("patch_manifest.rds", "patch_meta.csv", "patch_sample.rds",
           "recipe.rds", "points.csv", "predictor_type_table.csv",
           "qc_table.csv", "raster_table_used.csv")
  existing <- c(file.path(store_dir, own)[file.exists(file.path(store_dir, own))],
                list.files(store_dir, pattern = "^patches_w[0-9]+\\.rds$",
                           full.names = TRUE))
  if (length(existing) == 0L) return(invisible(character(0)))
  if (!isTRUE(overwrite)) {
    stop("store_dir already holds a patch store:\n  ", store_dir,
         "\n  Pass overwrite = TRUE to replace it, or write somewhere else.",
         call. = FALSE)
  }
  unlink(existing)
  say("Removed the previous store in ", store_dir, " (", length(existing),
      " file(s)).")
  invisible(existing)
}

.prep_read_points <- function(points) {
  if (is.character(points) && length(points) == 1L) {
    if (!file.exists(points)) stop("Point file not found: ", points, call. = FALSE)
    if (!requireNamespace("sf", quietly = TRUE)) {
      stop("Reading a spatial file needs the sf package.", call. = FALSE)
    }
    points <- sf::st_read(points, quiet = TRUE)
  }
  if (!is.data.frame(points)) {
    stop("points must be an sf object, a data.frame with coordinate columns, ",
         "or a path to a spatial file.", call. = FALSE)
  }
  if (inherits(points, "sf") && !requireNamespace("sf", quietly = TRUE)) {
    stop("points is an sf object but the sf package is not installed.",
         call. = FALSE)
  }
  points
}

# THE NAMES, CLEANED ONE BY ONE. janitor::make_clean_names() over the whole
# vector de-duplicates as it goes ("a_b", "a_b_2"), so stage 01's duplicate
# check after it could never fire: two files that clean to the same name were
# silently renamed instead of refused. Cleaning each name alone gives the same
# names when there is no clash and makes a clash visible.
.prep_clean <- function(x) {
  if (length(x) == 0L) return(character(0))
  vapply(x, janitor::make_clean_names, character(1), USE.NAMES = FALSE)
}

.prep_raster_table <- function(raster_dir, pattern, drop, say) {
  files <- list.files(raster_dir, pattern = pattern, full.names = TRUE,
                      recursive = FALSE)
  if (length(files) == 0L) {
    stop("No file matching '", pattern, "' in ", raster_dir, call. = FALSE)
  }
  all <- tibble::tibble(
    raster_file     = files,
    raster_name_raw = tools::file_path_sans_ext(basename(files)),
    predictor       = .prep_clean(tools::file_path_sans_ext(basename(files))))
  dup <- unique(all$predictor[duplicated(all$predictor)])
  if (length(dup) > 0L) {
    stop("These files clean to the same predictor name, so their channels ",
         "could not be told apart:\n  ",
         paste(vapply(dup, function(d) paste0(d, " <- ",
               paste(basename(all$raster_file[all$predictor == d]), collapse = ", ")),
               character(1)), collapse = "\n  "),
         "\n  Rename the files.", call. = FALSE)
  }

  drop_clean   <- unique(.prep_clean(drop))
  drop_present <- intersect(drop_clean, all$predictor)
  drop_missing <- setdiff(drop_clean, all$predictor)
  if (length(drop_present) > 0L) {
    say("Dropping as declared: ", paste(drop_present, collapse = ", "))
  }
  if (length(drop_missing) > 0L) {
    say("Drop entries that match no raster: ", paste(drop_missing, collapse = ", "))
  }

  use <- all %>%
    dplyr::filter(!predictor %in% drop_present) %>%
    dplyr::arrange(predictor)

  # ONE PREDICTOR PER FILE, ON ONE GRID. terra::rast() on a misaligned set
  # fails with "extents do not match" and names no file; a multi-band file
  # would shift every channel after it by one. Both are checked here, file by
  # file, so the message can say which.
  ref <- terra::rast(use$raster_file[1])
  bad_layers <- character(0)
  bad_geom   <- character(0)
  # The bytes a cell takes ON DISK, the widest over the files: it sizes the
  # GDAL block cache the extraction needs (see .prep_extract_patches).
  # Unknown types count as 8, which only makes the cache larger than needed.
  type_bytes <- c(INT1U = 1, INT1S = 1, INT2U = 2, INT2S = 2, INT4U = 4,
                  INT4S = 4, FLT4S = 4, INT8U = 8, INT8S = 8, FLT8S = 8)
  max_cell_bytes <- 1
  for (f in use$raster_file) {
    r <- terra::rast(f)
    if (terra::nlyr(r) != 1L) bad_layers <- c(bad_layers, basename(f))
    if (!terra::compareGeom(ref, r, stopOnError = FALSE)) {
      bad_geom <- c(bad_geom, basename(f))
    }
    dt <- tryCatch(terra::datatype(r)[1], error = function(e) NA_character_)
    b  <- if (!is.na(dt) && dt %in% names(type_bytes)) type_bytes[[dt]] else 8
    max_cell_bytes <- max(max_cell_bytes, b)
  }
  if (length(bad_layers) > 0L) {
    stop("One predictor per file, and these have more than one band: ",
         paste(bad_layers, collapse = ", "), call. = FALSE)
  }
  if (length(bad_geom) > 0L) {
    stop("These rasters are not on the same grid as ", basename(use$raster_file[1]),
         " (extent, rows/columns or CRS differ):\n  ",
         paste(bad_geom, collapse = ", "),
         "\n  Every predictor must stack pixel for pixel.", call. = FALSE)
  }
  list(all = all, use = use, drop_present = drop_present,
       drop_missing = drop_missing, max_cell_bytes = max_cell_bytes)
}

# Regex patterns over cleaned names, reported pattern by pattern so an
# unanchored pattern that matches more than intended is visible.
.prep_match <- function(patterns, preds, what, say) {
  if (is.null(patterns) || length(patterns) == 0L) return(character(0))
  hit <- preds[vapply(preds, function(p) any(grepl(paste(patterns, collapse = "|"), p)),
                      logical(1))]
  for (pat in patterns) {
    n <- sum(grepl(pat, preds))
    if (n == 0L) say("  ", what, " pattern '", pat, "' matches no raster.")
  }
  say("  ", what, ": ", length(hit), " channel(s) matched by ",
      length(patterns), " pattern(s)")
  hit
}

.prep_serialise_rules <- function(rules) {
  if (is.null(rules) || length(rules) == 0L) return("")
  paste(sprintf("%s=%s", names(rules), format(unname(rules))), collapse = ";")
}

.prep_report_risk <- function(channel_risk, run_profile, n_rows, say, verbose) {
  flagged <- dplyr::filter(channel_risk, risk != "")
  say("\n-- Channel risk --")
  if (nrow(flagged) == 0L) {
    say("  No channel flagged.")
    return(invisible(flagged))
  }
  say("  ", nrow(flagged), " channel(s) flagged -- the kind that has broken ",
      "MAPS here, not metrics:")
  if (verbose) print_wide(dplyr::arrange(flagged, risk, dplyr::desc(pct_na)), n = Inf)
  n_const <- sum(flagged$risk == "constant")
  if (n_const > 0L) {
    say("\n  ", n_const, " constant channel(s): zero information at the points ",
        "but non-zero somewhere on the map.")
    say("  Their weights never get a gradient, so they apply a seed-dependent ",
        "bias exactly where the network extrapolates.")
    if (identical(run_profile, "dev")) {
      say("  THIS IS A SUBSAMPLE -- do not act on it yet. A rare class is ",
          "constant at ", format(n_rows, big.mark = ","), " points because the ")
      say("  subsample missed it; dropping it would change the predictor set of ",
          "the full run from an artefact of the draw.")
    } else {
      say("  Add them to `drop`, or keep them deliberately.")
    }
  }
  n_na_ch <- sum(flagged$risk == "has_na")
  if (n_na_ch > 0L) {
    # Not "add them to drop", as for the constant channels: whether a channel
    # is worth the points it costs is a judgement about the variable, so the
    # report gives the count that judgement needs and names the option.
    say("\n  ", n_na_ch, " has_na channel(s): NA at points where other channels ",
        "have data. The QC drops every such point (n_na_at_points), and the map")
    say("  will have a hole wherever the channel is nodata. A channel that costs ",
        "many points is a candidate for `drop`.")
  }
  invisible(flagged)
}

.prep_report_blame <- function(blame_report, blame, n_points, n_valid, w_max,
                               say, verbose) {
  n_blamed <- sum(rowSums(blame) > 0)
  say("\n-- Which channels invalidated points (full-window rule) --")
  say("  window rule lost : ", format(n_blamed, big.mark = ","),
      " point(s) to non-finite values")
  say("  final valid      : ", format(n_valid, big.mark = ","), " / ",
      format(n_points, big.mark = ","), "  (",
      round(100 * (n_points - n_valid) / n_points, 2), "% removed)")
  top <- dplyr::filter(blame_report, n_invalidated > 0L)
  if (nrow(top) == 0L) {
    say("  No channel invalidated a single point.")
    return(invisible(NULL))
  }
  if (verbose) print_wide(dplyr::slice_head(top, n = 15), n = Inf)
  worst <- top[which.max(top$pct_sole_cause), ]
  if (worst$pct_sole_cause[1] > 0.1) {
    say("\n  WARNING: '", worst$predictor[1], "' ALONE lost ",
        round(worst$pct_sole_cause[1], 3), "% of the points. Sparse NA does ",
        "far more damage to the MAP,")
    say("  where each NA pixel becomes a hole of up to ", w_max, "x", w_max,
        ". Check its coverage before a full prediction run.")
  } else if (max(top$pct_invalidated) > 1) {
    say("\n  ", round(max(top$pct_invalidated), 2), "% of points lost, but no ",
        "channel is the SOLE cause of any: that is where the whole")
    say("  stack is nodata (coast, water, the raster edge). Dropping a ",
        "channel would recover none of them.")
  }
  invisible(NULL)
}

# ── the extraction ────────────────────────────────────────────────────────────
#
# ONE PASS, EVERY BAND READ ONCE: every point's centre, and the patches.
#
# The centre of every point is needed before anything else can happen -- the
# QC, the types and the variance filter all run on the values at the points.
# Stage 01 read them with terra::extract(), in series; this reads them here,
# in the same pass, from the same windows, in parallel. A point far enough
# from the edge gets its patches and its centre from one read; a point inside
# the raster but too close to its edge for the largest window still needs a
# centre (it stays in the point table, it only has no patch), so it gets a
# 1 x 1 read of its own; a point outside the raster gets nothing and stays NA,
# which the QC then drops -- terra::extract() returned NA for it too.
#
# READ ONLY WHAT THE PATCHES NEED.
#
# Stage 02 read, for every band and every chunk of 1,000 rows that held a
# point, the FULL WIDTH of the raster: 1,014 rows x 160,298 columns = 1.3 GB of
# doubles, to cut out a few hundred 15 x 15 patches. One band at a time that
# fit. The first parallel version of this function ran fifteen of those at
# once, and the copies qc_band_values() makes on the way pushed the peak to
# ~100 GB on a 63 GB machine -- the run of 2026-09-26 was interrupted for it.
#
# So the points of each row chunk are sorted by column and grouped, and each
# group is read as its own window: the rows its points span plus the half
# window, the columns likewise. A read is bounded by chunk_nrows x
# read_max_cols, whatever the grid -- 33 MB at the defaults, against 1.3 GB --
# and the memory per worker no longer depends on the width of the world.
#
# WHAT THIS DOES NOT SAVE, AND WHAT MAKES UP FOR IT. The SOC rasters are
# stored in strips of one full row, LZW-compressed (every one of the 187). A
# 15-column window still makes GDAL decompress the whole of each row it
# touches, so decompression is not what shrinks. What shrinks is everything
# after it: R no longer converts and allocates 160,298 doubles per row for the
# sake of 15. And each row is decompressed ONCE per band, not once per read,
# provided GDAL's block cache holds a chunk's rows -- which is why the cache
# is sized here, per worker, from the chunk and the row width, and why a band
# is opened once (readStart) and read window by window (readValues):
# terra::values() opens and closes the file on every call, and closing it
# empties the cache.
#
# ONE BAND PER CORE, AND THE RESULT DOES NOT DEPEND ON HOW MANY. Each band is
# its own file and each point belongs to exactly one read, so a worker's band
# depends on no other band, and the parent places the arrays by index. Nothing
# a worker does depends on what another did, or on the order they finish; and
# nothing depends on how the reads were cut, because a patch is the raster's
# cells around its point however they were fetched. The test suite checks both
# -- two cores against one, and one read per point against the defaults.
#
# The workers get the plan once (clusterExport(), as .prep_job) and a band
# index per call. Their functions are top-level on purpose: a closure defined inside
# dsm_prepare() would be serialised together with dsm_prepare()'s frame --
# which holds the patch arrays -- and sent to every worker.
.prep_extract_patches <- function(files, qc_table, xy, windows, chunk_nrows,
                                  n_cores, max_ram_gb, read_gap, read_max_cols,
                                  cell_bytes, say) {
  t0     <- Sys.time()
  ref    <- terra::rast(files[1])
  # terra returns the grid size as a double; the manifest keeps it as terra
  # gives it (stage 02 did), the indexing below uses integers.
  n_rows_raw <- terra::nrow(ref)
  n_cols_raw <- terra::ncol(ref)
  n_rows <- as.integer(n_rows_raw)
  n_cols <- as.integer(n_cols_raw)
  n_ch   <- length(files)
  n_pts  <- nrow(xy)

  cells   <- terra::cellFromXY(ref, xy)
  row_ids <- as.integer(terra::rowFromCell(ref, cells))
  col_ids <- as.integer(terra::colFromCell(ref, cells))

  # The edge check, once, against every window -- so all windows share one
  # surviving set of points. It also guarantees every patch read lies inside
  # the raster, so no read is ever clipped.
  inside  <- !is.na(row_ids) & !is.na(col_ids)
  edge_ok <- inside
  for (w in windows) {
    ok <- patch_centre_in_bounds(row_ids, col_ids, n_rows, n_cols, w)
    ok[is.na(ok)] <- FALSE
    edge_ok <- edge_ok & ok
  }
  say(sprintf("\nEdge check: %d of %d point(s) far enough from the edge for a %dx%d patch; %d more inside the raster (centre only); %d outside it",
              sum(edge_ok), n_pts, max(windows), max(windows),
              sum(inside & !edge_ok), sum(!inside)))
  if (!any(inside)) stop("No point falls inside the rasters.", call. = FALSE)

  # ── the reading plan: geometry, computed once, reused by every band ───────
  #
  # Two kinds of read. PATCH reads, for the points that get patches: their
  # windows plus the half window. CENTRE reads, for the points too close to
  # the edge: the cells themselves (half window 0). Both are grouped by
  # column within a row chunk the same way.
  h <- (max(windows) - 1L) %/% 2L
  reads <- list()
  n_chunks <- 0L
  for (cs in seq(1L, n_rows, by = chunk_nrows)) {
    ce <- min(cs + chunk_nrows - 1L, n_rows)
    in_rows <- inside & row_ids >= cs & row_ids <= ce
    in_rows[is.na(in_rows)] <- FALSE
    if (!any(in_rows)) next
    n_chunks <- n_chunks + 1L
    for (kind in c("patch", "centre")) {
      sel <- in_rows & (if (identical(kind, "patch")) edge_ok else !edge_ok)
      if (!any(sel)) next
      hh  <- if (identical(kind, "patch")) h else 0L
      idx <- which(sel)
      idx <- idx[order(col_ids[idx])]
      for (g in .prep_column_groups(col_ids[idx], hh, read_gap, read_max_cols)) {
        gi <- idx[g]
        r0 <- min(row_ids[gi]) - hh
        r1 <- max(row_ids[gi]) + hh
        c0 <- min(col_ids[gi]) - hh
        c1 <- max(col_ids[gi]) + hh
        width <- c1 - c0 + 1L
        lr <- row_ids[gi] - r0 + 1L
        lc <- col_ids[gi] - c0 + 1L
        reads[[length(reads) + 1L]] <- list(
          idx = gi, row = r0, nrows = r1 - r0 + 1L, col = c0, ncols = width,
          centre_cell = patch_cell_index(lr, lc, width, 1L)[, 1],
          cell_mats = if (identical(kind, "patch"))
            lapply(windows, function(w) patch_cell_index(lr, lc, width, w)) else NULL)
      }
    }
  }
  cells_read  <- sum(vapply(reads, function(rd) as.numeric(rd$nrows) * rd$ncols, numeric(1)))
  max_read_gb <- max(vapply(reads, function(rd) as.numeric(rd$nrows) * rd$ncols, numeric(1))) * 8 / 1e9
  say(sprintf("Reading plan: %s window read(s) per band over %d row chunk(s); %.3g%% of the raster's cells, largest read %.0f MB",
              format(length(reads), big.mark = ","), n_chunks,
              100 * cells_read / (as.numeric(n_rows) * n_cols), max_read_gb * 1e3))

  # ── memory: the GDAL cache that makes one decompression per row enough ────
  #
  # A chunk's rows, full width, at the bytes a cell takes on disk -- plus 10%.
  # Floored at 64 MB (GDAL's own minimum is sane, not generous) and capped at
  # 4 GB (a larger chunk should be a smaller chunk_nrows, not a larger cache).
  gdal_cache_mb <- ceiling(min(4096, max(64,
    1.1 * (chunk_nrows + 2 * h) * as.numeric(n_cols) * cell_bytes / 2^20)))
  arrays_gb <- n_pts * n_ch * sum(windows^2) * 8 / 1e9
  # Per worker: its cache, the largest read three times over (the read, the
  # copy qc_band_values() makes, a temporary), its band's arrays, and a base
  # R + terra session.
  per_worker_gb <- gdal_cache_mb / 1024 + 3 * max_read_gb +
    n_pts * sum(windows^2) * 8 / 1e9 + 0.4

  budget_gb <- max_ram_gb
  if (is.null(budget_gb) && requireNamespace("ps", quietly = TRUE)) {
    avail <- tryCatch(ps::ps_system_memory()$avail / 1e9, error = function(e) NA_real_)
    if (is.finite(avail)) budget_gb <- 0.7 * avail
  }
  n_workers <- min(n_cores, n_ch)
  if (!is.null(budget_gb)) {
    room <- budget_gb - arrays_gb - 1          # the parent: the arrays and itself
    fits <- max(1L, as.integer(floor(room / per_worker_gb)))
    if (fits < n_workers) {
      say(sprintf("  RAM caps the workers at %d of the %d cores asked for (%.1f GB budget, ~%.1f GB each).",
                  fits, n_workers, budget_gb, per_worker_gb))
      n_workers <- fits
    }
  }
  say(sprintf("RAM plan: patch arrays %.1f GB in the parent; %d worker(s) x ~%.1f GB (GDAL cache %d MB each)%s",
              arrays_gb, n_workers, per_worker_gb, gdal_cache_mb,
              if (is.null(budget_gb)) "; no budget (install ps to have one measured)"
              else sprintf("; budget %.1f GB", budget_gb)))

  keys <- patch_window_key(windows)
  patch_list <- stats::setNames(
    lapply(windows, function(w) array(NA_real_, dim = c(n_pts, n_ch, w, w))), keys)
  blame  <- matrix(FALSE, nrow = n_pts, ncol = n_ch)
  centre <- matrix(NA_real_, nrow = n_pts, ncol = n_ch)

  job <- list(files = files, rules = as.data.frame(qc_table), reads = reads,
              windows = windows, n_points = n_pts, gdal_cache_mb = gdal_cache_mb)

  # The assignment into patch_list is inline on purpose: done inside a helper
  # through <<-, R copies the whole multi-GB array on every band.
  if (n_workers == 1L) {
    say("Extracting ", n_ch, " band(s) on 1 core...")
    # In this session the cache is only ever RAISED, and put back afterwards:
    # it belongs to the user's R session, not to this function.
    old_cache <- .prep_gdal_cache()
    if (isTRUE(is.finite(old_cache)) && old_cache < gdal_cache_mb) {
      terra::gdalCache(gdal_cache_mb)
      on.exit(terra::gdalCache(old_cache), add = TRUE)
    }
    say("  GDAL cache in this session: ", format(.prep_gdal_cache()), " (asked ",
        gdal_cache_mb, " MB)")
    for (i in seq_len(n_ch)) {
      res <- .prep_band_worker(i, job$files, job$rules, job$reads, job$windows,
                               job$n_points)
      for (wi in seq_along(windows)) patch_list[[keys[wi]]][, i, , ] <- res$arrays[[wi]]
      if (length(res$invalid) > 0L) blame[res$invalid, i] <- TRUE
      centre[, i] <- res$centre
      if (i %% 20L == 0L) {
        say(sprintf("  %d / %d bands  (%.1f min)", i, n_ch,
                    as.numeric(difftime(Sys.time(), t0, units = "mins"))))
      }
    }
  } else {
    say("Extracting ", n_ch, " band(s) on ", n_workers, " cores...")
    cl <- parallel::makeCluster(n_workers)
    on.exit(parallel::stopCluster(cl), add = TRUE)
    fns <- .prep_worker_functions()
    parallel::clusterExport(cl, names(fns), envir = list2env(fns))
    # THE PLAN GOES TO EACH WORKER ONCE, where the band function finds it, and
    # clusterExport() puts it there -- not an assign() in this package's code,
    # which must never write into a global environment (CRAN's rule). R CMD
    # check --as-cran found the assign() the worker setup used to make and
    # cannot see that it ran only in a worker's own session; the workers'
    # global environments are theirs, and parallel's to fill.
    parallel::clusterExport(cl, ".prep_job", envir = list2env(list(.prep_job = job)))
    got <- parallel::clusterCall(cl, fns$.prep_worker_setup, job$gdal_cache_mb)
    say("  GDAL cache per worker: ", format(got[[1]]), " (asked ", gdal_cache_mb,
        " MB) -- read back, not assumed")
    batches <- split(seq_len(n_ch), ceiling(seq_len(n_ch) / n_workers))
    for (b in batches) {
      res_list <- parallel::parLapply(cl, b, fns$.prep_worker_band)
      for (k in seq_along(b)) {
        i <- b[k]
        for (wi in seq_along(windows)) {
          patch_list[[keys[wi]]][, i, , ] <- res_list[[k]]$arrays[[wi]]
        }
        if (length(res_list[[k]]$invalid) > 0L) blame[res_list[[k]]$invalid, i] <- TRUE
        centre[, i] <- res_list[[k]]$centre
      }
      rm(res_list)
      say(sprintf("  %d / %d bands  (%.1f min)", max(b), n_ch,
                  as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    }
  }
  say(sprintf("Extraction finished in %.1f min.",
              as.numeric(difftime(Sys.time(), t0, units = "mins"))))

  list(patch_list = patch_list, blame = blame, edge_ok = edge_ok,
       centre = centre, inside = inside,
       n_rows = n_rows_raw, n_cols = n_cols_raw, cell_size = terra::res(ref)[1],
       n_workers = n_workers, gdal_cache_mb = gdal_cache_mb,
       n_reads = length(reads))
}

# Group points, already sorted by column, into reads. A point joins the
# current group when its window starts no more than `gap` columns after the
# group's window ends, and the group stays no wider than `max_cols`. Returns
# positions into the sorted vector.
.prep_column_groups <- function(cols, h, gap, max_cols) {
  n <- length(cols)
  groups <- list()
  start  <- 1L
  g_lo   <- cols[1] - h
  g_hi   <- cols[1] + h
  for (k in seq_len(n)[-1L]) {
    lo <- cols[k] - h
    hi <- cols[k] + h
    if (lo - g_hi <= gap && (max(g_hi, hi) - g_lo + 1L) <= max_cols) {
      g_hi <- max(g_hi, hi)
    } else {
      groups[[length(groups) + 1L]] <- start:(k - 1L)
      start <- k
      g_lo  <- lo
      g_hi  <- hi
    }
  }
  groups[[length(groups) + 1L]] <- start:n
  groups
}

# One band: open once, every planned window, QC, then the centre of every
# point in the read and, for a patch read, each patch size. Returns the
# arrays (NA where a point has no patch), the points whose window this band
# left non-finite, and every point's centre (NA outside the raster).
.prep_band_worker <- function(i, files, rules, reads, windows, n_points) {
  r    <- terra::rast(files[i])
  rule <- rules[i, , drop = FALSE]
  arrays  <- lapply(windows, function(w) array(NA_real_, dim = c(n_points, w, w)))
  invalid <- logical(n_points)
  centre  <- rep(NA_real_, n_points)
  terra::readStart(r)
  on.exit(terra::readStop(r), add = TRUE)
  for (rd in reads) {
    v <- terra::readValues(r, row = rd$row, nrows = rd$nrows, col = rd$col,
                           ncols = rd$ncols, mat = FALSE)
    v <- qc_band_values(v, rule)
    centre[rd$idx] <- v[rd$centre_cell]
    if (!is.null(rd$cell_mats)) {
      for (wi in seq_along(windows)) {
        pb <- patch_band_assemble(v, rd$cell_mats[[wi]], windows[wi])
        arrays[[wi]][rd$idx, , ] <- pb$array
        if (!all(pb$valid)) invalid[rd$idx[!pb$valid]] <- TRUE
      }
    }
  }
  list(arrays = arrays, invalid = which(invalid), centre = centre)
}

# In each worker, once: a GDAL cache sized to hold a chunk's rows. Without it
# every worker gets GDAL's default -- 5% of the machine's RAM, each -- and
# fifteen of them would claim three quarters of it for cache. (The plan
# arrives beside it by clusterExport(), as .prep_job.)
.prep_worker_setup <- function(gdal_cache_mb) {
  tryCatch(terra::gdalCache(gdal_cache_mb), error = function(e) NULL)
  .prep_gdal_cache()
}

# The GDAL cache size as terra reports it, or NA. Read back after setting it,
# and printed: a wrong unit would cost speed silently. (It is MB -- the P1 run
# of 2026-09-26 read back exactly the 683 it asked for.)
.prep_gdal_cache <- function() {
  suppressWarnings(as.numeric(tryCatch(terra::gdalCache(), error = function(e) NA)))[1]
}

.prep_worker_band <- function(i) {
  j <- get(".prep_job", envir = globalenv())
  .prep_band_worker(i, j$files, j$rules, j$reads, j$windows, j$n_points)
}

# WHAT A BAND WORKER RECEIVES: these six, as copies whose environment is the
# worker's global one, where they find each other -- and nothing else of the
# framework, which a band does not need. Sent as they are, their environment
# would be the package's namespace, and a worker reading one loads the
# package: the INSTALLED copy, whatever this session loaded, so a session
# working on the source tree would extract with the code of the last install
# -- and each of the workers would load torch to read rasters.
.prep_worker_functions <- function() {
  nm <- c(".prep_band_worker", ".prep_worker_setup", ".prep_worker_band",
          ".prep_gdal_cache", "qc_band_values", "patch_band_assemble")
  fns <- mget(nm, envir = environment(.prep_worker_functions))
  lapply(fns, function(f) {
    environment(f) <- globalenv()
    f
  })
}
