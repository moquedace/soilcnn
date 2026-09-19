# Predictor QC and channel scaling
#
# Splits predictor conditioning into the two halves that must live at
# different points in the pipeline:
#
#   QC        — fold-INDEPENDENT. A physically impossible value is impossible
#               regardless of which points are in the training set. Applied
#               once, at extraction, and baked into the stored patches.
#
#   scaling   — fold-DEPENDENT. mu and sigma are ESTIMATED from the training
#               points, so with k folds there are k different scalings of the
#               same patches. Applied at tensor-build time, never at
#               extraction.
#
# Mixing the two is what makes cross-validation impossible without re-reading
# every raster k times. Keeping them apart costs one broadcast per fold.
#
# The scaling itself is affine and per channel, so every predictor type is the
# same operation with different constants:
#
#   continuous   (x - mu)   / sigma      mu, sigma estimated from train
#   percentage   (x - 0)    / 100        fixed, not estimated
#   dummy        (x - 0)    / 1          identity
#
# which is exactly the shape predictor_scaling.csv already has.

# ── QC rules ──────────────────────────────────────────────────────────────────

#' Build the per-predictor QC table.
#'
#' Two rule kinds, enough for every case the pipeline has needed so far:
#'   • a physical floor/ceiling outside which a value is *wrong* -> NA
#'   • a bounded range that interpolation can legitimately overshoot -> clamp
#'
#' @param predictors      Character vector of predictor names, in channel order.
#' @param na_below        Named numeric: predictors (or regex, see `by_regex`)
#'   whose values at or below the threshold are invalid and become NA.
#'   Defaults to none: a physical floor is a property of YOUR predictors, not
#'   of the framework. The soil example passes
#'   c("surface_temperature_celsius$" = -100) for a sensor nodata code.
#' @param clamp_range     Character vector of predictors whose values are
#'   clamped into `clamp_limits` instead of discarded.
#' @param clamp_limits    Length-2 numeric, the bounds used by `clamp_range`.
#' @param by_regex        Treat the names of `na_below` as regex patterns
#'   matched against `predictors` (default TRUE, matching current behaviour
#'   where the rule was keyed on a name suffix).
#' @return A tibble with one row per predictor: predictor, na_below,
#'   clamp_lower, clamp_upper.
make_qc_table <- function(predictors,
                          na_below     = NULL,
                          clamp_range  = character(0),
                          clamp_limits = c(0, 100),
                          by_regex     = TRUE) {
  floor_val <- rep(NA_real_, length(predictors))
  if (length(na_below) > 0L && is.null(names(na_below))) {
    stop("na_below must be a NAMED numeric vector: names are the predictor ",
         "names or regex patterns, values are the thresholds.", call. = FALSE)
  }
  for (pat in names(na_below)) {
    hit <- if (by_regex) grepl(pat, predictors) else predictors == pat
    floor_val[hit] <- na_below[[pat]]
  }

  in_clamp <- predictors %in% clamp_range

  tibble::tibble(
    predictor   = predictors,
    na_below    = floor_val,
    clamp_lower = ifelse(in_clamp, clamp_limits[1], NA_real_),
    clamp_upper = ifelse(in_clamp, clamp_limits[2], NA_real_)
  )
}

#' Apply the QC rule for one channel to a vector of raw raster values.
#'
#' Order matters and matches the original: the NA floor is applied first (a
#' sensor-invalid value must not be rescued by a later clamp), then the clamp.
#' Genuine NA / Inf pass through untouched either way.
#'
#' @param values Numeric vector of raw values for one channel.
#' @param rule   One row of make_qc_table().
qc_band_values <- function(values, rule) {
  if (!is.na(rule$na_below)) {
    bad <- !is.na(values) & is.finite(values) & values <= rule$na_below
    values[bad] <- NA_real_
  }
  if (!is.na(rule$clamp_lower) || !is.na(rule$clamp_upper)) {
    fin <- !is.na(values) & is.finite(values)
    lo  <- if (is.na(rule$clamp_lower)) -Inf else rule$clamp_lower
    hi  <- if (is.na(rule$clamp_upper))  Inf else rule$clamp_upper
    values[fin] <- pmin(pmax(values[fin], lo), hi)
  }
  values
}

# ── scaling ───────────────────────────────────────────────────────────────────

#' Estimate channel scaling from a SUBSET of points.
#'
#' This is the fold-dependent half. Pass the row indices of the fold's training
#' points and you get that fold's scaling; pass the whole training split and
#' you reproduce the single-holdout behaviour.
#'
#' @param points     Data frame of point values, one column per predictor.
#' @param type_table Tibble with columns predictor, is_dummy, is_percentage,
#'   in channel order.
#' @param rows       Integer row indices of the points to estimate from.
#' @param pct_scale  Divisor for percentage predictors (default 100).
#' @return A tibble: predictor, scaling_method, center, scale — in channel
#'   order, ready for scale_patches().
fit_scaling <- function(points, type_table, rows, pct_scale = 100) {
  if (length(rows) < 2L) {
    stop("This fold has ", length(rows), " training row(s); a mean and sd ",
         "cannot be estimated from fewer than 2.\n  Use fewer folds, a smaller ",
         "test_frac, or a grouping that leaves more rows in training.",
         call. = FALSE)
  }

  out <- tibble::tibble(
    predictor      = type_table$predictor,
    scaling_method = dplyr::case_when(
      type_table$is_dummy      ~ "none_dummy_0_1",
      type_table$is_percentage ~ "percentage_to_unit",
      TRUE                     ~ "zscore_train"
    ),
    center = 0,
    scale  = 1
  )

  is_z <- out$scaling_method == "zscore_train"
  if (any(is_z)) {
    sub <- points[rows, out$predictor[is_z], drop = FALSE]
    out$center[is_z] <- vapply(sub, mean, numeric(1), na.rm = TRUE)
    out$scale[is_z]  <- vapply(sub, stats::sd, numeric(1), na.rm = TRUE)
  }
  out$scale[out$scaling_method == "percentage_to_unit"] <- pct_scale

  bad <- !is.finite(out$center) | !is.finite(out$scale) | out$scale <= 0
  if (any(bad)) {
    out$degenerate <- bad
  } else {
    out$degenerate <- FALSE
  }
  out
}

#' Apply channel scaling to a patch tensor, in place of a re-extraction.
#'
#' @param x       torch tensor [N, C, H, W].
#' @param scaling Tibble from fit_scaling(), rows in channel order.
#' @param inplace Modify `x` instead of allocating a copy. The patch tensors
#'   are multi-GB (a 15x15 window over 37k points and 181 channels is ~6 GB in
#'   float32), so the copy is not free: use inplace = TRUE right after loading
#'   a tensor nothing else holds a reference to, and FALSE whenever the caller
#'   still needs the raw values.
#' @return The scaled tensor (the same object when `inplace`).
scale_patches <- function(x, scaling, inplace = FALSE) {
  n_ch <- x$shape[[2]]
  if (nrow(scaling) != n_ch) {
    stop("scaling has ", nrow(scaling), " rows but the tensor has ", n_ch,
         " channels -- channel order must match exactly.", call. = FALSE)
  }
  shp    <- c(1L, n_ch, 1L, 1L)
  centre <- torch::torch_tensor(as.numeric(scaling$center),
                                dtype = x$dtype, device = x$device)$view(shp)
  sd_t   <- torch::torch_tensor(as.numeric(scaling$scale),
                                dtype = x$dtype, device = x$device)$view(shp)
  if (inplace) {
    x$sub_(centre)
    x$div_(sd_t)
    x
  } else {
    (x - centre) / sd_t
  }
}

#' Same operation on a plain R array, for callers not holding a tensor.
#' Apply the same affine scaling to a [n_rows, n_channels] matrix.
#'
#' The table view of the same transform `scale_patches()` applies to patches.
#' One implementation of the arithmetic, three shapes -- tensor, array, matrix
#' -- because the map and the model must be built with the same constants and
#' the same formula, and two hand-written copies of an affine transform is
#' exactly how they stop agreeing.
#'
#' @param mat     Numeric matrix, columns in channel order.
#' @param scaling From fit_scaling(), rows in the SAME channel order.
scale_patches_matrix <- function(mat, scaling) {
  if (ncol(mat) != nrow(scaling)) {
    stop("matrix has ", ncol(mat), " columns but scaling has ", nrow(scaling),
         " rows -- channel order must match exactly.", call. = FALSE)
  }
  for (i in seq_len(ncol(mat))) {
    mat[, i] <- (mat[, i] - scaling$center[i]) / scaling$scale[i]
  }
  mat
}

scale_patches_array <- function(x, scaling) {
  n_ch <- dim(x)[2]
  if (nrow(scaling) != n_ch) {
    stop("scaling has ", nrow(scaling), " rows but the array has ", n_ch,
         " channels -- channel order must match exactly.", call. = FALSE)
  }
  for (i in seq_len(n_ch)) {
    x[, i, , ] <- (x[, i, , ] - scaling$center[i]) / scaling$scale[i]
  }
  x
}
