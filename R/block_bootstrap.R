# ── Inference on clustered test points: blocks, a bootstrap of them, a correlogram
#
# WHY THIS EXISTS (2026-10-04). The SOC 0-30 cm trial compares its validation
# designs on a common test set: per profile, the difference of two models'
# absolute errors, and its mean with a bootstrap interval. That bootstrap drew
# PROFILES, one by one, as if each were an independent piece of evidence. They
# are not drawn so: soil profiles come in surveys, farms and transects, and the
# test set itself was carved as whole blocks. Its 3,900 profiles sit in 303
# one-degree blocks; half of them in 11 blocks, 603 in the largest, 3 in the
# typical one. Where the models' errors are alike within a cluster, 603
# profiles carry far less than 603 profiles' evidence, and an interval that
# counts them so is too narrow.
#
# THE QUANTITY THAT MATTERS is the one whose mean is compared: the difference
# of two models' errors at a point, not the soil property and not either
# model's error alone. Whether IT resembles itself at short distances is what
# spatial_correlogram() measures; the clustering of the points is only the
# warning.
#
# TWO QUESTIONS, NOT ONE (weights =). A bootstrap of blocks corrects the
# INTERVAL under dependence; it does not change how much each block weighs in
# the mean, and the block of 603 still outweighs the block of 3. Which weight
# is right depends on the question:
#
#   "profile"  every test profile one vote: which model errs less on a
#              profile drawn from this data base. The mean over the points,
#              its interval from whole blocks drawn with replacement.
#   "block"    every block one vote, its profiles weighing 1/n_b each: which
#              model errs less in a region. Cell declustering (Deutsch &
#              Journel 1998), the device de Bruin et al. (2022) take up for
#              map accuracy from clustered samples. The ESTIMATE moves too,
#              not only the interval -- and a block of two profiles weighs as
#              much as one of 600, so its interval is wider: the honest price.
#
# Read together: where the two disagree, the advantage rests on the densely
# sampled regions. That is a finding, not noise. And "region" is still a
# region with profiles in it; ground with none is the area of applicability's
# question and kNNDM's, not this one's.
#
# THE BLOCKS. Square cells of equal AREA, in km, on a Lambert azimuthal
# equal-area projection centred on the points (equal_area_blocks()): a degree
# of longitude shrinks from 111 km at the equator to 71 km at 50 S, and blocks
# meant to weigh regions alike must be of equal area. The other natural blocks
# are the ones a test set was drawn by -- the spatial design's: a sample drawn
# by clusters has its variance between clusters, whatever the correlogram says.
#
# WHICH SIZE. Blocks smaller than the distance over which the differences
# resemble each other are themselves dependent, and the interval is still too
# narrow. block_bootstrap_by_size() gives the interval for growing blocks:
# where its width stops growing, the blocks no longer depend on each other.
# spatial_correlogram() says it from the values themselves: Moran's I by
# distance class, near zero beyond the distance that matters.

#' Square blocks of equal area, in kilometres.
#'
#' Each point falls in a square cell of `size_km` on a Lambert azimuthal
#' equal-area projection centred on the points (on the sphere of WGS84's
#' authalic radius), so every block covers the same ground wherever it lies --
#' which a grid in degrees does not: a degree of longitude is 111 km at the
#' equator and 71 km at 50 degrees.
#'
#' @param x,y Longitude and latitude in degrees (`coords = "lonlat"`), or
#'   easting and northing in metres of an equal-area projection of yours
#'   (`coords = "metres"`).
#' @param size_km The side of a block, in km.
#' @param coords "lonlat" (the default) or "metres".
#' @param centre For "lonlat": the projection's centre, `c(longitude,
#'   latitude)`. NULL: the points' own means.
#' @return A character vector, one block per point (the cell's column and row,
#'   "i_j"), with the size and the projection as attributes.
#' @examples
#' ex <- example_landscape()
#' blk <- equal_area_blocks(ex$profiles$x, ex$profiles$y, size_km = 2)
#' length(unique(blk))    # blocks holding a profile
#' table(table(blk))      # how many blocks hold 1, 2, ... profiles
#' @export
equal_area_blocks <- function(x, y, size_km, coords = c("lonlat", "metres"), centre = NULL) {
  coords <- match.arg(coords)
  xy <- .bb_projected_km(x, y, coords, centre)
  if (!is.numeric(size_km) || length(size_km) != 1L || !is.finite(size_km) || size_km <= 0) {
    stop("size_km must be one positive number of kilometres.", call. = FALSE)
  }
  structure(paste(floor(xy$e / size_km), floor(xy$n / size_km), sep = "_"),
            size_km = size_km, projection = xy$projection)
}

# The points in km: projected from longitude and latitude, or given in metres.
.bb_projected_km <- function(x, y, coords, centre = NULL) {
  if (!is.numeric(x) || !is.numeric(y) || length(x) != length(y) || length(x) == 0L) {
    stop("x and y must be numeric coordinates of the same length.", call. = FALSE)
  }
  if (any(!is.finite(x)) || any(!is.finite(y))) {
    stop("x and y must be finite: a point without coordinates belongs to no block.", call. = FALSE)
  }
  if (identical(coords, "metres")) {
    return(list(e = x / 1000, n = y / 1000, projection = "given, in metres"))
  }
  if (any(abs(x) > 180) || any(abs(y) > 90)) {
    stop("These coordinates are not longitude and latitude in degrees. Give coords = ",
         "\"metres\" for an equal-area projection of yours.", call. = FALSE)
  }
  centre <- centre %||% c(mean(x), mean(y))
  if (!is.numeric(centre) || length(centre) != 2L || any(!is.finite(centre))) {
    stop("centre must be c(longitude, latitude).", call. = FALSE)
  }
  c(.bb_laea_km(x, y, centre),
    projection = sprintf("+proj=laea +lon_0=%.4f +lat_0=%.4f, sphere of 6371.0072 km",
                         centre[1], centre[2]))
}

# Lambert azimuthal equal-area on the sphere (Snyder 1987, eq. 24-2 to 24-4),
# with WGS84's authalic radius: the sphere of the same area as the ellipsoid,
# so areas are the ellipsoid's to well under 1%.
.bb_laea_km <- function(lon, lat, centre) {
  r  <- 6371.0072
  l  <- lon * pi / 180
  p  <- lat * pi / 180
  l0 <- centre[1] * pi / 180
  p0 <- centre[2] * pi / 180
  cos_c <- sin(p0) * sin(p) + cos(p0) * cos(p) * cos(l - l0)
  # The projection runs to the antipode, where it tears: far from its centre
  # a square of the map is a sliver of the ground. Beyond 120 degrees, refuse.
  if (any(cos_c < -0.5)) {
    stop("Some points lie more than 120 degrees from the projection's centre: one ",
         "equal-area grid cannot hold them well. Give coords = \"metres\" for a projection ",
         "of yours, or split the points.", call. = FALSE)
  }
  k <- sqrt(2 / (1 + cos_c))
  list(e = r * k * cos(p) * sin(l - l0),
       n = r * k * (cos(p0) * sin(p) - sin(p0) * cos(p) * cos(l - l0)))
}

#' A metric, or two models' difference in it, with an interval from whole blocks.
#'
#' The points are drawn by blocks: each draw takes as many blocks as there
#' are, with replacement, and every point of a block drawn comes with it. With
#' `against`, the statistic is the difference `pred` less `against`, at the
#' same points. Two weightings answer two questions: "profile", every point one
#' vote; "block", every block one vote, its points weighing 1/n each. They
#' differ in the estimate as well as in the interval, and where they disagree
#' the advantage rests on the densely sampled blocks.
#'
#' @param obs Observations.
#' @param pred Predictions at the same points.
#' @param blocks One block per point: from [equal_area_blocks()], or the blocks
#'   the points were drawn by.
#' @param against NULL, or another model's predictions at the same points.
#' @param metric One or more of "mae", "rmse", "ccc" and "bias" (the mean of
#'   pred - obs).
#' @param weights "profile" and/or "block".
#' @param n_boot Draws of the blocks.
#' @param conf The interval's level.
#' @param seed Seed of the draws.
#' @return A `block_bootstrap`: a tibble, one row per metric and weighting, with
#'   `estimate` (pred's), `against` (the other model's), `difference`,
#'   `ci_low` and `ci_high` (of the difference when `against` is given, of the
#'   estimate otherwise), `n_points`, `n_blocks`, `n_effective` (for
#'   "profile": the number of independent points that would give the same
#'   interval) and `share_better` (the share of points, or of blocks, where
#'   `pred` errs less).
#' @examples
#' ex <- example_landscape()
#' obs <- ex$profiles$soc_stock
#' set.seed(1)
#' pred_a <- obs * exp(rnorm(length(obs), 0, 0.20))   # two models' predictions
#' pred_b <- obs * exp(rnorm(length(obs), 0, 0.25))
#' blk <- equal_area_blocks(ex$profiles$x, ex$profiles$y, size_km = 2)
#' block_bootstrap(obs, pred_a, blk, against = pred_b, metric = "mae", n_boot = 500)
#' @export
block_bootstrap <- function(obs, pred, blocks, against = NULL,
                            metric = c("mae", "rmse", "ccc", "bias"),
                            weights = c("profile", "block"), n_boot = 2000L, conf = 0.95,
                            seed = 42L) {
  metric  <- match.arg(metric, several.ok = TRUE)
  weights <- match.arg(weights, several.ok = TRUE)
  n <- length(obs)
  if (length(pred) != n || length(blocks) != n || (!is.null(against) && length(against) != n)) {
    stop("obs, pred, blocks (and against) must have one value per point.", call. = FALSE)
  }
  if (!is.numeric(n_boot) || length(n_boot) != 1L || !is.finite(n_boot) || n_boot < 100 ||
      n_boot != round(n_boot)) {
    stop("n_boot must be one whole number of 100 or more.", call. = FALSE)
  }
  if (!is.numeric(conf) || length(conf) != 1L || !is.finite(conf) || conf <= 0 || conf >= 1) {
    stop("conf must be one number between 0 and 1.", call. = FALSE)
  }
  keep <- is.finite(obs) & is.finite(pred) & !is.na(blocks)
  if (!is.null(against)) keep <- keep & is.finite(against)
  if (sum(keep) < 2L) stop("Fewer than two points with a value.", call. = FALSE)
  o  <- as.numeric(obs)[keep]
  pa <- as.numeric(pred)[keep]
  pb <- if (is.null(against)) NULL else as.numeric(against)[keep]
  b  <- as.integer(factor(as.character(blocks)[keep]))
  n_blk <- max(b)
  if (n_blk < 2L) stop("The points fall in one block: there is nothing to draw.", call. = FALSE)
  # The metrics are invariant to one shift of obs and pred together; centring
  # keeps the moments' one-pass sums exact to rounding.
  shift <- mean(o)
  o <- o - shift; pa <- pa - shift
  if (!is.null(pb)) pb <- pb - shift

  m_a <- .bb_block_sums(o, pa, b, n_blk)
  m_b <- if (is.null(pb)) NULL else .bb_block_sums(o, pb, b, n_blk)
  per_block <- tabulate(b, n_blk)
  alpha <- (1 - conf) / 2
  rows <- list()
  for (w in weights) {
    scale <- if (identical(w, "block")) 1 / per_block else rep(1, n_blk)
    mats <- list(a = m_a * scale)
    if (!is.null(m_b)) mats$b <- m_b * scale
    # The same draws for both models: the difference is paired by block.
    reps <- lapply(.bb_replicate_sums(mats, n_blk, n_boot, seed), .bb_metrics)
    rep_a <- reps$a
    rep_b <- reps$b
    est <- lapply(mats, function(m) {
      .bb_metrics(matrix(colSums(m), nrow = 1L, dimnames = list(NULL, colnames(m))))
    })
    est_a <- est$a
    est_b <- est$b
    # How many independent points would give the profile interval: the same
    # statistic drawn point by point, its variance against the blocks'.
    rep_iid <- NULL
    if (identical(w, "profile")) {
      iid <- list(a = .bb_block_sums(o, pa, seq_along(o), length(o)))
      if (!is.null(pb)) iid$b <- .bb_block_sums(o, pb, seq_along(o), length(o))
      rep_iid <- lapply(.bb_replicate_sums(iid, length(o), n_boot, seed + 1L), .bb_metrics)
    }
    for (mt in metric) {
      stat <- if (is.null(rep_b)) rep_a[[mt]] else rep_a[[mt]] - rep_b[[mt]]
      q <- stats::quantile(stat, c(alpha, 1 - alpha), names = FALSE, na.rm = TRUE)
      n_eff <- NA_real_
      if (!is.null(rep_iid)) {
        s_iid <- if (is.null(rep_iid$b)) rep_iid$a[[mt]] else rep_iid$a[[mt]] - rep_iid$b[[mt]]
        v_blk <- stats::var(stat, na.rm = TRUE)
        if (is.finite(v_blk) && v_blk > 0) {
          n_eff <- min(length(o), length(o) * stats::var(s_iid, na.rm = TRUE) / v_blk)
        }
      }
      rows[[length(rows) + 1L]] <- tibble::tibble(
        metric = mt, weights = w, estimate = est_a[[mt]],
        against = if (is.null(est_b)) NA_real_ else est_b[[mt]],
        difference = if (is.null(est_b)) NA_real_ else est_a[[mt]] - est_b[[mt]],
        ci_low = q[1], ci_high = q[2], n_points = length(o), n_blocks = n_blk,
        n_effective = n_eff,
        share_better = .bb_share_better(mt, w, o, pa, pb, b, n_blk))
    }
  }
  structure(dplyr::bind_rows(rows), conf = conf, n_boot = as.integer(n_boot),
            paired = !is.null(against), class = c("block_bootstrap", "tbl_df", "tbl", "data.frame"))
}

# Per block, the sums every metric is made of: the count, the absolute and the
# squared errors, the errors, and the moments of obs and pred.
.bb_block_sums <- function(o, p, b, n_blk) {
  e <- p - o
  m <- cbind(n = tabulate(b, n_blk), abs = .bb_rowsum(abs(e), b, n_blk),
             sq = .bb_rowsum(e^2, b, n_blk), err = .bb_rowsum(e, b, n_blk),
             sx = .bb_rowsum(o, b, n_blk), sy = .bb_rowsum(p, b, n_blk),
             sxx = .bb_rowsum(o^2, b, n_blk), syy = .bb_rowsum(p^2, b, n_blk),
             sxy = .bb_rowsum(o * p, b, n_blk))
  m
}

# A sum by block, in block order, one entry for every block.
.bb_rowsum <- function(v, b, n_blk) {
  out <- numeric(n_blk)
  s <- rowsum(v, b, reorder = TRUE)
  out[as.integer(rownames(s))] <- s[, 1]
  out
}

# The draws, summed: each draw takes n_blk blocks with replacement, and its
# row of each matrix in `mats` is the sum of the drawn blocks' rows. The counts
# are made a chunk of draws at a time, so a bootstrap of thousands of points
# drawn one by one never holds a count matrix of n_boot x points; the draws are
# the same whatever the chunk.
.bb_replicate_sums <- function(mats, n_blk, n_boot, seed, chunk = 250L) {
  out <- lapply(mats, function(m) {
    matrix(NA_real_, n_boot, ncol(m), dimnames = list(NULL, colnames(m)))
  })
  with_local_seed(seed, {
    for (s in seq.int(1L, n_boot, by = chunk)) {
      e <- min(n_boot, s + chunk - 1L)
      counts <- matrix(0, e - s + 1L, n_blk)
      for (r in seq_len(e - s + 1L)) {
        counts[r, ] <- tabulate(sample.int(n_blk, n_blk, replace = TRUE), n_blk)
      }
      for (k in names(mats)) out[[k]][s:e, ] <- counts %*% mats[[k]]
    }
  })
  out
}

# The metrics from summed columns (one row per draw), as calc_metrics() and
# ccc() compute them: population moments, the bias as pred - obs.
.bb_metrics <- function(s) {
  # Each column by name, and then without it: from a one-row matrix -- the
  # estimate's sums -- s[, "abs"] comes back as a scalar named "abs", and the
  # name rode into the result's columns (tests/testthat/test-blocks.R).
  col <- function(k) unname(s[, k])
  w   <- col("n")
  mx  <- col("sx") / w
  my  <- col("sy") / w
  vx  <- col("sxx") / w - mx^2
  vy  <- col("syy") / w - my^2
  cxy <- col("sxy") / w - mx * my
  den <- vx + vy + (mx - my)^2
  list(mae = col("abs") / w, rmse = sqrt(col("sq") / w), bias = col("err") / w,
       ccc = ifelse(den > 0, 2 * cxy / den, NA_real_))
}

# Where pred errs less than against: the share of points (profile) or of
# blocks (block, on the block's mean error). Not defined for the CCC, a
# property of the whole set, nor for the bias, whose sign is not "less".
.bb_share_better <- function(mt, w, o, pa, pb, b, n_blk) {
  if (is.null(pb) || mt %in% c("ccc", "bias")) return(NA_real_)
  loss <- if (identical(mt, "rmse")) function(p) (p - o)^2 else function(p) abs(p - o)
  d <- loss(pa) - loss(pb)
  if (identical(w, "profile")) return(mean(d < 0))
  mean(.bb_rowsum(d, b, n_blk) / tabulate(b, n_blk) < 0)
}

#' Print a `block_bootstrap`
#'
#' @param x   A `block_bootstrap`, from [block_bootstrap()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.block_bootstrap <- function(x, ...) {
  tab <- tibble::as_tibble(unclass(x))
  cat(sprintf("\n<block_bootstrap> %d point(s) in %d block(s), %d draws of the blocks, %.0f%% interval%s\n",
              tab$n_points[1], tab$n_blocks[1], attr(x, "n_boot"), 100 * attr(x, "conf"),
              if (isTRUE(attr(x, "paired"))) " of the difference pred - against" else ""))
  shown <- tibble::tibble(metric = tab$metric, weights = tab$weights,
                          estimate = signif(tab$estimate, 4))
  if (isTRUE(attr(x, "paired"))) {
    shown$against <- signif(tab$against, 4)
    shown$difference <- signif(tab$difference, 4)
  }
  shown$ci_low <- signif(tab$ci_low, 4)
  shown$ci_high <- signif(tab$ci_high, 4)
  shown$n_effective <- round(tab$n_effective)
  if (isTRUE(attr(x, "paired"))) shown$share_better <- round(tab$share_better, 3)
  print(shown, n = Inf)
  cat("  profile: every point one vote; the interval from whole blocks drawn with replacement.\n")
  cat("  block: every block one vote, its points 1/n each -- the estimate moves as well.\n")
  cat("  n_effective: how many independent points would give the profile interval.\n")
  if (isTRUE(attr(x, "paired"))) {
    cat("  difference: for mae and rmse below zero is pred erring less, for ccc above zero is\n")
    cat("  pred agreeing more; an interval holding zero is not evidence either way.\n")
  }
  invisible(x)
}

#' The interval as the blocks grow.
#'
#' [block_bootstrap()] for blocks of growing side. Blocks smaller than the
#' distance over which the values resemble each other are themselves
#' dependent, and their interval is still too narrow; where the width stops
#' growing, they no longer are.
#'
#' @param x,y Coordinates, as in [equal_area_blocks()].
#' @param obs,pred,against As in [block_bootstrap()].
#' @param sizes_km The block sides to try, in km.
#' @param metric One metric, as in [block_bootstrap()].
#' @param weights One weighting, as in [block_bootstrap()].
#' @param n_boot,conf,seed As in [block_bootstrap()].
#' @param coords,centre As in [equal_area_blocks()].
#' @return A `block_bootstrap_by_size`: a tibble, one row per size, with
#'   `size_km`, `n_blocks`, the `estimate` (or the `difference`), `ci_low`,
#'   `ci_high`, `width` and `n_effective`.
#' @examples
#' ex <- example_landscape()
#' obs <- ex$profiles$soc_stock
#' set.seed(1)
#' pred_a <- obs * exp(rnorm(length(obs), 0, 0.20))   # two models' predictions
#' pred_b <- obs * exp(rnorm(length(obs), 0, 0.25))
#' s <- block_bootstrap_by_size(ex$profiles$x, ex$profiles$y, obs, pred_a, against = pred_b,
#'                              sizes_km = c(1, 2, 5, 10), n_boot = 500)
#' s
#' plot(s)
#' @export
block_bootstrap_by_size <- function(x, y, obs, pred, against = NULL,
                                    sizes_km = c(25, 50, 100, 200, 400), metric = "mae",
                                    weights = "profile", n_boot = 2000L, conf = 0.95,
                                    seed = 42L, coords = c("lonlat", "metres"), centre = NULL) {
  coords <- match.arg(coords)
  if (length(metric) != 1L || length(weights) != 1L) {
    stop("One metric and one weighting at a time.", call. = FALSE)
  }
  if (!is.numeric(sizes_km) || length(sizes_km) == 0L || any(!is.finite(sizes_km)) ||
      any(sizes_km <= 0)) {
    stop("sizes_km must be positive numbers of kilometres.", call. = FALSE)
  }
  rows <- lapply(sort(unique(sizes_km)), function(s) {
    blk <- equal_area_blocks(x, y, s, coords = coords, centre = centre)
    # Blocks so large that the points fall in one: nothing to draw, and the
    # row says so rather than the curve stopping.
    if (length(unique(blk)) < 2L) {
      return(tibble::tibble(size_km = s, n_blocks = 1L, value = NA_real_, ci_low = NA_real_,
                            ci_high = NA_real_, width = NA_real_, n_effective = NA_real_))
    }
    r <- block_bootstrap(obs, pred, blk, against = against, metric = metric, weights = weights,
                         n_boot = n_boot, conf = conf, seed = seed)
    tibble::tibble(size_km = s, n_blocks = r$n_blocks,
                   value = if (is.null(against)) r$estimate else r$difference,
                   ci_low = r$ci_low, ci_high = r$ci_high, width = r$ci_high - r$ci_low,
                   n_effective = r$n_effective)
  })
  out <- dplyr::bind_rows(rows)
  names(out)[names(out) == "value"] <- if (is.null(against)) "estimate" else "difference"
  structure(out, metric = metric, weights = weights, conf = conf,
            class = c("block_bootstrap_by_size", "tbl_df", "tbl", "data.frame"))
}

#' Plot the interval as the blocks grow.
#'
#' @param x   A `block_bootstrap_by_size`, from [block_bootstrap_by_size()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
plot.block_bootstrap_by_size <- function(x, ...) {
  tab <- tibble::as_tibble(unclass(x))
  val <- if ("difference" %in% names(tab)) tab$difference else tab$estimate
  lim <- range(c(tab$ci_low, tab$ci_high, val, if ("difference" %in% names(tab)) 0),
               na.rm = TRUE)
  graphics::plot(tab$size_km, val, log = "x", ylim = lim, pch = 16,
                 xlab = "block side (km)",
                 ylab = sprintf("%s%s, %.0f%% interval", attr(x, "metric"),
                                if ("difference" %in% names(tab)) " difference" else "",
                                100 * attr(x, "conf")),
                 main = sprintf("The interval as the blocks grow (%s weights)", attr(x, "weights")))
  # An interval of length zero is a warning from graphics, not a mark.
  open <- is.finite(tab$ci_low) & is.finite(tab$ci_high) & tab$ci_high > tab$ci_low
  if (any(open)) {
    graphics::arrows(tab$size_km[open], tab$ci_low[open], tab$size_km[open], tab$ci_high[open],
                     angle = 90, code = 3, length = 0.04)
  }
  if ("difference" %in% names(tab)) graphics::abline(h = 0, lty = 2)
  invisible(x)
}

#' Moran's I by distance class: how far apart two points stop resembling each other.
#'
#' For each class of distance, Moran's I of `value` over the pairs of points
#' that far apart: the mean product of their deviations from the mean, over
#' the variance. Near zero, points that far apart are as unlike as any two;
#' the distance where it falls to zero is the scale blocks should exceed. For
#' two models compared at the same points, `value` is the difference of their
#' errors -- the quantity whose mean is compared.
#'
#' @param x,y Coordinates: longitude and latitude in degrees (great-circle
#'   distances) or metres (straight-line distances), as `coords` says.
#' @param value One number per point.
#' @param breaks_km The edges of the distance classes, in km.
#' @param coords "lonlat" (the default) or "metres".
#' @return A `spatial_correlogram`: a tibble with `from_km`, `to_km`,
#'   `n_pairs`, `moran_i`, and `expected`, Moran's I with no spatial structure
#'   (-1/(n-1)).
#' @examples
#' ex <- example_landscape()
#' obs <- ex$profiles$soc_stock
#' set.seed(1)
#' pred_a <- obs * exp(rnorm(length(obs), 0, 0.20))   # two models' predictions
#' pred_b <- obs * exp(rnorm(length(obs), 0, 0.25))
#' d <- abs(pred_a - obs) - abs(pred_b - obs)    # what a comparison of the two compares
#' cg <- spatial_correlogram(ex$profiles$x, ex$profiles$y, d,
#'                           breaks_km = c(0, 1, 2, 5, 10, 20))
#' cg
#' plot(cg)
#' @export
spatial_correlogram <- function(x, y, value, breaks_km = c(0, 5, 10, 25, 50, 100, 200, 400, 800),
                                coords = c("lonlat", "metres")) {
  coords <- match.arg(coords)
  if (length(value) != length(x) || length(x) != length(y)) {
    stop("x, y and value must have one entry per point.", call. = FALSE)
  }
  if (!is.numeric(breaks_km) || length(breaks_km) < 2L || any(!is.finite(breaks_km)) ||
      is.unsorted(breaks_km, strictly = TRUE) || breaks_km[1] < 0) {
    stop("breaks_km must be increasing distances in km, two or more.", call. = FALSE)
  }
  keep <- is.finite(x) & is.finite(y) & is.finite(value)
  x <- as.numeric(x)[keep]; y <- as.numeric(y)[keep]
  z <- as.numeric(value)[keep]
  n <- length(z)
  if (n < 3L) stop("Fewer than three points with a value.", call. = FALSE)
  z <- z - mean(z)
  v <- mean(z^2)
  if (v == 0) stop("value is constant: it resembles itself at every distance.", call. = FALSE)
  if (identical(coords, "lonlat")) {
    if (any(abs(x) > 180) || any(abs(y) > 90)) {
      stop("These coordinates are not longitude and latitude in degrees. Give coords = ",
           "\"metres\".", call. = FALSE)
    }
    lon <- x * pi / 180; lat <- y * pi / 180
  }
  k <- length(breaks_km) - 1L
  s_prod <- numeric(k); n_pair <- numeric(k)
  # Row by row, each point against the points after it: for 4,000 points 8
  # million pairs, never held as one matrix.
  for (i in seq_len(n - 1L)) {
    j <- (i + 1L):n
    d <- if (identical(coords, "lonlat")) {
      h <- sin((lat[j] - lat[i]) / 2)^2 + cos(lat[i]) * cos(lat[j]) * sin((lon[j] - lon[i]) / 2)^2
      2 * 6371.0088 * asin(pmin(1, sqrt(h)))
    } else {
      sqrt((x[j] - x[i])^2 + (y[j] - y[i])^2) / 1000
    }
    cls <- findInterval(d, breaks_km, left.open = FALSE, rightmost.closed = FALSE)
    ok <- cls >= 1L & cls <= k
    if (any(ok)) {
      s_prod <- s_prod + .bb_rowsum(z[i] * z[j][ok], cls[ok], k)
      n_pair <- n_pair + tabulate(cls[ok], k)
    }
  }
  out <- tibble::tibble(from_km = breaks_km[-length(breaks_km)], to_km = breaks_km[-1],
                        n_pairs = n_pair,
                        moran_i = ifelse(n_pair > 0, (s_prod / n_pair) / v, NA_real_),
                        expected = -1 / (n - 1))
  structure(out, n_points = n, class = c("spatial_correlogram", "tbl_df", "tbl", "data.frame"))
}

#' Print a `spatial_correlogram`
#'
#' @param x   A `spatial_correlogram`, from [spatial_correlogram()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.spatial_correlogram <- function(x, ...) {
  tab <- tibble::as_tibble(unclass(x))
  cat(sprintf("\n<spatial_correlogram> %d point(s), Moran's I by distance class\n",
              attr(x, "n_points")))
  print(tibble::tibble(from_km = tab$from_km, to_km = tab$to_km, n_pairs = tab$n_pairs,
                       moran_i = round(tab$moran_i, 3)), n = Inf)
  cat(sprintf("  With no spatial structure, about %.4f. Where it falls to that, points so far\n",
              tab$expected[1]))
  cat("  apart are as unlike as any two: the scale blocks should exceed.\n")
  invisible(x)
}

#' Plot a spatial correlogram.
#'
#' @param x   A `spatial_correlogram`, from [spatial_correlogram()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @export
plot.spatial_correlogram <- function(x, ...) {
  tab <- tibble::as_tibble(unclass(x))
  mid <- (tab$from_km + tab$to_km) / 2
  graphics::plot(mid, tab$moran_i, type = "b", pch = 16, xlab = "distance (km, class midpoint)",
                 ylab = "Moran's I", main = "Spatial correlogram",
                 ylim = range(c(0, tab$moran_i), na.rm = TRUE))
  graphics::abline(h = tab$expected[1], lty = 2)
  invisible(x)
}
