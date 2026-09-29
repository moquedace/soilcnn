# Resampling plans
#
# A fold constructor answers ONE question: which rows train, and which rows
# score, on each repetition. It answers nothing about scaling, tensors or
# models -- those read the plan and are unchanged by which plan they got.
#
# Every constructor takes the patch store's `meta` and returns a `fold_plan`,
# so the call site reads the same whichever one is used:
#
#   plan <- holdout(meta)                       # the fixed split from 01
#   plan <- random_folds(meta, k = 5)           # k-fold, ignores geography
#   plan <- spatial_folds(meta, k = 5, block_size = 50000)
#   plan <- region_folds(meta, group = meta$biome)
#
# WHY A PLAN OBJECT AND NOT A BARE LIST
# A bare list of indices cannot say how it was built, and a run whose folds
# cannot be described is a run that cannot be defended. The plan carries its
# method, its parameters and its per-row assignment, all of which get written
# next to the results.
#
# ONE CUT POINT, ONE CRITERION
# The plan carves the test set AND the folds, and it carves both the SAME way.
# A spatial validation sitting next to a random test set puts two numbers in
# one table that cannot be compared: measured on this project's own data, the
# random test came out 0.042 CCC EASIER than the spatial validation, while
# carrying a name that suggests it is the stricter of the two.
#
# So every constructor takes the same two numbers -- how many folds, and how
# much goes to test -- and the criterion (blocks, groups, at random) decides
# how both cuts are made. There is no half-spatial plan.
#
# THE TEST SET IS NOT RESAMPLED
# Once carved, the test rows stay test in EVERY fold and are never trained on.
# A test set that moves between folds has been seen by some model in the
# ensemble, and then it is not a test set -- it is a third validation set with
# a misleading name.
#
# NOTHING UPSTREAM DECIDES ROLES
# The patch store carries no role column. Stage 01 writes points, coordinates
# and values; who is test and who trains is answered here, in seconds, as many
# times as you like. That is what stops a change of split strategy from costing
# a re-extraction.

# -- small helpers ------------------------------------------------------------

`%||%` <- function(a, b) if (is.null(a)) b else a

# Run an expression under a given seed WITHOUT disturbing the caller's stream.
#
# Fold construction must not consume draws from the stream that initialises
# model weights: otherwise changing k would silently change every model's
# initialisation too, and the comparison between plans would be confounded.
with_local_seed <- function(seed, expr) {
  if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
    old <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    on.exit(assign(".Random.seed", old, envir = .GlobalEnv), add = TRUE)
  } else {
    on.exit(suppressWarnings(rm(".Random.seed", envir = .GlobalEnv)), add = TRUE)
  }
  set.seed(seed)
  force(expr)
}

# Draw WHOLE groups, in random order, until `frac` of the POINTS is reached.
#
# The one primitive behind every cut in this file. Drawing a fraction of GROUPS
# instead would miss the point target badly whenever group sizes are uneven,
# and they always are -- one spatial block can hold 5% of the data.
.draw_groups_for_frac <- function(group, frac, seed) {
  if (frac <= 0) return(integer(0))
  by_g    <- split(seq_along(group), group)
  order_g <- with_local_seed(seed, sample(names(by_g)))
  target  <- ceiling(frac * length(group))
  out     <- integer(0)
  for (g in order_g) {
    out <- c(out, by_g[[g]])
    if (length(out) >= target) break
  }
  sort(out)
}

# Carve the test set out of `group`, by the same criterion the folds will use.
#
# `test_ids` takes precedence over `test_frac`: a test set that is re-drawn on
# every run is not a test set. The example scripts freeze it to a file on the
# first run and pass it back afterwards.
.carve_test <- function(meta, group, test_frac, test_ids, seed) {
  if (!is.null(test_ids)) {
    if (!"sample_id" %in% names(meta)) {
      stop("meta needs sample_id to apply a frozen test set.", call. = FALSE)
    }
    pos <- match(as.character(test_ids), as.character(meta$sample_id))
    if (anyNA(pos)) {
      stop(sum(is.na(pos)), " frozen test sample_id(s) are not in meta -- the ",
           "frozen split and this dataset are not the same data.", call. = FALSE)
    }
    return(sort(pos))
  }
  if (is.null(test_frac) || test_frac <= 0) return(integer(0))
  if (test_frac >= 1) {
    stop("test_frac must be below 1 -- something has to be left to train on.",
         call. = FALSE)
  }
  # A different stream from the fold draw: changing k must not reshuffle the
  # test set, or two plans could never be compared on the same held-out data.
  .draw_groups_for_frac(group, test_frac, seed + 1000L)
}

# Everything after the test is carved: group -> k folds over what remains.
.plan_from_groups <- function(meta, group, k, test_frac, test_ids, seed,
                              method, params, extra = NULL) {
  n    <- length(group)
  test   <- .carve_test(meta, group, test_frac, test_ids, seed)
  pool   <- setdiff(seq_len(n), test)
  g_pool <- group[pool]

  # k = NULL means leave-one-group-out, and it can only be counted AFTER the
  # test is carved -- the groups that went to test are not available to be
  # folds. Resolved FIRST, because every check below compares against it.
  if (is.null(k)) k <- dplyr::n_distinct(g_pool)
  k <- as.integer(k)
  if (k < 2L) {
    stop("Only ", k, " group(s) left after the test set -- not enough to fold.",
         call. = FALSE)
  }
  if (length(pool) < k) {
    stop("Only ", length(pool), " row(s) left after the test set, for k = ", k,
         " folds.", call. = FALSE)
  }
  ug <- with_local_seed(seed, sample(unique(g_pool)))
  if (length(ug) < k) {
    stop("Only ", length(ug), " group(s) available for k = ", k, " folds -- ",
         "the grouping is too coarse for this many folds.", call. = FALSE)
  }
  sizes      <- as.integer(table(factor(g_pool, levels = ug))[ug])
  assignment <- .deal_groups(sizes, k)[match(g_pool, ug)]

  asg <- tibble::tibble(sample_id = meta$sample_id[pool], fold = assignment)
  if (!is.null(extra)) asg[[names(extra)]] <- extra[[1]][pool]

  .new_fold_plan(
    .folds_from_assignment(assignment, pool, test, k),
    method,
    c(params, list(k = k, test_frac = test_frac, n_test = length(test),
                   seed = seed)),
    meta, assignment = asg
  )
}

# Deal groups into k folds: largest group first, into whichever fold is
# currently smallest.
#
# WHY GREEDY AND NOT ROUND-ROBIN: spatial blocks and regions differ wildly in
# how many points they hold -- one block can carry 5% of the data. Round-robin
# would hand that block to an arbitrary fold and leave the folds badly
# unbalanced. Balance matters because each fold's metric is compared against
# the others', and a fold holding a quarter of its neighbour's points has a
# much noisier metric.
.deal_groups <- function(sizes, k) {
  ord   <- order(sizes, decreasing = TRUE)
  load  <- numeric(k)
  assig <- integer(length(sizes))
  for (g in ord) {
    j        <- which.min(load)
    assig[g] <- j
    load[j]  <- load[j] + sizes[g]
  }
  assig
}

# Turn a per-row fold assignment into the train/validation(/test) index lists.
.folds_from_assignment <- function(assignment, pool, test, k) {
  folds <- vector("list", k)
  for (j in seq_len(k)) {
    val <- pool[assignment == j]
    trn <- pool[assignment != j]
    if (length(val) == 0L || length(trn) == 0L) {
      stop("Fold ", j, " came out empty (", length(trn), " train, ",
           length(val), " validation). Use a smaller k, or larger blocks.",
           call. = FALSE)
    }
    idx <- list(train = trn, validation = val)
    if (length(test) > 0L) idx$test <- test
    folds[[j]] <- idx
  }
  folds
}

.new_fold_plan <- function(folds, method, params, meta, assignment = NULL) {
  structure(
    list(folds = folds, method = method, params = params,
         n_folds = length(folds), n_rows = nrow(meta),
         assignment = assignment),
    class = "fold_plan"
  )
}

# -- 1. holdout: the fixed split written by stage 01 --------------------------

#' A single random split: train / validation / test, one fold.
#'
#' The cheapest plan, and the right one when the rows really are independent.
#' On spatially clustered data it is the optimistic one -- and note that it is
#' random on BOTH cuts, which is what keeps its two numbers comparable with
#' each other. For a spatial question, use spatial_folds(): a spatial
#' validation beside a random test is the mismatch this file exists to prevent.
#'
#' @param meta            Point table; needs sample_id.
#' @param validation_frac Fraction of the non-test rows used to score.
#' @param test_frac       Fraction held out entirely.
#' @param test_ids        Optional frozen test sample_ids (see .carve_test).
#' @param seed            Draw seed.
# ── rows that must not be separated ───────────────────────────────────────────
#
# A ROW IS NOT ALWAYS AN INDEPENDENT OBSERVATION.
#
# In 3-D soil mapping one profile yields several rows -- 0-5, 5-15, 15-30 cm --
# at IDENTICAL coordinates, from the same pit, described by the same surveyor
# on the same day. Split those across training and validation and the model is
# scored on a depth of a profile it already learned: the covariates are
# byte-identical and the target is autocorrelated down the column. Wang et al.
# (2025, Geoderma 453:117131) name this and show the optimism it produces;
# leave-PROFILE-out is the fix.
#
# This framework was open to it. spatial_folds() and region_folds() happen to
# be safe -- rows at the same coordinates land in the same block -- but
# holdout() and random_folds() treated every row as its own unit, which is the
# exact failure the paper describes.
#
# WHY THE DEFAULT IS "auto" AND NOT NULL.
#
# Keeping a profile together is never wrong: with one row per profile it
# changes nothing at all, and with several it is the only correct answer.
# Failing to do it is silently wrong. A default that is safe in both cases and
# costs nothing in the common one does not deserve to be opt-in -- so the
# grouping is applied automatically when it MATTERS (the column exists and has
# duplicates), and the plan says so rather than doing it quietly.
#
# @param meta  Point table.
# @param group "auto", NULL/"row" for one group per row, a column name, or a
#   vector of group labels with one entry per row.
# @return character vector of group labels, with attr "note" describing it.
#' @noRd
.resolve_row_group <- function(meta, group = "auto") {
  n <- nrow(meta)
  rows_as_groups <- function(note) {
    g <- as.character(seq_len(n))
    attr(g, "note") <- note
    attr(g, "grouped") <- FALSE
    g
  }

  if (is.null(group) || identical(group, "row")) {
    return(rows_as_groups("every row is its own unit"))
  }

  if (identical(group, "auto")) {
    if (!"profile_id" %in% names(meta)) {
      return(rows_as_groups("every row is its own unit (no profile_id column)"))
    }
    pid <- as.character(meta$profile_id)
    if (anyNA(pid)) {
      stop("profile_id has ", sum(is.na(pid)), " missing value(s). A row that ",
           "cannot say which profile it belongs to cannot be kept with it.",
           call. = FALSE)
    }
    if (length(unique(pid)) == n) {
      return(rows_as_groups("one row per profile -- grouping changes nothing"))
    }
    g <- pid
    attr(g, "note") <- sprintf(
      paste0("grouped by profile_id: %d rows in %d profiles (up to %d rows ",
             "share one profile). Rows of one profile stay in the same fold ",
             "-- see Wang et al. 2025, Geoderma"),
      n, length(unique(pid)), max(table(pid)))
    attr(g, "grouped") <- TRUE
    return(g)
  }

  if (is.character(group) && length(group) == 1L && group %in% names(meta)) {
    g <- as.character(meta[[group]])
    if (anyNA(g)) {
      stop("Grouping column '", group, "' has ", sum(is.na(g)),
           " missing value(s).", call. = FALSE)
    }
    attr(g, "note") <- sprintf("grouped by %s: %d rows in %d groups",
                               group, n, length(unique(g)))
    attr(g, "grouped") <- length(unique(g)) < n
    return(g)
  }

  if (length(group) != n) {
    stop("`group` must be \"auto\", NULL, a column name of meta, or a vector ",
         "with one entry per row (got length ", length(group), " for ", n,
         " rows).", call. = FALSE)
  }
  g <- as.character(group)
  if (anyNA(g)) stop("`group` has missing values.", call. = FALSE)
  attr(g, "note") <- sprintf("grouped by the vector given: %d rows in %d groups",
                             n, length(unique(g)))
  attr(g, "grouped") <- length(unique(g)) < n
  g
}

#' A single train/validation/test split of a point table.
#'
#' The fold constructor behind holdout_cv(). Whole groups are drawn, never
#' single rows (see `group`), so the rows of one profile never straddle the
#' split.
#'
#' @param meta            Point table (the patch store's meta): needs
#'   sample_id, and profile_id for `group = "auto"` to have anything to do.
#' @param validation_frac Share of the non-test pool that validates.
#' @param test_frac       Share held out as the test set, drawn first.
#' @param test_ids        Sample ids that ARE the test set, in place of
#'   `test_frac`: a test set drawn again on every run is not a test set.
#' @param seed            Seed for this split only; the training seeds do not
#'   move it.
#' @param group           "auto" keeps the rows of one profile_id together when
#'   the table repeats profiles; NULL or "row" makes each row its own unit; a
#'   column name, or a vector with one label per row, names the groups.
#' @return A `fold_plan` with one fold.
#' @export
holdout <- function(meta, validation_frac = 0.15, test_frac = 0.15,
                    test_ids = NULL, seed = 42L, group = "auto") {
  if (!"sample_id" %in% names(meta)) {
    stop("meta needs a sample_id column.", call. = FALSE)
  }
  n     <- nrow(meta)
  group <- .resolve_row_group(meta, group)
  grp_note <- attr(group, "note")
  if (isTRUE(attr(group, "grouped"))) message("holdout: ", grp_note)

  test  <- .carve_test(meta, group, test_frac, test_ids, seed)
  pool  <- setdiff(seq_len(n), test)
  if (length(pool) < 2L) stop("Nothing left after the test set.", call. = FALSE)

  # The validation cut is drawn over GROUPS, not rows, for the same reason the
  # test cut is: half a profile in training and half in validation is the
  # leakage this argument exists to prevent.
  g_pool <- unique(group[pool])
  val_g  <- with_local_seed(seed,
    g_pool[sample(length(g_pool), max(1L, floor(validation_frac * length(g_pool))))])
  val <- sort(pool[group[pool] %in% val_g])
  trn <- setdiff(pool, val)
  if (length(trn) == 0L) {
    stop("validation_frac left no training rows.", call. = FALSE)
  }

  idx <- list(train = trn, validation = val)
  if (length(test) > 0L) idx$test <- test

  .new_fold_plan(list(idx), "holdout",
                 list(k = 1L, validation_frac = validation_frac,
                      test_frac = test_frac, n_test = length(test),
                      seed = seed, grouping = grp_note),
                 meta,
                 assignment = tibble::tibble(sample_id = meta$sample_id[pool],
                                             fold = 1L))
}

# -- 2. random k-fold ---------------------------------------------------------

#' Random k-fold over the train+validation pool.
#'
#' Ignores geography by construction. On spatially clustered data that makes
#' the validation metric optimistic -- the pipeline check measures it: a random
#' split put ~27% of validation in the SAME 250 m pixel as a training point.
#' Offered anyway, for two reasons: it is the right answer when the rows really
#' are independent, and running it against spatial_folds() on the same data is
#' the cleanest way to MEASURE how much geography is worth here.
#'
#' @param meta Patch store meta.
#' @param k    Number of folds.
#' @param seed Seed for the partition only (see with_local_seed): a fixed plan
#'   reproduces even when the training seeds change.
#' @inheritParams holdout
#' @return A `fold_plan` with `k` folds.
#' @export
random_folds <- function(meta, k = 5L, test_frac = 0, test_ids = NULL,
                        seed = 42L, group = "auto") {
  k <- as.integer(k)
  if (k < 2L) stop("k must be at least 2.", call. = FALSE)
  if (!"sample_id" %in% names(meta)) {
    stop("meta needs a sample_id column.", call. = FALSE)
  }
  g <- .resolve_row_group(meta, group)
  if (isTRUE(attr(g, "grouped"))) message("random_folds: ", attr(g, "note"))
  .plan_from_groups(meta, g, k, test_frac, test_ids, seed,
                    method = "random_folds",
                    params = list(grouping = attr(g, "note")))
}

# -- 3. spatial block k-fold --------------------------------------------------

#' Spatial block k-fold: whole blocks of ground go to the same fold.
#'
#' Points close enough to share raster cells -- and therefore share patch
#' content -- land in the SAME fold, so validation measures prediction at
#' unvisited places instead of interpolation between neighbours. The gap
#' between this and random_folds() on the same data IS the spatial optimism.
#'
#' @param meta       Patch store meta; needs x and y in the raster's CRS units.
#' @param k          Number of folds.
#' @param block_size Block side, in the SAME units as x/y (metres for a
#'   projected CRS, degrees for lon/lat). It should comfortably exceed the
#'   range over which the target autocorrelates, and at absolute minimum exceed
#'   the widest patch -- below that the folds still share patch pixels and the
#'   split only looks spatial. When NULL a size is derived from the extent so
#'   that roughly `blocks_per_fold * k` blocks result; that is a starting point
#'   for inspection, NOT a substitute for a variogram, and it is flagged in the
#'   plan's params as block_size_auto.
#' @param buffer     Exclusion radius, in the units of x/y: training points
#'   within this distance of a validation point are dropped from the fold.
#'   NULL (default) means no buffer -- and blocking ALONE does not guarantee
#'   separation, because a boundary can fall inside a cluster. Distance here is
#'   EUCLIDEAN, so to guarantee that no training patch shares a pixel with a
#'   validation patch, max(window) * resolution is exact under the default
#'   chebyshev metric. Under euclidean it takes sqrt(2) more, because patch
#'   overlap is a square condition and the diagonal escapes a circle.
#' @param buffer_metric "chebyshev" (default, the patch geometry) or
#'   "euclidean". See apply_buffer().
#' @param seed       Seed for dealing blocks to folds.
#' @param blocks_per_fold Target blocks per fold when `block_size` is NULL.
# ── choosing a block size, by measurement ─────────────────────────────────────
#
# THE TENSION. Bigger blocks separate better: a fold whose blocks are 2 degrees
# wide holds out whole landscapes, and a model cannot reach across the gap by
# memorising a neighbour. But a block is indivisible -- every point in it goes
# to the same fold -- so one oversized block unbalances the whole plan.
#
# WHY THIS CANNOT BE A CONSTANT. The right size depends on how the points are
# SPREAD, not on the method, and the same number is right and wrong for two
# datasets of the same size. It is also right and wrong for the same dataset at
# two sampling densities: block-subsampling keeps WHOLE blocks, so a 10% draw
# has a tenth of the blocks but blocks of the SAME size -- and a block that was
# 3% of the full data becomes 34% of the subsample.
#
# That is not hypothetical. It is exactly what happened here: block_size = 2
# was measured on the full point set (largest block 4.4%) and carried over to a
# 10% dev draw, where the largest block holds a THIRD of everything.
#
# So: measure, then choose. These two functions are what the choice is made
# from, and both are cheap enough to call before every run.

#' How lumpy is a blocking, at a given size?
#'
#' @param meta        Point table with x and y.
#' @param block_sizes Sizes to evaluate, in the units of x/y.
#' @return A tibble: one row per size, with the block count and the share of
#'   points held by the largest block.
#' @noRd
block_share <- function(meta, block_sizes = c(0.25, 0.5, 1, 2, 3)) {
  x <- as.numeric(meta$x); y <- as.numeric(meta$y)
  n <- length(x)
  if (n == 0L) stop("block_share(): there are no points to block.", call. = FALSE)
  rows <- lapply(block_sizes, function(bs) {
    blk <- paste(floor((x - min(x)) / bs), floor((y - min(y)) / bs), sep = "_")
    tab <- table(blk)
    tibble::tibble(block_size    = bs,
                   n_blocks      = length(tab),
                   largest_n     = as.integer(max(tab)),
                   largest_share = as.numeric(max(tab)) / n,
                   median_n      = as.numeric(stats::median(tab)))
  })
  dplyr::bind_rows(rows)
}

#' The largest block size this point set can afford.
#'
#' Largest, not smallest: separation is the thing being bought, so take as much
#' of it as the balance constraint allows.
#'
#' @param meta        Point table with x and y.
#' @param k           Number of folds the blocks must be dealt into.
#' @param max_share   Largest share of the points one block may hold. The
#'   default 0.10 is half of what a fold gets at k = 5 -- enough that no single
#'   block decides a fold.
#' @param min_blocks_per_fold At least this many blocks per fold, so the deal
#'   has something to balance WITH.
#' @param candidates  Sizes to consider, ascending.
#' @return The chosen size, with the evaluation table attached as "table".
#' @noRd
suggest_block_size <- function(meta, k = 5L, max_share = 0.10,
                               min_blocks_per_fold = 10L,
                               candidates = c(0.1, 0.25, 0.5, 1, 2, 3, 5)) {
  tab <- block_share(meta, sort(candidates))
  ok  <- tab$largest_share <= max_share &
         tab$n_blocks >= min_blocks_per_fold * as.integer(k)

  if (!any(ok)) {
    # Nothing qualifies: say so with the evidence rather than returning a
    # number that meets no constraint and looks deliberate.
    warning("No candidate block size keeps the largest block under ",
            round(100 * max_share), "% of the points with at least ",
            min_blocks_per_fold, " blocks per fold. The points are too ",
            "clustered for blocked folds at these sizes -- consider more ",
            "folds, region_folds() on a grouping that exists, or accepting ",
            "the imbalance deliberately.", call. = FALSE)
    chosen <- tab$block_size[which.min(tab$largest_share)]
  } else {
    chosen <- max(tab$block_size[ok])
  }
  attr(chosen, "table") <- tab
  attr(chosen, "max_share") <- max_share
  chosen
}

#' Print what suggest_block_size() measured.
#' @noRd
print_block_choice <- function(chosen) {
  tab <- attr(chosen, "table")
  ms  <- attr(chosen, "max_share")
  # Pulled out of the mutate() on purpose: `chosen` is both this function's
  # argument and the column being created, and relying on dplyr to resolve that
  # the way it happens to today is a bug waiting for a dplyr release.
  pick <- as.numeric(chosen)
  cat("\n-- Block size, measured on these points --\n")
  print_wide(dplyr::mutate(
    tab,
    largest_share = sprintf("%.1f%%", 100 * largest_share),
    chosen        = ifelse(block_size == pick, "  <--", "")), n = Inf)
  cat(sprintf("\n  chosen: %g  (largest block <= %.0f%% of the points)\n",
              as.numeric(chosen), 100 * ms))
  invisible(chosen)
}

#' Spatially blocked folds of a point table.
#'
#' The fold constructor behind spatial_cv(): the points are binned into square
#' blocks, whole blocks go to one fold, and the buffer then drops the training
#' points too close to a validation or test point.
#'
#' @param meta            Point table: needs x, y and sample_id.
#' @param k               Number of folds.
#' @param block_size      Side of a block, in the units of x/y. NULL sizes it
#'   from the extent, for `blocks_per_fold` blocks per fold.
#' @param buffer          Distance within which a training point is dropped
#'   from a fold, in the units of x/y; NULL for none.
#' @param buffer_metric   "chebyshev" (the default) or "euclidean". Chebyshev
#'   is exact for square patches: two patches share a pixel when their
#'   centres are within the window in both axes.
#' @param blocks_per_fold Blocks per fold, for a `block_size` of NULL.
#' @inheritParams holdout
#' @return A `fold_plan` with `k` folds.
#' @export
spatial_folds <- function(meta, k = 5L, test_frac = 0, block_size = NULL,
                          buffer = NULL,
                          buffer_metric = c("chebyshev", "euclidean"),
                          test_ids = NULL, seed = 42L,
                          blocks_per_fold = 10L) {
  buffer_metric <- match.arg(buffer_metric)
  k <- as.integer(k)
  if (k < 2L) stop("k must be at least 2.", call. = FALSE)
  if (!all(c("x", "y") %in% names(meta))) {
    stop("meta needs x and y columns for spatial folds.", call. = FALSE)
  }
  if (!"sample_id" %in% names(meta)) {
    stop("meta needs a sample_id column.", call. = FALSE)
  }
  x <- as.numeric(meta$x)
  y <- as.numeric(meta$y)
  if (anyNA(x) || anyNA(y)) {
    stop("x/y are NA for ", sum(is.na(x) | is.na(y)), " row(s).", call. = FALSE)
  }

  auto <- is.null(block_size)
  if (auto) {
    area <- (max(x) - min(x)) * (max(y) - min(y))
    block_size <- sqrt(max(area, .Machine$double.eps) /
                       max(1L, blocks_per_fold * k))
  }
  if (!is.finite(block_size) || block_size <= 0) {
    stop("block_size must be a positive finite number.", call. = FALSE)
  }

  blk <- paste(floor((x - min(x)) / block_size),
               floor((y - min(y)) / block_size), sep = "_")

  # A block is indivisible: every point in it goes to the same fold. So one
  # oversized block does not merely unbalance the plan, it DECIDES a fold --
  # and the fold it decides is then scored on whatever that one landscape
  # happens to be. Warned rather than refused, because there are point sets
  # where this is simply true and known; see suggest_block_size() to pick a
  # size from the data instead of carrying one over from another run.
  .share <- max(table(blk)) / length(blk)
  if (.share > 1 / k) {
    warning(sprintf(
      paste0("The largest block holds %.0f%% of the points, more than the ",
             "%.0f%% one fold gets at k = %d. A block cannot be split, so ",
             "that block alone decides a fold.\n  block_size = %g gives %d ",
             "block(s); suggest_block_size(meta, k = %d) measures what this ",
             "point set can afford."),
      100 * .share, 100 / k, k, block_size, length(unique(blk)), k),
      call. = FALSE)
  }

  # The test comes out of the SAME blocks the folds will use, so no test point
  # sits inside a block that also trains.
  plan <- .plan_from_groups(
    meta, blk, k, test_frac, test_ids, seed,
    method = "spatial_folds",
    params = list(block_size = block_size, block_size_auto = auto,
                  n_blocks = length(unique(blk))),
    extra  = list(block = blk)
  )
  # Blocking chooses where the cut falls; the buffer is what makes the cut
  # mean something. See apply_buffer() for why one without the other is not
  # enough -- it is measured there, not assumed.
  apply_buffer(plan, meta, buffer, metric = buffer_metric)
}

# -- 4. grouped folds (leave-region-out) --------------------------------------

#' Fold by a grouping column: no group is split across folds.
#'
#' For when the unit that must not leak is named rather than geometric -- a
#' survey campaign, a country, a laboratory, a soil map unit. With
#' k = number of groups this is leave-one-group-out.
#'
#' @param meta  Patch store meta.
#' @param group Group labels, one per row of `meta`.
#' @param k     Number of folds; defaults to one per group.
#' @inheritParams holdout
#' @return A `fold_plan`.
#' @export
region_folds <- function(meta, group, k = NULL, test_frac = 0,
                        test_ids = NULL, seed = 42L) {
  if (length(group) != nrow(meta)) {
    stop("group has ", length(group), " values but meta has ", nrow(meta),
         " rows.", call. = FALSE)
  }
  g <- as.character(group)
  if (anyNA(g)) {
    stop(sum(is.na(g)), " row(s) have no group label -- decide what they ",
         "belong to instead of letting them fall into an NA group.",
         call. = FALSE)
  }
  # k stays NULL when the caller wants leave-one-group-out: how many groups are
  # left is only known after the test set is carved.
  if (!is.null(k)) {
    k <- as.integer(k)
    if (k < 2L) stop("Need at least 2 groups (or k >= 2).", call. = FALSE)
  }

  .plan_from_groups(meta, g, k, test_frac, test_ids, seed,
                    method = "region_folds",
                    params = list(n_groups = dplyr::n_distinct(g)),
                    extra  = list(group = g))
}

# -- 5. buffer: the mechanism that actually separates folds -------------------
#
# A block boundary is drawn on the MAP, not on the data, so it lands wherever
# it lands -- including straight through a cluster of points. When that
# happens, part of the cluster trains and part of it validates, metres apart,
# and the fold is spatial in name only. Measured on this project's own test
# fixture: 1 cut site in 12 was enough for a third of the validation points to
# have a training neighbour within 1 km.
#
# Blocking decides WHERE the cut falls. The buffer is what makes the cut mean
# something: every training point within `buffer` of a validation point is
# DROPPED from that fold.
#
# CHOOSING THE BUFFER -- AND THE GEOMETRY TRAP IN IT.
#
# Two patches of width w pixels share at least one pixel exactly when their
# centres satisfy |drow| <= w-1 AND |dcol| <= w-1. That is a CHEBYSHEV
# condition -- a square -- not a Euclidean one.
#
# `apply_buffer()` measures EUCLIDEAN distance, so `buffer = w * res` does NOT
# deliver the guarantee it looks like it delivers. The escape is the diagonal:
# two centres 14 rows AND 14 columns apart (w = 15) sit 14*sqrt(2) ~ 19.8 cells
# away -- comfortably outside a 15-cell circle -- and their patches still share
# the corner pixel. Measured on the real run: with buffer = 15 * res, 0.51% to
# 0.98% of validation points per fold still shared patch pixels with training,
# even though "same raster cell" was a clean 0%.
#
# So `metric = "chebyshev"` is the default: a SQUARE buffer, which is the
# geometry of the thing being excluded. With it, `buffer = max(window) * res`
# is exact, and no training point is discarded for being diagonally far but
# circularly near.
#
#     chebyshev  buffer >= max(window) * resolution            exact
#     euclidean  buffer >= max(window) * resolution * sqrt(2)  41% wider,
#                                                              same guarantee,
#                                                              more training
#                                                              data thrown away
#
# `metric = "euclidean"` stays available for a target whose correlation really
# is isotropic in distance rather than tied to the patch grid.
#
# Above that floor the buffer becomes a statement about how far the target
# autocorrelates, which is a question about the soil, not about code.
#
# BLOCKING AND BUFFERING ARE NOT ALTERNATIVES. Buffering a RANDOM k-fold on
# clustered data removes every training point -- every cluster holds both
# training and validation rows, so every training row has a validation
# neighbour, so the buffer takes all of them. That is not a quirk of one
# dataset: it is what random CV on clustered data looks like once the leakage
# is taken away. Blocking is what makes buffering affordable, by putting whole
# clusters on one side of the line.
#
# THE BUFFER COSTS TRAINING DATA, and the plan says how much. Dropping 40% of
# the training set to buy honest validation may or may not be the right trade,
# but it has to be a visible one -- which is why the count is stored in the
# plan and printed, never silently absorbed.

#' Drop training points that sit within `buffer` of a validation point.
#'
#' Works on ANY plan, not only spatial ones: buffering a random k-fold is a
#' perfectly good way to find out what the leakage was worth.
#'
#' @param plan   A fold_plan.
#' @param meta   Patch store meta, with x and y (same units as `buffer`).
#' @param buffer Exclusion radius in the units of x/y. NULL or 0 returns the
#'   plan unchanged.
#' @return The plan, with buffered training sets and a `buffer_dropped` tibble.
#' @noRd
apply_buffer <- function(plan, meta, buffer,
                         metric = c("chebyshev", "euclidean"),
                         protect = c("validation", "test")) {
  stopifnot(inherits(plan, "fold_plan"))
  metric  <- match.arg(metric)
  protect <- match.arg(protect, c("validation", "test"), several.ok = TRUE)
  if (is.null(buffer) || buffer <= 0) return(plan)
  if (!all(c("x", "y") %in% names(meta))) {
    stop("meta needs x and y columns to apply a buffer.", call. = FALSE)
  }
  x <- as.numeric(meta$x)
  y <- as.numeric(meta$y)

  # ── THE TEST SET WAS NEVER BUFFERED ────────────────────────────────────────
  #
  # This function protected the VALIDATION set and nothing else. So the fold
  # plan could report "0% leakage" -- truthfully, about validation -- while
  # every training point next to a test block kept its place, and its 15x15
  # patch kept overlapping test patches pixel for pixel.
  #
  # The number that was protected is the one used to CHOOSE. The number that
  # was not is the one that goes in the paper.
  #
  # Validation is buffered against test too. It is the weaker of the two links
  # -- the model never fits validation rows, it only decides WHEN TO STOP on
  # them -- but early stopping is a decision made on data, and a stopping epoch
  # chosen on rows that overlap the test set is a small read of the test set.
  # It is cheap to close: the test is carved as whole blocks, so the points
  # within one patch span of it are a thin rim.
  #
  # Reported by cause, because "5.1% dropped" hides which promise it paid for.
  dropped <- vector("list", length(plan$folds))
  for (j in seq_along(plan$folds)) {
    f      <- plan$folds[[j]]
    n_before <- length(f$train)
    has_test <- length(f$test) > 0L

    near_val  <- if ("validation" %in% protect && length(f$validation) > 0L) {
      .near_any(x[f$train], y[f$train], x[f$validation], y[f$validation],
                buffer, metric)
    } else rep(FALSE, n_before)

    near_test <- if ("test" %in% protect && has_test) {
      .near_any(x[f$train], y[f$train], x[f$test], y[f$test], buffer, metric)
    } else rep(FALSE, n_before)

    plan$folds[[j]]$train <- f$train[!(near_val | near_test)]
    if (length(plan$folds[[j]]$train) == 0L) {
      stop("Fold ", j, ": the buffer removed every training point. The buffer ",
           "is large relative to the spacing of this data.", call. = FALSE)
    }

    n_val_before <- length(f$validation)
    n_val_drop   <- 0L
    if ("test" %in% protect && has_test && n_val_before > 0L) {
      nv <- .near_any(x[f$validation], y[f$validation], x[f$test], y[f$test],
                      buffer, metric)
      plan$folds[[j]]$validation <- f$validation[!nv]
      n_val_drop <- sum(nv)
      if (length(plan$folds[[j]]$validation) == 0L) {
        stop("Fold ", j, ": the buffer removed every validation point. Early ",
             "stopping has nothing left to watch.", call. = FALSE)
      }
    }

    n_drop <- sum(near_val | near_test)
    dropped[[j]] <- tibble::tibble(
      fold = j, n_train_before = n_before, n_dropped = n_drop,
      pct_dropped = round(100 * n_drop / n_before, 2),
      # Causes overlap -- a point can be near both -- so these do not have to
      # add up to n_dropped, and saying so here stops that reading as a bug.
      n_near_validation = sum(near_val), n_near_test = sum(near_test),
      n_validation_before = n_val_before, n_validation_dropped = n_val_drop)
  }

  plan$params$buffer         <- buffer
  plan$params$buffer_metric  <- metric
  plan$params$buffer_protect <- paste(protect, collapse = "+")
  plan$buffer_dropped        <- dplyr::bind_rows(dropped)
  plan
}

# For each (tx, ty), is there any (vx, vy) within `buffer`?
#
# Bucketed by the buffer size, testing the 9 neighbouring buckets, so the cost
# is linear instead of the n_train x n_validation product -- on the real data
# that product is ~2e8 per fold, the difference between a second and minutes.
# Distances inside the candidate set are exact: the buckets narrow the search,
# they never decide the answer.
.near_any <- function(tx, ty, vx, vy, buffer,
                      metric = c("chebyshev", "euclidean")) {
  metric <- match.arg(metric)
  if (length(tx) == 0L || length(vx) == 0L) return(rep(FALSE, length(tx)))
  b2 <- buffer^2
  ox <- min(c(tx, vx)); oy <- min(c(ty, vy))

  tb  <- paste(floor((tx - ox) / buffer), floor((ty - oy) / buffer), sep = "_")
  vbx <- floor((vx - ox) / buffer)
  vby <- floor((vy - oy) / buffer)
  v_by_bucket <- split(seq_along(vx), paste(vbx, vby, sep = "_"))

  hit <- logical(length(tx))
  for (bk in unique(tb)) {
    rows <- which(tb == bk)
    rc   <- as.integer(strsplit(bk, "_", fixed = TRUE)[[1]])
    cand <- unlist(v_by_bucket[paste(rep(rc[1] + (-1:1), each = 3L),
                                     rep(rc[2] + (-1:1), times = 3L),
                                     sep = "_")],
                   use.names = FALSE)
    if (length(cand) == 0L) next
    cx <- vx[cand]; cy <- vy[cand]
    hit[rows] <- if (metric == "chebyshev") {
      vapply(rows, function(i) {
        any(pmax(abs(cx - tx[i]), abs(cy - ty[i])) <= buffer)
      }, logical(1))
    } else {
      vapply(rows, function(i) {
        any((cx - tx[i])^2 + (cy - ty[i])^2 <= b2)
      }, logical(1))
    }
  }
  hit
}

#' How much does each fold still leak?
#'
#' Reports, per fold, the same quantities the pipeline check reports for the
#' fixed split: the share of validation points sharing a raster cell with a
#' training point, and sharing patch pixels at each window. Deliberately the
#' SAME criterion, so a number here can be compared with a number there rather
#' than being a second, incompatible notion of leakage.
#'
#' @param plan      A fold_plan.
#' @param meta      Patch store meta with x/y.
#' @param cell_size Raster resolution, in the units of x/y.
#' @param windows   Window sizes to report overlap for.
#' @return A tibble with a `fold` column: the overlap shares per role and
#'   window, fold by fold.
#' @export
fold_leakage_report <- function(plan, meta, cell_size, windows = c(3L, 9L, 15L)) {
  stopifnot(inherits(plan, "fold_plan"))
  col_id <- floor(as.numeric(meta$x) / cell_size)
  row_id <- floor(as.numeric(meta$y) / cell_size)

  out <- lapply(seq_along(plan$folds), function(j) {
    f    <- plan$folds[[j]]
    keep <- c(f$train, f$validation)
    role <- rep(c("train", "validation"),
                c(length(f$train), length(f$validation)))
    spatial_overlap_report(row_id[keep], col_id[keep], role,
                           windows = windows, reference = "train") %>%
      dplyr::mutate(fold = j, .before = 1)
  })
  dplyr::bind_rows(out)
}

# -- development subsampling --------------------------------------------------
#
# A pipeline you can run end to end in minutes is a pipeline you can fix
# without fear. Subsampling exists for that, and for nothing else: the number
# it produces is only ever comparable with another number from the same
# subsample.
#
# WHY WHOLE BLOCKS AND NOT RANDOM POINTS. Dropping 90% of points at random
# THINS the spatial clusters. Leakage falls, the buffer discards fewer points,
# and the folds come out looking cleaner than the data really is -- so a bug in
# the spatial logic hides behind a healthy-looking report. Keeping whole blocks
# preserves local density, which is the very phenomenon the spatial machinery
# exists to control.
#
# It is also faster for a reason that has nothing to do with statistics:
# extraction reads raster strips covering the points, so points concentrated in
# fewer regions mean fewer strips.

#' Keep roughly `frac` of the points, by whole spatial blocks.
#'
#' @param x,y        Coordinates, in the units the block size is given in.
#' @param frac       Target fraction of POINTS (not of blocks) to keep.
#' @param block_size Block side, same units as x/y.
#' @param seed       Seed for the block draw; independent of every other
#'   stream (see with_local_seed).
#' @return Integer row positions to keep, with attributes describing what was
#'   drawn -- a subsample whose composition cannot be reported is a subsample
#'   whose results cannot be interpreted.
#' @noRd
block_subsample <- function(x, y, frac, block_size, seed = 42L) {
  stopifnot(length(x) == length(y), frac > 0, frac <= 1, block_size > 0)
  n <- length(x)
  if (frac >= 1) return(seq_len(n))

  blk <- paste(floor((x - min(x)) / block_size),
               floor((y - min(y)) / block_size), sep = "_")

  # Same primitive the test set is carved with -- one implementation, so the
  # subsample and the split cannot drift apart in how they treat a block.
  keep <- .draw_groups_for_frac(blk, frac, seed)
  used <- unique(blk[keep])

  structure(
    keep,
    n_total        = n,
    n_kept         = length(keep),
    frac_requested = frac,
    frac_actual    = length(keep) / n,
    blocks_total   = dplyr::n_distinct(blk),
    blocks_kept    = length(used),
    block_size     = block_size,
    seed           = seed
  )
}

#' One line describing a block_subsample() result, for logs and metadata.
#' @noRd
describe_subsample <- function(idx) {
  sprintf(
    "%s of %s points (%.1f%%, requested %.1f%%) from %s of %s blocks of %s",
    format(attr(idx, "n_kept"), big.mark = ","),
    format(attr(idx, "n_total"), big.mark = ","),
    100 * attr(idx, "frac_actual"), 100 * attr(idx, "frac_requested"),
    format(attr(idx, "blocks_kept"), big.mark = ","),
    format(attr(idx, "blocks_total"), big.mark = ","),
    format(attr(idx, "block_size"))
  )
}

# -- refitting after selection ------------------------------------------------

#' A single train/validation split for the FINAL fit, by the plan's own rules.
#'
#' After cross-validation has chosen a configuration, the final model is
#' refitted on everything except the test set. It still needs somewhere to stop
#' (early stopping), and that somewhere must be carved the SAME way the
#' validation was carved during selection -- a spatially-selected model whose
#' final fit stops on a randomly drawn validation set has changed the question
#' between the two stages.
#'
#' The test set is taken from the tuning plan verbatim, never redrawn: the
#' model that gets tested must not have trained on any row the selection
#' already held out.
#'
#' @param plan            The fold_plan used for tuning.
#' @param meta            The same point table.
#' @param validation_frac Share of the non-test rows used to stop training.
#' @param predpoints      kNNDM plans only: the prediction points, for a plan
#'   made before knndm_folds() kept them with the plan. NULL takes the plan's.
#' @return A one-fold `fold_plan`.
#' @noRd
refit_split <- function(plan, meta, validation_frac = 0.15, predpoints = NULL) {
  stopifnot(inherits(plan, "fold_plan"))
  test_pos <- plan$folds[[1]]$test
  test_ids <- if (length(test_pos)) meta$sample_id[test_pos] else NULL
  k        <- max(2L, as.integer(round(1 / validation_frac)))
  seed     <- plan$params$seed %||% 42L

  # One fold of a k-fold plan with k = 1/validation_frac IS a holdout carved by
  # that plan's criterion -- so there is one implementation, not two.
  sub_plan <- switch(
    plan$method,
    spatial_folds = spatial_folds(
      meta, k = k, test_ids = test_ids,
      block_size    = plan$params$block_size,
      buffer        = plan$params$buffer,
      buffer_metric = plan$params$buffer_metric %||% "chebyshev",
      seed = seed),
    region_folds  = .refit_region_plan(plan, meta, k, test_ids, seed),
    knndm_folds   = .refit_knndm_plan(plan, meta, k, test_ids, seed, predpoints),
    random_folds  = random_folds(meta, k = k, test_ids = test_ids, seed = seed),
    holdout       = holdout(meta, validation_frac = validation_frac,
                            test_ids = test_ids, seed = seed),
    stop("Unknown plan method: ", plan$method, call. = FALSE)
  )

  if (identical(plan$method, "holdout")) return(sub_plan)

  # WHICH OF THE k FOLDS VALIDATES. Blocks and random folds come out balanced
  # -- blocks dealt greedily, rows drawn at random -- and the first is taken,
  # as it always was. kNNDM builds its folds from clusters and lets one hold
  # up to half the points (CAST's maxp), and a region is as large as it is:
  # there the fold closest in size to validation_frac of the pool validates,
  # so the refit trains on the share it was asked to.
  pick <- 1L
  if (plan$method %in% c("knndm_folds", "region_folds")) {
    f1 <- sub_plan$folds[[1]]
    n_pool <- length(f1$train) + length(f1$validation)
    sizes  <- vapply(sub_plan$folds, function(f) length(f$validation), integer(1))
    pick   <- which.min(abs(sizes - validation_frac * n_pool))
  }
  idx <- sub_plan$folds[[pick]]
  .new_fold_plan(list(idx), paste0("refit_", plan$method),
                 c(sub_plan$params, list(refit_of = plan$method,
                                         validation_frac = validation_frac,
                                         refit_fold = pick)),
                 meta, assignment = sub_plan$assignment)
}

# A kNNDM plan's criterion is its prediction points: the refit runs kNNDM
# again, over the non-test rows, against the same points, in the same frame
# and with the same CAST options. A plan from before knndm_folds() kept them
# has only its projection in params; its points must be given, and its CRS is
# taken as 4326 -- the default, and the one thing such a plan did not record.
.refit_knndm_plan <- function(plan, meta, k, test_ids, seed, predpoints) {
  kn   <- plan$knndm
  args <- kn$args %||% list()
  if (!is.null(predpoints) && !is.null(kn$predpoints) &&
      !isTRUE(all.equal(as.data.frame(predpoints), as.data.frame(kn$predpoints),
                        check.attributes = FALSE))) {
    stop("This kNNDM plan carries its own prediction points, and they are not ",
         "the ones given: the refit's validation must be cut against the points ",
         "the tuning folds were. Leave predpoints out.", call. = FALSE)
  }
  pp <- predpoints %||% kn$predpoints
  if (is.null(pp) && is.null(args$modeldomain)) {
    stop("This kNNDM plan does not carry its prediction points: it was made ",
         "before knndm_folds() kept them with the plan (2026-09-29). The refit's ",
         "validation is cut against them, as the tuning folds were -- pass the ",
         "same points as predpoints = <a data frame with x and y>.", call. = FALSE)
  }
  project_to <- if (!is.null(kn)) {
    kn$project_to
  } else {
    pr <- plan$params$projection
    if (is.null(pr) || identical(pr, "none (already projected)")) NULL else pr
  }
  do.call(knndm_folds, c(list(meta = meta, k = k, predpoints = pp,
                              test_ids = test_ids, crs = kn$crs %||% 4326,
                              project_to = project_to, seed = seed), args))
}

# A region plan keeps each folded row's group in its assignment (region_folds()
# writes it there). The test rows get a group of their own, which the frozen
# test set then keeps out of every fold; k is capped at the groups there are.
.refit_region_plan <- function(plan, meta, k, test_ids, seed) {
  asg <- plan$assignment
  if (is.null(asg) || !"group" %in% names(asg)) {
    stop("This region plan does not carry its groups, so the refit cannot cut ",
         "whole regions out of it.", call. = FALSE)
  }
  pos <- match(as.character(asg$sample_id), as.character(meta$sample_id))
  if (anyNA(pos)) {
    stop("The region plan names ", sum(is.na(pos)), " sample_id(s) this store ",
         "does not hold.", call. = FALSE)
  }
  g <- rep(".test", nrow(meta))
  g[pos] <- as.character(asg$group)
  test_pos <- if (length(test_ids)) match(as.character(test_ids), as.character(meta$sample_id)) else integer(0)
  if (!setequal(which(g == ".test"), test_pos)) {
    stop("The region plan's groups do not cover every row outside its test set.",
         call. = FALSE)
  }
  region_folds(meta, group = g, k = min(k, dplyr::n_distinct(asg$group)),
               test_ids = test_ids, seed = seed)
}

# -- validation and reporting -------------------------------------------------

#' Check a plan's folds are well formed, and describe them.
#'
#' Run this before any training: a broken plan caught here costs a second,
#' caught later it costs the whole run. Checks that, within each fold, train
#' and validation are disjoint; that test (when present) appears in neither;
#' and that across the folds every pooled row is validated exactly once -- the
#' property that makes the k metrics a partition of the pool rather than an
#' arbitrary set of overlapping subsets.
#'
#' @param plan  A `fold_plan`.
#' @param meta  The point table the plan was made for. Given, the grouping is
#'   proven against it: no group may be split across train and validation.
#' @param group How the rows group, as in holdout().
#' @return tibble, one row per fold.
#' @export
check_fold_plan <- function(plan, meta = NULL, group = "auto") {
  stopifnot(inherits(plan, "fold_plan"))
  val_seen <- integer(0)

  rows <- lapply(seq_along(plan$folds), function(j) {
    f <- plan$folds[[j]]
    if (length(intersect(f$train, f$validation)) > 0L) {
      stop("Fold ", j, ": ", length(intersect(f$train, f$validation)),
           " row(s) are in BOTH train and validation.", call. = FALSE)
    }
    if (!is.null(f$test)) {
      bad <- length(intersect(f$test, c(f$train, f$validation)))
      if (bad > 0L) {
        stop("Fold ", j, ": ", bad, " test row(s) also appear in train or ",
             "validation.", call. = FALSE)
      }
    }
    val_seen <<- c(val_seen, f$validation)
    tibble::tibble(fold = j, n_train = length(f$train),
                   n_validation = length(f$validation),
                   n_test = length(f$test %||% integer(0)))
  })

  # holdout is one fold and validates only its own slice, so "every pooled row
  # validated once" is a k-fold property, not a universal one.
  if (plan$method != "holdout") {
    dup <- sum(duplicated(val_seen))
    if (dup > 0L) {
      stop(dup, " row(s) are validated in more than one fold -- the folds do ",
           "not partition the pool.", call. = FALSE)
    }
  }
  sizes <- dplyr::bind_rows(rows)

  # -- the no-split-group property, PROVEN rather than trusted -----------------
  #
  # Every constructor here is supposed to keep a profile's rows together:
  # holdout() and random_folds() group explicitly, spatial_folds() and
  # region_folds() get it for free because rows at one coordinate fall in one
  # block. "Supposed to" is the operative phrase -- three of those are separate
  # code paths, and the property is what matters, not the four arguments that
  # are meant to produce it.
  #
  # So it is checked here, against the data, whenever `meta` is available. It
  # costs one table() over the row indices and it is the difference between a
  # plan that is correct and a plan that was built by code intended to be.
  if (!is.null(meta)) {
    g <- .resolve_row_group(meta, group)

    # A PLAN BUILT UNGROUPED ON PURPOSE IS NOT A DEFECT.
    #
    # holdout() and random_folds() record what they did in params$grouping. If
    # that says every row was its own unit, the user asked for it -- maybe
    # their profile_id means something else entirely -- and stopping would be
    # the framework overruling a deliberate choice. It still says so, loudly,
    # because "deliberate" and "forgotten" look identical from here.
    #
    # A plan that recorded nothing (spatial_folds, region_folds, or one built
    # before this existed) IS checked strictly: those get the property by
    # construction, so a violation means the construction is broken.
    declared   <- plan$params$grouping
    on_purpose <- !is.null(declared) && grepl("own unit", declared, fixed = TRUE)

    if (isTRUE(attr(g, "grouped"))) {
      for (j in seq_along(plan$folds)) {
        f     <- plan$folds[[j]]
        roles <- list(train = f$train, validation = f$validation)
        if (!is.null(f$test)) roles$test <- f$test
        assigned <- unlist(lapply(names(roles), function(r)
          stats::setNames(rep(r, length(roles[[r]])), g[roles[[r]]])))
        if (length(assigned) == 0L) next
        per_group <- tapply(assigned, names(assigned),
                            function(z) length(unique(z)))
        split_g <- names(per_group)[per_group > 1L]
        if (length(split_g) > 0L) {
          msg <- paste0(
            "Fold ", j, ": ", length(split_g), " group(s) are split across ",
            "roles -- e.g. ", paste(utils::head(split_g, 4), collapse = ", "),
            ".\n  Rows of one profile in both training and scoring is the ",
            "leakage described by Wang et al. (2025, Geoderma): identical ",
            "covariates, autocorrelated target.\n  Rebuild the plan with ",
            "group = \"auto\" (the default) or a grouping of your own.")
          if (on_purpose) warning(msg, call. = FALSE) else stop(msg, call. = FALSE)
        }
      }
    }
  }

  sizes
}

#' Print a fold plan, with its per-fold sizes.
#'
#' @param x   A `fold_plan`, from [resolve_resampling()] or a fold constructor.
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.fold_plan <- function(x, ...) {
  cat("<fold_plan> ", x$method, " | ", x$n_folds, " fold(s) | ",
      x$n_rows, " rows in the store\n", sep = "")
  if (length(x$params) > 0L) {
    cat("  params: ",
        paste(sprintf("%s=%s", names(x$params),
                      vapply(x$params,
                             function(z) paste(format(z), collapse = ","),
                             character(1))),
              collapse = " | "), "\n", sep = "")
  }
  print(check_fold_plan(x), n = Inf)
  if (!is.null(x$buffer_dropped)) {
    bd <- x$buffer_dropped
    cat("  buffer: ", format(sum(bd$n_dropped), big.mark = ","),
        " training point(s) dropped (",
        sprintf("%.1f%%", mean(bd$pct_dropped)),
        " per fold on average)\n", sep = "")
    # BY CAUSE. One number cannot say which promise was paid for, and the two
    # promises are different claims: "validation is independent" decides the
    # config, "test is independent" is the number that leaves the building.
    if ("n_near_test" %in% names(bd)) {
      cat("    near validation: ", format(sum(bd$n_near_validation), big.mark = ","),
          " | near test: ", format(sum(bd$n_near_test), big.mark = ","),
          " (a point can be both)\n", sep = "")
      if (sum(bd$n_validation_dropped) > 0L) {
        cat("    validation points dropped for being near test: ",
            format(sum(bd$n_validation_dropped), big.mark = ","), "\n", sep = "")
      }
    }
  }
  invisible(x)
}

# -- summarising repetitions --------------------------------------------------
#
# The part of caret worth borrowing outright is resamples(): once a config has
# been trained more than once, the honest report of it is a mean AND a spread,
# never the single number that happened to come out.
#
# A comparison table has one row per UNIT -- (config, fold, seed). These two
# functions turn that into the two things a person actually decides with:
#
#   summarise_resamples()  what each config scores, and how much it wobbles
#   seed_noise_floor()     how much any config wobbles for no reason at all
#
# The second is the one that changes behaviour. If retraining the SAME config
# with a different seed moves validation CCC by 0.02, then a 0.01 gap between
# two configs is not a finding, and a grid of 24 configs ranked by a single run
# each is mostly a ranking of luck -- the winner's curse, measured instead of
# argued about.

#' Aggregate a unit-level comparison table into one row per config.
#'
#' @param comparison Unit-level table from run_cnn_tuning()/run_cnn_resample().
#' @param metrics    Metric column names to aggregate (any that exist).
#' @param by         Grouping column; `config_id` unless you have a reason.
#' @return One row per config: n, mean, sd and se of each metric, ranked by the
#'   first metric. Failed units are excluded from the statistics but counted in
#'   `n_failed`, because a config that crashes 2 runs in 3 is not the same as
#'   one that completed all three.
#' @export
summarise_resamples <- function(comparison,
                                metrics = c("val_ccc", "val_mae", "val_rmse",
                                            "val_r2", "val_mqi",
                                            "test_ccc", "test_mae"),
                                by = "config_id") {
  if (nrow(comparison) == 0L) return(tibble::tibble())
  metrics <- intersect(metrics, names(comparison))
  if (length(metrics) == 0L) {
    stop("None of the requested metric columns are present.", call. = FALSE)
  }

  ok  <- comparison[comparison$status == "success", , drop = FALSE]
  bad <- comparison[comparison$status != "success", , drop = FALSE]

  agg <- ok %>%
    dplyr::group_by(dplyr::across(dplyr::all_of(by))) %>%
    dplyr::summarise(
      n_units  = dplyr::n(),
      n_folds  = dplyr::n_distinct(.data$fold),
      n_seeds  = dplyr::n_distinct(.data$seed),
      dplyr::across(dplyr::all_of(metrics),
                    list(mean = ~ mean(.x, na.rm = TRUE),
                         sd   = ~ stats::sd(.x, na.rm = TRUE)),
                    .names = "{.col}_{.fn}"),
      .groups = "drop"
    )

  # COMPLEXITY TRAVELS WITH THE SUMMARY.
  #
  # one_se() needs a measure of "simplest" in the same table as the means, and
  # n_params is a property of the CONFIG, not of the unit -- it is identical
  # across the folds and seeds of one config, so it summarises by taking the
  # first value rather than by averaging.
  #
  # Without this the column exists per unit and vanishes at the group_by, and
  # one_se() then fails on a table that "obviously" has what it needs. Any
  # future per-config constant belongs in this vector, not in a second join.
  const_cols <- intersect(c("n_params", "n_features", "model"), names(ok))
  if (length(const_cols) > 0L) {
    consts <- ok %>%
      dplyr::group_by(dplyr::across(dplyr::all_of(by))) %>%
      dplyr::summarise(dplyr::across(dplyr::all_of(const_cols),
                                     ~ dplyr::first(.x)), .groups = "drop")
    agg <- dplyr::left_join(agg, consts, by = by)
  }

  # Standard error of the mean over repetitions: the number that says whether
  # two configs are actually distinguishable. With n_units = 1 it is NA, which
  # is the correct answer -- one run gives no estimate of its own spread.
  key <- metrics[1]
  agg[[paste0(key, "_se")]] <- agg[[paste0(key, "_sd")]] / sqrt(agg$n_units)

  if (nrow(bad) > 0L) {
    nf <- bad %>%
      dplyr::group_by(dplyr::across(dplyr::all_of(by))) %>%
      dplyr::summarise(n_failed = dplyr::n(), .groups = "drop")
    agg <- dplyr::left_join(agg, nf, by = by)
  }
  # `agg$n_failed` on a tibble that has no such column warns before it returns
  # NULL, so ask by name instead of probing and catching.
  agg$n_failed <- if ("n_failed" %in% names(agg)) na_to_zero(agg$n_failed)
                  else rep(0L, nrow(agg))

  desc <- grepl("ccc|r2|nse|rpd", key)
  agg  <- agg[order(agg[[paste0(key, "_mean")]], decreasing = desc), ]
  agg$rank <- seq_len(nrow(agg))
  agg
}

# NA counts as zero failures: a config with no failed row gets no row in the
# join, not a missing value. Written out rather than pulling in tidyr for it.
na_to_zero <- function(x) {
  x[is.na(x)] <- 0L
  as.integer(x)
}

#' How much does a metric move when ONLY the seed changes?
#'
#' The noise floor of the whole experiment. Computed within (config, fold), so
#' nothing but the random draw differs between the runs being compared; then
#' summarised across all configs.
#'
#' Any difference between two configs smaller than this is not evidence. That
#' sentence is the entire reason the function exists, and it is why `oneSE`-style
#' selection is worth having: the best mean is frequently the luckiest draw.
#'
#' @param comparison Unit-level comparison table.
#' @param metric     Metric column to measure.
#' @return list(by_config = tibble, median_sd, max_sd, n_comparable)
#' @export
seed_noise_floor <- function(comparison, metric = "val_ccc") {
  if (!metric %in% names(comparison)) {
    stop("No column '", metric, "' in the comparison table.", call. = FALSE)
  }
  ok <- comparison[comparison$status == "success", , drop = FALSE]

  # A REPETITION IS A DISTINCT SEED, NOT A ROW. Two failures, one cause:
  # counting rows instead of counting what actually varies.
  #
  # Two rows of the same (config, fold) under the SAME seed are not two
  # repetitions -- they are one thing counted twice, and the sd between them
  # comes out 0, announcing a noise floor that does not exist. Zero is the
  # worst possible value here: it makes any difference between configs look
  # like evidence.
  #
  # Two rows whose metric came back NA are not two repetitions either. Without
  # the non-NA requirement the report announces "estimated over 2 combinations"
  # with an sd of NA -- a claim of coverage the data does not support, which is
  # worse than admitting it cannot be estimated.
  by_config <- ok %>%
    dplyr::group_by(.data$config_id, .data$fold) %>%
    dplyr::filter(
      dplyr::n_distinct(.data$seed[!is.na(.data[[metric]])]) > 1L,
      sum(!is.na(.data[[metric]])) > 1L
    ) %>%
    dplyr::summarise(
      n_seeds = dplyr::n(),
      mean    = mean(.data[[metric]], na.rm = TRUE),
      sd      = stats::sd(.data[[metric]], na.rm = TRUE),
      range   = .safe_range(.data[[metric]]),
      .groups = "drop"
    )

  list(
    metric       = metric,
    by_config    = by_config,
    n_comparable = nrow(by_config),
    median_sd    = if (nrow(by_config)) stats::median(by_config$sd, na.rm = TRUE)
                   else NA_real_,
    max_range    = if (nrow(by_config)) max(by_config$range, na.rm = TRUE)
                   else NA_real_
  )
}

# range() of an all-NA or empty vector returns -Inf/Inf WITH a warning, and
# dplyr reaches that case even when no group qualifies: with zero groups it
# still evaluates each expression once on an empty slice to infer the result
# type. Returning NA there is both correct and quiet -- an unknown spread is
# not an infinite one.
.safe_range <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0L) NA_real_ else diff(range(x))
}

# -- selection rules ----------------------------------------------------------
#
# Ranking by the best mean is one rule, not the only one, and on this project
# it is frequently the wrong one: the first spatially-validated run put the top
# three configs inside 0.019 of each other against a seed noise floor of
# 0.0275. Picking the top of that list is picking the luckiest draw.
#
# one_se() is caret's `selectionFunction = "oneSE"`: among the configs whose
# mean is within ONE STANDARD ERROR of the best, take the SIMPLEST. It is
# offered, never imposed -- `run_cnn_resample()` still ranks by the mean, and
# choosing this rule stays a decision someone makes on purpose.

#' Pick the simplest config within one standard error of the best.
#'
#' @param by_config  From summarise_resamples().
#' @param metric     Metric column stem, e.g. "val_ccc".
#' @param complexity Column holding the simplicity ordering (lower = simpler),
#'   or a numeric vector the same length. Defaults to `n_params`, which
#'   count_model_params() produces.
#' @param maximise   TRUE when higher is better (CCC, R2); FALSE for an error.
#' @return One row of `by_config`, with `within_one_se` (how many configs were
#'   tied) and `simpler_than_best` (whether the rule actually moved the choice)
#'   attached -- a selection rule that silently returns the same answer as the
#'   default should say so.
#' @export
one_se <- function(by_config, metric = "val_ccc", complexity = "n_params",
                   maximise = NULL) {
  mean_col <- paste0(metric, "_mean")
  se_col   <- paste0(metric, "_se")
  for (cl in c(mean_col, se_col)) {
    if (!cl %in% names(by_config)) {
      stop("by_config has no column '", cl, "' -- one_se() needs the mean AND ",
           "the standard error, which summarise_resamples() produces only when ",
           "a config was trained more than once.", call. = FALSE)
    }
  }
  if (is.null(maximise)) maximise <- grepl("ccc|r2|nse|rpd|mqi", metric)

  cx <- if (is.character(complexity) && length(complexity) == 1L) {
    if (!complexity %in% names(by_config)) {
      stop("No complexity column '", complexity, "'. \"Simplest\" needs a ",
           "definition -- add one with count_model_params(), or pass a numeric ",
           "vector. The framework will not invent an ordering.", call. = FALSE)
    }
    by_config[[complexity]]
  } else {
    complexity
  }
  stopifnot(length(cx) == nrow(by_config))

  mu <- by_config[[mean_col]]
  se <- by_config[[se_col]]
  best_i <- if (maximise) which.max(mu) else which.min(mu)
  if (!is.finite(se[best_i])) {
    stop("The best config has no standard error (it was trained once). ",
         "one_se() cannot tell a tie from a gap without repetitions.",
         call. = FALSE)
  }

  # The tolerance band comes from the BEST config's own standard error, which
  # is what makes this "within the noise of the winner" rather than an
  # arbitrary margin.
  within <- if (maximise) mu >= mu[best_i] - se[best_i]
            else          mu <= mu[best_i] + se[best_i]
  cand <- which(within & is.finite(cx))
  if (length(cand) == 0L) cand <- best_i

  pick <- cand[which.min(cx[cand])]
  out  <- by_config[pick, , drop = FALSE]
  attr(out, "within_one_se")    <- length(cand)
  attr(out, "simpler_than_best") <- !identical(pick, best_i)
  attr(out, "band") <- if (maximise) mu[best_i] - se[best_i]
                       else          mu[best_i] + se[best_i]
  out
}

#' Say what one_se() did, including when it did nothing.
#'
#' @param pick   From one_se().
#' @param metric The metric it selected on, for the report.
#' @param digits Digits of the threshold.
#' @return `pick`, invisibly.
#' @export
print_one_se <- function(pick, metric = "val_ccc", digits = 4L) {
  n_tied <- attr(pick, "within_one_se")
  cat("  one_se (", metric, "): ", n_tied,
      " config(s) within one standard error of the best",
      " (threshold ", round(attr(pick, "band"), digits), ")\n", sep = "")
  cat("    chosen: ", pick$config_id[1], sep = "")
  if (isTRUE(attr(pick, "simpler_than_best"))) {
    cat("  -- SIMPLER than the top-ranked config\n")
  } else {
    cat("  -- same as the top-ranked config; the rule changed nothing\n")
  }
  invisible(pick)
}

#' Print a noise-floor report in the terms it should be read in.
#'
#' @param nf     From seed_noise_floor().
#' @param digits Digits of the numbers.
#' @return `nf`, invisibly; NULL when no floor could be estimated.
#' @export
print_noise_floor <- function(nf, digits = 4L) {
  if (nf$n_comparable == 0L) {
    cat("  Noise floor: not estimable -- no config was trained under more ",
        "than one seed within the same fold.\n", sep = "")
    return(invisible(NULL))
  }
  cat("  Noise floor (", nf$metric, "), with only the seed changing:\n",
      sep = "")
  cat("    median sd between seeds : ", round(nf$median_sd, digits), "\n",
      sep = "")
  cat("    widest range observed   : ", round(nf$max_range, digits), "\n",
      sep = "")
  cat("    estimated over ", nf$n_comparable, " (config x fold) combination(s)\n",
      sep = "")
  cat("    -> a gap between configs smaller than this is NOT evidence\n")
  invisible(nf)
}

# ── Comparing two families on the same folds ──────────────────────────────────

#' Paired comparison between two model families.
#'
#' WHY PAIRED, AND WHY IT MATTERS HERE.
#'
#' Every family in stage 03b runs on the SAME fold plan with the SAME seeds, by
#' construction. So each family's units come in matched pairs: rf_context on
#' fold 2 seed 3 and the CNN on fold 2 seed 3 saw the identical training rows.
#'
#' Comparing the two MEANS and their separate spreads throws that away. Most of
#' the spread across units is the fold -- some folds are simply harder, and both
#' families suffer on them together. A paired comparison subtracts that shared
#' difficulty out before asking whether anything is left.
#'
#' The first 03b run made exactly this mistake in the reader's favour: it
#' reported +0.0091 CCC against a seed noise floor of 0.0383 and concluded
#' "smaller than the noise". The conclusion happened to hold, but the reasoning
#' did not -- the noise floor is the spread of ONE family across seeds, which is
#' not the standard error of a DIFFERENCE, and comparing a difference to it can
#' hide a real effect as easily as it can invent one.
#'
#' What is reported is a mean difference with its own standard error and a
#' paired t interval. Not a p-value on its own: with 9 pairs, "not significant"
#' is mostly a statement about 9, and the interval says how large an effect is
#' still compatible with the data -- which is the actual question when the
#' answer is "the convolution buys nothing".
#'
#' @param a,b       Comparison tibbles (or the `comparison` element of a fit).
#' @param metric    Column to compare. Higher-is-better is assumed for the
#'   verdict text unless the name matches a known error metric.
#' @param config_a,config_b Which config of each family to compare. Defaults to
#'   the best by mean `metric`, which is the config someone would deploy.
#' @param label_a,label_b Names for the report.
#' @param conf      Interval level.
#' @return An object of class "paired_comparison".
#' @export
paired_family_test <- function(a, b, metric = "val_ccc",
                               config_a = NULL, config_b = NULL,
                               label_a = "a", label_b = "b", conf = 0.95) {
  pick <- function(d, cfg, who) {
    if (inherits(d, "dsm_fit")) d <- d$comparison
    if (!is.data.frame(d)) stop("`", who, "` is not a comparison table.", call. = FALSE)
    need <- c("fold", "seed", "config_id", metric)
    miss <- setdiff(need, names(d))
    if (length(miss)) {
      stop("`", who, "` has no column(s): ", paste(miss, collapse = ", "),
           ".\n  A paired comparison needs fold and seed to pair ON.",
           call. = FALSE)
    }
    if ("status" %in% names(d)) d <- d[d$status == "success", , drop = FALSE]
    d <- d[is.finite(d[[metric]]), , drop = FALSE]
    chosen_by <- "caller"
    if (is.null(cfg)) {
      # A FALLBACK, NOT THE INTENDED PATH.
      #
      # Picking per metric makes each comparison internally optimal and the
      # REPORT incoherent: run this for CCC and again for MAE and "rf_context"
      # can name two different forests, with nothing on the page saying so. A
      # family should be represented by the config someone would deploy, chosen
      # once on the selection metric -- which is what the caller passes.
      #
      # When it does not, choose sensibly and SAY so: print.paired_comparison
      # marks a config that was chosen here rather than handed in.
      m <- tapply(d[[metric]], d$config_id, mean, na.rm = TRUE)
      # Error metrics are better when small; everything else here is a score.
      cfg <- names(m)[if (.metric_is_error(metric)) which.min(m) else which.max(m)]
      chosen_by <- paste0("chosen here: best ", metric)
    }
    d <- d[d$config_id == cfg, , drop = FALSE]
    if (nrow(d) == 0L) {
      stop("config '", cfg, "' has no successful units in `", who, "`.",
           call. = FALSE)
    }
    list(cfg = cfg, d = d, chosen_by = chosen_by)
  }

  A <- pick(a, config_a, "a"); B <- pick(b, config_b, "b")

  key <- function(d) paste(d$fold, d$seed, sep = "_")
  ka  <- key(A$d); kb <- key(B$d)
  if (anyDuplicated(ka) || anyDuplicated(kb)) {
    stop("A (fold, seed) pair appears twice within one config. ",
         "Pairing would be ambiguous.", call. = FALSE)
  }

  common <- intersect(ka, kb)
  # UNPAIRED UNITS ARE DROPPED, LOUDLY. Silently falling back to an unpaired
  # comparison because one family lost a fold to a failure is how a run reports
  # a difference between two different experiments.
  if (length(common) < 3L) {
    stop("Only ", length(common), " (fold, seed) pair(s) are shared by the two ",
         "families.\n  A paired comparison needs the same plan on both sides.",
         call. = FALSE)
  }
  dropped <- length(union(ka, kb)) - length(common)

  va <- A$d[[metric]][match(common, ka)]
  vb <- B$d[[metric]][match(common, kb)]
  d  <- va - vb

  n  <- length(d)
  md <- mean(d)
  se <- stats::sd(d) / sqrt(n)
  tq <- stats::qt(1 - (1 - conf) / 2, df = n - 1)

  # The unpaired SE, reported alongside, so the gain from pairing is visible
  # rather than asserted. When the folds dominate, this is much the larger.
  se_unpaired <- sqrt(stats::var(va) / n + stats::var(vb) / n)

  structure(list(
    metric = metric, label_a = A$cfg, label_b = B$cfg,
    chosen_a = A$chosen_by, chosen_b = B$chosen_by,
    name_a = label_a, name_b = label_b,
    n_pairs = n, dropped = dropped,
    mean_a = mean(va), mean_b = mean(vb),
    diff = md, se = se, se_unpaired = se_unpaired,
    ci = c(md - tq * se, md + tq * se),
    t = if (se > 0) md / se else NA_real_,
    p = if (se > 0) 2 * stats::pt(-abs(md / se), df = n - 1) else NA_real_,
    conf = conf, differences = d, pairs = common,
    higher_is_better = !.metric_is_error(metric)
  ), class = "paired_comparison")
}

.metric_is_error <- function(metric) {
  grepl("(^|_)(mae|rmse|mse|loss|bias|error)($|_)", tolower(metric))
}

#' Print a `paired_comparison`
#'
#' @param x   A `paired_comparison`, from [paired_family_test()].
#' @param ... Ignored.
#' @return `x`, invisibly.
#' @keywords internal
#' @export
print.paired_comparison <- function(x, ...) {
  cat("\nPaired comparison --", x$metric, "\n")
  cat(strrep("-", 62), "\n")
  note <- function(by) if (identical(by, "caller")) "" else paste0("   [", by, "]")
  cat(sprintf("  %-14s %-10s mean = %.4f%s\n", x$name_a, x$label_a,
              x$mean_a, note(x$chosen_a)))
  cat(sprintf("  %-14s %-10s mean = %.4f%s\n", x$name_b, x$label_b,
              x$mean_b, note(x$chosen_b)))
  cat(sprintf("  paired on %d (fold, seed) unit(s)%s\n", x$n_pairs,
              if (x$dropped > 0) sprintf("; %d unpaired dropped", x$dropped) else ""))
  cat("\n")
  cat(sprintf("  difference   = %+.4f  (SE %.4f)\n", x$diff, x$se))
  cat(sprintf("  %g%% CI       = [%+.4f, %+.4f]\n", 100 * x$conf, x$ci[1], x$ci[2]))
  cat(sprintf("  t(%d)        = %+.2f   p = %.3f\n", x$n_pairs - 1L, x$t, x$p))
  cat(sprintf("  unpaired SE  = %.4f  (pairing is worth %.1fx here)\n",
              x$se_unpaired,
              if (x$se > 0) x$se_unpaired / x$se else NA_real_))
  cat("\n")

  crosses <- x$ci[1] <= 0 && x$ci[2] >= 0
  if (crosses) {
    # The interval, not the p-value, is the useful statement: it bounds what is
    # still possible. "No difference" and "we could not resolve one" look the
    # same in a p-value and are not the same claim.
    big <- max(abs(x$ci))
    cat(sprintf("  -> NOT SEPARATED. The data are compatible with anything from\n"))
    cat(sprintf("     %+.4f to %+.4f, so an effect as large as %.4f %s cannot be\n",
                x$ci[1], x$ci[2], big, x$metric))
    cat(sprintf("     ruled out -- and neither can zero.\n"))
  } else {
    better <- if (xor(x$diff > 0, !x$higher_is_better)) x$name_a else x$name_b
    cat(sprintf("  -> SEPARATED at %g%%: %s is ahead, by %.4f to %.4f.\n",
                100 * x$conf, better, min(abs(x$ci)), max(abs(x$ci))))
  }
  invisible(x)
}
