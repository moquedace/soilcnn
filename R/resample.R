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
# THE TEST SET IS NOT RESAMPLED
# Rows marked `test` in dataset_role stay test in EVERY fold and are never
# trained on. Resampling partitions the train+validation pool only. A test set
# that moves between folds has been seen by some model in the ensemble, and
# then it is not a test set -- it is a third validation set with a misleading
# name.

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

# The pool that folds partition, and the test rows that ride along unchanged.
.fold_pool <- function(meta) {
  if (!"dataset_role" %in% names(meta)) {
    stop("meta has no dataset_role column -- cannot tell the pool from the ",
         "held-out test rows.", call. = FALSE)
  }
  pool <- which(meta$dataset_role %in% c("train", "validation"))
  test <- which(meta$dataset_role == "test")
  if (length(pool) == 0L) {
    stop("No rows with dataset_role in {train, validation}.", call. = FALSE)
  }
  list(pool = pool, test = test)
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

#' The single fixed split, expressed as a one-fold plan.
#'
#' The DEFAULT, deliberately: it reproduces exactly what the pipeline did
#' before resampling existed, so turning resampling on is a choice the user
#' makes, never something a version bump did to them. It is also the only plan
#' whose validation set is the one stage 01 wrote, which is what makes results
#' comparable with earlier runs.
#'
#' @param meta  Patch store meta.
#' @param roles Roles to extract, in order.
#' @return A one-fold `fold_plan`.
holdout <- function(meta, roles = c("train", "validation", "test")) {
  idx <- split_index_from_meta(meta, roles = roles)
  .new_fold_plan(list(idx), "holdout", list(roles = roles), meta)
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
random_folds <- function(meta, k = 5L, seed = 42L) {
  k <- as.integer(k)
  if (k < 2L) stop("k must be at least 2.", call. = FALSE)
  p <- .fold_pool(meta)
  if (k > length(p$pool)) {
    stop("k = ", k, " but the pool has only ", length(p$pool), " rows.",
         call. = FALSE)
  }

  assignment <- with_local_seed(seed, ((sample.int(length(p$pool)) - 1L) %% k) + 1L)

  .new_fold_plan(
    .folds_from_assignment(assignment, p$pool, p$test, k),
    "random_folds", list(k = k, seed = seed), meta,
    assignment = tibble::tibble(sample_id = meta$sample_id[p$pool],
                                fold = assignment)
  )
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
#'   validation patch you need max(window) * resolution * SQRT(2), not
#'   max(window) * resolution: patch overlap is a square condition and the
#'   diagonal escapes a circular buffer. See apply_buffer().
#' @param seed       Seed for dealing blocks to folds.
#' @param blocks_per_fold Target blocks per fold when `block_size` is NULL.
spatial_folds <- function(meta, k = 5L, block_size = NULL, buffer = NULL,
                          seed = 42L, blocks_per_fold = 10L) {
  k <- as.integer(k)
  if (k < 2L) stop("k must be at least 2.", call. = FALSE)
  if (!all(c("x", "y") %in% names(meta))) {
    stop("meta needs x and y columns for spatial folds.", call. = FALSE)
  }
  p <- .fold_pool(meta)
  x <- as.numeric(meta$x)[p$pool]
  y <- as.numeric(meta$y)[p$pool]
  if (anyNA(x) || anyNA(y)) {
    stop("x/y are NA for ", sum(is.na(x) | is.na(y)), " pooled row(s).",
         call. = FALSE)
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

  bx  <- floor((x - min(x)) / block_size)
  by  <- floor((y - min(y)) / block_size)
  blk <- paste(bx, by, sep = "_")
  ub  <- unique(blk)
  if (length(ub) < k) {
    stop("Only ", length(ub), " spatial block(s) for k = ", k,
         " folds -- block_size is too large for this extent.", call. = FALSE)
  }

  # Shuffle block order before the greedy pass: point tables usually arrive
  # sorted by geography, and dealing in that order would hand each fold one
  # contiguous strip of the map.
  ub       <- with_local_seed(seed, sample(ub))
  sizes    <- as.integer(table(factor(blk, levels = ub))[ub])
  blk_fold <- .deal_groups(sizes, k)
  assignment <- blk_fold[match(blk, ub)]

  plan <- .new_fold_plan(
    .folds_from_assignment(assignment, p$pool, p$test, k),
    "spatial_folds",
    list(k = k, block_size = block_size, block_size_auto = auto,
         n_blocks = length(ub), seed = seed), meta,
    assignment = tibble::tibble(sample_id = meta$sample_id[p$pool],
                                fold = assignment, block = blk)
  )
  # Blocking chooses where the cut falls; the buffer is what makes the cut
  # mean something. See apply_buffer() for why one without the other is not
  # enough -- it is measured there, not assumed.
  apply_buffer(plan, meta, buffer)
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
region_folds <- function(meta, group, k = NULL) {
  if (length(group) != nrow(meta)) {
    stop("group has ", length(group), " values but meta has ", nrow(meta),
         " rows.", call. = FALSE)
  }
  p <- .fold_pool(meta)
  g <- as.character(group)[p$pool]
  if (anyNA(g)) {
    stop(sum(is.na(g)), " pooled row(s) have no group label -- decide what ",
         "they belong to instead of letting them fall into an NA group.",
         call. = FALSE)
  }
  ug <- unique(g)
  if (is.null(k)) k <- length(ug)
  k <- as.integer(k)
  if (k < 2L) stop("Need at least 2 groups (or k >= 2).", call. = FALSE)
  if (k > length(ug)) {
    stop("k = ", k, " but there are only ", length(ug), " group(s).",
         call. = FALSE)
  }

  sizes      <- as.integer(table(factor(g, levels = ug))[ug])
  g_fold     <- .deal_groups(sizes, k)
  assignment <- g_fold[match(g, ug)]

  .new_fold_plan(
    .folds_from_assignment(assignment, p$pool, p$test, k),
    "region_folds", list(k = k, n_groups = length(ug)), meta,
    assignment = tibble::tibble(sample_id = meta$sample_id[p$pool],
                                fold = assignment, group = g)
  )
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
# So, with Euclidean distance:
#
#     buffer >= max(window) * resolution * sqrt(2)     (41% wider)
#
# A square buffer would be both exact and cheaper (buffer = w * res, no
# diagonal waste) and is the right geometry for square patches -- see the
# `metric` argument proposed in docs/revisao_e_prospeccao_2026_09.md §A0.
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
apply_buffer <- function(plan, meta, buffer) {
  stopifnot(inherits(plan, "fold_plan"))
  if (is.null(buffer) || buffer <= 0) return(plan)
  if (!all(c("x", "y") %in% names(meta))) {
    stop("meta needs x and y columns to apply a buffer.", call. = FALSE)
  }
  x <- as.numeric(meta$x)
  y <- as.numeric(meta$y)

  dropped <- vector("list", length(plan$folds))
  for (j in seq_along(plan$folds)) {
    f    <- plan$folds[[j]]
    near <- .near_any(x[f$train], y[f$train], x[f$validation], y[f$validation],
                      buffer)
    n_before <- length(f$train)
    plan$folds[[j]]$train <- f$train[!near]
    if (length(plan$folds[[j]]$train) == 0L) {
      stop("Fold ", j, ": the buffer removed every training point. The buffer ",
           "is large relative to the spacing of this data.", call. = FALSE)
    }
    dropped[[j]] <- tibble::tibble(
      fold = j, n_train_before = n_before, n_dropped = sum(near),
      pct_dropped = round(100 * sum(near) / n_before, 2))
  }

  plan$params$buffer  <- buffer
  plan$buffer_dropped <- dplyr::bind_rows(dropped)
  plan
}

# For each (tx, ty), is there any (vx, vy) within `buffer`?
#
# Bucketed by the buffer size, testing the 9 neighbouring buckets, so the cost
# is linear instead of the n_train x n_validation product -- on the real data
# that product is ~2e8 per fold, the difference between a second and minutes.
# Distances inside the candidate set are exact: the buckets narrow the search,
# they never decide the answer.
.near_any <- function(tx, ty, vx, vy, buffer) {
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
    hit[rows] <- vapply(rows, function(i) {
      any((cx - tx[i])^2 + (cy - ty[i])^2 <= b2)
    }, logical(1))
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
#' Needs R/diagnostics.R sourced (spatial_overlap_report).
#'
#' @param plan      A fold_plan.
#' @param meta      Patch store meta with x/y.
#' @param cell_size Raster resolution, in the units of x/y.
#' @param windows   Window sizes to report overlap for.
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
#' @return tibble, one row per fold.
check_fold_plan <- function(plan) {
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
  dplyr::bind_rows(rows)
}

#' Print a fold plan, with its per-fold sizes.
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
    cat("  buffer: ", format(sum(x$buffer_dropped$n_dropped), big.mark = ","),
        " training point(s) dropped (",
        sprintf("%.1f%%", mean(x$buffer_dropped$pct_dropped)),
        " per fold on average)\n", sep = "")
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
seed_noise_floor <- function(comparison, metric = "val_ccc") {
  if (!metric %in% names(comparison)) {
    stop("No column '", metric, "' in the comparison table.", call. = FALSE)
  }
  ok <- comparison[comparison$status == "success", , drop = FALSE]

  # Duas LINHAS nao bastam: duas repeticoes cuja metrica veio NA nao dao
  # espalhamento nenhum. Sem isto o relatorio anuncia "estimado em 2
  # combinacoes" com um sd que e NA -- uma afirmacao de cobertura que os dados
  # nao sustentam, que e pior do que admitir que nao da para estimar.
  # Repeticao e SEMENTE DISTINTA, nao linha.
  #
  # Duas linhas do mesmo (config, fold) com a MESMA semente nao sao duas
  # repeticoes -- sao a mesma coisa contada duas vezes, e o sd entre elas sai
  # 0, anunciando um piso de ruido inexistente. Acontece quando uma tabela
  # acumula linhas de esquemas diferentes; ver tambem a exigencia de valor
  # nao-NA logo abaixo. Em ambos os casos o erro e contar a linha em vez de
  # contar o que de fato varia.
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

#' Print a noise-floor report in the terms it should be read in.
print_noise_floor <- function(nf, digits = 4L) {
  if (nf$n_comparable == 0L) {
    cat("  Piso de ruido: nao estimavel -- nenhuma config foi treinada com ",
        "mais de uma semente no mesmo fold.\n", sep = "")
    return(invisible(NULL))
  }
  cat("  Piso de ruido (", nf$metric, "), so a semente muda:\n", sep = "")
  cat("    sd mediano entre sementes : ", round(nf$median_sd, digits), "\n",
      sep = "")
  cat("    maior amplitude observada : ", round(nf$max_range, digits), "\n",
      sep = "")
  cat("    estimado em ", nf$n_comparable, " combinacao(oes) (config x fold)\n",
      sep = "")
  cat("    -> diferenca entre configs menor que isso NAO e evidencia\n")
  invisible(nf)
}
