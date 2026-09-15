# Unit test: fold constructors (R/resample.R)
#
# A resampling plan is the one thing in the pipeline whose bugs do not look
# like bugs. A fold that leaks produces BETTER metrics, not an error -- the run
# finishes, the numbers improve, and nothing anywhere says the answer is wrong.
# So the properties have to be asserted mechanically, on data whose geography
# is known by construction.
#
# Verifies:
#   1. the PLAN carves the test set -- nothing upstream decides roles, and the
#      point table carries no dataset_role column at all
#   1b. the criterion cuts the test the SAME way it cuts the folds: a spatial
#      test comes out of whole blocks, a random one does not. And changing k
#      does not move the test set, or two plans could not be compared.
#   2. every plan: train and validation disjoint, test in neither
#   3. k-fold plans partition what is left after the test -- each row
#      validated exactly ONCE
#   4. the test set is identical in every fold and never trained on
#   5. spatial_folds(): no spatial block is split across folds -- but a
#      CLUSTER can be, because the grid is drawn on the map and not on the
#      data. Asserted as a fact of the method, not wished away.
#   6. how much each plan leaks, MEASURED: random leaks completely, blocking
#      alone still leaks through the cut clusters, and only the buffer closes
#      it to zero
#   7. region_folds(): no group split across folds; k = n_groups is LOGO
#   8. greedy dealing keeps folds balanced even with very uneven blocks
#   9. plans are reproducible from the seed, and different seeds differ
#  10. with_local_seed() does not disturb the caller's RNG stream
#  11. bad arguments fail loudly (k too large, missing x/y, NA groups)
#  12. apply_buffer(): only ever removes TRAINING points, never touches
#      validation or test, reports its cost, and is exact (buckets narrow the
#      search, they never decide the answer)
#  13. summarise_resamples() / seed_noise_floor(): means and spreads computed
#      by hand, and the winner's curse demonstrated -- the config holding the
#      single best run is NOT the config with the best mean
#  14. block_subsample(): keeps WHOLE blocks and therefore preserves local
#      density, where a random subsample of the same size destroys it
#  15. one_se(): on a fixture where the two rules DISAGREE -- the best mean and
#      the simplest model within one standard error of it are different
#      configs, which is the only case where the rule earns its existence
#
# Run: source("D:/.../tests/test_resample.R")    (CPU, no torch needed)

suppressMessages({
  library(tibble)
  library(dplyr)
})

# -- project root: works under source() in the console AND under Rscript ------

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
      if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
    }
  }
  stop("Project root not found. setwd() to the deep_learning_caret root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
source(file.path(root, "R", "utils.R"))
source(file.path(root, "R", "dataset.R"))
# diagnostics.R owns spatial_overlap_report(), which resample.R calls from
# fold_leakage_report() and which this file asserts directly.
source(file.path(root, "R", "diagnostics.R"))
source(file.path(root, "R", "resample.R"))

ok <- c()

# -- synthetic meta: clustered points, geography known by construction --------
#
# 12 clusters on a 4x3 grid of sites, 30 points each, scattered within 200 m of
# the site centre; sites are 50 km apart. So "same cluster" is unambiguous, and
# a spatial split with blocks >= 50 km MUST keep a cluster together while a
# random split cannot.

set.seed(1)
n_site <- 12L
n_per  <- 30L
site_x <- rep(seq(0, 150000, length.out = 4L), times = 3L)
site_y <- rep(seq(0, 100000, length.out = 3L), each  = 4L)

# NOTE: no dataset_role column. That is the contract -- the point table says
# where the points are and what they are worth, and NOTHING about who trains.
meta <- tibble::tibble(
  sample_id  = seq_len(n_site * n_per),
  profile_id = seq_len(n_site * n_per),
  site       = rep(seq_len(n_site), each = n_per),
  x          = rep(site_x, each = n_per) + runif(n_site * n_per, -200, 200),
  y          = rep(site_y, each = n_per) + runif(n_site * n_per, -200, 200)
)

cat("  synthetic points     : ", nrow(meta), " (", n_site, " sites x ", n_per,
    ", sites 50 km apart)\n", sep = "")

# -- 1. the plan carves the test set, by its own criterion --------------------

test_frac <- 0.15

hp <- holdout(meta, validation_frac = 0.2, test_frac = test_frac, seed = 7L)
ok["holdout_is_one_fold"]  <- hp$n_folds == 1L
ok["holdout_method_label"] <- hp$method == "holdout"
ok["holdout_three_roles"]  <- setequal(names(hp$folds[[1]]),
                                       c("train", "validation", "test"))
ok["holdout_covers_everything"] <- identical(
  sort(unlist(hp$folds[[1]], use.names = FALSE)), seq_len(nrow(meta)))

# -- plans under test ---------------------------------------------------------

k  <- 4L
rp <- random_folds(meta, k = k, test_frac = test_frac, seed = 7L)
sp <- spatial_folds(meta, k = k, test_frac = test_frac, block_size = 25000,
                    seed = 7L)
gp <- region_folds(meta, group = meta$site, test_frac = test_frac, seed = 7L)

# A plan with no test set is legitimate, and must be possible to ask for --
# the framework does not invent a test set nobody requested.
np <- spatial_folds(meta, k = k, block_size = 25000, seed = 7L)
ok["no_test_when_not_asked"] <- all(vapply(
  np$folds, function(f) is.null(f$test), logical(1)))

plans <- list(holdout = hp, random = rp, spatial = sp, region = gp)

# -- 2/3/4. structural properties, asserted for every plan --------------------

for (nm in names(plans)) {
  p <- plans[[nm]]

  # check_fold_plan() throws on overlap; reaching a tibble IS the assertion
  tab <- try(check_fold_plan(p), silent = TRUE)
  ok[paste0(nm, "_check_passes")] <- is.data.frame(tab) && nrow(tab) == p$n_folds

  ok[paste0(nm, "_train_val_disjoint")] <- all(vapply(
    p$folds, function(f) length(intersect(f$train, f$validation)) == 0L,
    logical(1)))

  # the test rows are the SAME rows in every fold of the plan, and never train
  p_test <- p$folds[[1]]$test
  ok[paste0(nm, "_test_exists")] <- length(p_test) > 0L
  ok[paste0(nm, "_test_constant")] <- all(vapply(
    p$folds, function(f) identical(sort(f$test), sort(p_test)), logical(1)))
  ok[paste0(nm, "_test_frac_respected")] <-
    abs(length(p_test) / nrow(meta) - test_frac) < 0.12   # whole groups overshoot
  ok[paste0(nm, "_test_never_trains")] <- all(vapply(
    p$folds,
    function(f) length(intersect(f$test, c(f$train, f$validation))) == 0L,
    logical(1)))

  if (nm != "holdout") {
    # every NON-TEST row validated exactly once across the k folds
    seen <- sort(unlist(lapply(p$folds, `[[`, "validation"), use.names = FALSE))
    ok[paste0(nm, "_partitions_pool")] <-
      identical(seen, sort(setdiff(seq_len(nrow(meta)), p_test)))
    # and trained on in exactly (folds - 1) of them. Uses the PLAN's own
    # fold count, not the k of this block: region_folds() defaults to
    # leave-one-group-out, so it has one fold per site, not k.
    trained <- table(unlist(lapply(p$folds, `[[`, "train"), use.names = FALSE))
    ok[paste0(nm, "_trained_k_minus_1")] <- all(trained == (p$n_folds - 1L))
  }
}

# -- 1b. the criterion cuts the TEST the same way it cuts the folds ----------
#
# The whole point of moving the split into the plan. A spatial validation next
# to a randomly drawn test set puts two incomparable numbers in one table --
# measured on the real data, the random test came out 0.042 CCC easier.

blk_of <- function(rows, bs = 25000) {
  paste(floor((meta$x[rows] - min(meta$x)) / bs),
        floor((meta$y[rows] - min(meta$y)) / bs), sep = "_")
}
sp_test <- sp$folds[[1]]$test
sp_pool <- setdiff(seq_len(nrow(meta)), sp_test)
ok["spatial_test_is_whole_blocks"] <-
  length(intersect(blk_of(sp_test), blk_of(sp_pool))) == 0L

# THE DISTINCTION THE REPORT MUST KEEP. An identical patch (two points in one
# raster cell) is a defect under any plan. Shared pixels between neighbours are
# not -- they are what neighbouring samples look like, and under a random plan
# they are the condition being measured. Conflating them turns a legitimate
# random split into a reported failure.
ov <- spatial_overlap_report(round(meta$y[sp_pool] / 250),
                             round(meta$x[sp_pool] / 250),
                             ifelse(seq_along(sp_pool) %% 4 == 0,
                                    "validation", "train"))
ok["report_flags_identical_patches"] <-
  all(ov$matters[grepl("identical", ov$criterion)])
ok["report_does_not_flag_shared_pixels"] <-
  !any(ov$matters[grepl("shares pixels", ov$criterion)])
ok["report_covers_both"] <- dplyr::n_distinct(ov$matters) == 2L

# What the block criterion promises is BLOCKS, not sites: the grid is drawn on
# the map at an origin unrelated to the points, so a boundary can still cut a
# cluster. Asserting "no site split" would demand a guarantee the method does
# not make, and would hide the very fact the buffer exists to handle.
n_site_cut_test <- length(intersect(meta$site[sp_test], meta$site[sp_pool]))
ok["spatial_test_cuts_few_sites"] <- n_site_cut_test <= 2L

# A random plan cuts nearly all of them -- and that is NOT a defect, it is what
# a random split is. Asserted so the difference between the two criteria is a
# measured fact rather than a claim.
rp_test <- rp$folds[[1]]$test
n_site_cut_rand <- length(intersect(
  meta$site[rp_test], meta$site[setdiff(seq_len(nrow(meta)), rp_test)]))
ok["random_test_cuts_many_sites"] <- n_site_cut_rand > n_site_cut_test

cat("  sites on both sides of the test: spatial ", n_site_cut_test,
    " | random ", n_site_cut_rand, "  (random makes no separation promise)\n",
    sep = "")

# CHANGING k MUST NOT MOVE THE TEST SET. Two plans that score on different
# held-out data cannot be compared, so the test draw runs on its own stream.
sp_k7 <- spatial_folds(meta, k = 7L, test_frac = test_frac,
                       block_size = 25000, seed = 7L)
ok["test_set_independent_of_k"] <- identical(sort(sp_k7$folds[[1]]$test),
                                             sort(sp_test))

# A frozen test set is honoured verbatim: a test set redrawn every run is not
# a test set.
frozen <- meta$sample_id[sp_test]
sp_frozen <- spatial_folds(meta, k = 3L, test_ids = frozen,
                           block_size = 25000, seed = 99L)
ok["frozen_test_is_honoured"] <- identical(sort(sp_frozen$folds[[1]]$test),
                                           sort(sp_test))
ok["frozen_test_rejects_alien_ids"] <- inherits(
  try(spatial_folds(meta, k = 3L, test_ids = c(frozen, 999999L),
                    block_size = 25000), silent = TRUE), "try-error")

cat("  test carved by the plan  : spatial ", length(sp_test),
    " pts (whole blocks) | random ", length(rp_test), " pts\n", sep = "")

# -- 5. spatial: a block is never split across folds --------------------------

blk_fold <- sp$assignment %>%
  dplyr::group_by(block) %>%
  dplyr::summarise(n_folds = dplyr::n_distinct(fold), .groups = "drop")
ok["spatial_block_not_split"] <- all(blk_fold$n_folds == 1L)

# A BLOCK is never split -- that is what the constructor guarantees. A SITE
# can be, and this is the point: the block grid is drawn on the MAP, at an
# origin that has nothing to do with where the points are, so a boundary can
# fall straight through a cluster. Asserted as a FACT of the method, not
# wished away: here 1 of the 12 sites is cut.
site_fold <- tibble::tibble(site = meta$site[match(sp$assignment$sample_id,
                                                   meta$sample_id)],
                            fold = sp$assignment$fold) %>%
  dplyr::group_by(site) %>%
  dplyr::summarise(n_folds = dplyr::n_distinct(fold), .groups = "drop")
n_site_cut <- sum(site_fold$n_folds > 1L)
ok["blocking_alone_can_cut_a_cluster"] <- n_site_cut > 0L

cat("  sites cut by a boundary  : ", n_site_cut, " of ", n_site,
    "  <- this is why the buffer exists\n", sep = "")

# -- 6. spatial really is more separated than random (measured) ---------------
#
# For each plan, the share of validation points that have a TRAINING point
# within 1 km. Random folds split clusters, so that share is near 1; spatial
# folds keep clusters whole, so it should be ~0.

near_share <- function(plan) {
  shares <- vapply(plan$folds, function(f) {
    vx <- meta$x[f$validation]; vy <- meta$y[f$validation]
    tx <- meta$x[f$train];      ty <- meta$y[f$train]
    hit <- vapply(seq_along(vx), function(i) {
      any((tx - vx[i])^2 + (ty - vy[i])^2 < 1000^2)
    }, logical(1))
    mean(hit)
  }, numeric(1))
  mean(shares)
}

sp_buf <- spatial_folds(meta, k = k, test_frac = test_frac,
                        block_size = 25000, buffer = 2000, seed = 7L)

share_random   <- near_share(rp)
share_spatial  <- near_share(sp)
share_buffered <- near_share(sp_buf)

ok["random_leaks_neighbours"] <- share_random > 0.95
ok["spatial_beats_random"]    <- share_spatial < share_random
# Blocking alone does NOT reach zero -- the cut cluster leaks. Asserting
# "< 0.01" here would have been the test lying to protect the method.
ok["blocking_alone_still_leaks"] <- share_spatial > 0
# The buffer is what closes it, and it has to close it COMPLETELY: buffer
# 2 km > the 400 m spread of a site, so no training point can survive next to
# a validation point.
ok["buffer_closes_the_leak"] <- share_buffered == 0

cat("  neighbour <1 km in train : random ",
    sprintf("%.1f%%", 100 * share_random), " | spatial ",
    sprintf("%.1f%%", 100 * share_spatial), " | spatial+buffer ",
    sprintf("%.1f%%", 100 * share_buffered), "\n", sep = "")

# -- 7. region_folds ----------------------------------------------------------

grp_fold <- gp$assignment %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(n_folds = dplyr::n_distinct(fold), .groups = "drop")
ok["region_group_not_split"] <- all(grp_fold$n_folds == 1L)
ok["region_logo_by_default"]  <- gp$n_folds <= n_site   # minus the test groups
ok["region_k_respected"]      <- region_folds(meta, meta$site, k = 3L)$n_folds == 3L

# -- 8. greedy dealing balances very uneven groups ----------------------------
#
# One group with 1000 members and nine with 10. Round-robin would put the big
# group with two small ones in some fold and leave others with 20 points;
# greedy must isolate it.
sizes <- c(1000L, rep(10L, 9L))
asg   <- .deal_groups(sizes, 3L)
loads <- vapply(1:3, function(j) sum(sizes[asg == j]), numeric(1))
ok["greedy_isolates_big_group"] <- sum(sizes[asg == asg[1]]) == 1000L
ok["greedy_balances_the_rest"] <- {
  small <- loads[-which.max(loads)]
  max(small) - min(small) <= 10L
}

# real data: folds should not differ wildly in size
sz <- check_fold_plan(sp)
ok["spatial_folds_balanced"] <-
  (max(sz$n_validation) / min(sz$n_validation)) < 2

# -- 9. reproducibility -------------------------------------------------------

ok["random_same_seed_same_plan"] <- identical(
  random_folds(meta, k = k, test_frac = test_frac, seed = 7L)$folds, rp$folds)
ok["random_diff_seed_diff_plan"] <- !identical(
  random_folds(meta, k = k, test_frac = test_frac, seed = 8L)$folds, rp$folds)
ok["spatial_same_seed_same_plan"] <- identical(
  spatial_folds(meta, k = k, test_frac = test_frac, block_size = 25000,
                seed = 7L)$folds, sp$folds)

# -- 10. fold construction does not consume the caller's RNG stream -----------
#
# If it did, changing k would silently change every model's weight
# initialisation, and two plans could never be compared on equal footing.

set.seed(99); before <- runif(3)
set.seed(99); invisible(random_folds(meta, k = 5L, seed = 123L)); after <- runif(3)
ok["random_folds_leaves_rng_alone"] <- identical(before, after)

set.seed(99); invisible(spatial_folds(meta, k = 5L, seed = 123L)); after2 <- runif(3)
ok["spatial_folds_leaves_rng_alone"] <- identical(before, after2)

# -- 11. bad arguments fail loudly --------------------------------------------

fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

ok["k_below_2_fails"]        <- fails(random_folds(meta, k = 1L))
ok["k_above_pool_fails"]     <- fails(random_folds(meta, k = 10000L))
ok["no_xy_fails"]            <- fails(spatial_folds(dplyr::select(meta, -x, -y)))
ok["huge_block_fails"]       <- fails(spatial_folds(meta, k = 4L,
                                                    block_size = 1e9))
ok["na_group_fails"]         <- fails(region_folds(meta, c(NA, meta$site[-1])))
ok["wrong_group_length_fails"] <- fails(region_folds(meta, meta$site[-1]))
ok["no_sample_id_fails"]     <- fails(random_folds(dplyr::select(meta,
                                                                -sample_id)))
ok["test_frac_1_fails"]      <- fails(random_folds(meta, k = 3L, test_frac = 1))

# -- 12. the buffer: what it guarantees, and what it costs -------------------

ok["buffer_recorded_in_params"] <- isTRUE(sp_buf$params$buffer == 2000)
ok["buffer_metric_is_chebyshev_by_default"] <-
  identical(sp_buf$params$buffer_metric, "chebyshev")

# THE GEOMETRY. Two square patches of width w share a pixel when the centres
# are within w-1 cells in BOTH axes -- a square condition. A circular buffer of
# radius w lets the diagonal through: at (14, 14) the distance is 14*sqrt(2) =
# 19.8, outside a circle of 15, and the patches still share the corner pixel.
# Measured on the real run: 0.51%-0.98% of validation points per fold still
# shared patch pixels while "same raster cell" reported a clean 0%.
tx <- c(0, 0); ty <- c(0, 0)
ok["euclidean_lets_the_diagonal_escape"] <-
  identical(.near_any(tx[1], ty[1], 14, 14, 15, "euclidean"), FALSE)
ok["chebyshev_catches_the_diagonal"] <-
  identical(.near_any(tx[1], ty[1], 14, 14, 15, "chebyshev"), TRUE)
ok["chebyshev_still_excludes_far_points"] <-
  identical(.near_any(tx[1], ty[1], 16, 16, 15, "chebyshev"), FALSE)
ok["buffer_cost_is_reported"]   <- is.data.frame(sp_buf$buffer_dropped) &&
  nrow(sp_buf$buffer_dropped) == k
ok["buffer_actually_dropped"]   <- sum(sp_buf$buffer_dropped$n_dropped) > 0L

# The buffer only ever REMOVES training points -- it must never touch the
# validation sets (that would change what is being measured) nor the test set.
ok["buffer_leaves_validation_alone"] <- all(vapply(seq_len(k), function(j)
  identical(sp_buf$folds[[j]]$validation, sp$folds[[j]]$validation),
  logical(1)))
ok["buffer_leaves_test_alone"] <- all(vapply(seq_len(k), function(j)
  identical(sp_buf$folds[[j]]$test, sp$folds[[j]]$test), logical(1)))
ok["buffer_only_removes"] <- all(vapply(seq_len(k), function(j)
  all(sp_buf$folds[[j]]$train %in% sp$folds[[j]]$train), logical(1)))

# Works on ANY plan, not only spatial ones: here on the fixed holdout, which
# has no notion of geography at all. A small buffer drops some neighbours
# without emptying anything.
hp_buf <- apply_buffer(hp, meta, 50)
ok["buffer_works_on_any_plan"] <- sum(hp_buf$buffer_dropped$n_dropped) > 0L
ok["buffer_on_holdout_keeps_training"] <-
  length(hp_buf$folds[[1]]$train) > 0L &&
  length(hp_buf$folds[[1]]$train) < length(hp$folds[[1]]$train)

# THE RESULT WORTH THE WHOLE FILE
#
# Buffering a RANDOM k-fold on clustered data removes every training point.
# Not a quirk: every cluster holds both training and validation points, so
# every training point has a validation neighbour, so the buffer takes all of
# them. Random cross-validation on clustered data cannot be repaired by
# buffering -- the honest version of it has no training data left. Blocking
# and buffering are not alternatives; blocking is what makes buffering
# affordable.
#
# And it has to STOP, loudly. An empty training set discovered five hours into
# a run is the same class of failure as the checks this project already exists
# to prevent.
ok["buffered_random_fold_is_impossible"] <- fails(apply_buffer(rp, meta, 2000))

cat("  random + 2 km buffer     : no training points left -> error (correct)\n")

# A buffer wider than the whole extent leaves nothing to train on, and that
# must be an error, not an empty training set discovered 5 hours later.
ok["absurd_buffer_fails"] <- fails(apply_buffer(sp, meta, 1e9))
ok["zero_buffer_is_a_no_op"] <- identical(apply_buffer(sp, meta, 0)$folds,
                                          sp$folds)

# .near_any(): exact, not merely bucketed. Two points 999 m apart are inside a
# 1 km buffer; 1001 m apart are outside -- and the bucket grid must not change
# that answer.
ok["near_any_is_exact"] <- identical(
  .near_any(c(0, 0), c(0, 0), c(999, 1001), c(0, 0), 1000),
  c(TRUE, TRUE)) &&
  identical(.near_any(c(0), c(0), c(1001), c(0), 1000), FALSE)

cat("  2 km buffer dropped      : ",
    format(sum(sp_buf$buffer_dropped$n_dropped), big.mark = ","),
    " training points (",
    sprintf("%.1f%%", mean(sp_buf$buffer_dropped$pct_dropped)),
    " per fold)\n", sep = "")

# auto block size must be flagged as a guess, not presented as a choice
ap <- spatial_folds(meta, k = 4L)
ok["auto_block_size_flagged"] <- isTRUE(ap$params$block_size_auto)
ok["explicit_block_not_flagged"] <- isFALSE(sp$params$block_size_auto)

# -- 13. summarise_resamples() / seed_noise_floor() ---------------------------
#
# Table arithmetic, so it can be checked against numbers computed by hand --
# and it decides which config wins, which makes it worth checking exactly.
#
# Fixture: config A is steadier but slightly worse on average; config B has one
# lucky seed that beats everything. Ranking single runs crowns B; ranking means
# crowns A. That is the winner's curse, in four rows.

cmp <- tibble::tibble(
  unit_id   = c("A_f1_s1", "A_f1_s2", "B_f1_s1", "B_f1_s2", "C_f1_s1"),
  config_id = c("A", "A", "B", "B", "C"),
  fold      = c(1L, 1L, 1L, 1L, 1L),
  seed      = c(42L, 43L, 42L, 43L, 42L),
  status    = c("success", "success", "success", "success", "failed"),
  val_ccc   = c(0.700, 0.720, 0.760, 0.600, NA_real_),
  val_mae   = c(5.0, 5.2, 4.8, 7.0, NA_real_)
)

sr <- summarise_resamples(cmp, metrics = c("val_ccc", "val_mae"))

ok["summary_one_row_per_config"] <- nrow(sr) == 2L   # C failed every unit
ok["summary_mean_is_right"] <- isTRUE(all.equal(
  sr$val_ccc_mean[sr$config_id == "A"], 0.710))
ok["summary_sd_is_right"] <- isTRUE(all.equal(
  sr$val_ccc_sd[sr$config_id == "B"], stats::sd(c(0.760, 0.600))))
ok["summary_se_is_sd_over_sqrt_n"] <- isTRUE(all.equal(
  sr$val_ccc_se[1], sr$val_ccc_sd[1] / sqrt(sr$n_units[1])))

# THE POINT: the mean ranks A first even though B holds the single best run.
ok["mean_beats_lucky_draw"] <- sr$config_id[sr$rank == 1L] == "A"
ok["lucky_single_run_would_win"] <-
  cmp$config_id[which.max(cmp$val_ccc)] == "B"

# Failed units are excluded from the statistics but never from view.
ok["failed_units_excluded_from_stats"] <- !any(is.na(sr$val_ccc_mean))
ok["failed_units_still_counted"] <-
  identical(sum(sr$n_units), 4L)

# A metric where lower is better must not be ranked as if higher were better;
# ranking follows the FIRST metric given, so asking for val_mae flips the order.
sr_mae <- summarise_resamples(cmp, metrics = c("val_mae", "val_ccc"))
ok["lower_is_better_ranks_ascending"] <- sr_mae$config_id[sr_mae$rank == 1L] == "A"

# One unit per config: sd and se are NA, because one run cannot estimate its
# own spread. Reporting 0 there would be a lie with a number on it.
one <- dplyr::filter(cmp, seed == 42L, status == "success")
sr1 <- summarise_resamples(one, metrics = "val_ccc")
ok["single_run_has_no_spread"] <- all(is.na(sr1$val_ccc_sd)) &&
  all(is.na(sr1$val_ccc_se))

# -- noise floor --------------------------------------------------------------

nf <- seed_noise_floor(cmp, metric = "val_ccc")
ok["noise_floor_uses_only_repeats"] <- nf$n_comparable == 2L

# Duas repeticoes cuja metrica veio NA nao sao duas repeticoes comparaveis.
# Contar a linha em vez do valor faria o relatorio anunciar cobertura que os
# dados nao tem -- pior do que dizer que nao da para estimar.
cmp_na <- cmp
cmp_na$val_ccc[cmp_na$config_id == "A"][2] <- NA_real_
nf_na <- seed_noise_floor(cmp_na, metric = "val_ccc")
ok["noise_floor_ignores_na_repeats"] <- nf_na$n_comparable == 1L
ok["noise_floor_sd_always_finite"]   <- is.finite(nf_na$median_sd)

# E duas linhas com a MESMA semente nao sao duas repeticoes: sao a mesma coisa
# contada duas vezes, e o sd entre elas sai 0 -- um piso de ruido inexistente,
# que faria qualquer diferenca entre configs parecer evidencia.
cmp_dup <- cmp
cmp_dup$seed[cmp_dup$config_id == "A"] <- 42L
nf_dup <- seed_noise_floor(cmp_dup, metric = "val_ccc")
ok["noise_floor_needs_distinct_seeds"] <-
  !("A" %in% nf_dup$by_config$config_id) && nf_dup$n_comparable == 1L
ok["noise_floor_median_sd"] <- isTRUE(all.equal(
  nf$median_sd, stats::median(c(stats::sd(c(0.700, 0.720)),
                                stats::sd(c(0.760, 0.600))))))
ok["noise_floor_max_range"] <- isTRUE(all.equal(nf$max_range, 0.160))

# With a single seed there is nothing to compare, and the report must say so
# rather than returning a confident zero.
nf1 <- seed_noise_floor(one, metric = "val_ccc")
ok["noise_floor_unknowable_with_one_seed"] <- nf1$n_comparable == 0L &&
  is.na(nf1$median_sd)

# Nenhuma das duas pode emitir aviso. Aviso em funcao de relatorio treina quem
# usa a ignorar avisos -- e o unico aviso que importa e sempre o proximo.
warns <- character(0)
withCallingHandlers({
  invisible(summarise_resamples(cmp, metrics = c("val_ccc", "val_mae")))
  invisible(summarise_resamples(one, metrics = "val_ccc"))
  invisible(seed_noise_floor(cmp, metric = "val_ccc"))
  invisible(seed_noise_floor(one, metric = "val_ccc"))
}, warning = function(w) {
  warns <<- c(warns, conditionMessage(w))
  invokeRestart("muffleWarning")
})
ok["aggregation_emits_no_warnings"] <- length(warns) == 0L

cat("  noise floor (fixture)    : median sd ",
    sprintf("%.3f", nf$median_sd), " | range ",
    sprintf("%.3f", nf$max_range),
    "  <- a gap smaller than this is not evidence\n", sep = "")

# -- 14. block_subsample() ----------------------------------------------------
#
# The property that matters is NOT "keeps 10% of points" -- it is "keeps whole
# blocks". A subsample that thins clusters makes the spatial machinery look
# healthier than it is, which is the one failure mode that would waste the
# whole development cycle it exists to speed up.

sub <- block_subsample(meta$x, meta$y, frac = 0.30, block_size = 25000,
                       seed = 3L)

ok["subsample_hits_the_target"] <- attr(sub, "frac_actual") >= 0.30 &&
  attr(sub, "frac_actual") < 0.45          # overshoots by at most one block
ok["subsample_keeps_whole_blocks"] <- {
  bx <- floor((meta$x - min(meta$x)) / 25000)
  by <- floor((meta$y - min(meta$y)) / 25000)
  blk <- paste(bx, by, sep = "_")
  kept <- unique(blk[sub])
  # every point of every kept block is present: no block is half in
  all(which(blk %in% kept) %in% sub)
}

# THE assertion: local density survives. Within the kept blocks, the mean
# number of neighbours under 1 km must be unchanged -- that is what a random
# subsample would destroy.
neigh_density <- function(rows) {
  xx <- meta$x[rows]; yy <- meta$y[rows]
  mean(vapply(seq_along(xx), function(i)
    sum((xx - xx[i])^2 + (yy - yy[i])^2 < 1000^2) - 1L, numeric(1)))
}
d_full  <- neigh_density(which(meta$sample_id %in% meta$sample_id))
d_block <- neigh_density(sub)
d_rand  <- neigh_density(with_local_seed(3L,
             sample(nrow(meta), length(sub))))

ok["block_subsample_preserves_density"] <- d_block > 0.8 * d_full
ok["random_subsample_destroys_density"] <- d_rand  < 0.6 * d_full

cat("  neighbours <1 km / point : full ", sprintf("%.1f", d_full),
    " | by block ", sprintf("%.1f", d_block),
    " | random ", sprintf("%.1f", d_rand), "
", sep = "")

ok["subsample_is_reproducible"] <- identical(
  as.integer(block_subsample(meta$x, meta$y, 0.30, 25000, seed = 3L)),
  as.integer(sub))
ok["subsample_frac_1_is_identity"] <- identical(
  as.integer(block_subsample(meta$x, meta$y, 1, 25000)), seq_len(nrow(meta)))
ok["subsample_describes_itself"] <- grepl("blocks of", describe_subsample(sub))

# It must not consume the caller's RNG stream either.
set.seed(7); b1 <- runif(3)
set.seed(7); invisible(block_subsample(meta$x, meta$y, 0.3, 25000, seed = 99L))
ok["subsample_leaves_rng_alone"] <- identical(b1, runif(3))

# -- 15. one_se(): the rule that exists because the ranking does not separate -
#
# Fixture built so the two rules DISAGREE, which is the only case worth
# asserting: cfg_B has the best mean, cfg_A is within one standard error of it
# and is far simpler. Ranking by the mean takes B; one_se takes A.

bc <- tibble::tibble(
  config_id    = c("cfg_B", "cfg_A", "cfg_C"),
  val_ccc_mean = c(0.530,   0.526,   0.480),
  val_ccc_sd   = c(0.025,   0.022,   0.028),
  val_ccc_se   = c(0.008,   0.007,   0.009),
  n_params     = c(11e6,    0.9e6,   0.4e6),
  n_units      = c(9L, 9L, 9L)
)

pick <- one_se(bc, complexity = "n_params")
ok["one_se_picks_the_simpler_tie"] <- pick$config_id == "cfg_A"
ok["one_se_counts_the_tied"]       <- attr(pick, "within_one_se") == 2L
ok["one_se_says_it_moved"]         <- isTRUE(attr(pick, "simpler_than_best"))
ok["one_se_band_is_best_minus_se"] <- isTRUE(all.equal(
  attr(pick, "band"), 0.530 - 0.008))

# cfg_C is 0.05 below the best -- far outside the band, and much simpler. The
# rule must NOT reach for it: "within one standard error" is the whole point.
ok["one_se_does_not_reach_outside"] <- pick$config_id != "cfg_C"

# When the best is also the simplest, the rule changes nothing -- and has to
# say so, or it looks like it did work it did not do.
bc2 <- bc
bc2$n_params <- c(0.4e6, 0.9e6, 11e6)
pick2 <- one_se(bc2, complexity = "n_params")
ok["one_se_can_agree_with_the_mean"] <- pick2$config_id == "cfg_B" &&
  !isTRUE(attr(pick2, "simpler_than_best"))

# For an error metric, lower is better, and the band opens upward.
bc3 <- dplyr::rename(bc, val_mae_mean = val_ccc_mean, val_mae_se = val_ccc_se)
bc3$val_mae_mean <- c(4.0, 4.005, 9.0)
bc3$val_mae_se   <- c(0.01, 0.01, 0.01)
pick3 <- one_se(bc3, metric = "val_mae", complexity = "n_params")
ok["one_se_handles_lower_is_better"] <- pick3$config_id == "cfg_A"

# "Simplest" is not something the framework may invent.
ok["one_se_demands_a_complexity"] <- inherits(
  try(one_se(dplyr::select(bc, -n_params)), silent = TRUE), "try-error")

# And it cannot tell a tie from a gap without repetitions.
bc4 <- bc; bc4$val_ccc_se <- NA_real_
ok["one_se_needs_repetitions"] <- inherits(
  try(one_se(bc4, complexity = "n_params"), silent = TRUE), "try-error")

cat("  one_se                   : the mean picks cfg_B (", bc$val_ccc_mean[1],
    "), one_se picks ", pick$config_id, " (",
    format(pick$n_params / 1e6, digits = 2), "M parameters against 11M)
",
    sep = "")

# =============================================================================
# block_share() and suggest_block_size()
#
# A block is indivisible: every point in it goes to the same fold. So one
# oversized block does not merely unbalance a plan, it DECIDES a fold -- and
# the fold is then scored on whatever that one landscape happens to be.
#
# The failure this guards against is specific and already happened: a
# block_size measured on the full point set was carried over to a 10%
# block-subsample, where the same size gives a tenth of the blocks at the SAME
# width, and a block that held 4.4% of the data held 34% of the draw.
#
# The fixture is built so the right answer is known by construction: one dense
# cluster inside a single degree, and a sparse spread around it. A large block
# swallows the cluster whole; a small one cuts it up.
# =============================================================================

set.seed(99)
bs_meta <- tibble::tibble(
  sample_id = seq_len(400L),
  # 300 points inside a 0.4 x 0.4 box, plus 100 spread over 20 x 20 degrees
  x = c(stats::runif(300L, 10.0, 10.4), stats::runif(100L, 0, 20)),
  y = c(stats::runif(300L, 10.0, 10.4), stats::runif(100L, 0, 20))
)

bshare <- block_share(bs_meta, c(0.1, 0.5, 5))
ok["block_share_one_row_per_size"] <- nrow(bshare) == 3L
ok["block_share_counts_blocks"]    <- all(bshare$n_blocks >= 1L)
# Bigger blocks can only merge, never split: the count falls, the worst share
# rises. Both are monotone, and a violation means the grid is not a grid.
ok["block_share_bigger_means_fewer"] <-
  all(diff(bshare$n_blocks) <= 0L)
ok["block_share_bigger_means_lumpier"] <-
  all(diff(bshare$largest_share) >= 0)
# A big block gathers the cluster; a small one cuts it into pieces.
#
# Asserted as a RELATIONSHIP, not as ">= 300". The block grid is anchored at
# min(x), which depends on the sparse points, so a cluster can straddle a
# boundary and be split between two blocks -- 185 and 115 here. That is not a
# defect: it is the same thing this file measures two blocks up as "sites cut
# by a boundary", and it is why the buffer exists. An absolute count would be
# asserting where the grid origin happens to land.
ok["block_share_gathers_the_cluster"] <-
  bshare$largest_n[bshare$block_size == 5] >
  4 * bshare$largest_n[bshare$block_size == 0.1]
# ...and even split, a 5-degree block holds far more than a random 1/12 share.
ok["block_share_big_block_beats_chance"] <-
  bshare$largest_share[bshare$block_size == 5] > 0.25

# suggest_block_size takes the LARGEST size that fits the constraint, because
# separation is the thing being bought.
sug <- suggest_block_size(bs_meta, k = 3L, max_share = 0.10,
                          min_blocks_per_fold = 5L,
                          candidates = c(0.1, 0.25, 0.5, 1, 5))
sug_tab <- attr(sug, "table")
fits <- sug_tab$block_size[sug_tab$largest_share <= 0.10 &
                           sug_tab$n_blocks >= 15L]
ok["suggest_returns_one_number"] <- length(as.numeric(sug)) == 1L
ok["suggest_respects_max_share"] <-
  sug_tab$largest_share[sug_tab$block_size == as.numeric(sug)] <= 0.10
ok["suggest_takes_the_largest_that_fits"] <-
  isTRUE(all.equal(as.numeric(sug), max(fits)))
ok["suggest_attaches_its_evidence"] <-
  is.data.frame(sug_tab) && nrow(sug_tab) == 5L

# When nothing fits it must SAY so, not return a number that meets no
# constraint and looks deliberate.
ok["suggest_warns_when_nothing_fits"] <- {
  w <- NULL
  withCallingHandlers(
    suggest_block_size(bs_meta, k = 3L, max_share = 0.001,
                       candidates = c(1, 5)),
    warning = function(cond) { w <<- conditionMessage(cond)
                               invokeRestart("muffleWarning") })
  is.character(w) && grepl("clustered", w)
}

# spatial_folds warns when the blocking it is HANDED lets one block decide a
# fold -- the case that was carried over silently.
ok["spatial_folds_warns_on_a_dominant_block"] <- {
  w <- NULL
  withCallingHandlers(
    spatial_folds(bs_meta, k = 3L, block_size = 5, buffer = NULL),
    warning = function(cond) { w <<- conditionMessage(cond)
                               invokeRestart("muffleWarning") })
  is.character(w) && grepl("decides a fold", w)
}
# ...and stays quiet when the blocking is balanced.
ok["spatial_folds_quiet_when_balanced"] <- {
  w <- NULL
  withCallingHandlers(
    spatial_folds(bs_meta, k = 3L, block_size = as.numeric(sug), buffer = NULL),
    warning = function(cond) { w <<- conditionMessage(cond)
                               invokeRestart("muffleWarning") })
  is.null(w)
}

cat("  block size               : cluster of 300 -> 5 deg gathers ",
    bshare$largest_n[bshare$block_size == 5], " into one block, 0.1 deg only ",
    bshare$largest_n[bshare$block_size == 0.1], "; suggested ",
    as.numeric(sug), "
", sep = "")

.report(ok, "test_resample")
