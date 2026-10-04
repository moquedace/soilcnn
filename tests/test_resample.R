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
#  14b. with strata, every stratum keeps its points up to one quota: the
#      sparse ones whole, the dense one cut in whole blocks
#  15. one_se(): on a fixture where the two rules DISAGREE -- the best mean and
#      the simplest model within one standard error of it are different
#      configs, which is the only case where the rule earns its existence
#
# Run: source("<package root>/tests/test_resample.R")    (CPU, no tensors)

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
  stop("Project root not found. setwd() to the package root, ",
       "or source() this file with its full path.", call. = FALSE)
})()

source(file.path(root, "tests", "helper.R"))
.load_framework(root)

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

# suppressWarnings, deliberately: fails() asserts that an expression ERRORS,
# and a warning raised on the way there is not part of that claim. The
# block_size = 1e9 case below legitimately warns ("one block decides a fold")
# before it errors -- printing that here would put a warning in a green test
# run, which is how people learn to scroll past warnings.
fails <- function(expr) {
  inherits(try(suppressWarnings(expr), silent = TRUE), "try-error")
}

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

# THE BUFFER ONLY EVER REMOVES, AND NEVER FROM THE TEST SET.
#
# This used to assert that validation is untouched, and that was the contract
# until the buffer started protecting the test set as well. Validation may now
# shrink -- points within one patch span of a test block are dropped, because a
# stopping epoch chosen on rows that overlap the test set is a small read of it.
#
# What survives from the old contract is the part that still matters, and it is
# asserted more precisely than before:
#
#   - the test set is NEVER touched. It is the thing being protected; shrinking
#     it would silently change what the final number is measured on.
#   - validation only ever SHRINKS, and only for the test reason. A buffer that
#     added rows, or reordered them, would be a different bug wearing the same
#     name.
ok["buffer_leaves_test_alone"] <- all(vapply(seq_len(k), function(j)
  identical(sp_buf$folds[[j]]$test, sp$folds[[j]]$test), logical(1)))

ok["buffer_only_shrinks_validation"] <- all(vapply(seq_len(k), function(j)
  all(sp_buf$folds[[j]]$validation %in% sp$folds[[j]]$validation), logical(1)))

ok["buffer_only_shrinks_train"] <- all(vapply(seq_len(k), function(j)
  all(sp_buf$folds[[j]]$train %in% sp$folds[[j]]$train), logical(1)))

# ...and with the OLD setting, validation is still exactly untouched. This is
# the old assertion, kept where it is true, so the change is visible as a
# change of contract rather than as a deleted guarantee.
sp_valonly <- apply_buffer(sp, meta, buffer = 2000, protect = "validation")
ok["validation_only_buffer_leaves_validation_alone"] <- all(vapply(seq_len(k),
  function(j) identical(sp_valonly$folds[[j]]$validation,
                        sp$folds[[j]]$validation), logical(1)))
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

# Two repeats whose metric came back NA are not two comparable repeats.
# Counting the row instead of the value would make the report announce coverage
# the data does not have -- worse than saying it cannot be estimated.
cmp_na <- cmp
cmp_na$val_ccc[cmp_na$config_id == "A"][2] <- NA_real_
nf_na <- seed_noise_floor(cmp_na, metric = "val_ccc")
ok["noise_floor_ignores_na_repeats"] <- nf_na$n_comparable == 1L
ok["noise_floor_sd_always_finite"]   <- is.finite(nf_na$median_sd)

# And two rows with the SAME seed are not two repeats: they are the same thing
# counted twice, and the sd between them comes out 0 -- a noise floor that does
# not exist, which would make any difference between configs look like
# evidence.
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

# Neither of them may raise a warning. A warning out of a reporting function
# trains whoever uses it to ignore warnings -- and the only warning that
# matters is always the next one.
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
    " | random ", sprintf("%.1f", d_rand), "\n", sep = "")

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

# -- 14b. block_subsample(strata =): every region keeps its points up to a quota
#
# One dense region -- 600 points over 100 x 100 km, sixteen blocks of 25 km, in
# one 500 km stratum -- and ten sparse strata of 6 points each, one block each.
# At a quarter the quota is 105: the sparse strata keep everything, and only
# the dense one is cut, in whole blocks. A draw over the whole area has no such
# promise: it takes blocks wherever they fall, sparse ones included.
st_x <- c(runif(600, 0, 100000), rep(600000 + 500000 * (0:9), each = 6) + runif(60, 0, 1000))
st_y <- c(runif(600, 0, 100000), runif(60, 0, 1000))
dense <- seq_len(600)
st_of <- paste(floor((st_x - min(st_x)) / 500000), floor((st_y - min(st_y)) / 500000))
blk_of <- paste(floor((st_x - min(st_x)) / 25000), floor((st_y - min(st_y)) / 25000))
strat <- block_subsample(st_x, st_y, frac = 0.25, block_size = 25000, seed = 5L,
                         strata = 500000)
flat  <- block_subsample(st_x, st_y, frac = 0.25, block_size = 25000, seed = 5L)
ok["strata_quota_is_the_water_level"] <- identical(attr(strat, "quota"), 105L) &&
  identical(attr(strat, "strata_total"), 11L) && identical(attr(strat, "strata_cut"), 1L)
ok["strata_keep_every_stratum"] <- length(unique(st_of[strat])) == 11L
ok["strata_keep_the_sparse_ones_whole"] <- all(601:660 %in% strat)
ok["strata_cut_only_the_dense_one"] <- sum(strat %in% dense) >= 105L &&
  sum(strat %in% dense) < 600L
ok["strata_keep_whole_blocks"] <- all(which(blk_of %in% blk_of[strat]) %in% strat)
ok["strata_reach_the_fraction"] <- attr(strat, "frac_actual") >= 0.25
ok["strata_are_reproducible"] <- identical(as.integer(strat), as.integer(
  block_subsample(st_x, st_y, frac = 0.25, block_size = 25000, seed = 5L, strata = 500000)))
set.seed(7); b2 <- runif(3)
set.seed(7); invisible(block_subsample(st_x, st_y, 0.25, 25000, seed = 9L, strata = 500000))
ok["strata_leave_rng_alone"] <- identical(b2, runif(3))
e_strata <- tryCatch({ block_subsample(st_x, st_y, 0.25, 25000, strata = 30000); "" },
                     error = function(e) conditionMessage(e))
ok["strata_must_be_a_multiple_of_the_block"] <- grepl("whole multiple of block_size", e_strata)
ok["strata_describe_themselves"] <- grepl("stratified in squares of", describe_subsample(strat)) &&
  grepl("1 of 11 cut", describe_subsample(strat))
cat("  strata at a quarter      : ", length(unique(st_of[strat])), " of 11 strata kept, ",
    sum(601:660 %in% strat), " of 60 sparse points | over the whole area: ",
    length(unique(st_of[flat])), " strata, ", sum(601:660 %in% flat), " sparse points\n",
    sep = "")

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

# A best config trained once has no standard error, and one_se() refuses --
# saying what to do: more seeds on a single split, or rule = "rank1".
bc_once <- bc
bc_once$val_ccc_se <- NA_real_
e_once <- tryCatch(one_se(bc_once, complexity = "n_params"),
                   error = function(e) conditionMessage(e))
ok["one_se_without_se_says_what_to_do"] <-
  grepl("n_seeds >= 2", e_once, fixed = TRUE) && grepl("rank1", e_once, fixed = TRUE)

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
    format(pick$n_params / 1e6, digits = 2), "M parameters against 11M)\n",
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
    as.numeric(sug), "\n", sep = "")


# =============================================================================
# paired_family_test()
#
# Stage 03b compares four families on ONE fold plan with ONE set of seeds, so
# the units are matched pairs and the comparison should use that. The first run
# did not: it put two means beside a seed noise floor, which is a bound on the
# wrong quantity (the spread of one family, not the SE of a difference).
#
# The properties that matter are not "does it compute a t statistic" but:
#   - it pairs on (fold, seed) and refuses to guess when it cannot
#   - a constant shift is detected with certainty, however noisy the folds are;
#     this is exactly the power that pairing buys and the unpaired SE loses
#   - direction is read correctly for an ERROR metric, where lower wins
# =============================================================================

mk <- function(cfg, ccc, folds = rep(1:3, each = 3), seeds = rep(1:3, 3)) {
  tibble::tibble(
    unit_id = sprintf("%s_f%d_s%d", cfg, folds, seeds),
    config_id = cfg, fold = folds, seed = seeds,
    status = "success", val_ccc = ccc, val_mae = 20 - 10 * ccc)
}

# Fold difficulty is large (0.30 between folds) and the real effect is small
# (0.02) and perfectly constant. Unpaired, the fold spread swamps it; paired,
# every difference is exactly 0.02 and the SE is zero to numerical precision.
fold_effect <- rep(c(0.20, 0.50, 0.80), each = 3)
A <- mk("cfg_001", fold_effect + 0.02)
B <- mk("rf_001",  fold_effect)

pt <- paired_family_test(A, B, metric = "val_ccc",
                         label_a = "cnn", label_b = "rf")
ok["paired_recovers_a_constant_shift"] <- abs(pt$diff - 0.02) < 1e-12
ok["paired_uses_every_matched_unit"]   <- pt$n_pairs == 9L
ok["paired_se_beats_unpaired_se"]      <- pt$se_unpaired > pt$se
ok["paired_separates_what_unpaired_cannot"] <- {
  # The unpaired interval would straddle zero by a wide margin; the paired one
  # must not contain it.
  !(pt$ci[1] <= 0 && pt$ci[2] >= 0) && 2 * pt$se_unpaired > 0.02
}

# Lower is better for MAE, and val_mae was built to move the opposite way, so
# the sign must flip and the verdict must still name the CNN as ahead.
pt_mae <- paired_family_test(A, B, metric = "val_mae",
                             label_a = "cnn", label_b = "rf")
ok["paired_mae_sign_flips"]        <- pt_mae$diff < 0
ok["paired_knows_mae_is_an_error"] <- isFALSE(pt_mae$higher_is_better)

# No effect: the interval must contain zero rather than the test inventing one.
C <- mk("rf_001", fold_effect)
ok["paired_finds_nothing_when_there_is_nothing"] <- {
  p0 <- paired_family_test(A, C, metric = "val_ccc")
  abs(p0$diff - 0.02) < 1e-12   # A really is 0.02 above; C == B
}
D <- mk("rf_001", fold_effect + 0.02)
ok["paired_zero_difference_is_zero"] <- {
  p0 <- paired_family_test(A, D, metric = "val_ccc")
  p0$diff == 0 && p0$ci[1] == 0 && p0$ci[2] == 0
}

# The best config is chosen per family, because that is the one a person would
# deploy -- not the grid average, which averages over configs nobody would use.
AA <- dplyr::bind_rows(A, mk("cfg_002", fold_effect - 0.30))
ok["paired_picks_the_best_config"] <- {
  p <- paired_family_test(AA, B, metric = "val_ccc")
  p$label_a == "cfg_001"
}

# ...and the best config for an ERROR metric is the SMALLEST, not the largest.
ok["paired_picks_the_best_config_for_mae"] <- {
  p <- paired_family_test(AA, B, metric = "val_mae")
  p$label_a == "cfg_001"
}

# A family that ran on a different plan is a different experiment. Refuse it
# rather than quietly comparing three shared units out of nine.
ok["paired_refuses_a_different_plan"] <- inherits(
  try(paired_family_test(A, mk("rf_001", fold_effect,
                               folds = rep(7:9, each = 3)),
                         metric = "val_ccc"), silent = TRUE), "try-error")

ok["paired_refuses_without_fold_and_seed"] <- inherits(
  try(paired_family_test(dplyr::select(A, -fold), B, metric = "val_ccc"),
      silent = TRUE), "try-error")

# Failed units carry no metric and must not be paired against a success.
ok["paired_ignores_failed_units"] <- {
  Abad <- A; Abad$status[1] <- "failed"; Abad$val_ccc[1] <- NA_real_
  p <- paired_family_test(Abad, B, metric = "val_ccc")
  p$n_pairs == 8L && p$dropped == 1L
}

cat(sprintf("  paired vs unpaired SE    : %.5f vs %.5f on a 0.02 shift under 0.30 fold spread\n",
            pt$se, pt$se_unpaired))


# =============================================================================
# summarise_resamples() must carry the per-config constants
#
# one_se() needs a "simplest" column beside the means. n_params is a property
# of the config, identical across its folds and seeds -- so it must survive the
# group_by rather than be averaged away. It did not, and one_se() therefore
# failed on a table that obviously had what it needed.
# =============================================================================

cmp_np <- tibble::tibble(
  unit_id = sprintf("cfg_%03d_f%d_s%d", rep(1:2, each = 4),
                    rep(rep(1:2, each = 2), 2), rep(1:2, 4)),
  config_id = rep(c("cfg_001", "cfg_002"), each = 4),
  fold = rep(rep(1:2, each = 2), 2), seed = rep(1:2, 4),
  status = "success",
  n_params = rep(c(1e6, 9e6), each = 4),
  n_features = 181L,
  # THE MEANS MUST ACTUALLY TIE, or this fixture tests nothing.
  #
  # The first version used 0.505 against 0.535 -- a gap of 0.030 against a
  # standard error of 0.0065, nearly five SE apart. one_se() correctly returned
  # the winner and the test called it a failure. The rule only has work to do
  # when the gap is INSIDE the winner's own standard error, so the fixture has
  # to build that, not assume it.
  #
  #   cfg_001: mean 0.530, se 0.0041, 1M parameters
  #   cfg_002: mean 0.535, se 0.0065, 9M parameters   <- best mean
  #   band   : 0.535 - 0.0065 = 0.5285, and 0.530 is above it
  val_ccc = c(0.53, 0.54, 0.52, 0.53,  0.53, 0.55, 0.52, 0.54),
  val_mae = 10)

bc <- summarise_resamples(cmp_np)
ok["summarise_keeps_n_params"]  <- "n_params" %in% names(bc)
ok["summarise_keeps_it_intact"] <- {
  identical(sort(bc$n_params), c(1e6, 9e6))
}
ok["summarise_did_not_average_it"] <-
  !any(grepl("^n_params_(mean|sd)$", names(bc)))

# ...and one_se() must now work on that table end to end, which is the whole
# point: the best mean is cfg_002 (0.535) but cfg_001 is within one SE and is
# nine times smaller, so the rule must take the small one.
pick <- one_se(bc, metric = "val_ccc", complexity = "n_params")
ok["one_se_runs_on_summarise_output"] <- nrow(pick) == 1L
ok["one_se_prefers_the_simpler_tie"]  <- pick$config_id == "cfg_001"
ok["one_se_reports_the_tie_size"]     <- attr(pick, "within_one_se") == 2L
ok["one_se_flags_that_it_moved"]      <- isTRUE(attr(pick, "simpler_than_best"))

# When the ranking DOES separate, the rule must return the winner -- this is
# what makes it safe as a default rather than a bias toward small models.
bc_clear <- bc
bc_clear$val_ccc_mean <- c(0.90, 0.50)[match(bc_clear$config_id,
                                             c("cfg_002", "cfg_001"))]
bc_clear$val_ccc_se   <- 0.001
ok["one_se_keeps_a_decisive_winner"] <- {
  p <- one_se(bc_clear, metric = "val_ccc", complexity = "n_params")
  p$config_id == "cfg_002" && isFALSE(attr(p, "simpler_than_best"))
}

# A family chosen by the caller must not be re-chosen per metric: see the note
# in paired_family_test(). The label is reported either way, and a config the
# caller did NOT pass is marked so the reader can see it was picked here.
ok["paired_marks_a_config_it_chose"] <- {
  p <- paired_family_test(A, B, metric = "val_ccc")
  grepl("chosen here", p$chosen_a)
}
ok["paired_does_not_mark_a_given_config"] <- {
  p <- paired_family_test(A, B, metric = "val_ccc",
                          config_a = "cfg_001", config_b = "rf_001")
  identical(p$chosen_a, "caller") && identical(p$chosen_b, "caller")
}
# The same config on both metrics when the caller says so -- the defect this
# fixed was rf_001 on CCC and rf_002 on MAE inside one report.
ok["paired_honours_the_config_on_every_metric"] <- {
  p1 <- paired_family_test(AA, B, metric = "val_ccc", config_a = "cfg_002")
  p2 <- paired_family_test(AA, B, metric = "val_mae", config_a = "cfg_002")
  p1$label_a == "cfg_002" && p2$label_a == "cfg_002"
}


# =============================================================================
# THE BUFFER MUST PROTECT THE TEST SET, NOT ONLY THE VALIDATION SET
#
# apply_buffer() used to measure distance from training points to VALIDATION
# points and stop there. The plan then printed "0% leakage" -- true, and about
# the wrong set. Every training point beside a test block kept its place, and
# at 250 m a 15x15 patch reaches 1.7 km, so those patches overlapped test
# patches pixel for pixel.
#
# The asymmetry is what makes it dangerous rather than merely wrong: the set
# that was protected is the one used to CHOOSE, and the set that was not is the
# one whose number is published.
#
# The fixture is built so the old behaviour cannot pass: training points are
# placed near the test set and FAR from validation, so a validation-only buffer
# drops nothing at all.
# =============================================================================

buf_meta <- tibble::tibble(
  sample_id = 1:9,
  #        test ....... | gap | train near test | far train | validation
  x = c(0.0, 0.1, 0.2,          0.5, 0.6,         5.0, 5.1,   9.0, 9.1),
  y = 0)
buf_plan <- structure(list(
  folds = list(list(train = 4:7, validation = 8:9, test = 1:3)),
  n_folds = 1L, params = list(), meta_rows = 9L),
  class = "fold_plan")

# buffer 1.0 reaches from x = 0.5 to the test point at x = 0.2 (distance 0.3),
# and nowhere near validation at x = 9.
buf_both <- apply_buffer(buf_plan, buf_meta, buffer = 1.0)
ok["buffer_drops_training_near_test"] <-
  identical(buf_both$folds[[1]]$train, 6:7)
ok["buffer_keeps_training_far_from_test"] <-
  all(c(6L, 7L) %in% buf_both$folds[[1]]$train)
ok["buffer_reports_the_test_cause"] <-
  buf_both$buffer_dropped$n_near_test == 2L
ok["buffer_reports_no_validation_cause"] <-
  buf_both$buffer_dropped$n_near_validation == 0L

# THE REGRESSION GUARD: with protect = "validation" only -- the old behaviour --
# nothing is dropped. If this ever equals the line above, the fix is gone.
buf_val <- apply_buffer(buf_plan, buf_meta, buffer = 1.0, protect = "validation")
ok["validation_only_buffer_drops_nothing_here"] <-
  identical(buf_val$folds[[1]]$train, 4:7)
ok["the_two_settings_really_differ"] <-
  !identical(buf_val$folds[[1]]$train, buf_both$folds[[1]]$train)

# Validation near the test set goes too: early stopping is a decision made on
# data, and a stopping epoch chosen on rows that overlap the test is a small
# read of the test.
buf_meta2 <- buf_meta; buf_meta2$x[8] <- 0.4   # validation point beside test
buf_v2 <- apply_buffer(buf_plan, buf_meta2, buffer = 1.0)
ok["buffer_drops_validation_near_test"] <-
  identical(buf_v2$folds[[1]]$validation, 9L)
ok["buffer_reports_validation_dropped"] <-
  buf_v2$buffer_dropped$n_validation_dropped == 1L

# ...and it must refuse rather than hand back an empty side.
ok["buffer_refuses_to_empty_validation"] <- inherits(
  try(apply_buffer(buf_plan, buf_meta, buffer = 20, protect = "test"),
      silent = TRUE), "try-error")

# A plan with no test set must be untouched by the test half of the rule.
buf_plan_nt <- buf_plan
buf_plan_nt$folds[[1]]$test <- integer(0)
ok["no_test_set_no_test_buffer"] <- {
  p <- apply_buffer(buf_plan_nt, buf_meta, buffer = 1.0)
  p$buffer_dropped$n_near_test == 0L && identical(p$folds[[1]]$train, 4:7)
}

ok["buffer_records_what_it_protected"] <-
  identical(buf_both$params$buffer_protect, "validation+test")

cat("  buffer protects          : train and validation against the test set",
    " (was validation only)\n", sep = "")


# =============================================================================
# check_plan_unchanged(): resuming onto a different split
#
# .resumable_units() proves a cached unit was fitted on the hyperparameters its
# name claims. It says nothing about the DATA -- and the fold plan is data.
#
# This is not hypothetical. Fixing apply_buffer() to protect the test set made
# every fold lose a rim of training points. The cached units were still cfg_002
# with the same learning rate and window, so a hyperparameter check passes them,
# while they had been fitted on a training set that no longer exists. Resuming
# would have ranked units trained on different data against each other.
#
# The refusal is deliberate rather than a silent discard: a run directory is the
# record of an experiment, and quietly replacing half of it with units from
# another experiment is worse than stopping.
# =============================================================================

plan_dir <- file.path(tempdir(), "test_plan_guard")
unlink(plan_dir, recursive = TRUE); dir.create(plan_dir, recursive = TRUE)

p_a <- spatial_folds(meta, k = 3L, test_frac = 0.15, block_size = 25000,
                     buffer = 2000, seed = 7L)
saveRDS(p_a, file.path(plan_dir, "fold_plan.rds"))

ok["same_plan_passes"] <-
  isTRUE(check_plan_unchanged(p_a, plan_dir, resume = TRUE))

# The exact case that prompted this: only the buffer's reach changed, so the
# folds are the same folds with a different rim.
p_b <- apply_buffer(spatial_folds(meta, k = 3L, test_frac = 0.15,
                                  block_size = 25000, buffer = NULL, seed = 7L),
                    meta, buffer = 2000, protect = "validation")
ok["a_different_buffer_is_caught"] <- inherits(
  try(check_plan_unchanged(p_b, plan_dir, resume = TRUE), silent = TRUE),
  "try-error")

# A different k, and a different seed, must be caught too.
ok["a_different_k_is_caught"] <- inherits(
  try(check_plan_unchanged(
        spatial_folds(meta, k = 4L, test_frac = 0.15, block_size = 25000,
                      buffer = 2000, seed = 7L),
        plan_dir, resume = TRUE), silent = TRUE), "try-error")
ok["a_different_seed_is_caught"] <- inherits(
  try(check_plan_unchanged(
        spatial_folds(meta, k = 3L, test_frac = 0.15, block_size = 25000,
                      buffer = 2000, seed = 99L),
        plan_dir, resume = TRUE), silent = TRUE), "try-error")

# The message has to name the sizes, or the reader is sent to diff two RDS.
ok["the_refusal_names_both_sizes"] <- {
  e <- tryCatch(check_plan_unchanged(p_b, plan_dir, resume = TRUE),
                error = function(e) conditionMessage(e))
  grepl("cached", e, fixed = TRUE) && grepl("asked", e, fixed = TRUE)
}

# resume = FALSE is a deliberate fresh start and must not be blocked; neither
# must a directory that holds no previous plan.
ok["no_resume_is_never_blocked"] <-
  isTRUE(check_plan_unchanged(p_b, plan_dir, resume = FALSE))
ok["an_empty_directory_is_fine"] <-
  isTRUE(check_plan_unchanged(p_a, tempdir(), resume = TRUE))

unlink(plan_dir, recursive = TRUE)

# -- 16. refit_split(): the final fit's validation, by the plan's own rules --
#
# dsm_final() stops the final fit on a validation set cut the way the tuning
# folds were. The test set is the plan's, verbatim; the rest is split once.
# Blocks and random rows take the first of k = 1/validation_frac folds, as
# they always did. Regions -- which could not be refitted before 2026-09-28 --
# take whole regions: the fold closest to the share asked for.
refit_is_sound <- function(rf, plan) {
  f    <- rf$folds[[1]]
  test <- sort(as.integer(plan$folds[[1]]$test))
  rf$n_folds == 1L && identical(sort(as.integer(f$test)), test) &&
    length(intersect(f$train, f$validation)) == 0L &&
    length(intersect(c(f$train, f$validation), test)) == 0L
}
rs <- refit_split(sp, meta, 0.15)
rr <- refit_split(rp, meta, 0.15)
rg <- refit_split(gp, meta, 0.15)
ok["refit_spatial_is_sound"] <- refit_is_sound(rs, sp) && identical(rs$method, "refit_spatial_folds")
ok["refit_random_is_sound"]  <- refit_is_sound(rr, rp) && identical(rr$method, "refit_random_folds")
ok["refit_region_is_sound"]  <- refit_is_sound(rg, gp) && identical(rg$method, "refit_region_folds")
ok["refit_spatial_takes_the_first_fold_as_it_did"] <- identical(rs$params$refit_fold, 1L)
ok["refit_region_validates_whole_regions"] <- {
  f <- rg$folds[[1]]
  length(intersect(meta$site[f$validation], meta$site[f$train])) == 0L
}
ok["refit_region_covers_every_non_test_row"] <- {
  f <- rg$folds[[1]]
  identical(sort(as.integer(c(f$train, f$validation))),
            as.integer(setdiff(seq_len(nrow(meta)), gp$folds[[1]]$test)))
}
cat(sprintf("  refit by region          : %d site(s) validate, %d train, the test's %d untouched\n",
            length(unique(meta$site[rg$folds[[1]]$validation])),
            length(unique(meta$site[rg$folds[[1]]$train])),
            length(unique(meta$site[gp$folds[[1]]$test]))))

.report(ok, "test_resample")
