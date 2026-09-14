# Unit test: fold constructors (R/resample.R)
#
# A resampling plan is the one thing in the pipeline whose bugs do not look
# like bugs. A fold that leaks produces BETTER metrics, not an error -- the run
# finishes, the numbers improve, and nothing anywhere says the answer is wrong.
# So the properties have to be asserted mechanically, on data whose geography
# is known by construction.
#
# Verifies:
#   1. holdout() reproduces the dataset_role split exactly
#   2. every plan: train and validation disjoint, test in neither
#   3. k-fold plans partition the pool -- each row validated exactly ONCE
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

meta <- tibble::tibble(
  sample_id    = seq_len(n_site * n_per),
  profile_id   = seq_len(n_site * n_per),
  site         = rep(seq_len(n_site), each = n_per),
  x            = rep(site_x, each = n_per) + runif(n_site * n_per, -200, 200),
  y            = rep(site_y, each = n_per) + runif(n_site * n_per, -200, 200),
  dataset_role = "train"
)
# a fixed holdout on top, the way stage 01 writes it
meta$dataset_role[seq(2, nrow(meta), by = 7)]  <- "validation"
meta$dataset_role[seq(5, nrow(meta), by = 11)] <- "test"

pool_ids <- meta$sample_id[meta$dataset_role %in% c("train", "validation")]
test_idx <- which(meta$dataset_role == "test")

cat("  pontos sinteticos    : ", nrow(meta), " (", n_site, " sitios x ", n_per,
    ", sitios a 50 km)\n", sep = "")
cat("  pool / teste         : ", length(pool_ids), " / ", length(test_idx), "\n",
    sep = "")

# -- 1. holdout reproduces the fixed split ------------------------------------

hp <- holdout(meta)
ok["holdout_is_one_fold"]   <- hp$n_folds == 1L
ok["holdout_method_label"]  <- hp$method == "holdout"
ok["holdout_train_matches"] <- identical(
  sort(hp$folds[[1]]$train), sort(which(meta$dataset_role == "train")))
ok["holdout_val_matches"] <- identical(
  sort(hp$folds[[1]]$validation),
  sort(which(meta$dataset_role == "validation")))
ok["holdout_test_matches"] <- identical(sort(hp$folds[[1]]$test), sort(test_idx))

# -- plans under test ---------------------------------------------------------

k  <- 4L
rp <- random_folds(meta, k = k, seed = 7L)
sp <- spatial_folds(meta, k = k, block_size = 25000, seed = 7L)
gp <- region_folds(meta, group = meta$site)

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

  # the test rows are the SAME rows in every fold, and never train
  ok[paste0(nm, "_test_constant")] <- all(vapply(
    p$folds, function(f) identical(sort(f$test), sort(test_idx)), logical(1)))
  ok[paste0(nm, "_test_never_trains")] <- all(vapply(
    p$folds,
    function(f) length(intersect(f$test, c(f$train, f$validation))) == 0L,
    logical(1)))

  if (nm != "holdout") {
    # every pooled row validated exactly once across the k folds
    seen <- sort(unlist(lapply(p$folds, `[[`, "validation"), use.names = FALSE))
    ok[paste0(nm, "_partitions_pool")] <-
      identical(seen, sort(which(meta$sample_id %in% pool_ids)))
    # and trained on in exactly (folds - 1) of them. Uses the PLAN's own
    # fold count, not the k of this block: region_folds() defaults to
    # leave-one-group-out, so it has one fold per site, not k.
    trained <- table(unlist(lapply(p$folds, `[[`, "train"), use.names = FALSE))
    ok[paste0(nm, "_trained_k_minus_1")] <- all(trained == (p$n_folds - 1L))
  }
}

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

cat("  sitios cortados por borda: ", n_site_cut, " de ", n_site,
    "  <- por isso o buffer existe\n", sep = "")

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

sp_buf <- spatial_folds(meta, k = k, block_size = 25000, buffer = 2000,
                        seed = 7L)

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

cat("  vizinho <1 km no treino  : random ",
    sprintf("%.1f%%", 100 * share_random), " | spatial ",
    sprintf("%.1f%%", 100 * share_spatial), " | spatial+buffer ",
    sprintf("%.1f%%", 100 * share_buffered), "\n", sep = "")

# -- 7. region_folds ----------------------------------------------------------

grp_fold <- gp$assignment %>%
  dplyr::group_by(group) %>%
  dplyr::summarise(n_folds = dplyr::n_distinct(fold), .groups = "drop")
ok["region_group_not_split"] <- all(grp_fold$n_folds == 1L)
ok["region_logo_by_default"]  <- gp$n_folds == n_site
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
  random_folds(meta, k = k, seed = 7L)$folds, rp$folds)
ok["random_diff_seed_diff_plan"] <- !identical(
  random_folds(meta, k = k, seed = 8L)$folds, rp$folds)
ok["spatial_same_seed_same_plan"] <- identical(
  spatial_folds(meta, k = k, block_size = 25000, seed = 7L)$folds, sp$folds)

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
ok["no_dataset_role_fails"]  <- fails(random_folds(dplyr::select(meta,
                                                                -dataset_role)))

# -- 12. the buffer: what it guarantees, and what it costs -------------------

ok["buffer_recorded_in_params"] <- isTRUE(sp_buf$params$buffer == 2000)
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

cat("  random + buffer 2 km     : sem pontos de treino -> erro (correto)\n")

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

cat("  buffer 2 km descartou    : ",
    format(sum(sp_buf$buffer_dropped$n_dropped), big.mark = ","),
    " pontos de treino (",
    sprintf("%.1f%%", mean(sp_buf$buffer_dropped$pct_dropped)),
    " por fold)\n", sep = "")

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

cat("  piso de ruido (fixture)  : sd mediano ",
    sprintf("%.3f", nf$median_sd), " | amplitude ",
    sprintf("%.3f", nf$max_range),
    "  <- diferenca menor que isso nao e evidencia\n", sep = "")

.report(ok, "test_resample")
