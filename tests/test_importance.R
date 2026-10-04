# Unit test: variable importance measures what it claims to measure
#
# WHY THIS FILE EXISTS.
#
# An importance table is read as a finding about the soil, and nobody checks a
# finding that agrees with what they expected. So the end-to-end checks here
# use models whose answer is known by construction, as tests/test_occlusion.R
# does: a model that reads two channels and ignores three must give the three
# exactly zero -- not small, zero -- and the two a loss; a model fed a regional
# gradient must lose it when the donors stay in their block. If the mask, the
# donors, the batching, the windows or the scores are wired wrongly, one of
# these comes out wrong.
#
# Verified:
#   1. the one-hot sets: found by prefix AND exclusivity at the points; a
#      shared first token that is two categoricals splits; a dummy named as
#      the prefix stays alone; a set never takes a channel's name
#   2. a grouping of the user's: replaces the rule; what it cannot mean is
#      refused; a set it cuts in two is said
#   3. the draws: a permutation with no fixed point, within each stratum; a
#      row alone keeps itself; the seed reproduces them
#   4. end to end, against known models: zero for what the model ignores; the
#      same donor in every window; a group with an unused channel is the used
#      channel alone; the mean fill gives the predicted change exactly; a
#      regional gradient loses its importance within blocks
#   5. from draws and models to the table: the sign, the mean, the spread
#   6. side by side: rankings aligned, agreement measured
#   7. the method refuses what cannot be meant
#   8. context, against known models: a centre reader loses only its centre, a
#      ring reader only its ring; per variable, each variable only where it is
#      read; a ring is the same ground in every window; a window the model
#      does not read costs nothing; the gate is read point by point
#   9. SHAP, against attributions known in closed form: a linear model's, for
#      both estimators; a square's by integrated gradients; zero for what is
#      not read; a group is the sum of its channels; every window counted;
#      the ring read is where the attribution sits; a real two-branch network
#      adds up; and attributions that do not add up stop
#  10. ALE, against effects known in closed form: a linear model's curve rises
#      by its weight times each bin's width, a square's by the squares' step;
#      what is not read is flat, exactly; the whole patch moves, in every
#      window; the curve is centred; a class's effect is the class difference
#  11. the importance as the AOA's weights: each channel its variable's, below
#      zero none, and a map takes them as they come or refuses them
#  12. the kernel, against Shapley values known in closed form: a linear
#      model's, a product's interaction split; nothing for what is not read;
#      exact sums; permutations exact for a linear model; one background point
#      in every window; and two SHAP importances compared point by point
#  13. SAGE, against Shapley values of the loss known in closed form: two
#      variables that carry one signal split it; nothing for what is not read;
#      exact sums; paired permutations exact for two variables; the loss asked
#      for
#
# Run: source("<package root>/tests/test_importance.R")

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)
})

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
  stop("Project root not found.", call. = FALSE)
})()
source(file.path(root, "tests", "helper.R"))
.load_framework(root)

ok <- logical(0)
err <- function(expr) {
  e <- tryCatch({ suppressMessages(expr); NULL }, error = function(e) conditionMessage(e))
  if (is.null(e)) "" else e
}
warned <- function(expr) {
  w <- NULL
  withCallingHandlers(suppressMessages(expr), warning = function(c) {
    w <<- c(w, conditionMessage(c)); invokeRestart("muffleWarning")
  })
  paste(w, collapse = " | ")
}

# ── 1. the one-hot sets ───────────────────────────────────────────────────────
#
# A store with every case the inference must get right: a categorical of three
# classes; two categoricals that share their first name token (a point has a
# soil class AND a drainage class, so together they are not exclusive); a
# binary map named like the prefix of a set beside it; a lone binary map; and a
# continuous channel.
set.seed(3)
n1 <- 40L
one_hot <- function(k) { m <- matrix(0, n1, k); m[cbind(seq_len(n1), sample.int(k, n1, TRUE))] <- 1; m }
lith  <- one_hot(3L); cls <- one_hot(2L); drn <- one_hot(2L)
wetx  <- matrix(0, n1, 2L)
wetx[cbind(seq_len(n1), sample.int(2L, n1, TRUE))] <- rbinom(n1, 1, 0.5)   # at most one of the two
pts1 <- tibble::tibble(clay = runif(n1),
                       lith_acid = lith[, 1], lith_basic = lith[, 2], lith_carbonate = lith[, 3],
                       soil_class_a = cls[, 1], soil_class_b = cls[, 2],
                       soil_drain_good = drn[, 1], soil_drain_poor = drn[, 2],
                       wet = rbinom(n1, 1, 0.3), wet_inland = wetx[, 1], wet_coastal = wetx[, 2],
                       flood = rbinom(n1, 1, 0.2))
ch1 <- names(pts1)
tt1 <- tibble::tibble(predictor = ch1, is_dummy = ch1 != "clay", is_percentage = FALSE)
fake <- list(store = list(predictors = ch1), points = pts1, type_table = tt1)

g <- importance_groups(fake)
var_of <- function(gr, ch) gr$variable[match(ch, gr$channel)]
ok["groups_follow_the_channel_order"] <- identical(g$channel, ch1)
ok["a_categorical_is_one_variable"] <-
  all(var_of(g, c("lith_acid", "lith_basic", "lith_carbonate")) == "lith")
ok["two_categoricals_sharing_a_token_are_two"] <-
  all(var_of(g, c("soil_class_a", "soil_class_b")) == "soil_class") &&
  all(var_of(g, c("soil_drain_good", "soil_drain_poor")) == "soil_drain")
ok["a_dummy_named_as_the_prefix_stays_alone"] <- var_of(g, "wet") == "wet"
ok["a_set_never_takes_a_channels_name"] <-
  all(var_of(g, c("wet_inland", "wet_coastal")) == "wet_classes")
ok["a_lone_binary_and_a_continuous_stay_alone"] <-
  var_of(g, "flood") == "flood" && var_of(g, "clay") == "clay"
ok["the_rule_is_said"] <- all(g$rule[g$channel %in% c("lith_acid", "soil_class_b")] == "one-hot set") &&
  g$rule[g$channel == "clay"] == "alone"
ok["channel_puts_every_channel_alone"] <- {
  gc_ <- importance_groups(fake, "channel")
  identical(gc_$variable, ch1) && all(gc_$rule == "alone")
}
# NOT EXCLUSIVE, NOT A SET. The same names over values where two "classes" are
# 1 together: the inference must believe the data over the names.
both <- fake
both$points$lith_basic[1:5] <- 1; both$points$lith_acid[1:5] <- 1
ok["names_alone_do_not_make_a_set"] <-
  all(var_of(importance_groups(both), c("lith_acid", "lith_basic", "lith_carbonate")) ==
        c("lith_acid", "lith_basic", "lith_carbonate"))

# ── 2. a grouping of the user's ───────────────────────────────────────────────
mine <- importance_groups(fake, list(texture = "clay",
                                     rock = c("lith_acid", "lith_basic", "lith_carbonate")))
ok["your_groups_replace_the_rule"] <- var_of(mine, "clay") == "texture" &&
  all(var_of(mine, c("lith_acid", "lith_carbonate")) == "rock") &&
  all(mine$rule[mine$variable %in% c("texture", "rock")] == "yours")
ok["what_you_do_not_name_keeps_the_rule"] <-
  all(var_of(mine, c("soil_class_a", "soil_class_b")) == "soil_class")
ok["a_data_frame_works_as_a_list_does"] <- identical(
  importance_groups(fake, data.frame(channel = c("clay", "flood"), variable = "misc")),
  importance_groups(fake, list(misc = c("clay", "flood"))))
ok["an_unknown_channel_is_refused"] <-
  grepl("does not have", err(importance_groups(fake, list(a = "silt"))))
ok["a_channel_in_two_variables_is_refused"] <-
  grepl("one variable only", err(importance_groups(fake, list(a = "clay", b = "clay"))))
ok["a_name_that_is_also_a_lone_channel_is_refused"] <-
  grepl("both a group of yours", err(importance_groups(fake, list(flood = c("wet_inland", "wet_coastal")))))
ok["a_set_cut_in_two_is_said"] <-
  grepl("splits the one-hot set", warned(importance_groups(fake, list(a = "lith_acid"))))
# In counts, not a list of every channel: a 33-class set made the first
# version's warning unreadable on the smoke (2026-10-03).
ok["the_split_is_said_in_counts"] <-
  grepl("1 in 1 variable\\(s\\) of yours, 2 left out \\(lith_basic, lith_carbonate\\)",
        warned(importance_groups(fake, list(a = "lith_acid"))))
ok["a_word_that_is_not_a_rule_is_refused"] <-
  grepl("must be", err(importance_groups(fake, "themes")))

# ── 3. the draws ──────────────────────────────────────────────────────────────
strata <- c("a", rep("b", 2L), rep("c", 3L), rep("d", 44L))
d1 <- .importance_donors(50L, strata, seed = 9L)
by_s <- split(seq_len(50L), strata)
ok["every_row_is_drawn_once_within_its_stratum"] <-
  all(vapply(by_s, function(g) identical(sort(d1[g]), g), logical(1)))
ok["no_row_of_a_shared_stratum_draws_itself"] <-
  !any(d1[strata != "a"] == seq_len(50L)[strata != "a"])
ok["a_row_alone_keeps_itself"] <- d1[1] == 1L
ok["the_seed_reproduces_the_draw"] <- identical(d1, .importance_donors(50L, strata, seed = 9L))
ok["another_seed_draws_otherwise"] <- !identical(d1, .importance_donors(50L, strata, seed = 10L))
ok["unrestricted_is_one_stratum"] <- {
  d0 <- .importance_donors(30L, NULL, seed = 1L)
  identical(sort(d0), seq_len(30L)) && !any(d0 == seq_len(30L))
}
ok["the_share_alone_is_counted"] <- isTRUE(all.equal(.importance_alone_share(strata), 1 / 50))
ok["two_rows_swap"] <- identical(.importance_donors(2L, NULL, seed = 4L), c(2L, 1L))

# ── 4. end to end, against known models ───────────────────────────────────────
#
# POSITIVE VALUES, as in the occlusion test: the scores clamp at zero by
# default, and a fixture below it would measure the clamp.
set.seed(17)
nv <- 60L; cv <- 5L
a5 <- array(runif(nv * cv * 5 * 5, 1, 6), dim = c(nv, cv, 5L, 5L))
a3 <- a5[, , 2:4, 2:4, drop = FALSE]                  # concentric: the 3x3 inside the 5x5
x5 <- torch_tensor(a5, dtype = torch_float())
x3 <- torch_tensor(a3, dtype = torch_float())
ring1 <- function(a, k) (apply(a[, k, 2:4, 2:4, drop = FALSE], 1, sum) - a[, k, 3, 3]) / 8

two_channel_reader <- nn_module(
  "two_channel_reader",
  forward = function(x) x[, 1, 3, 3] + (x[, 2, 2:4, 2:4]$sum(dim = c(2, 3)) - x[, 2, 3, 3]) / 8
)
y_two <- a5[, 1, 3, 3] + ring1(a5, 2L)
vars5 <- stats::setNames(as.list(seq_len(cv)), paste0("v", seq_len(cv)))
dons  <- lapply(1:3, function(d) .importance_donors(nv, NULL, seed = 100L + d))

r_two <- .importance_unit(two_channel_reader(), list(x5), y_two, y_two, vars5, donors = dons,
                          batch_size = 16L)
imp_of <- function(r, v) r$baseline$ccc - r$raw$ccc[r$raw$variable == v]
ok["the_known_model_reproduces_its_target"] <- r_two$baseline$ccc > 0.9999
ok["batches_give_what_one_pass_gives"] <- isTRUE(all.equal(
  .importance_forward(two_channel_reader(), list(x5), 16L),
  .importance_forward(two_channel_reader(), list(x5), 1000L)))
# THE ASSERTION THE FILE IS FOR: exactly zero, not small.
ok["what_the_model_ignores_costs_exactly_nothing"] <-
  all(vapply(c("v3", "v4", "v5"), function(v) all(imp_of(r_two, v) == 0), logical(1)))
ok["what_it_reads_costs_something"] <- all(imp_of(r_two, "v1") > 0.3) &&
  all(imp_of(r_two, "v2") > 0.05)
ok["one_row_per_variable_and_draw"] <- nrow(r_two$raw) == cv * 3L

# A GROUP WITH A CHANNEL THE MODEL IGNORES is the used channel alone: with the
# same donors, the predictions are the same numbers.
r_grp <- .importance_unit(two_channel_reader(), list(x5), y_two, y_two,
                          list(v1 = 1L, v1_and_v4 = c(1L, 4L)), donors = dons)
ok["a_group_with_an_unused_channel_is_the_used_one"] <-
  identical(r_grp$raw$ccc[r_grp$raw$variable == "v1"],
            r_grp$raw$ccc[r_grp$raw$variable == "v1_and_v4"])

# ONE DONOR FOR EVERY WINDOW. This model returns channel 2's centre plus the
# difference between channel 1's centre in the two windows -- zero for a real
# point, whose windows are concentric. Permuting channel 1 with the same donor
# in both keeps it zero, and the importance exactly zero; a donor per window
# would hand the network two places at once and the difference would show.
window_checker <- nn_module(
  "window_checker",
  forward = function(s, l) l[, 2, 3, 3] + (s[, 1, 2, 2] - l[, 1, 3, 3])
)
y_win <- a5[, 2, 3, 3]
r_win <- .importance_unit(window_checker(), list(x3, x5), y_win, y_win,
                          list(v1 = 1L, v2 = 2L), donors = dons, batch_size = 7L)
ok["the_windows_are_given_one_donor"] <- all(imp_of(r_win, "v1") == 0)
ok["and_the_variable_read_still_counts"] <- all(imp_of(r_win, "v2") > 0.3)

# THE MEAN FILL, BY HAND. Channel 1 set to 2.5 everywhere: the model then
# predicts 2.5 plus channel 2's ring, and that CCC is computable here.
fill5 <- rep(2.5, cv)
r_fill <- .importance_unit(two_channel_reader(), list(x5), y_two, y_two, list(v1 = 1L),
                           fill = fill5)
ok["the_mean_fill_gives_the_predicted_change"] <-
  isTRUE(all.equal(r_fill$raw$ccc, ccc(y_two, 2.5 + ring1(a5, 2L)), tolerance = 1e-5))
ok["donors_or_fill_one_of_the_two"] <-
  grepl("one of the two", err(.importance_unit(two_channel_reader(), list(x5), y_two, y_two,
                                               list(v1 = 1L), donors = dons, fill = fill5)))

# A REGIONAL GRADIENT LOSES ITS IMPORTANCE WITHIN BLOCKS. Channel 1 is one value
# per block of ten points -- a climate, seen at this scale -- and channel 2
# varies inside each block. Over all rows both matter; within blocks, every
# donor of channel 1 carries the point's own value, and only channel 2 is left.
blk <- rep(seq_len(6L), each = 10L)
ar <- array(runif(nv * 2 * 3 * 3, 1, 2), dim = c(nv, 2L, 3L, 3L))
ar[, 1, , ] <- rep(seq(1, 6, length.out = 6L)[blk], times = 9L)
xr <- torch_tensor(ar, dtype = torch_float())
centre_sum <- nn_module("centre_sum", forward = function(x) x[, 1, 2, 2] + x[, 2, 2, 2])
y_reg <- ar[, 1, 2, 2] + ar[, 2, 2, 2]
blocks <- list(store = list(meta = tibble::tibble(x = blk * 10 + runif(nv), y = 0.5)))
st <- .importance_strata(10, blocks, seq_len(nv))
ok["blocks_are_cut_from_the_coordinates"] <- length(unique(st)) == 6L
d_all <- lapply(1:3, function(d) .importance_donors(nv, NULL, d))
d_blk <- lapply(1:3, function(d) .importance_donors(nv, st, d))
r_all <- .importance_unit(centre_sum(), list(xr), y_reg, y_reg, list(v1 = 1L, v2 = 2L), donors = d_all)
r_blk <- .importance_unit(centre_sum(), list(xr), y_reg, y_reg, list(v1 = 1L, v2 = 2L), donors = d_blk)
ok["over_all_rows_the_gradient_matters"] <- all(imp_of(r_all, "v1") > 0.3)
ok["within_blocks_it_is_exactly_nothing"] <- all(imp_of(r_blk, "v1") == 0)
ok["within_blocks_the_local_variable_stays"] <- all(imp_of(r_blk, "v2") > 0.01)
ok["a_class_column_cuts_strata_too"] <- {
  dd <- list(store = list(meta = tibble::tibble(x = 1:4, y = 1:4)),
             points = tibble::tibble(region = c("n", "n", "s", NA)))
  s2 <- .importance_strata("region", dd, 1:4)
  s2[1] == s2[2] && s2[3] == "s" && grepl("^\\.none_", s2[4])
}
ok["a_column_that_is_not_there_is_refused"] <-
  grepl("is not a column", err(.importance_strata("biome", list(store = list(meta = tibble::tibble(x = 1, y = 1)),
                                                                   points = tibble::tibble(a = 1)), 1L)))

# ── 5. from draws and models to the table ─────────────────────────────────────
raw <- tibble::tibble(unit = rep(c("m1", "m2"), each = 4L), variable = rep(c("a", "a", "b", "b"), 2L),
                      draw = rep(1:2, 4L),
                      ccc = c(0.5, 0.7, 0.9, 0.9, 0.6, 0.6, 0.9, 0.9),
                      rmse = c(3, 2, 1.2, 1.2, 2.5, 2.5, 1.1, 1.3), rmse_transform = 0.2)
base <- tibble::tibble(unit = c("m1", "m2"), n = 10L, ccc = c(0.9, 0.9), rmse = c(1, 1),
                       rmse_transform = 0.1)
ag <- .importance_aggregate(raw, base, "ccc")
ta <- ag$table
ok["larger_is_more_important"] <- identical(ta$variable, c("a", "b")) && identical(ta$rank, 1:2)
ok["the_mean_is_over_draws_then_models"] <-
  isTRUE(all.equal(ta$importance[ta$variable == "a"], mean(c(0.3, 0.3))))
ok["the_spread_between_models_is_kept"] <- isTRUE(all.equal(ta$sd_models[ta$variable == "b"], 0)) &&
  isTRUE(all.equal(ta$sd_draws[ta$variable == "a"], sqrt(mean(c(stats::sd(c(0.4, 0.2))^2, 0)))))
ok["in_how_many_models_it_mattered"] <- ta$n_models_positive[ta$variable == "a"] == 2L &&
  ta$n_models_positive[ta$variable == "b"] == 0L
ok["the_rmse_rises_as_the_ccc_falls"] <-
  isTRUE(all.equal(ta$importance_rmse[ta$variable == "a"], mean(c(1.5, 1.5))))
ok["ranked_by_the_rmse_when_asked"] <-
  isTRUE(all.equal(.importance_aggregate(raw, base, "rmse")$table$importance[1], 1.5))
ok["the_share_of_the_baseline"] <- isTRUE(all.equal(ta$pct_of_baseline[1], 100 * 0.3 / 0.9))

# ── 6. side by side ───────────────────────────────────────────────────────────
fake_imp <- function(vars, imp, label) structure(list(
  table = tibble::tibble(rank = seq_along(vars), variable = vars, importance = imp),
  method = list(metric = "ccc"), label = label), class = "dsm_importance")
x_a <- fake_imp(c("a", "b", "c", "d"), c(4, 3, 2, 1), "A")
x_b <- fake_imp(c("d", "c", "b", "a"), c(4, 3, 2, 1), "B")
cmp_same <- compare_importance(one = x_a, two = x_a)
cmp_rev  <- compare_importance(x_a, x_b)
ok["the_same_ranking_agrees_fully"] <- isTRUE(all.equal(cmp_same$agreement[1, 2], 1))
ok["a_reversed_ranking_disagrees_fully"] <- isTRUE(all.equal(cmp_rev$agreement[1, 2], -1))
ok["names_become_labels"] <- identical(unname(cmp_same$labels), c("one", "two")) &&
  identical(unname(cmp_rev$labels), c("A", "B"))
ok["each_variable_once_with_every_rank"] <- nrow(cmp_rev$table) == 4L &&
  all(c("rank_i1", "rank_i2", "share_i1") %in% names(cmp_rev$table))
ok["one_importance_is_not_a_comparison"] <- grepl("two importances", err(compare_importance(x_a)))
ok["the_comparison_prints"] <- length(capture.output(print(cmp_rev))) > 5L

# ── 7. the method ─────────────────────────────────────────────────────────────
ok["draws_must_be_whole"] <- grepl("whole number", err(permutation_importance(draws = 0)))
ok["a_mean_has_no_draws"] <- permutation_importance(fill = "mean", draws = 7L)$draws == 1L
ok["a_mean_has_no_block"] <- grepl("do not go together",
                                   err(permutation_importance(within = 1, fill = "mean")))
ok["a_block_is_positive"] <- grepl("positive", err(permutation_importance(within = -1)))
ok["the_method_prints"] <- grepl("within blocks of 0.5",
                                 paste(capture.output(print(permutation_importance(within = 0.5))),
                                       collapse = " "))
ok["dsm_importance_wants_a_method"] <-
  grepl("importance method", err(dsm_importance("nowhere", fake, method = list(kind = "x"))))

# ── 8. context: rings, variables at rings, windows, the gate ──────────────────
#
# The same design as section 4, for places in the patch instead of variables.
# A ring is read here through a mask that is 0 at the centre, never as "the 3x3
# sum minus the centre": in float32 that difference moves in its last bits when
# only the centre is permuted, and "exactly zero" would fail for a reason that
# is arithmetic, not wiring.
ctx <- function(spec, ws, vars = vars5) .importance_context_targets(spec, ws, vars)
ring_mask5 <- torch_tensor(patch_ring_index(5L) == 1L, dtype = torch_float())
centre_reader5 <- nn_module("centre_reader5", forward = function(x) x[, 1, 3, 3])
ring1_reader5  <- nn_module("ring1_reader5",
                            forward = function(x) (x[, 2, , ] * ring_mask5)$sum(dim = c(2, 3)) / 8)
both_reader5   <- nn_module("both_reader5", forward = function(x) {
  x[, 1, 3, 3] + (x[, 2, , ] * ring_mask5)$sum(dim = c(2, 3)) / 8
})

tg_ring <- ctx(context_importance(), 5L)
ok["the_default_bands_are_centre_each_ring_and_context"] <-
  identical(names(tg_ring$targets), c("centre", "ring_01", "ring_02", "context")) &&
  identical(tg_ring$info$pixels, c(1, 8, 16, 24))
ok["a_band_of_yours_counts_its_ground"] <-
  identical(ctx(context_importance(bands = list(near = 0:1, far = 2L)), 5L)$info$pixels, c(9, 16))
ok["a_band_beyond_the_window_is_refused"] <-
  grepl("beyond the largest", err(ctx(context_importance(bands = list(far = 9L)), 5L)))

y_c <- a5[, 1, 3, 3]
y_r <- ring1(a5, 2L)
r_c <- .importance_unit(centre_reader5(), list(x5), y_c, y_c, tg_ring$targets, donors = dons)
r_r <- .importance_unit(ring1_reader5(), list(x5), y_r, y_r, tg_ring$targets, donors = dons)
zero <- function(r, ts) all(vapply(ts, function(b) all(imp_of(r, b) == 0), logical(1)))
ok["a_centre_reader_loses_only_its_centre"] <- all(imp_of(r_c, "centre") > 0.5) &&
  zero(r_c, c("ring_01", "ring_02", "context"))
ok["a_ring_reader_loses_only_its_ring"] <- all(imp_of(r_r, "ring_01") > 0.5) &&
  all(imp_of(r_r, "context") > 0.5) && zero(r_r, c("centre", "ring_02"))

tg_pv <- ctx(context_importance(per_variable = TRUE), 5L)
ok["per_variable_crosses_every_variable_with_centre_and_context"] <-
  length(tg_pv$targets) == cv * 2L && all(c("v1 | centre", "v2 | context") %in% names(tg_pv$targets))
y_b <- a5[, 1, 3, 3] + ring1(a5, 2L)
r_pv <- .importance_unit(both_reader5(), list(x5), y_b, y_b, tg_pv$targets, donors = dons)
ok["each_variable_counts_only_where_it_is_read"] <-
  all(imp_of(r_pv, "v1 | centre") > 0.3) && all(imp_of(r_pv, "v2 | context") > 0.05) &&
  zero(r_pv, c("v1 | context", "v2 | centre",
               paste(rep(c("v3", "v4", "v5"), 2L), rep(c("centre", "context"), each = 3L), sep = " | ")))

# THE SAME GROUND IN EVERY WINDOW. This model returns channel 2's centre plus
# the difference between one pixel of ring 1 as the 3x3 holds it and as the 5x5
# holds it -- the same ground, so zero. Ring 1 permuted with one donor in both
# windows keeps it zero; a ring masked at another place in one of the windows
# would not.
ring_aligner <- nn_module("ring_aligner",
                          forward = function(s, l) l[, 2, 3, 3] + (s[, 1, 1, 1] - l[, 1, 2, 2]))
tg_35 <- ctx(context_importance(), c(3L, 5L))
r_al <- .importance_unit(ring_aligner(), list(x3, x5), y_win, y_win, tg_35$targets, donors = dons)
ok["a_ring_is_the_same_ground_in_every_window"] <- zero(r_al, c("ring_01", "ring_02", "context"))
ok["and_the_centre_still_counts"] <- all(imp_of(r_al, "centre") > 0.3)

tg_win <- ctx(context_importance(by = "window"), c(3L, 5L))
ok["each_window_is_a_target"] <- identical(names(tg_win$targets), c("w03", "w05"))
large_reader <- nn_module("large_reader", forward = function(s, l) l[, 2, 3, 3])
r_w <- .importance_unit(large_reader(), list(x3, x5), y_win, y_win, tg_win$targets, donors = dons)
ok["a_window_the_model_does_not_read_costs_nothing"] <- zero(r_w, "w03")
ok["the_window_it_reads_costs"] <- all(imp_of(r_w, "w05") > 0.3)

# THE GATE, on a real two-branch network: one weight per point, between 0 and
# 1, and nothing to read when the branches are concatenated instead.
g35 <- make_manual_tune_grid(window_sizes = list(c(3L, 5L)), conv_channels = list(c(4L)),
                             embedding_dim = 8L, base_lr = 0.01, batch_size = 8L, dropout = 0,
                             gate_type = "scalar_per_sample", use_residual = FALSE,
                             use_se_block = FALSE)
torch_manual_seed(1)
gated <- build_cnn_from_config(g35[1, , drop = FALSE], cv)
gp <- .importance_gate_points(gated, list(x3, x5), 16L)
ok["the_gate_is_read_point_by_point"] <- is.data.frame(gp) && nrow(gp) == nv &&
  all(gp$gate >= 0 & gp$gate <= 1) && all(gp$norm_1 > 0) && all(gp$norm_2 > 0)
plain <- g35[1, , drop = FALSE]
plain$gate_type <- "no_gate_concat"
ok["no_gate_nothing_to_read"] <-
  is.null(.importance_gate_points(build_cnn_from_config(plain, cv), list(x3, x5), 16L))

ok["a_window_has_no_bands"] <-
  grepl("belong to", err(context_importance(by = "window", per_variable = TRUE)))
ok["bands_are_named_ring_numbers"] <-
  grepl("named list", err(context_importance(bands = list(0:1)))) &&
  grepl("0 or more", err(context_importance(bands = list(a = -1))))
ok["the_context_method_prints"] <-
  grepl("context by ring, per variable",
        paste(capture.output(print(context_importance(per_variable = TRUE))), collapse = " "))

# ── 9. SHAP, against attributions known in closed form ────────────────────────
#
# With a constant gradient w, the path integral is w times the distance
# travelled, whatever the path: a linear model's attributions are its weights
# times (x - reference), for both estimators. A square's integrated gradients
# are exact too: along the path the integrand is linear, and the midpoint rule
# integrates a line without error.
G5 <- diag(cv)
lin_reader <- nn_module("lin_reader", forward = function(x) {
  2 * x[, 1, 3, 3] - 0.5 * x[, 3, , ]$mean(dim = c(2, 3))
})
mean3  <- apply(a5[, 3, , , drop = FALSE], 1, mean)
y_lin  <- 2 * a5[, 1, 3, 3] - 0.5 * mean3
fill_b <- c(2.5, 3, 3.5, 4, 4.5)

r_ig <- .importance_shap_unit(lin_reader(), list(x5), y_lin, y_lin, G5, "integrated_gradients",
                              fill = fill_b, K = 10L, batch_size = 16L)
ok["ig_of_a_linear_model_is_its_weight_times_the_distance"] <-
  max(abs(r_ig$phi[, 1] - 2 * (a5[, 1, 3, 3] - 2.5))) < 1e-3 &&
  max(abs(r_ig$phi[, 3] + 0.5 * (mean3 - 3.5))) < 1e-3
ok["ig_gives_what_the_model_ignores_exactly_nothing"] <- all(r_ig$phi[, c(2, 4, 5)] == 0)
ok["ig_adds_up_exactly"] <- max(abs(r_ig$gap)) < 1e-3

sq_reader <- nn_module("sq_reader", forward = function(x) x[, 1, 3, 3]^2)
y_sq <- a5[, 1, 3, 3]^2
r_sq <- .importance_shap_unit(sq_reader(), list(x5), y_sq, y_sq, G5, "integrated_gradients",
                              fill = fill_b, K = 8L)
ok["ig_of_a_square_is_exact_by_the_midpoint_rule"] <-
  max(abs(r_sq$phi[, 1] - (a5[, 1, 3, 3]^2 - 2.5^2))) < 1e-3

# EXPECTED GRADIENTS, the same model: each point's value is the weight times
# the mean distance to the references it drew, which is computable here.
bg_arr <- array(runif(30 * cv * 25, 1, 6), dim = c(30L, cv, 5L, 5L))
x5_bg  <- torch_tensor(bg_arr, dtype = torch_float())
Ke <- 40L
dr <- with_local_seed(5L, list(j = matrix(sample.int(30L, nv * Ke, TRUE), nv, Ke),
                               a = matrix(runif(nv * Ke), nv, Ke)))
r_eg <- .importance_shap_unit(lin_reader(), list(x5), y_lin, y_lin, G5, "expected_gradients",
                              bg_inputs = list(x5_bg), draws = dr, K = Ke, batch_size = 16L)
ref1 <- matrix(bg_arr[dr$j, 1, 3, 3], nv, Ke)
ok["eg_of_a_linear_model_is_its_weight_times_the_mean_distance"] <-
  max(abs(r_eg$phi[, 1] - 2 * (a5[, 1, 3, 3] - rowMeans(ref1)))) < 1e-3
ok["eg_gives_what_the_model_ignores_exactly_nothing"] <- all(r_eg$phi[, c(2, 4, 5)] == 0)
ok["eg_adds_up_within_its_sampling"] <-
  is.data.frame(.importance_check_completeness(r_eg, "expected_gradients", "lin"))

G13 <- cbind(g13 = c(1, 0, 1, 0, 0), v2 = c(0, 1, 0, 0, 0), v4 = c(0, 0, 0, 1, 0),
             v5 = c(0, 0, 0, 0, 1))
r_g <- .importance_shap_unit(lin_reader(), list(x5), y_lin, y_lin, G13, "integrated_gradients",
                             fill = fill_b, K = 10L)
ok["a_group_is_the_sum_of_its_channels"] <-
  max(abs(r_g$phi[, 1] - (r_ig$phi[, 1] + r_ig$phi[, 3]))) < 1e-4

two_win <- nn_module("two_win", forward = function(s, l) s[, 1, 2, 2] + l[, 2, 3, 3])
y_tw <- a5[, 1, 3, 3] + a5[, 2, 3, 3]
r_tw <- .importance_shap_unit(two_win(), list(x3, x5), y_tw, y_tw, G5, "integrated_gradients",
                              fill = fill_b, K = 4L)
ok["shap_counts_every_window"] <-
  max(abs(r_tw$phi[, 1] - (a5[, 1, 3, 3] - 2.5))) < 1e-3 &&
  max(abs(r_tw$phi[, 2] - (a5[, 2, 3, 3] - 3))) < 1e-3 && max(abs(r_tw$gap)) < 1e-3

r_rs <- .importance_shap_unit(ring1_reader5(), list(x5), y_r, y_r, G5, "integrated_gradients",
                              fill = fill_b, K = 4L)
ring5 <- patch_ring_index(5L)
ok["the_attribution_sits_on_the_ring_read"] <-
  r_rs$ring_abs[2, 1, 2] > 0 && r_rs$ring_abs[2, 1, 1] == 0 && r_rs$ring_abs[2, 1, 3] == 0 &&
  all(r_rs$ring_abs[-2, , ] == 0)
ok["and_the_pixel_map_draws_the_ring"] <-
  all(r_rs$pix_abs[[1]][ring5 != 1L] == 0) && all(r_rs$pix_abs[[1]][ring5 == 1L] > 0)

# A REAL TWO-BRANCH NETWORK, with its gate, convolutions and normalisation: no
# closed form, but its attributions must add up, by both estimators.
bg35 <- list(x3[1:20, , , , drop = FALSE], x5[1:20, , , , drop = FALSE])
dr_real <- with_local_seed(6L, list(j = matrix(sample.int(20L, nv * 50L, TRUE), nv, 50L),
                                    a = matrix(runif(nv * 50L), nv, 50L)))
r_real <- .importance_shap_unit(gated, list(x3, x5), y_win, y_win, G5, "expected_gradients",
                                bg_inputs = bg35, draws = dr_real, K = 50L, batch_size = 16L)
ok["a_real_networks_expected_gradients_add_up"] <- all(is.finite(r_real$phi)) &&
  is.data.frame(.importance_check_completeness(r_real, "expected_gradients", "gated"))
r_real_ig <- .importance_shap_unit(gated, list(x3, x5), y_win, y_win, G5, "integrated_gradients",
                                   fill = fill_b, K = 64L, batch_size = 16L)
ok["a_real_networks_integrated_gradients_add_up"] <-
  is.data.frame(.importance_check_completeness(r_real_ig, "integrated_gradients", "gated"))

# AND WHAT DOES NOT ADD UP STOPS: off on average, or a path cut too coarsely.
ok["attributions_off_on_average_stop"] <- grepl("do not add up", err(
  .importance_check_completeness(list(delta = rep(1, 50), gap = 0.3 + seq(-0.01, 0.01, length.out = 50)),
                                 "expected_gradients", "m")))
ok["a_path_cut_too_coarsely_stops"] <- grepl("too coarsely", err(
  .importance_check_completeness(list(delta = rep(1, 10), gap = rep(0.1, 10)),
                                 "integrated_gradients", "m")))
ok["shap_wants_whole_samples"] <- grepl("whole number", err(shap_importance(samples = 0)))
ok["shap_prints_its_estimator"] <-
  grepl("expected gradients", paste(capture.output(print(shap_importance())), collapse = " "))

# ── 10. ALE, against effects known in closed form ─────────────────────────────
#
# The scaling is the identity here, so the inputs ARE the values the bins are
# cut from. A linear model's local effect in a bin is its weight times the
# bin's width, for every point in it; a square's is the difference of the
# edges' squares. Both are exact, so the curves are checked to the float.
ch5  <- paste0("v", seq_len(cv))
pts5 <- tibble::as_tibble(stats::setNames(lapply(seq_len(cv), function(k) a5[, k, 3, 3]), ch5))
fake5 <- list(store = list(predictors = ch5), points = pts5,
              type_table = tibble::tibble(predictor = ch5, is_dummy = FALSE, is_percentage = FALSE))
sc_id <- tibble::tibble(predictor = ch5, center = 0, scale = 1)
vars_c <- stats::setNames(as.list(seq_len(cv)), ch5)
plan5 <- .importance_ale_plan(fake5, vars_c, seq_len(nv), bins = 10L)
ok["the_bins_are_quantiles_of_the_points"] <- length(plan5$v1$edges) == 11L &&
  plan5$v1$edges[1] == min(a5[, 1, 3, 3]) && plan5$v1$edges[11] == max(a5[, 1, 3, 3])

r_ale <- .importance_ale_unit(lin_reader(), list(x5), plan5, fake5, seq_len(nv), sc_id, y_lin, y_lin)
e1 <- plan5$v1$edges
ok["a_linear_curve_rises_by_weight_times_width"] <-
  max(abs(diff(r_ale$effects$v1$ale) - 2 * diff(e1))) < 1e-4
# Channel 3 is read as the MEAN of its patch, and the bins are cut from its
# centre: the curve follows only because the whole patch moves. Moving the
# centre pixel alone would shift the mean by 1/25 of the step.
ok["the_whole_patch_moves"] <-
  max(abs(diff(r_ale$effects$v3$ale) + 0.5 * diff(plan5$v3$edges))) < 1e-4
ok["what_the_model_does_not_read_is_flat_exactly"] <-
  all(vapply(c("v2", "v4", "v5"), function(v) all(r_ale$effects[[v]]$ale == 0) &&
               r_ale$effects[[v]]$importance == 0, logical(1)))
mid1 <- (r_ale$effects$v1$ale[-1] + r_ale$effects$v1$ale[-11]) / 2
ok["the_curve_is_centred_on_the_points"] <- abs(sum(mid1 * r_ale$effects$v1$n)) < 1e-6
r_sq_ale <- .importance_ale_unit(sq_reader(), list(x5), plan5["v1"], fake5, seq_len(nv), sc_id, y_sq, y_sq)
ok["a_square_rises_by_the_squares_step"] <-
  max(abs(diff(r_sq_ale$effects$v1$ale) - diff(e1^2))) < 1e-3

# EVERY WINDOW MOVES. Channel 1 read in both windows: the curve rises twice as
# fast as for one -- a shift in one window only would give the slope of one.
both_windows <- nn_module("both_windows", forward = function(s, l) s[, 1, 2, 2] + l[, 1, 3, 3])
y_bw <- 2 * a5[, 1, 3, 3]
r_bw <- .importance_ale_unit(both_windows(), list(x3, x5), plan5["v1"], fake5, seq_len(nv), sc_id, y_bw, y_bw)
ok["every_window_moves_together"] <- max(abs(diff(r_bw$effects$v1$ale) - 2 * diff(e1))) < 1e-4

# A CATEGORICAL: channels 3 to 5 one-hot, pixel by pixel. With the whole patch
# made class j, the model returns the class's weight, so the effects differ by
# the weights' differences exactly; a binary map alone has two classes.
set.seed(23)
cls_px <- array(sample.int(3L, nv * 25L, TRUE), dim = c(nv, 5L, 5L))
a_cat <- a5
for (j in 1:3) a_cat[, 2L + j, , ] <- (cls_px == j) * 1
x_cat <- torch_tensor(a_cat, dtype = torch_float())
pts_cat <- tibble::as_tibble(stats::setNames(lapply(seq_len(cv), function(k) a_cat[, k, 3, 3]), ch5))
fake_cat <- list(store = list(predictors = ch5), points = pts_cat,
                 type_table = tibble::tibble(predictor = ch5, is_dummy = c(FALSE, FALSE, TRUE, TRUE, TRUE),
                                             is_percentage = FALSE))
cat_reader <- nn_module("cat_reader", forward = function(x) {
  x[, 3, 3, 3] + 2 * x[, 4, 3, 3] + 3 * x[, 5, 3, 3]
})
y_cat <- a_cat[, 3, 3, 3] + 2 * a_cat[, 4, 3, 3] + 3 * a_cat[, 5, 3, 3]
plan_cat <- .importance_ale_plan(fake_cat, list(v1 = 1L, cls = 3:5, v5 = 5L), seq_len(nv), 10L)
ok["a_set_is_categorical_and_a_lone_dummy_binary"] <-
  identical(plan_cat$cls$kind, "categorical") && !plan_cat$cls$binary &&
  identical(plan_cat$v5$classes, c("absent", "present"))
r_cat <- .importance_ale_unit(cat_reader(), list(x_cat), plan_cat, fake_cat, seq_len(nv), sc_id,
                              y_cat, y_cat)
eff <- r_cat$effects$cls$effect
ok["a_class_effect_is_the_class_difference"] <-
  abs((eff[2] - eff[1]) - 1) < 1e-5 && abs((eff[3] - eff[1]) - 2) < 1e-5
ok["the_classes_count_the_points"] <- sum(r_cat$effects$cls$n) == nv
ok["a_binary_map_is_present_against_absent"] <-
  abs(diff(r_cat$effects$v5$effect) - 3) < 1e-5

# ── 11. the importance as the AOA's weights ───────────────────────────────────
fk_imp <- function(imp) structure(list(
  table = tibble::tibble(variable = c("a", "set"), importance = imp),
  groups = tibble::tibble(channel = c("a", "s1", "s2"), variable = c("a", "set", "set"),
                          rule = c("alone", "one-hot set", "one-hot set")),
  method = list(kind = "permutation")), class = "dsm_importance")
w_ab <- importance_weights(fk_imp(c(0.3, -0.1)))
ok["each_channel_takes_its_variables_weight"] <- identical(names(w_ab), c("a", "s1", "s2")) &&
  w_ab[["a"]] == 0.3 && all(w_ab[c("s1", "s2")] == 0)
ok["every_channel_of_a_set_takes_the_sets_weight"] <-
  all(importance_weights(fk_imp(c(0.3, 0.2)))[c("s1", "s2")] == 0.2)
ok["an_importance_below_zero_everywhere_gives_no_weights"] <-
  grepl("zero or below", err(importance_weights(fk_imp(c(-1, 0)))))
ok["a_context_importance_gives_no_weights"] <-
  grepl("per ring or window", err(importance_weights(structure(list(method = list(kind = "context")),
                                                               class = "dsm_importance"))))
ok["a_map_takes_an_importance_as_its_weights"] <- identical(
  .predict_aoa_weights(fk_imp(c(0.3, 0.2)), c("a", "s1", "s2")), importance_weights(fk_imp(c(0.3, 0.2))))
ok["a_map_puts_named_weights_in_its_order"] <-
  identical(.predict_aoa_weights(c(s2 = 1, a = 2, s1 = 3), c("a", "s1", "s2")), c(a = 2, s1 = 3, s2 = 1))
ok["a_map_refuses_weights_it_cannot_place"] <-
  grepl("name every channel", err(.predict_aoa_weights(c(a = 1, x = 2), c("a", "s1")))) &&
  grepl("non-negative", err(.predict_aoa_weights(c(-1, 1), c("a", "b")))) &&
  grepl("value\\(s\\) for", err(.predict_aoa_weights(c(1, 2, 3), c("a", "b")))) &&
  grepl("all zero", err(.predict_aoa_weights(c(0, 0), c("a", "b"))))

# ── 12. the kernel: Shapley values of the variables, from coalitions ──────────
#
# In the game the kernel plays -- a variable outside a coalition taken from a
# background point -- a linear model's values are its weights times the
# distance to the background's mean, exactly; a product of two variables
# splits the interaction by Shapley's rule; what the model ignores gets
# exactly nothing; and every point's values add up exactly. With sampled
# permutations instead of every coalition, a linear model's values are exact
# too: each permutation gives a variable the same marginal.
bg8  <- x5_bg[1:8, , , , drop = FALSE]
bga8 <- bg_arr[1:8, , , , drop = FALSE]
r_k <- .importance_kernel_unit(lin_reader(), list(x5), G5, list(bg8), batch_size = 64L)
ok["kernel_linear_is_weight_times_distance_to_the_background_mean"] <-
  max(abs(r_k$phi[, 1] - 2 * (a5[, 1, 3, 3] - mean(bga8[, 1, 3, 3])))) < 1e-4 &&
  max(abs(r_k$phi[, 3] + 0.5 * (mean3 - mean(apply(bga8[, 3, , , drop = FALSE], 1, mean))))) < 1e-4
ok["kernel_gives_what_the_model_ignores_exactly_nothing"] <- all(r_k$phi[, c(2, 4, 5)] == 0)
ok["kernel_adds_up_exactly"] <- max(abs(r_k$gap)) < 1e-5

prod_reader <- nn_module("prod_reader", forward = function(x) x[, 1, 3, 3] * x[, 2, 3, 3])
r_pk <- .importance_kernel_unit(prod_reader(), list(x5), G5, list(bg8), batch_size = 64L)
b1 <- bga8[, 1, 3, 3]; b2 <- bga8[, 2, 3, 3]
xx1 <- a5[, 1, 3, 3]; xx2 <- a5[, 2, 3, 3]
v0 <- mean(b1 * b2); v1 <- xx1 * mean(b2); v2 <- mean(b1) * xx2; v12 <- xx1 * xx2
ok["kernel_splits_an_interaction_by_shapley"] <-
  max(abs(r_pk$phi[, 1] - 0.5 * ((v1 - v0) + (v12 - v2)))) < 1e-3 &&
  max(abs(r_pk$phi[, 2] - 0.5 * ((v2 - v0) + (v12 - v1)))) < 1e-3

r_kp <- .importance_kernel_unit(lin_reader(), list(x5), G5, list(bg8), exact_max = 2L,
                                permutations = 6L, batch_size = 64L)
ok["sampled_permutations_are_exact_for_a_linear_model"] <-
  max(abs(r_kp$phi - r_k$phi)) < 1e-4 && max(abs(r_kp$gap)) < 1e-5

r_k2 <- .importance_kernel_unit(two_win(), list(x3, x5), G5, bg35, batch_size = 64L)
ok["the_kernel_takes_one_background_point_in_every_window"] <-
  max(abs(r_k2$phi[, 1] - (a5[, 1, 3, 3] - mean(a5[1:20, 1, 3, 3])))) < 1e-4 &&
  max(abs(r_k2$phi[, 2] - (a5[, 2, 3, 3] - mean(a5[1:20, 2, 3, 3])))) < 1e-4
ok["a_kernel_that_does_not_add_up_stops"] <- grepl("do not add up", err(
  .importance_check_completeness(list(delta = rep(1, 5), gap = rep(0.01, 5)), "kernel", "m")))
ok["the_kernel_has_its_own_defaults"] <-
  shap_importance("kernel")$background == 16L && shap_importance("kernel")$max_points == 200L &&
  shap_importance()$background == 200L && is.null(shap_importance()$max_points)
ok["too_many_variables_for_the_kernel_are_refused"] <-
  grepl("up to 40", err(.importance_kernel_size(41L)))

# TWO SHAP IMPORTANCES OF THE SAME POINTS, compared point by point: the same
# values correlate at 1, a variable's sign flipped at -1.
fake_shap <- function(phi) structure(list(
  table = tibble::tibble(rank = 1:2, variable = c("a", "b"), importance = colMeans(abs(phi))),
  points = tibble::tibble(sample_id = seq_len(nrow(phi)), a = phi[, 1], b = phi[, 2]),
  method = list(kind = "shap", metric = "ccc"), label = "s"), class = "dsm_importance")
P <- matrix(stats::rnorm(40), 20, 2)
cmp_p <- compare_importance(one = fake_shap(P), two = fake_shap(cbind(P[, 1], -P[, 2])))
ok["two_shaps_are_compared_point_by_point"] <-
  isTRUE(all.equal(cmp_p$points_agreement$r, c(1, -1))) &&
  attr(cmp_p$points_agreement, "n_points") == 20L
ok["the_point_comparison_prints"] <- any(grepl("point by point", capture.output(print(cmp_p))))

ok["ale_wants_two_bins"] <- grepl("2 or more", err(ale_effect(bins = 1)))
ok["ale_wants_distinct_names"] <- grepl("distinct", err(ale_effect(variables = c("a", "a"))))
ok["ale_prints_its_bins"] <- grepl("ALE, 20 bin", paste(capture.output(print(ale_effect())), collapse = " "))

# ── 13. SAGE: Shapley values of the loss, known in closed form ────────────────
#
# A model that reads two variables, f = x1 + m3 (channel 1's centre, channel
# 3's patch mean), on points where channel 3 is a noisy copy of channel 1:
# the two carry one signal. In the game a coalition's prediction is
# K + [1 in S] a + [3 in S] c -- a and c each variable's distance to the
# background's mean, K the background's mean prediction -- and its loss is
# quadratic in them, so with r = y - K the Shapley values of the loss are
#   phi_1 = 2 mean(a r) - mean(a^2) - mean(a c),   phi_3 alike:
# the shared part, 2 mean(a c), split in halves. Left out one at a time, each
# gets mean(a c) less -- the shared part credited to neither, which is what a
# permutation sees. What the model does not read gets exactly nothing, and
# the values add up exactly to the loss of the background's mean prediction
# less the model's own. With two variables that matter, every permutation
# paired with its reverse sees each order once, so sampled permutations are
# exact too.
pair_reader <- nn_module("pair_reader", forward = function(x) {
  x[, 1, 3, 3] + x[, 3, , ]$mean(dim = c(2, 3))
})
copy_of_one <- function(a, seed) {
  noise <- with_local_seed(seed, rnorm(length(a[, 3, , ]), 0, 0.3))
  a[, 3, , ] <- a[, 1, 3, 3] + array(noise, dim(a[, 3, , ]))
  a
}
a_sh  <- copy_of_one(a5, 31L)
bg_sh <- copy_of_one(bg_arr[1:8, , , , drop = FALSE], 32L)
x_sh  <- torch_tensor(a_sh, dtype = torch_float())
xb_sh <- list(torch_tensor(bg_sh, dtype = torch_float()))
m3_sh <- apply(a_sh[, 3, , , drop = FALSE], 1, mean)
f_sh  <- a_sh[, 1, 3, 3] + m3_sh
y_sh  <- f_sh + with_local_seed(33L, rnorm(nv, 0, 0.5))
B1 <- mean(bg_sh[, 1, 3, 3])
B3 <- mean(apply(bg_sh[, 3, , , drop = FALSE], 1, mean))
a_c <- a_sh[, 1, 3, 3] - B1; c_c <- m3_sh - B3; r_c <- y_sh - (B1 + B3)
phi1 <- 2 * mean(a_c * r_c) - mean(a_c^2) - mean(a_c * c_c)
phi3 <- 2 * mean(c_c * r_c) - mean(c_c^2) - mean(a_c * c_c)
r_s <- .importance_sage_unit(pair_reader(), list(x_sh), y_sh, G5, xb_sh, batch_size = 64L)
ok["sage_is_the_shapley_value_of_the_loss"] <-
  abs(r_s$phi[1] - phi1) < 1e-3 && abs(r_s$phi[3] - phi3) < 1e-3
ok["the_two_copies_share_the_signal"] <- mean(a_c * c_c) > 0.5 &&
  abs(r_s$phi[1] - r_s$phi[3]) < 0.25 * abs(r_s$phi[1])
ok["sage_gives_what_the_model_ignores_exactly_nothing"] <- all(r_s$phi[c(2, 4, 5)] == 0)
ok["sage_adds_up_to_the_loss_explained_exactly"] <-
  abs(sum(r_s$phi) - (r_s$loss_empty - r_s$loss_model)) < 1e-8 && abs(r_s$gap) < 1e-8
ok["sage_losses_are_the_models_and_the_mean_predictions"] <-
  abs(r_s$loss_model - mean((f_sh - y_sh)^2)) < 1e-4 &&
  abs(r_s$loss_empty - mean((B1 + B3 - y_sh)^2)) < 1e-4
r_sp <- .importance_sage_unit(pair_reader(), list(x_sh), y_sh, G5, xb_sh, exact_max = 2L,
                              permutations = 6L, batch_size = 64L)
ok["paired_permutations_are_exact_for_two_variables_that_matter"] <-
  max(abs(r_sp$phi - r_s$phi)) < 1e-6 && abs(r_sp$gap) < 1e-8
r_sm <- .importance_sage_unit(pair_reader(), list(x_sh), y_sh, G5, xb_sh, loss = "mae",
                              batch_size = 64L)
ok["sage_takes_the_loss_asked_for"] <-
  abs(r_sm$loss_model - mean(abs(f_sh - y_sh))) < 1e-4 && abs(r_sm$gap) < 1e-8
ok["sage_wants_whole_numbers"] <- grepl("whole number", err(sage_importance(background = 1)))
ok["sage_prints_its_loss_and_no_metric"] <- any(grepl(
  "SAGE, mse, 16 background point\\(s\\), at most 200 point\\(s\\)$",
  capture.output(print(sage_importance()))))
ok["too_many_variables_for_sage_are_refused"] <-
  grepl("SAGE takes up to 40", err(.importance_kernel_size(41L, sage = TRUE)))

cat(sprintf("  one-hot sets found        : %s\n",
            paste(unique(g$variable[g$rule == "one-hot set"]), collapse = ", ")))
cat(sprintf("  known model, v1 / v2 / v3 : %.3f / %.3f / %.3f (drop in CCC, mean of 3 draws)\n",
            mean(imp_of(r_two, "v1")), mean(imp_of(r_two, "v2")), mean(imp_of(r_two, "v3"))))
cat(sprintf("  regional gradient         : %.3f over all rows, %.3f within blocks\n",
            mean(imp_of(r_all, "v1")), mean(imp_of(r_blk, "v1"))))
cat(sprintf("  ring reader, by band      : centre %.3f | ring 1 %.3f | ring 2 %.3f\n",
            mean(imp_of(r_r, "centre")), mean(imp_of(r_r, "ring_01")), mean(imp_of(r_r, "ring_02"))))
loo1 <- phi1 - mean(a_c * c_c); loo3 <- phi3 - mean(a_c * c_c)
cat(sprintf("  one signal in two, SAGE   : v1 %.2f / v3 %.2f of the loss explained; left out one at a time %.2f / %.2f\n",
            r_s$phi[1] / sum(r_s$phi), r_s$phi[3] / sum(r_s$phi), loo1 / sum(r_s$phi),
            loo3 / sum(r_s$phi)))
cat(sprintf("  SHAP adds up, real network: EG noise %.1f%%, IG off by %.2f%%\n",
            100 * .importance_check_completeness(r_real, "expected_gradients", "g")$rel_noise,
            100 * .importance_check_completeness(r_real_ig, "integrated_gradients", "g")$rel_noise))

.report(ok, "test_importance")
