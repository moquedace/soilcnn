# Unit test: inference on clustered test points -- blocks, their bootstrap, a correlogram
#
# WHY THIS FILE EXISTS.
#
# The SOC 0-30 cm trial compares its validation designs on a common test set
# whose 3,900 profiles sit in 303 one-degree blocks, half of them in 11. A
# bootstrap that draws the profiles one by one counts a block of 603 as 603
# independent pieces of evidence, and its interval comes out too narrow where
# the models' errors are alike within a block. So the checks here are built on
# data whose answer is known in closed form:
#
#   1. the blocks are of equal AREA: the projection's Jacobian is 1, wherever
#      on the sphere; points in metres are cut as given; and what cannot be
#      cut well is refused
#   2. the bootstrap of blocks: with "profile" weights the estimate is
#      calc_metrics()'s; with "block" weights every block weighs one vote --
#      the estimate moves, as in the hand example; independent points give an
#      n_effective of about n; errors identical within blocks of 25 give about
#      the number of blocks; a model against itself differs by exactly zero in
#      every draw (the draws are paired); the seed reproduces it and the
#      session's random numbers are left as they were
#   3. the interval as the blocks grow: blocks that cut a cluster are narrower
#      than blocks that hold it
#   4. the correlogram: Moran's I by hand on four points; great-circle
#      distances; a field constant within clusters resembles itself at short
#      distances and not at long ones, and noise resembles itself nowhere
#
# Run: source("<package root>/tests/test_block_bootstrap.R")

suppressMessages({
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

# ── 1. blocks of equal area ───────────────────────────────────────────────────
#
# Equal area is the Jacobian of the projection being 1 against the sphere: a
# small quadrilateral of the sphere -- R^2 x dlon x (sin lat2 - sin lat1) --
# keeps its area projected, wherever it lies. Checked on cells of 0.1 degree
# at the centre, far from it, near the equator and at 55 S, against a grid of
# degrees, whose cells shrink with the cosine of the latitude.
r_auth <- 6371.0072
sphere_area <- function(lon, lat, d) {
  r_auth^2 * (d * pi / 180) * (sin((lat + d) * pi / 180) - sin(lat * pi / 180))
}
projected_area <- function(lon, lat, d, centre) {
  p <- .bb_laea_km(c(lon, lon + d, lon + d, lon), c(lat, lat, lat + d, lat + d), centre)
  0.5 * abs(sum(p$e * c(p$n[-1], p$n[1]) - c(p$e[-1], p$e[1]) * p$n))
}
centre <- c(-65, -15)
spots <- list(c(-65, -15), c(-40, -55), c(-100, 25), c(-35, 0))
ratios <- vapply(spots, function(s) projected_area(s[1], s[2], 0.1, centre) /
                   sphere_area(s[1], s[2], 0.1), numeric(1))
ok["the_projection_keeps_area_everywhere"] <- all(abs(ratios - 1) < 1e-3)
ok["a_grid_of_degrees_does_not"] <-
  sphere_area(-40, -55, 1) / sphere_area(-35, 0, 1) < 0.6

blk_m <- equal_area_blocks(c(0, 99999, 100001, 250000), c(0, 0, 50000, 0), 100, coords = "metres")
ok["points_in_metres_are_cut_as_given"] <-
  identical(as.character(blk_m), c("0_0", "0_0", "1_0", "2_0"))
# The centre is given, so no cell edge falls between the two near points: at
# (-55, -25) they project to (522.7, 546.6) and (522.6, 546.5) km, both in cell
# 10_10, and the third to (627.1, 542.6), in 12_10.
blk_ll <- equal_area_blocks(c(-50, -50.001, -49), c(-20, -20.001, -20), 50, centre = c(-55, -25))
ok["near_points_share_a_block_and_far_ones_do_not"] <-
  identical(as.character(blk_ll), c("10_10", "10_10", "12_10")) && attr(blk_ll, "size_km") == 50
ok["what_cannot_be_cut_well_is_refused"] <-
  grepl("120 degrees", err(equal_area_blocks(c(-170, 10), c(0, 0), 50, centre = c(-170, 0)))) &&
  grepl("not longitude and latitude", err(equal_area_blocks(c(500000, 1), c(0, 1), 50))) &&
  grepl("positive number", err(equal_area_blocks(1, 1, -5))) &&
  grepl("finite", err(equal_area_blocks(c(1, NA), c(1, 1), 50)))

# ── 2. the bootstrap of blocks ────────────────────────────────────────────────
set.seed(11)
n_b <- 40L; m <- 25L
blk <- rep(sprintf("b%02d", seq_len(n_b)), each = m)
obs <- 30 + rnorm(n_b * m, sd = 8)
pred_a <- obs + rnorm(n_b * m, sd = 4)
pred_b <- obs + rnorm(n_b * m, sd = 5)
bb <- block_bootstrap(obs, pred_a, blk, against = pred_b, n_boot = 500L)
cm_a <- calc_metrics(obs, pred_a)
cm_b <- calc_metrics(obs, pred_b)
pr <- bb[bb$weights == "profile", ]
ok["profile_weights_give_calc_metrics_estimates"] <-
  abs(pr$estimate[pr$metric == "mae"] - cm_a$mae) < 1e-10 &&
  abs(pr$estimate[pr$metric == "rmse"] - cm_a$rmse) < 1e-10 &&
  abs(pr$estimate[pr$metric == "ccc"] - cm_a$ccc) < 1e-10 &&
  abs(pr$estimate[pr$metric == "bias"] - cm_a$bias) < 1e-10 &&
  abs(pr$difference[pr$metric == "mae"] - (cm_a$mae - cm_b$mae)) < 1e-10

# THE BLOCK WEIGHTS, BY HAND. One block of 500 points where pred errs by 0 and
# against by 2, and 99 blocks of 5 where both err by 1. Per profile the
# difference is -1000/995; per block, every block one vote, it is -2/100.
blk_e <- c(rep("big", 500L), rep(sprintf("s%02d", 1:99), each = 5L))
obs_e <- rep(10, 995L)
pa_e  <- obs_e + c(rep(0, 500L), rep(1, 495L))
pb_e  <- obs_e + c(rep(2, 500L), rep(1, 495L))
be <- block_bootstrap(obs_e, pa_e, blk_e, against = pb_e, metric = "mae", n_boot = 200L)
ok["profile_weights_give_the_big_block_its_points"] <-
  abs(be$difference[be$weights == "profile"] - (-1000 / 995)) < 1e-12
ok["block_weights_give_every_block_one_vote"] <-
  abs(be$difference[be$weights == "block"] - (-2 / 100)) < 1e-12 &&
  abs(be$estimate[be$weights == "block"] - 99 / 100) < 1e-12
ok["the_share_better_is_of_points_or_of_blocks"] <-
  abs(be$share_better[be$weights == "profile"] - 500 / 995) < 1e-12 &&
  abs(be$share_better[be$weights == "block"] - 1 / 100) < 1e-12

# A WEIGHTED CCC, BY HAND: with block weights each point weighs 1/n of its
# block, and the moments are the weighted ones.
wccc <- function(x, y, w) {
  w <- w / sum(w); mx <- sum(w * x); my <- sum(w * y)
  vx <- sum(w * (x - mx)^2); vy <- sum(w * (y - my)^2); cv <- sum(w * (x - mx) * (y - my))
  2 * cv / (vx + vy + (mx - my)^2)
}
blk_u <- c(rep("u1", 30L), rep("u2", 5L), rep("u3", 12L))
x_u <- c(rnorm(30L, 20, 3), rnorm(5L, 40, 3), rnorm(12L, 30, 3))
y_u <- x_u + rnorm(47L, sd = 4)
bw <- block_bootstrap(x_u, y_u, blk_u, metric = "ccc", weights = "block", n_boot = 200L)
ok["block_weights_give_the_weighted_ccc"] <-
  abs(bw$estimate - wccc(x_u, y_u, 1 / table(blk_u)[blk_u])) < 1e-10

# INDEPENDENT POINTS, EACH ITS OWN BLOCK: the block bootstrap is the point
# bootstrap, and n_effective is about n.
set.seed(12)
x_i <- rnorm(400L, 30, 8)
bi <- block_bootstrap(x_i, x_i + rnorm(400L, sd = 4), as.character(seq_len(400L)),
                      against = x_i + rnorm(400L, sd = 5), metric = "mae", weights = "profile",
                      n_boot = 2000L)
ok["independent_points_give_n_effective_about_n"] <- bi$n_effective > 0.85 * 400

# ERRORS IDENTICAL WITHIN BLOCKS OF 25: the difference of absolute errors is a
# block's, and 1,000 points carry the evidence of the 40 blocks. Drawn by
# points, the interval would be five times too narrow.
set.seed(13)
u <- rep(rnorm(n_b, sd = 3), each = m)
bc <- block_bootstrap(obs, obs + u, blk, against = obs, metric = "mae", weights = "profile",
                      n_boot = 2000L)
ok["errors_alike_within_blocks_give_n_effective_about_the_blocks"] <-
  bc$n_effective > 0.75 * n_b && bc$n_effective < 1.3 * n_b

# PAIRED DRAWS: a model against itself differs by exactly zero in every draw.
bz <- block_bootstrap(obs, pred_a, blk, against = pred_a, n_boot = 200L)
ok["the_draws_are_paired"] <- all(bz$difference == 0) && all(bz$ci_low == 0) &&
  all(bz$ci_high == 0)

set.seed(99)
seed_before <- .Random.seed
b1 <- block_bootstrap(obs, pred_a, blk, metric = "mae", n_boot = 300L, seed = 7L)
ok["the_session_s_random_numbers_are_left_alone"] <- identical(seed_before, .Random.seed)
b2 <- block_bootstrap(obs, pred_a, blk, metric = "mae", n_boot = 300L, seed = 7L)
ok["the_seed_reproduces_the_interval"] <- identical(b1$ci_low, b2$ci_low) &&
  identical(b1$ci_high, b2$ci_high)
ok["what_cannot_be_drawn_is_refused"] <-
  grepl("one value per point", err(block_bootstrap(obs, pred_a[-1], blk))) &&
  grepl("one block", err(block_bootstrap(obs, pred_a, rep("a", length(obs))))) &&
  grepl("100 or more", err(block_bootstrap(obs, pred_a, blk, n_boot = 10)))
ok["it_prints"] <- any(grepl("every block one vote", capture.output(print(bb))))

# ── 3. the interval as the blocks grow ───────────────────────────────────────
#
# 30 clusters of 30 points, each within 2 km, 100 km and more apart, the
# errors identical within a cluster: blocks of 1 km cut the clusters and draw
# their pieces as if independent; blocks of 50 km hold each cluster whole.
set.seed(14)
cl <- rep(seq_len(30L), each = 30L)
cx <- runif(30L, -60, -45)[cl] + rnorm(900L, sd = 0.01)
cy <- runif(30L, -30, -10)[cl] + rnorm(900L, sd = 0.01)
o_c <- 30 + rnorm(900L, sd = 8)
e_c <- rep(rnorm(30L, sd = 3), each = 30L)
bs <- block_bootstrap_by_size(cx, cy, o_c, o_c + e_c, against = o_c, sizes_km = c(1, 50),
                              n_boot = 1000L)
ok["blocks_that_cut_a_cluster_give_a_narrower_interval"] <-
  bs$width[bs$size_km == 1] < 0.7 * bs$width[bs$size_km == 50] &&
  bs$n_blocks[bs$size_km == 1] > bs$n_blocks[bs$size_km == 50]
grDevices::pdf(NULL)
size_warnings <- warned(plot(bs))
grDevices::dev.off()
ok["the_curve_draws_without_a_warning"] <- identical(size_warnings, "")

# ── 4. the correlogram ────────────────────────────────────────────────────────
#
# BY HAND: four points on a line, at 0, 1, 2 and 10 km, values 1, 1, -1, -1.
# Pairs within 1.5 km: (1,2) and (2,3), products 1 and -1, I = 0; between 1.5
# and 5 km: (1,3), I = -1; between 5 and 20 km: three pairs, I = -1/3.
cg <- spatial_correlogram(c(0, 1000, 2000, 10000), c(0, 0, 0, 0), c(1, 1, -1, -1),
                          breaks_km = c(0, 1.5, 5, 20), coords = "metres")
ok["morans_i_by_hand"] <- identical(cg$n_pairs, c(2, 1, 3)) &&
  isTRUE(all.equal(cg$moran_i, c(0, -1, -1 / 3))) && abs(cg$expected[1] + 1 / 3) < 1e-12
# Three points, the fewest a Moran's I reads: (0, 0) and (1, 0) on the equator
# are 111.2 km apart; (0, 5) is 556 and 567 km from them, beyond the classes.
cg_gc <- spatial_correlogram(c(0, 1, 0), c(0, 0, 5), c(1, -1, 0), breaks_km = c(0, 100, 120, 200))
ok["one_degree_on_the_equator_is_111_km"] <- identical(cg_gc$n_pairs, c(0, 1, 0))

cg_cl <- spatial_correlogram(cx, cy, e_c, breaks_km = c(0, 5, 50, 3000))
ok["a_field_alike_within_clusters_resembles_itself_near_and_not_far"] <-
  cg_cl$moran_i[1] > 0.95 && abs(cg_cl$moran_i[3]) < 0.2
set.seed(15)
cg_n <- spatial_correlogram(runif(600L, -60, -45), runif(600L, -30, -10), rnorm(600L),
                            breaks_km = c(0, 100, 400, 3000))
ok["noise_resembles_itself_nowhere"] <- all(abs(cg_n$moran_i) < 0.1, na.rm = TRUE)
ok["the_correlogram_refuses_what_it_cannot_read"] <-
  grepl("constant", err(spatial_correlogram(cx, cy, rep(1, length(cx))))) &&
  grepl("increasing", err(spatial_correlogram(cx, cy, e_c, breaks_km = c(10, 5))))
grDevices::pdf(NULL)
cg_warnings <- warned(plot(cg_cl))
grDevices::dev.off()
ok["the_correlogram_prints_and_draws"] <- identical(cg_warnings, "") &&
  any(grepl("the scale blocks should exceed", capture.output(print(cg_cl))))

cat(sprintf("  errors alike within 40 blocks of 25: n_effective %.0f of %d points\n",
            bc$n_effective, n_b * m))
cat(sprintf("  one big block against 99 small ones: %+.3f per profile, %+.3f per block\n",
            be$difference[be$weights == "profile"], be$difference[be$weights == "block"]))
cat(sprintf("  clusters cut by 1 km blocks: width %.3f, held whole by 50 km blocks: %.3f\n",
            bs$width[bs$size_km == 1], bs$width[bs$size_km == 50]))

.report(ok, "test_block_bootstrap")
