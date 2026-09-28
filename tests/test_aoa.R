# Unit test: the dissimilarity index and the area of applicability
#
# The AOA answers "where does the cross-validated error apply?". Its failure
# mode is the worst kind: a threshold that is too generous marks the whole map
# as trustworthy, which is exactly the answer everyone wants to see and the one
# nothing else would contradict.
#
# So the fixture is built so the right answer is known by construction: a
# training cloud in one place, and probe points at known distances from it.
#
# Verified:
#   1. DI is 0 for a point identical to a training point
#   2. DI grows monotonically as a probe moves away
#   3. DI is scale-free in the right sense: multiplying the whole space by a
#      constant leaves every DI unchanged
#   4. a point one average-pairwise-distance away has DI near 1
#   5. weights change distances in the direction they say they do
#   6. the threshold comes from ACROSS folds, never within
#   7. far points fall outside the AOA and near points inside
#   8. degenerate input is refused, not answered
#
# Run: source("D:/.../tests/test_aoa.R")     (no tensors: base R on small matrices)

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
.load_framework(root)

ok <- c()

# =============================================================================
# The fixture: 200 training points in a unit Gaussian cloud in 5 dimensions,
# and probes placed at known multiples of the cloud's own scale.
# =============================================================================

set.seed(20260915)
P <- 5L
xtr <- matrix(stats::rnorm(200L * P), ncol = P)
ref <- di_reference(xtr)

ok["reference_keeps_the_dimensions"] <- ref$p == P && ref$n == 200L
ok["avg_dist_is_positive"] <- ref$avg_dist > 0
# For a standard normal cloud in p dimensions the mean pairwise distance is
# close to sqrt(2p) -- here sqrt(10) ~ 3.16. This is a sanity band, not an
# identity: it catches a missing sqrt or a factor of two, which is what a
# distance implementation gets wrong.
ok["avg_dist_matches_theory"] <- abs(ref$avg_dist - sqrt(2 * P)) < 0.5

# -- 1. a training point is at distance zero from itself ----------------------
di_self <- dissimilarity_index(ref, xtr[1:5, , drop = FALSE])
ok["di_of_a_training_point_is_zero"] <- all(di_self < 1e-8)

# -- 2. monotone in distance --------------------------------------------------
# Probes marching away from the cloud along one axis. Beyond the cloud's own
# radius every step must increase the distance to the NEAREST training point.
far <- c(5, 10, 20, 40)
probes <- t(vapply(far, function(d) c(d, 0, 0, 0, 0), numeric(P)))
di_probes <- dissimilarity_index(ref, probes)
ok["di_increases_with_distance"] <- all(diff(di_probes) > 0)
ok["di_of_a_far_probe_is_large"]  <- di_probes[length(di_probes)] > 5

# -- 3. scale invariance ------------------------------------------------------
# The DI is a ratio of a distance to the mean pairwise distance, so blowing the
# whole space up by 100 must leave it EXACTLY where it was. A DI that moves
# under rescaling is reporting units, not dissimilarity.
ref100 <- di_reference(xtr * 100)
di100  <- dissimilarity_index(ref100, probes * 100)
ok["di_is_invariant_to_rescaling"] <-
  isTRUE(all.equal(di_probes, di100, tolerance = 1e-6))

# -- 4. one average pairwise distance away is DI ~ 1 --------------------------
# Built to be exact: take a training point and step avg_dist along one axis.
# Its nearest neighbour is then at most avg_dist away, so DI <= 1, and with a
# sparse cloud it should be close to 1 rather than far below.
step <- xtr[1, , drop = FALSE]
step[1, 1] <- step[1, 1] + ref$avg_dist
di_step <- dissimilarity_index(ref, step)
ok["di_one_avg_dist_away_is_at_most_one"] <- di_step <= 1 + 1e-8
ok["di_one_avg_dist_away_is_not_tiny"]    <- di_step > 0.2

# -- 5. weights do what they claim --------------------------------------------
# Put all the weight on dimension 1 and none elsewhere: a probe displaced along
# dimension 1 must become MORE dissimilar, and the same displacement along
# dimension 5 must become less.
w <- c(10, rep(0.01, P - 1L))
ref_w <- di_reference(xtr, weights = w)
p1 <- matrix(c(4, 0, 0, 0, 0), nrow = 1L)
p5 <- matrix(c(0, 0, 0, 0, 4), nrow = 1L)
ok["weight_raises_the_weighted_axis"] <-
  dissimilarity_index(ref_w, p1) > dissimilarity_index(ref_w, p5)
ok["unweighted_treats_axes_alike"] <- {
  # Loose on purpose. The claim is "not WILDLY different", against a weighted
  # case that differs by orders of magnitude -- a tight bound here would only
  # measure the fixture cloud's own asymmetry and fail on an unlucky draw.
  a <- dissimilarity_index(ref, p1); b <- dissimilarity_index(ref, p5)
  abs(a - b) / max(a, b) < 0.4
}
ok["weights_are_validated"] <- inherits(
  tryCatch(di_reference(xtr, weights = c(1, 2)), error = function(e) e), "error")
ok["negative_weights_refused"] <- inherits(
  tryCatch(di_reference(xtr, weights = rep(-1, P)), error = function(e) e),
  "error")

# =============================================================================
# 6. The threshold comes from ACROSS folds
#
# This is the property that makes the threshold mean something, and it is
# invisible in the output: a threshold computed within folds would use each
# point's own neighbours -- including the ones it trained beside -- and come
# out far too small, marking most of a map as outside the AOA for the wrong
# reason.
#
# The fixture makes it detectable: two tight clusters, one per fold. Within a
# fold every point has a neighbour at ~0 distance; ACROSS folds the nearest
# neighbour is a cluster-width away. The two answers differ by orders of
# magnitude, so an implementation that confuses them cannot pass.
# =============================================================================

clu <- rbind(matrix(stats::rnorm(100L * P, mean = 0,  sd = 0.05), ncol = P),
             matrix(stats::rnorm(100L * P, mean = 10, sd = 0.05), ncol = P))
fold2 <- rep(1:2, each = 100L)
ref_c <- di_reference(clu)
th    <- aoa_threshold(ref_c, fold2)
cv_di <- attr(th, "cv_di")

ok["threshold_is_finite"] <- is.finite(as.numeric(th)) && as.numeric(th) > 0
ok["cv_di_has_one_entry_per_training_row"] <- length(cv_di) == nrow(clu)
# Every cross-validated distance must be the BETWEEN-cluster one, not the
# within-cluster ~0. In DI units that is about 10 / avg_dist.
ok["threshold_uses_the_other_fold"] <- all(cv_di > 0.5 * 10 / ref_c$avg_dist)
ok["threshold_is_not_the_within_fold_distance"] <- min(cv_di) > 0.1

ok["threshold_needs_more_than_one_fold"] <- inherits(
  tryCatch(aoa_threshold(ref_c, rep(1L, nrow(clu))), error = function(e) e),
  "error")
ok["threshold_checks_the_fold_length"] <- inherits(
  tryCatch(aoa_threshold(ref_c, c(1L, 2L)), error = function(e) e), "error")

# =============================================================================
# 7. Inside and outside
# =============================================================================

th_n  <- aoa_threshold(ref, rep_len(1:4, nrow(xtr)))
di_in <- dissimilarity_index(ref, xtr[1:20, , drop = FALSE])
ok["training_points_are_inside"] <- all(inside_aoa(di_in, th_n))
ok["a_far_point_is_outside"] <-
  !inside_aoa(dissimilarity_index(ref, matrix(rep(50, P), nrow = 1L)), th_n)
ok["inside_aoa_is_elementwise"] <-
  length(inside_aoa(c(0, 1, 99), th_n)) == 3L

# =============================================================================
# 8. Degenerate input is refused, not answered
# =============================================================================

ok["identical_training_rows_are_refused"] <- inherits(
  tryCatch(di_reference(matrix(1, nrow = 10L, ncol = 3L)),
           error = function(e) e), "error")
ok["one_row_is_refused"] <- inherits(
  tryCatch(di_reference(matrix(1:3, nrow = 1L)), error = function(e) e),
  "error")
ok["column_mismatch_is_refused"] <- inherits(
  tryCatch(dissimilarity_index(ref, matrix(0, nrow = 2L, ncol = P + 1L)),
           error = function(e) e), "error")

# The chunked path and the whole-matrix path must agree exactly -- chunking is
# an implementation detail and must never be visible in the answer.
big <- matrix(stats::rnorm(500L * P), ncol = P)
ok["chunking_changes_nothing"] <- isTRUE(all.equal(
  dissimilarity_index(ref, big, chunk = 10000L),
  dissimilarity_index(ref, big, chunk = 37L), tolerance = 1e-10))

# print_aoa() is a reporting function, and a reporting function that throws
# turns a finished analysis into a lost one.
ok["print_aoa_does_not_throw"] <- !inherits(
  tryCatch(utils::capture.output(
    print_aoa(dissimilarity_index(ref, big), th_n, "probe points")),
    error = function(e) e), "error")

# ── the reference of a fitted model, built once (aoa_reference) ──────────────
#
# The map's DI, the AOA and the level-and-DI interval all measure against the
# same reference; built by one function, a calibration point and a map pixel
# cannot end up measured against two different ones.
set.seed(606)
n_r <- 120L
preds_r <- paste0("v", 1:4)
pts_r <- tibble::tibble(sample_id = seq_len(n_r), profile_id = seq_len(n_r),
                        x = stats::runif(n_r), y = stats::runif(n_r))
for (pr in preds_r) pts_r[[pr]] <- stats::rnorm(n_r, 50, 10)
plan_r <- random_folds(pts_r, k = 3L, test_frac = 0.2, seed = 1L)
qc_r   <- make_qc_table(preds_r)
sc_r   <- tibble::tibble(predictor = preds_r, center = 50, scale = 10)
aref   <- aoa_reference(pts_r, preds_r, qc_r, sc_r, plan_r)
test_pos <- plan_r$folds[[1]]$test
ok["the_reference_leaves_the_test_set_out"] <-
  length(test_pos) > 0L && !any(aref$cv$sample_id %in% pts_r$sample_id[test_pos])
ok["a_point_of_the_reference_has_di_zero"] <-
  all(aoa_di(aref, as.matrix(pts_r[aref$cv$sample_id[1:5], preds_r])) < 1e-6)
ok["the_cv_di_is_the_one_the_threshold_uses"] <-
  identical(aref$cv$cv_di, as.numeric(attr(aref$threshold, "cv_di")))
ok["a_permuted_scaling_is_refused"] <- inherits(
  try(aoa_reference(pts_r, preds_r, qc_r, sc_r[4:1, ], plan_r), silent = TRUE), "try-error")

cat(sprintf("  avg pairwise distance    : %.3f (theory sqrt(2p) = %.3f)\n",
            ref$avg_dist, sqrt(2 * P)))
cat(sprintf("  AOA threshold            : %.3f | training points inside: %.0f%%\n",
            as.numeric(th_n), 100 * mean(inside_aoa(
              dissimilarity_index(ref, xtr), th_n))))

.report(ok, "test_aoa")
