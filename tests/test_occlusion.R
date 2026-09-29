# Unit test: spatial occlusion measures what it claims to measure
#
# WHY THIS FILE EXISTS.
#
# spatial_occlusion() is the framework's answer to the question the project is
# built around -- does the network use the neighbourhood, or only the centre
# pixel? A diagnostic that answers that question WRONGLY is worse than not
# having it, because the answer is exactly the sort nobody double-checks: it
# agrees with whatever the reader already suspects.
#
# So the end-to-end tests here use models whose answer is known by construction.
# A module that reads only the centre pixel MUST report that hiding the centre
# costs everything and hiding the context costs nothing; a module that reads
# only ring 1 must report the mirror image. If the wiring is wrong anywhere --
# the mask, the permutation, the cache copy, the loader, the metric -- one of
# those two comes out wrong.
#
# Run: source("<package root>/tests/test_occlusion.R")

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

# ── 1. patch_ring_index ───────────────────────────────────────────────────────
#
# Ring d of a patch has 8d pixels, for every d >= 1. That closed form is the
# cheapest possible check that the rings are rings and not something else that
# happens to look like them at w = 3.

r5 <- patch_ring_index(5L)
ok["ring_matrix_has_the_right_shape"] <- identical(dim(r5), c(5L, 5L))
ok["centre_is_ring_zero"]             <- r5[3, 3] == 0L
ok["corner_is_the_outermost_ring"]    <- r5[1, 1] == 2L
ok["ring_counts_follow_8d"] <- all(vapply(c(3L, 5L, 9L, 15L), function(w) {
  r <- patch_ring_index(w)
  all(vapply(seq_len((w - 1L) %/% 2L),
             function(d) sum(r == d) == 8L * d, logical(1)))
}, logical(1)))
ok["rings_cover_the_patch"] <- all(vapply(c(3L, 9L, 15L), function(w) {
  sum(table(patch_ring_index(w))) == w * w
}, logical(1)))

# An even window has no single centre. Choosing one of the four silently would
# put every ring half a pixel off, and nothing downstream would notice.
ok["even_window_is_refused"] <- inherits(
  try(patch_ring_index(4L), silent = TRUE), "try-error")

# ── 2. occlude_patch_array ────────────────────────────────────────────────────

set.seed(11)
n <- 30L; ch <- 3L; w <- 5L
# TENSORS, because that is what build_fold_cache() puts in the cache. The first
# version of this file used base R arrays and passed nothing -- occlude_patch_array()
# had been written against an array too, so both were wrong in the same way and
# the test could not see it. A fixture that does not match the real
# representation tests the fixture.
arr  <- torch_tensor(array(rnorm(n * ch * w * w), dim = c(n, ch, w, w)),
                     dtype = torch_float())
ring <- patch_ring_index(w)
mask <- ring >= 2L                      # the outer rim

# Reading the masked pixels back out, the torch way: a [w, w] bool mask
# broadcast over [n, c, w, w], then the selected values. Written once so the
# test cannot check a different indexing than the code uses.
at <- function(a, m) {
  as.numeric(a$masked_select(
    torch_tensor(m, dtype = torch_bool())$expand_as(a)))
}

z <- occlude_patch_array(arr, mask, method = "zero")
ok["zero_hides_exactly_the_mask"] <- all(at(z, mask) == 0)
ok["zero_leaves_the_rest_untouched"] <-
  isTRUE(all.equal(at(z, !mask), at(arr, !mask)))

p <- occlude_patch_array(arr, mask, method = "permute", seed = 5L)
ok["permute_leaves_the_rest_untouched"] <-
  isTRUE(all.equal(at(p, !mask), at(arr, !mask)))

# THE PROPERTY THAT MAKES IT A PERMUTATION AND NOT NOISE: every value that was
# in the hidden region is still there, once. Only its owner changed. A method
# that invented values would change the marginal distribution, and the measured
# drop would then include "this input is impossible" as well as "this region
# was uninformative".
ok["permute_preserves_every_value"] <-
  isTRUE(all.equal(sort(at(p, mask)), sort(at(arr, mask))))

# ...and each sample's hidden region comes from ONE donor, kept whole. Drawing
# per pixel would destroy the region's own texture, so the drop would confound
# "uninformative" with "no longer a landscape".
rim_of <- function(t) {
  mm <- torch_tensor(mask, dtype = torch_bool())$expand_as(t)
  matrix(as.numeric(t$masked_select(mm)), nrow = n, byrow = TRUE)
}
rim_p <- rim_of(p); rim_a <- rim_of(arr)
donor_of <- vapply(seq_len(n), function(i) {
  hit <- which(vapply(seq_len(n), function(j)
    isTRUE(all.equal(rim_p[i, ], rim_a[j, ])), logical(1)))
  if (length(hit) == 1L) hit else NA_integer_
}, integer(1))
ok["each_sample_takes_one_whole_donor"] <- !anyNA(donor_of)
ok["no_sample_donates_to_itself"]       <- !any(donor_of == seq_len(n))

ok["empty_mask_is_a_no_op"] <- {
  o <- occlude_patch_array(arr, ring < 0L)
  isTRUE(all.equal(as.numeric(o), as.numeric(arr)))
}
ok["mismatched_mask_is_refused"] <- inherits(
  try(occlude_patch_array(arr, patch_ring_index(9L) >= 1L), silent = TRUE),
  "try-error")
# A base R array is not what the cache holds, and accepting one silently is how
# the first version of this pair stayed broken.
ok["a_base_array_is_refused"] <- inherits(
  try(occlude_patch_array(array(0, dim = c(n, ch, w, w)), mask), silent = TRUE),
  "try-error")

# ── 3-4. END TO END, against models whose answer is known ────────────────────
#
# The fixture: the target IS the centre pixel of channel 1 for one model, and
# the mean of ring 1 for the other. Both are exactly predictable, so the
# baseline CCC is ~1 and every drop is attributable.

nv <- 64L
wv <- 5L
cv <- 2L
# POSITIVE VALUES, because predict_loader() floors predictions at zero: the
# framework's target is a stock. An rnorm fixture put half the targets below
# zero, the clamp flattened those predictions, and the baseline CCC came out
# near 0.7 instead of 1 -- so every delta was measured against a broken
# reference, and the failure looked like the occlusion was wrong rather than
# the fixture. A fixture has to live in the domain the code assumes.
set.seed(7)
patches <- array(runif(nv * cv * wv * wv, min = 1, max = 6),
                 dim = c(nv, cv, wv, wv))
ring_v  <- patch_ring_index(wv)
centre_val <- patches[, 1, 3, 3]
ring1_val  <- apply(patches[, 1, , , drop = FALSE], 1,
                    function(s) mean(s[ring_v == 1L]))

# The cache mirrors build_fold_cache() exactly: tensors for the windows, and y
# as a [n, 1] float tensor, because .make_loaders_from_cache() hands both
# straight to torch::tensor_dataset().
make_cache <- function(y) {
  role <- list(w05 = torch_tensor(patches, dtype = torch_float()),
               y = torch_tensor(as.numeric(y),
                                dtype = torch_float())$view(c(-1L, 1L)))
  list(train = role, validation = role)
}
points_for <- function(y) tibble(
  profile_id = sprintf("p%02d", seq_len(nv)), sample_id = seq_len(nv),
  target_native = as.numeric(y), target_transform = as.numeric(y))

cfg <- tibble(config_id = "cfg_001", window_sizes = list(wv),
              batch_size = 32L)

# A module is used rather than a closure because predict_loader() calls
# model$eval() and do.call(model, inputs) -- the occlusion has to work against
# the real interface, not against a stand-in for it.
centre_reader <- nn_module(
  "centre_reader",
  initialize = function(w) { self$c0 <- (w + 1L) %/% 2L },
  forward = function(x) x[, 1, self$c0, self$c0]
)
ring_reader <- nn_module(
  "ring_reader",
  initialize = function(w) {
    self$m <- torch_tensor(patch_ring_index(w) == 1L, dtype = torch_float())
  },
  forward = function(x) {
    (x[, 1, , ] * self$m$to(device = x$device))$sum(dim = c(2, 3)) / 8
  }
)

dev <- torch_device("cpu")

occ_centre <- spatial_occlusion(
  centre_reader(wv), make_cache(centre_val), cfg, points_for(centre_val),
  role = "validation", device = dev, seed = 3L)
tc <- occ_centre$table

ok["baseline_reproduces_the_target"] <-
  tc$ccc[tc$scope == "baseline"] > 0.99

# The clamp must be honoured, and must be overridable -- a target that can go
# negative would otherwise be floored by the diagnostic itself, and the floor
# would look exactly like a finding.
#
# THE INPUT MOVES WITH THE TARGET. The first version of this check shifted the
# target by -3 and left the patches alone, so the model (which returns the raw
# centre pixel) was off by a constant 3. CCC penalises bias, so the baseline
# failed for that reason and not for the clamp -- a test that fails for the
# wrong reason is a test that will be "fixed" in the wrong place.
patches_neg <- patches - 3            # centre pixel now spans about -2 .. 3
centre_neg  <- patches_neg[, 1, 3, 3]
cache_neg <- list(
  train = list(w05 = torch_tensor(patches_neg, dtype = torch_float()),
               y = torch_tensor(as.numeric(centre_neg),
                                dtype = torch_float())$view(c(-1L, 1L))))
cache_neg$validation <- cache_neg$train

ok["the_negative_fixture_really_is_negative"] <- any(centre_neg < 0)

ok["clamp_floors_by_default"] <- {
  o <- spatial_occlusion(centre_reader(wv), cache_neg, cfg,
                         points_for(centre_neg), role = "validation",
                         device = dev, seed = 3L)
  o$table$ccc[o$table$scope == "baseline"] < 0.99
}
ok["clamp_can_be_opened"] <- {
  o <- spatial_occlusion(centre_reader(wv), cache_neg, cfg,
                         points_for(centre_neg), role = "validation",
                         device = dev, seed = 3L, clamp = c(-Inf, Inf))
  o$table$ccc[o$table$scope == "baseline"] > 0.99
}


# THE ASSERTION THE FILE IS FOR. A centre-only model must lose nothing when the
# whole neighbourhood is permuted away.
ok["centre_model_loses_nothing_to_context"] <-
  abs(tc$delta_ccc[tc$scope == "context_all"]) < 1e-6
ok["centre_model_collapses_without_its_pixel"] <-
  tc$delta_ccc[tc$scope == "centre_only_hidden"] < -0.5
ok["centre_model_is_flat_across_rings"] <-
  all(abs(tc$delta_ccc[grepl("^ring_", tc$scope)]) < 1e-6)

# ...and the mirror image, so the test cannot pass by reporting "no effect"
# for everything.
occ_ring <- spatial_occlusion(
  ring_reader(wv), make_cache(ring1_val), cfg, points_for(ring1_val),
  role = "validation", device = dev, seed = 3L)
tr <- occ_ring$table

ok["ring_model_reproduces_its_target"] <- tr$ccc[tr$scope == "baseline"] > 0.99
ok["ring_model_loses_nothing_to_its_centre"] <-
  abs(tr$delta_ccc[tr$scope == "centre_only_hidden"]) < 1e-6
ok["ring_model_collapses_without_ring_1"] <-
  tr$delta_ccc[tr$scope == "ring_01"] < -0.5
ok["ring_model_ignores_ring_2"] <-
  abs(tr$delta_ccc[tr$scope == "ring_02"]) < 1e-6
ok["ring_model_loses_everything_to_context"] <-
  tr$delta_ccc[tr$scope == "context_all"] < -0.5

# The two models must not produce the same report -- that would be the failure
# mode where the occlusion measures nothing and says so consistently.
ok["the_two_models_disagree"] <-
  tc$delta_ccc[tc$scope == "context_all"] != tr$delta_ccc[tr$scope == "context_all"]

# The pixel counts are a cheap cross-check on the masks: a 5x5 patch has 24
# non-centre pixels, and ring 1 has 8.
ok["context_hides_every_non_centre_pixel"] <-
  tc$n_pixels_hidden[tc$scope == "context_all"] == wv * wv - 1L
ok["ring_one_hides_eight_pixels"] <-
  tc$n_pixels_hidden[tc$scope == "ring_01"] == 8L

# "zero" must reach the same verdict here. The two methods exist to disagree on
# real data -- that disagreement is a finding about off-distribution
# sensitivity -- but on a model that provably ignores the context, both must
# report nothing.
occ_zero <- spatial_occlusion(
  centre_reader(wv), make_cache(centre_val), cfg, points_for(centre_val),
  role = "validation", device = dev, method = "zero")
ok["zero_method_agrees_on_a_centre_model"] <-
  abs(occ_zero$table$delta_ccc[occ_zero$table$scope == "context_all"]) < 1e-6

cat(sprintf("  centre-only model        : context %+.4f | centre %+.4f CCC\n",
            tc$delta_ccc[tc$scope == "context_all"],
            tc$delta_ccc[tc$scope == "centre_only_hidden"]))
cat(sprintf("  ring-1 model             : context %+.4f | centre %+.4f CCC\n",
            tr$delta_ccc[tr$scope == "context_all"],
            tr$delta_ccc[tr$scope == "centre_only_hidden"]))


# =============================================================================
# THE WRAPPER'S ARGUMENT SHAPE
#
# occlusion_report() passed fold_points_valid(store, idx) -- the whole named
# list(train =, validation =, test =) -- where spatial_occlusion() wants ONE
# role's tibble. The wrapper could never have worked.
#
# It survived because every test above calls spatial_occlusion() DIRECTLY with a
# tibble. The inner function was covered from three angles and the thing a user
# would actually call had never been called by anything. That is the defect
# class, not the defect: a wrapper is not tested by testing what it wraps.
#
# occlusion_report() itself needs a real store, a fold plan and a checkpoint on
# disk, so what is asserted here is the CONTRACT it violated -- and that the
# refusal names the fix rather than failing three frames down in
# check_point_contract() with "profile_id is missing", which sends the reader
# to the point table instead of to the argument.
# =============================================================================

full_list <- list(train      = points_for(centre_val)[1:10, ],
                  validation = points_for(centre_val),
                  test       = points_for(centre_val)[1:5, ])

ok["the_whole_role_list_is_refused"] <- inherits(
  try(spatial_occlusion(centre_reader(wv), make_cache(centre_val), cfg,
                        full_list, role = "validation", device = dev),
      silent = TRUE), "try-error")

ok["that_refusal_names_the_fix"] <- {
  e <- tryCatch(spatial_occlusion(centre_reader(wv), make_cache(centre_val), cfg,
                                  full_list, role = "validation", device = dev),
                error = function(e) conditionMessage(e))
  grepl("fold_points_valid", e, fixed = TRUE) && grepl("[[", e, fixed = TRUE)
}

# A tibble that is simply missing a contract column must still be caught, and
# by the contract check rather than by the list guard.
ok["an_incomplete_tibble_is_still_refused"] <- inherits(
  try(spatial_occlusion(centre_reader(wv), make_cache(centre_val), cfg,
                        dplyr::select(points_for(centre_val), -target_native),
                        role = "validation", device = dev),
      silent = TRUE), "try-error")

# ...and the wrapper now takes `role` as a real argument, so it can subset.
ok["occlusion_report_has_a_role_argument"] <-
  "role" %in% names(formals(occlusion_report))

.report(ok, "test_occlusion")
