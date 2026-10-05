# Unit test: kNNDM folds, and the argument contract that holds without CAST
#
# WHAT IS AND IS NOT TESTED HERE.
#
# The ALGORITHM is not re-derived and not re-checked: knndm_folds() calls
# CAST::knndm(), the reference implementation from the paper, on purpose. A
# published cross-validation method re-coded locally is a method that quietly
# differs from the one being cited, and no test written here would catch that
# because the test would share the same misunderstanding.
#
# What is tested is everything AROUND the call, which is where this framework
# can be wrong on its own:
#
#   - the argument contract, including the refusal to guess `predpoints`
#   - the projection, because comparing distances in degrees on global data is
#     comparing 111 km with 20 km and calling both "one unit"
#   - the fold plan that comes back: a partition, no leak between roles, and W
#     recorded where a reader will see it
#   - the front-end spec routing to the right constructor
#
# The parts needing CAST are skipped, loudly, when it is not installed -- the
# argument contract runs either way, which is why those checks were moved ahead
# of the dependency check in knndm_folds().
#
# Run: source("<package root>/tests/test_knndm.R")

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

# A global-ish spread, so the projection has something to do: at 60 degrees a
# degree of longitude is half what it is at the equator, and a method that
# compares distances in degrees would treat those as the same separation.
set.seed(21)
n_sites <- 40L; per_site <- 8L
sites <- tibble(
  site = seq_len(n_sites),
  cx = runif(n_sites, -60, 60),
  cy = runif(n_sites, -55, 65))
meta <- sites %>%
  dplyr::slice(rep(seq_len(n_sites), each = per_site)) %>%
  dplyr::mutate(
    x = cx + rnorm(dplyr::n(), 0, 0.15),
    y = cy + rnorm(dplyr::n(), 0, 0.15),
    sample_id = seq_len(dplyr::n()),
    profile_id = sprintf("p%04d", seq_len(dplyr::n()))) %>%
  dplyr::select(sample_id, profile_id, site, x, y)

predpts <- tibble(x = runif(600, -60, 60), y = runif(600, -55, 65))

# ── 1. the argument contract, with or without CAST ───────────────────────────

# THE REFUSAL THAT MATTERS. kNNDM without a prediction area is an expensive
# random split wearing the name of a spatial method. Defaulting it would be the
# single most damaging convenience this file could offer.
err <- tryCatch(knndm_folds(meta, k = 3L), error = function(e) e)
ok["refuses_without_predpoints"] <- inherits(err, "error")
ok["that_refusal_explains_why"] <-
  grepl("WHERE THE MAP WILL BE PREDICTED", conditionMessage(err), fixed = TRUE)

ok["refuses_k_below_2"] <- inherits(
  try(knndm_folds(meta, k = 1L, predpoints = predpts), silent = TRUE),
  "try-error")
ok["refuses_meta_without_coordinates"] <- inherits(
  try(knndm_folds(dplyr::select(meta, sample_id), k = 3L,
                  predpoints = predpts), silent = TRUE), "try-error")

# THE MAP SAMPLE COMES BACK AT ABOUT THE SIZE ASKED, however much of the
# raster is NA. A regular sample drops the cells it lands on without data, and
# a raster that was 70% sea returned 30% of the points asked for. Here the
# top 30 rows of 100 hold data.
land <- terra::rast(nrows = 100, ncols = 100, xmin = 0, xmax = 10, ymin = 0, ymax = 10,
                    crs = "EPSG:4326")
terra::values(land) <- ifelse(rep(seq_len(100), each = 100) <= 30, 1, NA)
ps <- prediction_sample(land, size = 500L)
on_land <- terra::extract(land, as.matrix(ps))
ok["prediction_sample_counts_only_cells_with_data"] <-
  nrow(ps) >= 450L && nrow(ps) <= 800L && all(!is.na(on_land[[ncol(on_land)]]))

# A size whose square overflows an integer (50000L) still draws.
ok["prediction_sample_takes_a_large_integer_size"] <-
  nrow(prediction_sample(land, size = 50000L)) > 0L

has_sf   <- requireNamespace("sf", quietly = TRUE)
has_cast <- requireNamespace("CAST", quietly = TRUE)

# ── 2. the projection ────────────────────────────────────────────────────────

if (has_sf) {
  m <- project_xy(meta$x, meta$y)
  ok["projection_returns_two_columns"] <- identical(dim(m), c(nrow(meta), 2L))
  ok["projection_is_in_metres"] <- max(abs(m)) > 1e5

  # THE POINT OF PROJECTING. Two pairs one degree of longitude apart, one at the
  # equator and one at 60 degrees north, are the same distance in degrees and
  # roughly half as far apart on the ground. A method comparing distances must
  # see the difference.
  eq <- project_xy(c(0, 1), c(0, 0))
  hi <- project_xy(c(0, 1), c(60, 60))
  d_eq <- sqrt(sum((eq[1, ] - eq[2, ])^2))
  d_hi <- sqrt(sum((hi[1, ] - hi[2, ])^2))
  ok["projection_shrinks_a_degree_at_high_latitude"] <- d_hi < 0.7 * d_eq

  # project_to = NULL means "already projected" and must not silently reproject.
  raw <- project_xy(meta$x, meta$y, to = NULL)
  ok["no_projection_leaves_coordinates_alone"] <-
    isTRUE(all.equal(as.numeric(raw[, 1]), as.numeric(meta$x)))
} else {
  cat("  sf missing               : projection checks skipped\n")
}

# ── 3-4. the plan, and the front end ─────────────────────────────────────────

if (has_cast && has_sf) {
  plan <- knndm_folds(meta, k = 3L, predpoints = predpts, seed = 7L)

  ok["plan_has_the_right_class"]  <- inherits(plan, "fold_plan")
  ok["plan_has_k_folds"]          <- plan$n_folds == 3L
  ok["plan_method_is_named"]      <- identical(plan$method, "knndm_folds")

  # Every row is validation exactly once, and never trains on the fold it is
  # validated in. check_fold_plan() is the framework's own proof, so it is used
  # rather than re-implemented here.
  sizes <- check_fold_plan(plan, meta = meta)
  ok["check_fold_plan_accepts_it"] <- is.data.frame(sizes) && nrow(sizes) == 3L
  ok["every_row_validates_once"] <-
    identical(sort(unlist(lapply(plan$folds, function(f) f$validation))),
              seq_len(nrow(meta)))
  ok["no_row_trains_and_validates"] <- all(vapply(plan$folds, function(f)
    length(intersect(f$train, f$validation)) == 0L, logical(1)))

  # W IS THE QUALITY OF THE PLAN and has to reach the reader. A fold plan whose
  # W is large is matching prediction badly, and that is the one number saying
  # so -- kept in params, which print.fold_plan() shows.
  ok["W_is_recorded"] <- is.numeric(plan$params$W) && is.finite(plan$params$W)
  ok["projection_is_recorded"] <- grepl("moll", plan$params$projection)

  # A frozen test set must be honoured exactly, because comparing two plans on
  # different held-out data compares two experiments.
  frozen <- meta$sample_id[1:40]
  plan_t <- knndm_folds(meta, k = 3L, predpoints = predpts,
                        test_ids = frozen, seed = 7L)
  ok["frozen_test_is_exact"] <-
    identical(sort(meta$sample_id[plan_t$folds[[1]]$test]), sort(frozen))
  ok["frozen_test_is_out_of_every_fold"] <- all(vapply(plan_t$folds,
    function(f) length(intersect(f$test, c(f$train, f$validation))) == 0L,
    logical(1)))
  ok["frozen_test_leaves_k_folds"] <- plan_t$n_folds == 3L

  # hold_out_test carves one kNNDM fold out of k + 1. It is off by default,
  # because a test set that changes with k is not a frozen test set.
  plan_h <- knndm_folds(meta, k = 3L, predpoints = predpts,
                        hold_out_test = TRUE, seed = 7L)
  ok["hold_out_test_produces_a_test_set"] <-
    length(plan_h$folds[[1]]$test) > 0L
  ok["hold_out_test_is_off_by_default"] <-
    length(plan$folds[[1]]$test) == 0L
  ok["held_out_test_is_roughly_one_of_k_plus_1"] <- {
    share <- length(plan_h$folds[[1]]$test) / nrow(meta)
    share > 0.10 && share < 0.45
  }

  # The calibration set, carved as the test set is: a kNNDM fold of k + 1 over
  # what the test left, in no fold; or frozen ids, never test ids.
  plan_c <- knndm_folds(meta, k = 3L, predpoints = predpts, hold_out_test = TRUE,
                        hold_out_calibration = TRUE, seed = 7L)
  ok["hold_out_calibration_carves_a_set_in_no_fold"] <-
    length(plan_c$calibration) > 0L &&
    length(intersect(plan_c$calibration, plan_c$folds[[1]]$test)) == 0L &&
    !any(plan_c$calibration %in% unlist(lapply(plan_c$folds, function(f) c(f$train, f$validation)))) &&
    is.data.frame(check_fold_plan(plan_c, meta = meta))
  plan_cf <- knndm_folds(meta, k = 3L, predpoints = predpts, seed = 7L,
                         calibration_ids = meta$sample_id[plan_c$calibration])
  ok["frozen_calibration_ids_are_taken_as_given"] <-
    setequal(plan_cf$calibration, plan_c$calibration)

  # TWO FOLDS NEED A maxp ABOVE ONE HALF. CAST's default is 0.5 and its bound
  # is strict, so kNNDM in two folds stopped inside CAST with a message that
  # named neither k nor where maxp goes. It is refused before CAST, by name --
  # and with a maxp in range, two folds are cut.
  e_k2 <- tryCatch(knndm_folds(meta, k = 2L, predpoints = predpts),
                   error = function(e) conditionMessage(e))
  ok["two_folds_under_the_default_maxp_are_refused_by_name"] <-
    grepl("maxp", e_k2, fixed = TRUE) && grepl("k >= 3", e_k2, fixed = TRUE)
  plan_k2 <- knndm_folds(meta, k = 2L, predpoints = predpts, maxp = 0.6, seed = 7L)
  ok["two_folds_are_cut_with_a_maxp_above_a_half"] <-
    plan_k2$n_folds == 2L && is.data.frame(check_fold_plan(plan_k2, meta = meta))

  # The front end must route to this constructor and not to a neighbour. The
  # switch() in resolve_resampling() selects by name, and a spec whose kind is
  # not a string once selected by POSITION instead -- a spatial request that
  # came back as a holdout, with no error.
  spec <- knndm_cv(k = 3L, predpoints = predpts, seed = 7L)
  ok["spec_is_a_resample_spec"] <- inherits(spec, "resample_spec")
  ok["spec_kind_is_the_string_knndm"] <-
    is.character(spec$kind) && identical(spec$kind, "knndm")

  fake <- structure(list(store = list(meta = meta, window_sizes = 3L),
                         cell_size = 0.00224579811173295),
                    class = "dsm_data")
  plan_api <- resolve_resampling(spec, fake, verbose = FALSE)
  ok["front_end_builds_a_knndm_plan"] <-
    identical(plan_api$method, "knndm_folds")
  ok["front_end_plan_matches_the_direct_call"] <-
    identical(plan_api$assignment$fold, plan$assignment$fold)

  # ── 5. the refit: the final model's validation, by the same criterion ─────
  #
  # dsm_final() stops the final fit on a validation set carved the way the
  # tuning folds were (refit_split()). For kNNDM that is kNNDM again, over the
  # non-test rows, against the same prediction points -- which the plan
  # therefore keeps beside itself.
  ok["the_plan_keeps_what_a_refit_needs"] <-
    isTRUE(all.equal(as.data.frame(plan_t$knndm$predpoints), as.data.frame(predpts))) &&
    identical(plan_t$knndm$crs, 4326) && grepl("moll", plan_t$knndm$project_to)
  rf   <- refit_split(plan_t, meta, validation_frac = 0.15)
  f    <- rf$folds[[1]]
  pool <- setdiff(seq_len(nrow(meta)), plan_t$folds[[1]]$test)
  ok["the_refit_is_one_knndm_fold"] <-
    rf$n_folds == 1L && identical(rf$method, "refit_knndm_folds")
  ok["the_refit_keeps_the_frozen_test_set"] <-
    identical(sort(f$test), sort(plan_t$folds[[1]]$test))
  ok["the_refit_splits_the_rest_once"] <-
    identical(sort(as.integer(c(f$train, f$validation))), as.integer(pool)) &&
    length(intersect(f$train, f$validation)) == 0L
  ok["the_refit_validates_about_the_share_asked"] <- {
    s <- length(f$validation) / length(pool)
    s > 0.05 && s < 0.35
  }
  ok["the_refit_is_reproducible"] <-
    identical(refit_split(plan_t, meta, validation_frac = 0.15)$folds[[1]], f)
  # A plan from before the points were kept: they must be given, and given,
  # the refit is the same one. Other points than a plan's own are refused.
  legacy <- plan_t
  legacy$knndm <- NULL
  e_old <- tryCatch(refit_split(legacy, meta, 0.15), error = function(e) conditionMessage(e))
  ok["an_old_plan_without_its_points_says_what_to_pass"] <- grepl("predpoints =", e_old, fixed = TRUE)
  ok["given_its_points_an_old_plan_refits_the_same_way"] <-
    identical(refit_split(legacy, meta, 0.15, predpoints = predpts)$folds[[1]], f)
  e_other <- tryCatch(refit_split(plan_t, meta, 0.15, predpoints = predpts[1:100, ]),
                      error = function(e) conditionMessage(e))
  ok["other_points_for_a_plan_with_its_own_are_refused"] <-
    grepl("Leave predpoints out", e_other, fixed = TRUE)

  cat(sprintf("  knndm W                  : %.4f (3 folds, %d points, %d predpoints)\n",
              plan$params$W, nrow(meta), nrow(predpts)))
  cat(sprintf("  knndm refit              : %d of %d non-test points validate (fold %d of 7)\n",
              length(f$validation), length(pool), rf$params$refit_fold))
} else {
  cat("  CAST missing             : fold-plan checks skipped\n")
  cat("                             install.packages(c(\"CAST\", \"sf\"))\n")
  # The dependency error must still name what to install, or a person on a
  # machine without CAST gets a stack trace instead of an instruction.
  ok["missing_dependency_names_itself"] <- {
    e <- tryCatch(knndm_folds(meta, k = 3L, predpoints = predpts),
                  error = function(e) conditionMessage(e))
    grepl("install.packages", e, fixed = TRUE)
  }
}

.report(ok, "test_knndm")
