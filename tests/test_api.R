# Unit test: the front end
#
# The engine is tested elsewhere. This tests the layer a USER touches, and it
# is the layer where a mistake is least likely to announce itself: a spec that
# quietly ignores an argument still produces a plan, still trains, and still
# reports a number.
#
# The two arguments that carry the most risk are the ones written as "auto",
# because "auto" is exactly where a wrong value looks deliberate:
#
#   buffer = "auto"      must be max(window) * cell_size -- the SQUARE
#                        separation distance. Half of it, or a radius, leaves
#                        patches sharing pixels while the report says 0%.
#   block_size = "auto"  must be measured on THESE points. A number carried
#                        from another run is how one fold of this project's own
#                        dev run came to hold a third of the data.
#
# Verified:
#   1. a spec is a spec, not a plan -- nothing is decided before it sees data
#   2. every kind of spec resolves to a valid fold plan
#   3. buffer = "auto" is exactly max(window) * cell_size
#   4. block_size = "auto" measures, and respects the balance constraint
#   5. a fold_plan passed as `resampling` is used unchanged
#   6. a frozen test set survives a change of resampling method
#   7. the failures a user will actually hit say what to do
#
# Run: source("D:/.../tests/test_api.R")     (no torch needed)

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
source(file.path(root, "R", "metrics.R"))
source(file.path(root, "R", "resample.R"))
source(file.path(root, "R", "dataset.R"))
source(file.path(root, "R", "model_registry.R"))
source(file.path(root, "R", "api.R"))

ok <- c()

# =============================================================================
# A dsm_data built by hand.
#
# No store on disk and no torch: everything tested here happens before a tensor
# is touched, and a test that needs 6 GB of patches to check an argument is a
# test that stops being run.
# =============================================================================

set.seed(4242)
# 48 sites of 8 points, one degree apart. Clustered enough that blocking has
# something to do, and NUMEROUS enough that suggest_block_size() can satisfy
# its own constraints -- with only a dozen sites every candidate size fails
# `min_blocks_per_fold` and the function correctly falls back with a warning,
# which would make this test assert against the fallback rather than the rule.
N_SITE <- 48L
PER    <- 8L
N      <- N_SITE * PER
site   <- rep(seq_len(N_SITE), each = PER)
meta <- tibble::tibble(
  sample_id  = seq_len(N),
  profile_id = sprintf("p%04d", seq_len(N)),
  x = ((site - 1L) %% 7L) * 1.0 + stats::runif(N, 0, 0.05),
  y = ((site - 1L) %/% 7L) * 1.0 + stats::runif(N, 0, 0.05),
  target_native    = stats::rlnorm(N, 3, 0.5),
  target_transform = log1p(stats::rlnorm(N, 3, 0.5))
)

CELL <- 0.00208333
fake_data <- structure(list(
  store = list(meta = meta, window_sizes = c(3L, 9L, 15L),
               predictors = c("a", "b"), n_channels = 2L),
  points     = meta,
  type_table = tibble::tibble(predictor = c("a", "b")),
  cell_size  = CELL,
  patch_dir  = "(none)",
  target_col = "soc"
), class = "dsm_data")

# =============================================================================
# 1. A spec decides nothing
# =============================================================================

sp <- spatial_cv(k = 4L, block_size = "auto", buffer = "auto")
ok["spec_is_a_spec"]          <- inherits(sp, "resample_spec")
ok["spec_is_not_a_plan"]      <- !inherits(sp, "fold_plan")
ok["spec_keeps_auto_as_auto"] <- identical(sp$block_size, "auto") &&
                                 identical(sp$buffer, "auto")
ok["spec_records_k"]          <- sp$k == 4L

# EVERY CONSTRUCTOR RECORDS ITS OWN KIND, AS A STRING.
#
# The assertion that was missing, and the cheapest one in the file. R partially
# matches a named argument against any formal declared BEFORE `...`, so while
# the builder was `function(kind, ...)`, `spatial_cv(k = 5)` set `kind` to 5
# and pushed the word "spatial" into `...`.
#
# Nothing errored. switch() on a NUMERIC EXPR ignores the alternative names and
# returns the nth one, so asking for spatial blocks returned a valid one-fold
# HOLDOUT plan -- the exact shape of failure this project keeps paying for: the
# wrong answer, well-formed, with no complaint anywhere.
ok["spatial_cv_records_its_kind"] <- identical(sp$kind, "spatial")
ok["random_cv_records_its_kind"]  <- identical(random_cv(k = 5L)$kind, "random")
ok["holdout_cv_records_its_kind"] <- identical(holdout_cv()$kind, "holdout")
ok["region_cv_records_its_kind"]  <-
  identical(region_cv(group = rep("a", 3L))$kind, "region")

# ...and the argument that triggered it is still carried, under its own name.
ok["k_survives_as_k_not_as_kind"] <- identical(random_cv(k = 7L)$k, 7L)
ok["region_cv_keeps_a_null_k"]    <- is.null(region_cv(group = "a", k = NULL)$k)

# A spec whose kind is not one of the four must be refused, not resampled by
# position.
bad_spec <- structure(list(kind = 3L, k = 2L), class = "resample_spec")
ok["a_non_character_kind_is_refused"] <- inherits(
  tryCatch(resolve_resampling(bad_spec, fake_data, verbose = FALSE),
           error = function(e) e), "error")

# =============================================================================
# 2. Every kind resolves to a valid plan
# =============================================================================

p_spatial <- suppressMessages(resolve_resampling(
  spatial_cv(k = 3L, block_size = 1, buffer = NULL, test_frac = 0.2),
  fake_data, verbose = FALSE))
p_random  <- suppressMessages(resolve_resampling(
  random_cv(k = 3L, test_frac = 0.2), fake_data, verbose = FALSE))
p_hold    <- suppressMessages(resolve_resampling(
  holdout_cv(validation_frac = 0.2, test_frac = 0.2), fake_data, verbose = FALSE))
p_region  <- suppressMessages(resolve_resampling(
  region_cv(group = as.character(site), test_frac = 0), fake_data, verbose = FALSE))

for (nm in c("p_spatial", "p_random", "p_hold", "p_region")) {
  pl <- get(nm)
  ok[paste0(nm, "_is_a_fold_plan")] <- inherits(pl, "fold_plan")
  # check_fold_plan proves the partition AND the no-split-group property; if it
  # throws, the plan is not a plan.
  ok[paste0(nm, "_passes_its_own_check")] <-
    !inherits(try(check_fold_plan(pl, meta = meta), silent = TRUE), "try-error")
}
ok["spatial_has_the_folds_asked_for"] <- p_spatial$n_folds == 3L
ok["holdout_has_one_fold"]            <- p_hold$n_folds == 1L
ok["region_uses_every_region"]        <- p_region$n_folds == N_SITE

# A test carved by the plan is held out of every fold, which is the property
# that makes it a test set rather than a label.
ok["test_never_trains"] <- all(vapply(p_spatial$folds, function(f)
  length(intersect(f$test, c(f$train, f$validation))) == 0L, logical(1)))

# =============================================================================
# 3. buffer = "auto" is the SQUARE separation distance
#
# Two patches of width w share a pixel when their centres are within w-1 cells
# in BOTH axes. Under the Chebyshev metric max(window) * cell_size is exact; a
# radius of half that, or a circular buffer, leaves the diagonal sharing pixels
# while the leakage report shows a clean zero.
# =============================================================================

p_auto <- suppressMessages(resolve_resampling(
  spatial_cv(k = 3L, block_size = 1, buffer = "auto", test_frac = 0),
  fake_data, windows = c(3L, 15L), verbose = FALSE))
ok["auto_buffer_is_window_times_cell"] <-
  isTRUE(all.equal(p_auto$params$buffer, 15 * CELL))
ok["auto_buffer_is_not_half"] <-
  !isTRUE(all.equal(p_auto$params$buffer, 7.5 * CELL))
ok["auto_buffer_metric_is_chebyshev"] <-
  identical(p_auto$params$buffer_metric %||% "chebyshev", "chebyshev")

# A number given explicitly is used as given, never "improved".
p_fixed <- suppressMessages(resolve_resampling(
  spatial_cv(k = 3L, block_size = 1, buffer = 0.5, test_frac = 0),
  fake_data, verbose = FALSE))
ok["explicit_buffer_is_respected"] <- isTRUE(all.equal(p_fixed$params$buffer, 0.5))

# And "auto" without a resolution must REFUSE, not invent one.
no_cell <- fake_data; no_cell$cell_size <- NULL
ok["auto_buffer_without_cell_size_refuses"] <- inherits(
  tryCatch(resolve_resampling(spatial_cv(buffer = "auto"), no_cell,
                              verbose = FALSE),
           error = function(e) e), "error")

# =============================================================================
# 4. block_size = "auto" measures
# =============================================================================

p_ab <- suppressMessages(resolve_resampling(
  spatial_cv(k = 3L, block_size = "auto", buffer = NULL, test_frac = 0,
             max_share = 0.10),
  fake_data, verbose = FALSE))
chosen <- p_ab$params$block_size
ok["auto_block_size_is_a_number"] <- is.numeric(chosen) && is.finite(chosen)
# The constraint it was given must actually hold on these points.
bs_tab <- block_share(meta, chosen)
ok["auto_block_size_respects_max_share"] <- bs_tab$largest_share[1] <= 0.10
ok["auto_block_size_is_recorded"] <- isTRUE(p_ab$params$block_size_auto) ||
                                     is.numeric(p_ab$params$block_size)

# =============================================================================
# 5-6. A plan can be passed straight through, and a test set can be frozen
# =============================================================================

ok["a_plan_passes_through_unchanged"] <-
  identical(resolve_resampling(p_spatial, fake_data), p_spatial)

frozen <- meta$sample_id[1:40]   # five whole sites, so a block plan can hold them
p_fr1 <- suppressMessages(resolve_resampling(
  spatial_cv(k = 3L, block_size = 1, buffer = NULL, test_frac = 0.2),
  fake_data, test_ids = frozen, verbose = FALSE))
p_fr2 <- suppressMessages(resolve_resampling(
  random_cv(k = 3L, test_frac = 0.2), fake_data, test_ids = frozen,
  verbose = FALSE))
test_of <- function(pl) sort(meta$sample_id[pl$folds[[1]]$test])
# The same rows are held out under BOTH methods -- which is the whole point of
# freezing it: a test set that changes with the method cannot compare them.
ok["frozen_test_survives_spatial"] <- all(frozen %in% test_of(p_fr1))
ok["frozen_test_survives_random"]  <- all(frozen %in% test_of(p_fr2))

# =============================================================================
# 7. The failures a user will actually hit
# =============================================================================

ok["unknown_model_is_refused"] <- inherits(
  tryCatch(dsm_train(fake_data, model = "not_a_model"),
           error = function(e) e), "error")

ok["dsm_train_needs_dsm_data"] <- inherits(
  tryCatch(dsm_train(list(a = 1), model = "rf"), error = function(e) e),
  "error")

# A grid asking for a window the data was not loaded with must name the fix.
if ("cnn" %in% list_models()$name) {
  small <- fake_data
  small$store$window_sizes <- 3L
  msg <- tryCatch(
    dsm_train(small, model = "cnn",
              resampling = p_spatial,
              tune_grid = tibble::tibble(config_id = "c1",
                                         window_sizes = list(c(3L, 21L)))),
    error = function(e) conditionMessage(e))
  ok["missing_window_names_the_fix"] <-
    is.character(msg) && grepl("dsm_load", msg) && grepl("21", msg)
}

# print methods must not error on a legitimate object -- a print that throws
# turns a finished run into a lost one.
ok["print_dsm_data_works"] <- !inherits(
  tryCatch(utils::capture.output(print(fake_data)), error = function(e) e),
  "error")
ok["print_spec_works"] <- !inherits(
  tryCatch(utils::capture.output(print(sp)), error = function(e) e), "error")

cat(sprintf("  auto buffer              : %.6f  (15 px x %.8f)\n",
            p_auto$params$buffer, CELL))
cat(sprintf("  auto block size          : %g  (largest block %.1f%% of points)\n",
            chosen, 100 * bs_tab$largest_share[1]))

.report(ok, "test_api")
