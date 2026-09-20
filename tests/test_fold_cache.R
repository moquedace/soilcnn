# Unit test: the fold cache scales by the fold's own training rows, and the
# helpers around it do what their one-line doc says
#
# WHY THIS FILE EXISTS.
#
# build_fold_cache() is where "scaling is fitted on THIS fold's training rows"
# stops being a sentence in the README and becomes arithmetic. Nothing tested
# that sentence directly: the end-to-end runs would still pass with scaling
# fitted on every row, because the CNN does not care where its z-scores came
# from -- only the leakage argument does. The same is true of the store's own
# store_complete verdict (refused since 2026-09-19), of the patch-centre check
# (the one diagnostic that can tell a store built from the wrong rasters), and
# of two torch helpers -- clone_state_dict() and set_optimizer_lr() -- that
# were used on every training path and asserted nowhere.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/tests/test_fold_cache.R")

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
for (m in c("utils", "patches", "preprocess", "dataset", "diagnostics")) {
  source(file.path(root, "R", paste0(m, ".R")))
}

ok <- c()

# ── a synthetic store: three channels of three kinds ─────────────────────────
#
# pred_1 continuous (z-scored), pred_2 a 0/1 dummy (left alone), pred_3 a
# percentage (divided by 100). One of each, so every branch of fit_scaling()
# is exercised by the same fold.
set.seed(3)
n_pts <- 40L; n_ch <- 3L; win <- 3L
preds <- c("pred_1", "pred_2", "pred_3")
centre <- (win + 1L) %/% 2L

arr <- array(0, dim = c(n_pts, n_ch, win, win))
arr[, 1, , ] <- rnorm(n_pts * win * win, mean = 50, sd = 10)
arr[, 2, , ] <- rbinom(n_pts * win * win, 1, 0.4)
arr[, 3, , ] <- runif(n_pts * win * win, 0, 100)

y_true <- 1 + abs(arr[, 1, centre, centre]) / 10
meta <- tibble::tibble(
  profile_id = seq_len(n_pts), sample_id = seq_len(n_pts),
  x = runif(n_pts, 0, 1e4), y = runif(n_pts, 0, 1e4),
  target_native = y_true, target_transform = log1p(y_true))

# the point table holds the centre pixel of every channel, as stage 01 does
points <- meta
for (k in seq_len(n_ch)) points[[preds[k]]] <- arr[, k, centre, centre]

type_table <- tibble::tibble(predictor = preds,
                             is_dummy = c(FALSE, TRUE, FALSE),
                             is_percentage = c(FALSE, FALSE, TRUE))

store_dir <- file.path(tempdir(), "dlc_fold_cache_store")
unlink(store_dir, recursive = TRUE)
dir.create(store_dir, recursive = TRUE, showWarnings = FALSE)
invisible(save_patch_window(arr, store_dir, win))
readr::write_csv2(meta, file.path(store_dir, "patch_meta.csv"))
write_manifest <- function(store_complete) {
  saveRDS(tibble::tibble(
    scaling_applied = FALSE, predictor_cols_final = paste(preds, collapse = ";"),
    n_channels = n_ch, windows_extracted = as.character(win),
    n_points_valid = n_pts, store_complete = store_complete),
    file.path(store_dir, "patch_manifest.rds"))
}

# ── 1. the store's own verdict is read ───────────────────────────────────────
write_manifest(FALSE)
r <- try(load_patch_store(store_dir, verbose = FALSE), silent = TRUE)
ok["an_incomplete_store_is_refused"] <-
  inherits(r, "try-error") && grepl("INCOMPLETE", conditionMessage(attr(r, "condition")))
write_manifest(TRUE)
store <- load_patch_store(store_dir, verbose = FALSE)
ok["a_complete_store_loads"] <- identical(store$predictors, preds)

# ── 2. scaling is fitted on the TRAINING rows of this fold, not on all rows ──
index <- list(train = 1:24, validation = 25:40)
fc <- suppressMessages(build_fold_cache(store, points, type_table, index))
sc <- fc$scaling

ok["zscore_centre_is_the_training_mean"] <-
  isTRUE(all.equal(sc$center[1], mean(points$pred_1[1:24])))
ok["zscore_centre_is_not_the_mean_of_all_rows"] <-
  !isTRUE(all.equal(sc$center[1], mean(points$pred_1)))
ok["zscore_scale_is_the_training_sd"] <-
  isTRUE(all.equal(sc$scale[1], sd(points$pred_1[1:24])))
ok["a_dummy_is_left_alone"] <- sc$center[2] == 0 && sc$scale[2] == 1
ok["a_percentage_is_divided_by_100"] <- sc$center[3] == 0 && sc$scale[3] == 100

# the arithmetic on the cached tensor is the arithmetic on the table
got  <- as.numeric(fc$cache$validation$w03[3, 1, centre, centre])
want <- (arr[27, 1, centre, centre] - sc$center[1]) / sc$scale[1]
ok["validation_row_3_is_store_row_27_scaled"] <- abs(got - want) < 1e-5
ok["cached_shapes_follow_the_index"] <-
  identical(as.integer(fc$cache$train$w03$shape), c(24L, n_ch, win, win)) &&
  identical(as.integer(fc$cache$validation$w03$shape), c(16L, n_ch, win, win))
ok["y_is_the_transformed_target_in_index_order"] <-
  isTRUE(all.equal(as.numeric(fc$cache$validation$y), log1p(y_true[25:40]),
                   tolerance = 1e-6))

# the store's raw tensor must survive: the next fold scales it again from raw
raw_after <- as.array(store$windows$w03)
ok["the_store_keeps_its_raw_tensor"] <-
  isTRUE(all.equal(raw_after, arr, tolerance = 1e-6, check.attributes = FALSE))

# fold_points_valid() rows pair with the cached tensors, row for row.
#
# Compared as VALUES, not with identical(). The store's meta comes back from
# read_csv2(), which guesses `double` for a column of whole numbers, so
# identical(fpv$validation$profile_id, 25:40) is FALSE on 25 vs 25L -- a
# storage type, not a mismatch of rows. align_points_to_meta() carries the
# same lesson in its own comment, and this assertion walked into it anyway.
fpv <- fold_points_valid(store, index)
ok["meta_rows_pair_with_the_tensor_rows"] <-
  isTRUE(all.equal(as.numeric(fpv$validation$profile_id), 25:40)) &&
  isTRUE(all.equal(as.numeric(fpv$train$profile_id), 1:24))
# ...and the pairing is by POSITION, so a shuffled index must follow
shuffled <- list(train = 1:24, validation = c(40L, 25:39))
ok["a_reordered_index_reorders_the_meta_rows_with_it"] <-
  isTRUE(all.equal(
    as.numeric(fold_points_valid(store, shuffled)$validation$profile_id),
    c(40, 25:39)))

# ── 3. what the cache refuses ────────────────────────────────────────────────
pts_const <- points; pts_const$pred_1[1:24] <- 7
r <- try(suppressMessages(build_fold_cache(store, pts_const, type_table, index)), silent = TRUE)
ok["a_channel_constant_within_the_fold_is_refused"] <-
  inherits(r, "try-error") && grepl("Degenerate", conditionMessage(attr(r, "condition")))

tt_swapped <- type_table[c(2, 1, 3), ]
r <- try(build_fold_cache(store, points, tt_swapped, index), silent = TRUE)
ok["a_type_table_in_another_channel_order_is_refused"] <-
  inherits(r, "try-error") && grepl("channel order", conditionMessage(attr(r, "condition")))

# ── 4. check_patch_centres: the store against the point table ────────────────
cc <- check_patch_centres(store_dir, points, preds)
ok["centres_agree_on_a_consistent_store"] <- isTRUE(cc$ok) && cc$n_mismatch == 0L

pts_bad <- points; pts_bad$pred_1[5] <- pts_bad$pred_1[5] + 1
cc <- check_patch_centres(store_dir, pts_bad, preds)
ok["one_moved_value_is_one_mismatch_in_the_right_channel"] <-
  !cc$ok && cc$n_mismatch == 1L && identical(cc$by_channel$predictor, "pred_1")

pts_na <- points; pts_na$pred_3[9] <- NA
cc <- check_patch_centres(store_dir, pts_na, preds)
ok["na_on_one_side_only_is_a_mismatch"] <- cc$n_mismatch == 1L

r <- try(check_patch_centres(store_dir, points[1:10, ], preds), silent = TRUE)
ok["a_point_table_of_another_length_is_refused"] <- inherits(r, "try-error")

# ── 5. two torch helpers every training path relies on ───────────────────────
torch::torch_manual_seed(1)
lin <- torch::nn_linear(2, 1)
sd0 <- lin$state_dict()
cl  <- clone_state_dict(sd0)
torch::with_no_grad(lin$weight$add_(100))
ok["a_cloned_state_dict_does_not_follow_the_model"] <-
  abs(as.numeric(cl$weight[1, 1]) - as.numeric(lin$weight[1, 1])) > 50
ok["a_cloned_state_dict_is_detached"] <- !cl$weight$requires_grad

opt <- torch::optim_adam(lin$parameters, lr = 0.1)
set_optimizer_lr(opt, 0.001)
ok["set_optimizer_lr_reaches_every_param_group"] <-
  all(vapply(opt$param_groups, function(g) g$lr, numeric(1)) == 0.001)

unlink(store_dir, recursive = TRUE, force = TRUE)

cat(sprintf("  scaling                  : centre %.3f from 24 training rows (all-row mean %.3f)\n",
            sc$center[1], mean(points$pred_1)))
cat("  store_complete = FALSE   : refused at load\n")
cat("  patch centres            : one moved value -> one mismatch, in its channel\n")

.report(ok, "test_fold_cache")
