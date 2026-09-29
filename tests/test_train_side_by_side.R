# Test: dsm_train() with its units side by side, each worker an R process
#
# Since 2026-09-29 the CNN's units train side by side by default: each worker
# claims units in fold order, builds a fold's cache once, writes a record per
# unit, and the session gathers the table (R/train_workers.R). What would go
# wrong without a word is what this checks:
#
#   1. every unit trains once, and writes what the fold loop writes: its row,
#      its checkpoint, its files -- and no test row while evaluate_test is FALSE
#   2. the numbers do not depend on how many workers trained them (T2): two
#      workers and one give the same table, to the bit
#   3. the run records how its units trained, and a resume that would train the
#      rest another way is refused: another threads_per_unit, the session, or a
#      run from before the record
#   4. a resume trains nothing that finished
#   5. a unit that fails is a row that says so, not a lost worker
#   6. the door: a device with in_session = FALSE, a thread count that is not
#      a whole number
#   7. the worker's memory trace, when asked, fold by fold and unit by unit
#
# Run: source("<package root>/tests/test_train_side_by_side.R")
# (starts worker processes and trains; ~2 min on CPU)

suppressMessages({
  library(torch)
  library(tibble)
  library(dplyr)
  library(readr)
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

# =============================================================================
# A store on disk: 64 points in four clusters, three channels, one window
# =============================================================================

set.seed(20260929)
n_pts <- 64L; n_ch <- 3L; win <- 3L
preds <- paste0("p", seq_len(n_ch))
CELL  <- 0.00224579811173295

site <- rep(1:4, each = n_pts / 4L)
x <- (site %% 2L) * 2 + stats::runif(n_pts, 0, 0.05)
y <- (site %/% 3L) * 2 + stats::runif(n_pts, 0, 0.05)
arr <- array(stats::rnorm(n_pts * n_ch * win * win), dim = c(n_pts, n_ch, win, win))
centre <- (win + 1L) %/% 2L
y_true <- 2 * arr[, 1, centre, centre] + stats::rnorm(n_pts, sd = 0.5)
y_true <- y_true - min(y_true) + 1

meta <- tibble::tibble(profile_id = seq_len(n_pts), sample_id = seq_len(n_pts),
                       x = x, y = y, target_native = y_true,
                       target_transform = log1p(y_true))

# Cleaned on ENTRY as well as on exit: tempdir() survives between source()s in
# one session, and a run that resumes its own previous output is not a test.
store_dir <- file.path(tempdir(), "dlc_side_store")
out_root  <- file.path(tempdir(), "dlc_side_out")
unlink(store_dir, recursive = TRUE)
unlink(out_root,  recursive = TRUE)
dir.create(store_dir, recursive = TRUE, showWarnings = FALSE)

invisible(save_patch_window(arr, store_dir, win))
readr::write_csv2(meta, file.path(store_dir, "patch_meta.csv"))
saveRDS(tibble::tibble(
  scaling_applied = FALSE, predictor_cols_final = paste(preds, collapse = ";"),
  n_channels = n_ch, windows_extracted = as.character(win), n_points_valid = n_pts,
  target_col = "soc_stock", target_transform = "log1p", cell_size = CELL
), file.path(store_dir, "patch_manifest.rds"))

points <- meta
for (j in seq_len(n_ch)) points[[preds[j]]] <- arr[, j, centre, centre]
type_table <- tibble::tibble(predictor = preds, is_dummy = FALSE, is_percentage = FALSE)

data <- suppressMessages(dsm_load(store_dir, points = points, type_table = type_table,
                                  target_col = "soc_stock", verbose = FALSE))
plan <- suppressMessages(resolve_resampling(
  spatial_cv(k = 2L, block_size = 1, buffer = "auto", test_frac = 0.2),
  data, verbose = FALSE))

# Two configs that differ in one hyperparameter; two folds, two seeds: eight
# units, in two folds' caches.
grid <- make_manual_tune_grid(
  window_sizes = list(c(win)), conv_channels = list(c(4L)),
  embedding_dim = c(8L, 16L), base_lr = 0.01, batch_size = 8L, dropout = 0.0,
  gate_type = "no_gate_concat", use_residual = FALSE, use_se_block = FALSE)
n_units <- nrow(grid) * plan$n_folds * 2L

train <- function(run_id, ...) {
  suppressMessages(dsm_train(
    data, model = "cnn", resampling = plan, tune_grid = grid, n_seeds = 2L,
    output_dir = out_root, run_id = run_id, n_epochs = 3L, patience = 3L,
    print_every = 100L, augment = FALSE, verbose = FALSE, ...))
}
err <- function(expr) tryCatch({ expr; "" }, error = function(e) conditionMessage(e))
numbers_of <- function(cmp) {
  cmp  <- as.data.frame(cmp)[order(cmp$unit_id), , drop = FALSE]
  keep <- vapply(cmp, is.numeric, logical(1)) & !names(cmp) %in% c("runtime_min", "rank")
  cmp[, keep, drop = FALSE]
}

# =============================================================================
# 1. Two workers side by side
# =============================================================================

side2 <- train("side2", n_cores = 2L, threads_per_unit = 1L)
rd2   <- file.path(out_root, "side2")
cmp2  <- side2$comparison

ok["side_by_side_returns_a_dsm_fit"] <- inherits(side2, "dsm_fit")
ok["every_unit_trained_once"] <- nrow(cmp2) == n_units &&
  dplyr::n_distinct(cmp2$unit_id) == n_units && all(cmp2$status == "success")
ok["the_units_have_the_fold_loops_names"] <- all(grepl("^cfg_\\d+_f[12]_s[12]$", cmp2$unit_id))
ok["the_seeds_are_the_fold_loops"] <- setequal(unique(cmp2$seed), c(42L, 43L))
ok["two_workers_ran"] <- identical(as.integer(side2$n_workers), 2L) &&
  identical(as.integer(side2$threads_per_unit), 1L)
ok["the_run_records_how_its_units_trained"] <-
  identical(readRDS(file.path(rd2, "threads.rds")), list(mode = "workers", threads = 1L))

recs <- lapply(cmp2$unit_id, function(u) readRDS(file.path(rd2, "units", paste0(u, ".rds"))))
ok["every_unit_wrote_its_record"] <- all(vapply(recs, function(r) identical(r$status, "success"), logical(1)))
ok["every_unit_wrote_its_checkpoint"] <-
  all(file.exists(file.path(rd2, "models", paste0(cmp2$unit_id, "_best.pt"))))
ok["no_record_is_left_half_written"] <-
  length(list.files(file.path(rd2, "units"), pattern = "[.]part$")) == 0L
ok["each_worker_has_its_log"] <- all(file.exists(file.path(rd2, "logs", c("worker_01.log", "worker_02.log"))))
ok["the_claims_are_cleared"] <- !dir.exists(file.path(rd2, ".claims"))
ok["the_scaling_of_every_fold_is_written"] <-
  all(file.exists(file.path(rd2, sprintf("scaling_fold%02d.csv", 1:2))))
ok["the_table_is_ranked_and_written"] <-
  all(file.exists(file.path(rd2, "comparison", c("comparison_all.rds", "comparison_ranked.csv",
                                                  "comparison_by_config.csv")))) &&
  nrow(side2$by_config) == nrow(grid) && all(side2$by_config$n_units == 4L)
ok["the_scaling_is_the_fold_caches"] <- {
  sc <- fit_scaling(data$points, data$type_table, plan$folds[[2]]$train)
  sc_disk <- safe_read_csv2(file.path(rd2, "scaling_fold02.csv"))
  isTRUE(all.equal(sc_disk$center, sc$center)) && isTRUE(all.equal(sc_disk$scale, sc$scale))
}

# No test row leaves a worker while evaluate_test is FALSE: every CSV a unit
# writes, read for its roles, as tests/test_resample_run.R reads the fold loop's.
roles <- unique(unlist(lapply(
  list.files(file.path(rd2, c("predictions", "metrics")), pattern = "[.]csv$", full.names = TRUE),
  function(p) { d <- safe_read_csv2(p); if ("dataset_role" %in% names(d)) unique(d$dataset_role) })))
ok["no_test_row_on_disk"] <- length(roles) > 0L && !"test" %in% roles && "validation" %in% roles
ok["the_test_columns_are_empty"] <- all(is.na(cmp2$test_ccc))

by_worker <- table(vapply(recs, function(r) r$worker, numeric(1)))

# =============================================================================
# 2. One worker, traced: the same numbers, and the trace fold by fold
# =============================================================================

old_opt <- options(dsm.train.trace_mem = TRUE)
side1 <- train("side1", n_cores = 1L, threads_per_unit = 1L)
options(old_opt)
rd1 <- file.path(out_root, "side1")

ok["one_worker_ran"] <- identical(as.integer(side1$n_workers), 1L)
ok["two_workers_and_one_give_the_same_numbers"] <- {
  a <- numbers_of(cmp2); b <- numbers_of(side1$comparison)
  identical(names(a), names(b)) && nrow(a) == n_units &&
    isTRUE(all.equal(a, b, tolerance = 0, check.attributes = FALSE))
}
tr <- readRDS(file.path(rd1, "logs", "worker_01_mem_trace.rds"))
ok["the_worker_traces_fold_by_fold"] <-
  identical(tr$phase, c("start", "store_loaded",
                        rep(c("fold_cache", rep("unit_trained", 4L)), 2L))) &&
  identical(tr$fold[tr$phase == "fold_cache"], 1:2) && all(is.finite(tr$r_heap_gb))
ok["no_trace_unless_asked"] <- !file.exists(file.path(rd2, "logs", "worker_01_mem_trace.rds"))

# =============================================================================
# 3-4. A resume: nothing finished trains again; another way of training is refused
# =============================================================================

before <- lapply(cmp2$unit_id, function(u) file.mtime(file.path(rd2, "units", paste0(u, ".rds"))))
again  <- train("side2", n_cores = 2L, threads_per_unit = 1L)
ok["a_resume_trains_nothing_that_finished"] <-
  identical(as.integer(again$n_workers), 0L) &&
  identical(lapply(cmp2$unit_id, function(u) file.mtime(file.path(rd2, "units", paste0(u, ".rds")))),
            before) &&
  isTRUE(all.equal(numbers_of(again$comparison), numbers_of(cmp2), tolerance = 0,
                   check.attributes = FALSE))

m_threads <- err(train("side2", n_cores = 2L, threads_per_unit = 2L))
ok["a_resume_with_another_thread_count_is_refused"] <-
  grepl("computed two ways", m_threads) && grepl("threads_per_unit = 1", m_threads)
m_session <- err(train("side2", n_cores = 1L, in_session = TRUE))
ok["a_resume_in_the_session_of_a_side_by_side_run_is_refused"] <-
  grepl("side by side with 1 thread", m_session) && grepl("computed two ways", m_session)

# A run from before the record: units on disk and no threads.rds.
old_dir <- file.path(out_root, "before_the_record")
dir.create(file.path(old_dir, "models"), recursive = TRUE)
invisible(file.create(file.path(old_dir, "models", "cfg_001_f1_s1_best.pt")))
ok["a_run_from_before_the_record_is_refused_side_by_side"] <-
  grepl("before dsm_train\\(\\) recorded", err(.train_threads_record(old_dir, "workers", 5L, TRUE)))
ok["and_goes_on_in_the_session"] <-
  identical(err(.train_threads_record(old_dir, "session", 3L, TRUE)), "") &&
  identical(readRDS(file.path(old_dir, "threads.rds")), list(mode = "session", threads = 3L))

# =============================================================================
# 5. A unit that fails is a row that says so
# =============================================================================

# embedding_dim = 0 cannot build a layer. dsm_train() refuses such a grid at
# the door, so the runner is called as dsm_train() calls it, side by side.
grid_bad <- grid
grid_bad$embedding_dim[1] <- 0L
bad <- suppressMessages(run_cnn_resample(
  tune_grid = grid_bad, store = data$store, points = data$points,
  type_table = data$type_table, plan = plan, transform = expm1, output_dir = out_root,
  device = NULL, run_id = "side_bad", n_seeds = 1L, in_session = FALSE,
  threads_per_unit = 1L, n_cores = 2L,
  n_epochs = 3L, patience = 3L, print_every = 100L, augment = FALSE))
cmp_bad <- bad$comparison
ok["a_failed_unit_is_a_row"] <- nrow(cmp_bad) == nrow(grid) * plan$n_folds &&
  sum(cmp_bad$status == "failed") == plan$n_folds &&
  all(nzchar(cmp_bad$error_message[cmp_bad$status == "failed"]))
ok["the_other_config_still_trained"] <- sum(cmp_bad$status == "success") == plan$n_folds &&
  nrow(bad$by_config) == 1L

# =============================================================================
# 6. The door
# =============================================================================

ok["a_device_with_in_session_false_is_refused"] <- grepl("in_session = TRUE", err(dsm_train(
  data, model = "cnn", resampling = plan, tune_grid = grid, output_dir = out_root,
  run_id = "door", device = torch::torch_device("cpu"), in_session = FALSE, verbose = FALSE)))
ok["threads_per_unit_must_be_whole"] <-
  grepl("whole number", err(dsm_train(data, model = "cnn", resampling = plan, tune_grid = grid,
                                      output_dir = out_root, run_id = "door",
                                      threads_per_unit = 1.5, verbose = FALSE))) &&
  grepl("whole number", err(dsm_train(data, model = "cnn", resampling = plan, tune_grid = grid,
                                      output_dir = out_root, run_id = "door",
                                      threads_per_unit = 0L, verbose = FALSE)))
ok["the_door_trains_nothing"] <- !dir.exists(file.path(out_root, "door", "models"))

# =============================================================================
# The session, for comparison: the same units one at a time in this process.
# Not asserted equal -- B6 measured a unit trained in a session that loaded
# torch with other threads at 1.2e-4 CCC from the same unit in a fresh
# process -- only reported.
# =============================================================================

sess <- train("session", n_cores = 1L, in_session = TRUE)
d_sess <- max(abs(numbers_of(sess$comparison)$val_ccc - numbers_of(cmp2)$val_ccc))
ok["the_session_records_its_threads"] <-
  identical(readRDS(file.path(out_root, "session", "threads.rds")),
            list(mode = "session", threads = 1L))

unlink(store_dir, recursive = TRUE)
unlink(out_root,  recursive = TRUE)

cat(sprintf("  side by side             : %d units, 2 workers x 1 thread (units per worker: %s)\n",
            n_units, paste(sprintf("%s=%d", names(by_worker), as.integer(by_worker)), collapse = ", ")))
cat(sprintf("  in the session           : max |val_ccc| difference from side by side %.2e\n", d_sess))

.report(ok, "test_train_side_by_side")
