# ══════════════════════════════════════════════════════════════════════════════
# T8 -- the tuning loop's memory, fold by fold: each fold's cache in tensors of
# its own, or in one buffer for every fold, on the dev store
#
# THE QUESTION.
#
# dsm_train() builds a fold's cache -- every role of every window, scaled by
# that fold's training rows -- trains the fold's units on it, releases it, and
# builds the next fold's. mimalloc, under libtorch on this Windows machine,
# neither returns nor reuses a freed tensor of ~250 MB or more (T6: a 1 GB
# tensor made and dropped three times left 3 GB). A training role is that
# large from the dev store's 15x15 window up, so each fold would leave its
# cache behind, and a run of k folds hold k caches by its end: on the full
# data set (windows ~11x larger) several GB a fold. T7 measured the final
# fit, which builds one cache; nothing had measured the loop.
#
# The fix is in the framework since 2026-09-29: one tensor per window, the
# store's length, made on the first fold, with each fold's roles written
# into consecutive slices of it (build_fold_cache(buffer = ), R/dataset.R).
# options(dsm.fold_buffer = FALSE) builds each fold's roles as tensors of
# their own, as before -- the comparison this script makes.
#
# WHAT IT MEASURES.
#
# The real loop, twice: dsm_train() on the deployed model's tuning plan and
# configuration, one seed a fold, a few epochs, in a fresh R process per arm
# (5 threads, as a tuning run here uses), with options(dsm.train.trace_mem =
# TRUE), under which the loop records the process's private memory, working
# set and peak -- after a collection each time -- before the first fold and,
# per fold, once its cache is built, once its unit trained, once it is
# released. Per fold after the first:
#
#   cache_added     what building the fold's cache added to the level the
#                   fold before left: without the buffer, the new tensors
#                   mimalloc could not serve from what it kept
#   left_for_next   what the fold left, over the fold before
#
# THE CHECKS: both arms trained and traced every fold (t8_01, t8_02); the
# buffer changes no number -- every numeric column of the tuning table the
# same, exactly, only the runtimes aside (t8_03); without the buffer a fold
# after the first adds at least half its largest role, which is the defect
# (t8_04); with it, less than a tenth (t8_05).
#
# WHAT IT WRITES, all under capability_sweep/t8_fold_loop/: the two arms'
# tuning runs, their logs, the trace of both (t8_trace_<commit>.csv) and the
# checks. The deployed model's tuning run is only read: its fold plan and its
# grid.
#
# COST: ~6 min.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/checks/_t8_fold_loop_memory.R")
# ══════════════════════════════════════════════════════════════════════════════

# The project root, found from where this file is: Rscript's --file, then the
# source() frame, then the working directory, climbing to R/cnn_architecture.R.
project_root <- (function() {
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
  for (d in cand) for (up in c(".", "..", "../..", "../../..")) {
    r <- normalizePath(file.path(d, up), winslash = "/", mustWork = FALSE)
    if (file.exists(file.path(r, "R", "cnn_architecture.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))
install_load_pkg(c("torch", "dplyr", "readr", "tibble", "ps", "callr"))
pkgload::load_all(project_root)
options(width = 200)

target_label <- "soc_stock_0_5cm"
base <- function(...) file.path(project_root, ..., "soc_stock_modeling", target_label)
metadata_dir <- base("outputs", "metadata")
data_dir     <- base("data", "processed")
patch_dir    <- base("outputs", "patches")
final_base   <- base("outputs", "final_model")
tuning_base  <- base("outputs", "tuning")
t8_dir       <- file.path(tuning_base, "capability_sweep", "t8_fold_loop")
code_tag     <- .git_commit_at(project_root)

t8_epochs  <- env_int("soc_t8_epochs", 3L)
t8_threads <- 5L                          # a tuning run's threads here (T1)

# ── The deployed model's configuration, and the plan it was tuned on ──────────
final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
summ       <- readRDS(file.path(final_base, final_run_id, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
tuning_src <- file.path(tuning_base, summ$tuning_run_id)
plan_file  <- file.path(tuning_src, "fold_plan.rds")
grid_file  <- file.path(tuning_src, "tune_grid.rds")
if (!all(file.exists(c(plan_file, grid_file)))) {
  stop("The deployed model's tuning run lacks its fold plan or its grid: ", tuning_src,
       call. = FALSE)
}
plan    <- readRDS(plan_file)
grid    <- readRDS(grid_file)
cfg_row <- grid[grid$config_id == config_id, , drop = FALSE]
if (nrow(cfg_row) != 1L) {
  stop("Configuration ", config_id, " is not in ", grid_file, call. = FALSE)
}
windows <- sort(as.integer(unlist(cfg_row$window_sizes)))
target_col <- safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1]

# What a fold's cache weighs, and its largest role, as float32: every role of
# every window the configuration reads.
n_ch <- as.integer(readRDS(file.path(patch_dir, "patch_manifest.rds"))$n_channels[1])
fold_gb <- dplyr::bind_rows(lapply(seq_along(plan$folds), function(j) {
  n <- lengths(plan$folds[[j]])
  tibble::tibble(fold = j, rows = paste(sprintf("%s %d", names(n), n), collapse = ", "),
                 cache_gb = sum(n) * n_ch * sum(windows^2) * 4 / 1e9,
                 largest_role_gb = max(n) * n_ch * max(windows)^2 * 4 / 1e9)
}))

message("\n", strrep("=", 78))
message("T8 -- the tuning loop's memory, fold by fold, with and without one buffer")
message(strrep("=", 78))
message("  model   : ", config_id, " of ", summ$tuning_run_id, " (deployed in ", final_run_id, ")")
message(sprintf("  plan    : %s, %d folds | window(s) %s, %d channels",
                plan$method, plan$n_folds, paste(windows, collapse = ", "), n_ch))
message(sprintf("  units   : one seed a fold, %d epochs, 1 process x %d threads per arm",
                t8_epochs, t8_threads))
message("  output  : ", t8_dir)
message(strrep("=", 78))

# ── One arm: the loop in a fresh R process ────────────────────────────────────
#
# A process of its own, so each arm starts from the same empty allocator and
# what one arm's folds leave cannot meet the other's. Everything it needs is
# an argument: callr runs the function in the child's global environment.
t8_arm <- function(project_root, fold_buffer, patch_dir, points_file, type_file,
                   raster_file, target_col, windows, plan, cfg_row, epochs, threads,
                   output_dir, run_id) {
  suppressMessages(pkgload::load_all(project_root, quiet = TRUE))
  options(dsm.fold_buffer = fold_buffer, dsm.train.trace_mem = TRUE)
  data <- dsm_load(patch_dir = patch_dir, points = points_file, type_table = type_file,
                   raster_table = raster_file, windows = windows, target_col = target_col,
                   verbose = FALSE)
  fit <- dsm_train(data, model = "cnn", resampling = plan, tune_grid = cfg_row,
                   n_seeds = 1L, output_dir = output_dir, run_id = run_id,
                   device = setup_torch_device(n_threads = threads, use_cuda = FALSE),
                   n_epochs = epochs, patience = epochs + 1L, print_every = 100L,
                   verbose = FALSE)
  list(comparison = as.data.frame(fit$comparison), run_dir = fit$run_dir)
}

arms <- tibble::tibble(arm = c("own_tensors", "one_buffer"), fold_buffer = c(FALSE, TRUE))
create_output_dirs(file.path(t8_dir, "tuning"))
runs <- list()
for (a in seq_len(nrow(arms))) {
  arm    <- arms$arm[a]
  run_id <- paste0("t8_", code_tag, "_", arm)
  unlink(file.path(t8_dir, "tuning", run_id), recursive = TRUE)    # T8's own run, at this commit
  log_file <- file.path(t8_dir, sprintf("t8_%s_%s.log", code_tag, arm))
  message(sprintf("\n  arm %-12s: dsm.fold_buffer = %s | log %s", arm, arms$fold_buffer[a], log_file))
  t0 <- Sys.time()
  out <- tryCatch(
    callr::r(t8_arm,
             args = list(project_root = project_root, fold_buffer = arms$fold_buffer[a],
                         patch_dir = patch_dir,
                         points_file = file.path(data_dir, "full_modeling_dataset_raw.csv"),
                         type_file = file.path(metadata_dir, "predictor_type_table.csv"),
                         raster_file = file.path(metadata_dir, "raster_table_used.csv"),
                         target_col = target_col, windows = windows, plan = plan,
                         cfg_row = cfg_row, epochs = t8_epochs, threads = t8_threads,
                         output_dir = file.path(t8_dir, "tuning"), run_id = run_id),
             env = c(callr::rcmd_safe_env(), OMP_NUM_THREADS = as.character(t8_threads),
                     MKL_NUM_THREADS = as.character(t8_threads)),
             stdout = log_file, stderr = "2>&1"),
    error = function(e) e)
  message(sprintf("  arm %-12s: %.1f min%s", arm,
                  as.numeric(difftime(Sys.time(), t0, units = "mins")),
                  if (inherits(out, "error")) paste(" -- FAILED:", conditionMessage(out)) else ""))
  trace_file <- file.path(t8_dir, "tuning", run_id, "logs", "train_mem_trace.rds")
  runs[[arm]] <- list(out = out,
                      trace = if (file.exists(trace_file)) readRDS(trace_file) else NULL)
}

ok_run <- function(arm) !inherits(runs[[arm]]$out, "error")
phases_expected <- c("start", rep(c("fold_cache", "fold_trained", "fold_released"), plan$n_folds))

# Per fold: the levels at its three marks, what building its cache added, and
# what it left for the next.
per_fold <- function(arm) {
  tr <- runs[[arm]]$trace
  if (is.null(tr)) return(NULL)
  at <- function(phase, j) {
    i <- which(tr$phase == phase & tr$fold %in% j)
    if (length(i)) tr$private_gb[i[1]] else NA_real_
  }
  start <- tr$private_gb[tr$phase == "start"][1]
  dplyr::bind_rows(lapply(seq_len(plan$n_folds), function(j) {
    before <- if (j == 1L) start else at("fold_released", j - 1L)
    tibble::tibble(arm = arm, fold = j, cache = at("fold_cache", j),
                   trained = at("fold_trained", j), released = at("fold_released", j),
                   cache_added = at("fold_cache", j) - before,
                   left_for_next = at("fold_released", j) - before)
  }))
}
pf <- dplyr::bind_rows(lapply(arms$arm, per_fold))
later <- function(arm, col) {
  v <- pf[[col]][pf$arm == arm & pf$fold >= 2L]
  if (length(v)) v else NA_real_
}
largest_later <- fold_gb$largest_role_gb[fold_gb$fold >= 2L]

required <- c("t8_01", "t8_02", "t8_03", "t8_04", "t8_05")
L <- check_ledger("T8")

ledger_check(L, "t8_01", "both arms trained every fold", {
  n_ok <- vapply(arms$arm, function(a) {
    if (!ok_run(a)) return(-1L)
    cmp <- runs[[a]]$out$comparison
    if (all(cmp$status == "success")) nrow(cmp) else -1L
  }, integer(1))
  list(ok = all(n_ok == plan$n_folds),
       measured = paste(sprintf("%s: %s", arms$arm,
                                ifelse(n_ok < 0L, "failed", paste(n_ok, "unit(s)"))), collapse = " | "))
})

ledger_check(L, "t8_02", "both arms traced every fold's three marks", {
  got <- vapply(arms$arm, function(a) {
    tr <- runs[[a]]$trace
    !is.null(tr) && identical(tr$phase, phases_expected) && all(is.finite(tr$private_gb))
  }, logical(1))
  list(ok = all(got), measured = sprintf("%d mark(s) expected per arm | traced: %s",
                                         length(phases_expected),
                                         paste(arms$arm[got], collapse = ", ")))
})

# THE SAME NUMBERS. The buffer holds the same values the tensors of their own
# held (tests/test_fold_cache.R compares them exactly), and each arm trains
# the same seed on the same threads, so every number in the tuning table must
# be the same -- not close. Only the runtimes differ.
ledger_check(L, "t8_03", "the buffer changes no number in the tuning table", {
  if (!all(vapply(arms$arm, ok_run, logical(1)))) {
    list(ok = FALSE, measured = "an arm did not finish")
  } else {
    num <- function(cmp) {
      cmp  <- cmp[order(cmp$unit_id), , drop = FALSE]
      rownames(cmp) <- NULL
      keep <- vapply(cmp, is.numeric, logical(1)) & names(cmp) != "runtime_min"
      cmp[, keep, drop = FALSE]
    }
    a <- num(runs$own_tensors$out$comparison)
    b <- num(runs$one_buffer$out$comparison)
    same_na <- identical(names(a), names(b)) && identical(is.na(as.matrix(a)), is.na(as.matrix(b)))
    worst <- if (same_na) max(0, abs(as.matrix(a) - as.matrix(b)), na.rm = TRUE) else NA_real_
    list(ok = same_na && identical(worst, 0),
         measured = sprintf("%d unit(s) x %d numeric column(s): worst |difference| %s",
                            nrow(a), ncol(a),
                            if (is.na(worst)) "(columns or NAs differ)" else sprintf("%.2e", worst)))
  }
})

# THE DEFECT, MEASURED: without the buffer, each fold after the first adds at
# least half of its largest role -- the one tensor surely past mimalloc's
# ~250 MB. If it does not, the loop did not leak on this store, and the
# buffer is shown here to be harmless but not needed: a finding, not a pass.
ledger_check(L, "t8_04", "without it, a later fold adds half its largest role", {
  added <- later("own_tensors", "cache_added")
  list(ok = all(is.finite(added)) && all(added >= 0.5 * largest_later),
       measured = sprintf("cache_added %s GB | half the largest role %s GB",
                          paste(sprintf("%.3f", added), collapse = ", "),
                          paste(sprintf("%.3f", 0.5 * largest_later), collapse = ", ")))
})

ledger_check(L, "t8_05", "with it, a later fold adds under a tenth of that role", {
  added <- later("one_buffer", "cache_added")
  list(ok = all(is.finite(added)) && all(added < 0.1 * largest_later),
       measured = sprintf("cache_added %s GB | a tenth of the largest role %s GB",
                          paste(sprintf("%.3f", added), collapse = ", "),
                          paste(sprintf("%.3f", 0.1 * largest_later), collapse = ", ")))
})

if (nrow(pf) > 0L) {
  message("\n-- The folds: what each cache weighs (float32, every role of every window) --")
  print_wide(dplyr::mutate(fold_gb, dplyr::across(dplyr::ends_with("_gb"), ~ round(.x, 3))), n = Inf)

  message("\n-- Per fold: private memory (GB) once its cache is built, once trained, once released --")
  message("   cache_added: what building the cache added to the level the fold before left")
  message("   left_for_next: what the fold left, over the fold before")
  print_wide(dplyr::mutate(pf, dplyr::across(-c(arm, fold), ~ round(.x, 3))), n = Inf)

  sm <- dplyr::bind_rows(lapply(arms$arm, function(a) {
    tr <- runs[[a]]$trace
    if (is.null(tr)) return(NULL)
    tibble::tibble(arm = a, private_start = tr$private_gb[1],
                   private_end = tr$private_gb[nrow(tr)],
                   growth_over_the_folds = tr$private_gb[nrow(tr)] - tr$private_gb[1],
                   peak_working_set = max(tr$peak_gb, na.rm = TRUE))
  }))
  message("\n-- The two arms, GB --")
  print_wide(dplyr::mutate(sm, dplyr::across(-arm, ~ round(.x, 2))), n = Inf)
  if (nrow(sm) == 2L) {
    d_left <- mean(later("own_tensors", "left_for_next") - later("one_buffer", "left_for_next"))
    message(sprintf(paste0("\n  Without the buffer each later fold left %.2f GB more. By proportion,",
                           " on the full data set (windows ~11x larger) that is ~%.0f GB a fold."),
                    d_left, 11 * d_left))
  }

  tr_all <- dplyr::bind_rows(lapply(arms$arm, function(a) {
    tr <- runs[[a]]$trace
    if (is.null(tr)) NULL else dplyr::mutate(tr, arm = a, .before = 1)
  }))
  create_output_dirs(t8_dir)
  safe_write_csv2(tr_all, file.path(t8_dir, sprintf("t8_trace_%s.csv", code_tag)))
}

create_output_dirs(t8_dir)
v <- ledger_verdict(L, required, file.path(t8_dir, "t8_checks.csv"))
message("\nTrace: ", file.path(t8_dir, sprintf("t8_trace_%s.csv", code_tag)))
