# ══════════════════════════════════════════════════════════════════════════════
# T1 -- how many torch threads train fastest on this machine
#
# THE QUESTION.
#
# Every training script here typed setup_torch_device(n_threads = 30): the
# author's workstation has 32 LOGICAL cores. It has 16 PHYSICAL ones, and the
# framework's rule (resolve_cores()) is physical minus one -- 15 -- because
# hyperthreads share a core's arithmetic units and torch on CPU is often
# slower, not faster, when its threads outnumber the cores. Nobody measured
# which is right for THIS network on THIS machine. The extraction taught the
# lesson on 2026-09-27: the intuition "more cores, faster" was wrong there,
# because the disk was the limit, and only a measurement said so.
#
# WHAT IS MEASURED.
#
# Seconds per training epoch of two configurations of the real network on the
# real dev store, at 5, 7, 15 and 30 threads:
#
#   heavy_3x15  dual branch 3 + 15, three conv blocks, flatten, SE -- the
#               costliest shape the grid draws, where threads have work to split
#   light_3     one 3x3 branch, two blocks, gap -- a cheap shape, where the
#               per-operation overhead is what threads cannot split
#
# Each thread count runs in its OWN R process, launched with OMP_NUM_THREADS
# and MKL_NUM_THREADS already set. OpenMP sizes its pool when torch loads;
# setting it from inside a session that has already loaded torch is exactly
# what B6 suspected of the 1.2e-4 CCC by which its subprocess differed. A
# fresh process per count takes that question off the table.
#
# 5 and 7 are there for a second question: 15 cores split into three units
# of 5 threads, or two of 7, trained side by side. If a unit at 5 threads is
# much more than a third as fast as at 15, splitting wins -- and the final
# refit with N seeds (step 3) is N independent units. The table estimates it;
# a run of units side by side is what would confirm it, because they share
# the memory bandwidth.
#
# EPOCHS PER CONFIG, AND WHY. The runner records a unit's time rounded to
# 0.01 min -- 0.6 s. Ten epochs of the light config are a few seconds, so the
# rounding alone would be several percent. 12 epochs of the heavy one and 60
# of the light one keep it near 1%, far under the noise the passes measure.
#
# TWICE, IN OPPOSITE ORDERS (15, 30, 7, 5 and back). The machine drifts --
# caches, the other programs, the temperature -- and a single pass would read
# the drift as a difference between thread counts. The two passes estimate
# the noise, and a difference smaller than it is not a result.
#
# The same seed at every count, so the numbers also say whether a run is
# reproducible at a fixed thread count, and whether the thread count itself
# changes them. That is reported, not checked: it decides nothing about speed.
#
# WHAT IT DECIDES. The n_cores the training scripts pass. If 15 is within the
# noise of the fastest, the scripts drop their 30 and take the default; if not,
# they pass the measured number, with this run as the reason.
#
# COST: 8 processes (4 counts x 2 passes), each loading the dev store's 3x3
# and 15x15 windows (~1.3 GB) and training the two configs -- roughly 15-25
# min, measured and printed per process. Leave the machine otherwise idle
# while it runs: anything else on the CPU is in the numbers.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t1_threads_benchmark.R")
# ══════════════════════════════════════════════════════════════════════════════

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/cnn_architecture.R.
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
install_load_pkg(c("torch", "coro", "dplyr", "readr", "tibble", "purrr",
                   "DescTools", "terra", "processx"))
pkgload::load_all(project_root)
options(width = 200)

# The path the workers are launched on: this file, found from the root.
t1_script <- file.path(project_root, "examples", "soc_stock_0_5cm",
                       "_t1_threads_benchmark.R")

target_label <- "soc_stock_0_5cm"
patch_dir    <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
data_dir     <- file.path(project_root, "data", "processed", "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
t1_dir       <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label,
                          "capability_sweep", "t1_threads")

# ── The settings ──────────────────────────────────────────────────────────────

t1_threads <- c(5L, 7L, 15L, 30L)
t1_order   <- list(c(15L, 30L, 7L, 5L),             # pass 1
                   c(5L, 7L, 30L, 15L))             # pass 2, the reverse
t1_epochs  <- c(heavy_3x15 = 12L, light_3 = 60L)    # see EPOCHS PER CONFIG
t1_timeout_min <- 30   # per process; a worker that takes longer has failed

# Two configurations bracketing the grid: the costliest shape and a cheap one.
t1_grid <- dplyr::bind_rows(
  make_manual_tune_grid(
    window_sizes = list(c(3L, 15L)), conv_channels = list(c(64L, 128L, 128L)),
    use_residual = TRUE, use_se_block = TRUE, embedding_dim = 384L,
    embed_pool = "flatten", gate_type = "vector_featurewise", dropout = 0.1,
    base_lr = 3e-4, weight_decay = 1e-4, batch_size = 256L, loss_fn = "smooth_l1"),
  make_manual_tune_grid(
    window_sizes = list(c(3L)), conv_channels = list(c(64L, 128L)),
    use_residual = TRUE, use_se_block = FALSE, embedding_dim = 256L,
    embed_pool = "gap", dropout = 0.1, base_lr = 3e-4, weight_decay = 1e-4,
    batch_size = 256L, loss_fn = "smooth_l1"))
t1_grid$config_id <- c("heavy_3x15", "light_3")

# Stage 03's training options, except the length: a fixed number of epochs
# (set per config below), no early stop, so every process does the same work.
t1_training_args <- list(
  es_min_delta = 0.0005, warmup_start_lr = 1e-5, lr_plateau_factor = 0.5,
  lr_plateau_patience = 25L, lr_plateau_min_delta = 0.0005, min_lr = 1e-6,
  gradient_clip = 1.0, print_every = 1L, augment = TRUE)

t1_phase <- tolower(trimws(Sys.getenv("soc_t1_phase", "")))

if (identical(t1_phase, "worker")) {
  # ── ONE THREAD COUNT, IN ITS OWN PROCESS ────────────────────────────────────
  n_thr   <- env_int("soc_t1_threads", NA_integer_)
  pass    <- env_int("soc_t1_pass", NA_integer_)
  out_csv <- Sys.getenv("soc_t1_out")
  if (is.na(n_thr) || is.na(pass) || !nzchar(out_csv)) {
    stop("T1 worker started without soc_t1_threads / soc_t1_pass / soc_t1_out.",
         call. = FALSE)
  }
  message("T1 worker: ", n_thr, " thread(s), pass ", pass, " | OMP_NUM_THREADS=",
          Sys.getenv("OMP_NUM_THREADS"))

  t_load <- Sys.time()
  data <- dsm_load(
    patch_dir    = patch_dir,
    points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
    type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
    raster_table = file.path(metadata_dir, "raster_table_used.csv"),
    windows      = c(3L, 15L),
    target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
    verbose      = FALSE)
  plan <- resolve_resampling(holdout_cv(validation_frac = 0.2, test_frac = 0.15, seed = 42L),
                             data, verbose = FALSE)
  load_sec <- as.numeric(difftime(Sys.time(), t_load, units = "secs"))

  runs_dir <- file.path(t1_dir, "runs")
  rows <- list()
  for (k in seq_len(nrow(t1_grid))) {
    cid    <- t1_grid$config_id[k]
    n_ep   <- t1_epochs[[cid]]
    run_id <- sprintf("t1_n%02d_p%d_%s", n_thr, pass, cid)
    unlink(file.path(runs_dir, run_id), recursive = TRUE)   # this script's own output
    fit <- do.call(dsm_train, c(list(
      data = data, model = "cnn", resampling = plan, tune_grid = t1_grid[k, ],
      n_seeds = 1L, base_seed = 42L, n_cores = n_thr,
      device = torch::torch_device("cpu"),
      output_dir = runs_dir, run_id = run_id, resume = FALSE, verbose = FALSE,
      n_epochs = n_ep, patience = n_ep + 1L),
      t1_training_args))
    cmp <- fit$comparison
    ran <- nrow(safe_read_csv2(file.path(runs_dir, run_id, "history",
                                         paste0(cmp$unit_id[1], "_history.csv"))))
    rows[[k]] <- tibble::tibble(
      threads       = n_thr,
      pass          = pass,
      config_id     = cid,
      n_epochs      = as.integer(ran),
      runtime_sec   = cmp$runtime_min[1] * 60,
      sec_per_epoch = cmp$runtime_min[1] * 60 / ran,
      val_ccc       = cmp$val_ccc[1],
      torch_threads = torch::torch_get_num_threads(),
      omp_env       = Sys.getenv("OMP_NUM_THREADS"),
      load_sec      = load_sec)
  }
  safe_write_csv2(dplyr::bind_rows(rows), out_csv)
  message("T1 worker done: ", out_csv)

} else {
  # ── THE PARENT: launch, collect, compare ────────────────────────────────────
  for (f in c(file.path(patch_dir, "patch_manifest.rds"),
              file.path(data_dir, "full_modeling_dataset_raw.csv"))) {
    if (!file.exists(f)) stop("The dev store is incomplete, missing: ", f, call. = FALSE)
  }
  rscript <- file.path(R.home("bin"), "Rscript.exe")
  if (!file.exists(rscript)) rscript <- file.path(R.home("bin"), "Rscript")
  if (!file.exists(rscript)) stop("Rscript not found under ", R.home("bin"), call. = FALSE)
  for (d in c("rows", "logs", "runs")) dir.create(file.path(t1_dir, d), recursive = TRUE,
                                                 showWarnings = FALSE)

  message("\n", strrep("=", 78))
  message("T1 -- torch threads against seconds per epoch, on ", .physical_cores(),
          " physical / ", parallel::detectCores(logical = TRUE), " logical core(s)")
  message(strrep("=", 78))
  message("  counts  : ", paste(t1_threads, collapse = ", "), "  (two passes, opposite orders)")
  message("  configs : ", paste(sprintf("%s (%d epochs)", names(t1_epochs), t1_epochs),
                               collapse = ", "), " | no early stop")
  message("  output  : ", t1_dir)
  message("  Leave the machine otherwise idle until it finishes.")
  message(strrep("=", 78), "\n")

  jobs <- do.call(rbind, lapply(seq_along(t1_order), function(p)
    data.frame(threads = t1_order[[p]], pass = p)))
  t_all <- Sys.time()
  exits <- integer(nrow(jobs))
  for (j in seq_len(nrow(jobs))) {
    n  <- jobs$threads[j]; p <- jobs$pass[j]
    out <- file.path(t1_dir, "rows", sprintf("t1_n%02d_p%d.csv", n, p))
    log <- file.path(t1_dir, "logs", sprintf("t1_n%02d_p%d.log", n, p))
    unlink(out)
    t0 <- Sys.time()
    pr <- processx::process$new(
      rscript, args = t1_script,
      env = c("current", soc_t1_phase = "worker", soc_t1_threads = as.character(n),
              soc_t1_pass = as.character(p), soc_t1_out = out,
              OMP_NUM_THREADS = as.character(n), MKL_NUM_THREADS = as.character(n)),
      stdout = log, stderr = "2>&1", cleanup = TRUE)
    pr$wait(timeout = t1_timeout_min * 60 * 1000)
    if (pr$is_alive()) {
      pr$kill()
      exits[j] <- NA_integer_
    } else {
      exits[j] <- pr$get_exit_status()
    }
    message(sprintf("  [%2d/%d] %2d thread(s), pass %d: %s  (%.1f min)", j, nrow(jobs), n, p,
                    if (file.exists(out)) "done" else paste0("NO RESULT -- see ", log),
                    as.numeric(difftime(Sys.time(), t0, units = "mins"))))
  }
  message(sprintf("\nAll processes finished in %.1f min.",
                  as.numeric(difftime(Sys.time(), t_all, units = "mins"))))

  rows_files <- file.path(t1_dir, "rows", sprintf("t1_n%02d_p%d.csv", jobs$threads, jobs$pass))
  res <- dplyr::bind_rows(lapply(rows_files[file.exists(rows_files)], safe_read_csv2))
  safe_write_csv2(res, file.path(t1_dir, "t1_timing.csv"))

  required <- sprintf("t1_%02d", 1:3)
  L <- check_ledger("T1")

  n_expect <- nrow(jobs) * nrow(t1_grid)
  # A process killed at the timeout has no exit code (NA): that is a failure,
  # said as one, not an NA the verdict has to interpret.
  ledger_check(L, "t1_01", "every process finished and wrote its timing",
               nrow(res) == n_expect && !anyNA(exits) && all(exits == 0L),
               sprintf("%d of %d row(s) | exit codes: %s", nrow(res), n_expect,
                       paste(exits, collapse = " ")))

  ledger_check(L, "t1_02", "each process ran at the thread count it was given", {
    okk <- nrow(res) > 0L && all(res$torch_threads == res$threads) &&
      all(as.character(res$omp_env) == as.character(res$threads))
    list(ok = okk, measured = sprintf("torch threads == asked: %s | OMP_NUM_THREADS == asked: %s",
                                      all(res$torch_threads == res$threads),
                                      all(as.character(res$omp_env) == as.character(res$threads))))
  })

  # The noise the comparison has to beat: the same count, the two passes.
  by_count <- res %>%
    dplyr::group_by(config_id, threads) %>%
    dplyr::summarise(n_pass = dplyr::n(),
                     sec_p1 = sec_per_epoch[pass == 1][1],
                     sec_p2 = sec_per_epoch[pass == 2][1],
                     sec    = mean(sec_per_epoch),
                     spread = abs(sec_p1 - sec_p2) / sec,
                     ccc_p1 = val_ccc[pass == 1][1],
                     ccc_p2 = val_ccc[pass == 2][1],
                     .groups = "drop")
  noise <- max(by_count$spread, na.rm = TRUE)
  ledger_check(L, "t1_03", "the two passes agree to within 10% at every count",
               is.finite(noise) && noise <= 0.10,
               sprintf("largest pass-to-pass spread %.1f%%", 100 * noise))

  base <- by_count %>% dplyr::filter(threads == 15L) %>%
    dplyr::select(config_id, sec_15 = sec)
  by_count <- by_count %>% dplyr::left_join(base, by = "config_id") %>%
    dplyr::mutate(speed_vs_15 = sec_15 / sec,
                  # Splitting 15 cores into units of `threads` each, run side by
                  # side. An ESTIMATE: it assumes the units do not slow each other
                  # down, and they share the memory bandwidth -- a real run of
                  # parallel units is what would confirm it.
                  units_side_by_side = pmax(1L, 15L %/% threads),
                  est_throughput_vs_15 = units_side_by_side * sec_15 / sec)

  message("\n-- Seconds per epoch (mean of the two passes) --")
  print_wide(dplyr::select(by_count, config_id, threads, sec_p1, sec_p2, sec,
                           spread, speed_vs_15, units_side_by_side,
                           est_throughput_vs_15), n = Inf)

  message("\n-- Reproducibility, reported and not judged --")
  message("  same count, both passes, identical val_ccc: ",
          sum(by_count$ccc_p1 == by_count$ccc_p2, na.rm = TRUE), " of ", nrow(by_count))
  for (cid in unique(by_count$config_id)) {
    v <- by_count$ccc_p1[by_count$config_id == cid]
    message(sprintf("  %-11s val_ccc across counts: %s  (range %.2e)", cid,
                    paste(sprintf("%.6f", v), collapse = ", "), diff(range(v, na.rm = TRUE))))
  }

  v <- ledger_verdict(L, required, file.path(t1_dir, "t1_checks.csv"))

  # THE DECISION, per config: the fastest count, and whether 15 -- the
  # default -- is within the noise of it.
  message("\n-- What it says --")
  for (cid in unique(by_count$config_id)) {
    b    <- by_count[by_count$config_id == cid, ]
    best <- b[which.min(b$sec), ]
    gap  <- b$sec[b$threads == 15L] / best$sec - 1
    message(sprintf("  %-11s fastest at %d thread(s), %.2f s/epoch; the default 15 is %.1f%% slower%s",
                    cid, best$threads, best$sec, 100 * gap,
                    if (gap <= noise) " -- within the noise" else ""))
  }
  if (!v$pass) {
    message("\nThe checks above did not all pass, so the timings are not to be read yet.")
  }
}
