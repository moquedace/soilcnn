# ══════════════════════════════════════════════════════════════════════════════
# T2 -- training units side by side: measured, not estimated
#
# THE QUESTION.
#
# T1 (2026-09-27) found that one CNN unit uses this CPU poorly: tripling its
# threads from 5 to 15 made the heavy configuration only 1.5x faster. So three
# units of 5 threads each, trained side by side, were ESTIMATED at 2.0x the
# throughput of one unit of 15 -- an estimate that assumed the units do not
# slow each other down. They share the memory bandwidth and the L3 cache, and
# only running them together says by how much.
#
# It matters because the final model is N independent units: the selected
# configuration refitted under N seeds (step 3, dsm_final()). If side by side
# is faster, that is how dsm_final() trains them; if not, it trains them one
# after another as stage 04 does now.
#
# WHAT IS MEASURED.
#
# The same six units -- T1's heavy_3x15 configuration, seeds 42 to 47, 12
# epochs each, no early stop -- trained three ways:
#
#   1x15   one process, 15 threads, the six units in sequence (stage 04 today)
#   3x5    three processes side by side, 5 threads each, two units each
#   2x7    two processes side by side, 7 threads each, three units each
#
# and the wall time from the moment every process of an arrangement is ready
# to the moment the last one finishes. A BARRIER starts them together: each
# process loads the store, writes a flag and waits for the others. Without it
# the three would read 1.3 GB each from the same hard disk at once, and the
# comparison would be about the disk -- which T1's companion measurement
# already showed is slow here -- instead of about training.
#
# The fold cache (the scaling and the tensors) is built INSIDE the timed span,
# by every process, because that is what dsm_final() will do: torch tensors
# cannot be handed from one R process to another, so each worker builds its own.
#
# TWICE, IN OPPOSITE ORDERS, as in T1: a machine under full load for half an
# hour warms up, and a single pass would credit the arrangement that ran first.
#
# THE CHECK THAT MATTERS MOST (t2_04). Seed 42 at 5 threads, trained BESIDE two
# other units, must give exactly the val_ccc it gave ALONE in T1 -- and the same
# at 7 and at 15 threads. T1 showed a unit is bit-identical at a fixed thread
# count and changes when the count changes. If running side by side changed
# it too -- an OpenMP that shrinks its pool under load would -- then a result
# would depend on how many units happened to fit on the machine, and the
# parallel design would be out whatever its speed.
#
# WHAT IT DECIDES: how dsm_final() spends n_cores -- one unit of 15 threads at a
# time, or units of fewer threads side by side -- and how many threads a unit
# gets, which then becomes a recorded part of every result.
#
# COST: 2 passes x 3 arrangements, ~20-30 min, measured and printed per
# arrangement. Leave the machine otherwise idle. Each process deletes the
# checkpoints it wrote (36 x ~49 MB) once its timings are on disk; the timings,
# logs and histories stay.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t2_parallel_units.R")
# ══════════════════════════════════════════════════════════════════════════════

# WHERE THIS PROJECT IS, FOUND RATHER THAN REMEMBERED.
#
# This was a hardcoded "D:/usuario_armazenamento/...", which meant the script
# ran on exactly one machine and had to be edited on every other. The same
# snippet is in every tests/*.R and in R/load_all.R: it asks Rscript (--file),
# then source() (the ofile of an enclosing frame), then the working directory,
# and climbs until it finds the directory that holds R/load_all.R.
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
    if (file.exists(file.path(r, "R", "load_all.R"))) return(r)
  }
  stop("Project root not found. source() this script by its full path, or ",
       "setwd() into the project first.", call. = FALSE)
})()
source(file.path(project_root, "utils", "install_load_pkg.R"))
install_load_pkg(c("torch", "coro", "dplyr", "readr", "tibble", "purrr",
                   "DescTools", "terra", "processx", "ps"))
source(file.path(project_root, "R", "load_all.R"))
options(width = 200)

t2_script <- file.path(project_root, "examples", "soc_stock_0_5cm",
                       "_t2_parallel_units.R")

target_label <- "soc_stock_0_5cm"
patch_dir    <- file.path(project_root, "outputs", "patches", "soc_stock_modeling", target_label)
data_dir     <- file.path(project_root, "data", "processed", "soc_stock_modeling", target_label)
metadata_dir <- file.path(project_root, "outputs", "metadata", "soc_stock_modeling", target_label)
sweep_dir    <- file.path(project_root, "outputs", "tuning", "soc_stock_modeling", target_label,
                          "capability_sweep")
t2_dir       <- file.path(sweep_dir, "t2_parallel_units")
t1_timing    <- file.path(sweep_dir, "t1_threads", "t1_timing.csv")

# ── The settings ──────────────────────────────────────────────────────────────

t2_arms <- list(
  "1x15" = list(workers = 1L, threads = 15L),
  "3x5"  = list(workers = 3L, threads = 5L),
  "2x7"  = list(workers = 2L, threads = 7L))
t2_order  <- list(c("1x15", "3x5", "2x7"),          # pass 1
                  c("2x7", "3x5", "1x15"))          # pass 2, the reverse
t2_seeds  <- 42:47                                  # six: divisible by 1, 2 and 3
t2_epochs <- 12L                                    # T1's, so seed 42 can be compared
t2_timeout_min <- 45   # per arrangement

# T1's heavy configuration, exactly: t2_04 compares against T1's numbers.
t2_grid <- make_manual_tune_grid(
  window_sizes = list(c(3L, 15L)), conv_channels = list(c(64L, 128L, 128L)),
  use_residual = TRUE, use_se_block = TRUE, embedding_dim = 384L,
  embed_pool = "flatten", gate_type = "vector_featurewise", dropout = 0.1,
  base_lr = 3e-4, weight_decay = 1e-4, batch_size = 256L, loss_fn = "smooth_l1")
t2_grid$config_id <- "heavy_3x15"

t2_training_args <- list(
  n_epochs = t2_epochs, patience = t2_epochs + 1L, es_min_delta = 0.0005,
  warmup_start_lr = 1e-5, lr_plateau_factor = 0.5, lr_plateau_patience = 25L,
  lr_plateau_min_delta = 0.0005, min_lr = 1e-6, gradient_clip = 1.0,
  print_every = 1L, augment = TRUE)

t2_phase <- tolower(trimws(Sys.getenv("soc_t2_phase", "")))

if (identical(t2_phase, "worker")) {
  # ── ONE PROCESS OF ONE ARRANGEMENT ──────────────────────────────────────────
  arm_id    <- Sys.getenv("soc_t2_arm")
  pass      <- env_int("soc_t2_pass", NA_integer_)
  worker    <- env_int("soc_t2_worker", NA_integer_)
  n_workers <- env_int("soc_t2_n_workers", NA_integer_)
  n_thr     <- env_int("soc_t2_threads", NA_integer_)
  seed0     <- env_int("soc_t2_base_seed", NA_integer_)
  n_seeds   <- env_int("soc_t2_n_seeds", NA_integer_)
  out_csv   <- Sys.getenv("soc_t2_out")
  ready_dir <- Sys.getenv("soc_t2_ready_dir")
  if (anyNA(c(pass, worker, n_workers, n_thr, seed0, n_seeds)) ||
      !nzchar(out_csv) || !nzchar(ready_dir) || !nzchar(arm_id)) {
    stop("T2 worker started without its settings.", call. = FALSE)
  }
  message("T2 worker ", worker, "/", n_workers, " of ", arm_id, ", pass ", pass,
          ": seeds ", seed0, "..", seed0 + n_seeds - 1L, " at ", n_thr,
          " thread(s) | OMP_NUM_THREADS=", Sys.getenv("OMP_NUM_THREADS"))

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

  # THE BARRIER: say ready, wait for the others, then start together.
  file.create(file.path(ready_dir, sprintf("w%d.flag", worker)))
  t_wait <- Sys.time()
  repeat {
    if (length(list.files(ready_dir, pattern = "\\.flag$")) >= n_workers) break
    if (as.numeric(difftime(Sys.time(), t_wait, units = "mins")) > 20) {
      stop("T2 worker ", worker, ": the other process(es) never became ready.",
           call. = FALSE)
    }
    Sys.sleep(0.2)
  }
  t_start <- Sys.time()

  runs_dir <- file.path(t2_dir, "runs")
  run_id   <- sprintf("t2_%s_p%d_w%d", arm_id, pass, worker)
  unlink(file.path(runs_dir, run_id), recursive = TRUE)   # this script's own output
  fit <- do.call(dsm_train, c(list(
    data = data, model = "cnn", resampling = plan, tune_grid = t2_grid,
    n_seeds = n_seeds, base_seed = seed0, n_cores = n_thr,
    device = torch::torch_device("cpu"),
    output_dir = runs_dir, run_id = run_id, resume = FALSE, verbose = FALSE),
    t2_training_args))
  t_end <- Sys.time()

  mi <- ps::ps_memory_info(ps::ps_handle())
  peak_gb <- as.numeric(if ("peak_wset" %in% names(mi)) mi[["peak_wset"]] else mi[["rss"]]) / 1e9

  cmp <- fit$comparison
  n_ep <- vapply(cmp$unit_id, function(u) {
    nrow(safe_read_csv2(file.path(runs_dir, run_id, "history", paste0(u, "_history.csv"))))
  }, integer(1))
  safe_write_csv2(tibble::tibble(
    arm           = arm_id,
    pass          = pass,
    worker        = worker,
    threads       = n_thr,
    seed          = cmp$seed,
    n_epochs      = as.integer(n_ep),
    runtime_sec   = cmp$runtime_min * 60,
    sec_per_epoch = cmp$runtime_min * 60 / n_ep,
    val_ccc       = cmp$val_ccc,
    torch_threads = torch::torch_get_num_threads(),
    omp_env       = Sys.getenv("OMP_NUM_THREADS"),
    load_sec      = load_sec,
    t_start       = as.numeric(t_start),
    t_end         = as.numeric(t_end),
    peak_gb       = peak_gb), out_csv)
  # The checkpoints are this script's own and nobody reads them: 36 of ~49 MB.
  unlink(file.path(runs_dir, run_id, "models"), recursive = TRUE)
  message("T2 worker done: ", out_csv)

} else {
  # ── THE PARENT: launch each arrangement, collect, compare ──────────────────
  for (f in c(file.path(patch_dir, "patch_manifest.rds"),
              file.path(data_dir, "full_modeling_dataset_raw.csv"))) {
    if (!file.exists(f)) stop("The dev store is incomplete, missing: ", f, call. = FALSE)
  }
  rscript <- file.path(R.home("bin"), "Rscript.exe")
  if (!file.exists(rscript)) rscript <- file.path(R.home("bin"), "Rscript")
  if (!file.exists(rscript)) stop("Rscript not found under ", R.home("bin"), call. = FALSE)
  for (d in c("rows", "logs", "runs", "ready")) {
    dir.create(file.path(t2_dir, d), recursive = TRUE, showWarnings = FALSE)
  }

  message("\n", strrep("=", 78))
  message("T2 -- the same 6 units, one after another or side by side")
  message(strrep("=", 78))
  message("  arrangements : ", paste(names(t2_arms), collapse = ", "),
          "  (two passes, opposite orders)")
  message("  units        : heavy_3x15, seeds ", min(t2_seeds), "-", max(t2_seeds),
          ", ", t2_epochs, " epochs each, no early stop")
  message("  output       : ", t2_dir)
  message("  Leave the machine otherwise idle until it finishes.")
  message(strrep("=", 78), "\n")

  exits <- list()
  t_all <- Sys.time()
  for (p in seq_along(t2_order)) {
    for (arm_id in t2_order[[p]]) {
      arm    <- t2_arms[[arm_id]]
      chunks <- split(t2_seeds, rep(seq_len(arm$workers), each = length(t2_seeds) / arm$workers))
      ready_dir <- file.path(t2_dir, "ready", sprintf("%s_p%d", arm_id, p))
      unlink(ready_dir, recursive = TRUE)
      dir.create(ready_dir, recursive = TRUE, showWarnings = FALSE)
      procs <- list()
      for (w in seq_len(arm$workers)) {
        tag <- sprintf("t2_%s_p%d_w%d", arm_id, p, w)
        out <- file.path(t2_dir, "rows", paste0(tag, ".csv"))
        unlink(out)
        procs[[w]] <- processx::process$new(
          rscript, args = t2_script,
          env = c("current", soc_t2_phase = "worker", soc_t2_arm = arm_id,
                  soc_t2_pass = as.character(p), soc_t2_worker = as.character(w),
                  soc_t2_n_workers = as.character(arm$workers),
                  soc_t2_threads = as.character(arm$threads),
                  soc_t2_base_seed = as.character(chunks[[w]][1]),
                  soc_t2_n_seeds = as.character(length(chunks[[w]])),
                  soc_t2_out = out, soc_t2_ready_dir = ready_dir,
                  OMP_NUM_THREADS = as.character(arm$threads),
                  MKL_NUM_THREADS = as.character(arm$threads)),
          stdout = file.path(t2_dir, "logs", paste0(tag, ".log")), stderr = "2>&1",
          cleanup = TRUE)
      }
      t0 <- Sys.time()
      for (w in seq_along(procs)) {
        left_ms <- max(1, t2_timeout_min * 60 * 1000 -
                          1000 * as.numeric(difftime(Sys.time(), t0, units = "secs")))
        procs[[w]]$wait(timeout = left_ms)
        # A process still alive at the timeout is killed and has NO exit code:
        # NA, which t2_01 reads as the failure it is.
        alive <- procs[[w]]$is_alive()
        if (alive) procs[[w]]$kill()
        exits[[sprintf("%s_p%d_w%d", arm_id, p, w)]] <-
          if (alive) NA_integer_ else procs[[w]]$get_exit_status()
      }
      message(sprintf("  pass %d, %-4s: %d process(es) finished in %.1f min (loading included)",
                      p, arm_id, arm$workers,
                      as.numeric(difftime(Sys.time(), t0, units = "mins"))))
    }
  }
  message(sprintf("\nAll arrangements finished in %.1f min.",
                  as.numeric(difftime(Sys.time(), t_all, units = "mins"))))

  rows_files <- list.files(file.path(t2_dir, "rows"), pattern = "\\.csv$", full.names = TRUE)
  res <- dplyr::bind_rows(lapply(rows_files, safe_read_csv2))
  safe_write_csv2(res, file.path(t2_dir, "t2_units.csv"))
  exit_codes <- unlist(exits)

  required <- sprintf("t2_%02d", 1:4)
  L <- check_ledger("T2")

  n_expect <- length(t2_order) * length(t2_arms) * length(t2_seeds)
  ledger_check(L, "t2_01", "every process finished and wrote its units",
               nrow(res) == n_expect && !anyNA(exit_codes) && all(exit_codes == 0L),
               sprintf("%d of %d unit row(s) | exit codes: %s", nrow(res), n_expect,
                       paste(exit_codes, collapse = " ")))

  ledger_check(L, "t2_02", "each process ran at the thread count it was given", {
    okk <- nrow(res) > 0L && all(res$torch_threads == res$threads) &&
      all(as.character(res$omp_env) == as.character(res$threads))
    list(ok = okk, measured = sprintf("torch threads == asked: %s | OMP_NUM_THREADS == asked: %s",
                                      all(res$torch_threads == res$threads),
                                      all(as.character(res$omp_env) == as.character(res$threads))))
  })

  # Determinism UNDER LOAD: the same seed at the same count, in the two passes.
  ledger_check(L, "t2_03", "the same seed at the same thread count gives the same val_ccc in both passes", {
    d <- res %>% dplyr::group_by(threads, seed) %>%
      dplyr::summarise(n = dplyr::n(), same = dplyr::n_distinct(val_ccc) == 1L, .groups = "drop")
    list(ok = nrow(d) > 0L && all(d$n == 2L) && all(d$same),
         measured = sprintf("%d of %d (seed, threads) pair(s) identical across passes",
                            sum(d$same), nrow(d)))
  })

  # Side by side against ALONE: seed 42 here against seed 42 in T1.
  ledger_check(L, "t2_04", "seed 42 beside other units gives the val_ccc it gave alone in T1", {
    if (!file.exists(t1_timing)) {
      list(ok = FALSE, measured = paste("T1's timings are not on disk:", t1_timing,
                                        "-- this check compares against them; run T1 first"))
    } else {
      t1 <- safe_read_csv2(t1_timing) %>%
        dplyr::filter(config_id == "heavy_3x15") %>%
        dplyr::group_by(threads) %>% dplyr::summarise(t1_ccc = dplyr::first(val_ccc), .groups = "drop")
      here <- res %>% dplyr::filter(seed == 42L) %>%
        dplyr::group_by(threads) %>% dplyr::summarise(t2_ccc = dplyr::first(val_ccc), .groups = "drop") %>%
        dplyr::inner_join(t1, by = "threads")
      list(ok = nrow(here) == 3L && all(here$t2_ccc == here$t1_ccc),
           measured = paste(sprintf("%d thr: %.6f vs %.6f", here$threads, here$t2_ccc, here$t1_ccc),
                            collapse = " | "))
    }
  })

  # ── The comparison ─────────────────────────────────────────────────────────
  by_arm <- res %>%
    dplyr::group_by(arm, pass) %>%
    dplyr::summarise(workers = dplyr::n_distinct(worker), threads = dplyr::first(threads),
                     units = dplyr::n(),
                     wall_min = (max(t_end) - min(t_start)) / 60,
                     sec_per_epoch = mean(sec_per_epoch),
                     peak_gb_per_process = max(peak_gb),
                     .groups = "drop")
  ref <- by_arm %>% dplyr::filter(arm == "1x15") %>% dplyr::select(pass, wall_ref = wall_min)
  by_arm <- by_arm %>% dplyr::left_join(ref, by = "pass") %>%
    dplyr::mutate(speedup_vs_1x15 = wall_ref / wall_min) %>%
    dplyr::select(-wall_ref)

  # What a unit costs beside others against what it cost ALONE in T1 at the
  # same count: the slowdown the neighbours cause.
  if (file.exists(t1_timing)) {
    alone <- safe_read_csv2(t1_timing) %>% dplyr::filter(config_id == "heavy_3x15") %>%
      dplyr::group_by(threads) %>% dplyr::summarise(sec_alone = mean(sec_per_epoch), .groups = "drop")
    by_arm <- by_arm %>% dplyr::left_join(alone, by = "threads") %>%
      dplyr::mutate(slowdown_from_neighbours = sec_per_epoch / sec_alone)
  }

  message("\n-- Wall time for the same 6 units, per arrangement and pass --")
  print_wide(by_arm, n = Inf)

  summ <- by_arm %>% dplyr::group_by(arm) %>%
    dplyr::summarise(wall_min = mean(wall_min), speedup = mean(speedup_vs_1x15),
                     spread = diff(range(speedup_vs_1x15)), .groups = "drop") %>%
    dplyr::arrange(wall_min)
  message("\n-- Mean of the two passes --")
  print_wide(summ, n = Inf)

  v <- ledger_verdict(L, required, file.path(t2_dir, "t2_checks.csv"))

  message("\n-- What it says --")
  best <- summ[1, ]
  message(sprintf("  fastest: %s, %.1f min for 6 units -- %.2fx the sequential 1x15 (passes differ by %.2f)",
                  best$arm, best$wall_min, best$speedup, best$spread))
  if (!v$pass) {
    message("\nThe checks above did not all pass, so the timings are not to be read yet.")
  }
}
