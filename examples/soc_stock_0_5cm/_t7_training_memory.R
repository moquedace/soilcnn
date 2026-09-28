# ══════════════════════════════════════════════════════════════════════════════
# T7 -- where a training worker's memory goes: its working set and its private
# memory after every phase and every epoch, on the dev store
#
# THE QUESTION.
#
# T2 (2026-09-27) found, without looking for it, ~10 GB at the peak of a
# training worker holding 1.26 GB of loaded windows -- about 8x -- and about
# the same whether the process trained 2 units or 6 (10.0 against 10.3 GB).
# On the full data set (41 thousand points, windows ~11x larger) the same
# proportion is over 100 GB for one process: the final fit could not run
# there, side by side or not. But a peak says only how high memory once went.
# It says neither which phase set it nor whether the level climbs from epoch
# to epoch or from unit to unit -- the two shapes a leak takes -- and a level
# that climbs can stay under a peak set earlier.
#
# The first suspect is known. mimalloc, under libtorch on this Windows
# machine, neither returns nor reuses a freed tensor of ~250 MB or more (T6),
# and that is what the map's worker leaked until its step's window was made
# once (2504916). A training worker's setup frees tensors of that size: the
# copies a window passes through on its way from an R array to a float
# tensor, the clone the fold cache scales, and the store's raw windows once
# the cache exists.
#
# WHAT IT MEASURES.
#
# The REAL fit, not a copy of its code: dsm_final() on a throwaway copy of the
# deployed model's tuning run, with options(dsm.final.trace_mem = TRUE), under
# which the worker records -- after a collection each time -- its working set,
# its private memory, its peak working set and R's own heap:
#
#   start          the framework loaded, nothing else
#   store_loaded   dsm_load(): the windows as float tensors
#   fold_cache     build_fold_cache(): scaled, and split by role
#   store_dropped  the store's raw windows released
#   and per unit   unit_start (its loaders built), every epoch, unit_trained,
#                  unit_released (after the unit's collection)
#
# One worker of 5 threads, the shape of a real fit; the deployed
# configuration; three seeds of 30 epochs each, without early stopping. Three
# units in ONE process, so what a unit leaves behind, the next one meets. The
# worker runs with MIMALLOC_SHOW_STATS=1, and mimalloc's own account at its
# exit is printed.
#
# WHAT IT PRINTS: the level after each setup phase and what each phase added;
# per unit, the level at its start, at epochs 1, 10 and the last, and after
# it, with the slope per epoch; the level each unit leaves for the next; the
# phase that set the peak. The checks say only that the measurement is
# complete and that the original tuning run was not touched: what the numbers
# mean is read from the tables.
#
# WHAT IT WRITES, all under capability_sweep/t7_memory/: a final run of three
# seeds (three checkpoints, their tables and logs), the copy of the tuning run
# it was fitted from, the trace and the checks.
#
# COST: ~10 min.
#
# Run: source("D:/usuario_armazenamento/cassio/R/deep_learning_caret/examples/soc_stock_0_5cm/_t7_training_memory.R")
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
t7_dir       <- file.path(tuning_base, "capability_sweep", "t7_memory")
code_tag     <- .git_commit_at(project_root)

t7_seeds   <- 42:44                       # three units, one after another, in one worker
t7_epochs  <- env_int("soc_t7_epochs", 30L)
t7_threads <- 5L                          # dsm_final()'s default: T2's best

# ── The deployed model: its configuration, and the tuning run it came from ────
final_run_id <- latest_run_dir(final_base, prefix = "final_",
                               require_file = file.path("comparison", "final_run_summary.rds"),
                               label = "final_run_id")
summ       <- readRDS(file.path(final_base, final_run_id, "comparison", "final_run_summary.rds"))
config_id  <- selected_config_id(summ, final_run_id)
cfg_row    <- summ$selected_cfgs[summ$selected_cfgs$config_id == config_id, , drop = FALSE]
windows    <- sort(as.integer(unlist(cfg_row$window_sizes)))
tuning_src <- file.path(tuning_base, summ$tuning_run_id)

# ── A throwaway copy of the tuning run ────────────────────────────────────────
#
# dsm_final() freezes its choice in the tuning run it is given, and the
# original's comparison/selection.rds is the evidence that the deployed model
# was chosen before the test set was seen (B2's reasoning). A memory
# measurement has no business near it. The fit runs on a copy carrying what
# dsm_final() reads -- the plan, the grid, the ranking, the clamp when there
# is one, the cross-validated predictions -- and not the frozen selection;
# the original is fingerprinted before and after (t7_03).
src_files <- c("fold_plan.rds", "tune_grid.rds", "clamp.rds",
               file.path("comparison", c("comparison_ranked.csv", "comparison_by_config.csv")))
src_files <- src_files[file.exists(file.path(tuning_src, src_files))]
must <- c("fold_plan.rds", "tune_grid.rds", file.path("comparison", "comparison_ranked.csv"))
if (!all(must %in% src_files)) {
  stop("The deployed model's tuning run lacks ", paste(setdiff(must, src_files), collapse = ", "),
       ": ", tuning_src, call. = FALSE)
}
fingerprint <- function() {
  f <- unique(c(file.path(tuning_src, src_files),
                list.files(file.path(tuning_src, "comparison"), full.names = TRUE)))
  list(md5 = tools::md5sum(f), n_files = length(list.files(tuning_src, recursive = TRUE)))
}
before <- fingerprint()

tuning_copy <- file.path(t7_dir, "tuning", summ$tuning_run_id)
unlink(tuning_copy, recursive = TRUE)                  # T7's own copy, from a run before
create_output_dirs(file.path(tuning_copy, c("comparison", "predictions")))
copied <- c(file.copy(file.path(tuning_src, src_files), file.path(tuning_copy, src_files)),
            file.copy(list.files(file.path(tuning_src, "predictions"), full.names = TRUE),
                      file.path(tuning_copy, "predictions")))
if (!all(copied)) stop("Could not copy the tuning run into ", tuning_copy, call. = FALSE)

# ── The data, as stage 04 loads it ────────────────────────────────────────────
data <- dsm_load(
  patch_dir    = patch_dir,
  points       = file.path(data_dir, "full_modeling_dataset_raw.csv"),
  type_table   = file.path(metadata_dir, "predictor_type_table.csv"),
  raster_table = file.path(metadata_dir, "raster_table_used.csv"),
  windows      = windows,
  target_col   = safe_read_csv2(file.path(metadata_dir, "target_config.csv"))$target_col[1],
  verbose      = FALSE)

run_id  <- paste0("t7_", code_tag)
run_dir <- file.path(t7_dir, "final", run_id)
message("\n", strrep("=", 78))
message("T7 -- a training worker's memory, phase by phase and epoch by epoch")
message(strrep("=", 78))
message("  model   : ", config_id, " of ", summ$tuning_run_id, " (deployed in ", final_run_id, ")")
message(sprintf("  data    : %s points x %d channels, window(s) %s",
                format(nrow(data$store$meta), big.mark = ","), data$store$n_channels,
                paste(windows, collapse = ", ")))
message(sprintf("  units   : seeds %s, %d epochs each, no early stop, 1 worker x %d threads",
                paste(t7_seeds, collapse = ", "), t7_epochs, t7_threads))
message("  output  : ", run_dir)
message("  About 10 minutes; the worker's own log is ", file.path(run_dir, "logs", "worker_01.log"))
message(strrep("=", 78))

# ── The fit, traced ───────────────────────────────────────────────────────────
unlink(run_dir, recursive = TRUE)                      # T7's own run, at this commit
old_opts <- options(dsm.final.trace_mem = TRUE,
                    dsm.final.worker_env = c(MIMALLOC_SHOW_STATS = "1"))
t0  <- Sys.time()
fin <- tryCatch(
  dsm_final(tuning = tuning_copy, data = data, config = config_id, seeds = t7_seeds,
            training = list(n_epochs = t7_epochs, patience = t7_epochs + 1L, print_every = 10L),
            n_cores = t7_threads, threads_per_unit = t7_threads,
            output_dir = file.path(t7_dir, "final"), run_id = run_id, verbose = FALSE),
  error = function(e) e)
options(old_opts)
message(sprintf("\nThe fit took %.1f min.", as.numeric(difftime(Sys.time(), t0, units = "mins"))))
after <- fingerprint()

trace_file <- file.path(run_dir, "logs", "worker_01_mem_trace.rds")
tr <- if (file.exists(trace_file)) readRDS(trace_file) else NULL
setup_phases <- c("start", "store_loaded", "fold_cache", "store_dropped")

required <- c("t7_01", "t7_02", "t7_03")
L <- check_ledger("T7")

ledger_check(L, "t7_01", "the fit finished every seed",
             !inherits(fin, "error") &&
               identical(sort(as.integer(unique(fin$all_seed_results$seed))), t7_seeds),
             if (inherits(fin, "error")) conditionMessage(fin) else
               sprintf("%d seed(s) by %d worker(s)", length(unique(fin$all_seed_results$seed)),
                       fin$n_workers))

ledger_check(L, "t7_02", "the worker traced every phase and every epoch", {
  if (is.null(tr)) {
    list(ok = FALSE, measured = paste("no trace at", trace_file))
  } else {
    ut <- tr[!is.na(tr$unit_id), , drop = FALSE]
    n_ep <- vapply(split(ut$phase, ut$unit_id), function(p) sum(p == "epoch"), integer(1))
    has_marks <- vapply(split(ut$phase, ut$unit_id), function(p)
      all(c("unit_start", "unit_trained", "unit_released") %in% p), logical(1))
    list(ok = all(setup_phases %in% tr$phase) && length(n_ep) == length(t7_seeds) &&
           all(n_ep == t7_epochs) && all(has_marks),
         measured = sprintf("%d mark(s) | epochs traced per unit: %s", nrow(tr),
                            paste(n_ep, collapse = ", ")))
  }
})

ledger_check(L, "t7_03", "the original tuning run is untouched",
             identical(before$md5, after$md5) && identical(before$n_files, after$n_files),
             sprintf("%d file(s) fingerprinted | %d file(s) in the run before, %d after",
                     length(before$md5), before$n_files, after$n_files))

if (!is.null(tr) && nrow(tr) > 0L) {
  # ── the setup ───────────────────────────────────────────────────────────────
  st <- tr[is.na(tr$unit_id), c("phase", "seconds", "rss_gb", "private_gb", "peak_gb", "r_heap_gb")]
  st$private_added <- c(NA_real_, diff(st$private_gb))
  message("\n-- The setup: the level after each phase, GB (private_added: what the phase kept) --")
  print_wide(dplyr::mutate(st, dplyr::across(-phase, ~ round(.x, 2))), n = Inf)

  # ── the units ───────────────────────────────────────────────────────────────
  ut <- tr[!is.na(tr$unit_id), , drop = FALSE]
  at <- function(u, phase, epoch = NA_integer_) {
    i <- if (is.na(epoch)) which(u$phase == phase) else which(u$phase == phase & u$epoch == epoch)
    if (length(i)) u$private_gb[i[1]] else NA_real_
  }
  slope_mb <- function(ep, col) {
    ep <- ep[ep$epoch >= 5L, , drop = FALSE]      # past the first epochs' allocations
    if (nrow(ep) < 3L || !any(is.finite(ep[[col]]))) return(NA_real_)
    1000 * unname(stats::coef(stats::lm(ep[[col]] ~ ep$epoch))[2])
  }
  per_unit <- dplyr::bind_rows(lapply(split(ut, ut$unit_id), function(u) {
    ep <- u[u$phase == "epoch", , drop = FALSE]
    last <- if (nrow(ep)) max(ep$epoch) else NA_integer_
    tibble::tibble(
      unit_id = u$unit_id[1], start = at(u, "unit_start"), epoch_1 = at(u, "epoch", 1L),
      epoch_10 = at(u, "epoch", 10L), epoch_last = at(u, "epoch", last),
      trained = at(u, "unit_trained"), released = at(u, "unit_released"),
      mb_per_epoch_private = slope_mb(ep, "private_gb"),
      mb_per_epoch_rss = slope_mb(ep, "rss_gb"),
      peak = max(u$peak_gb), minutes = (max(u$seconds) - min(u$seconds)) / 60)
  }))
  message("\n-- Per unit: private memory (GB) at its start, at epochs 1, 10 and the last, and after it --")
  message("   mb_per_epoch_*: the slope of the level over epochs 5 onwards, in MB an epoch")
  print_wide(dplyr::mutate(per_unit, dplyr::across(-unit_id, ~ round(.x, 3))), n = Inf)

  rel <- per_unit$released
  message(sprintf("\n  What each unit leaves for the next (private, GB): %s  (%+.2f GB a unit)",
                  paste(sprintf("%.2f", rel), collapse = " -> "),
                  if (length(rel) > 1L) (rel[length(rel)] - rel[1]) / (length(rel) - 1L) else NA_real_))

  pk <- max(tr$peak_gb, na.rm = TRUE)
  first <- which(tr$peak_gb >= pk - 0.005)[1]
  message(sprintf("  Peak working set %.2f GB, first reached by the mark '%s'%s -- set between it and the mark before.",
                  pk, tr$phase[first],
                  if (!is.na(tr$unit_id[first])) paste0(" of ", tr$unit_id[first]) else ""))
  if (!inherits(fin, "error")) {
    message(sprintf("  dsm_final()'s estimate for such a worker: %.1f GB (1.5 + 7 x the windows, from T2).",
                    fin$per_worker_gb_estimate))
  }
  create_output_dirs(t7_dir)
  safe_write_csv2(tr, file.path(t7_dir, "t7_trace.csv"))
}

# ── mimalloc, in its own words ────────────────────────────────────────────────
log_file <- file.path(run_dir, "logs", "worker_01.log")
if (file.exists(log_file)) {
  lines <- readLines(log_file, warn = FALSE)
  hit <- grep("heap stats|mimalloc", lines, ignore.case = TRUE)
  if (length(hit)) {
    message("\n-- mimalloc's account at the worker's exit (", log_file, ") --")
    message(paste(utils::head(lines[hit[1]:length(lines)], 60), collapse = "\n"))
  } else {
    message("\n  mimalloc printed no statistics into ", log_file,
            " -- MIMALLOC_SHOW_STATS did not reach the worker.")
  }
}

create_output_dirs(t7_dir)
v <- ledger_verdict(L, required, file.path(t7_dir, "t7_checks.csv"))
message("\nTrace: ", file.path(t7_dir, "t7_trace.csv"))
