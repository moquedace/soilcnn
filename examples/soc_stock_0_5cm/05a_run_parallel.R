source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a — Orchestrate parallel spatial prediction
#
# Single entry point: set n_workers below and run this script ONCE. It spawns
# n_workers background `Rscript 05_predict_spatial.R <id> <n_workers>`
# processes, waits for all of them, then automatically runs
# 05b_merge_spatial_parts.R to mosaic the results. No manual terminal juggling.
# ══════════════════════════════════════════════════════════════════════════════

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

# ── Settings ────────────────────────────────────────────────────────────────────

# IMPORTANT: this is a GLOBAL raster (160k+ columns), so RAM — not CPU cores —
# is the binding constraint. Each worker pays for the FULL row width regardless
# of how few rows it owns (~240 MB/row at 187 channels), creating a per-worker
# RAM floor around ~7-8 GB peak that persists even at a tiny max_strip_ram_gb.
# n_workers and max_strip_ram_gb (in 05_predict_spatial.R) must be tuned
# TOGETHER against total RAM.
#
# REVISED after observing real usage: n_workers=4 with max_strip_ram_gb=5
# measured ~62.7/63.1 GB (99%!) on a 64 GB machine — the empirical per-worker
# peak (~15.7 GB) ran higher than the earlier estimate (~13.4 GB), leaving
# almost no margin. Dropped to n_workers=3 (same max_strip_ram_gb=5, so the
# I/O over-read ratio per block is unchanged) -> ~3 x 15.7 = ~47 GB estimated
# peak, ~16 GB headroom. CPU was only ~20% utilised at 4 workers anyway (32
# logical threads, RAM was always the binding constraint, not cores), so
# losing one worker costs little real throughput.
n_workers <- 3   # <<< EDIT THIS together with max_strip_ram_gb in 05_predict_spatial.R

# ── Paths ─────────────────────────────────────────────────────────────────────

script_dir   <- file.path(project_root, "examples", "soc_stock_0_5cm")
worker_script <- file.path(script_dir, "05_predict_spatial.R")
merge_script  <- file.path(script_dir, "05b_merge_spatial_parts.R")

log_dir <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs",
                     format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found at: ", rscript_bin)

# ── Launch workers ───────────────────────────────────────────────────────────────

message(sprintf("Launching %d worker(s)...", n_workers))
message("Per-worker logs: ", log_dir)

procs <- vector("list", n_workers)
for (w in seq_len(n_workers)) {
  log_file <- file.path(log_dir, sprintf("worker_%02dof%02d.log", w, n_workers))
  procs[[w]] <- processx::process$new(
    rscript_bin,
    args   = c(worker_script, as.character(w), as.character(n_workers)),
    stdout = log_file,
    stderr = log_file,
    cleanup = TRUE
  )
  message(sprintf("  Worker %2d/%2d started (PID %d) -> %s",
                  w, n_workers, procs[[w]]$get_pid(), basename(log_file)))
}

# ── Wait for completion, polling so progress can be observed ───────────────────

message("\nWaiting for all workers to finish. Tail any worker_*.log file in:")
message("  ", log_dir)
message("to watch its progress (read/scale/predict timing, RSS, valid pixels per block).\n")

t0 <- Sys.time()
repeat {
  alive <- vapply(procs, function(p) p$is_alive(), logical(1))
  if (!any(alive)) break
  Sys.sleep(30)
  el <- Sys.time() - t0
  message(sprintf("  [%.1f %s elapsed] %d/%d worker(s) still running...",
                  el, units(el), sum(alive), n_workers))
}

exit_codes <- vapply(procs, function(p) p$get_exit_status(), integer(1))
message("\nWorker exit codes: ", paste(exit_codes, collapse = ", "),
        " (0 = success)")

if (any(exit_codes != 0L, na.rm = TRUE)) {
  failed <- which(exit_codes != 0L)
  stop("Worker(s) ", paste(failed, collapse = ", "), " failed (non-zero exit code). ",
       "Check their logs in: ", log_dir, " before re-running.")
}

message("\nAll workers finished successfully (", format(Sys.time() - t0, units = "auto"), ").")

# ── Merge ─────────────────────────────────────────────────────────────────────

message("\nRunning merge step (05b_merge_spatial_parts.R)...\n")
merge_log <- file.path(log_dir, "merge.log")
merge_result <- processx::run(rscript_bin, args = merge_script,
                              stdout = "|", stderr = "|", echo = TRUE,
                              error_on_status = FALSE)
writeLines(c(merge_result$stdout, merge_result$stderr), merge_log)

if (merge_result$status != 0L) {
  stop("Merge step failed (exit code ", merge_result$status, "). See: ", merge_log)
}

message("\n── All done ──────────────────────────────────────────────────────────")
message("  Worker logs: ", log_dir)
message("  Merge log:   ", merge_log)
