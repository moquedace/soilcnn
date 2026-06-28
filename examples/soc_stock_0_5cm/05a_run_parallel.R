source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a — Orchestrate parallel spatial prediction (sharded job queue)
#
# Single entry point: set n_shards / max_concurrent below and run this script
# ONCE. It splits the raster into n_shards row-partitions (each one a
# `Rscript 05_predict_spatial.R <shard_id> <n_shards>` call) and runs at most
# max_concurrent of them at a time, queueing the rest. As soon as one finishes
# its slot is handed to the next pending shard. Once every shard is done it
# automatically runs 05b_merge_spatial_parts.R to mosaic the results.
#
# WHY SHARDED (not just "n_workers long-lived processes"):
# Windows R sessions accumulate memory fragmentation over a long run that
# gc() does NOT return to the OS — RSS creeps upward over hours even when the
# actual per-block working set is constant. Three long-lived workers (each
# covering ~1/3 of the raster, ~9+ hours) hit 100% system RAM well after
# starting clean, even though the same settings looked safe in the first
# couple of hours. Many SHORT-LIVED shards sidestep this: each process exits
# (full OS memory reclaim, guaranteed) long before fragmentation becomes
# dangerous, and the next shard in that slot starts from a clean process.
#
# RAM is still bounded by max_concurrent (concurrently running shards), NOT by
# n_shards — splitting into more, smaller shards does not increase peak RAM,
# it only shortens each process's lifetime (less time to fragment) and adds a
# bit of process-startup overhead (model loading etc., a few seconds each).
# ══════════════════════════════════════════════════════════════════════════════

project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

# ── Settings ────────────────────────────────────────────────────────────────────

# Total row-partitions. Smaller shards = shorter-lived processes = less time
# for memory fragmentation to build up before a clean exit, at the cost of a
# few seconds of repeated startup overhead (package load, model load) per shard.
#
# REVISED (again): n_shards=30 still let individual shards run for DAYS when
# their row range landed on a sustained-dense band (observed: ~2000-2800
# s/block for 213+ CONSECUTIVE blocks in one shard -- likely an equatorial
# band crossing several continents at once). Block count per shard is purely
# row-range-driven (density-independent), so worst-case wall-clock time is
# (blocks_per_shard x worst_observed_rate) regardless of n_shards -- and at
# n_shards=30 that worst case is ~25 days, which is exactly what blew up RAM
# again (2 such multi-day shards fragmenting simultaneously exhausted the
# whole system, breaking even unrelated file/network I/O).
#
# n_shards=1000 bounds blocks/shard to ~32, so even a shard landing entirely
# in the densest observed band caps out around ~25 h (~1 day) of process
# lifetime -- a large improvement on the 5+ days that caused the last
# failure. Overhead cost: ~500 sequential launches (1000 / max_concurrent=2)
# x ~30s setup (packages + 5 models + raster metadata) =~ 4.2 h total, small
# relative to the job's multi-week timeline.
n_shards <- 1000

# How many shards run AT THE SAME TIME. This — not n_shards — is what bounds
# peak RAM (same per-process floor as before: this is a GLOBAL raster, so each
# concurrent process pays for the full row width regardless of how few rows it
# owns). Keep this at the value already tuned against max_strip_ram_gb in
# 05_predict_spatial.R (currently max_strip_ram_gb=5, tuned for 3 concurrent
# processes on 64 GB RAM).
# REVISED: 3 concurrent shards at max_strip_ram_gb=5 OOM'd ("cannot allocate
# vector of size 4.5 Gb") after only 20 blocks (minutes, not hours) -- so this
# was NOT primarily long-run fragmentation, individual per-process peaks
# (observed up to ~14 GB on a single shard) simply add up past 64 GB when 3
# happen to spike together. Dropped to 2 concurrent + max_strip_ram_gb=4 in
# 05_predict_spatial.R for more headroom (~2 x 13 GB ≈ 26 GB estimated, vs the
# ~63 GB ceiling) at the cost of less parallelism (CPU was never the
# bottleneck here, so this mainly costs wall-clock time, not correctness).
max_concurrent <- 2

poll_interval_s <- 30

# ── Paths ─────────────────────────────────────────────────────────────────────

script_dir    <- file.path(project_root, "examples", "soc_stock_0_5cm")
worker_script <- file.path(script_dir, "05_predict_spatial.R")
merge_script  <- file.path(script_dir, "05b_merge_spatial_parts.R")

log_dir <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs",
                     format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found at: ", rscript_bin)

# ── Launch a bounded queue of shards ───────────────────────────────────────────

message(sprintf("Running %d shard(s), %d at a time...", n_shards, max_concurrent))
message("Per-shard logs: ", log_dir, "\n")

pending    <- seq_len(n_shards)
active     <- list()            # named list: shard_id (as character) -> process
exit_codes <- integer(n_shards)
launch_log <- function(shard_id) {
  log_file <- file.path(log_dir, sprintf("shard_%02dof%02d.log", shard_id, n_shards))
  p <- processx::process$new(
    rscript_bin,
    args   = c(worker_script, as.character(shard_id), as.character(n_shards)),
    stdout = log_file,
    stderr = log_file,
    cleanup = TRUE
  )
  message(sprintf("  Shard %2d/%2d started (PID %d) -> %s",
                  shard_id, n_shards, p$get_pid(), basename(log_file)))
  p
}

# Prime the queue
while (length(active) < max_concurrent && length(pending) > 0) {
  sid <- pending[1]; pending <- pending[-1]
  active[[as.character(sid)]] <- launch_log(sid)
}

t0 <- Sys.time()
n_done <- 0L
while (length(active) > 0 || length(pending) > 0) {
  Sys.sleep(poll_interval_s)

  finished_ids <- character(0)
  for (sid_chr in names(active)) {
    p <- active[[sid_chr]]
    if (!p$is_alive()) {
      sid <- as.integer(sid_chr)
      exit_codes[sid] <- p$get_exit_status()
      n_done <- n_done + 1L
      status <- if (exit_codes[sid] == 0L) "OK" else paste0("FALHOU (exit ", exit_codes[sid], ")")
      message(sprintf("  Shard %2d/%2d finished: %s", sid, n_shards, status))
      finished_ids <- c(finished_ids, sid_chr)
    }
  }
  active[finished_ids] <- NULL

  while (length(active) < max_concurrent && length(pending) > 0) {
    sid <- pending[1]; pending <- pending[-1]
    active[[as.character(sid)]] <- launch_log(sid)
  }

  el <- Sys.time() - t0
  message(sprintf("  [%.1f %s elapsed] %d/%d done, %d running, %d queued",
                  el, units(el), n_done, n_shards, length(active), length(pending)))
}

message("\nAll shards finished (", format(Sys.time() - t0, units = "auto"), ").")
message("Exit codes: ", paste(exit_codes, collapse = ", "), " (0 = success)")

if (any(exit_codes != 0L)) {
  failed <- which(exit_codes != 0L)
  stop("Shard(s) ", paste(failed, collapse = ", "), " failed (non-zero exit code). ",
       "Check their logs in: ", log_dir, " before re-running.")
}

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
message("  Shard logs: ", log_dir)
message("  Merge log:  ", merge_log)
