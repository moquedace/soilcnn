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

# env_int() and latest_run_dir() live in R/utils.R. Loaded here, first, and
# nothing more of the framework: this is an orchestrator, not a model.
source(file.path(project_root, "R", "utils.R"))

source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a — Spatial prediction orchestrator with 2D tiling
#
# Splits the raster into n_row_shards x n_col_shards rectangles. Each shard is
# an independent process that reads only its tile's column range (+ a half_w_max
# margin on each side), cutting the RAM floor proportionally.
#
# Example with a global raster of 160 k columns, 187 bands:
#   1D tiling (n_col_shards=1, the previous scheme, retired): ~240 MB/row ->
#     floor ~7-8 GB -> max_concurrent=2
#   2D tiling (n_col_shards=4, the current scheme): ~60 MB/row -> floor ~2 GB ->
#     max_concurrent=8
#
# n_col_shards must be chosen so that:
#   ceil(r_ncol / n_col_shards) >> 2 * half_w_max  (tile >> margin)
# For half_w_max = 16 (window 33), n_col_shards <= 160 k / (10 * 32) = 500 is
# reasonable (e.g. 4-8 already reduce enough without excessive overhead).
#
# Output tiles go to raster/parts_2d/ with the suffix _rXXXofYYY_cXXXofYYY.
# At the end it runs 05b_merge_spatial_parts.R to assemble the final maps.
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuration ──────────────────────────────────────────────────────────────

# Row shards. More shards = each process lives
# less time = less RAM fragmentation.
n_row_shards <- 250

# ── OVERRIDABLE, and the reason is not convenience ────────────────────────────
#
# These three were literals with no override, so running at any other shard grid
# meant editing this file -- and the only person who ever wants another grid is
# someone testing the sharding itself, who then has to remember to edit it back.
# A test that requires a source edit is a test that runs once.
#
# The idiom is 05_predict_spatial.R:60-72's: an environment variable wins when
# it is set, the literal stands otherwise, so nothing changes for a normal run.
#
# soc_n_row_shards / soc_n_col_shards / soc_max_concurrent

# Column shards: each shard reduces strip_ncol by 1/n_col_shards -> RAM
# per process proportional. n_col_shards=4 splits ~240 MB/row into ~60 MB.
n_col_shards <- 4

# Total shards = n_row_shards * n_col_shards
# With 250 x 4 = 1000 shards (same total number as 05a, but a 2D layout)

# Concurrent processes.
# Root cause of the high RSS identified and fixed: it was not output_block_rows,
# it was batch_size in build_patches_multi (the window validity check indexed
# strip_values for the whole chunk, producing a transient peak of
# ~batch_size * n_pos * n_channels * 8 bytes, independent of the block
# size). Cutting batch_size 4096 -> 512 and removing the duplicated reindexing
# in build_patches_multi, the retest (05a_test.R, row 1, 4 col-shards) gave:
#   sparse shards (ocean): RSS ~2.2 GB, ~2.7 min/shard
#   dense shard (the same region that used to hit 37 GB): RSS peak 13.2 GB,
#     stable across the 16 blocks (no more progressive degradation),
#     34.5 min/shard (89 min before, for the same shard)
# Machine: 32 cores, ~64 GB RAM.
#
# UPDATE (after a full smoke test at 1 km, the same strip width of ~40 thousand
# columns per block as the 4 col-shards config here): the RSS peak actually
# measured came out at 13.9-14.9 GB/shard (a little above the 13.2 GB reference
# above -- that reference is from before the final model had 10 seeds; more
# seeds does not change the RSS peak much -- what dominates the peak is the
# window validity check buffer, sized by batch_size, not by n_seeds -- but the
# extra margin is worth it anyway). Besides that, this machine does NOT have a
# GPU accessible to torch (cuda_is_available()==FALSE, cuda_device_count()==0,
# confirmed in the logs of the 1 km test -- "Device: cpu") -- the work is 100%
# CPU-bound, so raising max_concurrent beyond what is needed to fit in RAM does
# NOT speed up the total (the machine's total CPU is fixed); it only cuts the
# safety margin. Hence:
# -> max_concurrent=3: worst case 3 x 14.9 GB ~= 45 GB / 64 GB (margin ~19 GB).
#    threads_per_worker = 32/3 = 10.
# -> Re-evaluate via 05c_estimate_eta.R as the first real shards finish.
max_concurrent <- 3


n_row_shards   <- env_int("soc_n_row_shards",   n_row_shards)
n_col_shards   <- env_int("soc_n_col_shards",   n_col_shards)
max_concurrent <- env_int("soc_max_concurrent", max_concurrent)
poll_interval_s <- 30

# ── Paths ──────────────────────────────────────────────────────────────────────

script_dir    <- file.path(project_root, "examples", "soc_stock_0_5cm")
worker_script <- file.path(script_dir, "05_predict_spatial.R")
merge_script  <- file.path(script_dir, "05b_merge_spatial_parts.R")

log_dir <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs",
                     format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found: ", rscript_bin)

# ── Resolve config_id/target_label (same logic as 05_predict_spatial.R) ───────
# Needed here only to know where the already generated tiles are (resume).

target_label <- "soc_stock_0_5cm"
final_run_id <- "latest"

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (identical(final_run_id, "latest")) {
  # By time, and only a run that FINISHED: stage 04 creates its directory
  # before its own validations, so a failed 04 leaves a final_<timestamp> that
  # the old name-sort would have handed to every worker.
  final_run_id <- latest_run_dir(
    final_model_base, prefix = "final_",
    require_file = file.path("comparison", "final_run_summary.rds"),
    label = "final_run_id")
}
final_run_dir <- file.path(final_model_base, final_run_id)
summary_file  <- file.path(final_run_dir, "comparison", "final_run_summary.rds")
if (!file.exists(summary_file)) stop("final_run_summary.rds not found: ", summary_file)
config_id <- selected_config_id(readRDS(summary_file), final_run_id)

output_log_dir <- file.path(project_root, "outputs", "spatial_prediction",
                            "soc_stock_modeling", target_label, config_id, "log")

message(sprintf("config_id: %s | final_run_id: %s | logs in: %s",
                config_id, final_run_id, output_log_dir))

# ── Build the list of all shards (row x col cartesian product) ─────────────────

shards <- expand.grid(row_shard = seq_len(n_row_shards),
                       col_shard = seq_len(n_col_shards))
# Sorts by row first (iterates row by row) for a progressive mosaic
shards <- shards[order(shards$row_shard, shards$col_shard), ]
n_shards_total <- nrow(shards)

message(sprintf("2D tiling: %d x %d = %d shards, %d at a time",
                n_row_shards, n_col_shards, n_shards_total, max_concurrent))
message("Logs: ", log_dir, "\n")

# ── Resume: skips shards already finished successfully ────────────────────────
# The .tif tiles are created (writeStart) with the correct dimensions already
# at the START of block 1 -- if the process dies halfway (e.g. block 11/16),
# the file can still open without an error in terra, only with blocks missing.
# Checking only the existence of the .tif would give a false positive (an
# incomplete tile accepted as finished -> a silent hole in the final mosaic).
#
# Reliable signal: prediction_config_r###of###_c###of###.csv in cfg_022/log/
# is only written AFTER the writeStop() of every raster and after the sanity
# checks pass (see 05_predict_spatial.R). If the process dies before that,
# that CSV never exists -- so its existence implies a shard that is
# 100% finished and validated.

shard_config_file <- function(rs, cs) {
  suf <- sprintf("_r%03dof%03d_c%03dof%03d", rs, n_row_shards, cs, n_col_shards)
  file.path(output_log_dir, paste0("prediction_config", suf, ".csv"))
}

# A MARKER WITHOUT ITS TILES IS NOT "DONE". The config CSV is written by 05
# after the tiles; if the tiles were since deleted (a parts_2d/ cleared by
# hand, a 05b that was told to remove them) the marker alone would make this
# script skip the shard and 05b then fail on a tile count. Refused here, where
# the fix is one sentence.
parts_dir <- file.path(dirname(output_log_dir), "raster", "parts_2d")
shard_already_done <- function(rs, cs) {
  marker <- file.exists(shard_config_file(rs, cs))
  if (!marker) return(FALSE)
  suf <- sprintf("_r%03dof%03d_c%03dof%03d[.]tif$", rs, n_row_shards, cs, n_col_shards)
  tiles <- list.files(parts_dir, pattern = suf)
  if (length(tiles) == 0L) {
    stop("Shard [r", rs, "/c", cs, "] has its prediction_config marker but no ",
         "tile under ", parts_dir, ".\n  Delete ", basename(shard_config_file(rs, cs)),
         " to have the shard re-run, or restore the tiles.", call. = FALSE)
  }
  TRUE
}

message("Checking for shards already finished (resume)...")
done_mask <- vapply(seq_len(n_shards_total), function(idx)
  shard_already_done(shards$row_shard[idx], shards$col_shard[idx]), logical(1))
n_already_done <- sum(done_mask)
if (n_already_done > 0L) {
  message(sprintf("  %d/%d shards already finished -- skipping (resume).",
                  n_already_done, n_shards_total))
}

# ── Queue ──────────────────────────────────────────────────────────────────────

pending    <- which(!done_mask)          # index into the rows of `shards`
active     <- list()                     # idx -> process
exit_codes <- integer(n_shards_total)
exit_codes[done_mask] <- 0L

launch_shard <- function(idx) {
  rs <- shards$row_shard[idx]
  cs <- shards$col_shard[idx]
  log_file <- file.path(log_dir,
    sprintf("shard_r%03dof%03d_c%03dof%03d.log", rs, n_row_shards, cs, n_col_shards))
  # THE ENVIRONMENT IS PASSED, NOT ASSUMED.
  #
  # 05_predict_spatial.R:115-118 reads soc_predict_raster_dir to decide WHICH
  # rasters to predict over. Spawning without `env` left that to processx
  # inheriting the parent environment -- true today, undocumented, and silent
  # when it is not: the worker falls back to raster_table_used.csv, which is the
  # 250 m grid, and 05b later merges 250 m tiles under 20 km filenames. Nothing
  # in either script would say so.
  #
  # c("current", ...) keeps the inherited environment and adds to it, so a
  # session that did not set the variable behaves exactly as before.
  p <- processx::process$new(
    rscript_bin,
    args = c(worker_script,
             as.character(rs), as.character(cs),
             as.character(n_row_shards), as.character(n_col_shards),
             as.character(max_concurrent)),
    env = if (nzchar(Sys.getenv("soc_predict_raster_dir"))) {
      c("current", soc_predict_raster_dir = Sys.getenv("soc_predict_raster_dir"))
    } else NULL,
    stdout  = log_file,
    stderr  = log_file,
    cleanup = TRUE
  )
  message(sprintf("  Shard [r%03d/c%03d] started (PID %d) -> %s",
                  rs, cs, p$get_pid(), basename(log_file)))
  p
}

while (length(active) < max_concurrent && length(pending) > 0) {
  idx <- pending[1]; pending <- pending[-1]
  active[[as.character(idx)]] <- launch_shard(idx)
}

t0 <- Sys.time()
n_done <- 0L

while (length(active) > 0 || length(pending) > 0) {
  Sys.sleep(poll_interval_s)

  finished_ids <- character(0)
  for (idx_chr in names(active)) {
    p <- active[[idx_chr]]
    if (!p$is_alive()) {
      idx  <- as.integer(idx_chr)
      rs   <- shards$row_shard[idx]
      cs   <- shards$col_shard[idx]
      exit_codes[idx] <- p$get_exit_status()
      n_done <- n_done + 1L
      status <- if (exit_codes[idx] == 0L) "OK"
                else paste0("FAILED (exit ", exit_codes[idx], ")")
      message(sprintf("  [r%03d/c%03d] finished: %s", rs, cs, status))
      finished_ids <- c(finished_ids, idx_chr)
    }
  }
  active[finished_ids] <- NULL

  while (length(active) < max_concurrent && length(pending) > 0) {
    idx <- pending[1]; pending <- pending[-1]
    active[[as.character(idx)]] <- launch_shard(idx)
  }

  el <- Sys.time() - t0
  message(sprintf("  [%.1f %s elapsed] %d/%d done (%d already done + %d this session), %d running, %d queued",
                  as.numeric(el), units(el),
                  n_already_done + n_done, n_shards_total, n_already_done, n_done,
                  length(active), length(pending)))
}

el_total <- Sys.time() - t0
message(sprintf("\nAll shards finished in %.1f %s.",
                as.numeric(el_total), units(el_total)))

failed <- which(exit_codes != 0L)
if (length(failed) > 0) {
  failed_info <- shards[failed, ]
  msg <- paste(sprintf("[r%d/c%d]", failed_info$row_shard, failed_info$col_shard),
               collapse = ", ")
  stop("Failed shards: ", msg, "\nLogs in: ", log_dir)
}

# ── Merge ──────────────────────────────────────────────────────────────────────

message("\nRunning merge (05b_merge_spatial_parts.R)...\n")
merge_log    <- file.path(log_dir, "merge.log")
merge_result <- processx::run(rscript_bin, args = merge_script,
                              stdout = "|", stderr = "|", echo = TRUE,
                              error_on_status = FALSE)
writeLines(c(merge_result$stdout, merge_result$stderr), merge_log)

if (merge_result$status != 0L) {
  stop("Merge failed (exit ", merge_result$status, "). See: ", merge_log)
}

message("\n── All done ──────────────────────────────────────────────────────")
message("  Logs: ", log_dir)
message("  Merge: ", merge_log)
