project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

# ══════════════════════════════════════════════════════════════════════════════
# 05c -- ETA for the 2D job (05a_run_parallel.R)
#
# Re-runnable at any time while 05a is running. Automatically detects the most
# recent run with logs in the 2D format
# (shard_rXXXofYYY_cXXXofZZZ.log). Does not interfere with 05c.
# ══════════════════════════════════════════════════════════════════════════════

max_concurrent <- 3   # <<< keep identical to 05a_run_parallel.R

log_root <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs")
# THE RUN WITH THE NEWEST SHARD LOG, which is the one being written right now.
# The old loop took the newest directory by NAME and then looked for logs in
# it; a run in progress is the one whose log was touched last, and that is the
# question an ETA script is asking.
source(file.path(project_root, "R", "utils.R"))
run_dir <- file.path(log_root, latest_run_dir(
  log_root, prefix = "",
  require_pattern = "^shard_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+[.]log$",
  label = "2D run"))

now <- Sys.time()
message("2D run: ", basename(run_dir), "  |  now: ", format(now))

log_files <- list.files(run_dir,
  pattern = "^shard_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.log$",
  full.names = TRUE)
log_files <- sort(log_files)

# Extracts n_row_shards and n_col_shards from the first filename
name_pat    <- "shard_r([0-9]+)of([0-9]+)_c([0-9]+)of([0-9]+)\\.log$"
first_name  <- basename(log_files[1])
n_row_shards <- as.integer(sub(paste0(".*", name_pat), "\\2", first_name))
n_col_shards <- as.integer(sub(paste0(".*", name_pat), "\\4", first_name))
n_shards_total <- n_row_shards * n_col_shards

block_pattern <- "Block ([0-9]+)/([0-9]+)"
error_pattern <- "^Erro|^Error|Execu..o interrompida"

parse_shard_log <- function(f) {
  bn      <- basename(f)
  row_id  <- as.integer(sub(paste0(".*", name_pat), "\\1", bn))
  col_id  <- as.integer(sub(paste0(".*", name_pat), "\\3", bn))
  finfo   <- file.info(f)
  started <- finfo$ctime
  lines   <- tryCatch(readLines(f, warn = FALSE), error = function(e) character(0))

  has_error <- any(grepl(error_pattern, lines))
  finished  <- any(grepl("Spatial prediction complete", lines))

  block_lines <- lines[grepl(block_pattern, lines)]
  done <- 0L; total <- NA_integer_
  if (length(block_lines) > 0) {
    m     <- regmatches(block_lines[length(block_lines)],
                        regexec(block_pattern, block_lines[length(block_lines)]))[[1]]
    done  <- as.integer(m[2])
    total <- as.integer(m[3])
  }

  # For shards already complete/ERROR, use the log's mtime (last write, when the
  # shard actually finished) instead of "now" -- otherwise the elapsed grows with
  # the idle time after it ended (the script may be run hours later), which
  # artificially inflates s/block and the ETA. Only a "running" shard uses "now".
  end_time <- if (finished || has_error) finfo$mtime else now

  list(row_id = row_id, col_id = col_id,
       done = done, total = total,
       elapsed_s = as.numeric(end_time - started, units = "secs"),
       has_error = has_error, finished = finished)
}

shards <- lapply(log_files, parse_shard_log)

message(sprintf("\n%d/%d shard(s) started  (%d x %d grid)\n",
                length(shards), n_shards_total, n_row_shards, n_col_shards))
message(sprintf("%-16s %8s %8s %8s %10s %10s",
                "shard [r/c]", "done", "total", "pct", "s/block", "status"))
message(strrep("-", 66))

rate_num <- 0; rate_den <- 0L
total_known <- integer(0)
any_error   <- FALSE

for (s in shards) {
  status <- if (s$has_error) "ERROR" else if (s$finished) "complete" else "running"
  if (s$has_error) any_error <- TRUE
  if (!is.na(s$total)) total_known <- c(total_known, s$total)

  rate_here <- if (s$done > 0) s$elapsed_s / s$done else NA_real_
  if (!s$has_error && s$done > 0) {
    rate_num <- rate_num + s$elapsed_s
    rate_den <- rate_den + s$done
  }

  pct <- if (!is.na(s$total) && s$total > 0)
    sprintf("%.1f%%", 100 * s$done / s$total) else "-"

  message(sprintf("[r%03d/c%03d]      %8d %8s %8s %10s %10s",
                  s$row_id, s$col_id,
                  s$done,
                  ifelse(is.na(s$total), "?", s$total),
                  pct,
                  ifelse(is.na(rate_here), "-", sprintf("%.1f", rate_here)),
                  status))
}
message(strrep("-", 66))

if (any_error) {
  message("\n[WARNING] At least one shard ended with an error -- the estimate ignores",
          " that shard. Check the ERROR logs.")
}

n_not_started <- n_shards_total - length(shards)
done_total    <- sum(vapply(shards, function(s) s$done, integer(1)))
avg_total_per_shard <- if (length(total_known) > 0) mean(total_known) else NA_real_

if (is.na(avg_total_per_shard) || rate_den == 0L) {
  message("\nStill not enough data. Run it again in a few minutes.")
} else {
  est_total_all  <- avg_total_per_shard * n_shards_total
  remaining_blks <- est_total_all - done_total
  avg_rate       <- rate_num / rate_den

  remaining_wall_s <- (remaining_blks * avg_rate) / max_concurrent
  eta              <- now + remaining_wall_s
  remaining_dt     <- eta - now

  message(sprintf("\nEstimated total blocks (%d shards): %s",
                  n_shards_total, format(round(est_total_all), big.mark = ",")))
  message(sprintf("Blocks completed: %s (%.1f%%)",
                  format(done_total, big.mark = ","),
                  100 * done_total / est_total_all))
  message(sprintf("Average rate: %.1f s/block", avg_rate))
  message(sprintf("Shards queued (not started): %d", n_not_started))
  message(sprintf("\nETA: %s  (~%.1f %s remaining)",
                  format(eta), as.numeric(remaining_dt), units(remaining_dt)))
}
