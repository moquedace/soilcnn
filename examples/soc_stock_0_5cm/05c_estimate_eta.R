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

# ══════════════════════════════════════════════════════════════════════════════
# 05c -- ETA for the 2D job (05a_run_parallel.R)
#
# Re-runnable at any time while 05a is running. Automatically detects the most
# recent run with logs in the 2D format
# (shard_rXXXofYYY_cXXXofZZZ.log). Does not interfere with 05c.
# ══════════════════════════════════════════════════════════════════════════════


log_root <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs")
# THE RUN WITH THE NEWEST SHARD LOG, which is the one being written right now.
# The old loop took the newest directory by NAME and then looked for logs in
# it; a run in progress is the one whose log was touched last, and that is the
# question an ETA script is asking.
source(file.path(project_root, "R", "utils.R"))

# The same override 05a and 05 read, so the ETA divides by the concurrency
# the run actually has. This was a literal 3 with a comment asking the reader
# to keep it in step with 05a by hand.
max_concurrent <- env_int("soc_max_concurrent", 3L)

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
  # An unreadable log used to become an empty one, i.e. a shard "not yet
  # started" -- the ETA then counted it as pending work.
  lines   <- tryCatch(readLines(f, warn = FALSE), error = function(e) {
    stop("Cannot read worker log ", f, " (", conditionMessage(e), ")", call. = FALSE)
  })

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
