project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

# ══════════════════════════════════════════════════════════════════════════════
# 05c — Estimate remaining time for the running sharded spatial prediction
#
# Re-runnable at any point while 05a_run_parallel.R's shard queue is running
# (or after it finishes). Unlike a fixed set of long-lived workers that all
# start together, shards start at DIFFERENT times (queued, max_concurrent at
# once), so each shard's own progress is measured against ITS OWN start time
# (the log file's creation time), not a single shared launch timestamp.
#
# The whole job is modelled as one pool of "block work" shared across
# max_concurrent concurrent slots: remaining wall-clock ≈ remaining blocks x
# average seconds-per-block / max_concurrent. Set max_concurrent below to
# match whatever you used in 05a_run_parallel.R.
# ══════════════════════════════════════════════════════════════════════════════

max_concurrent <- 2   # <<< keep this in sync with 05a_run_parallel.R

log_root <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs")
run_dirs <- list.dirs(log_root, recursive = FALSE, full.names = FALSE)
if (length(run_dirs) == 0) stop("No worker log runs found in: ", log_root)

latest_run <- sort(run_dirs, decreasing = TRUE)[1]
run_dir    <- file.path(log_root, latest_run)
now        <- Sys.time()

message("Run: ", latest_run, "  |  now: ", format(now))

log_files <- list.files(run_dir, pattern = "^shard_[0-9]+of[0-9]+\\.log$", full.names = TRUE)
if (length(log_files) == 0) stop("No shard_*.log files found in: ", run_dir,
                                 " (old worker_*.log naming? this script expects the sharded orchestrator)")
log_files <- sort(log_files)

block_pattern <- "Block ([0-9]+)/([0-9]+)"
error_pattern <- "^Erro|^Error|Execu..o interrompida"
name_pattern  <- "shard_([0-9]+)of([0-9]+)\\.log$"

n_shards <- as.integer(sub(paste0(".*", name_pattern), "\\2", log_files[1]))

parse_shard_log <- function(f) {
  shard_id  <- as.integer(sub(paste0(".*", name_pattern), "\\1", f))
  started   <- file.info(f)$ctime
  lines     <- tryCatch(readLines(f, warn = FALSE), error = function(e) character(0))
  has_error <- any(grepl(error_pattern, lines))
  finished  <- any(grepl("Spatial prediction complete", lines))

  block_lines <- lines[grepl(block_pattern, lines)]
  done  <- 0L
  total <- NA_integer_
  if (length(block_lines) > 0) {
    m <- regmatches(block_lines[length(block_lines)],
                    regexec(block_pattern, block_lines[length(block_lines)]))[[1]]
    done  <- as.integer(m[2])
    total <- as.integer(m[3])
  }
  elapsed_s <- as.numeric(now - started, units = "secs")
  list(shard_id = shard_id, done = done, total = total, elapsed_s = elapsed_s,
       has_error = has_error, finished = finished)
}

shards <- lapply(log_files, parse_shard_log)

message(sprintf("\n%d/%d shard(s) have started so far.\n", length(shards), n_shards))
message(sprintf("%-10s %8s %8s %8s %10s %10s", "shard", "done", "total", "pct", "s/block", "status"))
message(strrep("-", 60))

rate_num <- 0; rate_den <- 0L   # pooled: sum(elapsed) / sum(done) across started shards
total_known <- integer(0)
any_error <- FALSE

for (s in shards) {
  status <- if (s$has_error) "ERRO" else if (s$finished) "completo" else "rodando"
  if (s$has_error) any_error <- TRUE

  if (!is.na(s$total)) total_known <- c(total_known, s$total)

  rate_here <- if (s$done > 0) s$elapsed_s / s$done else NA_real_
  if (!s$has_error && s$done > 0) {
    rate_num <- rate_num + s$elapsed_s
    rate_den <- rate_den + s$done
  }

  pct <- if (!is.na(s$total) && s$total > 0) sprintf("%.1f%%", 100 * s$done / s$total) else "-"
  message(sprintf("%-10d %8d %8s %8s %10s %10s",
                  s$shard_id, s$done, ifelse(is.na(s$total), "?", s$total), pct,
                  ifelse(is.na(rate_here), "-", sprintf("%.1f", rate_here)), status))
}
message(strrep("-", 60))

if (any_error) {
  message("\n[ATENCAO] Pelo menos um shard terminou com erro -- a estimativa abaixo ",
          "ignora o trabalho desse shard. Verifique o(s) log(s) ERRO; o job nao estara ",
          "de fato completo so porque os outros terminaram.")
}

n_started      <- length(shards)
n_not_started  <- n_shards - n_started
done_total     <- sum(vapply(shards, function(s) s$done, integer(1)))

avg_total_per_shard <- if (length(total_known) > 0) mean(total_known) else NA_real_

if (is.na(avg_total_per_shard) || rate_den == 0L) {
  message("\nAinda nao ha dados suficientes (nenhum shard com blocos processados). Rode de novo em alguns minutos.")
} else {
  est_total_all_shards <- avg_total_per_shard * n_shards
  remaining_blocks      <- est_total_all_shards - done_total
  avg_rate_s_per_block  <- rate_num / rate_den

  remaining_wall_s <- (remaining_blocks * avg_rate_s_per_block) / max_concurrent
  eta              <- now + remaining_wall_s

  message(sprintf("\nBlocos totais estimados (todos os %d shards): %s", n_shards,
                  format(round(est_total_all_shards), big.mark = ",")))
  message(sprintf("Blocos concluidos até agora: %s (%.1f%%)",
                  format(done_total, big.mark = ","), 100 * done_total / est_total_all_shards))
  message(sprintf("Ritmo medio observado: %.1f s/bloco (por slot concorrente)", avg_rate_s_per_block))
  message(sprintf("Shards ainda na fila (nao iniciados): %d", n_not_started))
  message(sprintf("\nETA estimado do job inteiro: %s  (faltam ~%.1f horas)",
                  format(eta), remaining_wall_s / 3600))
}
