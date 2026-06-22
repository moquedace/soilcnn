project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

# ══════════════════════════════════════════════════════════════════════════════
# 05c — Estimate remaining time for the running parallel spatial prediction
#
# Re-runnable at any point while 05a_run_parallel.R's workers are running (or
# after they finish). Reads the latest worker_logs/<timestamp>/ run, takes the
# last "Block X/Y" line logged by each worker, and combines it with elapsed
# wall-clock time since launch (the timestamp folder name) to estimate a
# per-worker completion time. The job as a whole finishes only when the
# SLOWEST worker finishes, so the headline ETA is the max across workers —
# not the average.
#
# Accuracy improves the longer you wait between runs: very early in a worker's
# row range the throughput estimate can be biased by an atypical land/ocean
# density in just-processed blocks (seen in the single-process diagnostic
# run earlier). Treat early estimates (few blocks done) as rough.
# ══════════════════════════════════════════════════════════════════════════════

log_root <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs")
run_dirs <- list.dirs(log_root, recursive = FALSE, full.names = FALSE)
if (length(run_dirs) == 0) stop("No worker log runs found in: ", log_root)

latest_run <- sort(run_dirs, decreasing = TRUE)[1]
run_dir    <- file.path(log_root, latest_run)

start_time    <- as.POSIXct(latest_run, format = "%Y%m%d_%H%M%S")
now           <- Sys.time()
elapsed_total <- as.numeric(now - start_time, units = "secs")

message("Run: ", latest_run, "  |  started: ", format(start_time),
        "  |  elapsed: ", format(now - start_time, digits = 3))

log_files <- list.files(run_dir, pattern = "^worker_\\d+of\\d+\\.log$", full.names = TRUE)
if (length(log_files) == 0) stop("No worker_*.log files found in: ", run_dir)
log_files <- sort(log_files)

block_pattern <- "Block ([0-9]+)/([0-9]+)"
error_pattern <- "^Erro|^Error|Execu..o interrompida"

parse_worker_log <- function(f) {
  lines <- tryCatch(readLines(f, warn = FALSE), error = function(e) character(0))
  has_error <- any(grepl(error_pattern, lines))
  finished  <- any(grepl("Spatial prediction complete", lines))

  block_lines <- lines[grepl(block_pattern, lines)]
  if (length(block_lines) == 0) {
    return(list(file = basename(f), done = 0L, total = NA_integer_,
               has_error = has_error, finished = finished))
  }
  m <- regmatches(block_lines[length(block_lines)],
                  regexec(block_pattern, block_lines[length(block_lines)]))[[1]]
  list(file = basename(f), done = as.integer(m[2]), total = as.integer(m[3]),
       has_error = has_error, finished = finished)
}

results <- lapply(log_files, parse_worker_log)

message("\n", sprintf("%-22s %8s %8s %8s %10s %18s", "worker", "done", "total", "pct", "s/block", "ETA"))
message(strrep("-", 90))

eta_candidates <- c()
any_error <- FALSE

for (r in results) {
  if (r$has_error) {
    any_error <- TRUE
    message(sprintf("%-22s %8s %8s %8s %10s %18s", r$file, r$done,
                    ifelse(is.na(r$total), "?", r$total), "ERRO", "-", "FALHOU - ver log"))
    next
  }
  if (r$finished || (!is.na(r$total) && r$done >= r$total)) {
    message(sprintf("%-22s %8d %8d %8s %10s %18s", r$file, r$done,
                    ifelse(is.na(r$total), r$done, r$total), "100%", "-", "concluido"))
    eta_candidates <- c(eta_candidates, as.numeric(now))
    next
  }
  if (r$done == 0L || is.na(r$total)) {
    message(sprintf("%-22s %8d %8s %8s %10s %18s", r$file, r$done, "?", "-", "-", "sem blocos ainda"))
    next
  }

  rate_s     <- elapsed_total / r$done
  remaining  <- rate_s * (r$total - r$done)
  eta_time   <- now + remaining
  pct        <- 100 * r$done / r$total

  flag <- if (r$done < 10L) " (amostra pequena)" else ""

  message(sprintf("%-22s %8d %8d %7.1f%% %10.1f %18s%s",
                  r$file, r$done, r$total, pct, rate_s,
                  format(eta_time, "%Y-%m-%d %H:%M"), flag))

  eta_candidates <- c(eta_candidates, as.numeric(eta_time))
}

message(strrep("-", 90))

if (any_error) {
  message("\n[ATENCAO] Pelo menos um worker terminou com erro -- a estimativa de ETA ",
          "abaixo IGNORA esse worker e nao sera o tempo real de conclusao do job ",
          "completo. Verifique o(s) log(s) marcados ERRO antes de confiar no merge final.")
}

if (length(eta_candidates) > 0) {
  overall_eta <- as.POSIXct(max(eta_candidates), origin = "1970-01-01", tz = Sys.timezone())
  message(sprintf("\nETA do job inteiro (worker mais lento): %s  (faltam ~%s)",
                  format(overall_eta), format(overall_eta - now, digits = 3)))
} else {
  message("\nNenhum worker com blocos suficientes para estimar ainda. Rode de novo em alguns minutos.")
}
