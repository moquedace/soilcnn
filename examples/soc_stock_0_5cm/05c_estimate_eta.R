project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

# ══════════════════════════════════════════════════════════════════════════════
# 05c — ETA do job 2D (05a_run_parallel.R)
#
# Re-rodável a qualquer momento enquanto o 05a estiver rodando. Detecta
# automaticamente o run mais recente com logs no formato 2D
# (shard_rXXXofYYY_cXXXofZZZ.log). Não interfere no 05c.
# ══════════════════════════════════════════════════════════════════════════════

max_concurrent <- 3   # <<< manter igual ao 05a_run_parallel.R

log_root <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs")
run_dirs <- list.dirs(log_root, recursive = FALSE, full.names = FALSE)
if (length(run_dirs) == 0) stop("Nenhum run encontrado em: ", log_root)

# Encontra o run mais recente que tenha logs 2D (ignora runs do 05a/TEST2D)
run_dir <- NULL
for (rd in sort(run_dirs, decreasing = TRUE)) {
  candidate <- file.path(log_root, rd)
  files_2d  <- list.files(candidate,
    pattern = "^shard_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.log$")
  if (length(files_2d) > 0) { run_dir <- candidate; break }
}
if (is.null(run_dir)) stop("Nenhum run 2D encontrado em: ", log_root)

now <- Sys.time()
message("Run 2D: ", basename(run_dir), "  |  now: ", format(now))

log_files <- list.files(run_dir,
  pattern = "^shard_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.log$",
  full.names = TRUE)
log_files <- sort(log_files)

# Extrai n_row_shards e n_col_shards do primeiro filename
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

  # Para shards ja completo/ERRO, usa o mtime do log (ultima escrita, quando o
  # shard de fato terminou) em vez de "now" -- senao o elapsed cresce com o
  # tempo ocioso apos o termino (o script pode ser rodado horas depois), o que
  # infla artificialmente s/bloco e o ETA. Só shard "rodando" usa "now".
  end_time <- if (finished || has_error) finfo$mtime else now

  list(row_id = row_id, col_id = col_id,
       done = done, total = total,
       elapsed_s = as.numeric(end_time - started, units = "secs"),
       has_error = has_error, finished = finished)
}

shards <- lapply(log_files, parse_shard_log)

message(sprintf("\n%d/%d shard(s) iniciados  (%d x %d grid)\n",
                length(shards), n_shards_total, n_row_shards, n_col_shards))
message(sprintf("%-16s %8s %8s %8s %10s %10s",
                "shard [r/c]", "done", "total", "pct", "s/bloco", "status"))
message(strrep("-", 66))

rate_num <- 0; rate_den <- 0L
total_known <- integer(0)
any_error   <- FALSE

for (s in shards) {
  status <- if (s$has_error) "ERRO" else if (s$finished) "completo" else "rodando"
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
  message("\n[ATENCAO] Pelo menos um shard terminou com erro -- estimativa ignora",
          " esse shard. Verifique os logs ERRO.")
}

n_not_started <- n_shards_total - length(shards)
done_total    <- sum(vapply(shards, function(s) s$done, integer(1)))
avg_total_per_shard <- if (length(total_known) > 0) mean(total_known) else NA_real_

if (is.na(avg_total_per_shard) || rate_den == 0L) {
  message("\nAinda sem dados suficientes. Rode de novo em alguns minutos.")
} else {
  est_total_all  <- avg_total_per_shard * n_shards_total
  remaining_blks <- est_total_all - done_total
  avg_rate       <- rate_num / rate_den

  remaining_wall_s <- (remaining_blks * avg_rate) / max_concurrent
  eta              <- now + remaining_wall_s
  remaining_dt     <- eta - now

  message(sprintf("\nBlocos totais estimados (%d shards): %s",
                  n_shards_total, format(round(est_total_all), big.mark = ",")))
  message(sprintf("Blocos concluidos: %s (%.1f%%)",
                  format(done_total, big.mark = ","),
                  100 * done_total / est_total_all))
  message(sprintf("Ritmo medio: %.1f s/bloco", avg_rate))
  message(sprintf("Shards na fila (nao iniciados): %d", n_not_started))
  message(sprintf("\nETA: %s  (faltam ~%.1f %s)",
                  format(eta), as.numeric(remaining_dt), units(remaining_dt)))
}
