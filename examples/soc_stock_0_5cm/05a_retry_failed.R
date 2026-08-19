project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a_retry_failed — Re-executa apenas os shards que falharam
#
# Use este script quando 05a_run_parallel.R terminar com shards com exit != 0.
# Configure retry_shards abaixo com os IDs dos shards que falharam.
# Ao final chama 05b_merge_spatial_parts.R para mosaico completo (inclui tiles
# dos shards que ja passaram + os reprocessados agora).
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuração ──────────────────────────────────────────────────────────────

n_shards       <- 1000   # deve ser o mesmo valor usado no 05a original
max_concurrent <- 2
poll_interval_s <- 30

# Shards a reprocessar (IDs que falharam na execução anterior):
retry_shards <- 27:1000

# ── Paths ─────────────────────────────────────────────────────────────────────

script_dir    <- file.path(project_root, "examples", "soc_stock_0_5cm")
worker_script <- file.path(script_dir, "05_predict_spatial.R")
merge_script  <- file.path(script_dir, "05b_merge_spatial_parts.R")

log_dir <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs",
                     format(Sys.time(), "%Y%m%d_%H%M%S"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found at: ", rscript_bin)

# ── Fila de retry ──────────────────────────────────────────────────────────────

message(sprintf("Retry: %d shard(s) para reprocessar, %d por vez...",
                length(retry_shards), max_concurrent))
message("Logs: ", log_dir, "\n")

pending    <- retry_shards
active     <- list()
exit_codes <- integer(0)   # named: shard_id -> exit code

launch_log <- function(shard_id) {
  log_file <- file.path(log_dir, sprintf("shard_%02dof%02d.log", shard_id, n_shards))
  p <- processx::process$new(
    rscript_bin,
    args   = c(worker_script, as.character(shard_id), as.character(n_shards)),
    stdout = log_file,
    stderr = log_file,
    cleanup = TRUE
  )
  message(sprintf("  Shard %4d/%d started (PID %d) -> %s",
                  shard_id, n_shards, p$get_pid(), basename(log_file)))
  p
}

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
      sid  <- as.integer(sid_chr)
      code <- p$get_exit_status()
      exit_codes[sid_chr] <- code
      n_done <- n_done + 1L
      status <- if (code == 0L) "OK" else paste0("FALHOU (exit ", code, ")")
      message(sprintf("  Shard %4d/%d finished: %s", sid, n_shards, status))
      finished_ids <- c(finished_ids, sid_chr)
    }
  }
  active[finished_ids] <- NULL

  while (length(active) < max_concurrent && length(pending) > 0) {
    sid <- pending[1]; pending <- pending[-1]
    active[[as.character(sid)]] <- launch_log(sid)
  }

  el <- Sys.time() - t0
  message(sprintf("  [%.1f %s elapsed] %d/%d retry shards done, %d rodando, %d na fila",
                  as.numeric(el), units(el),
                  n_done, length(retry_shards), length(active), length(pending)))
}

el_total <- Sys.time() - t0
message(sprintf("\nRetry concluido em %.1f %s.", as.numeric(el_total), units(el_total)))

failed_now <- as.integer(names(exit_codes)[exit_codes != 0L])
if (length(failed_now) > 0) {
  stop("Shard(s) ainda com falha: ", paste(failed_now, collapse = ", "),
       "\nLogs em: ", log_dir)
}

message("Todos os shards de retry passaram (exit 0).")

# ── Merge completo (inclui tiles dos shards anteriores que passaram) ──────────

message("\nRodando merge (05b_merge_spatial_parts.R)...\n")
merge_log    <- file.path(log_dir, "merge.log")
merge_result <- processx::run(rscript_bin, args = merge_script,
                              stdout = "|", stderr = "|", echo = TRUE,
                              error_on_status = FALSE)
writeLines(c(merge_result$stdout, merge_result$stderr), merge_log)

if (merge_result$status != 0L) {
  stop("Merge falhou (exit ", merge_result$status, "). Ver: ", merge_log)
}

message("\n── Tudo pronto ───────────────────────────────────────────────────────")
message("  Logs deste retry: ", log_dir)
message("  Merge log:        ", merge_log)
