project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(file.path(project_root, "utils", "install_load_pkg.R"))

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a_test — Teste do pipeline 2D antes de rodar o job completo
#
# Roda apenas test_n_shards shards escolhidos do grid n_row_shards x n_col_shards.
# O objetivo é verificar:
#   1. Geometria dos tiles de saída (extensão, resolução, nrows/ncols)
#   2. Valores produzidos (mediana, sd, mask) — plausibilidade
#   3. RSS por processo (piso de RAM real com 2D)
#   4. Throughput (s/bloco) para estimar ETA do job real
#
# Ao final imprime um diagnóstico comparando RAM e throughput com o 05
# (referência: ~7-8 GB piso, ~130 s/bloco após otimização).
#
# NÃO roda o merge — os tiles de teste ficam em raster/parts_2d/ para
# inspeção manual. Apague-os antes de rodar o 06a real.
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuração de TESTE ──────────────────────────────────────────────────────
# Estes valores devem ser os MESMOS que você planeja usar no 06a real.
# Só test_n_shards e max_concurrent podem ser menores no teste.

n_row_shards <- 250      # igual ao que usará no 06a real
n_col_shards <- 4        # igual ao que usará no 06a real

# Quantos shards rodar no teste. 4 = 1 por coluna (testa toda a largura).
# Aumente para 8 se quiser testar 2 linhas de shards.
test_n_shards <- 4

# Quais shards rodar. "auto" = distribui pelos 4 tiles de coluna na linha 1
# (testa geometria em todas as colunas). Ou defina manualmente, ex:
#   test_shards <- list(c(1,1), c(1,2), c(1,3), c(1,4))
test_shards <- "auto"

# Concorrência e RAM (pode ser menor que o real para o teste)
max_concurrent <- 2
poll_interval_s <- 15

# ── Paths ──────────────────────────────────────────────────────────────────────

script_dir    <- file.path(project_root, "examples", "soc_stock_0_5cm")
worker_script <- file.path(script_dir, "05_predict_spatial.R")

log_dir <- file.path(project_root, "outputs", "spatial_prediction", "_worker_logs",
                     paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_TEST2D"))
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)

rscript_bin <- file.path(R.home("bin"), "Rscript.exe")
if (!file.exists(rscript_bin)) stop("Rscript.exe not found: ", rscript_bin)

# ── Montar lista de shards de teste ──────────────────────────────────────────

if (identical(test_shards, "auto")) {
  # 1 shard por tile de coluna, todos na linha 1 — testa a geometria completa
  # da largura do raster com poucos blocos (linha 1 = latitudes mais altas = geralmente menos pixels validos = rapido)
  test_shards <- lapply(seq_len(n_col_shards), function(cs) c(1L, cs))
  if (test_n_shards > n_col_shards) {
    # adiciona shards da linha do meio (mais densos, melhor teste de throughput)
    mid_row <- ceiling(n_row_shards / 2)
    extra <- lapply(seq_len(min(test_n_shards - n_col_shards, n_col_shards)),
                    function(cs) c(mid_row, cs))
    test_shards <- c(test_shards, extra)
  }
  test_shards <- test_shards[seq_len(min(test_n_shards, length(test_shards)))]
}

n_test <- length(test_shards)
message(sprintf("Teste 2D: %d x %d grid | rodando %d shard(s) | %d por vez",
                n_row_shards, n_col_shards, n_test, max_concurrent))
message(sprintf("Shards de teste: %s",
                paste(sapply(test_shards, function(x) sprintf("[r%d/c%d]", x[1], x[2])),
                      collapse = " ")))
message("Logs: ", log_dir, "\n")

# ── Fila ───────────────────────────────────────────────────────────────────────

pending    <- seq_len(n_test)
active     <- list()
exit_codes <- integer(n_test)
rss_peak   <- numeric(n_test)   # RSS max lido do log de cada shard

launch_shard <- function(idx) {
  rs <- test_shards[[idx]][1]
  cs <- test_shards[[idx]][2]
  log_file <- file.path(log_dir,
    sprintf("test_r%03dof%03d_c%03dof%03d.log", rs, n_row_shards, cs, n_col_shards))
  p <- processx::process$new(
    rscript_bin,
    args = c(worker_script,
             as.character(rs), as.character(cs),
             as.character(n_row_shards), as.character(n_col_shards),
             as.character(max_concurrent)),
    stdout  = log_file,
    stderr  = log_file,
    cleanup = TRUE
  )
  message(sprintf("  Shard [r%03d/c%03d] PID %d -> %s",
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
      idx <- as.integer(idx_chr)
      rs  <- test_shards[[idx]][1]
      cs  <- test_shards[[idx]][2]
      exit_codes[idx] <- p$get_exit_status()
      n_done <- n_done + 1L
      status <- if (exit_codes[idx] == 0L) "OK" else paste0("FALHOU (exit ", exit_codes[idx], ")")
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
  message(sprintf("  [%.1f %s] %d/%d done, %d rodando, %d aguardando",
                  as.numeric(el), units(el), n_done, n_test, length(active), length(pending)))
}

el_total <- Sys.time() - t0

# ── Diagnóstico dos logs ───────────────────────────────────────────────────────

message("\n", strrep("═", 70))
message("DIAGNÓSTICO DO TESTE 2D")
message(strrep("═", 70))

parse_worker_log <- function(idx) {
  rs <- test_shards[[idx]][1]
  cs <- test_shards[[idx]][2]
  log_file <- file.path(log_dir,
    sprintf("test_r%03dof%03d_c%03dof%03d.log", rs, n_row_shards, cs, n_col_shards))
  if (!file.exists(log_file)) return(NULL)
  lines <- readLines(log_file, warn = FALSE)

  # strip_ncol vs r_ncol
  strip_line <- lines[grepl("strip_ncol", lines)]
  strip_ncol <- NA_integer_; r_ncol <- NA_integer_; ram_factor <- NA_real_
  if (length(strip_line) > 0) {
    m <- regmatches(strip_line[1], regexpr("strip_ncol (\\d+) vs r_ncol (\\d+)", strip_line[1]))
    if (length(m) > 0) {
      nums <- as.integer(regmatches(m, gregexpr("\\d+", m))[[1]])
      if (length(nums) >= 2) { strip_ncol <- nums[1]; r_ncol <- nums[2] }
      ram_factor <- if (!is.na(r_ncol) && strip_ncol > 0) r_ncol / strip_ncol else NA_real_
    }
  }

  # output_block_rows
  blk_line <- lines[grepl("Auto output_block_rows", lines)]
  output_block_rows <- NA_integer_
  if (length(blk_line) > 0) {
    m <- regmatches(blk_line[1], regexpr("= (\\d+)", blk_line[1]))
    if (length(m) > 0) output_block_rows <- as.integer(sub("= ", "", m))
  }

  # RSS máximo
  rss_vals <- as.numeric(regmatches(lines,
    gregexpr("RSS ([0-9]+(?:\\.[0-9]+)?) MB", lines, perl = TRUE)) |>
    lapply(function(x) sub("RSS ", "", x)) |> unlist())
  rss_peak_mb <- if (length(rss_vals) > 0) max(rss_vals, na.rm = TRUE) else NA_real_

  # s/bloco (predict)
  block_lines <- lines[grepl("predict [0-9]+\\.[0-9]+s", lines)]
  predict_times <- as.numeric(regmatches(block_lines,
    gregexpr("[0-9]+\\.[0-9]+(?=s \\| RSS)", block_lines, perl = TRUE)) |> unlist())
  median_predict_s <- if (length(predict_times) > 0) median(predict_times) else NA_real_

  # n_valid total (ultima linha de Block)
  n_valid <- NA_integer_
  blk_summary <- lines[grepl("^Block [0-9]+/[0-9]+", lines)]
  if (length(blk_summary) > 0) {
    m <- regmatches(blk_summary[length(blk_summary)],
                    regexpr("valid ([0-9,]+)", blk_summary[length(blk_summary)]))
    if (length(m) > 0) n_valid <- as.integer(gsub(",", "", sub("valid ", "", m)))
  }

  # exit
  ok <- exit_codes[idx] == 0L

  list(rs = rs, cs = cs, ok = ok,
       strip_ncol = strip_ncol, r_ncol = r_ncol, ram_factor = ram_factor,
       output_block_rows = output_block_rows,
       rss_peak_mb = rss_peak_mb, median_predict_s = median_predict_s,
       n_valid = n_valid, log = log_file)
}

results <- lapply(seq_len(n_test), parse_worker_log)

message(sprintf("\n%-20s %10s %10s %10s %12s %12s %8s",
                "shard", "strip_ncol", "ram_fator", "blk_rows", "RSS_pico_MB", "pred_s/blk", "status"))
message(strrep("-", 85))

for (r in results) {
  if (is.null(r)) next
  message(sprintf("[r%03d/c%03d]          %10s %10s %10s %12s %12s %8s",
                  r$rs, r$cs,
                  ifelse(is.na(r$strip_ncol), "?", format(r$strip_ncol, big.mark = ",")),
                  ifelse(is.na(r$ram_factor),  "?", sprintf("%.1fx", r$ram_factor)),
                  ifelse(is.na(r$output_block_rows), "?", r$output_block_rows),
                  ifelse(is.na(r$rss_peak_mb), "?", sprintf("%.0f", r$rss_peak_mb)),
                  ifelse(is.na(r$median_predict_s), "?", sprintf("%.1f", r$median_predict_s)),
                  if (r$ok) "OK" else "FALHOU"))
}

message(strrep("-", 85))

# Verifica geometria dos tiles produzidos
message("\n── Verificação de geometria dos tiles ──────────────────────────────")
target_label <- "soc_stock_0_5cm"

output_dir <- file.path(project_root, "outputs", "spatial_prediction",
                        "soc_stock_modeling", target_label)
parts_dir  <- file.path(output_dir, "raster", "parts_2d")
config_id  <- "auto"

if (identical(config_id, "auto")) {
  final_model_base <- file.path(project_root, "outputs", "final_model",
                                "soc_stock_modeling", target_label)
  run_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^final_", run_dirs)]
  if (length(run_dirs) > 0) {
    final_run_id <- sort(run_dirs, decreasing = TRUE)[1]
    summary_path <- file.path(final_model_base, final_run_id, "comparison",
                              "final_run_summary.rds")
    if (file.exists(summary_path)) {
      config_id <- readRDS(summary_path)$selected_cfgs$config_id[1]
    }
  }
}

test_tiles <- list.files(parts_dir,
  pattern = paste0("^", target_label, "_", config_id,
                   "_ensemble_median.*_r[0-9]+of[0-9]+_c[0-9]+of[0-9]+\\.tif$"),
  full.names = TRUE)

if (length(test_tiles) > 0) {
  metadata_dir  <- file.path(project_root, "outputs", "metadata",
                             "soc_stock_modeling", target_label)
  raster_table  <- readr::read_csv2(file.path(metadata_dir, "raster_table_used.csv"),
                                    show_col_types = FALSE)
  full_template <- terra::rast(raster_table$raster_file[1])

  for (tf in sort(test_tiles)) {
    r <- terra::rast(tf)
    geom_ok <- terra::compareGeom(r, full_template, stopOnError = FALSE,
                                  res = TRUE, crs = TRUE, ext = FALSE)
    message(sprintf("  %s", basename(tf)))
    message(sprintf("    nrow=%d ncol=%d | ext [%.4f %.4f %.4f %.4f] | res OK=%s",
                    terra::nrow(r), terra::ncol(r),
                    terra::xmin(r), terra::xmax(r), terra::ymin(r), terra::ymax(r),
                    geom_ok))
  }
} else {
  message("  Nenhum tile de teste encontrado em: ", parts_dir)
}

# Recomendações finais
message("\n── Recomendações para o job real (05a_run_parallel.R) ─────────────")
ok_results <- Filter(function(r) !is.null(r) && r$ok, results)
if (length(ok_results) > 0) {
  rss_vals    <- sapply(ok_results, function(r) r$rss_peak_mb)
  pred_vals   <- sapply(ok_results, function(r) r$median_predict_s)
  ram_factors <- sapply(ok_results, function(r) r$ram_factor)

  avg_rss  <- mean(rss_vals,  na.rm = TRUE)
  avg_pred <- mean(pred_vals, na.rm = TRUE)
  avg_rf   <- mean(ram_factors, na.rm = TRUE)

  # RAM disponível: 64 GB = 64000 MB; margem de 30%
  ram_total_mb    <- 64000
  ram_budget_mb   <- ram_total_mb * 0.70   # 70% para shards
  safe_concurrent <- floor(ram_budget_mb / max(avg_rss, 1))
  safe_concurrent <- min(safe_concurrent, parallel::detectCores())

  message(sprintf("  RAM pico por shard  : %.0f MB (media dos %d shards de teste)",
                  avg_rss, length(ok_results)))
  message(sprintf("  Reducao de RAM vs 05: %.1fx (strip_ncol / r_ncol)",
                  avg_rf))
  message(sprintf("  Throughput          : %.1f s/bloco (predict)",
                  avg_pred))
  message(sprintf("  max_concurrent seguro (70%% de 64 GB): %d processos",
                  safe_concurrent))

  n_blocks_est <- n_row_shards * n_col_shards * 32   # ~32 blocos/shard (estimativa)
  eta_h <- (n_blocks_est * avg_pred) / safe_concurrent / 3600
  message(sprintf("  ETA estimado (job real, %d x %d = %d shards): ~%.0f h (~%.1f dias)",
                  n_row_shards, n_col_shards, n_row_shards * n_col_shards,
                  eta_h, eta_h / 24))

  message(sprintf("\n  -> Edite 05a_run_parallel.R: max_concurrent <- %d", safe_concurrent))
} else {
  message("  Nenhum shard completou com sucesso — verifique os logs em: ", log_dir)
}

message(sprintf("\nTeste concluido em %.1f %s. Logs em: %s",
                as.numeric(el_total), units(el_total), log_dir))
message("Tiles de teste em: ", parts_dir)
message("Apague os tiles de teste antes de rodar o 06a real.")
