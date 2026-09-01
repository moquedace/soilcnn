project_root <- "D:/usuario_armazenamento/cassio/R/deep_learning_caret"

source(
  "https://raw.githubusercontent.com/moquedace/funcs/refs/heads/main/utils/install_load_pkg.R"
)

pkg <- c("processx")
install_load_pkg(pkg)

# ══════════════════════════════════════════════════════════════════════════════
# 05a — Orquestrador de predição espacial com tiling 2D
#
# Divide o raster em n_row_shards x n_col_shards retângulos. Cada shard é um
# processo independente que lê apenas a faixa de colunas do seu tile (+ margem
# half_w_max de cada lado), reduzindo o piso de RAM proporcionalmente.
#
# Exemplo com raster global 160 k colunas, 187 bandas:
#   tiling 1D (n_col_shards=1, esquema anterior, retirado): ~240 MB/linha ->
#     piso ~7-8 GB -> max_concurrent=2
#   tiling 2D (n_col_shards=4, esquema atual): ~60 MB/linha -> piso ~2 GB ->
#     max_concurrent=8
#
# n_col_shards deve ser escolhido de modo que:
#   ceil(r_ncol / n_col_shards) >> 2 * half_w_max  (tile >> margem)
# Para half_w_max = 16 (janela 33), n_col_shards <= 160 k / (10 * 32) = 500 e
# razoavel (ex: 4-8 ja dao reducao suficiente sem overhead excessivo).
#
# Tiles de saida vao para raster/parts_2d/ com sufixo _rXXXofYYY_cXXXofYYY.
# Ao final roda 05b_merge_spatial_parts.R para montar os mapas finais.
# ══════════════════════════════════════════════════════════════════════════════

# ── Configuração ───────────────────────────────────────────────────────────────

# Fatias de linha. Mais fatias = processo vive
# menos tempo = menos fragmentacao de RAM.
n_row_shards <- 250

# Fatias de coluna: cada fatia reduz strip_ncol por 1/n_col_shards -> RAM
# por processo proporcional. n_col_shards=4 divide ~240 MB/linha em ~60 MB.
n_col_shards <- 4

# Total de shards = n_row_shards * n_col_shards
# Com 250 x 4 = 1000 shards (igual ao 05a em numero total, mas layout 2D)

# Processos simultâneos.
# Causa raiz do RSS alto identificada e corrigida: nao era output_block_rows,
# e sim batch_size em build_patches_multi (a checagem de validade da janela
# indexava strip_values para todo o chunk, gerando um pico transitorio de
# ~batch_size * n_pos_janela * n_channels * 8 bytes, independente do tamanho
# do bloco). Reduzindo batch_size 4096 -> 512 e eliminando a reindexacao
# duplicada em build_patches_multi, reteste (05a_test.R, linha 1, 4 col-shards)
# deu:
#   shards sparse (oceano): RSS ~2.2 GB, ~2.7 min/shard
#   shard denso (mesma regiao que antes batia 37 GB): RSS pico 13.2 GB,
#     estavel ao longo dos 16 blocos (sem mais degradacao progressiva),
#     34.5 min/shard (antes 89 min para o mesmo shard)
# Maquina: 32 nucleos, ~64 GB RAM.
#
# ATUALIZACAO (apos smoke test completo em 1 km, mesma largura de strip ~40 mil
# colunas por bloco que a config 4 col-shards aqui): RSS pico medido de verdade
# ficou em 13.9-14.9 GB/shard (um pouco acima dos 13.2 GB de referencia acima --
# essa referencia e de antes do modelo final ter 10 seeds; mais seeds nao muda
# muito o pico de RSS -- quem domina o pico e o buffer de checagem de validade
# de janela, dimensionado por batch_size, nao por n_seeds -- mas ainda assim
# vale a margem extra). Alem disso, esta maquina NAO tem GPU acessivel ao torch
# (cuda_is_available()==FALSE, cuda_device_count()==0, confirmado nos logs do
# teste 1 km -- "Device: cpu") -- o trabalho e 100% CPU-bound, entao subir
# max_concurrent alem do necessario para caber na RAM NAO acelera o total (a
# CPU total da maquina e fixa); so reduz a margem de seguranca. Por isso:
# -> max_concurrent=3: pior caso 3 x 14.9 GB ~= 45 GB / 64 GB (margem ~19 GB).
#    threads_per_worker = 32/3 = 10.
# -> Reavalie via 05c_estimate_eta.R conforme os primeiros shards reais terminam.
max_concurrent <- 3

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

# ── Resolve config_id/target_label (mesma logica do 05_predict_spatial.R) ─────
# Necessario aqui so para saber onde ficam os tiles ja gerados (resume).

target_label <- "soc_stock_0_5cm"
final_run_id <- "latest"

metadata_dir     <- file.path(project_root, "outputs", "metadata",
                              "soc_stock_modeling", target_label)
final_model_base <- file.path(project_root, "outputs", "final_model",
                              "soc_stock_modeling", target_label)

if (identical(final_run_id, "latest")) {
  run_dirs <- list.dirs(final_model_base, recursive = FALSE, full.names = FALSE)
  run_dirs <- run_dirs[grepl("^final_", run_dirs)]
  if (length(run_dirs) == 0) stop("Nenhum final_* encontrado em: ", final_model_base)
  final_run_id <- sort(run_dirs, decreasing = TRUE)[1]
}
final_run_dir <- file.path(final_model_base, final_run_id)
summary_file  <- file.path(final_run_dir, "comparison", "final_run_summary.rds")
if (!file.exists(summary_file)) stop("final_run_summary.rds nao encontrado: ", summary_file)
config_id <- readRDS(summary_file)$selected_cfgs$config_id[1]

output_log_dir <- file.path(project_root, "outputs", "spatial_prediction",
                            "soc_stock_modeling", target_label, config_id, "log")

message(sprintf("config_id: %s | final_run_id: %s | logs em: %s",
                config_id, final_run_id, output_log_dir))

# ── Gerar lista de todos os shards (produto cartesiano row x col) ──────────────

shards <- expand.grid(row_shard = seq_len(n_row_shards),
                       col_shard = seq_len(n_col_shards))
# Ordena por linha primeiro (itera linha a linha) para mosaico progressivo
shards <- shards[order(shards$row_shard, shards$col_shard), ]
n_shards_total <- nrow(shards)

message(sprintf("Tiling 2D: %d x %d = %d shards, %d por vez",
                n_row_shards, n_col_shards, n_shards_total, max_concurrent))
message("Logs: ", log_dir, "\n")

# ── Resume: pula shards ja concluidos com sucesso ─────────────────────────────
# Os tiles .tif sao criados (writeStart) com as dimensoes corretas ja no
# INICIO do bloco 1 -- se o processo morrer no meio (ex: bloco 11/16), o
# arquivo pode continuar abrindo sem erro no terra, so com blocos faltando.
# Checar so a existencia dos .tif daria falso-positivo (tile incompleto
# aceito como concluido -> buraco silencioso no mosaico final).
#
# Sinal confiavel: prediction_config_r###of###_c###of###.csv em cfg_022/log/
# so e escrito DEPOIS do writeStop() de todos os rasters e das checagens de
# sanidade passarem (ver 05_predict_spatial.R). Se o processo morre antes
# disso, esse CSV nunca existe -- entao a existencia dele implica shard
# 100% concluido e validado.

shard_config_file <- function(rs, cs) {
  suf <- sprintf("_r%03dof%03d_c%03dof%03d", rs, n_row_shards, cs, n_col_shards)
  file.path(output_log_dir, paste0("prediction_config", suf, ".csv"))
}

shard_already_done <- function(rs, cs) {
  file.exists(shard_config_file(rs, cs))
}

message("Verificando shards ja concluidos (resume)...")
done_mask <- vapply(seq_len(n_shards_total), function(idx)
  shard_already_done(shards$row_shard[idx], shards$col_shard[idx]), logical(1))
n_already_done <- sum(done_mask)
if (n_already_done > 0L) {
  message(sprintf("  %d/%d shards ja concluidos -- pulando (resume).",
                  n_already_done, n_shards_total))
}

# ── Fila ───────────────────────────────────────────────────────────────────────

pending    <- which(!done_mask)          # índice nas linhas de `shards`
active     <- list()                     # idx -> process
exit_codes <- integer(n_shards_total)
exit_codes[done_mask] <- 0L

launch_shard <- function(idx) {
  rs <- shards$row_shard[idx]
  cs <- shards$col_shard[idx]
  log_file <- file.path(log_dir,
    sprintf("shard_r%03dof%03d_c%03dof%03d.log", rs, n_row_shards, cs, n_col_shards))
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
                else paste0("FALHOU (exit ", exit_codes[idx], ")")
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
  message(sprintf("  [%.1f %s elapsed] %d/%d done (%d ja prontos + %d nesta sessao), %d rodando, %d na fila",
                  as.numeric(el), units(el),
                  n_already_done + n_done, n_shards_total, n_already_done, n_done,
                  length(active), length(pending)))
}

el_total <- Sys.time() - t0
message(sprintf("\nTodos os shards concluidos em %.1f %s.",
                as.numeric(el_total), units(el_total)))

failed <- which(exit_codes != 0L)
if (length(failed) > 0) {
  failed_info <- shards[failed, ]
  msg <- paste(sprintf("[r%d/c%d]", failed_info$row_shard, failed_info$col_shard),
               collapse = ", ")
  stop("Shards com falha: ", msg, "\nLogs em: ", log_dir)
}

# ── Merge ──────────────────────────────────────────────────────────────────────

message("\nRodando merge (05b_merge_spatial_parts.R)...\n")
merge_log    <- file.path(log_dir, "merge.log")
merge_result <- processx::run(rscript_bin, args = merge_script,
                              stdout = "|", stderr = "|", echo = TRUE,
                              error_on_status = FALSE)
writeLines(c(merge_result$stdout, merge_result$stderr), merge_log)

if (merge_result$status != 0L) {
  stop("Merge falhou (exit ", merge_result$status, "). Ver: ", merge_log)
}

message("\n── Tudo pronto ───────────────────────────────────────────────────")
message("  Logs: ", log_dir)
message("  Merge: ", merge_log)
